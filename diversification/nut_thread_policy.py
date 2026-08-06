"""The nut-threading policy of examples/solve_nut_and_bolt.py, as a sampler.

This is that solve script with its lifted constants read from a theta dict instead of argparse,
and nothing else changed. The phase machine — settle/hover/descend/grasp/lift/over/lower/thread,
and inside thread the pinch/seek/wind/open/rewind/reclose state machine with its pacing, bind
escape and width-gated reclose — is the original, because that machine is what tolerates the
sampling: every target is recomputed from measured state, so a different pinch force or spawn
position is absorbed rather than followed off a cliff.

Frozen constants (see params_nut_thread.yaml for why each one) stay literals here on purpose:
dt, down_yaw, HAND_OFFSET, GRIP_KP, the OSC gains, the xy-integrator caps, the z floor, the
+8 mm rewind lift, stop_dz.

solve(env, theta, rec) runs one episode and returns a verdict dict. The env is built by the
caller (run_episode.py), which is where the environment half of theta is applied.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
)

OPEN, CLOSE = 0.04, 0.0
HAND_OFFSET = 0.058          # frozen: measured panda_hand -> fingertip
GRIP_KP = 8000.0             # frozen: pinch force = KP * overshoot
DOWN_YAW = -0.5              # frozen: centres wrist-roll q7 for stroke range
STOP_DZ = 0.005              # frozen: bottom-out ~0.001, pads graze below ~0.004


def solve(env, theta: dict[str, Any], rec=None, max_sec: float = 300.0,
          log_every_s: float = 8.0) -> dict[str, Any]:
    """Run one threading episode. Returns the verdict; `rec.step(env)` is called per control step."""
    device = env.device
    scene, robot = env.scene, env.robot
    osc = robot.controller.controllers[0]
    # frozen: raised after run 1 stalled turning the engaged nut
    osc._kp = torch.tensor([150.0, 150.0, 150.0, 600.0, 600.0, 600.0], device=device)
    osc._kd = 2.0 * osc._kp.sqrt()
    osc.cfg.rot_scale = 0.15
    art = robot.articulation
    ee_idx = art.body_names.index("panda_hand")
    lf_idx = art.body_names.index("panda_leftfinger")
    rf_idx = art.body_names.index("panda_rightfinger")
    fj1 = art.find_joints(["panda_finger_joint1"])[0]
    down = quat_from_euler_xyz(*(torch.tensor([v], device=device)
                                 for v in (3.14159, 0.0, DOWN_YAW)))[0]
    dim = robot.action_dim
    DECIM = robot.control_period
    CTRL_HZ = 1.0 / (env.dt * DECIM)
    env.reset()

    def ee_pose():
        return art.data.body_pos_w[0, ee_idx], art.data.body_quat_w[0, ee_idx]

    def servo(goal_pos, goal_quat, grip):
        p, q = ee_pose()
        a = torch.zeros(1, dim, device=device)
        a[0, 0:3] = ((goal_pos - p) / osc.cfg.pos_scale).clamp(-1.0, 1.0)
        qe = quat_mul(goal_quat.unsqueeze(0), quat_conjugate(q.unsqueeze(0)))
        a[0, 3:6] = (axis_angle_from_quat(qe)[0] / osc.cfg.rot_scale).clamp(-1.0, 1.0)
        a[0, 6:8] = grip
        return a

    def nut():
        return scene.nuts[0].data.root_pos_w[0]

    def grasp_center():
        return 0.5 * (art.data.body_pos_w[0, lf_idx] + art.data.body_pos_w[0, rf_idx])

    def bolt():
        return scene.bolts[0].data.root_pos_w[0]

    def yaw_of(q):
        ex = quat_apply(q.unsqueeze(0), torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        return math.degrees(math.atan2(ex[1].item(), ex[0].item()))

    def yaw_about_z(qbase, ang):
        yz = quat_from_euler_xyz(*(torch.tensor([v], device=device) for v in (0.0, 0.0, ang)))
        return quat_mul(yz, qbase.unsqueeze(0))[0]

    Z = lambda z: torch.tensor([0.0, 0.0, z], device=device)

    # ---- theta ---------------------------------------------------------------------------------
    pinch_n = float(theta["pinch_n"])
    pinch_wind_n = float(theta["pinch_wind_n"])
    lean_cmd = float(theta["lean"])
    seek_max = math.radians(float(theta["seek_deg"]))
    sweep = math.radians(float(theta["sweep_deg"]))
    hover_dz = float(theta["hover_dz"])
    carry_dz = float(theta["carry_dz"])
    lower_mps = float(theta["lower_mps"])
    w_land_rad = math.radians(float(theta["w_land_deg"]))
    nut_center = bool(theta["nut_center"])
    pm = float(theta["phase_mult"])

    SEC = lambda s: max(1, round(s * CTRL_HZ))
    T_SETTLE, T_HOVER, T_DESCEND, T_GRASP, T_LIFT, T_OVER = (
        SEC(s * pm) for s in (0.5, 3.0, 3.0, 1.0, 2.5, 1.8))
    T_PINCH, T_OPEN, T_RECLOSE, T_QUIET = SEC(0.5), SEC(0.4), SEC(0.5), SEC(0.5)
    D_WIND = float(theta["wind_rate"]) / CTRL_HZ
    D_REWIND = float(theta["rewind_rate"]) / CTRL_HZ

    phase, ph_t = "settle", 0
    st: dict[str, Any] = {
        "p_flat": None, "grip_off": None, "z_contact": None, "lower_z": None, "nut_hist": [],
        "t_state": "pinch", "wound": 0.0, "t": 0, "dz_seek0": None, "stroke": 0, "caught": False,
        "ee0": None, "stall_t": 0, "w_start": 0.0, "w_land": 0.0, "align": None, "flat_ok": False,
        "stall_run": 0, "xy_tgt": None, "xy_tgt_g": None, "dz_touch": None, "rc_at_band": False,
        "err_descend": None,
    }
    yaw_acc = {"ee": 0.0, "nut": 0.0, "ee_prev": None, "nut_prev": None}

    def unwrap(key, val):
        prev = yaw_acc[key + "_prev"]
        if prev is not None:
            yaw_acc[key] += (val - prev + 180.0) % 360.0 - 180.0
        yaw_acc[key + "_prev"] = val
        return yaw_acc[key]

    def pinch_target(newtons):
        return st["p_flat"] - newtons / GRIP_KP

    def wide_target():
        return st["p_flat"] + 0.005

    LOG_EVERY = SEC(log_every_s)
    LOST_FOR = SEC(3.0)
    steps, lost_t, lost = 0, 0, False
    for i in range(1, SEC(max_sec) + 1):
        nx, bx = nut(), bolt()
        dz = (nx - bx)[2].item()
        p, q = ee_pose()
        ee_u = unwrap("ee", yaw_of(q))
        unwrap("nut", yaw_of(scene.nuts[0].data.root_quat_w[0]))

        if phase == "settle":
            action = servo(p, down, OPEN)
            if ph_t >= T_SETTLE:
                phase, ph_t = "hover", 0
        elif phase == "hover":
            tgt = nx + Z(HAND_OFFSET + hover_dz)
            if nut_center:
                if st["xy_tgt_g"] is None:
                    st["xy_tgt_g"] = nx[:2].clone()
                st["xy_tgt_g"] += (0.01 * (nx[:2] - grasp_center()[:2])).clamp(-3e-4, 3e-4)
                st["xy_tgt_g"] = nx[:2] + (st["xy_tgt_g"] - nx[:2]).clamp(-0.03, 0.03)
                tgt[0:2] = st["xy_tgt_g"]
            action = servo(tgt, down, OPEN)
            if ph_t >= T_HOVER:
                phase, ph_t = "descend", 0
        elif phase == "descend":
            tgt = nx + Z(HAND_OFFSET)
            if nut_center:
                st["xy_tgt_g"] += (0.01 * (nx[:2] - grasp_center()[:2])).clamp(-3e-4, 3e-4)
                st["xy_tgt_g"] = nx[:2] + (st["xy_tgt_g"] - nx[:2]).clamp(-0.03, 0.03)
                tgt[0:2] = st["xy_tgt_g"]
            action = servo(tgt, down, OPEN)
            if ph_t >= T_DESCEND:
                st["err_descend"] = (p - (nx + Z(HAND_OFFSET)))
                phase, ph_t = "grasp", 0
        elif phase == "grasp":
            tgt = nx + Z(HAND_OFFSET)
            if nut_center and st["xy_tgt_g"] is not None:
                tgt[0:2] = st["xy_tgt_g"]
            action = servo(tgt, down, CLOSE)
            if ph_t >= T_GRASP:
                st["p_flat"] = art.data.joint_pos[0, fj1].item()
                st["grip_off"] = p[2].item() - nx[2].item()
                err = st["err_descend"]
                print(f"[cal] across-flats grip: finger={st['p_flat'] * 1e3:.2f}mm "
                      f"(width {2 * st['p_flat'] * 1e3:.1f}mm) grip_off={st['grip_off'] * 1e3:.1f}mm"
                      + (f" | descend reach-err=({err[0] * 1e3:+.1f},{err[1] * 1e3:+.1f},"
                         f"{err[2] * 1e3:+.1f})mm" if err is not None else ""), flush=True)
                phase, ph_t = "lift", 0
        elif phase == "lift":
            action = servo(bx + Z(HAND_OFFSET + carry_dz), down, CLOSE)
            if ph_t >= T_LIFT:
                phase, ph_t = "over", 0
        elif phase == "over":
            tgt = bx + Z(HAND_OFFSET + carry_dz)
            if nut_center:
                if st["xy_tgt"] is None:
                    st["xy_tgt"] = bx[:2].clone()
                st["xy_tgt"] += (0.01 * (bx[:2] - nx[:2])).clamp(-3e-4, 3e-4)
                st["xy_tgt"] = bx[:2] + (st["xy_tgt"] - bx[:2]).clamp(-0.025, 0.025)
                tgt[0:2] = st["xy_tgt"]
            action = servo(tgt, down, CLOSE)
            if ph_t >= T_OVER:
                phase, ph_t = "lower", 0
                st["lower_z"] = p[2].item()
        elif phase == "lower":
            st["lower_z"] -= lower_mps / CTRL_HZ
            st["nut_hist"].append(nx[2].item())
            st["nut_hist"] = st["nut_hist"][-T_QUIET:]
            tgt = torch.tensor([bx[0], bx[1], st["lower_z"]], device=device)
            if nut_center:
                st["xy_tgt"] += (0.01 * (bx[:2] - nx[:2])).clamp(-3e-4, 3e-4)
                st["xy_tgt"] = bx[:2] + (st["xy_tgt"] - bx[:2]).clamp(-0.025, 0.025)
                tgt[0:2] = st["xy_tgt"]
            action = servo(tgt, down, CLOSE)
            quiet = len(st["nut_hist"]) == T_QUIET and (max(st["nut_hist"]) - min(st["nut_hist"])) < 2e-4
            if (quiet and st["lower_z"] < p[2].item() - 0.002 and dz < 0.035) or dz < 0.0245:
                st["z_contact"] = p[2].item()
                st["dz_touch"] = dz
                print(f"[touch] nut down on bolt: dz={dz * 1e3:.1f}mm hand_z={p[2].item():.4f} "
                      f"(lower_z target {st['lower_z']:.4f})", flush=True)
                if st["xy_tgt"] is not None:
                    bias = st["xy_tgt"] - bx[:2]
                    print(f"[centre] learned xy bias=({bias[0] * 1e3:+.1f},{bias[1] * 1e3:+.1f})mm "
                          f"nut_lat={(nx[:2] - bx[:2]).norm() * 1e3:.1f}mm", flush=True)
                jp = art.data.joint_pos[0].tolist()
                jl = art.data.joint_pos_limits[0]
                margins = [min(jp[k] - jl[k, 0].item(), jl[k, 1].item() - jp[k]) for k in range(7)]
                print("[reach] threading-pose arm q(rad)=["
                      + ", ".join(f"{v:+.2f}" for v in jp[:7]) + f"] min_margin={min(margins):+.2f}"
                      f" | bolt=({bx[0]:.3f},{bx[1]:.3f},{bx[2]:.3f}) nut=({nx[0]:.3f},{nx[1]:.3f})",
                      flush=True)
                phase, ph_t = "thread", 0
        elif phase == "thread":
            ts = st["t_state"]
            grip = pinch_target(pinch_n if ts in ("pinch", "seek") else pinch_wind_n)
            lean = 0.0
            if ts == "pinch":
                st["t"] += 1
                if st["t"] >= T_PINCH:
                    drop = (st["dz_touch"] - dz) if st["dz_touch"] is not None else 0.0
                    if drop > 8e-4:
                        print(f"[seek] pre-engaged at touchdown (settle drop {drop * 1e3:.2f}mm) "
                              f"— winding directly", flush=True)
                        st["caught"], st["t_state"] = True, "wind"
                    else:
                        st["t_state"], st["t"] = "seek", 0
                        st["dz_seek0"] = dz
            elif ts == "seek":
                st["wound"] += D_WIND
                lean = lean_cmd
                if st["dz_seek0"] - dz > 4e-4:
                    print(f"[seek] caught the thread start after "
                          f"{math.degrees(st['wound']):.0f}deg "
                          f"(dropped {(st['dz_seek0'] - dz) * 1e3:.2f}mm)", flush=True)
                    st["caught"], st["t_state"] = True, "wind"
                elif dz - st["dz_seek0"] > 5e-4:
                    print(f"[seek] nut rising ({(dz - st['dz_seek0']) * 1e3:.2f}mm) after "
                          f"{math.degrees(st['wound']):.0f}deg — already engaged, winding", flush=True)
                    st["caught"], st["t_state"] = True, "wind"
                elif st["wound"] >= seek_max:
                    st["t_state"] = "wind"
            elif ts == "wind":
                if st["ee0"] is None:
                    st["ee0"], st["stall_t"], st["w_start"] = ee_u, 0, st["wound"]
                    st["flat_ok"], st["stall_run"], st["t"] = False, 0, 0
                gpos_now = art.data.joint_pos[0, fj1].item()
                if not st["flat_ok"]:
                    grip, st["t"] = pinch_target(pinch_n), st["t"] + 1
                    if gpos_now < st["p_flat"] + 5e-4 or st["t"] >= SEC(1.5):
                        if gpos_now >= st["p_flat"] + 5e-4:
                            print(f"[wind] stroke {st['stroke']}: no flat contact after 1.5s "
                                  f"(grip {gpos_now * 1e3:.2f}mm) — clamping anyway", flush=True)
                        st["flat_ok"] = True
                        st["ee0"] = ee_u
                else:
                    swept = math.radians(st["ee0"] - ee_u)
                    lag = st["w_start"] - st["wound"] - swept
                    lean = lean_cmd
                    st["stall_run"] = st["stall_run"] + 1 if lag >= 0.35 else 0
                    w_end = st["w_start"] - sweep
                    if st["stall_run"] >= SEC(2.5):
                        print(f"[wind] stroke {st['stroke']} BOUND (lag "
                              f"{math.degrees(lag):.0f}deg for 2.5s) at wound "
                              f"{math.degrees(st['wound']):.0f}deg — recycling grip", flush=True)
                        st["t_state"], st["t"], st["ee0"] = "open", 0, None
                        st["stroke"] += 1
                    elif st["wound"] > w_end:
                        if lag < 0.35:
                            st["wound"] = max(w_end, st["wound"] - D_WIND)
                    else:
                        st["stall_t"] += 1
                        if lag < 0.26 or st["stall_t"] >= SEC(1.5):
                            if lag >= 0.26:
                                print(f"[wind] stroke {st['stroke']} ended "
                                      f"{math.degrees(lag):.0f}deg short", flush=True)
                            st["t_state"], st["t"], st["ee0"] = "open", 0, None
                            st["stroke"] += 1
            elif ts == "open":
                grip, st["t"] = wide_target(), st["t"] + 1
                if st["t"] >= T_OPEN:
                    st["w_land"] = w_land_rad
                    st["t_state"] = "rewind"
            elif ts == "rewind":
                grip = wide_target()
                st["wound"] = min(st["w_land"], st["wound"] + D_REWIND)
                if st["wound"] >= st["w_land"]:
                    st["t_state"], st["t"], st["rc_at_band"] = "reclose", 0, False
            elif ts == "reclose":
                st["t"] += 1
                gpos_now = art.data.joint_pos[0, fj1].item()
                if not st["rc_at_band"]:
                    grip = wide_target()
                    if (p[2].item() - (nx[2].item() + st["grip_off"])) < 1.5e-3 or st["t"] >= SEC(2.0):
                        st["rc_at_band"], st["t"] = True, 0
                elif st["t"] <= T_RECLOSE:
                    grip = pinch_target(pinch_n)
                elif gpos_now > st["p_flat"] + 3e-4 and st["t"] < T_RECLOSE + SEC(2.5):
                    grip = pinch_target(pinch_n)
                    st["wound"] -= 0.6 / CTRL_HZ
                else:
                    st["t_state"] = "wind"
            zt = nx[2].item() + st["grip_off"] - lean
            if ts in ("open", "rewind") and art.data.joint_pos[0, fj1].item() > st["p_flat"] + 3.5e-3:
                zt += 0.008           # frozen: width-gated rewind lift
            zt = max(zt, st["z_contact"] - 0.021)   # frozen: z floor
            if nut_center and st["xy_tgt"] is not None:
                err = bx[:2] - nx[:2]
                if err.norm() > 5e-4:
                    st["xy_tgt"] += (0.01 * err).clamp(-2e-6, 2e-6)   # frozen: thread-phase cap
                tgt_xy = (st["xy_tgt"][0].item(), st["xy_tgt"][1].item())
            else:
                tgt_xy = (bx[0].item(), bx[1].item())
            action = servo(torch.tensor([tgt_xy[0], tgt_xy[1], zt], device=device),
                           yaw_about_z(down, st["wound"]), grip)
            if dz <= STOP_DZ:
                print(f"[done] dz={dz * 1e3:.1f}mm after {st['stroke']} strokes", flush=True)
                phase, ph_t = "finish", 0
        elif phase == "finish":
            zt = min(p[2].item() + 0.03 / CTRL_HZ, st["z_contact"] + 0.08)
            action = servo(torch.tensor([bx[0], bx[1], zt], device=device), down, OPEN)
            if ph_t >= SEC(2.0):
                break

        tstate = st["t_state"] if phase == "thread" else ""
        env.step(action, render=(rec.observe(env, action, phase, tstate)
                                 if rec is not None else False))
        if rec is not None:
            rec.capture(env)
        steps += 1
        ph_t += 1
        # Stop an episode whose nut is no longer on the bolt. A threading nut sits within ~2 mm of
        # the axis; 20 mm for three seconds means it was knocked onto the table, and the phase
        # machine will then wind air for the rest of the budget — ~50 minutes of GPU per dead
        # episode. This stops the RECORDING, exactly like the success check; it does not change
        # what the policy does.
        if phase == "thread":
            lost_t = lost_t + 1 if (nut() - bolt())[:2].norm().item() > 0.020 else 0
            if lost_t >= LOST_FOR:
                print(f"[lost] nut {(nut() - bolt())[:2].norm().item() * 1e3:.1f}mm off the bolt "
                      f"axis for 3s after {st['stroke']} strokes — ending the episode", flush=True)
                lost = True
                break
        if i % LOG_EVERY == 0:
            off = nut() - bolt()
            gpos = art.data.joint_pos[0, fj1].item()
            wz = scene.nuts[0].data.root_ang_vel_w[0, 2].item()
            tag = f"{phase}" + (f"/{st['t_state']}#{st['stroke']}" if phase == "thread" else "")
            print(f"  {i:6d} [{tag:>14s}] dz={off[2] * 1e3:6.2f}mm "
                  f"lat={off[:2].norm() * 1e3:5.1f}mm | wound={math.degrees(st['wound']):+6.1f} "
                  f"eeYaw={ee_u:+7.1f} nutYaw={yaw_acc['nut']:+7.1f} wz={wz:+5.2f} "
                  f"| grip={gpos * 1e3:5.2f}mm p_flat="
                  + (f"{st['p_flat'] * 1e3:5.2f}mm" if st["p_flat"] is not None else "  n/a")
                  + f" seated={int(scene.seated()[0, 0])}", flush=True)

    off = nut() - bolt()
    return {
        "success": bool(scene.success()[0]),
        "seated": int(scene.seated()[0, 0]),
        "dz_mm": round(off[2].item() * 1e3, 2),
        "lat_mm": round(off[:2].norm().item() * 1e3, 2),
        "strokes": st["stroke"],
        "caught": bool(st["caught"]),
        "ctrl_steps": steps,
        "final_phase": phase,
        "lost_nut": lost,
        "t_state": st["t_state"] if phase == "thread" else "",
    }
