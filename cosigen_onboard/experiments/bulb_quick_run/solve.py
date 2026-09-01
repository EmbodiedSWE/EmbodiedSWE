"""solve — Fable's scripted solver for assembly.bulb.franka.osc.

The bulb spawns LYING on its side (screw axis horizontal) — unlike the nut task, the part must be
reoriented 90 deg before it can be threaded. A rigidly-held 90 deg flip ends with the hand
HORIZONTAL, and you cannot screw with a horizontal hand (that rotation axis is not the wrist roll),
so the plan is place-then-regrasp:

  grasp the glass belly (= the COM, so gravity exerts ~no torque in the pinch) with the hand down and
  the fingers closing across the bulb axis -> lift -> ramp the goal orientation -90 deg about the
  finger-close axis (the pinch never fights the turn) so the cap swings down, hand ends horizontal ->
  servo the BULB origin (live feedback, not the hand) onto the socket axis -> lower with touch-detect
  until the cap rests in the bore -> release, retreat -> the bulb stands in the socket mouth ->
  regrasp from straight above -> thread with wrist-roll wind strokes.

Threading differs from the nut in one key way: the glass is ROUND, so torque transmission is pure
pad FRICTION (no hex wrench geometry). That kills the whole hex-alignment machinery (reclose at any
angle, no corner cam-back, no corner-bind) but makes coupling pinch-force-limited: firm pinch for
strokes; slip is graceful (a slipping clutch still delivers its max friction torque, which is the
same press+twist regime the free-body smoke threads with). The z closed-loop follows the bulb down
at the measured grip offset, with a small lean as the press.

Run from the repo root:

    .venv/bin/python experiments/bulb_franka_osc_fable/solve.py --headless
    .venv/bin/python experiments/bulb_franka_osc_fable/solve.py --livestream 2
"""

from __future__ import annotations

import argparse
import math
import os
import threading

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--max_sec", type=float, default=220.0, help="sim-time budget (s)")
parser.add_argument("--carry_n", type=float, default=40.0, help="pinch per finger while carrying/reorienting (N)")
parser.add_argument("--pinch_wind_n", type=float, default=30.0, help="pinch per finger for wind strokes (N) — sets the friction torque budget")
parser.add_argument("--lean", type=float, default=0.003, help="z press-lean below the follow height while winding (m)")
parser.add_argument("--wind_rate", type=float, default=2.0, help="tighten rate (rad/s)")
parser.add_argument("--rewind_rate", type=float, default=3.0, help="unwind rate (rad/s)")
parser.add_argument("--sweep_deg", type=float, default=120.0, help="stroke sweep (deg)")
parser.add_argument("--stop_h", type=float, default=0.024, help="stop threading once bulb-above-socket <= this (m; seat_z=0.027, full seat ~0.0218)")
parser.add_argument("--bulb_friction", type=float, default=-1.0, help="override the scene's bulb friction dial (<0 = stock 0.01)")
parser.add_argument("--dt", type=float, default=0.0, help="sim dt override (0 = scene default 1/240)")
parser.add_argument("--torque_dt", type=float, default=-1.0, help="OSC target-latch dt; -1 = latch every physics step")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
livestream_on = args.livestream > 0

app = AppLauncher(args).app

import torch  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
)

import robobench  # noqa: E402
from robobench.core import EnvCfg  # noqa: E402
from robobench.robots.franka import FrankaRobotCfg  # noqa: E402
from robobench.suites.assembly.scenes import BulbAssemblySceneCfg  # noqa: E402

OPEN = 0.04                 # finger position target, fully open (m)
GRIP_KP = 8000.0            # gripper stiffness we build the robot with (pinch force = KP * overshoot)
PAD_TIP = 0.103             # panda_hand -> pad TIPS along the approach axis (nut session, measured)
GRASP_DROP = 0.099          # panda_hand -> pad contact-band centre (tips + ~4mm band)
PALM = 0.066                # panda_hand -> palm/knuckle underside (r3: hand bottomed at dome+0.066
                            # exactly). HARD LIMIT: pads can never reach below ~2mm above the bulb's
                            # equator when it stands — the palm hits the dome first.
BELLY = 0.044               # bulb origin (cap end) -> glass belly = COM along the bulb axis (baked USD)
NECK = 0.025                # bulb origin -> neck waist (between the Ø20 cap and the glass). The PICK
                            # grasps here: a waist is self-centering and cannot squeeze out, unlike the
                            # sloped glass barrel (r1/r2: flat pads on the barrel watermelon-seed it).
NECK_CLOSE = 0.007          # finger target at the waist (r1 equilibrium: 9.9mm at ~23 N, rigid carry)
GLASS_R = 0.024             # glass half-width at the belly
BULB_TOP = 0.083            # bulb origin -> glass dome tip (probe bbox)
REST_H = 0.0323             # bulb origin above socket origin after a free release onto the thread (probe)
SEAT_H = 0.0218             # ... fully seated
PSI = math.pi / 2           # hand-down yaw: fingers close along world x (perp. to the lying bulb's y axis)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    if args.torque_dt != 0:  # latch the OSC pose target faster than the 15 Hz default
        from robobench.robots.franka import FrankaRobot

        dt_eff = args.dt if args.dt > 0 else 1.0 / 240.0
        FrankaRobot.TORQUE_CONTROL_DT = dt_eff if args.torque_dt < 0 else args.torque_dt
    # Natural mounted layout (2026-07-14, was surface_z=0.20 + bulb_row_x0=-0.12 which sank the robot
    # 20 cm into the table on camera): robot base IN the table's mount cutout at (-0.09, 0, 0), table
    # top at the scene default z=0. The work is pulled toward the base to restore the session's
    # verified reach band on the flat table — a first try with the bulb at 0.69 m from the base ended
    # REORIENT_STUCK (descend stalled 60 mm high; too far for a table-level top-down pick). Now:
    # socket 0.50 m ahead of the base, bulb beside it at the verified 0.43 m pick radius.
    # Bulb on the +y side: at -y the descend pose parks the wrist at q7=-2.11 (0.8 rad from the stop
    # the wind strokes turn toward); mirrored, q7 rests at +0.46 with maximal margin.
    scene_kw = dict(socket_slots=((-0.09, 0.0),), bulb_init_xy=((-0.24, 0.25),))
    if args.bulb_friction >= 0:
        scene_kw["bulb_friction"] = args.bulb_friction
    scene_cfg = BulbAssemblySceneCfg(**scene_kw)
    robot_cfg = FrankaRobotCfg(nullspace_dof_pos=(), gripper_stiffness=GRIP_KP)
    sim_overrides = {"dt": args.dt} if args.dt > 0 else {}
    env = EnvCfg(
        scene="bulb", scene_cfg=scene_cfg, robot="franka", robot_cfg=robot_cfg, control_mode="osc", env_spacing=2,
        sim_overrides=sim_overrides,
    ).build(num_envs=1, device=device)
    scene, robot = env.scene, env.robot
    osc = robot.controller.controllers[0]
    # per-step latching caps usable torque at ~kp_rot * rot_scale; nut session: 600 * 0.15 converges.
    osc._kp = torch.tensor([150.0, 150.0, 150.0, 600.0, 600.0, 600.0], device=device)
    osc._kd = 2.0 * osc._kp.sqrt()
    osc.cfg.rot_scale = 0.15
    art = robot.articulation
    ee_idx = art.body_names.index("panda_hand")
    fj1 = art.find_joints(["panda_finger_joint1"])[0]
    render = (not args.headless) or livestream_on
    n, dim = env.num_envs, robot.action_dim
    DECIM = robot.control_period
    CTRL_HZ = 1.0 / (env.dt * DECIM)
    down = quat_from_euler_xyz(*(torch.tensor([v], device=device) for v in (math.pi, 0.0, PSI)))[0]
    print(f"[cfg] dt={env.dt:.5f} control={CTRL_HZ:.0f}Hz | carry={args.carry_n}N wind_pinch={args.pinch_wind_n}N "
          f"lean={args.lean * 1e3:.0f}mm sweep={args.sweep_deg}deg bulb_mu={scene.cfg.bulb_friction}", flush=True)
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

    bulb, socket = scene.bulbs[0], scene.sockets[0]

    def bulb_pos():
        return bulb.data.root_pos_w[0]

    def sock_pos():
        return socket.data.root_pos_w[0]

    def bulb_h():  # bulb origin above socket origin (m)
        return (bulb_pos() - sock_pos())[2].item()

    def bulb_tilt():  # deg off vertical
        up = quat_apply(bulb.data.root_quat_w, torch.tensor([[0.0, 0.0, 1.0]], device=device))[0]
        return math.degrees(math.acos(max(-1.0, min(1.0, up[2].item()))))

    def bulb_axis():  # world direction of the bulb's local +z (cap -> dome)
        return quat_apply(bulb.data.root_quat_w, torch.tensor([[0.0, 0.0, 1.0]], device=device))[0]

    def yaw_of(q):
        ex = quat_apply(q.unsqueeze(0), torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        return math.degrees(math.atan2(ex[1].item(), ex[0].item()))

    def yaw_about_z(qbase, ang):
        yz = quat_from_euler_xyz(*(torch.tensor([v], device=device) for v in (0.0, 0.0, ang)))
        return quat_mul(yz, qbase.unsqueeze(0))[0]

    def pitch_about_x(qbase, ang):  # world-side rotation about x (the finger-close axis)
        qx = quat_from_euler_xyz(*(torch.tensor([v], device=device) for v in (ang, 0.0, 0.0)))
        return quat_mul(qx, qbase.unsqueeze(0))[0]

    V = lambda x, y, z: torch.tensor([x, y, z], device=device)

    SEC = lambda s: max(1, round(s * CTRL_HZ))
    # hover/descend lengthened 1.7/1.5 -> 3.0/3.0 for the flat-table layout: the pick pose needs a
    # bigger posture swing from home than the old raised-surface band (probe: ~5 s to converge).
    T_SETTLE, T_HOVER, T_DESCEND, T_GRASP, T_LIFT = (SEC(s) for s in (0.5, 3.0, 3.0, 1.0, 1.5))
    T_REOR, T_CARRY, T_RELEASE, T_RETREAT, T_REHOVER, T_REDESC, T_PINCH = (
        SEC(s) for s in (2.5, 1.5, 0.6, 1.5, 2.5, 2.0, 0.6))
    T_OPEN, T_RECLOSE, T_QUIET = SEC(0.4), SEC(0.5), SEC(0.4)
    LOWER_MPS = 0.012
    SWEEP = math.radians(args.sweep_deg)
    D_WIND, D_REWIND = args.wind_rate / CTRL_HZ, args.rewind_rate / CTRL_HZ

    phase, ph_t = "settle", 0
    st = {
        "p_glass": None,      # finger joint pos pinching the glass belly (m)
        "grip_off": None,     # hand_z - bulb_z at the top regrasp (z-follow offset)
        "lower_z": None,      # descending EE-target z during touch-detect insert
        "bulb_hist": [],      # recent bulb z (touch detect)
        "theta": 0.0,         # reorient pitch ramp (rad)
        "carry_p": None,      # grasp-point world target held during the reorient
        "rel_open": None,     # release: finger target = measured width + real clearance
        "rel_p": None,        # EE pose frozen at release entry (don't drift into the bulb)
        "t_state": "pinch", "t": 0, "wound": 0.0, "stroke": 0,
        "ee0": None, "stall_t": 0, "w_start": 0.0, "stall_run": 0,
        "yaw0_ee": 0.0, "yaw0_bulb": 0.0,  # per-stroke coupling metric
        "z_floor": None,
    }
    yaw_acc = {"ee": 0.0, "bulb": 0.0, "ee_prev": None, "bulb_prev": None}

    def unwrap(key, val):
        prev = yaw_acc[key + "_prev"]
        if prev is not None:
            yaw_acc[key] += (val - prev + 180.0) % 360.0 - 180.0
        yaw_acc[key + "_prev"] = val
        return yaw_acc[key]

    def pinch(newtons):
        return st["p_glass"] - newtons / GRIP_KP

    def wide():
        return st["p_glass"] + 0.006  # 6 mm off the glass per side — clears the Ø48 during rewinds

    LOG_EVERY = SEC(0.8)
    verdict = "TIMEOUT"

    for i in range(1, SEC(args.max_sec) + 1):
        bp, sp = bulb_pos(), sock_pos()
        h = bulb_h()
        p, q = ee_pose()
        ee_u = unwrap("ee", yaw_of(q))
        bulb_u = unwrap("bulb", yaw_of(bulb.data.root_quat_w[0]))
        d = bulb_axis()
        belly = bp + BELLY * d  # widest glass ring (= COM) — regrasp target on the STANDING bulb
        neck = bp + NECK * d    # neck waist — pick target on the LYING bulb (self-centering)

        if phase == "settle":
            action = servo(p, down, OPEN)
            if ph_t >= T_SETTLE:
                phase, ph_t = "hover", 0
        elif phase == "hover":  # over the lying bulb's neck
            action = servo(neck + V(0, 0, GRASP_DROP + 0.07), down, OPEN)
            if ph_t >= T_HOVER:
                phase, ph_t = "descend", 0
        elif phase == "descend":
            action = servo(neck + V(0, 0, GRASP_DROP), down, OPEN)
            if ph_t >= T_DESCEND:
                err = (neck + V(0, 0, GRASP_DROP) - p)
                print(f"[dbg] descend err mm: ({err[0] * 1e3:+.1f},{err[1] * 1e3:+.1f},{err[2] * 1e3:+.1f})", flush=True)
                phase, ph_t = "grasp", 0
        elif phase == "grasp":  # close onto the waist: the concave neck self-centers and cannot eject
            # (r1/r2: any pinch on the sloped Ø48 barrel watermelon-seeds the bulb out of the jaws)
            action = servo(neck + V(0, 0, GRASP_DROP), down, NECK_CLOSE)
            if ph_t >= T_GRASP:
                st["p_glass"] = art.data.joint_pos[0, fj1].item()
                print(f"[cal] neck grip: finger={st['p_glass'] * 1e3:.2f}mm (width {2 * st['p_glass'] * 1e3:.1f}mm)", flush=True)
                phase, ph_t = "lift", 0
        elif phase == "lift":  # straight up to the carry height, holding the carry pinch
            tgt = V(p[0].item(), p[1].item(), sp[2].item() + 0.28 + GRASP_DROP)
            action = servo(tgt, down, pinch(args.carry_n))
            if ph_t >= T_LIFT:
                phase, ph_t = "reorient", 0
                st["carry_p"] = V(p[0].item(), p[1].item(), sp[2].item() + 0.28)  # grasp-point target (hand is DROP above)
        elif phase == "reorient":  # closed-loop pitch about world x until the BULB axis reads vertical
            # (the lying bulb settles TILTED — cap end droops to the table — so the needed swing is
            # measured, not the nominal 90 deg. delta = residual rotation about x that verticalizes it.)
            delta = math.atan2(d[1].item(), d[2].item())
            rate = (math.pi / 2) / T_REOR
            st["theta"] = max(-2.3, min(0.2, st["theta"] + max(-rate, min(rate, delta))))
            th = st["theta"]
            goal_q = pitch_about_x(down, th)
            # EE = grasp point - DROP * approach(theta), approach(theta) = (0, sin, -cos)
            ee_t = st["carry_p"] + V(0.0, -GRASP_DROP * math.sin(th), GRASP_DROP * math.cos(th))
            action = servo(ee_t, goal_q, pinch(args.carry_n))
            if abs(delta) < math.radians(3.0) and ph_t >= T_REOR:
                print(f"[reor] bulb vertical: hand pitch {math.degrees(th):+.0f}deg tilt={bulb_tilt():.1f}deg", flush=True)
                phase, ph_t = "carry", 0
            elif ph_t >= 4 * T_REOR:
                verdict = "REORIENT_STUCK"
                print(f"[abort] reorient stuck: pitch {math.degrees(th):+.0f}deg tilt={bulb_tilt():.1f}deg", flush=True)
                phase, ph_t = "abort", 0
        elif phase == "carry":  # servo the BULB ORIGIN over the socket axis, 60 mm up
            goal_q = pitch_about_x(down, st["theta"])
            err = sp + V(0, 0, 0.06) - bp  # where the bulb origin should be, minus where it is
            ee_t = p + err
            action = servo(ee_t, goal_q, pinch(args.carry_n))
            if ph_t >= T_CARRY and err[:2].norm().item() < 0.004:
                phase, ph_t = "insert", 0
                st["lower_z"] = p[2].item()
                st["bulb_hist"] = []
        elif phase == "insert":  # touch-detect lower: bulb z quiets while the target keeps sinking
            goal_q = pitch_about_x(down, st["theta"])
            st["lower_z"] -= LOWER_MPS / CTRL_HZ
            st["bulb_hist"].append(bp[2].item())
            st["bulb_hist"] = st["bulb_hist"][-T_QUIET:]
            err_xy = (sp - bp)[:2]
            ee_t = V(p[0].item() + err_xy[0].item(), p[1].item() + err_xy[1].item(), st["lower_z"])
            action = servo(ee_t, goal_q, pinch(args.carry_n))
            quiet = len(st["bulb_hist"]) == T_QUIET and (max(st["bulb_hist"]) - min(st["bulb_hist"])) < 2e-4
            if (quiet and st["lower_z"] < p[2].item() - 0.003 and h < REST_H + 0.02) or h < REST_H - 0.001:
                print(f"[touch] cap down in the bore: h={h * 1e3:.1f}mm lat={(bp - sp)[:2].norm() * 1e3:.1f}mm "
                      f"tilt={bulb_tilt():.1f}deg", flush=True)
                phase, ph_t = "release", 0
        elif phase == "release":  # open relative to the MEASURED width (r1 opened vs a stale reading
            # and left ~0.1 mm of clearance — the retreat then dragged the bulb over)
            if st["rel_open"] is None:
                st["rel_open"] = art.data.joint_pos[0, fj1].item() + 0.010
                st["rel_p"] = p.clone()
            goal_q = pitch_about_x(down, st["theta"])
            action = servo(st["rel_p"], goal_q, st["rel_open"])
            if ph_t >= T_RELEASE:
                phase, ph_t = "backout", 0
        elif phase == "backout":  # slide the jaws off along the hand approach axis (horizontal), no turning
            goal_q = pitch_about_x(down, st["theta"])
            ee_t = st["rel_p"] + V(0.0, 0.09 * min(1.0, ph_t / SEC(1.0)), 0.0)
            action = servo(ee_t, goal_q, st["rel_open"])
            if ph_t >= SEC(1.2):
                phase, ph_t = "retreat", 0
        elif phase == "retreat":  # now clear of the bulb: up + away while re-pointing down
            frac = min(1.0, ph_t / T_RETREAT)
            goal_q = pitch_about_x(down, st["theta"] * (1.0 - frac))  # swing back to hand-down as we leave
            ee_t = V(sp[0].item(), sp[1].item() + 0.14, sp[2].item() + 0.18 + 0.17 * frac)
            action = servo(ee_t, goal_q, OPEN)
            if ph_t >= T_RETREAT + SEC(0.5):
                tl = bulb_tilt()
                print(f"[stand] released bulb: h={h * 1e3:.1f}mm lat={(bp - sp)[:2].norm() * 1e3:.1f}mm tilt={tl:.1f}deg", flush=True)
                if tl > 40.0 or h < 0.020:
                    verdict = "FELL_ON_RELEASE"
                    phase, ph_t = "abort", 0
                else:
                    phase, ph_t = "rehover", 0
        elif phase == "rehover":  # straight above the standing bulb, hand down, fingers wide
            ee_t = V(sp[0].item(), sp[1].item(), sp[2].item() + REST_H + BULB_TOP + PAD_TIP + 0.02)
            action = servo(ee_t, down, OPEN)
            if ph_t >= T_REHOVER:
                phase, ph_t = "redescend", 0
        elif phase == "redescend":  # straddle the glass, palm 1 mm above the dome (the physical floor —
            # r4 aimed the band below the equator and the palm ploughed the dome, knocking the bulb out).
            # The band then sits ~2-6 mm above the equator; μ (bulb_friction dial) must self-lock it.
            ee_t = V(sp[0].item(), sp[1].item(), bp[2].item() + BULB_TOP + PALM + 0.001)
            action = servo(ee_t, down, OPEN)
            if ph_t >= T_REDESC and abs((ee_t - p)[2].item()) < 0.002:
                phase, ph_t = "thread", 0
                st["t_state"], st["t"] = "pinch", 0
            elif ph_t >= 3 * T_REDESC:  # convergence timeout: pinch anyway
                phase, ph_t = "thread", 0
                st["t_state"], st["t"] = "pinch", 0
        elif phase == "thread":
            ts = st["t_state"]
            grip = pinch(args.pinch_wind_n)
            lean = 0.0
            if ts == "pinch":  # absolute force-calibrated close, then re-measure the width in THIS grip
                grip = GLASS_R - args.pinch_wind_n / GRIP_KP
                st["t"] += 1
                if st["t"] >= T_PINCH:
                    st["p_glass"] = art.data.joint_pos[0, fj1].item()  # rigid contact: joint pos = glass surface
                    st["grip_off"] = p[2].item() - bp[2].item()
                    st["z_floor"] = sp[2].item() + SEAT_H + st["grip_off"] - 0.002
                    print(f"[cal] regrasp: grip={art.data.joint_pos[0, fj1].item() * 1e3:.2f}mm "
                          f"(ref {st['p_glass'] * 1e3:.2f}) grip_off={st['grip_off'] * 1e3:.1f}mm", flush=True)
                    st["t_state"], st["ee0"] = "wind", None
            elif ts == "wind":  # tighten stroke, paced by the real EE yaw
                if st["ee0"] is None:
                    st["ee0"], st["stall_t"], st["w_start"], st["stall_run"] = ee_u, 0, st["wound"], 0
                    st["yaw0_ee"], st["yaw0_bulb"] = ee_u, bulb_u
                swept = math.radians(st["ee0"] - ee_u)  # actual CW turn of the jaws this stroke (rad)
                lag = st["w_start"] - st["wound"] - swept
                lean = args.lean
                st["stall_run"] = st["stall_run"] + 1 if lag >= 0.35 else 0
                w_end = st["w_start"] - SWEEP
                if st["stall_run"] >= SEC(2.5):  # binding: end the stroke, recycle the grip
                    print(f"[wind] stroke {st['stroke']} BOUND (lag {math.degrees(lag):.0f}deg) at "
                          f"wound {math.degrees(st['wound']):.0f} — recycling", flush=True)
                    st["t_state"], st["t"], st["ee0"] = "open", 0, None
                elif st["wound"] > w_end:
                    if lag < 0.35:
                        st["wound"] = max(w_end, st["wound"] - D_WIND)
                else:
                    st["stall_t"] += 1
                    if lag < 0.26 or st["stall_t"] >= SEC(1.5):
                        st["t_state"], st["t"], st["ee0"] = "open", 0, None
            elif ts == "open":
                grip, st["t"] = wide(), st["t"] + 1
                if st["t"] == 1:  # stroke report: how much of the jaw turn the bulb followed
                    dee, dbu = st["yaw0_ee"] - ee_u, st["yaw0_bulb"] - bulb_u
                    cpl = (dbu / dee * 100.0) if abs(dee) > 5 else float("nan")
                    print(f"[stroke {st['stroke']}] jaws {dee:+.0f}deg bulb {dbu:+.0f}deg ({cpl:.0f}% coupled) "
                          f"h={h * 1e3:.2f}mm", flush=True)
                    st["stroke"] += 1
                if st["t"] >= T_OPEN:
                    st["t_state"] = "rewind"
            elif ts == "rewind":  # unwind while clear of the glass; land at +45 for the next stroke
                grip = wide()
                st["wound"] = min(math.radians(45.0), st["wound"] + D_REWIND)
                if st["wound"] >= math.radians(45.0):
                    st["t_state"], st["t"] = "reclose", 0
            elif ts == "reclose":  # round glass: reclose at any angle, straight to firm
                st["t"] += 1
                if st["t"] >= T_RECLOSE:
                    st["t_state"], st["ee0"] = "wind", None
            zt = bp[2].item() + st["grip_off"] - lean if st["grip_off"] is not None else p[2].item()
            if st["z_floor"] is not None:
                zt = max(zt, st["z_floor"])
            action = servo(V(sp[0].item(), sp[1].item(), zt), yaw_about_z(down, st["wound"]), grip)
            if bulb_tilt() > 30.0:
                verdict = "TILT_JAM"
                print(f"[abort] bulb tilt {bulb_tilt():.0f}deg at h={h * 1e3:.1f}mm", flush=True)
                phase, ph_t = "abort", 0
            elif h <= args.stop_h:
                print(f"[done] h={h * 1e3:.1f}mm after {st['stroke']} strokes", flush=True)
                phase, ph_t = "finish", 0
        elif phase == "finish":  # open, rise, settle — the seated bulb must hold on its own
            zt = min(p[2].item() + 0.03 / CTRL_HZ, sp[2].item() + 0.35)
            action = servo(V(sp[0].item(), sp[1].item(), zt), down, OPEN)
            if ph_t >= SEC(2.0):
                verdict = "DONE"
                break
        elif phase == "abort":
            action = servo(V(p[0].item(), p[1].item(), sp[2].item() + 0.35), down, OPEN)
            if ph_t >= SEC(1.5):
                break

        env.step(action, render=render)
        ph_t += 1

        if i % LOG_EVERY == 0:
            off = bulb_pos() - sock_pos()
            gpos = art.data.joint_pos[0, fj1].item()
            wz = bulb.data.root_ang_vel_w[0, 2].item()
            tag = f"{phase}" + (f"/{st['t_state']}#{st['stroke']}" if phase == "thread" else "")
            print(f"  {i:5d} [{tag:>14s}] h={off[2] * 1e3:6.2f}mm lat={off[:2].norm() * 1e3:4.1f}mm "
                  f"tilt={bulb_tilt():4.1f} | wound={math.degrees(st['wound']):+6.1f} eeYaw={ee_u:+7.1f} "
                  f"bulbYaw={bulb_u:+7.1f} wz={wz:+5.2f} | grip={gpos * 1e3:5.2f}mm "
                  f"seated={int(scene.seated()[0, 0])}", flush=True)

    off = bulb_pos() - sock_pos()
    seated = int(scene.seated()[0, 0])
    print(f"SOLVE[fable] | {verdict} | seated={seated} | h={off[2] * 1e3:.1f}mm lat={off[:2].norm() * 1e3:.1f}mm "
          f"tilt={bulb_tilt():.1f}deg | strokes={st['stroke']}", flush=True)
    threading.Timer(10.0, lambda: os._exit(0)).start()  # Isaac teardown hangs; free the GPU regardless
    env.close()


if __name__ == "__main__":
    main()
    app.close()
    os._exit(0)
