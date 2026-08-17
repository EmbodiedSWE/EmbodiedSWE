"""Teleport solution for WindmillTollgateScene (sim_gen task `roll_ball_i358`) — the
task's legitimacy certificate.

The task's load-bearing interactions and how they are executed:

1. SILENCE THE MILL (contact dynamics — never bypassed): the brake beam is
   teleported ONLY through free air to a hover just outside the exterior bore mouth
   (transport = carrying the grasped bar; 26 mm bar in the 80 mm jaw), then a HELD
   wrench emulating the grasp — gravity compensation + lateral/vertical PD onto the
   bore lane + a slow axial velocity servo — threads it through the 34 mm mouth, the
   30 mm wall port and the interior sleeve until the tip crosses the blade sweep
   circle. The beam is RELEASED; the next blade pass slams into its flank and the
   motor stalls against it. The jam itself — blade-on-beam contact against a live
   motor — is pure physics; the scene's pose-window stall detector must verdict it.
2. DELIVER THE BALL (contact dynamics — never bypassed): the ball is teleported
   ONLY along open ground to the apron approach (transport), then bowled up the
   apron with a velocity-regulated CoM force AIMED THROUGH THE AXLE LINE (impact
   lever arm ~0, so the hit cannot kick the stalled rotor), and the force is CUT
   well before the doorway plane: the climb through the door, the deflection off
   the stalled blade's flank, the creep along the flank around the blade tip, and
   the downhill run into the corner pad all happen HANDS-OFF under gravity on the
   tilted floor. No force is ever applied to anything inside the roofed courtyard.
   If a bowl under- or over-shoots, the ball is re-staged (teleported back out to
   the same open-ground start — reversing, in free transport, the door path it
   physically rolled in on) and re-bowled: the delivery that counts is the final,
   entirely hands-off run.

Pod force-frame quirk: some pods rotate an applied wrench by the body's rotation
since its reference orientation (fatal for a ROLLING ball); every push loop probes
the correct `encode_force` mode from measured progress instead of assuming it.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.<task>.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import math
import os
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from .scene import _qapply, encode_force  # noqa: F401  (registers the scene)
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import _qapply, encode_force  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.windmill_tollgate")().build(num_envs=args.num_envs,
                                                      device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback
    # so distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def loc(body) -> torch.Tensor:
        return scene._local(body.data.root_pos_w)[0]

    def report(tag: str) -> None:
        bl = loc(scene.ball)
        kl = loc(scene.beam)
        tip = scene.beam_tip_local()[0]
        print(f"[solve] {tag:12s} | ball=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) beam=({float(kl[0]):+.3f},{float(kl[1]):+.3f},"
              f"{float(kl[2]):.3f}) tip_x={float(tip[0]):+.3f} "
              f"yaw={math.degrees(float(scene.mill_yaw()[0])):+.1f}deg "
              f"stalled={bool(scene.stalled()[0])} "
              f"eng={bool(scene._engaged[0])} sil={bool(scene._silenced[0])} "
              f"ent={bool(scene._entered[0])} on_pad={bool(scene.ball_on_pad()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline --------------------------------------
    step(120)  # free bodies settle onto the ground; the motor spins the mill up
    bl, kl = loc(scene.ball), loc(scene.beam)
    yaw_a = float(scene.mill_yaw()[0])
    print(f"[solve] layout readback (seed {args.seed}): "
          f"ball=({float(bl[0]):+.3f},{float(bl[1]):+.3f}) "
          f"beam=({float(kl[0]):+.3f},{float(kl[1]):+.3f}) "
          f"yaw0={math.degrees(yaw_a):+.1f}deg "
          f"spin_sign={float(scene.spin_sign[0]):+.0f}", flush=True)
    report("reset")
    assert c.ball_x_rng[0] - 0.02 < float(bl[0]) < c.ball_x_rng[1] + 0.02
    assert c.ball_y_rng[0] - 0.02 < float(bl[1]) < c.ball_y_rng[1] + 0.02
    assert c.beam_x_rng[0] - 0.03 < float(kl[0]) < c.beam_x_rng[1] + 0.03
    # the mill must actually be running (the whole point of the toll-gate)
    step(60)
    dyaw = math.atan2(math.sin(float(scene.mill_yaw()[0]) - yaw_a),
                      math.cos(float(scene.mill_yaw()[0]) - yaw_a))
    print(f"[solve] mill check: {math.degrees(dyaw):+.1f} deg over 0.5 s", flush=True)
    assert abs(dyaw) > 0.5, "mill is not turning at reset"
    assert dyaw * float(scene.spin_sign[0]) > 0, "spin direction readback mismatch"
    assert not bool(scene.stalled()[0]), "free-running mill must not read stalled"
    s0 = print_score("P0 reset+settle (mill running, both free bodies outside)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.02, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: THREAD THE BRAKE BEAM (held wrench, contact) -----------------
    # Transport: ONE pose write carries the grasped beam through free air to a hover
    # just outside the exterior mouth, long axis on the bore lane. Everything after
    # is contact: the held insertion, the blade slam, the stall.
    court = scene._court_pos(all_ids)
    zm = (c.chan_z0 + c.chan_z1) / 2                    # 0.105 lane centre
    tip_start = c.wall_in + c.wall_t + c.sleeve_out_len + 0.015
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = court[:, 0] + tip_start + c.beam_len / 2
    st[:, 1] = court[:, 1] + c.chan_y
    st[:, 2] = court[:, 2] + zm
    st[:, 3] = 1.0
    scene.beam.write_root_state_to_sim(st, all_ids)
    q_ref_beam = scene.beam.data.root_quat_w.clone()

    m_beam = c.beam_density * c.beam_len * c.beam_s * c.beam_s   # ~1.05 kg
    grav = m_beam * 9.81
    kp, kd = 60.0, 8.0            # lateral/vertical hold (K*dt/m well under 1)
    kr, kw = 0.4, 0.05            # small-angle orientation hold
    kx, v_des = 40.0, -0.06       # axial velocity servo (slow, regulated)
    f_ax_cap = 5.0
    tip_goal = c.engage_x - 0.007  # push a hair past the rubric line
    mode = 0
    win_i, win_x = 0, float(scene.beam_tip_local()[0, 0])
    stalls = 0
    inserted = False
    for i in range(2400):
        tip_x = float(scene.beam_tip_local()[0, 0])
        if tip_x < tip_goal:
            inserted = True
            break
        p = scene.beam.data.root_pos_w
        v = scene.beam.data.root_lin_vel_w
        w = scene.beam.data.root_ang_vel_w
        q = scene.beam.data.root_quat_w
        tgt_y = court[:, 1] + c.chan_y
        tgt_z = court[:, 2] + zm
        f_world = torch.zeros(n, 3, device=device)
        f_world[:, 0] = (kx * (v_des - v[:, 0])).clamp(-f_ax_cap, f_ax_cap)
        f_world[:, 1] = (kp * (tgt_y - p[:, 1]) - kd * v[:, 1]).clamp(-8.0, 8.0)
        f_world[:, 2] = (grav + kp * (tgt_z - p[:, 2]) - kd * v[:, 2]).clamp(0.0, 20.0)
        # orientation hold: small-angle rotvec PD toward identity (world frame)
        sgn = torch.sign(q[:, :1]) + (q[:, :1] == 0).float()
        rotvec = 2.0 * q[:, 1:] * sgn
        t_world = (-kr * rotvec - kw * w).clamp(-0.5, 0.5)
        f_arg = encode_force(mode, q_ref_beam, q, f_world)
        t_arg = encode_force(mode, q_ref_beam, q, t_world)
        scene.beam.set_external_force_and_torque(
            f_arg.view(n, 1, 3).contiguous(), t_arg.view(n, 1, 3).contiguous(),
            env_ids=all_ids, is_global=True)
        env.step(no_action)
        if i - win_i >= 40:  # progress probe: wrong force-frame mode shows up as
            # regression (tip moving AWAY from the wall) or a persistent stall
            x = float(scene.beam_tip_local()[0, 0])
            if x > win_x + 0.006:
                mode = 1 - mode
                stalls = 0
                print(f"[solve] insert: moving away (tip {win_x:+.3f} -> {x:+.3f}); "
                      f"force-frame mode -> {mode}", flush=True)
            elif x > win_x - 0.002:
                stalls += 1
                if stalls >= 3:
                    mode = 1 - mode
                    stalls = 0
                    f_ax_cap = 5.0
                    print(f"[solve] insert: persistent stall at tip={x:+.3f}; "
                          f"force-frame mode -> {mode}", flush=True)
                else:
                    f_ax_cap = min(f_ax_cap + 1.0, 9.0)
                    print(f"[solve] insert: stalled at tip={x:+.3f} -> "
                          f"cap={f_ax_cap:.1f} N (a passing blade can block the "
                          f"tip for a beat)", flush=True)
            else:
                stalls = 0
            win_i, win_x = i, x
    clear(scene.beam)
    report("inserted")
    assert inserted, "beam insertion never reached the engagement depth"
    # release: the beam settles onto the bore floor, tail in the sleeves; the
    # motor-driven blade now slams into the flank and must stall against it.
    stalled_ok = False
    for i in range(1800):
        env.step(no_action)
        if bool(scene.stalled()[0]) and bool(scene._silenced[0]):
            stalled_ok = True
            break
        if i % 300 == 299:
            print(f"[solve] waiting for stall: "
                  f"yaw={math.degrees(float(scene.mill_yaw()[0])):+.1f}deg "
                  f"tip_x={float(scene.beam_tip_local()[0, 0]):+.3f}", flush=True)
    report("stalled")
    assert stalled_ok, "mill never stalled against the released beam"
    assert bool(scene.beam_engaged_now()[0]), "beam left the engagement pose"
    s1 = print_score("P1 beam threaded, mill jammed + silenced (blade-on-beam)")
    assert s1 >= s0 - 1e-6 and s1 >= 0.49, f"expected engaged+silenced, got {s1}"

    # ---------------- phase 2: BOWL THE BALL through the doorway ----------------------------
    # Transport: pose writes only along OPEN GROUND outside the courtyard (start of
    # each attempt). The bowl itself is a regulated CoM push, aimed through the
    # axle line (x = axle_x: impact lever arm ~0 on the stalled rotor) and CUT at
    # y < door_cut, well outside the wall plane — from there the climb, the
    # doorway, the flank ride and the corner drop are hands-off physics.
    aim_x = c.axle_x                                    # -0.04: through the axle
    start = torch.zeros(n, 13, device=device)
    start[:, 0] = court[:, 0] + aim_x
    start[:, 1] = court[:, 1] - 0.41
    start[:, 2] = court[:, 2] + c.ball_r + 0.001
    start[:, 3] = 1.0
    door_cut = -(c.wall_in + c.wall_t) - 0.02           # cut all force here (-0.22)
    delivered = False
    mode_b = 0
    for attempt, v_bowl in enumerate((0.55, 0.65, 0.50, 0.78, 0.90)):
        scene.ball.write_root_state_to_sim(start, all_ids)
        q_ref_ball = scene.ball.data.root_quat_w.clone()
        step(10)  # touch down
        win_i, win_y = 0, float(loc(scene.ball)[1])
        stalls_b = 0
        pushing = True
        for i in range(2400):
            bl3 = loc(scene.ball)
            by = float(bl3[1])
            if pushing and by > door_cut:
                pushing = False
                clear(scene.ball)
                print(f"[solve] bowl {attempt}: hands off at y={by:+.3f} "
                      f"v={float(scene.ball.data.root_lin_vel_w[0, 1]):.2f} m/s",
                      flush=True)
            if pushing:
                v = scene.ball.data.root_lin_vel_w
                p = scene.ball.data.root_pos_w
                f_world = torch.zeros(n, 3, device=device)
                f_world[:, 1] = (3.0 * (v_bowl - v[:, 1])).clamp(-2.5, 2.5)
                f_world[:, 0] = (6.0 * (court[:, 0] + aim_x - p[:, 0])
                                 - 1.2 * v[:, 0]).clamp(-1.5, 1.5)
                f_arg = encode_force(mode_b, q_ref_ball,
                                     scene.ball.data.root_quat_w, f_world)
                scene.ball.set_external_force_and_torque(
                    f_arg.view(n, 1, 3).contiguous(), zero_wrench,
                    env_ids=all_ids, is_global=True)
                env.step(no_action)
                if i - win_i >= 40:  # rolling-body force-frame probe on y progress
                    if by < win_y - 0.006:
                        mode_b = 1 - mode_b
                        stalls_b = 0
                        print(f"[solve] bowl: moving away (y {win_y:+.3f} -> "
                              f"{by:+.3f}); force-frame mode -> {mode_b}", flush=True)
                    elif by < win_y + 0.004:
                        stalls_b += 1
                        if stalls_b >= 2:
                            mode_b = 1 - mode_b
                            stalls_b = 0
                            print(f"[solve] bowl: persistent stall at y={by:+.3f}; "
                                  f"force-frame mode -> {mode_b}", flush=True)
                    else:
                        stalls_b = 0
                    win_i, win_y = i, by
                continue
            # hands-off: gravity on the tilted floor does the delivery
            env.step(no_action)
            if bool(scene.success()[0]):
                delivered = True
                break
            still = float(scene.ball.data.root_lin_vel_w[0].norm()) < 0.04
            if still and by < -0.30:
                print(f"[solve] bowl {attempt}: rolled back out (y={by:+.3f})",
                      flush=True)
                break  # under-shot: never made it in — retry faster
            if still and i > 600 and bool(scene.ball_inside()[0]) \
                    and not bool(scene.ball_on_pad()[0]):
                print(f"[solve] bowl {attempt}: parked off-pad at "
                      f"({float(bl3[0]):+.3f},{by:+.3f})", flush=True)
                break  # parked somewhere wrong — re-stage and retry
        if delivered:
            print(f"[solve] bowl {attempt}: delivered at v_bowl={v_bowl:.2f}",
                  flush=True)
            break
        report(f"bowl{attempt}-end")
    clear(scene.ball)
    report("delivered")
    assert delivered, "no bowl attempt delivered the ball to the pad"
    s2 = print_score("P2 ball bowled through the doorway onto the corner pad")
    assert s2 >= s1 - 1e-6, "score decreased across delivery"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after delivery)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3.3 simulated seconds, hands-off) ------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.3 s (motor still driving against the beam)")
    ok = hold and bool(scene.success()[0]) and s3 >= s2 - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001 — Kit teardown hangs; die loudly now
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({exc})", flush=True)
        os._exit(1)
