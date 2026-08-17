"""Teleport solution for SauceChuteScene (sim_gen task
`living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray_i166`) — the
task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. DOCK (transport teleport): one pose write carries the EMPTY tray from its jittered
   spawn zone across free space to the catch dock on the ground under the spout. It
   settles through real ground contact; `tray_docked` (position + upright + still) is
   asserted on the settled pose, not at the write instant.
2. PUSH (applied force — the fingertip surrogate): a gentle velocity-servo force
   (F_y = clamp(K*(v_des - v_y), -6 N, 0), K = 15, v_des = -0.22 m/s, K*dt/m = 0.36)
   is applied at the RED can's centre, world -y, ONLY while the can is within the
   30 mm stroke a fingertip poked through the 55 x 50 mm rear port could reach
   (can y > cradle_y - 0.030). This pod's set_external_force_and_torque rotates
   wrenches by the body's rotation-since-reset, so the world-frame force is
   pre-encoded each step as quat_apply_inverse(q_now, F_world) (reset quat is
   identity). ~3.3 N pops the can over the 5 mm cradle ridge (2.3 N quasi-static
   requirement). The wrench is then zeroed and EVERYTHING after is hands-off physics:
   the 2-deg ramp rolls the can ~0.24 m to the spout, it flies off the edge, drops
   144 mm and lands inside the docked tray through real contact. Asserted: the can
   exited the chute, is inside the tray, was never floored, and the `caught` latch set.
3. DELIVER (transport teleport): one state write moves tray AND can TOGETHER by the
   same rigid delta (relative pose preserved — carrying a loaded tray), placing the
   tray bottom 2 mm above the pad top; both velocities zeroed. It settles onto the pad
   through real contact and success() is asserted on the settled state.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

The single-Franka-arm strategy for the same plan (wall-top pinch carries of the tray;
closed-jaw fingertip poke through the rear port) lives in TASK.md as the embodiment
argument.

Run (forge): python -u -m simgen_tasks.living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray_i166.solve --headless [--seed N]
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

import os
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.sauce_chute")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        cp, tp, dp, pp = rel(scene.can), rel(scene.tray), rel(scene.decoy), rel(scene.pad)
        print(f"[solve] {tag:10s} | can=({float(cp[0]):+.3f},{float(cp[1]):+.3f},"
              f"{float(cp[2]):.3f}) tray=({float(tp[0]):+.3f},{float(tp[1]):+.3f},"
              f"{float(tp[2]):.3f}) pad=({float(pp[0]):+.3f},{float(pp[1]):+.3f}) "
              f"decoy=({float(dp[0]):+.3f},{float(dp[1]):+.3f}) "
              f"|vc|={float(scene.can.data.root_lin_vel_w[0].norm()):.4f} "
              f"|vt|={float(scene.tray.data.root_lin_vel_w[0].norm()):.4f} "
              f"docked={bool(scene._docked[0])} min_y={float(scene._min_y[0]):.3f} "
              f"exited={bool(scene._exited[0])} caught={bool(scene._caught[0])} "
              f"delivered={bool(scene._delivered[0])} floored={bool(scene._floored[0])} "
              f"in_tray={bool(scene.in_tray(scene.can)[0])} "
              f"on_pad={bool(scene.tray_on_pad()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    cp0, tp0, dp0, pp0 = rel(scene.can), rel(scene.tray), rel(scene.decoy), rel(scene.pad)
    print(f"[solve] layout readback (seed {args.seed}): can=({float(cp0[0]):+.3f},"
          f"{float(cp0[1]):+.3f},{float(cp0[2]):.3f}) tray=({float(tp0[0]):+.3f},"
          f"{float(tp0[1]):+.3f}) pad=({float(pp0[0]):+.3f},{float(pp0[1]):+.3f}) "
          f"decoy=({float(dp0[0]):+.3f},{float(dp0[1]):+.3f})", flush=True)
    report("reset")
    assert abs(float(cp0[1]) - c.cradle_y) < 0.010, "can not seated in the cradle (y)"
    assert abs(float(cp0[2]) - c.can_rest_z) < 0.008, "can not resting on the channel floor"
    assert not bool(scene.in_tray(scene.can)[0]), "can spawned inside the tray"
    assert not bool(scene._floored[0]), "floored latch set at reset"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"baseline score not ~0 ({s0:.3f})"

    # ---------------- phase 1: DOCK — transport the empty tray under the spout --------------
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1], st[:, 2] = c.dock_target[0], c.dock_target[1], 0.002
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins
    scene.tray.write_root_state_to_sim(st, all_ids)
    step(60)
    report("docked")
    assert bool(scene.tray_docked()[0]), "tray did not settle docked under the spout"
    assert bool(scene._docked[0]), "docked latch not set"
    s1 = print_score("P1 empty tray docked under the spout")
    assert s1 >= s0 - 1e-6, "score decreased across docking"
    assert s1 >= c.w_dock - 0.001, f"dock credit missing ({s1:.3f})"

    # ---------------- phase 2: PUSH — fingertip-force nudge, then hands-off physics ---------
    # Servo force only within the fingertip stroke; frame-drag pre-encoding per step.
    K, CAP, V_DES = 15.0, 6.0, -0.22
    stroke_end = c.cradle_y - 0.030
    zeros3 = torch.zeros(n, 1, 3, device=device)
    max_push = 0.0
    push_steps = 0
    for _ in range(480):  # <= 4 s of pushing before we call it a stall
        cy = float(rel(scene.can)[1])
        if cy <= stroke_end:
            break
        vy = float(scene.can.data.root_lin_vel_w[0, 1])
        fy = max(-CAP, min(0.0, K * (V_DES - vy)))
        f_w = torch.zeros(n, 3, device=device)
        f_w[:, 1] = fy
        f_in = quat_apply_inverse(scene.can.data.root_quat_w, f_w).unsqueeze(1)
        scene.can.set_external_force_and_torque(f_in, zeros3, env_ids=all_ids)
        env.step(no_action)
        max_push = max(max_push, abs(fy))
        push_steps += 1
    scene.can.set_external_force_and_torque(zeros3, zeros3, env_ids=all_ids)
    print(f"[solve] push done: {push_steps} steps, peak |F|={max_push:.2f} N "
          f"(cap {CAP:.0f}), can y={float(rel(scene.can)[1]):.3f}", flush=True)
    assert float(rel(scene.can)[1]) <= stroke_end + 1e-3, \
        "push stalled inside the fingertip stroke"
    assert max_push <= CAP + 1e-6, "push force exceeded the fingertip cap"
    # Hands off: ramp roll, spout drop, tray catch — pure gravity + contact.
    for _ in range(720):  # <= 6 s
        env.step(no_action)
        if bool(scene._caught[0]) and bool(scene.settled()[0]):
            break
    step(30)
    report("caught")
    assert bool(scene._exited[0]), "can never left the chute (no exit latch)"
    assert not bool(scene._floored[0]), "can landed OUTSIDE the tray (floored latch)"
    assert bool(scene.in_tray(scene.can)[0]), "can not inside the tray after the drop"
    assert bool(scene._caught[0]), "caught latch not set"
    s2 = print_score("P2 can pushed, rolled, dropped and CAUGHT in the tray")
    assert s2 >= s1 - 1e-6, "score decreased across the catch"
    assert s2 >= c.w_dock + c.w_roll + c.w_exit + c.w_catch - 0.005, \
        f"catch credit missing ({s2:.3f})"

    # ---------------- phase 3: DELIVER — carry the loaded tray onto the pad -----------------
    # One rigid-delta state write for tray AND can together (relative pose preserved),
    # velocities zeroed; the 2 mm drop onto the pad is real contact.
    tray_st = scene.tray.data.root_state_w[all_ids].clone()
    can_st = scene.can.data.root_state_w[all_ids].clone()
    target = scene.pad.data.root_pos_w[all_ids].clone()
    # gentle set-down: 0.5 mm above the pad top — a big drop jolts the capsule into a
    # slow wall-to-wall rolling oscillation that takes many seconds to damp out
    target[:, 2] = scene.env_origins[:, 2] + c.pad_size[2] + 0.0005
    delta = target - tray_st[:, 0:3]
    tray_st[:, 0:3] += delta
    can_st[:, 0:3] += delta
    tray_st[:, 7:13] = 0.0
    can_st[:, 7:13] = 0.0
    scene.tray.write_root_state_to_sim(tray_st, all_ids)
    scene.can.write_root_state_to_sim(can_st, all_ids)
    streak = 0
    for _ in range(1200):  # wait (<= 10 s) for the set-down transient to ring down
        env.step(no_action)
        # require a sustained-success streak: a velocity-threshold "still" can fire
        # spuriously at the turning point of a rolling oscillation
        streak = streak + 1 if bool(scene.success()[0]) else 0
        if streak >= 60:
            break
    report("delivered")
    assert bool(scene.tray_on_pad()[0]), "tray not resting on the pad"
    assert bool(scene.in_tray(scene.can)[0]), "can spilled during the delivery settle"
    assert bool(scene._delivered[0]), "delivered latch not set"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after delivery)", flush=True)
        os._exit(1)
    s3 = print_score("P3 loaded tray delivered onto the pad")
    assert s3 >= 0.99, f"success did not map to score 1.0 ({s3:.3f})"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) -------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s3 - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001 — die loudly, never hang until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(exc).__name__}: {exc})", flush=True)
        os._exit(1)
