"""Teleport solution for LatchCanisterScene (sim_gen task `close_jar_i54`) — the
task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. BALL DROP (teleport = transport only; containment by gravity): one root-state
   write carries the RED ball from its ground slot to a free hover ~60 mm above the
   canister mouth, zero velocity. The write satisfies nothing by itself (the hover
   point is ABOVE `cavity_z_hi`); the ball then FALLS through the open mouth and
   comes to rest on the canister floor — `ball_in` is produced by ballistics and
   contact. If a bounce ejects it, it is lifted and re-dropped; entry is always
   gravitational. The BLUE decoy is never touched.
2. LID SEATING (teleport = transport only; seating by gravity): the lid is carried
   to a hover ~18 mm above the seat plane, yaw-aligned with the canister (exactly
   what a Franka wrist does before releasing), zero velocity — the hover height is
   ABOVE `seat_z_hi`, so the write does not satisfy `lid_seated`. Gravity drops it
   the last stretch into the lip-ring recess; the seat plane and lateral capture do
   the aligning. Mis-seats are lifted and re-dropped.
3. TAB LOCKING (applied torque through the joint): each orange turn-tab is swung
   from OPEN to LOCKED by a pure world-z torque on the tab body — the exact wrench
   of a fingertip pushing the bar sideways — velocity-regulated (~0.9 rad/s cruise),
   released at -80 deg where the strongly-damped vertical hinge parks it (~7 deg of
   coast). `tabs_locked` is judged from the resulting BODY pose.
   A pure z torque is invariant under the pod force-frame drag quirk (the drag
   rotates wrenches by the body's z-rotation since reset, which fixes the z axis),
   so no mode probing is needed. Stalls escalate the torque cap, then re-open and
   re-try; if the tab cannot reach the locked window the run FAILS (it does not
   teleport the tab).
4. ORDER (forced by the scene's interlocks): ball -> lid -> tabs. This solve also
   demonstrates the order dependency implicitly: the rubric's order-coupled latches
   only advance in this sequence.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.close_jar_i54.solve --headless [--seed N]
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
    from .scene import _qapply
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import _qapply

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.latch_canister")().build(num_envs=args.num_envs,
                                                   device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def can_local(pos_local) -> torch.Tensor:
        """Canister-frame point -> world (live canister pose)."""
        p = torch.tensor(pos_local, device=device).expand(n, 3)
        return scene.canister.data.root_pos_w + _qapply(
            scene.canister.data.root_quat_w, p)

    def report(tag: str) -> None:
        th = scene.tab_theta()[0]
        lid_loc = scene._local(scene.lid.data.root_pos_w)[0]
        red_loc = scene._local(scene.red.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} | tabs=({float(th[0]):+6.1f},{float(th[1]):+6.1f})deg "
              f"lid_loc=({float(lid_loc[0]):+.3f},{float(lid_loc[1]):+.3f},"
              f"{float(lid_loc[2]):+.3f}) "
              f"red_loc_z={float(red_loc[2]):+.3f} "
              f"ball_in={bool(scene.ball_in()[0])} "
              f"lid_seated={bool(scene.lid_seated()[0])} "
              f"locked={[bool(v) for v in scene.tabs_locked()[0]]} "
              f"decoy_out={bool(scene.decoy_out()[0])} still={bool(scene.still()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(150)
    can_p = (scene.canister.data.root_pos_w - scene.env_origins)[0]
    qc = scene.canister.data.root_quat_w[0]
    yaw = float(torch.rad2deg(2.0 * torch.atan2(qc[3], qc[0])))
    red_xy = (scene.red.data.root_pos_w - scene.env_origins)[0, :2]
    blue_xy = (scene.blue.data.root_pos_w - scene.env_origins)[0, :2]
    lid_xy = (scene.lid.data.root_pos_w - scene.env_origins)[0, :2]
    th0 = scene.tab_theta()[0]
    slot = "slotA" if float(scene.red_slot[0]) > 0 else "slotB"
    print(f"[solve] layout readback (seed {args.seed}): "
          f"canister=({float(can_p[0]):+.3f},{float(can_p[1]):+.3f}) yaw={yaw:+.1f}deg "
          f"tabs=({float(th0[0]):+.1f},{float(th0[1]):+.1f})deg "
          f"red=({float(red_xy[0]):+.3f},{float(red_xy[1]):+.3f}) [{slot}] "
          f"blue=({float(blue_xy[0]):+.3f},{float(blue_xy[1]):+.3f}) "
          f"lid=({float(lid_xy[0]):+.3f},{float(lid_xy[1]):+.3f})", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert float(th0.abs().max()) < 25.0, "tabs must start near OPEN"
    assert not bool(scene.ball_in()[0]), "red ball must start outside"
    assert not bool(scene.lid_seated()[0]), "lid must start off the canister"
    s0 = print_score("P0 reset+settle (canister open and empty)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: drop the RED ball in (containment by gravity) ---------------
    got = False
    for attempt in range(3):
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = can_local((0.0, 0.0, 0.150))  # hover ABOVE cavity_z_hi
        st[:, 3] = 1.0
        scene.red.write_root_state_to_sim(st, all_ids)
        step(180)  # free fall through the mouth + rattle-down + settle, hands-off
        got = bool(scene.ball_in()[0]) and \
            float(scene.red.data.root_lin_vel_w[0].norm()) < 0.05
        if got:
            break
        print(f"[solve] ball drop attempt {attempt} missed "
              f"(ball_in={bool(scene.ball_in()[0])}); re-dropping", flush=True)
    report("ball-drop")
    if not got:
        print("SIM_GEN_SOLVE: FAIL (ball never came to rest in the cavity)", flush=True)
        os._exit(1)
    s1 = print_score("P1 red ball dropped through the open mouth")
    assert s1 >= c.w_ball - 1e-6, f"P1 score {s1} (expect ball latch = {c.w_ball})"
    assert not bool(scene.success()[0]), "cannot be success with the lid off"

    # ---------------- phase 2: lay the lid into the recess (seating by gravity) ------------
    seated = False
    for attempt in range(3):
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = can_local((0.0, 0.0, 0.112))  # 18 mm above the seat, ABOVE seat_z_hi
        st[:, 3:7] = scene.canister.data.root_quat_w  # yaw-aligned release
        scene.lid.write_root_state_to_sim(st, all_ids)
        step(180)  # gravity seats it; lateral capture by the lip ring
        seated = bool(scene.lid_seated()[0]) and \
            float(scene.lid.data.root_lin_vel_w[0].norm()) < 0.05
        if seated:
            break
        print(f"[solve] lid drop attempt {attempt} mis-seated "
              f"(lid_seated={bool(scene.lid_seated()[0])}); re-dropping", flush=True)
    report("lid-drop")
    if not seated:
        print("SIM_GEN_SOLVE: FAIL (lid never seated in the recess)", flush=True)
        os._exit(1)
    s2 = print_score("P2 lid seated in the lip-ring recess")
    assert s2 >= s1 - 1e-6 and s2 >= c.w_ball + c.w_lid - 1e-6, \
        f"P2 score {s2} (expect ball+lid = {c.w_ball + c.w_lid})"
    assert not bool(scene.success()[0]), "cannot be success with the tabs open"

    # ---------------- phases 3+4: swing each tab locked (applied z torque) -----------------
    def lock_tab(body, idx: int, tag: str) -> bool:
        """Velocity-regulated pure-z torque swing OPEN -> LOCKED; released at -80 deg
        where the damped hinge parks it. Judged from the resulting body pose."""
        # Gain sizing: tab inertia about the pivot is ~4.4e-5 kg*m^2, so discrete
        # stability at 120 Hz needs k/I * dt < 2 -> k < ~0.01. k = 0.004 cruises
        # smoothly; the cap escalates only if a real obstruction resists.
        tau_cap = 0.02
        for round_ in range(2):
            win_i, win_th = 0, float(scene.tab_theta()[0, idx])
            for i in range(1200):
                th = float(scene.tab_theta()[0, idx])
                if th <= -80.0:
                    clear_wrench(body)
                    step(90)  # damped-hinge park + settle, hands-off
                    ok = bool(scene.tabs_locked()[0, idx])
                    print(f"[solve] {tag}: released at {th:+.1f} deg -> "
                          f"locked={ok}", flush=True)
                    return ok
                w = float(body.data.root_ang_vel_w[0, 2])
                w_des = -0.9 if th > -60.0 else -0.4
                tau = max(-tau_cap, min(tau_cap, 0.004 * (w_des - w)))
                t_world = torch.zeros(n, 1, 3, device=device)
                t_world[0, 0, 2] = tau
                body.set_external_force_and_torque(zero_wrench, t_world,
                                                   env_ids=all_ids, is_global=True)
                env.step(no_action)
                if i - win_i >= 60:  # stall probe every 0.5 s
                    if th > win_th - 3.0:
                        tau_cap = min(tau_cap + 0.02, 0.08)
                        print(f"[solve] {tag}: stalled at {th:+.1f} deg; "
                              f"torque cap -> {tau_cap:.2f} N*m", flush=True)
                    win_i, win_th = i, th
            clear_wrench(body)
            if round_ == 0:
                # re-open and retry once from the top with the escalated cap
                print(f"[solve] {tag}: swing timed out at "
                      f"{float(scene.tab_theta()[0, idx]):+.1f} deg; re-opening",
                      flush=True)
                for i in range(600):
                    th = float(scene.tab_theta()[0, idx])
                    if th >= -5.0:
                        break
                    w = float(body.data.root_ang_vel_w[0, 2])
                    tau = max(-0.04, min(0.04, 0.004 * (0.9 - w)))
                    t_world = torch.zeros(n, 1, 3, device=device)
                    t_world[0, 0, 2] = tau
                    body.set_external_force_and_torque(zero_wrench, t_world,
                                                       env_ids=all_ids, is_global=True)
                    env.step(no_action)
                clear_wrench(body)
                step(60)
        return False

    ok = lock_tab(scene.tab_e, 0, "tab-east")
    report("tab-east")
    if not ok:
        print("SIM_GEN_SOLVE: FAIL (east tab never locked)", flush=True)
        os._exit(1)
    s3 = print_score("P3 east tab swung across the lid")
    assert s3 >= s2 - 1e-6 and s3 >= 0.60 - 1e-6, f"P3 score {s3} (expect cap 0.60)"

    ok = lock_tab(scene.tab_w, 1, "tab-west")
    step(120)  # full settle, hands-off (still-counter needs 60 consecutive)
    report("tab-west")
    if not ok or not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (west tab / final state not successful)", flush=True)
        os._exit(1)
    s4 = print_score("P4 both tabs locked; canister sealed")
    assert s4 >= s3 - 1e-6, "score decreased across the tab phase"
    assert s4 >= 1.0 - 1e-6, f"success must score 1.0, got {s4}"

    # ---------------- phase 5: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s5 >= s4 - 1e-6
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
    main()
