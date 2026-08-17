"""Teleport solution for TiltDispenserScene (sim_gen task `basketball_in_hoop_i225`) —
the task's legitimacy certificate.

This solve uses NO teleports at all: the ball spawns sealed inside the cage and the
robot-side interaction is ONE sustained mechanism press, applied as honest dynamics:

  1. READ (perception stand-in): the open side is read from the scene state (the
     shutter covers the other port) — the binary decision the seed task never makes.
  2. PRESS (dynamics): a constant hinge torque on the CAGE body — the exact
     equivalent of ~11 N pressed down on the yellow paddle at its 0.195 m arm
     (`press_torque / paddle_cx`, asserted Franka-scale in the scene cfg) — swings
     the cage toward the open side until the joint's 15-degree tilt stop.
  3. HOLD (dynamics): the torque is HELD at the stop while gravity rolls the ball
     down the now-downhill V-floor, out through the open port, and into the catch
     bin below. Nothing ever touches the ball but the cage, gravity and the bin.
  4. RELEASE (dynamics): the wrench is cleared BEFORE success is ever True; the
     keel swings the cage back level and the ball settles in the bin hands-off.
     The final resting state is never spawned.

Wrench notes: the torque is applied in the cage BODY frame along local +y — the
hinge axis is body-fixed, so a constant body-frame vector is the physically-correct
"hand holds the paddle down" wrench; `enable_external_forces_every_iteration` is set
in the scene cfg and the wrench is re-issued every step.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.5 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still
holds.

Run (forge): python -u -m simgen_tasks.basketball_in_hoop_i225.solve --headless [--seed N]
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tilt_dispenser")().build(num_envs=args.num_envs,
                                                    device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_rows = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def press(tau: float) -> None:
        """Hold the paddle down: torque about the cage's BODY-frame y (the hinge
        axis is body-fixed, so this is the constant press wrench)."""
        t_rows = zero_rows.clone()
        t_rows[:, 0, 1] = tau
        scene.cage.set_external_force_and_torque(zero_rows, t_rows, env_ids=all_ids)

    def clear_press() -> None:
        scene.cage.set_external_force_and_torque(zero_rows, zero_rows, env_ids=all_ids)

    def report(tag: str) -> None:
        p = scene.ball_canon()[0]
        print(f"[solve] {tag:12s} | tilt_c={float(scene.tilt_canon_deg()[0]):+.2f}deg"
              f" ball_canon=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f})"
              f" latches=({float(scene.tilt_latch[0]):.0f},"
              f"{float(scene.exit_latch[0]):.0f},{float(scene.bin_latch[0]):.0f})"
              f" in_bin={bool(scene.in_open_bin()[0])}"
              f" settled={bool(scene.settled()[0])}"
              f" blin={float(scene.ball.data.root_lin_vel_w[0].norm()):.4f}"
              f" bang={float(scene.ball.data.root_ang_vel_w[0].norm()):.4f}"
              f" success={bool(scene.success()[0])}"
              f" score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, read the scene --------------------------------
    step(90)  # ball rolls to the dish valley, transients die
    side = float(scene.side[0])
    p0 = scene.ball_canon()[0]
    print(f"[solve] layout readback (seed {args.seed}): open side="
          f"{side:+.0f} ball_canon=({float(p0[0]):+.3f},{float(p0[1]):+.3f},"
          f"{float(p0[2]):.3f}) tilt={float(scene.tilt_deg()[0]):+.2f}deg", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle (null baseline)")
    assert s0 <= 0.02, "null credit at reset — rubric leak"

    # the press torque, signed toward the OPEN side (local +y tilt = local +x down;
    # the open port is at local x = side)
    tau = side * c.press_torque

    # ---------------- phase 1: press the open-side paddle to the tilt stop ------------------
    reached = False
    for i in range(600):  # up to 5 s
        press(tau)
        env.step(no_action)
        if float(scene.tilt_canon_deg()[0]) >= c.tilt_stop_deg - 1.0:
            reached = True
            print(f"[solve] press: tilt stop reached @step {i}", flush=True)
            break
    report("press")
    assert reached, "cage never reached the tilt stop under the press torque"
    # keep pressing a beat so the tilt latch (which needs sustained >= latch angle)
    # is definitely sampled past the stop
    for _ in range(30):
        press(tau)
        env.step(no_action)
    s1 = print_score("P1 paddle pressed, cage held at the open-side tilt stop")
    assert s1 >= s0 - 1e-6 and s1 >= 0.20 - 1e-6, "tilt stage credit missing"

    # ---------------- phase 2: hold at the stop while gravity dispenses the ball ------------
    dispensed = False
    for i in range(720):  # up to 6 s held
        press(tau)
        env.step(no_action)
        if float(scene.bin_latch[0]) > 0.5:
            dispensed = True
            print(f"[solve] hold: ball entered the bin window @step {i}", flush=True)
            break
    report("hold")
    assert dispensed, "ball never rolled out into the bin window while held"
    s2 = print_score("P2 held at the stop — ball out the open port, into the bin")
    assert s2 >= s1 - 1e-6 and s2 >= 0.60 - 1e-6, "exit/bin stage credit missing"

    # ---------------- phase 3: release, hands-off settle ------------------------------------
    clear_press()
    for i in range(40):  # up to 10 s hands-off (cage swings level, ball stops rolling)
        if bool(scene.success()[0]):
            break
        step(30)
        if i % 4 == 3:
            p = scene.ball_canon()[0]
            print(f"[solve] settle wait {i}: canon=({float(p[0]):+.3f},"
                  f"{float(p[1]):+.3f},{float(p[2]):+.3f}) "
                  f"blin={float(scene.ball.data.root_lin_vel_w[0].norm()):.4f} "
                  f"bang={float(scene.ball.data.root_ang_vel_w[0].norm()):.4f}",
                  flush=True)
    report("released")
    s3 = print_score("P3 released — cage back level, ball at rest in the bin")
    assert s3 >= s2 - 1e-6, "score decreased across the release"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after press-hold-release)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.5 simulated seconds, hands-off) ------------
    hold, flickers = True, 0
    for i in range(420):  # 420 steps = 3.5 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                p = scene.ball_canon()[0]
                print(f"[solve] persist flicker @step {i}: "
                      f"in_bin={bool(scene.in_open_bin()[0])} "
                      f"settled={bool(scene.settled()[0])} "
                      f"tilt_c={float(scene.tilt_canon_deg()[0]):+.2f} "
                      f"canon=({float(p[0]):+.3f},{float(p[1]):+.3f},"
                      f"{float(p[2]):+.3f}) "
                      f"blin={float(scene.ball.data.root_lin_vel_w[0].norm()):.4f} "
                      f"bang={float(scene.ball.data.root_ang_vel_w[0].norm()):.4f}",
                      flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/420 steps", flush=True)
    report("persist")
    s4 = print_score("P4 persistence 3.5 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s3 - 1e-6
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
