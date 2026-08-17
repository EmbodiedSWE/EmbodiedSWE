"""Teleport solution for PinLockDrawerScene (sim_gen task `pull_cube_i245`) — the
task's legitimacy certificate.

Every load-bearing interaction happens through CONTACT DYNAMICS via the scene's force
plant (`scene.drive`: bounded world-frame forces at a body's CoM — the push/pull a
fingertip or pinch grasp exerts). The demonstrated policy is the forced order from
describe():

  1. UN-LATCH + EXTRACT the yellow GUARD pin: press it up to its hole ceiling with a
     constant bounded force (hard-stop press — the lift that clears its arm over the
     housing's catch posts), slide it ~4 cm along the housing's +y axis with a
     closed-loop velocity servo (F = K*(v_des - v), 0 <= F <= Fmax) while holding the
     press, drop the lift once the arm has passed the far catch face, then servo the
     rest of the way out of both rail holes and the drawer notches onto the support
     ledge. The pin slides through its holes under contact the whole way.
  2. Teleport the freed, resting pin to a floor depot — TRANSPORT ONLY (the in-reach
     pick-and-carry a Franka does with a pinch grasp; the pin is already fully outside
     the mechanism when it is picked).
  3. Same for the green LOCK pin (its path is only free because the guard's arm is
     gone — smoke proves it jams otherwise).
  4. PULL the drawer by a servo force along the housing's -x axis until its lug hits
     the stop lintel; release; the drawer and riding cube settle by contact.

Prints `SIM_GEN_SCORE <score>` at each settled phase boundary (non-decreasing:
0 -> 0.15 -> 0.30 -> 1.0), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
held at every poll.

Run (forge): python -u -m simgen_tasks.pull_cube_i245.solve --headless [--seed N]
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

_DEPOT = {"pin1": (0.60, 0.45), "pin2": (0.60, 0.60)}  # floor depots, far off the plinth


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pin_lock_drawer")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def wait_settled(max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            ok = True
            for b in (scene.drawer, scene.pin1, scene.pin2, scene.cube):
                ok = ok and bool(scene.settled(b)[0])
            if ok:
                break

    def report(tag: str) -> None:
        d = float(scene.drawer_d()[0])
        t1 = float(scene.pin_travel(scene.pin1)[0])
        t2 = float(scene.pin_travel(scene.pin2)[0])
        print(f"[solve] {tag:14s} | d={d:+.4f} (need >= {c.d_thresh:.3f}) "
              f"pin1_travel={t1:+.4f} pin2_travel={t2:+.4f} "
              f"clear=({bool(scene.pin_clear(scene.pin1)[0])},"
              f"{bool(scene.pin_clear(scene.pin2)[0])}) "
              f"cube_in_bay={bool(scene.cube_in_bay()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def servo_pull(row: int, body, dir_w: torch.Tensor, goal_fn, goal: float,
                   v_des: float, k0: float, fmax: float, budget: int, tag: str,
                   bias: torch.Tensor | None = None) -> None:
        """Closed-loop velocity servo: world force along dir_w at the body CoM, gain-
        escalating on stall (velocity-servo gain-stall recipe), with a frame-encoding
        toggle as the last-resort fallback. Breaks on POSITION readback (goal_fn).
        `bias` (n,3) is a constant world force added every step (the ceiling press
        held through the guard pin's lifted slide); it is kept during relax windows."""
        rest = bias if bias is not None else 0.0
        gain, fruitless, last, win = k0, 0, float(goal_fn()[0]), 0
        for i in range(budget):
            v = float((body.data.root_lin_vel_w[0] * dir_w[0]).sum())
            f = max(0.0, min(gain * (v_des - v), fmax))
            scene.drive[:, row, :] = dir_w * f
            if bias is not None:
                scene.drive[:, row, :] += bias
            env.step(no_action)
            prog = float(goal_fn()[0])
            if prog >= goal:
                print(f"[solve] {tag}: reached {prog:+.4f} in {i + 1} steps "
                      f"(gain {gain:.1f})", flush=True)
                break
            win += 1
            if win >= 60:
                if prog - last < 0.002:
                    fruitless += 1
                    if fruitless >= 3:
                        scene.drive[:, row, :] = rest
                        step(30)
                        scene.drive_encode = not scene.drive_encode
                        gain, fruitless = k0, 0
                        print(f"[solve] {tag}: encoding toggled "
                              f"(drive_encode={scene.drive_encode})", flush=True)
                    else:
                        gain *= 1.6
                        print(f"[solve] {tag}: stall at {prog:+.4f} — gain -> "
                              f"{gain:.1f}", flush=True)
                else:
                    fruitless = 0
                last, win = prog, 0
        scene.drive[:, row, :] = rest
        step(1)
        assert float(goal_fn()[0]) >= goal - 0.004, \
            f"{tag}: servo did not reach its goal ({float(goal_fn()[0]):+.4f} < {goal})"

    def teleport_pin(pin, name: str) -> None:
        """TRANSPORT ONLY: the pin is already fully extracted and resting on the
        support ledge (outside the mechanism, in easy reach); carry it to the floor
        depot and set it down flat with zero velocity."""
        dx, dy = _DEPOT[name]
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = dx
        st[:, 1] = dy
        st[:, 2] = c.pin_w / 2 + 0.001
        st[:, 0:3] += scene.env_origins
        st[:, 3] = 1.0
        pin.write_root_state_to_sim(st, all_ids)

    # ---------------- phase 0: reset, settle, baseline --------------------------------------
    wait_settled(240)
    report("reset")
    d0 = float(scene.drawer_d()[0])
    assert d0 < 0.012, "drawer must start (near) closed"
    assert not bool(scene.pin_clear(scene.pin1)[0]), "guard pin must start seated"
    assert not bool(scene.pin_clear(scene.pin2)[0]), "lock pin must start seated"
    assert bool(scene.cube_in_bay()[0]), "cube must start riding in the bay"
    s_prev = print_score("P0 reset+settle (drawer pinned shut, cube aboard)")
    assert s_prev <= 0.02, "null-state score must be ~0"

    ax, ay = scene.housing_axes()
    pull_y = ay.clone()  # pin extraction direction (housing +y in world)
    pull_x = -ax.clone()  # drawer opening direction (housing -x in world)

    # ---------------- phase 1: un-latch (lift) + extract the GUARD pin, park it -------------
    # P1a: press the pin up to its hole ceiling (hard-stop press, ~2x pin weight) —
    # the lift that raises its arm clear of the catch posts. No servo tuning needed.
    lift_f = torch.zeros(n, 3, device=device)
    lift_f[:, 2] = 2.5
    scene.drive[:, 0, :] = lift_f
    step(90)
    lift = float(scene.pin1.data.root_pos_w[0, 2] - scene.env_origins[0, 2]) - c.pin_z
    print(f"[solve] P1a ceiling press: lift={lift * 1000:.1f} mm "
          f"(need >= {0.7 * c.guard_lift * 1000:.1f})", flush=True)
    assert lift >= 0.7 * c.guard_lift, "guard pin failed to rise to its hole ceiling"
    # P1b: slide +y while pressed up until the arm has passed the far catch face.
    servo_pull(0, scene.pin1, pull_y, lambda: scene.pin_travel(scene.pin1),
               c.guard_drop_travel, 0.05, 5.0, 5.0, 900, "P1b lifted slide", bias=lift_f)
    # P1c: drop the lift (shaft lands back on the flush hole-bottom/ledge plane),
    # then servo flat the rest of the way out.
    scene.drive[:, 0, :] = 0.0
    step(45)
    servo_pull(0, scene.pin1, pull_y, lambda: scene.pin_travel(scene.pin1),
               c.pin_travel_out, 0.06, 5.0, 5.0, 1500, "P1c guard-pin extract")
    wait_settled(300)
    report("P1-extracted")
    assert bool(scene.pin_clear(scene.pin1)[0]), "guard pin must be clear after extraction"
    teleport_pin(scene.pin1, "pin1")
    wait_settled(240)
    s_now = print_score("P1 guard pin extracted through its holes and parked")
    assert s_now >= s_prev - 1e-6 and s_now >= 0.14, "P1 score regression"
    s_prev = s_now

    # ---------------- phase 2: extract the LOCK pin, park it --------------------------------
    servo_pull(1, scene.pin2, pull_y, lambda: scene.pin_travel(scene.pin2),
               c.pin_travel_out, 0.06, 5.0, 5.0, 1500, "P2 lock-pin extract")
    wait_settled(300)
    report("P2-extracted")
    assert bool(scene.pin_clear(scene.pin2)[0]), "lock pin must be clear after extraction"
    teleport_pin(scene.pin2, "pin2")
    wait_settled(240)
    s_now = print_score("P2 lock pin extracted (path free only because the guard is out)")
    assert s_now >= s_prev - 1e-6 and s_now >= 0.29, "P2 score regression"
    s_prev = s_now

    # ---------------- phase 3: pull the drawer to its hard stop -----------------------------
    servo_pull(2, scene.drawer, pull_x, lambda: scene.drawer_d(),
               0.150, 0.07, 12.0, 8.0, 1500, "P3 drawer pull")
    wait_settled(480)
    report("P3-open")
    s_now = print_score("P3 drawer at the lintel stop, cube riding in the bay")
    assert s_now >= s_prev - 1e-6, "P3 score regression"
    s_prev = s_now
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after open+settle)", flush=True)
        os._exit(1)
    assert abs(s_prev - 1.0) < 1e-6, "success must score exactly 1.0"

    # ---------------- persistence (>= 3.3 simulated seconds, hands off) --------------------
    hold, flickers = True, 0
    for i in range(400):
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:
                loc = scene.drawer_local()[0]
                lv = float(scene.drawer.data.root_lin_vel_w[0].norm())
                cv = float(scene.cube.data.root_lin_vel_w[0].norm())
                print(f"[solve] persist flicker @step {i}: d={float(scene.drawer_d()[0]):+.4f} "
                      f"y={float(loc[1]):+.4f} z={float(loc[2]):+.4f} "
                      f"drawer_lin={lv:.4f} cube_lin={cv:.4f} "
                      f"in_bay={bool(scene.cube_in_bay()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s4 = print_score("P-persist persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s_prev - 1e-6
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
