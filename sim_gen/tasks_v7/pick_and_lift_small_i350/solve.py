"""Solution for PinLatchDumbwaiterScene (sim_gen task `pick_and_lift_small_i350`) —
the task's legitimacy certificate.

Teleports are TRANSPORT ONLY (one hover placement of the blue cube above the car
mouth, zero velocity); every load-bearing interaction is contact dynamics:

  1. LOAD (gravity + contact): the cube is released above the car's open top and
     FALLS through the drop corridor (past the latch pin) into the car; seating,
     bouncing and coming to rest are pure contact. Retried from the hover if a
     bounce leaves it off-target.
  2. PIN PULL (contact friction): a bias-ramp + velocity-damped axial force on the
     pin (what a parallel-jaw pinch-and-pull on the knob applies) slides the rod
     out of both slots against the spring-preloaded friction. The force cap starts
     finger-scale and escalates only on a detected stall.
  3. RIDE (pure physics, hands off): the scene's lift spring hoists car + cube the
     full stroke into the sealed penthouse and holds them pressed on the top stop.
  4. SETTLE + persistence: hands-off until success(), 2 s extra margin, then a
     >= 3.3 simulated-second window with success() checked every substep. Nothing
     is welded, pinned or held at the end — the spring press is the scene's own
     standing mechanism.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), and `SIM_GEN_SOLVE: SUCCESS` only if success persists.

Run (forge): python -u -m simgen_tasks.pick_and_lift_small_i350.solve --headless [--seed N]
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

_qapply = scene_mod._qapply
_qinv = scene_mod._qinv

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pin_latch_dumbwaiter")().build(num_envs=args.num_envs,
                                                          device=device)
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

    def wrench(body, f_w: torch.Tensor) -> None:
        """Apply a WORLD force (n,3) to `body`, expressed in its CURRENT link frame
        (`is_global=True` silently drops torques on this stack — transform manually,
        the house convention). Re-set every substep while pushing."""
        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f_w).unsqueeze(1),
            torch.zeros(n, 1, 3, device=device), env_ids=all_ids)

    zero3 = torch.zeros(n, 3, device=device)

    def cut(body) -> None:
        wrench(body, zero3)

    def tower_local(pos_w: torch.Tensor) -> torch.Tensor:
        return _qapply(_qinv(scene.tower.data.root_quat_w),
                       pos_w - scene.tower.data.root_pos_w)

    def pin_ends_y() -> tuple[float, float]:
        """Tower-frame y of the rod tail and head (env 0)."""
        q = scene.pin.data.root_quat_w
        p = scene.pin.data.root_pos_w
        tail = torch.tensor([0.0, -c.pin_tail, 0.0], device=device).expand(n, 3)
        head = torch.tensor([0.0, c.pin_head, 0.0], device=device).expand(n, 3)
        y1 = tower_local(p + _qapply(q, tail))[0, 1]
        y2 = tower_local(p + _qapply(q, head))[0, 1]
        return float(y1), float(y2)

    def report(tag: str) -> None:
        q = float(scene.q_car()[0])
        cl = scene._car_local(scene.cube.data.root_pos_w)[0]
        y1, y2 = pin_ends_y()
        print(f"[solve] {tag:12s} | q={q:.4f}/{c.stroke:.3f} "
              f"cube_car=({float(cl[0]):+.3f},{float(cl[1]):+.3f},{float(cl[2]):+.3f}) "
              f"in_car={bool(scene.cube_in_car()[0])} pin_y=({y1:+.3f},{y2:+.3f}) "
              f"clear={bool(scene.pin_clear()[0])} "
              f"L={int(scene._loaded[0])}P={int(scene._pin_out[0])}R={int(scene._risen[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f} "
              f"| v_car={float(scene.car.data.root_lin_vel_w[0].norm()):.4f} "
              f"v_cube={float(scene.cube.data.root_lin_vel_w[0].norm()):.4f} "
              f"v_pin={float(scene.pin.data.root_lin_vel_w[0].norm()):.4f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: settle at spawn ---------------------------------------------
    step(120)
    report("spawn")
    prev = print_score("P0 spawn settled (car pinned, cube on the floor)")
    assert prev < 0.05, "null-state score should be ~0"
    q0 = float(scene.q_car()[0])
    assert abs(q0 - c.q_pin) < 0.010, f"car not pinned at spawn (q={q0:.4f})"

    # ---------------- phase 1: LOAD — hover release, gravity drop into the car -------------
    # Teleport = transport only: place the cube at zero velocity in free air above
    # the car's open mouth (inside the drop corridor, clear of the pin), release.
    tq = scene.tower.data.root_quat_w
    tp = scene.tower.data.root_pos_w
    hover_loc = torch.tensor([0.009, 0.0, 0.150], device=device).expand(n, 3)
    loaded = False
    for attempt in range(4):
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = tp + _qapply(tq, hover_loc)
        st[:, 3:7] = tq  # faces parallel to the car walls
        scene.cube.write_root_state_to_sim(st, all_ids)
        for _ in range(240):  # 2 s: fall + seat + settle
            env.step(no_action)
            if bool(scene.cube_in_car()[0]) \
                    and float(scene.cube.data.root_lin_vel_w[0].norm()) < 0.03:
                break
        if bool(scene.cube_in_car()[0]):
            loaded = True
            break
        print(f"[solve] drop attempt {attempt} missed — retrying", flush=True)
    assert loaded, "cube never seated in the car"
    step(60)
    report("loaded")
    s1 = print_score("P1 cube dropped through the corridor into the car (gravity)")
    assert bool(scene._loaded[0]), "loaded latch did not set"
    assert s1 >= prev - 1e-6, "score decreased across load"
    prev = s1

    # ---------------- phase 2: PIN PULL — axial friction slide out of the slots ------------
    # Handle side from the knob's tower-frame y sign; pull along the tower's y axis.
    y1, y2 = pin_ends_y()
    sign = 1.0 if y2 > y1 else -1.0
    print(f"[solve] pin handle on {'+y' if sign > 0 else '-y'} side", flush=True)
    v_des = 0.10
    f_cap = 4.0
    bias = 0.0
    k_damp = 1.5
    clear_goal = c.wall_out + 0.020
    pulled = False
    stall_ref, stall_ctr = min(abs(y1), abs(y2)), 0
    for i in range(2400):  # up to 20 s
        ey = torch.zeros(n, 3, device=device)
        ey[:, 1] = sign
        d_w = quat_apply(scene.tower.data.root_quat_w, ey)
        v_along = float((scene.pin.data.root_lin_vel_w * d_w).sum(dim=-1)[0])
        if v_along < 0.5 * v_des:
            bias = min(bias + 0.02, f_cap)
        elif v_along > 1.5 * v_des:
            bias = max(bias * 0.90, 0.2)
        f_mag = max(0.0, min(bias + k_damp * (v_des - v_along), f_cap))
        wrench(scene.pin, d_w * f_mag)
        env.step(no_action)
        y1, y2 = pin_ends_y()
        prog = min(abs(y1), abs(y2))  # inner-most rod end distance from centre
        if min(y1, y2) > clear_goal or max(y1, y2) < -clear_goal:
            pulled = True
            break
        if prog > stall_ref + 0.004:
            stall_ref, stall_ctr = prog, 0
        else:
            stall_ctr += 1
            if stall_ctr >= 360:  # 3 s without 4 mm of progress
                f_cap = min(f_cap * 1.5, 12.0)
                bias, stall_ctr = 0.0, 0
                print(f"[solve] pin stall @i={i} prog={prog:.3f} — "
                      f"escalate cap to {f_cap:.1f}N", flush=True)
        if i % 240 == 239:
            print(f"[solve] pull i={i}: ends=({y1:+.3f},{y2:+.3f}) v={v_along:.3f} "
                  f"bias={bias:.2f} q={float(scene.q_car()[0]):.3f}", flush=True)
    cut(scene.pin)
    assert pulled, "pin never cleared the slots"
    report("pin_out")
    s2 = print_score("P2 pin pulled out of both slots (friction under spring preload)")
    assert bool(scene._pin_out[0]), "pin_out latch did not set"
    assert s2 >= prev - 1e-6, "score decreased across the pull"
    prev = s2

    # ---------------- phase 3: RIDE — hands off, the spring hoists car + cube --------------
    risen = False
    for i in range(720):  # up to 6 s
        env.step(no_action)
        if float(scene.q_car()[0]) >= c.stroke - c.top_tol \
                and float(scene.car.data.root_lin_vel_w[0].norm()) < 0.05:
            risen = True
            break
        if i % 120 == 119:
            print(f"[solve] ride i={i}: q={float(scene.q_car()[0]):.3f} "
                  f"v_car={float(scene.car.data.root_lin_vel_w[0].norm()):.3f} "
                  f"in_car={bool(scene.cube_in_car()[0])}", flush=True)
    assert risen, "car never reached the top stop"
    report("risen")
    s3 = print_score("P3 spring ride to the top stop (pure physics)")
    assert bool(scene._risen[0]), "risen latch did not set"
    assert s3 >= prev - 1e-6, "score decreased across the ride"
    prev = s3

    # ---------------- phase 4: settle to success + margin ----------------------------------
    for j in range(32):  # up to 8 s of hands-off settling
        if bool(scene.success()[0]):
            break
        step(30)
        if j % 4 == 3:
            report(f"settle{j}")
    if bool(scene.success()[0]):
        step(240)  # 2 s extra hands-off margin before the strict persistence window
    report("delivered")
    s4 = print_score("P4 success settled (car held on the top stop by the spring)")
    assert s4 >= prev - 1e-6, "score decreased across settle"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after ride+settle)", flush=True)
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3.3 simulated seconds, hands off) -----------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                print(f"[solve] persist flicker @step {i}: "
                      f"q={float(scene.q_car()[0]):.4f} "
                      f"in={bool(scene.cube_in_car()[0])} "
                      f"set={bool(scene.settled()[0])} "
                      f"v_car={float(scene.car.data.root_lin_vel_w[0].norm()):.4f} "
                      f"v_cube={float(scene.cube.data.root_lin_vel_w[0].norm()):.4f} "
                      f"v_pin={float(scene.pin.data.root_lin_vel_w[0].norm()):.4f}",
                      flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
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
    try:
        main()
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
