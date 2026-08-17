"""Teleport solution for StowflatScene (sim_gen task
`libero_kitchen_scene5_close_the_top_drawer_of_the_cabinet_i418`) — the task's
legitimacy certificate.

NO teleports are needed at all: every interaction is a force-limited contact
proxy for a hand, and every load-bearing outcome is delivered by contact
dynamics under gravity.

1. PERCEPTION: the drawer's sampled opening q0 and the cargo poses (carton x,
   side, yaw; companion pose) are read back from the episode state — never
   hard-coded. The tip direction (toward the drawer centreline) comes from the
   MEASURED carton side.
2. TIP (a fingertip push): a small horizontal force (0.45 N — far below the
   ~0.65 N sliding limit, but 1.9x the tipping moment at its application
   height) is applied at a point on the carton's upper third, pushing it
   sideways toward the drawer centreline. The carton pivots on its bottom
   edge; once past the diagonal, the force is REMOVED and gravity lays it
   flat on the drawer floor, inside the drawer. Nothing else is touched.
3. CLOSE (the seed's push, now unblocked): a force-limited PD push
   (<= 1.2 N — gentle enough that the flat cargo rides the floor without
   sliding) drives the drawer to its seated position; the force is released.
4. HANDS-OFF: the drawer holds its seat (horizontal slide, no stored energy),
   the cargo lies stowed inside; success() turns True on the live settled
   state and persists.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
rubric latches), then holds HANDS-OFF >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.
The whole solve is then repeated on a SECOND seed (fresh episode, no score
prints — the rubric restarts at 0 on reset) to certify seed robustness.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene5_close_the_top_drawer_of_the_cabinet_i418.solve
             --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.stowflat_cabinet")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    from isaaclab.utils.math import quat_apply_inverse

    print("[solve] " + "=" * 70, flush=True)
    print(scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        print(f"[solve] {tag:12s} | q={float(scene.drawer_open()[0]):+.4f} "
              f"axz={float(scene.carton_axis_z()[0]):.2f} "
              f"flat={bool(scene.carton_flat()[0])} in={bool(scene.carton_in()[0])} "
              f"comp={bool(scene.comp_in()[0])} "
              f"latch(f/c)=({int(scene._fflat[0])},{float(scene._fclose[0]):.2f}) "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    zero3 = torch.zeros(n, 1, 3, device=device)

    def carton_wrench_off() -> None:
        scene.carton.set_external_force_and_torque(zero3, zero3)

    def drawer_force_off() -> None:
        scene.drawer.set_external_force_and_torque(zero3, zero3)

    def episode(seed: int, announce: bool) -> bool:
        """One full episode on `seed`. SIM_GEN_SCORE is printed only when
        `announce` (the rubric restarts at 0 on reset, and the score stream
        must be non-decreasing)."""
        env.reset(seed=seed)
        s_prev = 0.0

        def print_score(tag: str) -> float:
            nonlocal s_prev
            s = float(scene.score()[0])
            print(f"[solve] phase boundary: {tag}", flush=True)
            if announce:
                print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
            assert s >= s_prev - 1e-6, f"score decreased: {s_prev:.4f} -> {s:.4f}"
            s_prev = s
            return s

        # ---------------- phase 0: settle, layout readback, baseline --------------------
        step(120)
        q0 = float(scene.q0[0])
        cl = scene._drawer_frame(scene.carton)[0]
        bl = scene._drawer_frame(scene.comp)[0]
        side = 1.0 if float(cl[1]) > 0 else -1.0
        print(f"[solve] layout readback (seed {seed}): drawer opening q0={q0:+.4f} "
              f"carton=({float(cl[0]):+.3f},{float(cl[1]):+.3f},{float(cl[2]):+.3f}) "
              f"side={side:+.0f} "
              f"comp=({float(bl[0]):+.3f},{float(bl[1]):+.3f},{float(bl[2]):+.3f}) "
              f"axz={float(scene.carton_axis_z()[0]):.3f}", flush=True)
        report("reset")
        assert bool(scene._finite()[0]), "NaN/inf after settle"
        assert abs(float(scene.drawer_open()[0]) - q0) < 0.006, \
            "drawer must rest at its sampled opening"
        assert float(scene.carton_axis_z()[0]) > 0.95, "carton must start upright"
        assert bool(scene.carton_in()[0]) and bool(scene.comp_in()[0]), \
            "cargo must start inside the drawer"
        assert not bool(scene.success()[0]), "fresh reset must not be success"
        s = print_score("P0 reset+settle (drawer out, carton upright in the front band)")
        assert s <= 0.05, f"baseline score should be ~0, got {s}"

        # ---------------- phase 1: TIP — fingertip push lays the carton flat ------------
        # 0.45 N horizontal at a body point 45 mm above the CoM (upper third):
        # 1.9x the tipping moment, 0.7x the sliding limit. Direction: toward the
        # drawer centreline, from the MEASURED side. Released past the diagonal;
        # gravity finishes the lay-down.
        f_w = torch.zeros(n, 3, device=device)
        f_w[:, 1] = -side * 0.45
        r_b = torch.zeros(n, 3, device=device)
        r_b[:, 2] = 0.045
        tipped = None
        for i in range(480):
            q = scene.carton.data.root_quat_w
            f_b = quat_apply_inverse(q, f_w)
            t_b = torch.cross(r_b, f_b, dim=-1)
            scene.carton.set_external_force_and_torque(
                f_b.reshape(n, 1, 3), t_b.reshape(n, 1, 3))
            env.step(no_action)
            if float(scene.carton_axis_z()[0]) < 0.65:
                tipped = i
                break
        carton_wrench_off()
        assert tipped is not None, \
            f"carton must pass the diagonal (axz={float(scene.carton_axis_z()[0]):.2f})"
        step(180)  # hands off: gravity lays it flat; everything settles
        report("tipped")
        assert bool(scene.carton_flat()[0]), "carton must lie flat inside the drawer"
        assert bool(scene.comp_in()[0]), "companion must remain inside"
        assert abs(float(scene.drawer_open()[0]) - q0) < 0.010, \
            "the tip must not move the drawer"
        s = print_score(f"P1 carton tipped flat inside the drawer ({tipped} push steps)")
        assert s >= c.w_flat - 1e-6, f"P1 score {s:.3f} below the flat credit"

        # ---------------- phase 2: CLOSE — the seed's push, now unblocked ---------------
        # Gentle force-limited PD (<= 1.2 N): max drawer+cargo acceleration
        # ~2.3 m/s^2, far below the ~5 m/s^2 friction limit — the flat cargo
        # rides the floor without sliding.
        for _ in range(700):
            v = scene.drawer.data.root_lin_vel_w[:, 0]
            f = (60.0 * (0.002 - scene.drawer_open()) - 25.0 * v).clamp(-1.2, 1.2)
            fw = torch.zeros(n, 3, device=device)
            fw[:, 0] = f
            fb = quat_apply_inverse(scene.drawer.data.root_quat_w, fw)
            scene.drawer.set_external_force_and_torque(fb.reshape(n, 1, 3), zero3)
            env.step(no_action)
            if float(scene.drawer_open()[0]) <= 0.004:
                break
        drawer_force_off()
        report("pushed")
        assert float(scene.drawer_open()[0]) <= c.q_goal, "drawer must be seated"

        won_at = None
        for i in range(300):
            env.step(no_action)
            if bool(scene.success()[0]):
                won_at = i
                break
        report("closed")
        assert won_at is not None and bool(scene.success()[0]), \
            (f"success must arrive after the push settles: "
             f"q={float(scene.drawer_open()[0]):+.4f} "
             f"flat={bool(scene.carton_flat()[0])} comp={bool(scene.comp_in()[0])}")
        s = print_score(f"P2 drawer seated with the cargo stowed ({won_at} settle steps)")
        assert s >= 1.0 - 1e-6, "success must score 1.0"

        # ---------------- persistence (>= 3 simulated seconds, hands-off) ---------------
        hold = True
        for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
            step(40)
            hold = hold and bool(scene.success()[0])
        report("persist")
        s_final = print_score("P-final persistence 3.3 s")
        ok = hold and bool(scene.success()[0]) and s_final >= 1.0 - 1e-6
        print(f"[solve] episode seed={seed}: {'OK' if ok else 'FAILED'}", flush=True)
        return ok

    ok = episode(args.seed, announce=True)
    ok = episode(args.seed + 1, announce=False) and ok

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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
