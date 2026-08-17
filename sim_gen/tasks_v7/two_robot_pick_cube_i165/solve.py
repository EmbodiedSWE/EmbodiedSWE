"""Teleport solution for SortingTowerScene (sim_gen task `two_robot_pick_cube_i165`) —
the task's legitimacy certificate.

Teleports are TRANSPORT ONLY (relocating a disc the solver is already holding in free
air). Every load-bearing interaction is contact dynamics:

1. PERCEPTION: the tower pose, WHICH slab slot holds which disc, and which size the
   red distractor is are read back from the episode state — never hard-coded. The
   red disc is identified and deliberately left untouched.
2. INSERT x3 (gravity + contact): in ASCENDING size order — the order the keyed
   geometry demands — each GREEN disc is TELEPORTED from its slab slot to free air
   3 cm above the tower's mouth rim, axis vertical, and RELEASED. Gravity drops it
   through the mouth funnel; the internal funnels and tube walls — real contacts —
   guide it down and seat it at its own keyed depth. The rubric judges the settled
   seated pose, not the carry. (Any other order fails: a seated disc wedges the
   shaft, and a larger disc inserted early strands the smaller ones on top —
   smoke.py proves both with real drops.)
3. VERIFY: hands off; all three discs seated at their bands, red outside, settled;
   success() must hold and keep holding for >= 3.3 more simulated seconds before
   SIM_GEN_SOLVE: SUCCESS prints.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the rubric
latches seat credit and success only adds).

Run (forge): python -u -m simgen_tasks.two_robot_pick_cube_i165.solve --headless [--seed N]
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
    env = ENVS.get("simgen.sorting_tower")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        seat = scene.seated()[0]
        locs = [scene._tower_local(scene.cargo[k].data.root_pos_w)[0] for k in range(3)]
        loc_s = " ".join(
            f"d{k}=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f})"
            for k, p in enumerate(locs))
        print(f"[solve] {tag:16s} | seated={[bool(v) for v in seat]} {loc_s} "
              f"red_in={bool(scene.red_inside()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ----- phase 0: settle + perception ----------------------------------------------------------
    step(60)
    report("settled start")
    print(f"[solve] perception: slot permutation (disc->slot az idx) = "
          f"{scene.slot_of[0].tolist()}, active red = size {int(scene.active_red[0])} "
          f"(leave it alone), tower z_seats = "
          f"{[f'{v:.3f}' for v in scene.z_seat_t]}", flush=True)
    s0 = print_score("start")
    assert s0 < 0.05, f"fresh episode must score ~0, got {s0}"

    # ----- phases 1..3: INSERT, ascending size order ---------------------------------------------
    # Transport-only teleport: hold the disc in free air 3 cm above the mouth rim,
    # axis vertical, and release. Gravity + funnels do all the seating work.
    prev = s0
    for k in range(3):
        drop = scene.tower_world([0.0, 0.0, c.mz1 + 0.03 + c.disc_h[k] / 2])
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = drop
        st[:, 3:7] = scene.tower.data.root_quat_w  # flat: axis parallel to the shaft
        scene.cargo[k].write_root_state_to_sim(st)
        # fall + funnel descent + settle; break as soon as this disc is seated & calm
        for i in range(720):
            env.step(no_action)
            if i >= 90 and bool((scene.seated()[:, k] & scene.settled()).all()):
                break
        report(f"inserted d{k}")
        assert bool(scene.seated()[:, k].all()), \
            f"disc {k} must be seated at its keyed band after its drop"
        s = print_score(f"disc {k} seated")
        assert s >= prev, "score must not decrease"
        assert s >= (k + 1) * c.w_seat - 0.01, f"seat latch {k} must pay, got {s}"
        prev = s

    # ----- phase 4: VERIFY — hands off, success must hold and keep holding ----------------------
    step(60)
    report("verify")
    ok = scene.success()
    assert bool(ok.all()), "success() must hold before the persistence window"
    s3 = print_score("success reached")
    assert s3 >= 0.999, f"live success must score 1.0, got {s3}"

    step(400)  # >= 3.3 simulated seconds, hands off
    report("persistence")
    ok = scene.success()
    s4 = float(scene.score()[0])
    print(f"SIM_GEN_SCORE {s4:.4f}", flush=True)
    if bool(ok.all()) and s4 >= 0.999:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        os._exit(0)
    print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)
    os._exit(1)


try:
    main()
except Exception as e:  # noqa: BLE001
    print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
    os._exit(1)
