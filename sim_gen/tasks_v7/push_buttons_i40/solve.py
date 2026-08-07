"""Teleport solution for CounterweightScaleScene (sim_gen task `push_buttons_i40`) —
the task's legitimacy certificate.

Teleportation is TRANSPORT ONLY: each selected weight block is teleported to a hover
point a few centimetres ABOVE the blue pan's opening and released. Everything
load-bearing happens through CONTACT DYNAMICS: the block free-falls into the pan,
the pan floor carries its weight into the hanger, the revolute beam responds to the
changed moment, and the pendulum bob settles the beam — level only if the loaded
mass truly equals the reference mass. Nothing is ever teleported into the pan, no
pose is pinned, and after the last drop the scene is HANDS-OFF: gravity alone
produces the judged equilibrium.

Plan (oracle side): read k = number of reference cubes from the scene, decompose k
in binary over the block units {1, 2, 4} (each k in 1..7 has exactly one subset),
drop the selected blocks largest-first at tiling offsets inside the pan, settle
between drops. If a drop bounces out (readback), re-transport above the pan and drop
again.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.push_buttons_i40.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.counterweight_scale")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(kk: int) -> None:
        for _ in range(kk):
            env.step(no_action)

    def report(tag: str) -> None:
        t = math.degrees(float(scene.tilt()[0]))
        print(f"[solve] {tag:16s} | tilt={t:+6.2f} deg blue_units="
              f"{float(scene.blue_units()[0]):.1f} refs_home={bool(scene.refs_home()[0])} "
              f"cand_ok={bool(scene.cand_ok()[0])} level={bool(scene.level()[0])} "
              f"still={bool(scene.still()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, read the task instance -----------------------
    step(360)  # beam swings onto the red-side limit and everything comes to rest
    k = int(scene.n_ref[0])
    subset = [j for j, (_nm, _s, u) in enumerate(c.cand_specs) if k & u]
    total = sum(c.cand_specs[j][2] for j in subset)
    assert total == k, f"binary decomposition failed: k={k} subset={subset}"
    names = [c.cand_specs[j][0] for j in subset]
    print(f"[solve] readback (seed {args.seed}): k={k} reference cubes -> subset "
          f"{names} (units {[c.cand_specs[j][2] for j in subset]})", flush=True)
    bp = (scene.base.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve]   scale axis at ({float(bp[0]):+.3f},{float(bp[1]):+.3f}), "
          f"tilt={math.degrees(float(scene.tilt()[0])):+.2f} deg (red-side limit)", flush=True)
    report("reset")
    s_prev = print_score("P0 reset+settle")

    # ---------------- phases 1..len(subset): drop each selected block ----------------------
    # Hover point: pan-local (ox, oy, hover_z) mapped through the pan's live pose —
    # the pan self-levels, so the drop is always onto a level floor. The block's
    # BOTTOM starts 10 mm above the rim top (pan-local -0.058): it clears the rim,
    # falls ~42 mm onto the floor, and the pan+beam absorb the impact through
    # contact with minimal recoil.
    tile = {"large": (-0.025, -0.025), "medium": (0.030, 0.025), "small": (0.030, -0.030)}
    from isaaclab.utils.math import quat_apply

    def settle_scale(max_rounds: int = 12) -> None:
        """Hands-off wait until the beam and pans stop moving (or the budget ends)."""
        for _ in range(max_rounds):
            step(60)
            if bool(scene.still()[0]):
                break

    def drop(j: int, phase: int) -> None:
        nonlocal s_prev
        nm, size, units = c.cand_specs[j]
        body = scene.cands[j]
        ok = False
        for attempt in range(3):
            ox, oy = tile[nm]
            if attempt > 0:  # retry: aim dead centre
                ox, oy = 0.0, 0.0
            loc = torch.tensor([ox, oy, -0.058 + size / 2], device=device).expand(n, 3)
            pq = scene.pan_blue.data.root_quat_w
            pp = scene.pan_blue.data.root_pos_w
            st = torch.zeros(n, 13, device=device)
            st[:, 0:3] = pp + quat_apply(pq, loc)
            st[:, 3:7] = pq  # match the pan's yaw so the cube faces the rims squarely
            body.write_root_state_to_sim(st, all_ids)
            step(60)   # free fall + impact through contact
            settle_scale()  # beam responds and settles, hands off
            if bool(scene.cand_in_blue()[0, j]):
                ok = True
                break
            print(f"[solve] block {nm}: not in the pan after drop "
                  f"(attempt {attempt}), re-transporting", flush=True)
        if not ok:
            report("FAIL-drop")
            print(f"SIM_GEN_SOLVE: FAIL (block {nm} would not stay in the blue pan)",
                  flush=True)
            os._exit(1)
        report(f"drop-{nm}")
        s_now = print_score(f"P{phase} drop {nm} ({units}u) + settle")
        assert s_now >= s_prev - 1e-6, f"score decreased across drop {nm}"
        s_prev = s_now

    # largest first: the heaviest impact lands on the emptiest pan
    for phase, j in enumerate(sorted(subset, key=lambda jj: -c.cand_specs[jj][2]), start=1):
        drop(j, phase)

    # ---------------- final settle: wait for the balanced equilibrium ----------------------
    for _ in range(20):
        if bool(scene.success()[0]):
            break
        step(60)
    report("balanced")
    s_prev = print_score("P9 balanced + settled")
    if not bool(scene.success()[0]):
        print("SIM_GEN_SOLVE: FAIL (no success after loading the subset)", flush=True)
        os._exit(1)

    # ---------------- persistence (>= 3.3 simulated seconds, no intervention) --------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P10 persistence 3.3 s")
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
    main()
