"""Teleport solution for HanoiPlatesScene (sim_gen task
libero_kitchen_scene2_put_the_black_bowl_at_the_back_on_the_plate_i56) — the task's
legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): each of the 7 Hanoi moves is ONE root-state write that
   carries a single plate from its current resting place to the free-space point
   ~6 mm ABOVE THE TIP of the destination post, flat, zero velocity. The lift off the
   source post is a free vertical translation (9-13 mm of radial slack around a
   straight post — nothing to unlatch), and the write satisfies no rubric clause by
   itself: at the drop point the plate is airborne and OUTSIDE the threaded z-span.
2. THREADING + STACKING (gravity + contact): from the drop point the plate falls, the
   post enters its aperture, and the plate slides ~60-110 mm down the post until it
   lands on the plank / green pad / the boss ring of the plate below, tilting,
   contacting and settling under real dynamics. Every fact success() checks (threaded
   xy/z on the target post, flatness, stack order, stillness) is produced by this
   contact interaction, never written. If a plate cocked and wedged on the post the
   run would simply fail — no force is ever applied to coax the rubric.
3. RULES (trajectory facts): the scene monitors one-plate-at-a-time, no-parking and
   never-larger-on-smaller at every physics step and latches violations permanently.
   The solution moves strictly one plate per phase and only ever drops onto legal
   supports, so the latches stay clear — the canonical 7-move recursion
   S->T, M->spare, S->spare, L->T, S->start, M->T, S->T is what the rules force.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: stage credit
is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success() first
turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

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
    from . import scene as scene_mod  # noqa: F401
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

PLATE = {0: "red", 1: "yellow", 2: "blue"}


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.hanoi_plates")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        thr = scene.threaded()[0]  # (3 plates, 3 posts)
        posts = ["-" if not thr[i].any() else str(int(thr[i].float().argmax()))
                 for i in range(3)]
        z = scene._plate_loc()[0, :, 2]
        print(f"[solve] {tag:16s} | plate->post R:{posts[0]} Y:{posts[1]} B:{posts[2]} "
              f"z=({float(z[0]):.3f},{float(z[1]):.3f},{float(z[2]):.3f}) "
              f"rest={scene.at_rest()[0].tolist()} violated={bool(scene._violated[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def drop(plate_idx: int, post_idx: int, tag: str) -> None:
        """One Hanoi move: teleport plate `plate_idx` to 6 mm above the tip of post
        `post_idx` (flat, zero velocity), then hands-off until it threads down and
        comes to rest. The threading and landing are pure gravity + contact."""
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0:2] = scene._post_xy[post_idx]
        loc[:, 2] = c.base_h + c.post_h + c.plate_t / 2 + 0.006
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.rack.data.root_pos_w + quat_apply(scene.rack.data.root_quat_w, loc)
        st[:, 3:7] = scene.rack.data.root_quat_w
        scene.plates[plate_idx].write_root_state_to_sim(st, all_ids)
        for _ in range(360):
            env.step(no_action)
            thr = scene.threaded()[0, plate_idx, post_idx]
            if bool(thr) and bool(scene.at_rest()[0, plate_idx]):
                break
        step(30)  # extra settle, hands-off
        report(tag)
        assert bool(scene.threaded()[0, plate_idx, post_idx]), \
            f"{tag}: {PLATE[plate_idx]} plate failed to thread onto post {post_idx}"
        assert bool(scene.at_rest()[0, plate_idx]), f"{tag}: plate never came to rest"
        assert not bool(scene._violated[0]), f"{tag}: a rule latch fired"

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)  # tower settles onto the start post
    S = int(scene.start_post[0])
    T = int(scene.target_post[0])
    P = 3 - S - T
    rp = (scene.rack.data.root_pos_w - scene.env_origins)[0]
    rq = scene.rack.data.root_quat_w[0]
    ryaw = math.degrees(2.0 * math.atan2(float(rq[3]), float(rq[0])))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"rack=({float(rp[0]):+.3f},{float(rp[1]):+.3f}) yaw={ryaw:+.1f}deg "
          f"start_post={S} target_post={T} spare={P}", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    thr = scene.threaded()[0]
    assert all(bool(thr[i, S]) for i in range(3)), "tower must start on the start post"
    assert not bool(scene._violated[0]), "fresh reset must not violate"
    s0 = print_score("P0 reset+settle (tower on the start post)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phases 1..7: the canonical Hanoi recursion ---------------------------
    moves = [(2, T, "M1 blue->target"),
             (1, P, "M2 yellow->spare"),
             (2, P, "M3 blue->spare (onto yellow)"),
             (0, T, "M4 red->target (the unlocked move)"),
             (2, S, "M5 blue->start"),
             (1, T, "M6 yellow->target (onto red)"),
             (2, T, "M7 blue->target (onto yellow)")]
    expect = [0.15, 0.15, 0.15, 0.50, 0.50, 0.75, 1.0]
    prev = s0
    for (pi, po, tag), exp in zip(moves, expect):
        drop(pi, po, tag)
        s = print_score(tag)
        assert s >= prev - 1e-6, f"{tag}: score decreased {prev} -> {s}"
        assert s >= exp - 1e-6, f"{tag}: expected >= {exp}, got {s}"
        prev = s

    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after move 7)", flush=True)
        os._exit(1)

    # ---------------- phase 8: persistence (>= 3.3 simulated seconds, hands-off) -----------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s_end = print_score("P8 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s_end >= prev - 1e-6
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
