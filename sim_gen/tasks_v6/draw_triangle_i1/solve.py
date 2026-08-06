"""Teleport solution for ShapeSorterScene — the task's legitimacy certificate.

NullRobot scene-level env. Teleportation is TRANSPORT ONLY: each piece is carried across
free space to a hover pose ABOVE the lid opening whose rim matches its color (aperture
alignment attitude, zero velocity, bottom ~22 mm above the lid top) and RELEASED there.
The load-bearing interaction this task requires — the aperture passage into containment —
happens through contact dynamics: the piece falls, threads the opening (the card passes
the slot only because its long axis was aligned with it), impacts the compartment floor,
and settles INSIDE the sealed compartment. No piece is ever written into a contained
pose (every release point is fully OUTSIDE the box volume, above the lid), nothing is
pinned against physics, and no scene state other than piece root poses in free space is
ever touched. Every rubric gate (centre inside the assigned compartment, exact piece top
below the lid underside, settled) is earned by the physics end state.

Phases (each boundary prints `SIM_GEN_SCORE`, non-decreasing):
  0. reset(seed), settle; readback (box pose, color->compartment assignment, piece
     spawns) proves the episode is seed-specific -> score ~0
  1-3. POST piece k (cube, cylinder, card): transport-teleport to hover above the
     matching aperture, release, fall + thread + settle >= 1.5 s. Score climbs
     0.25 -> 0.50 -> 1.00 as pieces are contained.
  4. VERIFY (+ bounded repair: extra settle, then re-hover + re-drop any piece whose
     compartment is wrong — at most 2 rounds; a correctly designed drop needs none),
     then require success() and score == 1.0 exactly.
  5. PERSIST: >= 3.5 s of pure simulation, zero intervention; success re-verified at
     every poll and at the end -> print `SIM_GEN_SOLVE: SUCCESS`.

Run (forge): python -u -m simgen_tasks.draw_triangle_i1.solve --headless [--seed N]
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

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.draw_triangle_i1 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

DROP_CLEAR = 0.022  # free-fall gap between piece bottom and the lid top at release


def hard_exit(code: int) -> None:
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        app.close()
    finally:
        os._exit(code)


def main() -> None:  # noqa: PLR0915
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.shape_sorter")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    origin = scene.env_origins[0]

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        print(f"[solve] {tag:12s} | {scene.status_report()}", flush=True)

    last_score = [0.0]

    def score_line(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        if s < last_score[0] - 1e-6:
            print(f"SIM_GEN_SOLVE: FAILURE (score decreased at {tag}: "
                  f"{last_score[0]:.4f} -> {s:.4f})", flush=True)
            hard_exit(1)
        last_score[0] = s
        return s

    # drop attitude half-heights (cube flat, cylinder axis-up, card upright long-axis-x)
    drop_half = (c.cube_edge / 2, c.cyl_h / 2, c.card_size[2] / 2)

    def aperture_xy(i: int) -> tuple[float, float, float]:
        """World (x, y) of piece i's assigned aperture centre + the box yaw (readback)."""
        bxy, yaw = scene.box_frame()
        by = float(yaw[0])
        slot = int(scene.assign[0, i])
        ly = (slot - 1) * c.bin_pitch
        return (float(bxy[0, 0]) - math.sin(by) * ly,
                float(bxy[0, 1]) + math.cos(by) * ly, by)

    def hover_drop(i: int) -> None:
        """TRANSPORT teleport: carry piece i to a hover pose above its matching aperture
        (aligned with the box axes, zero velocity, fully outside the box volume) and
        release. The aperture passage and landing are pure contact dynamics."""
        x, y, byaw = aperture_xy(i)
        st = torch.zeros(1, 13, device=device)
        st[0, 0], st[0, 1] = x, y
        st[0, 2] = c.surface_z + c.lid_z1 + DROP_CLEAR + drop_half[i]
        st[0, 3] = math.cos(byaw / 2)
        st[0, 6] = math.sin(byaw / 2)
        st[0, 0:3] += origin
        scene.pieces[i].write_root_state_to_sim(st, torch.arange(1, device=device))
        step(180)  # fall + thread the opening + impact + settle (1.5 s)

    # ================= phase 0: reset + settle ================================================
    env.reset(seed=args.seed)
    step(60)
    bxy, yaw = scene.box_frame()
    assign = scene.assign[0].tolist()
    spawns = []
    for p in scene.pieces:
        q = (p.data.root_pos_w[0] - origin)
        spawns.append((round(float(q[0]), 3), round(float(q[1]), 3)))
    print(f"[solve] seed={args.seed} readback: box=({float(bxy[0, 0]):.3f},"
          f"{float(bxy[0, 1]):.3f},{math.degrees(float(yaw[0])):.1f}deg) "
          f"assign(red,green,blue)={assign} pieces={spawns}", flush=True)
    report("settled")
    score_line("reset")

    # ================= phases 1-3: POST each piece (hover-release, physical threading) ========
    for i, name in enumerate(("cube", "cylinder", "card")):
        hover_drop(i)
        report(f"posted {name}")
        score_line(name)

    # ================= phase 4: verify (+ bounded repair by re-transport) =====================
    rounds = 0
    while not bool(scene.success()[0]) and rounds < 2:
        rounds += 1
        step(240)  # extra settle first: a slowly rolling cylinder just needs time
        wrong = [i for i in range(3)
                 if int(scene.piece_bin()[0, i]) != int(scene.assign[0, i])]
        print(f"[solve] repair round {rounds}: wrong={wrong}", flush=True)
        for i in wrong:
            hover_drop(i)
        report(f"repair {rounds}")
    step(60)  # everything at rest before judging
    report("posted all")
    s = score_line("posted all")
    if not bool(scene.success()[0]) or abs(s - 1.0) > 1e-3:
        print("SIM_GEN_SOLVE: FAILURE (pieces not all contained / score not 1.0)", flush=True)
        hard_exit(1)

    # ================= phase 5: PERSIST (>= 3.5 s, no intervention) ===========================
    ok = True
    for _ in range(14):  # 14 * 30 = 420 steps = 3.5 s
        step(30)
        ok = ok and bool(scene.success()[0])
    report("persist")
    s_end = score_line("persist")
    if ok and bool(scene.success()[0]) and abs(s_end - 1.0) <= 1e-3:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        hard_exit(0)
    print("SIM_GEN_SOLVE: FAILURE (success did not persist)", flush=True)
    hard_exit(1)


if __name__ == "__main__":
    main()
