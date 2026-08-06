"""Teleport solution for BarTriangleScene — the task's legitimacy certificate.

NullRobot scene-level env. Teleportation is TRANSPORT ONLY: each bar is carried across
free space to a hover pose 30 mm ABOVE its planned frame slot (planned yaw, zero
velocity) and RELEASED there. The placement itself — the load-bearing interaction this
task requires — happens through contact dynamics: the bar falls, impacts the mat, and
settles flat at ground height, and only the SETTLED pose the sim reports can close the
per-corner gaps. No bar is ever written into a seated/settled pose (release is a full
bar-height-plus above rest), nothing is pinned against physics, and no scene state
other than bar root poses in free space is ever touched. Every rubric gate (flat at
ground height at BOTH ends, settled, entirely on the mat, three closed corner gaps,
distinct-ends cycle, Heron-anchored area) is earned by the physics end state.

Phases (each boundary prints `SIM_GEN_SCORE`, non-decreasing):
  0. reset(seed), settle; layout readback proves the episode is seed-specific -> ~0
  1. PLAN: read the settled mat centre; `frame_layout` solves the ideal frame (uniform
     ~24 mm corner gaps -> 16 mm margin under the 40 mm tolerance, ~6 mm edge clearance
     between neighbouring bars) centred on the mat.
  2-4. PLACE bar k (k = 0, 1, 2): transport-teleport to hover 30 mm above the planned
     slot, release, settle >= 1 s under gravity/contact. Partial credit climbs
     0.05 -> 0.30 -> 1.00 as bars validate and corners close.
  5. REPAIR (rarely needed): while success() is False, re-hover + re-drop the placed
     bar deviating most from plan (at most 2 rounds), then require success and
     score == 1.0 exactly.
  6. PERSIST: >= 3.5 s of pure simulation, zero intervention; success re-verified at
     every poll and at the end -> print `SIM_GEN_SOLVE: SUCCESS`.

Run (forge): python -u -m simgen_tasks.draw_triangle_i8.solve --headless [--seed N]
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
    from simgen_tasks.draw_triangle_i8 import scene as scene_mod
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod

frame_layout = scene_mod.frame_layout

DROP_H = 0.030  # release height above rest: a full bar-height + 12 mm of free fall
PLAN_ROT = 0.35  # frame rotation on the mat (any value works; fixed for repeatability)
PLAN_GAP = 0.024  # designed corner gap: 16 mm margin under joint_tol = 40 mm


def hard_exit(code: int) -> None:
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        app.close()
    finally:
        os._exit(code)


def main() -> None:  # noqa: PLR0915
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bar_triangle")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(1, device=device)
    origin = scene.env_origins[0]

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        print(f"[solve] {tag:12s} | {scene.corner_report()}", flush=True)

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

    def bar_pose(i: int) -> tuple[float, float, float]:
        """Env-local (x, y, yaw of the long axis) of bar i."""
        from isaaclab.utils.math import quat_apply

        p = scene.bars[i].data.root_pos_w[0] - origin
        q = scene.bars[i].data.root_quat_w[0]
        ax = quat_apply(q.unsqueeze(0), torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        return float(p[0]), float(p[1]), math.atan2(float(ax[1]), float(ax[0]))

    def hover_drop(i: int, x: float, y: float, yaw: float) -> None:
        """TRANSPORT teleport: carry bar i to a hover pose above its planned slot and
        release it (zero velocity). The landing is pure contact dynamics."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0], st[0, 1] = x, y
        st[0, 2] = c.surface_z + c.bar_w / 2 + DROP_H
        st[0, 3] = math.cos(yaw / 2)
        st[0, 6] = math.sin(yaw / 2)
        st[0, 0:3] += origin
        scene.bars[i].write_root_state_to_sim(st, all_ids)
        step(120)  # fall + impact + settle (1.0 s)

    # ================= phase 0: reset + settle ================================================
    env.reset(seed=args.seed)
    step(60)
    mat = (scene.pad.data.root_pos_w[0] - origin)
    mx, my = float(mat[0]), float(mat[1])
    spawns = [bar_pose(i) for i in range(3)]
    print(f"[solve] seed={args.seed} layout readback: mat=({mx:.3f},{my:.3f}) bars="
          + " ".join(f"({x:.3f},{y:.3f},{math.degrees(w):.0f}deg)" for x, y, w in spawns),
          flush=True)
    report("settled")
    score_line("reset")

    # ================= phase 1: PLAN ==========================================================
    # Ideal frame centred on the settled mat: per-corner retractions solved so every
    # endpoint gap lands at ~24 mm (16 mm under tolerance), bars ~6 mm apart at corners.
    centers, yaws, verts, plan_gaps = frame_layout(c, (mx, my), PLAN_ROT, corner_gap=PLAN_GAP)
    print(f"[solve] plan: centers={[(round(p[0], 3), round(p[1], 3)) for p in centers]} "
          f"gaps_mm={[round(g * 1000, 1) for g in plan_gaps]}", flush=True)

    # ================= phases 2-4: PLACE (hover-release, physical landing) ===================
    for k in range(3):
        hover_drop(k, centers[k][0], centers[k][1], yaws[k])
        report(f"placed b{k}")
        score_line(f"bar {k}")

    # ================= phase 5: verify (+ repair by re-transport if ever needed) ==============
    rounds = 0
    while not bool(scene.success()[0]) and rounds < 2:
        rounds += 1
        dev = []
        for i in range(3):
            x, y, w = bar_pose(i)
            dyaw = abs((w - yaws[i] + math.pi / 2) % math.pi - math.pi / 2)
            dev.append(math.hypot(x - centers[i][0], y - centers[i][1]) + 0.05 * dyaw)
        worst = max(range(3), key=lambda i: dev[i])
        print(f"[solve] repair round {rounds}: re-dropping bar {worst} "
              f"(deviation {dev[worst] * 1000:.1f} mm-eq)", flush=True)
        hover_drop(worst, centers[worst][0], centers[worst][1], yaws[worst])
        report(f"repair {rounds}")
    step(60)  # everything at rest before judging
    report("built")
    s = score_line("built")
    if not bool(scene.success()[0]) or abs(s - 1.0) > 1e-3:
        print("SIM_GEN_SOLVE: FAILURE (frame did not close / score not 1.0)", flush=True)
        hard_exit(1)

    # ================= phase 6: PERSIST (>= 3.5 s, no intervention) ===========================
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
