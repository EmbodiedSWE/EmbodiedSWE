"""Teleport solution for TiltMazeScene — the task's legitimacy certificate.

NullRobot scene-level env. NO teleport of the ball is ever needed (or used): the whole
solution is the same indirect actuation a robot would perform — bounded torques written
to the scene's `tilt_drive` plant buffer (the wrench a fingertip pressing a yellow rim
tab exerts about the gimbal), while GRAVITY moves the ball:
  - the DEPART/CROSS leg: a small southward tilt (~6 deg equivalent) rolls the ball out
    of the open bay, through the roofed channel, into the cross corridor;
  - the BRANCH leg: read the episode's goal side (the beacon), tilt toward it (with a
    little residual south pressure) until the ball falls into the GOAL well — the
    rubric's `potted` latch fires only from this continuous, physical path;
  - the RELEASE: zero drive; the centering spring re-levels the tray and the ball
    settles in the well — success() requires exactly this released, quiet state.

Phases (each boundary prints `SIM_GEN_SCORE`, non-decreasing):
  0. reset(seed), settle                                    -> score ~0
  1. SOUTH: closed-loop tilt until the ball is in the cross
     corridor; assert `departed` + `crossed`                -> score ~0.25
  2. BRANCH: tilt toward the beacon side until `potted`     -> score 0.70
  3. RELEASE: zero drive, spring re-levels, ball settles    -> score 1.00 (success)
  4. PERSIST: >= 3 s of pure simulation, no intervention; success must hold at every
     poll and at the end -> print `SIM_GEN_SOLVE: SUCCESS`.

Run (forge): python -u -m simgen_tasks.approach_grasp_banana_i69.solve --headless [--seed N]
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

try:
    from .scene import TiltMazeScene  # registers "tilt_maze" + env
except ImportError:  # direct-file fallback
    from scene import TiltMazeScene

assert TiltMazeScene is not None


def hard_exit(code: int) -> None:
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        app.close()
    finally:
        os._exit(code)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.tilt_maze")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        p = scene.ball_local()[0].tolist()
        print(f"[solve] {tag:10s} | ball_loc=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
              f"tilt={math.degrees(float(scene.tray_tilt()[0])):5.2f}deg "
              f"side={float(scene.side[0]):+.0f} "
              f"dep={bool(scene.departed[0])} cross={bool(scene.crossed[0])} "
              f"branch={float(scene.branch[0]):.2f} pot={bool(scene.potted[0])} "
              f"goal_well={bool(scene.in_goal_well()[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])}",
              flush=True)

    def score_line() -> float:
        s = float(scene.score()[0])
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ================= phase 0: reset + settle ================================================
    env.reset(seed=args.seed)
    step(60)
    report("settled")
    s0 = score_line()
    p = scene.ball_local()[0]
    if not (abs(float(p[0])) < c.chan_hw and c.slat_y_hi < float(p[1]) < c.bay_y_hi
            and abs(float(p[2]) - c.ball_rest_floor) < 0.01):
        print("SIM_GEN_SOLVE: FAILURE (ball not resting in the start bay after reset)",
              flush=True)
        hard_exit(1)

    # ================= phase 1: SOUTH (roll out of the bay into the cross corridor) ===========
    # Equilibrium mapping: drive_x = +kappa*sin(a) tilts the south edge down by ~a;
    # drive_y = side*kappa*sin(a) tilts the goal-side edge down. Mild x-centering keeps
    # the ball off the channel walls. All torques stay far below drive_max.
    south = c.kappa * math.sin(math.radians(6.0))
    ok = False
    for i in range(1800):
        p = scene.ball_local()[0]
        if float(p[1]) < -0.055:
            ok = True
            break
        cx = -c.kappa * max(-0.05, min(0.05, 1.5 * float(p[0])))  # center the channel
        scene.tilt_drive[:, 0] = south
        scene.tilt_drive[:, 1] = cx
        env.step(no_action)
        if i % 240 == 0:
            report(f"south{i:4d}")
    report("crossed")
    s1 = score_line()
    if not (ok and bool(scene.departed[0]) and bool(scene.crossed[0])):
        print("SIM_GEN_SOLVE: FAILURE (ball did not traverse into the cross corridor)",
              flush=True)
        hard_exit(1)
    if s1 < s0 - 1e-6:
        print("SIM_GEN_SOLVE: FAILURE (score decreased across the south leg)", flush=True)
        hard_exit(1)

    # ================= phase 2: BRANCH (toward the beacon; fall into the goal well) ===========
    side = float(scene.side[0])
    branch = side * c.kappa * math.sin(math.radians(7.0))
    hold_s = c.kappa * math.sin(math.radians(2.5))
    ok = False
    for i in range(1800):
        if bool(scene.potted[0]):
            ok = True
            break
        scene.tilt_drive[:, 0] = hold_s
        scene.tilt_drive[:, 1] = branch
        env.step(no_action)
        if i % 240 == 0:
            report(f"branch{i:3d}")
    report("potted")
    s2 = score_line()
    if not ok:
        print("SIM_GEN_SOLVE: FAILURE (ball never potted in the goal well)", flush=True)
        hard_exit(1)
    if s2 < s1 - 1e-6:
        print("SIM_GEN_SOLVE: FAILURE (score decreased across the branch leg)", flush=True)
        hard_exit(1)

    # ================= phase 3: RELEASE (spring re-levels; ball settles in the well) ==========
    scene.tilt_drive[:] = 0.0
    step(300)  # 2.5 s: spring recentering + ball settling
    report("released")
    s3 = score_line()
    if not bool(scene.success()[0]):
        print("SIM_GEN_SOLVE: FAILURE (no settled success after release)", flush=True)
        hard_exit(1)
    if s3 < s2 - 1e-6 or abs(s3 - 1.0) > 1e-3:
        print("SIM_GEN_SOLVE: FAILURE (score not 1.0 at success)", flush=True)
        hard_exit(1)

    # ================= phase 4: PERSIST (>= 3 s, no intervention) ==============================
    ok = True
    for _ in range(14):  # 14 * 30 = 420 steps = 3.5 s
        step(30)
        ok = ok and bool(scene.success()[0])
    report("persist")
    s4 = score_line()
    if ok and bool(scene.success()[0]) and s4 >= s3 - 1e-6:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        hard_exit(0)
    print("SIM_GEN_SOLVE: FAILURE (success did not persist)", flush=True)
    hard_exit(1)


if __name__ == "__main__":
    main()
