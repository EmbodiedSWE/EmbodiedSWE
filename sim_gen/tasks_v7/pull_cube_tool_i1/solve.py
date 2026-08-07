"""Teleport solution for CarouselFerryScene — the task's legitimacy certificate.

NullRobot scene-level env. Teleportation is used for TRANSPORT ONLY (lifting the cube
off the platter and across free space to above the open-top tray); every load-bearing
interaction goes through contact dynamics:
  - the FERRY: a bounded torque on the platter's free axle (the same wrench a peg push
    exerts, applied through the scene's `platter_drive` plant buffer) rotates the
    carousel while the cube rides the platter top by real friction — the rubric's
    `ferried` latch only fires from this physical carried transport;
  - the PLACEMENT: the cube is released ABOVE the tray mouth and falls, impacts, and
    settles between the walls under gravity and contact — never spawned seated.

Phases (each boundary prints `SIM_GEN_SCORE`, non-decreasing):
  0. reset(seed), settle                                  -> score ~0
  1. FERRY: closed-loop axle torque until the cube azimuth reaches the near sector and
     the platter settles; assert `ferried`                -> score 0.30
  2. DELIVER: teleport the cube to 9.5 cm above the tray-mat centre (clear of the walls),
     zero velocity; it drops in and settles               -> score 1.00 (success)
  3. PERSIST: >= 3 s of pure simulation, no intervention; success must hold at every
     poll and at the end -> print `SIM_GEN_SOLVE: SUCCESS`.

Run (forge): python -u -m simgen_tasks.pull_cube_tool_i1.solve --headless [--seed N]
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
    from .scene import CarouselFerryScene  # registers "carousel_ferry" + env
except ImportError:  # direct-file fallback
    from scene import CarouselFerryScene

assert CarouselFerryScene is not None


def _wrap(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


def hard_exit(code: int) -> None:
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        app.close()
    finally:
        os._exit(code)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.carousel_ferry")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        cube = (scene.cube.data.root_pos_w[0] - scene.env_origins[0]).tolist()
        print(f"[solve] {tag:10s} | cube=({cube[0]:.3f},{cube[1]:.3f},{cube[2]:.3f}) "
              f"az={math.degrees(float(scene.cube_azimuth()[0])):7.1f}deg "
              f"sweep={math.degrees(float(scene.net_sweep[0])):7.1f}deg "
              f"riding={bool(scene.riding()[0])} ferried={bool(scene.ferried[0])} "
              f"placed={bool(scene.placed[0])} in_tray={bool(scene.in_tray()[0])} "
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
    if not bool(scene.riding()[0]):
        print("SIM_GEN_SOLVE: FAILURE (cube not riding the platter after reset)", flush=True)
        hard_exit(1)

    # ================= phase 1: FERRY (axle torque, cube rides by friction) ===================
    # Closed loop on the CUBE's azimuth (0 = far, pi = near): torque = kp*e - kd*omega,
    # bounded to what a fingertip push on a peg exerts (~5 N at the 0.26 m peg circle).
    kp, kd, tau_max = 2.0, 1.5, 1.2
    done_rot = False
    for i in range(2400):
        az = float(scene.cube_azimuth()[0])
        om = float(scene.platter_rate()[0])
        e = _wrap(math.pi - az)
        if abs(e) < math.radians(4.0) and abs(om) < 0.05:
            done_rot = True
            break
        scene.platter_drive[:] = max(-tau_max, min(tau_max, kp * e - kd * om))
        env.step(no_action)
        if i % 240 == 0:
            report(f"ferry{i:4d}")
    scene.platter_drive[:] = 0.0
    step(90)  # platter spins down under the axle's viscous friction; cube settles
    report("ferried")
    s1 = score_line()
    if not (done_rot and bool(scene.ferried[0]) and bool(scene.riding()[0])):
        print("SIM_GEN_SOLVE: FAILURE (ferry did not complete or latch)", flush=True)
        hard_exit(1)
    if s1 < s0 - 1e-6:
        print("SIM_GEN_SOLVE: FAILURE (score decreased across the ferry)", flush=True)
        hard_exit(1)

    # ================= phase 2: DELIVER (transport teleport + physical drop) ==================
    # Transport only: the cube is moved across free space to hover ABOVE the open tray
    # mouth (clear of the walls), then dropped — insertion happens through gravity and
    # contact against the mat and walls.
    tray = scene.tray_xy()[0]
    drop_z = c.wall_top + c.cube_size / 2 + 0.025  # bottom face ~2.5 cm above the wall top
    st = torch.zeros(1, 13, device=device)
    st[0, 0] = tray[0]
    st[0, 1] = tray[1]
    st[0, 2] = drop_z
    st[0, 0:3] += scene.env_origins[0]
    st[0, 3:7] = scene.cube.data.root_quat_w[0]  # keep the carried yaw
    scene.cube.write_root_state_to_sim(st, torch.arange(1, device=device))
    step(240)  # drop + settle (2 s)
    report("delivered")
    s2 = score_line()
    if not bool(scene.success()[0]):
        print("SIM_GEN_SOLVE: FAILURE (cube did not settle inside the tray)", flush=True)
        hard_exit(1)
    if s2 < s1 - 1e-6 or abs(s2 - 1.0) > 1e-3:
        print("SIM_GEN_SOLVE: FAILURE (score not 1.0 at success)", flush=True)
        hard_exit(1)

    # ================= phase 3: PERSIST (>= 3 s, no intervention) ==============================
    ok = True
    for _ in range(14):  # 14 * 30 = 420 steps = 3.5 s
        step(30)
        ok = ok and bool(scene.success()[0])
    report("persist")
    s3 = score_line()
    if ok and bool(scene.success()[0]) and s3 >= s2 - 1e-6:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        hard_exit(0)
    print("SIM_GEN_SOLVE: FAILURE (success did not persist)", flush=True)
    hard_exit(1)


if __name__ == "__main__":
    main()
