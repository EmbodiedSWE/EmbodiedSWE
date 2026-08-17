"""Teleport solution for ShelfDropDispatchScene — the task's legitimacy certificate.

NullRobot scene-level env. Teleportation is used for TRANSPORT ONLY (lifting the cube
out of the catch pit and across free space to above the pedestal top); every
load-bearing interaction goes through contact dynamics:
  - the RELEASE: a bounded horizontal force on the blue column's CoM (the same push a
    fingertip exerts on its face, applied through the scene's `prop_force` plant
    buffer) slides the loaded column along its channel out from under the shelf —
    real friction under real load; the shelf then swings down on its damped hinge
    purely under gravity;
  - the DELIVERY: entirely passive — the cube slides down the released ramp and into
    the walled catch pit under gravity and contact; the solver never touches it;
  - the PLACEMENT: the cube is released ABOVE the pedestal top and falls, impacts,
    and settles on it — never spawned seated.

Phases (each boundary prints `SIM_GEN_SCORE`, non-decreasing):
  0. reset(seed), settle                                     -> score ~0
  1. EXTRACT: closed-loop y-velocity servo force on the column until it is fully clear
     of the shelf width; the shelf swings down; wait for the cube to slide into the
     pit and settle; assert `released` + `delivered`         -> score 0.55
  2. PLACE: teleport the cube to 3 cm above the pedestal top centre, zero velocity;
     it drops and settles                                    -> score 1.00 (success)
  3. PERSIST: >= 3.5 s of pure simulation, no intervention; success must hold at every
     poll and at the end -> print `SIM_GEN_SOLVE: SUCCESS`.

Run (forge): python -u -m simgen_tasks.pull_cube_tool_i186.solve --headless [--seed N]
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
    from .scene import ShelfDropDispatchScene  # registers "shelf_drop_dispatch" + env
except ImportError:  # direct-file fallback
    from scene import ShelfDropDispatchScene

assert ShelfDropDispatchScene is not None


def hard_exit(code: int) -> None:
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        app.close()
    finally:
        os._exit(code)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.shelf_drop_dispatch")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        cube = scene.cube_pos()[0].tolist()
        print(f"[solve] {tag:10s} | cube=({cube[0]:.3f},{cube[1]:.3f},{cube[2]:.3f}) "
              f"pitch={math.degrees(float(scene.shelf_pitch()[0])):6.1f}deg "
              f"prop_y={float(scene.prop_y()[0]):6.3f} "
              f"released={bool(scene.released[0])} delivered={bool(scene.delivered[0])} "
              f"in_pit={bool(scene.in_pit()[0])} on_ped={bool(scene.on_pedestal()[0])} "
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
    cube0 = scene.cube_pos()[0]
    on_shelf = abs(float(cube0[2]) - (c.shelf_top + c.cube_size / 2)) < 0.02
    level = abs(math.degrees(float(scene.shelf_pitch()[0]))) < 2.0
    if not (on_shelf and level):
        print("SIM_GEN_SOLVE: FAILURE (initial state not level shelf + cube on it)", flush=True)
        hard_exit(1)

    # ================= phase 1: EXTRACT (force on the column; passive delivery) ===============
    # Closed loop on the column's y velocity: F = kv * (v_des - v), bounded to a
    # fingertip-scale push (|F| <= 12 N). Pull toward -y until the column is fully
    # clear of the shelf width, then let go: the shelf swings down under gravity
    # (damped hinge) and the cube slides into the pit on its own.
    kv, f_max, v_des = 40.0, 12.0, -0.10
    y_goal = -(c.prop_clear_y + 0.03)
    done_ext = False
    for i in range(900):
        y = float(scene.prop_y()[0])
        if y <= y_goal:
            done_ext = True
            break
        vy = float(scene.prop.data.root_lin_vel_w[0, 1])
        f = max(-f_max, min(f_max, kv * (v_des - vy)))
        scene.prop_force[:, 1] = f
        env.step(no_action)
        if i % 120 == 0:
            report(f"extract{i:3d}")
    scene.prop_force[:] = 0.0
    report("extracted")
    if not done_ext:
        print("SIM_GEN_SOLVE: FAILURE (column extraction did not complete)", flush=True)
        hard_exit(1)

    # Passive delivery: shelf swings down, cube slides down the ramp into the pit.
    ok_del = False
    for i in range(1200):
        env.step(no_action)
        if (bool(scene.released[0]) and bool(scene.delivered[0])
                and bool(scene.in_pit()[0])
                and float(scene.cube.data.root_lin_vel_w[0].norm()) < 0.03
                and float(scene.shelf.data.root_ang_vel_w[0].norm()) < 0.05):
            ok_del = True
            break
        if i % 240 == 0:
            report(f"deliver{i:4d}")
    step(60)
    report("delivered")
    s1 = score_line()
    if not ok_del:
        print("SIM_GEN_SOLVE: FAILURE (gravity delivery into the pit did not complete)",
              flush=True)
        hard_exit(1)
    if s1 < s0 - 1e-6 or s1 < 0.53:
        print("SIM_GEN_SOLVE: FAILURE (release+delivery credit not latched)", flush=True)
        hard_exit(1)

    # ================= phase 2: PLACE (transport teleport + physical drop) ====================
    # Transport only: the cube is lifted out of the open pit and moved across free
    # space to hover ABOVE the pedestal top, then dropped — seating happens through
    # gravity and contact against the pedestal face.
    ped = scene.ped_xy()[0]
    st = torch.zeros(1, 13, device=device)
    st[0, 0] = ped[0]
    st[0, 1] = ped[1]
    st[0, 2] = c.ped_rest_z + 0.03  # bottom face 3 cm above the pedestal top
    st[0, 0:3] += scene.env_origins[0]
    st[0, 3] = 1.0  # upright (the arm reorients the cube in hand)
    scene.cube.write_root_state_to_sim(st, torch.arange(1, device=device))
    step(240)  # drop + settle (2 s)
    report("placed")
    s2 = score_line()
    if not bool(scene.success()[0]):
        print("SIM_GEN_SOLVE: FAILURE (cube did not settle centered on the pedestal)",
              flush=True)
        hard_exit(1)
    if s2 < s1 - 1e-6 or abs(s2 - 1.0) > 1e-3:
        print("SIM_GEN_SOLVE: FAILURE (score not 1.0 at success)", flush=True)
        hard_exit(1)

    # ================= phase 3: PERSIST (>= 3.5 s, no intervention) ============================
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
