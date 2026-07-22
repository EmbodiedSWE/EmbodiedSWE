"""Franka binding smoke for push_cube_ref: boot, home, reach, and a scripted
robot solution (push the cube into the goal with the closed gripper).

Run:  python -m sim_gen.tasks.push_cube_ref.binding_smoke \
          --video sim_gen/artifacts/push_cube_ref_franka.mp4
"""

from __future__ import annotations

import argparse

import numpy as np

from sim_gen.core import Checks, Recorder
from sim_gen.core.robots import bind_franka
from sim_gen.tasks.push_cube_ref.scene import PushCubeScene, PushCubeSceneCfg

PUSH_Z = 0.025          # TCP height while pushing (cube half = 0.02)
STANDOFF = 0.07         # pre-push distance behind the cube
DOWN = (0.0, 1.0, 0.0, 0.0)  # hand pointing straight down (180 deg about x, wxyz)


def solve_with_franka(scene, arm, max_nudges: int = 6) -> None:
    """Scripted solution: closed-loop pushing — re-aim behind the cube and nudge it
    toward the goal until it sits inside the radius. Fingers-down orientation keeps
    the contact geometry predictable."""
    arm.set_gripper(0.0, steps=50)  # closed fist = pushing tool
    for _ in range(max_nudges):
        cube = scene.body_pos("cube")
        dist = float(np.linalg.norm(cube[:2] - scene._goal_xy))
        if dist < scene.cfg.goal_radius * 0.5:
            break
        u = (scene._goal_xy - cube[:2]) / max(dist, 1e-9)
        behind = cube[:2] - u * STANDOFF
        # overhead approach -> drop behind the cube -> push a bounded segment
        arm.move_to([*behind, 0.15], quat=DOWN, steps=200)
        arm.move_to([*behind, PUSH_Z], quat=DOWN, steps=150)
        push_len = min(dist + 0.01, 0.15) + STANDOFF - 0.045  # cube rides ~4.5cm ahead
        end = behind + u * push_len
        arm.move_to([*end, PUSH_Z], quat=DOWN, steps=350)
        arm.move_to([*end, 0.15], quat=DOWN, steps=120)   # retract before re-aiming
    scene.settle(200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="push_cube_ref_franka.mp4")
    args = parser.parse_args()

    c = Checks()
    cfg = PushCubeSceneCfg(camera_distance=1.9, camera_lookat=(-0.1, 0.0, 0.2))
    scene, arm = bind_franka(PushCubeScene, cfg)
    rec = Recorder()
    scene.attach_recorder(rec)

    # boot + home
    scene.reset(seed=0)
    arm.go_home()
    scene.step(100)
    c.check("boot: no NaN with robot attached", bool(np.isfinite(scene.data.qpos).all()))
    tcp0 = arm.tcp_pos()
    scene.step(200)
    hold = float(np.linalg.norm(arm.tcp_pos() - tcp0))
    c.check("home hold: TCP steady", hold < 0.01, f"drift {hold:.4f} m")

    # reach: TCP above the cube
    err = arm.move_to(scene.body_pos("cube") + np.array([0, 0, 0.10]))
    c.check("reach: TCP above cube", err < 0.02, f"err {err:.3f} m")

    # scripted robot solutions on 2 instances
    for seed in (0, 1):
        scene.reset(seed=seed)
        arm.go_home()
        solve_with_franka(scene, arm)
        c.check(f"franka solves seed {seed}", scene.success(),
                f"dist {scene._cube_goal_dist():.3f}, score {scene.score():.2f}")

    rec.save(args.video)
    print(f"[binding_smoke] video -> {args.video}", flush=True)
    ok = c.finish()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
