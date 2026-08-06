"""Franka binding smoke for push_cube_d1: boot, home hold, and reach checks.

A scripted robot die-flip is not included yet (edge-tipping with the closed gripper
needs its own calibration pass); this smoke validates the binding layer — the robot
boots in the scene, holds home, and can reach the die across instances.

Run:  python -m sim_gen.tasks.push_cube_d1.binding_smoke \
          --video sim_gen/artifacts/push_cube_d1_franka.mp4
"""

from __future__ import annotations

import argparse

import numpy as np

from sim_gen.core import Checks, Recorder
from sim_gen.core.robots import bind_franka
from sim_gen.tasks.push_cube_d1.scene import FlipCubeScene, FlipCubeSceneCfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="push_cube_d1_franka.mp4")
    args = parser.parse_args()

    c = Checks()
    cfg = FlipCubeSceneCfg(camera_distance=1.9, camera_lookat=(-0.1, 0.0, 0.2))
    scene, arm = bind_franka(FlipCubeScene, cfg)
    rec = Recorder()
    scene.attach_recorder(rec)

    scene.reset(seed=0)
    arm.go_home()
    scene.step(100)
    c.check("boot: no NaN with robot attached", bool(np.isfinite(scene.data.qpos).all()))
    tcp0 = arm.tcp_pos()
    scene.step(200)
    c.check("home hold: TCP steady",
            float(np.linalg.norm(arm.tcp_pos() - tcp0)) < 0.01)

    # reach the die across 3 instances (above + beside, the pre-tip poses)
    for seed in (0, 1, 2):
        scene.reset(seed=seed)
        arm.go_home()
        die = scene.body_pos("cube")
        e_above = arm.move_to(die + np.array([0, 0, 0.10]))
        e_side = arm.move_to(die + np.array([-0.06, 0, 0.03]))
        c.check(f"reach seed {seed}: above + beside the die",
                e_above < 0.03 and e_side < 0.03,
                f"above {e_above:.3f} m, beside {e_side:.3f} m")
        c.check(f"reach seed {seed}: die undisturbed by reach",
                not scene.success())

    rec.save(args.video)
    print(f"[binding_smoke] video -> {args.video}", flush=True)
    ok = c.finish()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
