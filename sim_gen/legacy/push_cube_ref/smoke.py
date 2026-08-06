"""Null-robot smoke for push_cube_ref: oracle solution + the test-case battery.

Run:  python -m sim_gen.tasks.push_cube_ref.smoke --video artifacts/push_cube_ref.mp4
"""

from __future__ import annotations

import argparse

import numpy as np

from sim_gen.core import Checks, Recorder
from sim_gen.tasks.push_cube_ref.scene import CUBE_HALF, PushCubeScene


def oracle_solution(scene: PushCubeScene, offset: float = 0.0) -> None:
    """Carry the cube to (goal + offset along x), release, settle."""
    target = np.array([*scene._goal_xy, CUBE_HALF + 0.03]) + np.array([offset, 0, 0])
    lift = scene.body_pos("cube") + np.array([0, 0, 0.08])
    scene.carry("cube", lift, steps=60)
    scene.carry("cube", target, steps=140)
    scene.settle(200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="push_cube_ref_smoke.mp4")
    args = parser.parse_args()

    c = Checks()
    scene = PushCubeScene()
    rec = Recorder()
    scene.attach_recorder(rec)

    # --- physical gates ------------------------------------------------------------
    scene.reset(seed=0)
    p0 = scene.body_pos("cube").copy()
    scene.settle(400)
    drift = float(np.linalg.norm(scene.body_pos("cube") - p0))
    c.check("settle: cube at rest, no drift", drift < 0.01, f"drift {drift:.4f} m")
    c.check("settle: no NaN", bool(np.isfinite(scene.data.qpos).all()))

    scene.reset(seed=0)
    scene.step(200)
    h1 = scene.state_hash()
    scene.reset(seed=0)
    scene.step(200)
    c.check("determinism: same seed, same trajectory", h1 == scene.state_hash())

    scene.reset(seed=0)
    a = scene.body_pos("cube").copy()
    scene.reset(seed=1)
    b = scene.body_pos("cube").copy()
    c.check("randomization is real across seeds", float(np.linalg.norm(a - b)) > 0.005)

    # --- success-criterion gates ------------------------------------------------------
    scene.reset(seed=0)
    scene.settle(scene.cfg.episode_steps // 2)
    c.check("null policy: no success", not scene.success())
    c.check("null policy: score ~ 0", scene.score() < 0.05, f"score {scene.score():.3f}")

    # oracle on 3 instances, with rubric monotonicity probes
    for seed in (0, 1, 2):
        scene.reset(seed=seed)
        s_start = scene.score()
        mid = np.array([*((scene.body_pos("cube")[:2] + scene._goal_xy) / 2), CUBE_HALF + 0.03])
        scene.carry("cube", scene.body_pos("cube") + np.array([0, 0, 0.08]), steps=60)
        scene.carry("cube", mid, steps=80)
        s_mid = scene.score()
        scene.carry("cube", np.array([*scene._goal_xy, CUBE_HALF + 0.03]), steps=80)
        scene.settle(200)
        s_end = scene.score()
        c.check(f"oracle seed {seed}: reaches success()", scene.success(),
                f"dist {scene._cube_goal_dist():.3f}")
        c.check(f"rubric monotone seed {seed}", s_start <= s_mid <= s_end and s_end == 1.0,
                f"{s_start:.2f} -> {s_mid:.2f} -> {s_end:.2f}")

    # --- negative controls ---------------------------------------------------------
    scene.reset(seed=3)
    oracle_solution(scene, offset=scene.cfg.goal_radius * 1.5)  # near-miss outside radius
    c.check("near-miss outside radius must FAIL", not scene.success(),
            f"dist {scene._cube_goal_dist():.3f} vs r={scene.cfg.goal_radius}")
    c.check("near-miss score pinned below 1", scene.score() < 1.0)

    scene.reset(seed=4)
    scene.carry("cube", np.array([*scene._goal_xy, 0.25]), steps=100)  # hover over goal
    c.check("hover over goal (in air) must FAIL", not scene.success())

    # --- tolerance calibration sweep --------------------------------------------------
    knee = []
    for off_frac in (0.0, 0.5, 0.9, 1.3):
        scene.reset(seed=5)
        oracle_solution(scene, offset=scene.cfg.goal_radius * off_frac)
        knee.append((off_frac, scene.success()))
    c.check("sweep: inside-radius offsets pass", all(ok for f, ok in knee if f < 1.0),
            str(knee))
    c.check("sweep: outside-radius offset fails", not dict(knee)[1.3], str(knee))

    rec.save(args.video)
    print(f"[smoke] video -> {args.video}", flush=True)
    ok = c.finish()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
