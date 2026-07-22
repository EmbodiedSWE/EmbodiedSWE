"""Null-robot smoke for push_cube_d1: oracle solution + the test-case battery.

Run:  python -m sim_gen.tasks.push_cube_d1.smoke --video artifacts/push_cube_d1.mp4
"""

from __future__ import annotations

import argparse

import numpy as np

from sim_gen.core import Checks, Recorder
from sim_gen.tasks.push_cube_d1.scene import BASE_QUATS, CUBE_HALF, FlipCubeScene

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
HOVER = 0.12


def aligned_target(scene: FlipCubeScene, q_target) -> np.ndarray:
    """Pick the sign of q_target on the same quaternion cover as the cube's current
    orientation, so carry()'s linear interpolation takes the short way around."""
    q = np.asarray(q_target, dtype=np.float64)
    return q if float(np.dot(scene.body_quat("cube"), q)) >= 0 else -q


def reorient_solution(scene: FlipCubeScene, q_target=IDENTITY) -> None:
    """Oracle: lift the cube, rotate it in the air to q_target, lower, settle."""
    p = scene.body_pos("cube")
    q = aligned_target(scene, q_target)
    scene.carry("cube", (p[0], p[1], HOVER), steps=60)
    scene.carry("cube", (p[0], p[1], HOVER), to_quat=q, steps=100)
    scene.carry("cube", (p[0], p[1], CUBE_HALF + 0.002), steps=80)
    scene.settle(250)


# Standard oracle export (pipeline convention): the validator re-runs this under its
# own instrumentation for the measured difficulty label + persistence gates.
null_solution = reorient_solution


def seed_strategy_push(scene: FlipCubeScene, dist: float = 0.25) -> None:
    """The SEED task's plan: translate the cube across the table, orientation
    untouched. Must accomplish nothing here."""
    p = scene.body_pos("cube")
    d = -p[:2] / (np.linalg.norm(p[:2]) + 1e-9)  # toward the table center: stays on
    scene.carry("cube", (*(p[:2] + dist * d), CUBE_HALF + 0.002), steps=150)
    scene.settle(200)


def start_face(scene: FlipCubeScene) -> tuple[int, int]:
    """Which body axis points up: (axis index, sign)."""
    up_body = scene.data.body("cube").xmat.reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])
    i = int(np.argmax(np.abs(up_body)))
    return i, int(np.sign(up_body[i]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="push_cube_d1_smoke.mp4")
    args = parser.parse_args()

    c = Checks()
    scene = FlipCubeScene()
    rec = Recorder()
    scene.attach_recorder(rec)

    # --- physical gates ------------------------------------------------------------
    scene.reset(seed=0)
    p0, u0 = scene.body_pos("cube").copy(), scene.blue_up()
    scene.settle(400)
    drift = float(np.linalg.norm(scene.body_pos("cube") - p0))
    c.check("settle: cube at rest, no drift", drift < 0.01, f"drift {drift:.4f} m")
    c.check("settle: orientation holds", abs(scene.blue_up() - u0) < 0.05,
            f"blue_up {u0:.3f} -> {scene.blue_up():.3f}")
    c.check("settle: no NaN", bool(np.isfinite(scene.data.qpos).all()))

    scene.reset(seed=0)
    scene.step(200)
    h1 = scene.state_hash()
    scene.reset(seed=0)
    scene.step(200)
    c.check("determinism: same seed, same trajectory", h1 == scene.state_hash())

    # tipping behavior depends on the cube's friction and mass — verify they applied
    mu = float(scene.model.geom("cube_g").friction[0])
    mass = float(scene.model.body("cube").mass[0])
    c.check("physics readback: cube friction & mass as configured",
            abs(mu - 1.0) < 1e-9 and abs(mass - 0.05) < 1e-6,
            f"mu {mu}, mass {mass}")

    faces, xys = set(), []
    for seed in range(8):
        scene.reset(seed=seed)
        faces.add(start_face(scene))
        xys.append(scene.body_pos("cube")[:2].copy())
    spread = max(float(np.linalg.norm(a - b)) for a in xys for b in xys)
    c.check("randomization: spawn xy varies", spread > 0.02, f"max spread {spread:.3f} m")
    c.check("randomization: starting face varies", len(faces) >= 3, str(sorted(faces)))
    c.check("randomization: no seed starts solved (blue up)", (2, 1) not in faces)

    # --- success-criterion gates -----------------------------------------------------
    scene.reset(seed=0)
    scene.settle(scene.cfg.episode_steps // 2)
    c.check("null policy: no success", not scene.success())
    c.check("null policy: score ~ 0", scene.score() < 0.05, f"score {scene.score():.3f}")

    # oracle on 3 instances, with rubric monotonicity probes at mid-rotation
    for seed in (0, 1, 2):
        scene.reset(seed=seed)
        s_start = scene.score()
        p = scene.body_pos("cube")
        q_end = aligned_target(scene, IDENTITY)
        q_half = scene.body_quat("cube") + 0.5 * (q_end - scene.body_quat("cube"))
        q_half /= np.linalg.norm(q_half)
        scene.carry("cube", (p[0], p[1], HOVER), steps=60)
        scene.carry("cube", (p[0], p[1], HOVER), to_quat=q_half, steps=60)
        s_mid = scene.score()
        scene.carry("cube", (p[0], p[1], HOVER), to_quat=q_end, steps=60)
        scene.carry("cube", (p[0], p[1], CUBE_HALF + 0.002), steps=80)
        scene.settle(250)
        s_end = scene.score()
        c.check(f"oracle seed {seed}: reaches success()", scene.success(),
                f"blue_up {scene.blue_up():.3f}")
        c.check(f"rubric monotone seed {seed}",
                s_start <= s_mid + 1e-9 and s_mid <= s_end and s_end == 1.0,
                f"{s_start:.2f} -> {s_mid:.2f} -> {s_end:.2f}")

    # --- negative controls -----------------------------------------------------------
    scene.reset(seed=3)  # A: the seed task's strategy — planar transport
    seed_strategy_push(scene)
    c.check("seed strategy (planar slide) must FAIL", not scene.success(),
            f"blue_up {scene.blue_up():.3f}")
    c.check("seed strategy scores ~ 0", scene.score() < 0.05, f"score {scene.score():.3f}")

    scene.reset(seed=4)  # B: reorient, but to an adjacent face (90 deg off)
    reorient_solution(scene, q_target=BASE_QUATS["+x"])
    c.check("adjacent face up must FAIL", not scene.success(),
            f"blue_up {scene.blue_up():.3f}, score {scene.score():.3f}")

    scene.reset(seed=5)  # C: reorient the wrong way round — blue face down
    reorient_solution(scene, q_target=BASE_QUATS["-z"])
    c.check("blue face down must FAIL", not scene.success(),
            f"blue_up {scene.blue_up():.3f}")

    scene.reset(seed=6)  # D: correct orientation but held in the air
    p = scene.body_pos("cube")
    scene.carry("cube", (p[0], p[1], 0.25), to_quat=aligned_target(scene, IDENTITY),
                steps=120)
    c.check("blue up but hovering must FAIL", not scene.success() and scene.score() < 1.0,
            f"z {scene.body_pos('cube')[2]:.3f}, score {scene.score():.3f}")

    # --- tolerance calibration: the physical knee is the cube's 45-deg tipping angle --
    knee = []
    for deg in (10, 30, 60, 80):
        scene.reset(seed=7)
        th = np.radians(deg)
        # balance the cube on an edge at tilt `th` (support height h*(cos+sin)), release
        scene.set_body_pose("cube", (0, 0, CUBE_HALF * (np.cos(th) + np.sin(th)) + 5e-4),
                            (np.cos(th / 2), np.sin(th / 2), 0.0, 0.0))
        scene.settle(600)
        knee.append((deg, scene.success()))
    c.check("calibration: sub-45deg tilts fall back to blue-up",
            all(ok for d, ok in knee if d < 45), str(knee))
    c.check("calibration: past-45deg tilts tip over and fail",
            not any(ok for d, ok in knee if d > 45), str(knee))

    rec.save(args.video)
    print(f"[smoke] video -> {args.video}", flush=True)
    ok = c.finish()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
