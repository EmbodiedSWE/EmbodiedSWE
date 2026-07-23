"""Null-robot smoke for pull_cube_tool_d1: oracle solution + the test-case battery.

Run:  python -m sim_gen.tasks.pull_cube_tool_d1.smoke --video sim_gen/artifacts/pull_cube_tool_d1_smoke.mp4
"""

from __future__ import annotations

import argparse

import numpy as np

from sim_gen.core import Checks, Recorder
from sim_gen.tasks.pull_cube_tool_d1.scene import TrenchBridgeScene


def oracle_solution(scene: TrenchBridgeScene, bin_offset=(0.0, 0.0), skip_return: bool = False,
                    stage_scores: list | None = None) -> None:
    """The intended 4-stage plan: bridge → cross on the bridge → deposit → return plank."""
    c = scene.cfg

    def mark():
        if stage_scores is not None:
            stage_scores.append(scene.score())

    mark()
    # ① bridge: carry the plank over the trench at the cube's y-lane, lay it flat across
    y_lane = float(np.clip(scene.body_pos("cube")[1], -0.30, 0.30))
    p = scene.body_pos("plank")
    scene.carry("plank", (p[0], p[1], 0.08), steps=50)
    scene.carry("plank", (0.0, y_lane, 0.03), to_quat=(1.0, 0.0, 0.0, 0.0), steps=120)
    scene.carry("plank", (0.0, y_lane, c.plank_half[2] + 0.001), steps=50)
    scene.settle(100)
    mark()
    # ② cross: set the cube down on the bridge's far end and slide it to mid-span
    q = scene.body_pos("cube")
    z_ride = 2 * c.plank_half[2] + c.cube_half + 0.001
    scene.carry("cube", (q[0], q[1], 0.06), steps=40)
    scene.carry("cube", (c.gap_half + 0.03, y_lane, 0.06), steps=100)
    scene.carry("cube", (c.gap_half + 0.03, y_lane, z_ride), steps=40)
    scene.carry("cube", (0.0, y_lane, z_ride), steps=60)
    scene.settle(80)  # rests mid-span on the bridge — the crossing latch fires here
    mark()
    # ③ deposit: lift off the bridge, over the bin walls, drop in, settle
    bx, by = scene._bin_xy + np.asarray(bin_offset, dtype=np.float64)
    scene.carry("cube", (0.0, y_lane, 0.10), steps=40)
    scene.carry("cube", (bx, by, 0.10), steps=100)
    scene.carry("cube", (bx, by, 0.045), steps=40)
    scene.settle(150)
    mark()
    if skip_return:
        return
    # ④ return: bring the plank home to the rest pad
    scene.carry("plank", (0.0, y_lane, 0.12), steps=50)
    scene.carry("plank", (*scene._rest_xy, 0.12), steps=110)
    scene.carry("plank", (*scene._rest_xy, c.plank_half[2] + 0.001), steps=60)
    scene.settle(250)
    mark()


def seed_strategy_drag(scene: TrenchBridgeScene) -> None:
    """The SEED task's plan: hook-and-drag the cube along the surface toward the base.
    Dragging means the object slides on its support — over the open trench there is
    none, so the cube is released where the surface ends and falls into the pit."""
    y = float(scene.body_pos("cube")[1])
    z = scene.cfg.cube_half + 0.001
    scene.carry("cube", (scene.cfg.gap_half + 0.03, y, z), steps=120)  # drag to the trench edge
    scene.carry("cube", (0.0, y, z), steps=50)                          # drag onto... nothing
    scene.settle(300)


def park_plank_on_pad(scene: TrenchBridgeScene) -> None:
    scene.carry("plank", (*scene.body_pos("plank")[:2], 0.08), steps=40)
    scene.carry("plank", (*scene._rest_xy, 0.08), steps=100)
    scene.carry("plank", (*scene._rest_xy, scene.cfg.plank_half[2] + 0.001), steps=40)
    scene.settle(120)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="pull_cube_tool_d1_smoke.mp4")
    args = parser.parse_args()

    c = Checks()
    scene = TrenchBridgeScene()
    cfg = scene.cfg
    rec = Recorder()
    scene.attach_recorder(rec)

    # --- physical gates ------------------------------------------------------------
    scene.reset(seed=0)
    p_cube, p_plank = scene.body_pos("cube").copy(), scene.body_pos("plank").copy()
    scene.settle(400)
    drift = max(float(np.linalg.norm(scene.body_pos("cube") - p_cube)),
                float(np.linalg.norm(scene.body_pos("plank") - p_plank)))
    c.check("settle: cube+plank at rest, no drift", drift < 0.01, f"max drift {drift:.4f} m")
    c.check("settle: no NaN", bool(np.isfinite(scene.data.qpos).all()))

    # the mechanism depends on plank geometry and the cube/plank mass ratio — verify applied
    plank_len = 2 * float(scene.model.geom("plank_g").size[0])
    need = 2 * (cfg.gap_half + cfg.span_overlap)
    c.check("physics readback: plank out-spans the trench", plank_len > need,
            f"plank {plank_len:.3f} m vs required {need:.3f} m")
    m_cube = float(scene.model.body("cube").mass[0])
    m_plank = float(scene.model.body("plank").mass[0])
    c.check("physics readback: masses as configured (cube heavier than plank end can hold)",
            abs(m_cube - cfg.cube_mass) < 1e-6 and abs(m_plank - cfg.plank_mass) < 1e-6,
            f"cube {m_cube} kg, plank {m_plank} kg")

    scene.reset(seed=0)
    scene.step(200)
    h1 = scene.state_hash()
    scene.reset(seed=0)
    scene.step(200)
    c.check("determinism: same seed, same trajectory", h1 == scene.state_hash())

    scene.reset(seed=0)
    a = np.concatenate([scene.body_pos("cube"), scene.body_pos("plank"),
                        scene._bin_xy, scene._rest_xy])
    scene.reset(seed=1)
    b = np.concatenate([scene.body_pos("cube"), scene.body_pos("plank"),
                        scene._bin_xy, scene._rest_xy])
    c.check("randomization is real across seeds (cube/plank/bin/pad)",
            float(np.linalg.norm(a - b)) > 0.01, f"delta {float(np.linalg.norm(a - b)):.3f}")

    # --- success-criterion gates -----------------------------------------------------
    scene.reset(seed=0)
    scene.settle(cfg.episode_steps // 2)
    c.check("null policy: no success", not scene.success())
    c.check("null policy: score ~ 0", scene.score() < 0.05, f"score {scene.score():.3f}")

    # oracle on 3 instances with rubric probes at every stage boundary
    for seed in (0, 1, 2):
        scene.reset(seed=seed)
        ss: list[float] = []
        oracle_solution(scene, stage_scores=ss)
        c.check(f"oracle seed {seed}: reaches success()", scene.success(),
                f"crossed={scene._crossed} in_bin={scene._cube_in_bin()} rest={scene._plank_in_rest()}")
        mono = all(ss[i] <= ss[i + 1] + 1e-9 for i in range(len(ss) - 1))
        c.check(f"rubric monotone over stages seed {seed}",
                mono and ss[0] < 0.05 and ss[-1] == 1.0,
                " -> ".join(f"{s:.2f}" for s in ss))

    hold_ok = True
    for _ in range(10):
        scene.step(30)
        hold_ok = hold_ok and scene.success()
    c.check("success is stable over a 300-step hold", hold_ok)

    # --- negative controls -----------------------------------------------------------
    scene.reset(seed=10)  # A: the seed's own strategy — surface drag, no bridge
    seed_strategy_drag(scene)
    c.check("seed strategy (drag across) must FAIL — cube falls in the pit",
            not scene.success() and scene.body_pos("cube")[2] < -0.05,
            f"cube z {scene.body_pos('cube')[2]:.3f}")
    c.check("seed strategy scores ~ 0", scene.score() < 0.05, f"score {scene.score():.3f}")

    scene.reset(seed=11)  # B: fly the cube into the bin without ever bridging
    q = scene.body_pos("cube")
    scene.carry("cube", (q[0], q[1], 0.12), steps=50)
    scene.carry("cube", (*scene._bin_xy, 0.12), steps=120)
    scene.carry("cube", (*scene._bin_xy, 0.045), steps=40)
    scene.settle(200)
    c.check("fly-over into the bin (no bridge) must FAIL despite cube in bin",
            scene._cube_in_bin() and not scene.success(),
            f"in_bin={scene._cube_in_bin()} crossed={scene._crossed}")
    c.check("fly-over scores ~ 0 (delivery gated on crossing)", scene.score() < 0.05,
            f"score {scene.score():.3f}")

    scene.reset(seed=12)  # C: wrong order — park the plank on the pad FIRST, then cross
    park_plank_on_pad(scene)
    seed_strategy_drag(scene)
    c.check("wrong order (return plank before crossing) must FAIL",
            not scene.success() and scene.score() < 0.05,
            f"plank_in_rest={scene._plank_in_rest()} score {scene.score():.3f}")

    scene.reset(seed=13)  # D: full delivery but the bridge left in place
    oracle_solution(scene, skip_return=True)
    c.check("bridge left in place must FAIL", not scene.success(),
            f"plank_in_rest={scene._plank_in_rest()}")
    c.check("bridge left in place earns partial credit only",
            0.5 < scene.score() < 1.0, f"score {scene.score():.3f}")

    scene.reset(seed=14)  # E: cantilevered "bridge" — near end short of the trench edge
    y_lane = float(np.clip(scene.body_pos("cube")[1], -0.30, 0.30))
    p = scene.body_pos("plank")
    scene.carry("plank", (p[0], p[1], 0.08), steps=40)
    scene.carry("plank", (0.085, y_lane, 0.03), to_quat=(1.0, 0.0, 0.0, 0.0), steps=110)
    scene.carry("plank", (0.085, y_lane, cfg.plank_half[2] + 0.001), steps=40)
    scene.settle(80)
    q = scene.body_pos("cube")
    scene.carry("cube", (q[0], q[1], 0.06), steps=40)
    scene.carry("cube", (-0.015, y_lane, 0.06), steps=90)
    scene.carry("cube", (-0.015, y_lane, 2 * cfg.plank_half[2] + cfg.cube_half + 0.002), steps=40)
    scene.settle(300)  # plank end is unsupported: it tips under the cube
    c.check("cantilever (not spanning) must not latch a crossing and must FAIL",
            not scene._crossed and not scene.success(),
            f"cube z {scene.body_pos('cube')[2]:.3f} bridged={scene._bridged}")

    # --- tolerance calibration: knee at the bin wall ----------------------------------
    knee = []
    for i, frac in enumerate((0.0, 0.6, 3.0)):
        scene.reset(seed=20 + i)
        oracle_solution(scene, bin_offset=(0.0, frac * cfg.bin_tol))
        knee.append((frac, scene.success()))
    c.check("calibration: in-tolerance bin offsets succeed",
            all(ok for f, ok in knee if f < 1.0), str(knee))
    c.check("calibration: offset past the bin wall fails", not dict(knee)[3.0], str(knee))

    rec.save(args.video)
    print(f"[smoke] video -> {args.video}", flush=True)
    ok = c.finish()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
