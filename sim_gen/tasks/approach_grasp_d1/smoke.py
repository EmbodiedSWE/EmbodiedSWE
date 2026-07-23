"""Null-robot smoke for approach_grasp_d1: teleport-oracle solution + test battery.

Run:  MUJOCO_GL=egl python -m sim_gen.tasks.approach_grasp_d1.smoke --video sim_gen/artifacts/approach_grasp_d1_smoke.mp4
"""

from __future__ import annotations

import argparse

import numpy as np

from sim_gen.core import Checks, Recorder
from sim_gen.tasks.approach_grasp_d1.scene import StackTowerScene

CLEAR_Z = 0.28  # transit height: clears the finished tower (top ~0.22) with margin


def _pick_place(scene: StackTowerScene, body: str, target_xy, rest_z: float,
                settle_steps: int = 150) -> None:
    """Lift a block, traverse at height, set it down 3 mm above rest_z, release, settle."""
    p = scene.body_pos(body)
    scene.carry(body, np.array([p[0], p[1], CLEAR_Z]), to_quat=(1.0, 0.0, 0.0, 0.0), steps=70)
    scene.carry(body, np.array([target_xy[0], target_xy[1], CLEAR_Z]), steps=90)
    scene.carry(body, np.array([target_xy[0], target_xy[1], rest_z + 0.003]), steps=80)
    scene.settle(settle_steps)


def _build_tower(scene: StackTowerScene, top_offset_x: float = 0.0) -> None:
    """Bottom-up tower on the pedestal; the small block may be offset (calibration probe)."""
    c = scene.cfg
    _pick_place(scene, "block_l", scene._ped_xy, scene._ped_top + c.half_l)
    lp = scene.body_pos("block_l")
    _pick_place(scene, "block_m", lp[:2], lp[2] + c.half_l + c.half_m)
    mp = scene.body_pos("block_m")
    _pick_place(scene, "block_s", mp[:2] + np.array([top_offset_x, 0.0]),
                mp[2] + c.half_m + c.half_s)
    scene.settle(300)


def oracle_solution(scene: StackTowerScene) -> None:
    """The intended solution: place large on the pedestal, stack medium, stack small."""
    _build_tower(scene, top_offset_x=0.0)


def _grasp_lift_hold(scene: StackTowerScene, body: str, hold_pos=None) -> None:
    """The SEED's strategy: approach a block, 'grasp' it, lift it, hold it stably aloft."""
    p = scene.body_pos(body)
    target = np.array([p[0], p[1], p[2] + 0.15]) if hold_pos is None else np.asarray(hold_pos, dtype=float)
    scene.carry(body, target, steps=80)
    scene.carry(body, target, steps=300)  # sustained stable hold in the air


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="approach_grasp_d1_smoke.mp4")
    args = parser.parse_args()

    c = Checks()
    scene = StackTowerScene()
    rec = Recorder()
    scene.attach_recorder(rec)
    cfg = scene.cfg

    # --- physical gates ------------------------------------------------------------
    scene.reset(seed=0)
    p0 = {b: scene.body_pos(b) for b in ("block_l", "block_m", "block_s")}
    scene.settle(400)
    drift = max(float(np.linalg.norm(scene.body_pos(b) - p0[b])) for b in p0)
    c.check("settle: blocks at rest, no drift", drift < 0.01, f"max drift {drift:.4f} m")
    c.check("settle: no NaN", bool(np.isfinite(scene.data.qpos).all()))

    scene.reset(seed=0)
    scene.step(200)
    h1 = scene.state_hash()
    scene.reset(seed=0)
    scene.step(200)
    c.check("determinism: same seed, same trajectory", h1 == scene.state_hash())

    scene.reset(seed=0)
    a = np.concatenate([scene.body_pos("block_l"), scene.body_pos("block_s"), scene._ped_xy])
    scene.reset(seed=1)
    b = np.concatenate([scene.body_pos("block_l"), scene.body_pos("block_s"), scene._ped_xy])
    c.check("randomization is real (blocks + pedestal)", float(np.linalg.norm(a - b)) > 0.005)

    masses = [float(scene.model.body(n).mass[0]) for n in ("block_l", "block_m", "block_s")]
    c.check("mass readback: cfg applied, ordered l>m>s",
            abs(masses[0] - cfg.mass_l) < 1e-9 and abs(masses[1] - cfg.mass_m) < 1e-9
            and abs(masses[2] - cfg.mass_s) < 1e-9 and masses[0] > masses[1] > masses[2],
            f"masses {masses}")
    fric = float(scene.model.geom("table").friction[0])
    c.check("friction readback: table sliding friction applied", abs(fric - 1.0) < 1e-9,
            f"friction {fric}")

    # --- success-criterion gates -----------------------------------------------------
    scene.reset(seed=0)
    scene.settle(cfg.episode_steps // 2)
    c.check("null policy: no success", not scene.success())
    c.check("null policy: score ~ 0", scene.score() < 0.05, f"score {scene.score():.3f}")

    # oracle on 3 instances, with rubric monotonicity across stage boundaries
    for seed in (0, 1, 2):
        scene.reset(seed=seed)
        s0 = scene.score()
        _pick_place(scene, "block_l", scene._ped_xy, scene._ped_top + cfg.half_l)
        s1 = scene.score()
        lp = scene.body_pos("block_l")
        _pick_place(scene, "block_m", lp[:2], lp[2] + cfg.half_l + cfg.half_m)
        s2 = scene.score()
        mp = scene.body_pos("block_m")
        _pick_place(scene, "block_s", mp[:2], mp[2] + cfg.half_m + cfg.half_s)
        scene.settle(300)
        s3 = scene.score()
        c.check(f"oracle seed {seed}: reaches success()", scene.success())
        c.check(f"rubric monotone seed {seed}", s0 < s1 < s2 < s3 == 1.0,
                f"{s0:.2f} -> {s1:.2f} -> {s2:.2f} -> {s3:.2f}")

    # exported oracle + success stability over an extended hold
    scene.reset(seed=0)
    oracle_solution(scene)
    ok0 = scene.success()
    scene.step(300)
    c.check("success stable over +300-step hold", ok0 and scene.success())

    # --- negative controls ---------------------------------------------------------
    # (A) the seed's own strategy: grasp a block, lift it, hold it stably in the air
    scene.reset(seed=3)
    _grasp_lift_hold(scene, "block_s")
    c.check("seed strategy (grasp-lift-hold) must FAIL", not scene.success())
    c.check("seed strategy earns ~0 score", scene.score() < 0.05, f"score {scene.score():.3f}")

    # (A') seed strategy aimed at the goal: hold the LARGE block right above the pedestal
    scene.reset(seed=3)
    _grasp_lift_hold(scene, "block_l", hold_pos=[*scene._ped_xy, 0.22])
    c.check("hover above pedestal must FAIL", not scene.success())
    c.check("hover score stays low (shaping only)", scene.score() <= 0.25,
            f"score {scene.score():.3f}")

    # (B) wrong-order: small -> medium -> large, same pedestal, same care
    scene.reset(seed=4)
    _pick_place(scene, "block_s", scene._ped_xy, scene._ped_top + cfg.half_s)
    sp = scene.body_pos("block_s")
    _pick_place(scene, "block_m", sp[:2], sp[2] + cfg.half_s + cfg.half_m)
    mp = scene.body_pos("block_m")
    _pick_place(scene, "block_l", mp[:2], mp[2] + cfg.half_m + cfg.half_l)
    scene.settle(300)
    c.check("wrong-order tower (small-first) must FAIL", not scene.success())
    c.check("wrong-order score pinned below 1", scene.score() < 1.0, f"score {scene.score():.3f}")

    # (C) correct order but on the table, off the pedestal
    scene.reset(seed=4)
    spot = np.array([-0.48, -0.38])
    _pick_place(scene, "block_l", spot, cfg.half_l)
    lp = scene.body_pos("block_l")
    _pick_place(scene, "block_m", lp[:2], lp[2] + cfg.half_l + cfg.half_m)
    mp = scene.body_pos("block_m")
    _pick_place(scene, "block_s", mp[:2], mp[2] + cfg.half_m + cfg.half_s)
    scene.settle(300)
    c.check("correct-order tower OFF the pedestal must FAIL", not scene.success())

    # --- latch persistence -----------------------------------------------------------
    scene.reset(seed=6)
    _pick_place(scene, "block_l", scene._ped_xy, scene._ped_top + cfg.half_l)
    s_placed = scene.score()
    c.check("large-block placement earns latched credit", s_placed >= 0.40,
            f"score {s_placed:.3f}")
    snap = scene.get_state()
    scene.step(150)
    scene.set_state(snap)
    c.check("latches survive state save/restore", abs(scene.score() - s_placed) < 1e-9,
            f"{scene.score():.3f} vs {s_placed:.3f}")
    lp = scene.body_pos("block_l")
    pm = scene.body_pos("block_m")
    scene.carry("block_m", np.array([pm[0], pm[1], CLEAR_Z]), steps=70)
    scene.carry("block_m", np.array([lp[0], lp[1], CLEAR_Z]), steps=90)
    c.check("credit does not evaporate under continued correct behavior",
            scene.score() >= s_placed, f"{scene.score():.3f} vs {s_placed:.3f}")

    # --- tolerance calibration sweep ---------------------------------------------------
    knee = []
    for frac in (0.0, 0.4, 0.8, 1.6):
        scene.reset(seed=5)
        _build_tower(scene, top_offset_x=frac * cfg.align_tol)
        knee.append((frac, scene.success()))
    c.check("sweep: within-tolerance offsets succeed", all(ok for f, ok in knee if f < 1.0),
            str(knee))
    c.check("sweep: beyond-tolerance offset fails", not dict(knee)[1.6], str(knee))

    rec.save(args.video)
    print(f"[smoke] video -> {args.video}", flush=True)
    ok = c.finish()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
