"""Smoke / rubric-REJECTION battery for DieRollScene (sim_gen task
`slide_block_to_target_i135`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — torque-driven quarter-rolls that end blue-up on
the disc — is the acceptance evidence that the rubric ACCEPTS a correct outcome).
Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome
and asserts the rubric REJECTS it; no probe in this battery ever reaches success(),
and a final audit check asserts exactly that.

  1-2. settle/no-NaN     — reset layout settles finite: die at rest on the floor,
                           disc flat at its slot, all still, blue NOT up, score ~0;
  3-4. randomization     — READBACK over 8 seeded resets: the sampled up-face varies
                           and is NEVER blue; die xy, disc xy and spawn yaw (blue-
                           normal azimuth) spreads are real;
  5.  null policy        — 240 idle steps -> score ~0, no success;
  6.  seed strategy      — the end state the SEED's plan produces here (slide the
                           block onto the target): the die, spawn orientation
                           PRESERVED, settled centred ON the disc -> in-zone reads
                           True but NOT success, score <= 0.45 (position alone
                           cannot satisfy an orientation goal);
  7.  blue-down in zone  — die settled on the disc blue straight DOWN (the hardest
                           wrong face: two rolls short) -> NOT success, score <= 0.45;
  8.  blue-side in zone  — die settled on the disc blue SIDEWAYS (one roll short)
                           -> NOT success, score <= 0.45;
  9.  tolerance gate     — die in the zone at rest height, quiet, blue tilted 25 deg
                           off vertical (just past the 20 deg tolerance), judged
                           live -> in-zone AND settled read True, yet NOT success
                           (then relocated before it can topple flat on its own);
  10. latched credit     — after probe 6's zone/approach credit, the die teleported
                           far away again: latched score UNCHANGED, still no success;
  11. position near-miss — die settled blue-UP but centred just outside the zone
                           radius -> NOT success, score <= 0.61;
  12. airborne           — die blue-up in free fall directly over the disc, judged
                           mid-air -> NOT success (resting-height gate; then
                           relocated before it can land);
  13. rejection audit    — success() was never True at ANY judged point;
  14. final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes some driver versions and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

FACE_NAMES = scene_mod.FACE_NAMES

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

_C = math.sqrt(0.5)
Q_BLUE_UP = (1.0, 0.0, 0.0, 0.0)
Q_BLUE_DOWN = (0.0, 1.0, 0.0, 0.0)
Q_BLUE_SIDE = (_C, 0.0, -_C, 0.0)  # +x (red) up, blue sideways
Q_WHITE_UP = (0.0, 1.0, 0.0, 0.0)  # same as blue-down: white (-z) up
_H25 = math.radians(25.0) / 2.0
Q_BLUE_TILT25 = (math.cos(_H25), 0.0, math.sin(_H25), 0.0)  # blue 25 deg off vertical


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.die_roll")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    s = c.size

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -1.00, 0.85)) + o),
                                tuple(np.array((0.32, 0.00, 0.05)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action)
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    ever_success = [False]

    def judge() -> tuple[float, bool]:
        sc, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return sc, ok

    def die_p() -> torch.Tensor:
        return (scene.die.data.root_pos_w - scene.env_origins)[0]

    def disc_p() -> torch.Tensor:
        return (scene.disc.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        p = die_p()
        b = scene.blue_normal_w()[0]
        sc, ok = judge()
        print(f"[smoke] {tag:16s} | die=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):.3f}) up_face={FACE_NAMES[int(scene.up_face_idx()[0])]} "
              f"blue_z={float(b[2]):+.3f} d_disc={float(scene.dist_to_disc()[0]):.3f} "
              f"in_zone={bool(scene.in_zone()[0])} blue_up={bool(scene.blue_up()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"blue_ever={bool(scene._blue_ever[0])} app={float(scene._app_max[0]):.3f} "
              f"zone_ever={bool(scene._zone_ever[0])} score={sc:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_die(x: float, y: float, z: float, quat=None) -> None:
        """Probe constructor: teleport the die to a world pose (orientation given, or
        the CURRENT one preserved when quat is None), zero velocity."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = float(x), float(y), float(z)
        if quat is None:
            st[:, 3:7] = scene.die.data.root_quat_w[0]
        else:
            st[:, 3:7] = torch.tensor(quat, device=device)
        st[:, 0:3] += scene.env_origins
        scene.die.write_root_state_to_sim(st, all_ids)

    def still() -> bool:
        return (float(scene.die.data.root_lin_vel_w[0].norm()) < c.settle_speed
                and float(scene.die.data.root_ang_vel_w[0].norm()) < c.settle_omega)

    def finite_all() -> bool:
        return bool(torch.isfinite(scene.die.data.root_state_w).all()
                    and torch.isfinite(scene.disc.data.root_state_w).all())

    FAR = (0.10, -0.30)  # relocation spot: > approach_d0 from every disc slot draw

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    report("reset")
    step(90)
    report("show")
    p = die_p()
    check("settle: states finite, die at rest on the floor (z ~ s/2), disc flat at "
          "its slot, all still, blue NOT up",
          finite_all() and abs(float(p[2]) - s / 2) < 0.010
          and abs(float(disc_p()[2]) - c.disc_h / 2) < 0.008 and still()
          and not bool(scene.blue_up()[0]) and int(scene.up_face_idx()[0]) != 4)
    sc, ok = judge()
    check("settle: score ~0 at reset (<= 0.02, the spawn gap clears the approach "
          "ramp), no success", sc <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(5)
        p = die_p()
        d = disc_p()
        b = scene.blue_normal_w()[0]
        reads.append((float(p[0]), float(p[1]), float(d[0]), float(d[1]),
                      int(scene.up_face_idx()[0]), float(b[0]), float(b[1]),
                      float(b[2])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (die_x, die_y, disc_x, disc_y, up_face, "
          f"blue_nx, blue_ny, blue_nz):\n{arr}", flush=True)
    faces = arr[:, 4].astype(int)
    check("randomization: sampled up-face VARIES across 8 seeded resets (>= 2 "
          "distinct) and is NEVER blue (readback)",
          len(set(faces.tolist())) >= 2 and all(f != 4 for f in faces))
    die_spread = float((arr[:, 0:2].max(axis=0) - arr[:, 0:2].min(axis=0)).max())
    disc_spread = float((arr[:, 2:4].max(axis=0) - arr[:, 2:4].min(axis=0)).max())
    # yaw evidence: azimuth of the blue normal when blue is sideways follows the
    # free spawn yaw — at least two azimuths must differ by > 10 deg.
    az = [r[5:7] for r in reads if abs(r[7]) < 0.5]
    min_dot = 2.0
    for i in range(len(az)):
        for j in range(i + 1, len(az)):
            u = np.array(az[i]) / (np.linalg.norm(az[i]) + 1e-9)
            v = np.array(az[j]) / (np.linalg.norm(az[j]) + 1e-9)
            min_dot = min(min_dot, float(u @ v))
    check("randomization: die xy jitter (> 8 mm), disc xy jitter (> 8 mm) and spawn "
          "yaw spread (blue-normal azimuths differ > 10 deg) are real (readback)",
          die_spread > 0.008 and disc_spread > 0.008
          and len(az) >= 2 and min_dot < math.cos(math.radians(10.0)))

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    sc, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          sc <= 0.02 and not ok)

    # =========================== 6. seed strategy: slide onto the target ====================
    # The seed's entire plan — push the block flat across the floor onto the target
    # marker — executed here as its END STATE: the die, spawn orientation PRESERVED
    # (a planar slide changes no orientation), settled centred ON the disc.
    env.reset(seed=41)
    step(30)
    d = disc_p()
    place_die(float(d[0]), float(d[1]), s / 2 + c.disc_h + 0.002, quat=None)
    step(150)
    report("seed-strategy")
    sc6, ok = judge()
    check("seed strategy: die slid onto the disc with its SPAWN orientation — in-zone "
          "reads True, yet NOT success and score <= 0.45 (position alone cannot "
          "satisfy an orientation goal)",
          bool(scene.in_zone()[0]) and still() and not bool(scene.blue_up()[0])
          and not ok and sc6 <= 0.45)

    # =========================== 7-8. wrong face in the zone (constructed) ==================
    env.reset(seed=51)
    step(30)
    d = disc_p()
    place_die(float(d[0]), float(d[1]), s / 2 + c.disc_h + 0.002, quat=Q_BLUE_DOWN)
    step(150)
    report("blue-down-zone")
    sc, ok = judge()
    check("blue-down in zone: die settled on the disc blue straight DOWN (two rolls "
          "short) — NOT success, score <= 0.45",
          bool(scene.in_zone()[0]) and still()
          and float(scene.blue_normal_w()[0, 2]) < -0.9 and not ok and sc <= 0.45)

    env.reset(seed=61)
    step(30)
    d = disc_p()
    place_die(float(d[0]), float(d[1]), s / 2 + c.disc_h + 0.002, quat=Q_BLUE_SIDE)
    step(150)
    report("blue-side-zone")
    sc, ok = judge()
    check("blue-side in zone: die settled on the disc blue SIDEWAYS (one roll short) "
          "— NOT success, score <= 0.45",
          bool(scene.in_zone()[0]) and still()
          and abs(float(scene.blue_normal_w()[0, 2])) < 0.3 and not ok and sc <= 0.45)

    # =========================== 9. orientation tolerance gate ==============================
    # Blue 25 deg off vertical (just past the 20 deg tolerance), zero velocity, in
    # the zone at rest height: in-zone AND settled read True, the tolerance gate
    # ALONE must reject. Judged on the written state with NO intervening step (a
    # cube cannot rest at 25 deg — one gravity step already exceeds the settle
    # gate), then relocated before it can topple flat on its own.
    env.reset(seed=71)
    step(30)
    d = disc_p()
    place_die(float(d[0]), float(d[1]), 0.058, quat=Q_BLUE_TILT25)
    report("tilt-25deg")
    sc, ok = judge()
    tol_ok = (bool(scene.in_zone()[0]) and bool(scene.settled()[0])
              and not bool(scene.blue_up()[0]) and not ok
              and float(scene.blue_normal_w()[0, 2]) > 0.85)
    place_die(FAR[0], FAR[1], s / 2 + 0.003, quat=Q_WHITE_UP)  # relocate pre-topple
    step(60)
    check("tolerance gate: die in the zone, quiet, blue 25 deg off vertical — "
          "in-zone and settled read True, yet NOT success (the 20 deg tolerance is "
          "load-bearing)", tol_ok)

    # =========================== 10. latched credit survives regression =====================
    env.reset(seed=81)
    step(30)
    d = disc_p()
    place_die(float(d[0]), float(d[1]), s / 2 + c.disc_h + 0.002, quat=Q_WHITE_UP)
    step(120)
    report("zone-credit")
    sc10a, ok = judge()
    place_die(FAR[0], FAR[1], s / 2 + 0.003, quat=Q_WHITE_UP)
    step(90)
    report("regressed")
    sc10b, ok = judge()
    check("latched credit: teleporting the die far away after earning zone/approach "
          f"credit leaves the latched score unchanged ({sc10a:.3f} -> {sc10b:.3f}), "
          "still no success",
          sc10a >= 0.35 and abs(sc10b - sc10a) < 1e-3 and not ok
          and float(scene.dist_to_disc()[0]) > c.zone_r + 0.05)

    # =========================== 11. position near-miss (blue up) ===========================
    env.reset(seed=91)
    step(30)
    d = disc_p()
    place_die(float(d[0]) + c.zone_r + 0.025, float(d[1]),
              s / 2 + c.disc_h + 0.003, quat=Q_BLUE_UP)
    step(150)
    report("near-miss-pos")
    sc, ok = judge()
    check("position near-miss: die settled blue-UP just outside the zone radius — "
          "NOT success, score <= 0.61 (blue-up alone does not finish the task)",
          bool(scene.blue_up()[0]) and still()
          and float(scene.dist_to_disc()[0]) > c.zone_r + 0.005
          and not ok and sc <= 0.61)

    # =========================== 12. airborne fly-through ===================================
    # Blue-up in free fall directly over the disc: judged mid-air the resting-height
    # gate must reject; the die is relocated before it can land on the disc.
    env.reset(seed=101)
    step(30)
    d = disc_p()
    place_die(float(d[0]), float(d[1]), 0.35, quat=Q_BLUE_UP)
    step(2)
    report("airborne")
    sc, ok = judge()
    air_ok = (bool(scene.blue_up()[0]) and float(die_p()[2]) > c.rest_z_hi + 0.05
              and float(scene.dist_to_disc()[0]) < c.zone_r
              and not bool(scene.in_zone()[0]) and not ok)
    place_die(FAR[0], FAR[1], s / 2 + 0.003, quat=Q_WHITE_UP)  # relocate pre-landing
    step(60)
    check("airborne: die blue-up in free fall directly over the disc, judged mid-air "
          "— NOT success (the resting-height gate is load-bearing)", air_ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.die_roll")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, ok in checks:
            if not ok:
                print(f"[smoke]   FAILED: {nm}", flush=True)
    code = 0 if all_ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()
