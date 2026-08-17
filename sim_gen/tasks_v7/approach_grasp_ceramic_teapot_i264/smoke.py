"""Smoke / rubric-REJECTION battery for TeapotBayonetScene (sim_gen task
`approach_grasp_ceramic_teapot_i264`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — hover the lid over the mouth, press it through the
keyed throat, twist it past the lock threshold — is the acceptance evidence that the
rubric ACCEPTS a correct outcome). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome and asserts the rubric REJECTS it; no probe in
this battery ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: lid upright on the ground in
                            its spawn ring, pot at its jittered pose; score ~0;
  3-4. randomization      — READBACK over 8 seeded resets: pot xy AND free yaw (the
                            notch phase) vary; lid ring radius stays in band while its
                            polar slot, own yaw, and the relative fold all vary;
  5.  null policy         — 300 idle steps -> score ~0, no success, no lid creep;
  6.  SEED strategy       — the seed's whole plan ("approach, grasp, hold the object
                            aloft") end state: the lid HELD above the pot mouth, then
                            set back down beside the pot. Carry latch only (<= 0.16),
                            NOT success — holding is worth almost nothing here;
  7.  rim-rest at angle   — lid dropped on the mouth already ROTATED to the lock angle
                            (lugs on the solid ring): rests ~24 mm too high, NO entered
                            credit, NOT success — right angle, wrong depth;
  8.  seated-unrotated    — lid pressed through the notches onto the ledge but NOT
                            twisted: seated() true yet NOT success (0.38 <= score <=
                            0.45); then a slow upward pull EXTRACTS it (readback above
                            the rim) — physical proof the unlocked lid is not captive;
  9.  under-rotated       — seated lid twisted only ~25 deg (< lock_min 36): NOT
                            success, partial rotation credit only (score <= 0.78);
  10. latched regression  — teleporting that lid back out to the ground leaves the
                            latched credit unchanged and success stays gone;
  11. inverted lid        — lid dropped knob-down into the mouth: cap perches on the
                            rim, origin far above the seat band, upright fails ->
                            NOT success (wrong orientation cannot fake the lock);
  12. rejection audit     — success() was never True at ANY judged point;
  13. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.approach_grasp_ceramic_teapot_i264.smoke --headless
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
# RTX recipe: kit mis-decodes the driver version and silently rejects RTX -> the
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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.teapot_bayonet")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.10, -0.95, 0.85)) + o),
                                tuple(np.array((0.10, 0.00, 0.14)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

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
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def lid_p() -> torch.Tensor:
        return scene._lid_pos()[0]

    def pot_p() -> torch.Tensor:
        return scene._pot_pos()[0]

    def pot_yaw() -> float:
        return float(scene._yaw(scene.pot.data.root_quat_w)[0])

    def fold_deg() -> float:
        return float(torch.rad2deg(scene.rel_yaw_fold())[0])

    def report(tag: str) -> None:
        lp, pp = lid_p(), pot_p()
        dxy = float((lp[:2] - pp[:2]).norm())
        s, ok = judge()
        print(f"[smoke] {tag:18s} | lid=({float(lp[0]):+.3f},{float(lp[1]):+.3f},"
              f"{float(lp[2]):.4f}) dxy={dxy:.4f} fold={fold_deg():5.1f}deg "
              f"up_z={float(scene.lid_up_z()[0]):+.3f} "
              f"car={float(scene.carried[0]):.0f} ent={float(scene.entered[0]):.0f} "
              f"rot={math.degrees(float(scene.rot_max[0])):4.1f} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def place_lid(x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0),
                  settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += scene.env_origins
        scene.lid.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def clear_wrench() -> None:
        scene.lid.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    lp, pp = lid_p(), pot_p()
    rad = float((lp[:2] - pp[:2]).norm())
    fin0 = bool(torch.isfinite(scene.lid.data.root_state_w).all()
                and torch.isfinite(scene.pot.data.root_state_w).all())
    check("settle: states finite; lid upright at ground height inside its spawn ring, "
          f"pot at its jittered pose (readback z={float(lp[2]):.4f}, rad={rad:.3f})",
          fin0 and abs(float(lp[2])) < 0.008
          and c.lid_r_lo - 0.03 < rad < c.lid_r_hi + 0.03
          and float(scene.lid_up_z()[0]) > 0.99
          and abs(float(pp[0]) - c.pot_nom[0]) <= c.pot_jitter + 0.005)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(5)
        lp, pp = lid_p(), pot_p()
        d = lp[:2] - pp[:2]
        lid_yaw = float(scene._yaw(scene.lid.data.root_quat_w)[0])
        reads.append((float(pp[0]), float(pp[1]), pot_yaw(),
                      float(d.norm()), math.atan2(float(d[1]), float(d[0])),
                      lid_yaw, math.radians(fold_deg())))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (pot_x, pot_y, pot_yaw, lid_rad, lid_ang, "
          f"lid_yaw, fold):\n{np.array2string(arr, precision=3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: pot xy jitter AND free pot yaw (the notch phase) vary "
          "across 8 seeds (readback)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.8
          and float(np.abs(arr[:, 0] - c.pot_nom[0]).max()) <= c.pot_jitter + 0.005)
    check("randomization: lid ring radius in band on every seed; polar slot, lid yaw "
          "and relative fold all vary (readback)",
          bool(np.all((arr[:, 3] > c.lid_r_lo - 0.02) & (arr[:, 3] < c.lid_r_hi + 0.02)))
          and spread[4] > 0.8 and spread[5] > 0.8 and spread[6] > 0.25)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(60)
    p0 = lid_p()[:2].clone()
    step(300)
    report("null-policy")
    drift = float((lid_p()[:2] - p0).norm())
    s, ok = judge()
    check("null policy: score ~0 and no success after 300 idle steps, and the lid does "
          f"not creep (drift {drift * 1000:.1f} mm)", s <= 0.02 and not ok and drift < 0.03)

    # =========================== 6. SEED strategy ============================================
    # The seed's whole plan: approach the teapot's object, close the jaw on it, HOLD it
    # (aloft, near the pot). Emulated honestly: the lid held fixed above the pot mouth
    # for ~0.5 s, judged mid-hold, then set back down beside the pot. The rubric must
    # give the carry latch at most and never success.
    env.reset(seed=41)
    step(30)
    pp = pot_p()
    hold = torch.zeros(n, 13, device=device)
    hold[:, 0], hold[:, 1], hold[:, 2] = float(pp[0]), float(pp[1]), 0.32
    hold[:, 3] = 1.0
    hold[:, 0:3] += scene.env_origins
    for _ in range(60):
        scene.lid.write_root_state_to_sim(hold, all_ids)
        step(1)
    report("seed-held")
    s_held, ok_held = judge()
    place_lid(float(pp[0]) + 0.30, float(pp[1]) + 0.10, 0.004, settle_steps=90)
    report("seed-setdown")
    s_down, ok_down = judge()
    check("seed strategy (lid grasped and HELD above the pot mouth, then set down "
          "beside the pot): carry latch only (score <= 0.16 both judged points), "
          "NOT success",
          s_held <= 0.16 and s_down <= 0.16 and not ok_held and not ok_down
          and float(scene.carried[0]) == 1.0)

    # =========================== 7. rim-rest at the lock angle ==============================
    # Right ANGLE, wrong DEPTH: the lid arrives over the mouth already rotated to the
    # lock angle, so its lugs land on the SOLID ring and it perches ~24 mm too high.
    env.reset(seed=51)
    step(30)
    pp = pot_p()
    place_lid(float(pp[0]), float(pp[1]), c.ring_z1 + 0.012,
              quat=yaw_quat(pot_yaw() + math.radians(45.0)), settle_steps=120)
    report("rim-rest")
    lp = lid_p()
    s, ok = judge()
    check("rim-rest at the lock angle: lugs perch on the solid ring "
          f"(z={float(lp[2]):.4f} > seat band), fold in the lock window yet NOT "
          "success, and NO entered credit (score <= 0.16)",
          float(lp[2]) > c.seat_z_hi + 0.006 and 30.0 < fold_deg() < 54.0
          and not ok and float(scene.entered[0]) == 0.0 and s <= 0.16)

    # =========================== 8. seated-unrotated + pull-out proof =======================
    env.reset(seed=61)
    step(30)
    pp = pot_p()
    place_lid(float(pp[0]), float(pp[1]), c.ring_z1 + 0.015,
              quat=yaw_quat(pot_yaw()), settle_steps=150)
    report("seated-unrot")
    lp = lid_p()
    s8, ok8 = judge()
    seated8 = bool(scene.seated()[0])
    check("seated-unrotated: lid dropped through the notches seats on the ledge "
          f"(z={float(lp[2]):.4f}, fold={fold_deg():.1f} deg) — seated() true, entered "
          "credit in, but NOT locked, NOT success (0.38 <= score <= 0.45)",
          seated8 and fold_deg() < 8.0 and not ok8 and 0.38 <= s8 <= 0.45)
    # Physical captivity counter-proof: a slow upward pull EXTRACTS the unrotated lid.
    m = c.lid_mass
    extracted = False
    for _ in range(400):
        vz = float(scene.lid.data.root_lin_vel_w[0, 2])
        fz = min(max(m * 60.0 * (0.15 - vz) + m * 9.81, 0.0), 8.0)
        f = torch.tensor([0.0, 0.0, fz], device=device)
        scene.lid.set_external_force_and_torque(
            f.view(1, 1, 3).expand(n, 1, 3).contiguous(), zero_wrench,
            env_ids=all_ids, is_global=True)
        env.step(no_action)
        if float(lid_p()[2]) > c.ring_z1 + 0.030:
            extracted = True
            break
    clear_wrench()
    report("pulled-out")
    check("pull-out proof: the UNROTATED seated lid extracts under a slow upward pull "
          f"(z={float(lid_p()[2]):.3f} readback above the rim) — the unlocked state "
          "is genuinely not captive", extracted and float(lid_p()[2]) > c.ring_z1 + 0.020)

    # =========================== 9. under-rotated near-miss =================================
    env.reset(seed=71)
    step(30)
    pp = pot_p()
    place_lid(float(pp[0]), float(pp[1]), c.seat_z + 0.004,
              quat=yaw_quat(pot_yaw() + math.radians(25.0)), settle_steps=100)
    report("under-rotated")
    lp = lid_p()
    s9, ok9 = judge()
    check("under-rotated near-miss: lid seated but twisted only "
          f"~{fold_deg():.1f} deg (< lock_min {c.lock_min_deg:.0f}) — NOT success, "
          "partial rotation credit only (0.35 <= score <= 0.78)",
          bool(scene.seated()[0]) and 15.0 < fold_deg() < 34.0 and not ok9
          and 0.35 <= s9 <= 0.78)

    # =========================== 10. latched credit survives regression =====================
    place_lid(float(pp[0]) + 0.32, float(pp[1]) - 0.12, 0.004, settle_steps=60)
    report("regressed")
    s10, ok10 = judge()
    check("latched regression: teleporting the under-rotated lid back out to the "
          f"ground leaves the latched credit unchanged ({s9:.3f} -> {s10:.3f}) and "
          "success stays gone", abs(s10 - s9) < 0.02 and not ok10)

    # =========================== 11. inverted lid ===========================================
    env.reset(seed=81)
    step(30)
    pp = pot_p()
    place_lid(float(pp[0]), float(pp[1]), 0.32, quat=(0.0, 1.0, 0.0, 0.0),
              settle_steps=150)
    report("inverted")
    lp = lid_p()
    s, ok = judge()
    check("inverted lid (knob-down in the mouth): cap perches on the rim, origin far "
          f"above the seat band (z={float(lp[2]):.3f}), upright fails -> NOT success",
          float(lp[2]) > c.ring_z1 + 0.020 and float(scene.lid_up_z()[0]) < -0.5
          and not ok and s <= 0.16)

    # =========================== 12-13. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.lid.data.root_state_w).all()
           and torch.isfinite(scene.pot.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.teapot_bayonet")
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
