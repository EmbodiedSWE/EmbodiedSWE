"""Smoke / rubric-REJECTION battery for BallastHatchScene (sim_gen task
`stack_cube_i183`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — gravity-dropped ballast actuating the hatch, then a
nonprehensile doorway push — is the acceptance evidence that the rubric ACCEPTS a
correct outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partial) outcome and asserts the rubric REJECTS it; no probe in this battery ever
reaches success(), and a final audit check asserts exactly that.

  1.   settle/no-NaN      — reset layout settles finite: gate CLOSED (angle readback),
                            parcel in its lane, score ~0, no success;
  2-3. randomization      — READBACK over 6 seeded resets: apparatus xy + yaw, parcel
                            lane pose, and ballast slot polar coords all move;
  4.   null policy        — 300 idle steps: gate still closed, parcel still in the
                            lane, score ~0;
  5.   SEED strategy      — the seed's whole plan is stack-cube-on-cube: the parcel is
                            set down ON TOP of a ballast cube (stack verified by
                            readback) — worthless here, score ~0;
  6.   sealed roof        — parcel dropped onto the vault roof rests ON it (readback),
                            never inside: the doorway is the only way in;
  7.   direct-to-chamber  — parcel teleported INSIDE the chamber, settled: physically
                            contained and still, yet NOT success (pathway latches
                            unearned), score ~0;
  8.   closed-gate push   — the solve-strength push against the CLOSED flap: the parcel
                            verifiably MOVES (actuation proof), then stalls OUTSIDE the
                            doorway; the flap holds (pushing presses it shut), score ~0;
  9.   one cube too few   — ONE ballast cube dropped in: the gate verifiably responds
                            (partial angle readback) but stays below the latch, and the
                            same push is still physically BLOCKED by the low flap;
  10.  ballast-only       — all four cubes in: gate at the limit stop, open latch
                            earned — but that alone is only 0.30, NOT success;
  11.  door near-miss     — parcel left IN the doorway slot: doorway latch earns 0.60
                            but NOT success; pulled back out, the latched credit
                            survives regression (and still no success);
  12.  self-reclosing     — ballast removed: the gate re-CLOSES on its own (readback)
                            and the latched credit survives the mechanism resetting;
  13.  monotonicity       — the battery's earned stages are strictly increasing:
                            null ~0 < gate-open 0.30 < doorway 0.60;
  14.  rejection audit    — success() was never True at ANY judged point;
  15.  final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.stack_cube_i183.smoke --headless
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ballast_hatch")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.95, -0.90, 0.75)) + o),
                                tuple(np.array((0.0, 0.0, 0.15)) + o),
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

    def gate_deg() -> float:
        return math.degrees(float(scene.gate_open_angle()[0]))

    def report(tag: str) -> None:
        p = scene.payload_vault_local()[0]
        s, ok = judge()
        print(f"[smoke] {tag:18s} | parcel_vault=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):+.3f}) gate={gate_deg():+.1f}deg "
              f"open={bool(scene.lat_open[0])} door={bool(scene.lat_door[0])} "
              f"cham={bool(scene.in_chamber()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_local(body, lx: float, ly: float, lz: float, settle_steps: int = 30,
                    yaw_align: bool = True) -> None:
        """Kinematic probe placement in the VAULT's local frame (instrumentation, not a
        solution) + REAL physics steps before judging (the zero-step trap)."""
        v = torch.tensor([lx, ly, lz], device=device).unsqueeze(0).expand(n, 3)
        w = scene.vault.data.root_pos_w + quat_apply(scene.vault.data.root_quat_w, v)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = w
        st[:, 3:7] = scene.vault.data.root_quat_w if yaw_align \
            else torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def drop_ballast(i: int, settle_steps: int = 300) -> None:
        """Hover cube i above its hopper CELL's CURRENT opening (gate-pose readback),
        cell-aligned, release — the gravity-drop loading used by solve.py."""
        mw = scene.cell_mouth_world(i)[0]
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = mw[0], mw[1], mw[2] + 0.032
        st[:, 3:7] = scene.cell_drop_quat()
        scene.ballast[i].write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def push_payload(max_steps: int) -> float:
        """The solve-strength doorway push (velocity servo, |F| <= 0.7 N, body frame).
        Returns the parcel's final vault-local x."""
        v_des, k_v, cap_along = 0.12, 5.0, 0.70
        k_y, k_dy, cap_lat = 6.0, 2.0, 0.40
        for _ in range(max_steps):
            qv = scene.vault.data.root_quat_w
            p = scene.payload_vault_local()[0]
            if float(p[0]) < 0.02:
                break
            v_local = quat_apply_inverse(qv, scene.payload.data.root_lin_vel_w)[0]
            f_along = max(-cap_along, min(cap_along, k_v * (v_des + float(v_local[0]))))
            f_lat = max(-cap_lat, min(cap_lat,
                                      k_y * (0.0 - float(p[1])) - k_dy * float(v_local[1])))
            f_vault = torch.tensor([-f_along, f_lat, 0.0], device=device).unsqueeze(0)
            f_body = quat_apply_inverse(scene.payload.data.root_quat_w,
                                        quat_apply(qv, f_vault))
            scene.payload.set_external_force_and_torque(
                f_body.view(n, 1, 3).contiguous(), zero_wrench, env_ids=all_ids)
            env.step(no_action)
        scene.payload.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                    env_ids=all_ids)
        step(60)
        return float(scene.payload_vault_local()[0][0])

    def all_finite() -> bool:
        ok = (torch.isfinite(scene.vault.data.root_state_w).all()
              and torch.isfinite(scene.gate.data.root_state_w).all()
              and torch.isfinite(scene.payload.data.root_state_w).all())
        for b in scene.ballast:
            ok = ok and torch.isfinite(b.data.root_state_w).all()
        return bool(ok)

    pz = c.payload_size / 2 + 0.0005

    # =========================== 1. settle / no-NaN =========================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    p = scene.payload_vault_local()[0]
    s, ok = judge()
    check("settle: states finite; gate CLOSED (|angle| < 3 deg readback), parcel in its "
          "approach lane, score ~0, no success",
          all_finite() and abs(gate_deg()) < 3.0 and 0.18 < float(p[0]) < 0.34
          and abs(float(p[1])) < 0.06 and s <= 0.02 and not ok)

    # =========================== 2-3. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        vp = (scene.vault.data.root_pos_w - scene.env_origins)[0]
        q = scene.vault.data.root_quat_w[0]
        yaw = math.atan2(2 * float(q[0]) * float(q[3]), 1 - 2 * float(q[3]) ** 2)
        pl = scene.payload_vault_local()[0]
        b0 = scene.vault_local(scene.ballast[0].data.root_pos_w)[0]
        reads.append((float(vp[0]), float(vp[1]), yaw, float(pl[0]), float(pl[1]),
                      math.atan2(float(b0[1]), float(b0[0])),
                      math.hypot(float(b0[0]), float(b0[1]))))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (vault_x, vault_y, yaw, parcel_lx, parcel_ly, "
          f"ballast0_ang, ballast0_r):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: apparatus xy + yaw vary across seeded resets (readback)",
          spread[0] > 0.005 and spread[1] > 0.005 and spread[2] > 0.5)
    check("randomization: parcel lane pose AND ballast slot polar coords vary across "
          "seeded resets (readback)",
          spread[3] > 0.015 and spread[4] > 0.005 and spread[5] > 0.02
          and spread[6] > 0.005)

    # =========================== 4. null policy fails =======================================
    env.reset(seed=31)
    step(300)
    report("null-policy")
    p = scene.payload_vault_local()[0]
    s, ok = judge()
    check("null policy: after 300 idle steps the gate is still closed (readback), the "
          "parcel still in its lane, score ~0, no success",
          abs(gate_deg()) < 3.0 and float(p[0]) > 0.15 and s <= 0.02 and not ok)
    s_null = s

    # =========================== 5. SEED strategy ===========================================
    # The seed's whole plan is stack-cube-on-cube (relative-pose detector). Constructed
    # literally here: parcel set down ON TOP of a ballast cube, settled, stack verified
    # by readback — and worth nothing.
    env.reset(seed=41)
    step(10)
    b0 = scene.ballast[0].data.root_pos_w[0]
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1] = b0[0], b0[1]
    st[:, 2] = b0[2] + c.cube_size / 2 + c.payload_size / 2 + 0.002
    st[:, 3:7] = scene.ballast[0].data.root_quat_w
    scene.payload.write_root_state_to_sim(st, all_ids)
    step(150)
    report("seed-strategy")
    dz = float(scene.payload.data.root_pos_w[0][2] - scene.ballast[0].data.root_pos_w[0][2])
    s, ok = judge()
    check("seed strategy (stack cube on cube): parcel settled ON TOP of a ballast cube "
          "(stack verified by readback dz), gate untouched — score ~0, no success",
          dz > 0.035 and abs(gate_deg()) < 3.0 and s <= 0.02 and not ok)

    # =========================== 6. sealed roof =============================================
    env.reset(seed=51)
    step(10)
    place_local(scene.payload, 0.0, 0.0, 0.20, settle_steps=150)
    report("roof-drop")
    p = scene.payload_vault_local()[0]
    s, ok = judge()
    check("sealed roof: parcel dropped onto the vault from above rests ON the roof "
          "(readback z > 0.10), not inside — the doorway is the only way in; score ~0",
          float(p[2]) > 0.10 and not bool(scene.in_chamber()[0]) and s <= 0.02 and not ok)

    # =========================== 7. direct-to-chamber teleport ==============================
    env.reset(seed=61)
    step(10)
    place_local(scene.payload, -0.010, 0.0, pz + 0.003, settle_steps=120)
    report("direct-inside")
    s, ok = judge()
    check("direct-to-chamber teleport: parcel verified physically INSIDE the chamber and "
          "settled, yet NOT success (gate/doorway latches unearned), score ~0",
          bool(scene.in_chamber()[0]) and bool(scene.settled()[0]) and not ok
          and s <= 0.02)

    # =========================== 8. closed-gate push ========================================
    env.reset(seed=71)
    step(10)
    place_local(scene.payload, 0.19, 0.0, pz, settle_steps=30)
    x_final = push_payload(600)
    report("closed-gate-push")
    s, ok = judge()
    check("closed-gate push: the solve-strength push verifiably MOVED the parcel "
          "(readback x 0.19 -> < 0.16) but it stalled OUTSIDE the doorway against the "
          "closed flap (x > 0.112), the gate held closed (< 6 deg), score ~0",
          x_final < 0.16 and x_final > 0.112 and abs(gate_deg()) < 6.0
          and not bool(scene.in_chamber()[0]) and s <= 0.02 and not ok)

    # =========================== 9. one cube too few ========================================
    env.reset(seed=81)
    step(10)
    drop_ballast(0, settle_steps=360)
    ang_1 = gate_deg()
    report("one-cube")
    lat_after_one = bool(scene.lat_open[0])
    # stage OUTBOARD of the ~30 deg-swung flap (its bottom edge reaches x ~0.21)
    place_local(scene.payload, 0.26, 0.0, pz, settle_steps=30)
    x_final = push_payload(700)
    report("one-cube-push")
    s, ok = judge()
    check("one cube too few: the gate verifiably RESPONDS to one ballast cube (angle "
          f"readback {ang_1:.1f} deg in (6, 34)) but stays below the 35 deg latch, and "
          "the push is still physically blocked by the low flap (x > 0.112, doorway "
          "latch False), score ~0",
          6.0 < ang_1 < 34.0 and not lat_after_one and x_final > 0.112
          and not bool(scene.lat_door[0]) and s <= 0.02 and not ok)

    # =========================== 10. ballast-only ===========================================
    # Park the parcel clear of the flap's swing arc, then load the remaining cubes.
    place_local(scene.payload, 0.34, 0.10, pz, settle_steps=20)
    for i in (1, 2, 3):
        drop_ballast(i, settle_steps=280)
    step(120)
    report("ballast-only")
    ang_4 = gate_deg()
    s, ok = judge()
    check("ballast-only: four cubes hold the gate at the limit stop (angle readback "
          f"{ang_4:.1f} >= 50 deg), open latch earned — but that alone is only 0.30, "
          "NOT success",
          ang_4 >= 50.0 and bool(scene.lat_open[0]) and abs(s - c.stage_scores[0]) < 0.02
          and not ok)
    s_open = s

    # =========================== 11. door near-miss + latched credit ========================
    # Gate is open, the doorway slot is free space: parcel left IN the slot earns the
    # doorway latch but NOT success; pulled back out, the credit survives regression.
    place_local(scene.payload, 0.090, 0.0, pz + 0.002, settle_steps=60)
    report("door-nearmiss")
    s_door, ok_door = judge()
    place_local(scene.payload, 0.26, 0.0, pz, settle_steps=60)
    report("pulled-back-out")
    s_back, ok_back = judge()
    check("door near-miss: parcel left IN the doorway slot earns the doorway latch "
          "(0.60) but NOT success; pulled back OUT of the doorway the latched credit "
          "survives regression, still no success",
          bool(scene.lat_door[0]) and abs(s_door - c.stage_scores[1]) < 0.02
          and not ok_door and abs(s_back - s_door) < 0.02 and not ok_back)

    # =========================== 12. self-reclosing =========================================
    # Remove the ballast: the bell-crank's own preload must re-close the hatch, and the
    # latched credit must survive the mechanism resetting.
    for i, ang in enumerate((30.0, 120.0, -120.0, -30.0)):
        a = math.radians(ang)
        place_local(scene.ballast[i], 0.45 * math.cos(a), 0.45 * math.sin(a),
                    c.cube_size / 2 + 0.002, settle_steps=5)
    step(600)
    report("ballast-removed")
    s, ok = judge()
    check("self-reclosing: with the ballast removed the gate re-CLOSES on its own "
          f"(angle readback {gate_deg():.1f} < 8 deg) and the latched credit survives "
          "the mechanism resetting (score unchanged), still no success",
          gate_deg() < 8.0 and abs(s - s_door) < 0.02 and not ok)

    # =========================== 13. monotonicity ===========================================
    check("monotonicity: the battery's earned stages are strictly increasing "
          f"(null {s_null:.2f} < gate-open {s_open:.2f} < doorway {s_door:.2f}), and "
          "success (1.0) was never granted",
          s_null + 0.10 < s_open and s_open + 0.10 < s_door)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ballast_hatch")
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
    try:
        main()
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - die fast, don't wait for the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)
