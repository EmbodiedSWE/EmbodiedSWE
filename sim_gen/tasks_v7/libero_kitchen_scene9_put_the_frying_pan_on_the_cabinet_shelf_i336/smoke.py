"""Smoke / rubric-REJECTION battery for ComboShutterPantryScene (sim_gen task
`libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf_i336`) — NullRobot,
teleported probe states, RECORDED.

This is NOT a solution (solve.py — force-align both shutter plates, then force-push the
pan flat through the registered gate — is the acceptance evidence that the rubric
ACCEPTS a correct outcome). Every teleport here is instrumentation that CONSTRUCTS a
wrong (or partial) outcome and asserts the rubric REJECTS it; no probe in this battery
ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: plates seated at their spawn
                            offsets, pan + saucer at rest on the counter; score ~0;
  3-5. randomization      — READBACK over 8 seeded resets: cabinet xy + free yaw; BOTH
                            plate offsets vary, span BOTH signs and stay in the spawn
                            band (each alone gate-blocking); pan xy + saucer side vary;
  6.  null policy         — 240 idle steps -> score ~0, no success;
  7.  SEED strategy       — the seed's whole plan ("set the object down on top of the
                            fixture") = pan parked ON THE ROOF of the pantry: verified
                            above the roof plane, NOT success, score ~0;
  8.  blocked-gate press  — both plates misaligned (spawn state), the pan DRIVEN at the
                            doorway with the solve's own bounded governor for ~3.3 s:
                            the probe asserts the pan MOVED (>= 3 cm, not vacuous) yet
                            never entered the doorway slab; plates still misaligned
                            (the press must not walk them open), score ~0;
  9.  single-plate press  — ONE plate registered, the other still misaligned; same
                            press: still blocked, score ~0.15 (one alignment latch);
  10. straddle near miss  — both plates registered, the pan honestly pushed but STOPPED
                            mid-doorway: the transit latch fires LEGITIMATELY, yet NOT
                            success (not inside), score <= 0.70;
  11. teleport inject     — pan WRITTEN directly into the alcove (physically consistent
                            rest pose, gates never registered): pan_inside verified
                            True, yet the transit pathway latch stays 0 -> NOT success,
                            score ~0;
  12. wrong object        — saucer put INSIDE the alcove first, then the pan honestly
                            pushed through the registered gate: pan in AND saucer in ->
                            success remains False (decoy-out constraint; ordered so no
                            prefix of the construction satisfies the goal);
  13. flipped pan         — pan flipped upside-down, honestly pushed through the
                            registered gate to the goal x: upright clause rejects;
  14. latched credit      — registering both plates then shoving one far off leaves the
                            latched alignment score unchanged (~0.50) and success gone;
  15. rejection audit     — success() was never True at ANY judged point;
  16. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf_i336.smoke --headless
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
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.combo_shutter_pantry")().build(num_envs=args.num_envs,
                                                          device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.05, -0.95, 0.90)) + o),
                                tuple(np.array((0.05, 0.0, 0.28)) + o),
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

    def cab_pose() -> tuple[torch.Tensor, torch.Tensor]:
        return scene.cabinet.data.root_pos_w[0:1], scene.cabinet.data.root_quat_w[0:1]

    def cab_axes() -> tuple[torch.Tensor, torch.Tensor]:
        yaw = float(scene._yaw_of(scene.cabinet)[0])
        xd = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0], device=device)
        yd = torch.tensor([-math.sin(yaw), math.cos(yaw), 0.0], device=device)
        return xd, yd

    def place_in_cab(body, lx: float, ly: float, lz: float, q_extra=None,
                     settle_steps: int = 30) -> None:
        """Kinematic probe placement in the CABINET frame (instrumentation, not a
        solution) + REAL physics steps before judging (the zero-step trap)."""
        cp, cq = cab_pose()
        w = cp + quat_apply(cq, torch.tensor([[lx, ly, lz]], device=device))
        q = cq if q_extra is None else quat_mul(cq, torch.tensor([q_extra], device=device))
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = w
        st[:, 3:7] = q
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def place_plate(body, slot_x: float, offset: float, settle_steps: int = 30) -> None:
        place_in_cab(body, slot_x, offset, c.plate_z0 + 0.001, settle_steps=settle_steps)

    def stage_pan(flipped: bool = False, settle_steps: int = 50) -> None:
        """The transport move solve.py performs: set the pan down on the bare counter
        in front of the gate, handle trailing (or upside-down for the flip probe)."""
        q = [0.0, 1.0, 0.0, 0.0] if flipped else [0.0, 0.0, 0.0, 1.0]
        lz = 0.038 if flipped else 0.004  # flipped: body origin = disc TOP
        place_in_cab(scene.pan, -0.150, 0.0, lz, q_extra=q, settle_steps=settle_steps)

    def clear_pan_wrench() -> None:
        scene.pan.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def push_pan(x_target: float, budget: int) -> float:
        """The solve's bounded planar push (position-carrot spring toward a waypoint
        advancing along the doorway centreline, force capped at 5 N — mirrors
        solve.py; position-referenced so GPU phantom-velocity noise cannot veer
        or stall it). The commanded WORLD force is pre-encoded into the pan's
        LIVE body frame and applied with the default (body-frame) call: on these
        pods `is_global=True` rotates the wrench by the body's rotation since its
        FIRST-ever application, and that stale reference persists across
        env.reset — so every check after the first push would get a force rotated
        by the yaw difference between the two seeds' layouts. Returns the pan's
        final cabinet-frame x after force-off settle."""
        x_ref = float(scene.cab_local(scene.pan.data.root_pos_w)[0][0])
        kx, dx_, cap = 120.0, 8.0, 5.0
        ky, dy_, capy = 250.0, 10.0, 4.5
        for _i in range(budget):
            loc = scene.cab_local(scene.pan.data.root_pos_w)[0]
            x, y = float(loc[0]), float(loc[1])
            if x > x_target:
                break
            x_ref = min(x_ref + 0.05 / 120.0, x_target + cap / kx + 0.02)
            xd, yd = cab_axes()
            v = scene.pan.data.root_lin_vel_w[0]
            vx = float(torch.dot(v, xd))
            vy = float(torch.dot(v, yd))
            fx = max(-cap, min(cap, kx * (x_ref - x) - dx_ * vx))
            fy = max(-capy, min(capy, -ky * y - dy_ * vy))
            fb = quat_apply_inverse(scene.pan.data.root_quat_w[0:1],
                                    (xd * fx + yd * fy).view(1, 3))
            fw = fb.view(1, 1, 3).expand(n, 1, 3).contiguous()
            scene.pan.set_external_force_and_torque(fw, zero_wrench, env_ids=all_ids)
            env.step(no_action)
        clear_pan_wrench()
        step(60)
        return float(scene.cab_local(scene.pan.data.root_pos_w)[0][0])

    def report(tag: str) -> None:
        aF = float(scene.plate_offset(scene.plate_f)[0])
        aR = float(scene.plate_offset(scene.plate_r)[0])
        pl = scene.cab_local(scene.pan.data.root_pos_w)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | aF={aF * 1000:+.1f}mm aR={aR * 1000:+.1f}mm "
              f"pan_cab=({float(pl[0]):+.3f},{float(pl[1]):+.3f},{float(pl[2]):.3f}) "
              f"inside={bool(scene.pan_inside()[0])} "
              f"upright={bool(scene.pan_upright()[0])} "
              f"decoy_in={bool(scene.decoy_in_alcove()[0])} "
              f"lF={float(scene.alignF_latch[0]):.2f} lR={float(scene.alignR_latch[0]):.2f} "
              f"lB={float(scene.both_latch[0]):.2f} lT={float(scene.transit_latch[0]):.2f} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def all_finite() -> bool:
        return bool(torch.isfinite(scene.cabinet.data.root_state_w).all()
                    and torch.isfinite(scene.plate_f.data.root_state_w).all()
                    and torch.isfinite(scene.plate_r.data.root_state_w).all()
                    and torch.isfinite(scene.pan.data.root_state_w).all()
                    and torch.isfinite(scene.decoy.data.root_state_w).all())

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    pl = scene.cab_local(scene.pan.data.root_pos_w)[0]
    check("settle: states finite; both plates seated at rest; pan flat at counter "
          "height (readback)",
          all_finite() and bool(scene.plate_seated(scene.plate_f, c.slot_xF)[0])
          and bool(scene.plate_seated(scene.plate_r, c.slot_xR)[0])
          and abs(float(pl[2])) < 0.010 and bool(scene.pan_upright()[0])
          and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-5. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(5)
        cp = (scene.cabinet.data.root_pos_w - scene.env_origins)[0]
        cy = float(scene._yaw_of(scene.cabinet)[0])
        aF = float(scene.plate_offset(scene.plate_f)[0])
        aR = float(scene.plate_offset(scene.plate_r)[0])
        pp = scene.cab_local(scene.pan.data.root_pos_w)[0]
        dp = scene.cab_local(scene.decoy.data.root_pos_w)[0]
        reads.append((float(cp[0]), float(cp[1]), cy, aF, aR,
                      float(pp[0]), float(pp[1]), float(dp[1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (cab_x, cab_y, cab_yaw, aF, aR, pan_x, "
          f"pan_y, decoy_y):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: cabinet xy + free yaw vary across seeded resets (readback)",
          spread[0] > 0.01 and spread[1] > 0.01 and spread[2] > 0.5)
    band_ok = bool((np.abs(arr[:, 3:5]) > c.off_lo - 0.006).all()
                   and (np.abs(arr[:, 3:5]) < c.off_hi + 0.006).all())
    signs_ok = bool((arr[:, 3] > 0).any() and (arr[:, 3] < 0).any()
                    and (arr[:, 4] > 0).any() and (arr[:, 4] < 0).any())
    check("randomization: BOTH plate offsets vary, span BOTH signs, and stay in the "
          "gate-blocking spawn band (readback)",
          spread[3] > 0.02 and spread[4] > 0.02 and band_ok and signs_ok)
    check("randomization: pan xy varies and the saucer spawns on BOTH sides (readback)",
          spread[5] > 0.01 and spread[6] > 0.02
          and bool((arr[:, 7] > 0).any() and (arr[:, 7] < 0).any()))

    # =========================== 6. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 7. SEED strategy ===========================================
    # The seed's whole plan is "set the object down on top of the fixture". Here that is
    # the pan parked ON THE ROOF of the pantry. The rubric must refuse it.
    env.reset(seed=41)
    step(10)
    roof_top = c.alc_h + c.roof_t
    place_in_cab(scene.pan, 0.21, 0.0, roof_top + 0.004, settle_steps=60)
    report("seed-strategy")
    s, ok = judge()
    pz = float(scene.cab_local(scene.pan.data.root_pos_w)[0, 2])
    check("seed strategy (pan set down ON TOP of the pantry roof): verified above the "
          "roof plane, NOT success, score ~0",
          pz > roof_top - 0.005 and not bool(scene.pan_inside()[0])
          and not ok and s <= 0.02)

    # =========================== 8. blocked-gate press ======================================
    # Both plates at their spawn misalignments: the solve's own bounded push must NOT
    # get the pan into the doorway slab — and the probe must really push (not vacuous),
    # and must not walk the plates into registration.
    env.reset(seed=51)
    step(30)
    aF0 = float(scene.plate_offset(scene.plate_f)[0])
    aR0 = float(scene.plate_offset(scene.plate_r)[0])
    stage_pan()
    x_before = float(scene.cab_local(scene.pan.data.root_pos_w)[0][0])
    x_after = push_pan(x_target=0.30, budget=400)  # 3.3 s of honest pushing
    report("blocked-press")
    s, ok = judge()
    aF1 = float(scene.plate_offset(scene.plate_f)[0])
    aR1 = float(scene.plate_offset(scene.plate_r)[0])
    check("blocked gate: with both plates misaligned the driven pan MOVED >= 3 cm but "
          "never entered the doorway slab; plates not walked open; score ~0",
          abs(aF0) > c.pass_tol and abs(aR0) > c.pass_tol
          and (x_after - x_before) > 0.03 and x_after < c.slab_x0
          and abs(aF1) > c.pass_tol and abs(aR1) > c.pass_tol
          and float(scene.transit_latch[0]) < 0.5 and not ok and s <= 0.02)

    # =========================== 9. single-plate press ======================================
    env.reset(seed=61)
    step(30)
    place_plate(scene.plate_r, c.slot_xR, 0.0)  # rear registered, front still off
    aF0 = float(scene.plate_offset(scene.plate_f)[0])
    stage_pan()
    x_after = push_pan(x_target=0.30, budget=400)
    report("single-press")
    s, ok = judge()
    check("single plate registered: the driven pan is still blocked before the slab, "
          "transit latch 0, score ~0.15 (one alignment latch only)",
          abs(aF0) > c.pass_tol and x_after < c.slab_x0
          and float(scene.transit_latch[0]) < 0.5
          and float(scene.alignR_latch[0]) > 0.5 and not ok and 0.13 <= s <= 0.16)

    # =========================== 10. straddle near miss =====================================
    env.reset(seed=71)
    step(30)
    place_plate(scene.plate_r, c.slot_xR, 0.0)
    place_plate(scene.plate_f, c.slot_xF, 0.0)
    stage_pan()
    x_mid = push_pan(x_target=0.045, budget=700)  # stop INSIDE the doorway slab
    report("straddle")
    s, ok = judge()
    check("straddle near miss: pan honestly stopped mid-doorway — transit latch fires "
          "LEGITIMATELY, yet NOT success (not inside), score <= 0.70",
          c.slab_x0 < x_mid < c.entry_x - 0.02
          and float(scene.transit_latch[0]) > 0.5
          and not bool(scene.pan_inside()[0]) and not ok and s <= 0.705)

    # =========================== 11. teleport inject ========================================
    env.reset(seed=81)
    step(30)
    place_in_cab(scene.pan, 0.22, 0.0, 0.004, settle_steps=40)
    report("inject")
    s, ok = judge()
    check("teleport inject: pan WRITTEN into the alcove (pan_inside verified True, at "
          "rest) — gates never registered, transit latch 0 -> NOT success, score ~0",
          bool(scene.pan_inside()[0]) and bool(scene.pan_upright()[0])
          and float(scene.transit_latch[0]) < 0.5 and not ok and s <= 0.02)

    # =========================== 12. wrong object ===========================================
    # Saucer FIRST (so no prefix of this construction ever satisfies the goal), then
    # the pan honestly through the registered gate.
    env.reset(seed=91)
    step(30)
    # decoy at (0.288, 0.075): solidly inside, and 25 mm edge clearance from the
    # delivered pan (target 0.175 -> pan leading edge 0.240 vs decoy near edge 0.243
    # at dy=0.075 -> centre distance 0.136 > r_pan + r_decoy = 0.110)
    place_in_cab(scene.decoy, 0.288, 0.075, c.decoy_h / 2 + 0.003, settle_steps=40)
    place_plate(scene.plate_r, c.slot_xR, 0.0)
    place_plate(scene.plate_f, c.slot_xF, 0.0)
    stage_pan()
    x_in = push_pan(x_target=0.175, budget=1200)
    report("wrong-object")
    s, ok = judge()
    check("wrong object: saucer inside the alcove AND the pan honestly pushed through "
          "the registered gate (pan_inside verified) — success remains False",
          bool(scene.decoy_in_alcove()[0]) and x_in > c.entry_x
          and bool(scene.pan_inside()[0]) and float(scene.transit_latch[0]) > 0.5
          and not ok)

    # =========================== 13. flipped pan ============================================
    env.reset(seed=101)
    step(30)
    place_plate(scene.plate_r, c.slot_xR, 0.0)
    place_plate(scene.plate_f, c.slot_xF, 0.0)
    stage_pan(flipped=True)
    x_flip = push_pan(x_target=0.19, budget=1200)
    report("flipped")
    s, ok = judge()
    check("flipped pan: pushed upside-down through the registered gate to the goal x — "
          "the upright clause rejects (NOT success)",
          x_flip > c.entry_x and not bool(scene.pan_upright()[0]) and not ok)

    # =========================== 14. latched credit survives regression =====================
    env.reset(seed=111)
    step(10)
    place_plate(scene.plate_r, c.slot_xR, 0.0)
    place_plate(scene.plate_f, c.slot_xF, 0.0)
    report("both-registered")
    s_in, _ = judge()
    place_plate(scene.plate_f, c.slot_xF, 0.10, settle_steps=40)  # shove one far off
    report("plate-shoved")
    s_out, ok = judge()
    aF_now = abs(float(scene.plate_offset(scene.plate_f)[0]))
    check("latched credit: de-registering a plate after both were registered leaves "
          "the latched alignment score unchanged (~0.50) and success gone",
          s_in >= 0.49 and abs(s_out - s_in) < 0.02 and aF_now > c.align_tol and not ok)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.combo_shutter_pantry")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, okc in checks:
            if not okc:
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
    except BaseException as exc:  # noqa: BLE001 — die loudly, don't wait for the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL (exception: {exc!r})", flush=True)
        os._exit(2)
