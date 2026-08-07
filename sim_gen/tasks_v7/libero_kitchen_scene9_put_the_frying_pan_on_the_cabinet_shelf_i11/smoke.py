"""Smoke / rubric-REJECTION battery for MatchboxDrawerScene (sim_gen task
`libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf_i11`) — NullRobot,
teleported probe states, RECORDED.

This is NOT a solution (solve.py — drive the drawer open with a bounded force, drop the
cube in under gravity, drive it shut — is the acceptance evidence that the rubric
ACCEPTS a correct outcome). Every teleport here is instrumentation that CONSTRUCTS a
wrong (or partial) outcome and asserts the rubric REJECTS it; no probe in this battery
ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: tray shut and seated, cubes at
                            rest on the counter; score ~0 at rest, no success;
  3-4. randomization      — READBACK over 8 seeded resets: cabinet xy + free yaw, tray
                            seating dither, payload + decoy cabinet-frame xy all move,
                            and the payload spawns on BOTH sides across seeds;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's whole plan ("set the object down on top of the
                            fixture") = red cube settled ON TOP of the drawer unit's
                            roof: verified above the roof plane, NOT success, score ~0;
  7.  shut-drawer inject  — red cube WRITTEN directly inside the SHUT drawer's cavity
                            (a physically consistent pose — but no aperture was ever
                            opened): in_cavity AND enclosed AND shut all verified True,
                            yet the load pathway latch stays 0 -> NOT success, score ~0;
  8.  open-not-closed     — drawer opened (probe teleport of the tray), cube honestly
                            dropped in under gravity, drawer LEFT OPEN: loaded latch set,
                            yet NOT success, score <= 0.60;
  9.  almost-closed       — from the loaded-open state, tray AND cube shifted together
                            to a 30 mm opening (> close_tol): still NOT success (shut
                            requires < 12 mm), score <= 0.85;
  10. wrong object        — the BLUE decoy loaded into the open drawer and the drawer
                            shut around it: shut + decoy-in-cavity verified, success
                            remains False (identity + decoy-out constraint);
  11. porch dump          — red cube left ON THE PORCH in front of the shut drawer:
                            settled, NOT success, score ~0;
  12. latched credit      — loading the cube (honest drop into the opened drawer) then
                            removing it leaves the latched open + load score unchanged
                            (and success is gone);
  13. monotonicity        — a wider opening latches strictly more opening credit than a
                            narrower one;
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf_i11.smoke --headless
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
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.matchbox_drawer")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.85, -0.75, 0.85)) + o),
                                tuple(np.array((0.0, 0.0, 0.25)) + o),
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

    def place_in_cab(body, lx: float, ly: float, lz: float,
                     settle_steps: int = 30) -> None:
        """Kinematic probe placement in the CABINET frame (instrumentation, not a
        solution) + REAL physics steps before judging (the zero-step trap)."""
        cp, cq = cab_pose()
        off = torch.tensor([[lx, ly, lz]], device=device)
        w = cp + quat_apply(cq, off)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = w
        st[:, 3:7] = cq
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def place_tray(opening: float, settle_steps: int = 30) -> None:
        place_in_cab(scene.tray, c.x_closed + opening, 0.0, c.plate_top + 0.002,
                     settle_steps=settle_steps)

    def drop_into_tray(body, settle_steps: int = 120) -> None:
        """The honest loading move solve.py performs: hover 100 mm above the open
        cavity (in the tray frame), then free fall + settle."""
        tp = scene.tray.data.root_pos_w[0:1]
        tq = scene.tray.data.root_quat_w[0:1]
        w = tp + quat_apply(tq, torch.tensor([[0.0, 0.0, 0.100]], device=device))
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = w
        st[:, 3:7] = tq
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def shift_together(delta: float, settle_steps: int = 40) -> None:
        """Shift tray AND payload by the same distance along the pull axis (a probe
        that preserves their relative pose, so nothing penetrates)."""
        _, cq = cab_pose()
        d = quat_apply(cq, torch.tensor([[delta, 0.0, 0.0]], device=device))
        for body in (scene.tray, scene.payload):
            st = body.data.root_state_w[0:1].clone()
            st[:, 0:3] += d
            st[:, 7:13] = 0.0
            body.write_root_state_to_sim(st.expand(n, 13).contiguous(), all_ids)
        step(settle_steps)

    def report(tag: str) -> None:
        o = float(scene.opening()[0])
        pl = scene.cab_local(scene.payload.data.root_pos_w)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | opening={o * 1000:+.1f}mm "
              f"shut={bool(scene.shut()[0])} "
              f"pay_cab=({float(pl[0]):+.3f},{float(pl[1]):+.3f},{float(pl[2]):.3f}) "
              f"in_cav={bool(scene.in_cavity(scene.payload)[0])} "
              f"encl={bool(scene.enclosed(scene.payload)[0])} "
              f"decoy_cav={bool(scene.in_cavity(scene.decoy)[0])} "
              f"open_l={float(scene.open_latch[0]):.2f} "
              f"load_l={float(scene.load_latch[0]):.2f} "
              f"shut_l={float(scene.shut_latch[0]):.2f} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    pay_cab = scene.cab_local(scene.payload.data.root_pos_w)[0]
    fin0 = bool(torch.isfinite(scene.cabinet.data.root_state_w).all()
                and torch.isfinite(scene.tray.data.root_state_w).all()
                and torch.isfinite(scene.payload.data.root_state_w).all()
                and torch.isfinite(scene.decoy.data.root_state_w).all())
    check("settle: states finite; tray shut + seated; red cube at rest on the counter "
          "(readback height)",
          fin0 and bool(scene.shut()[0]) and bool(scene.tray_seated()[0])
          and abs(float(pay_cab[2]) - c.cube_s / 2) < 0.010 and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(5)
        cp = (scene.cabinet.data.root_pos_w - scene.env_origins)[0]
        cy = float(scene._yaw_of(scene.cabinet)[0])
        pl = scene.cab_local(scene.payload.data.root_pos_w)[0]
        dl = scene.cab_local(scene.decoy.data.root_pos_w)[0]
        reads.append((float(cp[0]), float(cp[1]), cy, float(scene.opening()[0]),
                      float(pl[0]), float(pl[1]), float(dl[0]), float(dl[1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (cab_x, cab_y, cab_yaw, opening, pay_x, "
          f"pay_y, decoy_x, decoy_y):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: cabinet xy + free yaw and the tray seating dither vary "
          "across seeded resets (readback)",
          spread[0] > 0.01 and spread[1] > 0.01 and spread[2] > 0.5
          and spread[3] > 0.0005)
    both_sides = bool((arr[:, 5] > 0).any() and (arr[:, 5] < 0).any())
    check("randomization: payload + decoy cabinet-frame xy vary and the payload spawns "
          "on BOTH sides of the porch across seeds (readback)",
          spread[4] > 0.01 and spread[5] > 0.05 and spread[6] > 0.01
          and spread[7] > 0.05 and both_sides)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy ===========================================
    # The seed's whole plan is "set the object down on top of the fixture". Here that is
    # the red cube parked ON THE ROOF of the drawer unit. The rubric must refuse it.
    env.reset(seed=41)
    step(10)
    roof_top = c.plate_top + c.inner_h + c.roof_t
    place_in_cab(scene.payload, -0.09, 0.0, roof_top + c.cube_s / 2 + 0.003,
                 settle_steps=60)
    report("seed-strategy")
    s, ok = judge()
    pz = float(scene.cab_local(scene.payload.data.root_pos_w)[0, 2])
    check("seed strategy (red cube set down ON TOP of the drawer unit): verified above "
          "the roof plane, NOT success, score ~0",
          pz > roof_top - 0.005 and not bool(scene.in_cavity(scene.payload)[0])
          and not bool(scene.enclosed(scene.payload)[0]) and not ok and s <= 0.02)

    # =========================== 7. teleport into the SHUT drawer ===========================
    # The anti-teleport pathway latch: a red cube WRITTEN directly inside the shut
    # drawer's cavity (consistent, non-penetrating pose) earns nothing, because it was
    # never in the cavity while the drawer was open.
    env.reset(seed=51)
    step(10)
    tp = scene.tray.data.root_pos_w[0:1]
    tq = scene.tray.data.root_quat_w[0:1]
    w = tp + quat_apply(tq, torch.tensor(
        [[0.0, 0.0, c.tray_floor_t + c.cube_s / 2 + 0.002]], device=device))
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = w
    st[:, 3:7] = tq
    scene.payload.write_root_state_to_sim(st, all_ids)
    step(40)
    report("shut-inject")
    s, ok = judge()
    check("shut-drawer inject: cube verified IN the cavity, ENCLOSED, drawer SHUT — "
          "yet the load pathway latch stays 0 -> NOT success, score ~0",
          bool(scene.in_cavity(scene.payload)[0]) and bool(scene.enclosed(scene.payload)[0])
          and bool(scene.shut()[0]) and float(scene.load_latch[0]) < 0.5
          and not ok and s <= 0.05)

    # =========================== 8. loaded but left OPEN ====================================
    env.reset(seed=61)
    step(10)
    place_tray(c.open_target, settle_steps=40)
    drop_into_tray(scene.payload)
    report("left-open")
    s, ok = judge()
    check("open-not-closed: cube honestly dropped into the OPEN drawer (load latch set) "
          "but the drawer stays open -> NOT success, score <= 0.60",
          bool(scene.in_cavity(scene.payload)[0]) and float(scene.load_latch[0]) > 0.5
          and float(scene.opening()[0]) > c.open_gate and not ok and s <= 0.60)

    # =========================== 9. almost closed ===========================================
    shift_together(-(c.open_target - 0.030))
    report("almost-closed")
    o_now = float(scene.opening()[0])
    s, ok = judge()
    check("almost-closed near miss: tray + cube shifted together to a ~30 mm opening "
          "(> close_tol) — cube in the cavity, yet NOT success, score <= 0.85",
          bool(scene.in_cavity(scene.payload)[0]) and 0.015 < o_now < 0.055
          and not bool(scene.shut()[0]) and not ok and s <= 0.85)

    # =========================== 10. wrong object ===========================================
    env.reset(seed=71)
    step(10)
    place_tray(c.open_target, settle_steps=40)
    drop_into_tray(scene.decoy)
    shift_together_decoy = float(scene.opening()[0])
    _, cq = cab_pose()
    d = quat_apply(cq, torch.tensor([[-shift_together_decoy, 0.0, 0.0]], device=device))
    for body in (scene.tray, scene.decoy):
        stt = body.data.root_state_w[0:1].clone()
        stt[:, 0:3] += d
        stt[:, 7:13] = 0.0
        body.write_root_state_to_sim(stt.expand(n, 13).contiguous(), all_ids)
    step(60)
    report("wrong-object")
    s, ok = judge()
    check("wrong object: the BLUE decoy loaded and the drawer shut around it (verified) "
          "— success remains False (identity + decoy-out constraint)",
          bool(scene.in_cavity(scene.decoy)[0]) and bool(scene.shut()[0]) and not ok)

    # =========================== 11. porch dump =============================================
    env.reset(seed=81)
    step(10)
    place_in_cab(scene.payload, 0.10, 0.0, c.plate_top + c.cube_s / 2 + 0.003,
                 settle_steps=60)
    report("porch-dump")
    s, ok = judge()
    check("porch dump: red cube left on the porch in front of the shut drawer — "
          "NOT success, score ~0",
          not bool(scene.in_cavity(scene.payload)[0])
          and not bool(scene.enclosed(scene.payload)[0]) and not ok and s <= 0.02)

    # =========================== 12. latched credit survives regression =====================
    env.reset(seed=91)
    step(10)
    place_tray(c.open_target, settle_steps=40)
    drop_into_tray(scene.payload)
    report("loaded")
    s_in, _ = judge()
    place_in_cab(scene.payload, 0.24, 0.20, c.cube_s / 2 + 0.003, settle_steps=60)
    report("cube-removed")
    s_out, ok = judge()
    check("latched credit: removing the loaded cube leaves the latched open + load "
          "score unchanged (and success is gone)",
          s_in >= 0.50 and abs(s_out - s_in) < 0.02 and not ok
          and not bool(scene.in_cavity(scene.payload)[0]))

    # =========================== 13. opening monotonicity ===================================
    env.reset(seed=101)
    step(10)
    place_tray(0.050, settle_steps=10)
    a_half = float(scene.open_latch[0])
    place_tray(0.110, settle_steps=10)
    a_wide = float(scene.open_latch[0])
    check("monotonicity: a wider opening latches strictly more opening credit "
          f"({a_half:.3f} < {a_wide:.3f})", a_half + 0.10 < a_wide)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.cabinet.data.root_state_w).all()
           and torch.isfinite(scene.tray.data.root_state_w).all()
           and torch.isfinite(scene.payload.data.root_state_w).all()
           and torch.isfinite(scene.decoy.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.matchbox_drawer")
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
    except BaseException as exc:  # noqa: BLE001 — die loudly, don't wait for the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL (exception: {exc!r})", flush=True)
        os._exit(2)
