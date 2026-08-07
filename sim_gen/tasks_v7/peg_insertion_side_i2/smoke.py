"""Smoke / rubric-REJECTION battery for HaspPinScene (sim_gen task
`peg_insertion_side_i2`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — seat the bar under gravity, thread the pin under a
floating-hand force controller — is the acceptance evidence that the rubric ACCEPTS a
correct outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partial) outcome and asserts the rubric REJECTS it; no probe in this battery ever
reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: bar flat on the floor, pin and
                            decoy lying on their sides; score ~0 at rest, no success;
  3-4. randomization      — READBACK over 6 seeded resets: stand xy + yaw, bar xy + yaw,
                            pin xy, and the approach baseline d0 all move;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy /     — the seed's whole plan ("put the peg in the fixed block's
      out-of-order          hole, done") = pin dropped into the BARE bore first: head
                            lands captive on the plate, shaft in the bore -> NOT
                            success, NOT linked, score ~0 (order latch: no pin credit
                            before the bar seats);
  7.  near-miss seat      — bar laid on the plate 14 mm off the bore axis (align_tol is
                            10 mm), settled -> NOT seated, NOT success, score <= 0.20;
  8.  pin ON TOP          — bar correctly seated, blue pin lying HORIZONTALLY on top of
                            it, settled -> NOT linked (not through anything), score
                            <= 0.45 (approach + seat credit only);
  9.  wrong object        — bar correctly seated, RED decoy offered to the hole: its
                            30 mm shaft cannot enter -> NOT linked, NOT success,
                            score <= 0.45 (identity matters);
  10. axis-line loophole  — pin stood upright on the GROUND in the gap under the plate,
                            axis passing through both aligned holes -> _linked is False
                            (the head-seated clause rejects it: nothing carries the pin);
  11. latched credit      — seating the bar then removing it leaves the latched
                            approach + seat score unchanged (and success is gone);
  12. monotonicity        — moving the bar closer to the bore axis latches strictly
                            more approach credit than a farther placement;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.peg_insertion_side_i2.smoke --headless
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
    env = ENVS.get("simgen.hasp_pin_link")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.85, -0.75, 0.65)) + o),
                                tuple(np.array((0.02, 0.05, 0.06)) + o),
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

    def stand_pose() -> tuple[torch.Tensor, float]:
        sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
        q = scene.stand.data.root_quat_w[0]
        return sp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def report(tag: str) -> None:
        bl = scene._stand_local(scene.bar.data.root_pos_w)[0]
        _, tip_w, head_w = scene._pin_ends_w(scene.pin)
        tip_z = float((tip_w - scene.env_origins)[0, 2])
        head_z = float((head_w - scene.env_origins)[0, 2])
        s, ok = judge()
        print(f"[smoke] {tag:16s} | bar_local=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) seated={bool(scene.bar_seated()[0])} "
              f"tip_z={tip_z:.3f} head_z={head_z:.3f} linked={bool(scene.pin_linked()[0])} "
              f"appr={float(scene.approach_latch[0]):.3f} "
              f"seat={float(scene.seat_latch[0]):.3f} pin={float(scene.pin_latch[0]):.3f} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_body(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0),
                   settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def lying_quat(yaw: float) -> tuple[float, float, float, float]:
        """q = qz(yaw) * qy(90 deg): pin local +z -> horizontal."""
        c45 = math.cos(math.pi / 4)
        cy2, sy2 = math.cos(yaw / 2), math.sin(yaw / 2)
        return (cy2 * c45, -sy2 * c45, cy2 * c45, sy2 * c45)

    def seat_bar(offset_x_local: float = 0.0) -> None:
        """Hover the bar over the bore axis (optionally offset along stand local x)
        and let gravity seat it — the same honest seat solve.py performs."""
        sp, syaw = stand_pose()
        ca, sa = math.cos(syaw), math.sin(syaw)
        place_body(scene.bar, float(sp[0]) + ca * offset_x_local,
                   float(sp[1]) + sa * offset_x_local,
                   c.plate_top + 0.008 + c.bar_t / 2, quat=yaw_quat(syaw),
                   settle_steps=120)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    bar_z = float((scene.bar.data.root_pos_w - scene.env_origins)[0, 2])
    pin_z = float((scene.pin.data.root_pos_w - scene.env_origins)[0, 2])
    fin0 = bool(torch.isfinite(scene.bar.data.root_state_w).all()
                and torch.isfinite(scene.pin.data.root_state_w).all()
                and torch.isfinite(scene.decoy.data.root_state_w).all())
    check("settle: states finite; bar flat on the floor and pin lying on its side "
          "(readback heights)",
          fin0 and abs(bar_z - c.bar_t / 2) < 0.008 and abs(pin_z - c.pin_head_r) < 0.010
          and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        sp, syaw = stand_pose()
        b = (scene.bar.data.root_pos_w - scene.env_origins)[0]
        qb = scene.bar.data.root_quat_w[0]
        byaw = 2.0 * math.atan2(float(qb[3]), float(qb[0]))
        p = (scene.pin.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(sp[0]), float(sp[1]), syaw, float(b[0]), float(b[1]), byaw,
                      float(p[0]), float(p[1]), float(scene.d0[0])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (stand_x, stand_y, stand_yaw, bar_x, bar_y, "
          f"bar_yaw, pin_x, pin_y, d0):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: stand xy + yaw vary across seeded resets (readback)",
          spread[0] > 0.005 and spread[1] > 0.005 and spread[2] > 0.2)
    check("randomization: bar xy + yaw, pin xy, and the approach baseline d0 vary "
          "across seeded resets (readback)",
          spread[3] > 0.005 and spread[4] > 0.005 and spread[5] > 0.2
          and spread[6] > 0.005 and spread[7] > 0.005 and spread[8] > 0.005)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy / out-of-order ============================
    # The seed's whole plan is "put the peg into the fixed block's hole — done". Here
    # that is exactly the out-of-order end state: pin dropped into the BARE platform
    # bore first. Its head lands captive on the plate, its shaft hangs in the bore, the
    # bar never moved. The rubric must refuse it entirely (order latch: pin credit is
    # gated on the bar being seated).
    env.reset(seed=41)
    step(10)
    sp, _syaw = stand_pose()
    place_body(scene.pin, float(sp[0]), float(sp[1]),
               c.plate_top + 0.005 + c.pin_half, settle_steps=120)
    report("seed-strategy")
    s, ok = judge()
    _, tip_w, head_w = scene._pin_ends_w(scene.pin)
    tip_loc = scene._stand_local(tip_w)[0]
    head_z = float((head_w - scene.env_origins)[0, 2])
    in_bore_xy = abs(float(tip_loc[0])) < c.tip_xy_tol and abs(float(tip_loc[1])) < c.tip_xy_tol
    check("seed strategy / out-of-order (pin into the BARE bore, bar untouched): pin "
          "verified in the bore with head captive on the plate, yet NOT linked, NOT "
          "success, score <= 0.05",
          in_bore_xy and head_z < c.eye_top_seated - c.head_seat_low
          and not bool(scene.pin_linked()[0]) and not ok and s <= 0.05)

    # =========================== 7. near-miss seat ==========================================
    env.reset(seed=51)
    step(10)
    seat_bar(offset_x_local=0.014)  # align_tol is 0.010
    report("near-miss-seat")
    s, ok = judge()
    bl = scene._stand_local(scene.bar.data.root_pos_w)[0]
    off = max(abs(float(bl[0])), abs(float(bl[1])))
    check("near-miss: bar resting ON the plate but 14 mm off the bore axis — NOT "
          "seated, NOT success, score <= 0.20",
          off > c.align_tol and not bool(scene.bar_seated()[0]) and not ok and s <= 0.20)

    # =========================== 8. pin ON TOP of the seated bar ============================
    env.reset(seed=61)
    step(10)
    seat_bar()
    sp, syaw = stand_pose()
    place_body(scene.pin, float(sp[0]), float(sp[1]),
               c.eye_top_seated + c.pin_head_r + 0.002, quat=lying_quat(syaw + math.pi / 2),
               settle_steps=120)
    report("pin-on-top")
    s, ok = judge()
    check("pin resting HORIZONTALLY on top of the correctly seated bar: NOT linked, "
          "NOT success, score <= 0.45 (approach + seat credit only)",
          bool(scene.bar_seated()[0]) and not bool(scene.pin_linked()[0]) and not ok
          and s <= 0.45)

    # =========================== 9. wrong object ============================================
    env.reset(seed=71)
    step(10)
    seat_bar()
    sp, _syaw = stand_pose()
    place_body(scene.decoy, float(sp[0]), float(sp[1]),
               c.eye_top_seated + 0.004 + c.pin_half, settle_steps=120)
    report("wrong-object")
    s, ok = judge()
    dec_tip_z = float((scene._pin_ends_w(scene.decoy)[1] - scene.env_origins)[0, 2])
    check("wrong object: the RED decoy offered to the aligned holes cannot enter (its "
          "tip never passes the eye), is NOT linked, NOT success, score <= 0.45",
          dec_tip_z > c.plate_top - 0.005 and not bool(scene._linked(scene.decoy)[0])
          and not ok and s <= 0.45)

    # =========================== 10. axis-line loophole probe ===============================
    # A pin STANDING ON THE GROUND in the open gap under the plate has its axis passing
    # through both aligned holes — an infinite-line containment test would accept it.
    # Our head-seated clause must reject it: nothing carries the pin. (The pose
    # interpenetrates the plate underside — this is a transient judged probe, not a
    # settled outcome; judged after 2 real steps, then discarded.)
    place_body(scene.decoy, float(sp[0]) + 0.30, float(sp[1]), c.decoy_head_r + 0.002,
               quat=lying_quat(0.0), settle_steps=10)  # decoy out of the way first
    place_body(scene.pin, float(sp[0]), float(sp[1]), 0.001 + c.pin_half, settle_steps=2)
    report("underworld")
    _, tip_w, head_w = scene._pin_ends_w(scene.pin)
    tip_loc = scene._stand_local(tip_w)[0]
    head_z = float((head_w - scene.env_origins)[0, 2])
    on_axis = abs(float(tip_loc[0])) < c.tip_xy_tol and abs(float(tip_loc[1])) < c.tip_xy_tol
    s, ok = judge()
    check("axis-line loophole: pin standing on the ground UNDER the plate, axis "
          "through both holes — head not seated => NOT linked, NOT success",
          on_axis and head_z < c.eye_top_seated - c.head_seat_low
          and not bool(scene.pin_linked()[0]) and not ok)

    # =========================== 11. latched credit survives regression =====================
    env.reset(seed=81)
    step(10)
    seat_bar()
    report("seated")
    s_in, _ = judge()
    place_body(scene.bar, -0.24, -0.12, c.bar_t / 2 + 0.002, settle_steps=60)
    report("bar-removed")
    s_out, ok = judge()
    check("latched credit: removing the seated bar leaves the latched approach + seat "
          "score unchanged (and success is gone)",
          s_in >= 0.35 and abs(s_out - s_in) < 0.02 and not ok
          and not bool(scene.bar_seated()[0]))

    # =========================== 12. approach monotonicity ==================================
    env.reset(seed=91)
    step(5)
    sp, _syaw = stand_pose()
    b0 = (scene.bar.data.root_pos_w - scene.env_origins)[0]
    dx, dy = float(sp[0] - b0[0]), float(sp[1] - b0[1])
    place_body(scene.bar, float(b0[0]) + 0.5 * dx, float(b0[1]) + 0.5 * dy, 0.25,
               settle_steps=3)
    a_half = float(scene.approach_latch[0])
    place_body(scene.bar, float(b0[0]) + 0.85 * dx, float(b0[1]) + 0.85 * dy, 0.25,
               settle_steps=3)
    a_near = float(scene.approach_latch[0])
    check("monotonicity: moving the bar closer to the bore axis latches strictly more "
          f"approach credit ({a_half:.3f} < {a_near:.3f})", a_half + 0.10 < a_near)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.stand.data.root_state_w).all()
           and torch.isfinite(scene.bar.data.root_state_w).all()
           and torch.isfinite(scene.pin.data.root_state_w).all()
           and torch.isfinite(scene.decoy.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.hasp_pin_link")
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
