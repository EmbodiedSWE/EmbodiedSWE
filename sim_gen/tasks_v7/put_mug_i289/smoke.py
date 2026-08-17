"""Smoke / rubric-REJECTION battery for MugTrapScene (sim_gen task
`put_mug_i289`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — flip-hover the mug over the ball, gravity-drop
capture, force-herd the captive ball into the disc, hands-off settle — is the
acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport
here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled
state and asserts the rubric REJECTS it; no probe in this battery ever reaches
success(), and a final audit check asserts exactly that.

  1-2.  settle/no-NaN  — reset layout settles finite: disc in its band, ball on
                         the floor >= min_sep from the disc, mug upright at rest;
                         score ~0, no success;
  3-4.  randomization  — READBACK over 6 seeded resets: disc BODY pose + ball xy
                         vary with the separation constraint holding; mug xy +
                         free yaw vary, its spawn side FLIPS, never on the ball;
  5.    null policy    — 240 idle steps -> score ~0, no success;
  6.    seed strategy  — the seed's outcome (carry the mug and set it down
                         UPRIGHT centred on the target) earns NOTHING;
  7.    bare delivery  — the ball alone settled at the disc centre, uncovered:
                         score ~0, no success;
  8.    capture-only   — a real hover-drop capture FAR from the disc: flip +
                         captured latches (~0.30), no success;
  9.    wrong topology — mug UPRIGHT in the disc with the ball settled INSIDE the
                         cup (the sibling receptacle end state): score ~0, no
                         success — the mug must be mouth-DOWN over the ball;
  10.   beside the rim — inverted mug resting centred in the disc, ball settled
                         just OUTSIDE its rim yet still inside the disc: flip
                         latch only (~0.10), no success;
  11.   near miss      — a real capture with the ball just OUTSIDE the disc edge:
                         captured fires (~0.30) but delivered does NOT, no
                         success;
  12.   fly-through    — the exact success geometry teleported in cold (covered
                         ball at the disc centre, zero velocity) judged WITHOUT
                         stepping -> NOT success (pose-jump guard);
  13.   fly-through B  — a few real frames after the teleport (streak << full),
                         then the ball yanked out: success never fired at any
                         point, latched credit persists (~0.60), no success;
  14.   held aloft     — covered xy geometry with the mug held HIGH above the
                         ball (z readback): rest-band clause rejects it cold;
  15.   ball on top    — ball placed on the upturned mug's base in the disc:
                         on-floor clause rejects it; no captured/delivered latch;
  16.   rejection audit— success() was never True at ANY judged point;
  17.   final no-NaN   — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.put_mug_i289.smoke --headless
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)",
                                             flush=True), os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.mug_trap")().build(num_envs=args.num_envs,
                                              device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.35, -1.15, 0.95)) + o),
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
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def report(tag: str) -> None:
        s, ok = judge()
        mp = (scene.mug.data.root_pos_w - scene.env_origins)[0]
        bp = (scene.ball.data.root_pos_w - scene.env_origins)[0]
        print(f"[smoke] {tag:14s} | mug_z={float(mp[2]):.3f} "
              f"up_z={float(scene._mug_up_z()[0]):+.3f} "
              f"ball=({float(bp[0]):+.3f},{float(bp[1]):+.3f},"
              f"{float(bp[2]):.3f}) d_zone={float(scene.ball_zone_dist()[0]):.3f} "
              f"flip={bool(scene.flipped_now()[0])} "
              f"cov={bool(scene.covered_now()[0])} "
              f"zone={bool(scene.in_zone_now()[0])} still={int(scene._still[0])} "
              f"latch(f/c/d)={int(scene._flipped[0])}/{int(scene._captured[0])}/"
              f"{int(scene._delivered[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def write_state(obj, pos_w: torch.Tensor, quat: torch.Tensor,
                    settle_steps: int = 0) -> None:
        """One root-state write at a WORLD pose (already env-origin absolute)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w.unsqueeze(0)
        st[:, 3:7] = quat.unsqueeze(0)
        obj.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    q_inv = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)  # pi about x

    def env_pt(x: float, y: float, z: float) -> torch.Tensor:
        return torch.tensor([x, y, z], device=device) + scene.env_origins[0]

    def zone_pt(dx: float, dy: float, z: float) -> torch.Tensor:
        zc = scene._zone_xy[0]
        return env_pt(float(zc[0]) + dx, float(zc[1]) + dy, z)

    def cap_drop(settle: int = 120) -> None:
        """The solve's capture: inverted hover over the ball's LIVE position,
        gravity drop (a real contact capture, streak earned on real frames)."""
        bp = (scene.ball.data.root_pos_w - scene.env_origins)[0]
        write_state(scene.mug,
                    env_pt(float(bp[0]), float(bp[1]), c.inv_rest_z + 0.030),
                    q_inv, settle_steps=settle)

    def settle_until_still(max_chunks: int = 20, chunk: int = 30) -> None:
        for _ in range(max_chunks):
            step(chunk)
            if int(scene._still[0]) >= c.still_steps:
                break

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.zone.data.root_state_w).all()
            and torch.isfinite(scene.mug.data.root_state_w).all()
            and torch.isfinite(scene.ball.data.root_state_w).all())
    zc0 = scene._zone_xy[0]
    bp0 = (scene.ball.data.root_pos_w - scene.env_origins)[0]
    mp0 = (scene.mug.data.root_pos_w - scene.env_origins)[0]
    check("settle: states finite, disc inside its band, ball on the floor at "
          "least min_sep from the disc, mug upright at rest (readback)",
          bool(fin0)
          and c.zone_x_range[0] - 0.005 <= float(zc0[0]) <= c.zone_x_range[1] + 0.005
          and abs(float(zc0[1])) <= c.zone_y_amp + 0.005
          and abs(float(bp0[2]) - c.ball_r) < 0.010
          and float(scene.ball_zone_dist()[0]) >= c.ball_zone_min_sep - 0.02
          and float(scene._mug_up_z()[0]) > 0.95
          and abs(float(mp0[2]) - c.body_h / 2) < 0.012
          and float(scene.mug.data.root_lin_vel_w[0].norm()) < c.settle_lin_mug
          and float(scene.ball.data.root_lin_vel_w[0].norm()) < c.settle_lin_ball)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        zc1 = scene._zone_xy[0]
        zb1 = (scene.zone.data.root_pos_w - scene.env_origins)[0]
        bp1 = (scene.ball.data.root_pos_w - scene.env_origins)[0]
        mp1 = (scene.mug.data.root_pos_w - scene.env_origins)[0]
        mq1 = scene.mug.data.root_quat_w[0]
        myaw1 = 2.0 * math.atan2(float(mq1[3]), float(mq1[0]))
        reads.append((float(zc1[0]), float(zc1[1]), float(zb1[0]), float(zb1[1]),
                      float(bp1[0]), float(bp1[1]),
                      float(scene.ball_zone_dist()[0]),
                      float(mp1[0]), float(mp1[1]), myaw1,
                      float((mp1[:2] - bp1[:2]).norm())))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (zone_xy, zone_body_xy, ball_xy, "
          f"d_ball_zone, mug_xy, mug_yaw, d_mug_ball):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: disc centre varies across seeded resets, the disc BODY "
          "really moves with it (readback), and the ball-disc separation holds "
          "every seed",
          spread[0] > 0.015 and spread[1] > 0.03
          and abs(spread[0] - spread[2]) < 0.01 and abs(spread[1] - spread[3]) < 0.01
          and spread[4] > 0.015 and spread[5] > 0.03
          and float(arr[:, 6].min()) >= c.ball_zone_min_sep - 0.02)
    check("randomization: mug xy + free yaw vary, the mug's spawn side FLIPS "
          "across seeds, and the mug never spawns within 15 cm of the ball",
          spread[7] > 0.015 and spread[9] > 0.3
          and float(arr[:, 8].min()) < -0.05 and float(arr[:, 8].max()) > 0.05
          and float(arr[:, 10].min()) >= 0.15)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. seed strategy: carry & set down =========================
    # The seed's whole plan — carry the mug and set it down UPRIGHT on the target
    # marking — earns nothing here: the mug is a tool, not the payload.
    torch.manual_seed(41)
    env.reset()
    step(10)
    write_state(scene.mug, zone_pt(0.0, 0.0, c.body_h / 2 + 0.002), ident,
                settle_steps=120)
    report("set-down")
    s, ok = judge()
    check("seed strategy: mug carried and set down UPRIGHT centred on the disc "
          "(the seed's put-on-target outcome) earns nothing — no success, "
          "score <= 0.02",
          float(scene._mug_up_z()[0]) > 0.95
          and float((scene.mug.data.root_pos_w - scene.env_origins)[0, :2]
                    .sub(scene._zone_xy[0]).norm()) < 0.02
          and not ok and s <= 0.02)

    # =========================== 7. bare delivery: ball alone in the disc ===================
    torch.manual_seed(51)
    env.reset()
    step(10)
    write_state(scene.ball, zone_pt(0.0, 0.0, c.ball_r + 0.002), ident,
                settle_steps=0)
    settle_until_still()
    report("bare-delivery")
    s, ok = judge()
    check("bare delivery: the ball settled at the disc centre UNCOVERED (readback: "
          "in zone, genuinely still) scores ~0 and is not success",
          bool(scene.in_zone_now()[0]) and int(scene._still[0]) >= c.still_steps
          and not ok and s <= 0.02)

    # =========================== 8. capture-only, far from the disc =========================
    torch.manual_seed(61)
    env.reset()
    step(10)
    cap_drop(settle=120)
    report("capture-only")
    s, ok = judge()
    check("capture-only: a real hover-drop capture far from the disc (readback: "
          "covered, settled) fires flip+captured (~0.30) but NOT delivered, no "
          "success",
          bool(scene.covered_now()[0]) and bool(scene._captured[0])
          and not bool(scene._delivered[0])
          and float(scene.ball_zone_dist()[0]) > c.zone_r + 0.05
          and not ok and 0.28 <= s <= 0.32)

    # =========================== 9. wrong topology: upright mug, ball inside ================
    # The sibling-receptacle end state: ball INSIDE the UPRIGHT mug standing in
    # the disc. Here the mug must be mouth-DOWN over the ball — rejected.
    torch.manual_seed(71)
    env.reset()
    step(10)
    write_state(scene.mug, zone_pt(0.0, 0.0, c.body_h / 2 + 0.002), ident)
    mugp = scene.mug.data.root_pos_w[0]
    in_cup = mugp + torch.tensor(
        [0.0, 0.0, -c.body_h / 2 + c.floor_t + c.ball_r + 0.002], device=device)
    write_state(scene.ball, in_cup, ident)
    settle_until_still()
    report("upright-cup")
    s, ok = judge()
    bl = (scene.ball.data.root_pos_w - scene.mug.data.root_pos_w)[0]
    check("wrong topology: ball settled INSIDE the UPRIGHT mug standing in the "
          "disc (readback: ball in the cup, in zone, still) is NOT success and "
          "scores ~0 — mouth-down cover required",
          float(scene._mug_up_z()[0]) > 0.95
          and float(bl[:2].norm()) < c.cover_r and float(bl[2]) < 0.0
          and bool(scene.in_zone_now()[0])
          and int(scene._still[0]) >= c.still_steps
          and not ok and s <= 0.02)

    # =========================== 10. beside the rim, both in the disc =======================
    torch.manual_seed(81)
    env.reset()
    step(10)
    # Geometry: contact-free needs axis separation >= body_r + ball_r = 63 mm.
    # Mug nudged +x inside the disc, ball on the handle-free -x side (the handle
    # points along +x under q_inv): separation 70 mm (7 mm wall clearance), ball
    # only 55 mm from the disc centre (20 mm inside the zone edge).
    write_state(scene.mug, zone_pt(0.015, 0.0, c.inv_rest_z + 0.002), q_inv)
    write_state(scene.ball, zone_pt(-0.055, 0.0, c.ball_r + 0.002), ident)
    settle_until_still()
    report("beside-rim")
    s, ok = judge()
    d_ax = float((scene.ball.data.root_pos_w[:, :2]
                  - scene.mug.data.root_pos_w[:, :2]).norm(dim=-1)[0])
    check("beside the rim: inverted mug resting centred in the disc, ball settled "
          "just OUTSIDE its rim yet inside the disc (readback) — flip latch only "
          "(~0.10), no success",
          bool(scene.flipped_now()[0]) and bool(scene.in_zone_now()[0])
          and d_ax > c.cover_r and not bool(scene.covered_now()[0])
          and int(scene._still[0]) >= c.still_steps
          and not ok and 0.09 <= s <= 0.12)

    # =========================== 11. near miss: covered just outside the disc ===============
    torch.manual_seed(91)
    env.reset()
    step(10)
    write_state(scene.ball, zone_pt(c.zone_r + 0.035, 0.0, c.ball_r + 0.002),
                ident, settle_steps=30)
    cap_drop(settle=120)
    report("near-miss")
    s, ok = judge()
    check("near miss: a real capture with the ball settled just OUTSIDE the disc "
          "edge (readback) — captured fires (~0.30) but delivered does NOT, no "
          "success",
          bool(scene.covered_now()[0]) and bool(scene._captured[0])
          and not bool(scene._delivered[0])
          and c.zone_r < float(scene.ball_zone_dist()[0]) < c.zone_r + 0.06
          and not ok and 0.28 <= s <= 0.32)

    # =========================== 12-13. fly-through / teleport guard ========================
    torch.manual_seed(101)
    env.reset()
    step(60)  # build an (irrelevant) stillness streak first
    write_state(scene.ball, zone_pt(0.0, 0.0, c.ball_r + 0.001), ident)
    write_state(scene.mug, zone_pt(0.0, 0.0, c.inv_rest_z + 0.001), q_inv)
    s0, ok0 = judge()
    report("fly-through")
    check("fly-through: the exact success geometry (covered ball at the disc "
          "centre, zero velocity) teleported in and judged WITHOUT stepping is "
          "NOT success (pose-jump guard)",
          bool(scene.covered_now()[0]) and bool(scene.in_zone_now()[0])
          and not ok0)
    mid_ok = False
    # 70 real frames: the teleport keeps the drift WINDOW failing for the first
    # `still_steps` (45) of them, so the streak only reaches ~25 — enough for
    # the latch gate (10) to fire honestly, never enough for success (45).
    for _ in range(70):
        step(1)
        mid_ok = mid_ok or bool(scene.success()[0])
    ever_success[0] = ever_success[0] or mid_ok
    write_state(scene.ball, env_pt(0.10, 0.0, c.ball_r + 0.001), ident,
                settle_steps=90)
    report("yank-out")
    s, ok = judge()
    check("fly-through B: a few real frames after the teleport, then the ball "
          "yanked back out — success never fired at any point, latched credit "
          "persists (~0.60), no success",
          not mid_ok and not ok and 0.55 <= s <= 0.65
          and bool(scene._delivered[0]) and not bool(scene.covered_now()[0]))

    # =========================== 14. held-aloft cover =======================================
    torch.manual_seed(111)
    env.reset()
    step(10)
    write_state(scene.ball, zone_pt(0.0, 0.0, c.ball_r + 0.002), ident,
                settle_steps=30)
    write_state(scene.mug, zone_pt(0.0, 0.0, 0.20), q_inv)  # held high, no step
    s, ok = judge()
    report("held-aloft")
    check("held aloft: covered xy geometry with the mug held HIGH above the ball "
          "(z readback ~0.20) is rejected cold — not resting, no cover, no "
          "success, no fresh latch",
          float((scene.mug.data.root_pos_w - scene.env_origins)[0, 2]) > 0.15
          and not bool(scene.flipped_now()[0])
          and not bool(scene.covered_now()[0])
          and not ok and s <= 0.02)

    # =========================== 15. ball on top of the upturned mug ========================
    torch.manual_seed(121)
    env.reset()
    step(10)
    write_state(scene.mug, zone_pt(0.0, 0.0, c.inv_rest_z + 0.002), q_inv,
                settle_steps=60)
    top = scene.mug.data.root_pos_w[0] + torch.tensor(
        [0.0, 0.0, c.body_h / 2 + c.ball_r + 0.002], device=device)
    write_state(scene.ball, top, ident)
    settle_until_still()
    report("ball-on-top")
    s, ok = judge()
    bz = float((scene.ball.data.root_pos_w - scene.env_origins)[0, 2])
    d_ax = float((scene.ball.data.root_pos_w[:, :2]
                  - scene.mug.data.root_pos_w[:, :2]).norm(dim=-1)[0])
    check("ball on top: ball placed on the upturned mug's base in the disc — "
          "either it rests up there (on-floor clause) or it rolls off outside "
          "the rim (readback); no captured/delivered latch, no success, "
          "score <= 0.12",
          (bz > c.ball_r + c.ball_ground_tol or d_ax > c.cover_r)
          and not bool(scene.covered_now()[0]) and not bool(scene._captured[0])
          and not bool(scene._delivered[0]) and not ok and s <= 0.12)

    # =========================== 16-17. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = (torch.isfinite(scene.zone.data.root_state_w).all()
           and torch.isfinite(scene.mug.data.root_state_w).all()
           and torch.isfinite(scene.ball.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.mug_trap")
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
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
