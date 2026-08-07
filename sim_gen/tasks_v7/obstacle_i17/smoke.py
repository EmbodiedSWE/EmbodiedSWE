"""Smoke / rubric-REJECTION battery for SkittleGalleryScene (sim_gen task
`obstacle_i17`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — stage the ball, floating-hand throw down the red
lane, ballistic tunnel shot — is the acceptance evidence that the rubric ACCEPTS a
correct outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partial) outcome and asserts the rubric REJECTS it; no probe in this battery ever
reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: both pins upright at standing
                            height, ball and decoy resting on the open floor; score ~0
                            at rest, no success;
  3-4. randomization      — READBACK over 6 seeded resets: gallery xy + yaw move, the
                            RED lane takes BOTH sides, ball/decoy scatter and the
                            approach baseline d0 all move;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's whole plan ("carry the payload over the wall,
                            set it down at the goal") = ball SET DOWN at rest beside
                            the standing red pin inside the cell: ball verified in the
                            red cell, yet NOT success (no pin was felled), score <= 0.55
                            (transport alone earns at most partial credit);
  7.  wrong lane          — the WHITE pin felled while red stands = the unrecoverable
                            failure: NOT success;
  8.  both pins down      — red genuinely down (red_down() True) but white down too:
                            the white-standing clause alone must reject it;
  9.  chamber clause      — red pin lying flat and low but OUTSIDE the gallery (open
                            floor): red_down() False — a pin is only "felled" in its
                            own cell;
  10. wrong object        — the ORANGE decoy driven at the red mouth at 1.5 m/s: too
                            big for funnel/tunnel, never enters the chamber, the pin
                            behind is untouched -> no success, no meaningful score;
  11. latched credit      — ball delivered into the red cell then removed: the latched
                            approach + entry score survives unchanged (and no success);
  12. monotonicity        — moving the ball closer to the red mouth latches strictly
                            more approach credit than a farther placement;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.obstacle_i17.smoke --headless
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
    env = ENVS.get("simgen.skittle_gallery")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.00, -1.05, 0.70)) + o),
                                tuple(np.array((0.00, 0.05, 0.08)) + o),
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

    def gallery_pose() -> tuple[torch.Tensor, float]:
        fp = (scene.gallery.data.root_pos_w - scene.env_origins)[0]
        q = scene.gallery.data.root_quat_w[0]
        return fp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def to_world(lx: float, ly: float) -> tuple[float, float]:
        fp, yaw = gallery_pose()
        cy, sy = math.cos(yaw), math.sin(yaw)
        return (float(fp[0]) + lx * cy - ly * sy, float(fp[1]) + lx * sy + ly * cy)

    def report(tag: str) -> None:
        bl = scene._fix_local(scene.ball.data.root_pos_w)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | ball_local=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) red_upz={float(scene._up_z(scene.red)[0]):+.3f} "
              f"red_z={float(scene._z_rel(scene.red)[0]):.3f} "
              f"white_upz={float(scene._up_z(scene.white)[0]):+.3f} "
              f"down={bool(scene.red_down()[0])} stand={bool(scene.white_standing()[0])} "
              f"in_cell={bool(scene.ball_in_red_cell()[0])} "
              f"appr={float(scene.appr_latch[0]):.3f} pass={float(scene.pass_latch[0]):.3f} "
              f"tilt={float(scene.tilt_latch[0]):.3f} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_local(body, lx: float, ly: float, z: float,
                    quat=(1.0, 0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0),
                    settle_steps: int = 30) -> None:
        """Kinematic probe placement in GALLERY-LOCAL xy (instrumentation, not a
        solution) + REAL physics steps before judging (the zero-step trap)."""
        wx, wy = to_world(lx, ly)
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = wx, wy, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 7], st[:, 8], st[:, 9] = vel
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def lying_quat(world_yaw: float) -> tuple[float, float, float, float]:
        """q = qz(yaw) * qy(90 deg): pin local +z -> horizontal (lying flat)."""
        c45 = math.cos(math.pi / 4)
        cy2, sy2 = math.cos(world_yaw / 2), math.sin(world_yaw / 2)
        return (cy2 * c45, -sy2 * c45, cy2 * c45, sy2 * c45)

    def fin_all() -> bool:
        return bool(torch.isfinite(scene.gallery.data.root_state_w).all()
                    and torch.isfinite(scene.red.data.root_state_w).all()
                    and torch.isfinite(scene.white.data.root_state_w).all()
                    and torch.isfinite(scene.ball.data.root_state_w).all()
                    and torch.isfinite(scene.decoy.data.root_state_w).all())

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    ball_z = float(scene._z_rel(scene.ball)[0])
    red_z = float(scene._z_rel(scene.red)[0])
    white_z = float(scene._z_rel(scene.white)[0])
    check("settle: states finite; both pins upright at standing height, ball resting "
          "on the floor (readback)",
          fin_all() and abs(ball_z - c.ball_r) < 0.008
          and abs(red_z - c.pin_z0) < 0.010 and abs(white_z - c.pin_z0) < 0.010
          and float(scene._up_z(scene.red)[0]) > 0.99
          and float(scene._up_z(scene.white)[0]) > 0.99 and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        fp, fyaw = gallery_pose()
        bl = scene._fix_local(scene.ball.data.root_pos_w)[0]
        dl = scene._fix_local(scene.decoy.data.root_pos_w)[0]
        reads.append((float(fp[0]), float(fp[1]), fyaw, float(scene.red_side[0]),
                      float(bl[0]), float(bl[1]), float(dl[0]), float(dl[1]),
                      float(scene.d0[0])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (fix_x, fix_y, fix_yaw, red_side, ball_x, "
          f"ball_y, decoy_x, decoy_y, d0):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: gallery xy + yaw vary AND the RED lane takes both sides "
          "across seeded resets (readback)",
          spread[0] > 0.005 and spread[1] > 0.005 and spread[2] > 0.05
          and arr[:, 3].min() < -0.5 and arr[:, 3].max() > 0.5)
    check("randomization: ball xy, decoy xy, and the approach baseline d0 vary "
          "across seeded resets (readback)",
          spread[4] > 0.01 and spread[5] > 0.005 and spread[6] > 0.01
          and spread[7] > 0.005 and spread[8] > 0.005)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy ===========================================
    # The seed's whole plan is "carry the payload over the obstacle and SET IT DOWN at
    # the goal". Here that end state — the ball placed at rest beside the standing red
    # pin inside the cell (physically impossible for the embodied agent: the gallery is
    # roofed and sealed; constructed by teleport as pure instrumentation) — must NOT be
    # success: the task is about FELLING the pin, and gentle delivery fells nothing.
    env.reset(seed=41)
    step(10)
    side = float(scene.red_side[0])
    place_local(scene.ball, side * c.lane_off, c.pin_y - 0.08, c.ball_r + 0.002,
                settle_steps=120)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (ball SET DOWN at rest beside the standing red pin, in-cell "
          "delivery verified): NOT success, score <= 0.55 — transport without the "
          "strike earns at most partial credit",
          bool(scene.ball_in_red_cell()[0]) and float(scene._up_z(scene.red)[0]) > 0.95
          and not ok and s <= 0.55)

    # =========================== 7. wrong lane (white felled) ===============================
    env.reset(seed=51)
    step(10)
    side = float(scene.red_side[0])
    _fp, fyaw = gallery_pose()
    place_local(scene.white, -side * c.lane_off, c.pin_y, 0.024,
                quat=lying_quat(fyaw), settle_steps=90)
    report("wrong-lane")
    s, ok = judge()
    check("wrong lane: WHITE pin felled while red stands — the unrecoverable failure "
          "— NOT success",
          not bool(scene.white_standing()[0]) and float(scene._up_z(scene.red)[0]) > 0.95
          and not ok)

    # =========================== 8. both pins down ==========================================
    place_local(scene.red, side * c.lane_off, c.pin_y, 0.024,
                quat=lying_quat(fyaw), settle_steps=90)
    report("both-down")
    s, ok = judge()
    check("both pins down: red is GENUINELY down (red_down() True) but the "
          "white-standing clause alone rejects success",
          bool(scene.red_down()[0]) and not bool(scene.white_standing()[0]) and not ok)

    # =========================== 9. chamber clause ==========================================
    env.reset(seed=61)
    step(10)
    side = float(scene.red_side[0])
    _fp, fyaw = gallery_pose()
    place_local(scene.red, 0.0, -0.30, 0.024, quat=lying_quat(fyaw), settle_steps=90)
    report("outside-chamber")
    s, ok = judge()
    check("chamber clause: red pin lying flat and LOW but on the open floor OUTSIDE "
          "the gallery — red_down() False (a pin only counts felled in its own cell), "
          "NOT success",
          float(scene._up_z(scene.red)[0]) < 0.5
          and float(scene._z_rel(scene.red)[0]) < c.down_z
          and not bool(scene.red_down()[0]) and not ok)

    # =========================== 10. wrong object ===========================================
    env.reset(seed=71)
    step(10)
    side = float(scene.red_side[0])
    _fp, fyaw = gallery_pose()
    # park the ball far off to the side so the decoy run is unobstructed
    place_local(scene.ball, -side * 0.30, -0.75, c.ball_r + 0.002, settle_steps=20)
    fwd = (-math.sin(fyaw) * 1.5, math.cos(fyaw) * 1.5, 0.0)
    place_local(scene.decoy, side * c.lane_off, -0.45, c.decoy_r + 0.002, vel=fwd,
                settle_steps=1)
    max_y = -10.0
    for _ in range(240):
        step(1)
        max_y = max(max_y, float(scene._fix_local(scene.decoy.data.root_pos_w)[0][1]))
    report("decoy-shot")
    s, ok = judge()
    check("wrong object: ORANGE decoy driven at the red mouth at 1.5 m/s reaches the "
          "wall but NEVER enters the chamber (too big for funnel/tunnel); the pin "
          "behind is untouched, NOT success",
          max_y > -0.30 and max_y < 0.0 and not bool(scene._in_chamber(scene.decoy)[0])
          and float(scene._up_z(scene.red)[0]) > 0.95 and not ok)

    # =========================== 11. latched credit survives regression =====================
    env.reset(seed=81)
    step(10)
    side = float(scene.red_side[0])
    place_local(scene.ball, side * c.lane_off, c.pin_y - 0.08, c.ball_r + 0.002,
                settle_steps=90)
    report("delivered")
    s_in, _ = judge()
    place_local(scene.ball, 0.0, -0.80, c.ball_r + 0.002, settle_steps=60)
    report("ball-removed")
    s_out, ok = judge()
    check("latched credit: removing the delivered ball leaves the latched approach + "
          "entry score unchanged (and still no success)",
          s_in >= 0.35 and abs(s_out - s_in) < 0.02 and not ok)

    # =========================== 12. approach monotonicity ==================================
    env.reset(seed=91)
    step(5)
    side = float(scene.red_side[0])
    place_local(scene.ball, side * c.lane_off, -0.32, 0.20, settle_steps=3)
    a_far = float(scene.appr_latch[0])
    place_local(scene.ball, side * c.lane_off, -0.12, 0.20, settle_steps=3)
    a_near = float(scene.appr_latch[0])
    check("monotonicity: moving the ball closer to the red mouth latches strictly "
          f"more approach credit ({a_far:.3f} < {a_near:.3f})", a_far + 0.10 < a_near)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", fin_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.skittle_gallery")
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
