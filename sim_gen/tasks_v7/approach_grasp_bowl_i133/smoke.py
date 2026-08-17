"""Smoke / rubric-REJECTION battery for CliffSweepScene (sim_gen task
`approach_grasp_bowl_i133`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — cage the balls under the scoop dome, plow them over
the cliff, brake and let them drop into the basin — is the acceptance evidence that the
rubric ACCEPTS a correct outcome). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome and asserts the rubric REJECTS it; no probe in
this battery ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: balls at sphere height on the
                            stage, scoop resting at its dome height; score ~0, no success;
  3-4. randomization      — READBACK over 8 seeded resets: the cliff SIDE flips both
                            ways and the kinematic basin/end-rail teleports follow it;
                            ball cluster xy, scoop xy and yaw all vary; spawn pairwise
                            ball gaps respected;
  5.  null policy         — 300 idle steps -> score ~0, no success, and the balls do
                            not creep (GPU sphere-creep regression);
  6.  SEED strategy       — the seed's whole plan ("grasp the object, lift it, carry it
                            to the goal") executed on all three balls: each ball is
                            raised above `lift_z` then set down IN the basin. The end
                            state is visually complete (all balls in the basin, settled,
                            scoop parked) yet the spoil latch REJECTS it: NOT success,
                            score capped at 0.20;
  7.  beside-basin miss   — balls settled on the GROUND next to the basin (outside the
                            side wall) -> no delivery credit, NOT success;
  8.  cliff-edge miss     — balls settled on the STAGE at the cliff lip (past the red
                            stripe but not over) -> no delivery credit, NOT success;
  9.  captured-only       — scoop set down over the pack on the stage: capture latch
                            fires (score >= 0.08) but NOT success;
  10. scoop-in-basin      — all three balls delivered but the scoop dropped into the
                            basin too: delivery credit only (<= 0.85), NOT success —
                            the tool must end parked on the stage;
  11. unsettled judged    — balls IN the basin but still moving fast (velocity written)
                            with the scoop off-stage: NOT success at that instant;
  12. latched delivery    — removing a delivered ball back onto the stage leaves the
                            latched delivery credit unchanged and success stays gone;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.approach_grasp_bowl_i133.smoke --headless
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
    env = ENVS.get("simgen.cliff_sweep")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.55, -1.45, 1.10)) + o),
                                tuple(np.array((0.10, 0.00, 0.06)) + o),
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

    def bp() -> torch.Tensor:
        """(3,3) ball centres, env-local."""
        return scene._ball_pos()[0]

    def side() -> float:
        return float(scene.side[0])

    def report(tag: str) -> None:
        p = bp()
        inb = scene._in_basin(scene._ball_pos())[0]
        sp = scene._scoop_pos()[0]
        s, ok = judge()
        print(f"[smoke] {tag:18s} | balls_z=[" + ",".join(
            f"{float(p[i, 2]):.3f}" for i in range(3)) + "] in_basin=[" + ",".join(
            "T" if bool(inb[i]) else "f" for i in range(3)) + "] "
            f"scoop_z={float(sp[2]):.3f} parked={bool(scene.scoop_parked()[0])} "
            f"cap={float(scene.captured[0]):.0f} "
            f"del={float(scene.delivered[0].sum()):.0f} "
            f"spoiled={bool(scene.spoiled[0])} score={s:.3f} success={ok} "
            f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_body(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0),
                   vel=(0.0, 0.0, 0.0), settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 7], st[:, 8], st[:, 9] = vel
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    ball_rest_z = c.stage_h + c.ball_r  # ~0.140 on the stage
    basin_ball_z = c.floor_t + c.ball_r + 0.003  # ~0.031 on the basin floor

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    p = bp()
    sp = scene._scoop_pos()[0]
    fin0 = bool(torch.isfinite(scene.scoop.data.root_state_w).all()
                and all(torch.isfinite(b.data.root_state_w).all() for b in scene.balls)
                and torch.isfinite(scene.basin.data.root_state_w).all())
    z_ok = all(abs(float(p[i, 2]) - ball_rest_z) < 0.008 for i in range(3))
    check("settle: states finite; balls at sphere height on the stage and scoop at its "
          "dome resting height (readback)",
          fin0 and z_ok and abs(float(sp[2]) - c.scoop_rest_z) < 0.010)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(5)
        p = bp()
        sp = scene._scoop_pos()[0]
        q = scene.scoop.data.root_quat_w[0]
        yaw = float(torch.atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                                1 - 2 * (q[2] * q[2] + q[3] * q[3])))
        bas_x = float((scene.basin.data.root_pos_w - scene.env_origins)[0, 0])
        rail_x = float((scene.end_rail.data.root_pos_w - scene.env_origins)[0, 0])
        gap = float(min((p[i, :2] - p[j, :2]).norm()
                        for i in range(3) for j in range(i + 1, 3)))
        reads.append((side(), float(p[0, 0]), float(p[0, 1]), float(sp[0]) * side(),
                      float(sp[1]), yaw, bas_x, rail_x, gap))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (side, b0x, b0y, scoop_mx, scoop_y, yaw, "
          f"basin_x, rail_x, min_gap):\n{arr}", flush=True)
    sides = set(arr[:, 0].tolist())
    follows = all(r[6] * r[0] > 0.5 and r[7] * r[0] < -0.3 for r in reads)
    check("randomization: cliff side flips both ways across seeds and the kinematic "
          "basin + end-rail teleports follow it (readback)",
          sides == {1.0, -1.0} and follows)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: ball cluster xy, scoop xy and yaw all vary; spawn pairwise "
          "ball gaps respected (readback)",
          spread[1] > 0.02 and spread[2] > 0.02 and spread[4] > 0.03 and spread[5] > 0.5
          and float(arr[:, 8].min()) > c.ball_gap - 0.004)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(60)
    p0 = bp()[:, :2].clone()
    step(300)
    report("null-policy")
    drift = float((bp()[:, :2] - p0).norm(dim=-1).max())
    s, ok = judge()
    check("null policy: score ~0 and no success after 300 idle steps, and the balls do "
          f"not creep (max drift {drift * 1000:.1f} mm)",
          s <= 0.02 and not ok and drift < 0.03)

    # =========================== 6. SEED strategy ============================================
    # The seed's whole plan: grasp the (graspable, 40 mm) object, LIFT it, carry it to
    # the goal. Executed honestly here: each ball is raised above lift_z, then set down
    # inside the basin. The final tableau is exactly the goal picture — and the spoil
    # latch must refuse it anyway.
    env.reset(seed=41)
    step(30)
    sd = side()
    for i in range(3):
        p = bp()
        place_body(scene.balls[i], float(p[i, 0]), float(p[i, 1]), 0.30,
                   settle_steps=3)  # the lift — trips the latch
        place_body(scene.balls[i], sd * (0.45 + 0.06 * i), (i - 1) * 0.12,
                   basin_ball_z, settle_steps=40)
    step(120)
    report("seed-strategy")
    s, ok = judge()
    inb = scene._in_basin(scene._ball_pos())[0]
    tableau = bool(inb.all()) and bool(scene.scoop_parked()[0])
    check("seed strategy (each ball lifted above lift_z and carried into the basin): "
          "final tableau complete (all in basin, scoop parked) yet spoiled -> NOT "
          "success, score <= 0.20",
          tableau and bool(scene.spoiled[0]) and not ok and s <= 0.20 + 1e-5)

    # =========================== 7. beside-basin near-miss ==================================
    env.reset(seed=51)
    step(30)
    sd = side()
    for i in range(3):
        place_body(scene.balls[i], sd * (0.45 + 0.06 * i), 0.40 + 0.05 * i,
                   c.ball_r + 0.003, settle_steps=30)
    step(90)
    report("beside-basin")
    s, ok = judge()
    inb = scene._in_basin(scene._ball_pos())[0]
    check("beside-basin miss: balls settled on the ground OUTSIDE the basin side wall "
          "— no delivery credit (score <= 0.02), NOT success",
          not bool(inb.any()) and s <= 0.02 and not ok)

    # =========================== 8. cliff-edge near-miss ====================================
    env.reset(seed=61)
    step(30)
    sd = side()
    for i in range(3):
        place_body(scene.balls[i], sd * 0.35, (i - 1) * 0.10, ball_rest_z + 0.002,
                   settle_steps=30)
    step(90)
    report("cliff-edge")
    s, ok = judge()
    inb = scene._in_basin(scene._ball_pos())[0]
    zs = bp()[:, 2]
    check("cliff-edge miss: balls settled ON the stage at the lip (still at stage "
          "height) — no delivery credit (score <= 0.02), NOT success",
          not bool(inb.any()) and bool((zs > 0.10).all()) and s <= 0.02 and not ok)

    # =========================== 9. captured-only ============================================
    env.reset(seed=71)
    step(30)
    pts = bp()[:, :2].clone()
    ctr = pts.mean(dim=0)
    for _ in range(40):  # approx min-enclosing-circle centre
        d = (pts - ctr).norm(dim=-1)
        ctr = ctr + 0.25 * (pts[int(d.argmax())] - ctr)
    place_body(scene.scoop, float(ctr[0]), float(ctr[1]), c.scoop_rest_z + 0.004,
               quat=(1.0, 0.0, 0.0, 0.0) if side() > 0 else (0.0, 0.0, 0.0, 1.0),
               settle_steps=90)
    report("captured-only")
    s, ok = judge()
    check("captured-only: scoop set down over the pack on the stage — capture latch "
          "fires (0.08 <= score <= 0.25) but NOT success",
          float(scene.captured[0]) == 1.0 and 0.08 <= s <= 0.25 and not ok)

    # =========================== 10. scoop-in-basin loophole ================================
    env.reset(seed=81)
    step(30)
    sd = side()
    place_body(scene.scoop, sd * 0.58, 0.0, c.floor_t + c.scoop_wall_h / 2 + 0.004,
               quat=(1.0, 0.0, 0.0, 0.0), settle_steps=60)
    for i in range(3):
        place_body(scene.balls[i], sd * 0.43, (i - 1) * 0.12, basin_ball_z,
                   settle_steps=40)
    step(120)
    report("scoop-in-basin")
    s10, ok = judge()
    inb = scene._in_basin(scene._ball_pos())[0]
    check("scoop-in-basin: all three balls delivered but the scoop fell in too — "
          "delivery credit only (0.70 <= score <= 0.85), NOT success (tool must end "
          "parked on the stage)",
          bool(inb.all()) and not bool(scene.scoop_parked()[0]) and not ok
          and 0.70 <= s10 <= 0.85 + 1e-5)

    # =========================== 11. unsettled judged immediately ===========================
    # (fresh reset; scoop moved OFF the stage first so the balls settling later can
    # never complete the success conjunction — the audit stays honest)
    env.reset(seed=91)
    step(30)
    sd = side()
    place_body(scene.scoop, 0.0, 0.60, c.scoop_wall_h / 2 + 0.004, settle_steps=30)
    for i in range(3):
        place_body(scene.balls[i], sd * 0.45, (i - 1) * 0.12, basin_ball_z,
                   vel=(sd * 0.5, 0.0, 0.0), settle_steps=0)
    step(2)
    report("moving-in-basin")
    s, ok = judge()
    inb = scene._in_basin(scene._ball_pos())[0]
    vmax = float(scene._ball_vel()[0].max())
    check("unsettled: balls IN the basin but still moving fast — NOT success at the "
          f"judged instant (max |v|={vmax:.2f} m/s)",
          bool(inb.all()) and vmax > 0.10 and not ok)
    step(180)  # let them stop against the far wall (scoop off-stage: still no success)

    # =========================== 12. latched delivery survives regression ===================
    # Continue from the settled state of check 11's episode: delivery latches are in.
    report("delivered-settled")
    s_in, _ = judge()
    place_body(scene.balls[1], 0.0, 0.0, ball_rest_z + 0.002, settle_steps=60)
    report("ball-removed")
    s_out, ok = judge()
    inb = scene._in_basin(scene._ball_pos())[0]
    check("latched delivery: removing a delivered ball back onto the stage leaves the "
          f"latched credit unchanged ({s_in:.3f} -> {s_out:.3f}) and success stays gone",
          s_in >= 0.70 and abs(s_out - s_in) < 0.02 and not bool(inb[1]) and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.scoop.data.root_state_w).all()
           and torch.isfinite(scene.basin.data.root_state_w).all()
           and torch.isfinite(scene.end_rail.data.root_state_w).all()
           and all(torch.isfinite(b.data.root_state_w).all() for b in scene.balls))
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.cliff_sweep")
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
