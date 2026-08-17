"""Smoke / rubric-REJECTION battery for MoatBridgeScene (sim_gen task
`approach_grasp_screwdriver_i192`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — lay the plank as a bridge by drop+settle, then
force-roll the ball across into the dock — is the acceptance evidence that the rubric
ACCEPTS a correct outcome). Every teleport here is instrumentation that CONSTRUCTS a
wrong (or partial) outcome and asserts the rubric REJECTS it; no probe in this battery
ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: ball at rest height on the
                            near platform, plank flat on the near platform; score ~0,
                            no success;
  3-4. randomization      — READBACK over 6 seeded resets: moat width (far-platform
                            body x), dock x AND y, ball xy, plank xy AND yaw all move;
                            the ball always spawns on the NEAR side;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy       — the seed's whole plan ("grasp the graspable object, carry
                            it to the target") = the graspable PLANK settled on the
                            dock, ball untouched -> NOT success, score ~0 (identity
                            matters: only the ball in the dock counts);
  7.  ball in the moat    — ball released inside the moat airspace (below the height
                            gate) settles on the moat floor -> NOT success, score ~0,
                            crossing/docking latches stay 0 (the moat earns nothing);
  8.  cantilever plank    — plank level at rim height but NOT reaching the far rim
                            (stable cantilever): bridged_now False, bridged latch 0;
  9.  hover loophole      — ball held in the AIR over the dock centre (judged
                            transiently): in_dock z-band rejects; docked latch stays 0
                            and crossed latch stays 0 (too high);
  10. bridge + mid-deck   — a REAL bridge (drop+settle, bridged latches) with the ball
                            resting mid-deck: partial credit only, NOT success;
  11. mouth near-miss     — ball settled INSIDE the dock mouth but short of the accept
                            plane (dock-frame x < dock_x_lo): crossed latched, yet
                            in_dock False, NOT success, score <= 0.85 (< 1.0);
  12. latched credit      — removing the ball back to the near platform leaves the
                            latched score unchanged and success stays gone;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.approach_grasp_screwdriver_i192.smoke --headless
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
    env = ENVS.get("simgen.moat_bridge")().build(num_envs=args.num_envs, device=device)
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
                                tuple(np.array((0.05, 0.00, 0.10)) + o),
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

    def ball_p() -> torch.Tensor:
        return (scene.ball.data.root_pos_w - scene.env_origins)[0]

    def plank_p() -> torch.Tensor:
        return (scene.plank.data.root_pos_w - scene.env_origins)[0]

    def dock_p() -> torch.Tensor:
        return (scene.dock.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        b = ball_p()
        d = scene._dock_rel()[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | ball=({float(b[0]):+.3f},{float(b[1]):+.3f},"
              f"{float(b[2]):.3f}) dock_rel=({float(d[0]):+.3f},{float(d[1]):+.3f}) "
              f"plank_z={float(plank_p()[2]):.3f} bridged={bool(scene.bridged_now()[0])} "
              f"latches=({float(scene.prog_latch[0]):.2f},"
              f"{float(scene.bridged_latch[0]):.0f},{float(scene.crossed_latch[0]):.0f},"
              f"{float(scene.docked_latch[0]):.0f}) in_dock={bool(scene.in_dock()[0])} "
              f"settled={bool(scene.settled()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0),
              settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    fin0 = all(bool(torch.isfinite(getattr(scene, nm).data.root_state_w).all())
               for nm in ("near_plat", "far_plat", "dock", "plank", "ball"))
    b0, p0 = ball_p(), plank_p()
    check("settle: states finite; ball at rest height on the near platform and the "
          "plank flat on the near platform (readback)",
          fin0 and abs(float(b0[2]) - (c.plat_h + c.ball_r)) < 0.008
          and float(b0[0]) < -0.10
          and abs(float(p0[2]) - (c.plat_h + c.deck_t / 2)) < 0.008
          and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads, yaws = [], []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        fp = (scene.far_plat.data.root_pos_w - scene.env_origins)[0]
        dk = dock_p()
        bb = ball_p()
        pp = plank_p()
        q = scene.plank.data.root_quat_w[0]
        reads.append((float(fp[0]), float(dk[0]), float(dk[1]), float(bb[0]),
                      float(bb[1]), float(pp[0]), float(pp[1])))
        yaws.append(abs(float(q[3])))  # |sin(yaw/2)|: quat double-cover safe
    arr = np.array(reads)
    print(f"[smoke] randomization readback (far_plat_x, dock_x, dock_y, ball_x, "
          f"ball_y, plank_x, plank_y):\n{arr}", flush=True)
    print(f"[smoke] plank |q_z| across seeds: {[f'{y:.3f}' for y in yaws]}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: moat width (far platform x) and dock x AND y vary across "
          "seeded resets (readback)",
          spread[0] > 0.01 and spread[1] > 0.01 and spread[2] > 0.05)
    check("randomization: ball xy and plank xy and plank yaw vary; the ball always "
          "spawns on the NEAR side (readback)",
          spread[3] > 0.03 and spread[4] > 0.05 and spread[5] > 0.01
          and spread[6] > 0.03 and (max(yaws) - min(yaws)) > 0.10
          and bool((arr[:, 3] < -0.15).all()))

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy ============================================
    # The seed's whole plan is "grasp the graspable object and carry it to the target".
    # Here that is exactly the graspable PLANK carried to the dock and settled there,
    # ball untouched. The rubric must refuse it entirely: only the BALL counts.
    env.reset(seed=41)
    step(10)
    dk = dock_p()
    place(scene.plank, float(dk[0]), float(dk[1]), c.plat_h + c.wall_h + c.deck_t,
          settle_steps=90)
    report("seed-strategy")
    s, ok = judge()
    pd = plank_p()
    plank_at_dock = (abs(float(pd[0]) - float(dk[0])) < 0.10
                     and abs(float(pd[1]) - float(dk[1])) < 0.10)
    check("seed strategy (graspable PLANK carried to the dock and settled, ball "
          "untouched): plank verified at the dock, yet NOT success, score <= 0.02",
          plank_at_dock and not ok and s <= 0.02)

    # =========================== 7. ball lost to the moat ===================================
    # Released inside the moat airspace BELOW the progress height gate: settles on the
    # moat floor. The moat must earn nothing: no progress, no crossing, no docking.
    env.reset(seed=51)
    step(10)
    _near, far = scene._rims()
    gap_mid = float(far[0]) / 2
    place(scene.ball, gap_mid, 0.0, c.plat_h - 0.03, settle_steps=90)
    report("moat-loss")
    s, ok = judge()
    check("ball in the moat: settles on the moat floor (z < 0.07), NOT success, "
          "score <= 0.02, crossing/docking latches stay 0",
          float(ball_p()[2]) < 0.07 and bool(scene.settled()[0]) and not ok
          and s <= 0.02 and float(scene.crossed_latch[0]) == 0.0
          and float(scene.docked_latch[0]) == 0.0)

    # =========================== 8. cantilever plank is not a bridge ========================
    # Level, at rim height, hanging over the moat — but NOT reaching the far rim
    # (centre of mass stays over the near platform: a stable cantilever).
    env.reset(seed=61)
    step(10)
    dk = dock_p()
    place(scene.plank, -0.02, float(dk[1]), c.plat_h + c.deck_t / 2 + 0.003,
          settle_steps=120)
    report("cantilever")
    check("cantilever plank: level at rim height but short of the far rim — "
          "bridged_now False, bridged latch stays 0",
          not bool(scene.bridged_now()[0]) and float(scene.bridged_latch[0]) == 0.0
          and abs(float(plank_p()[2]) - (c.plat_h + c.deck_t / 2)) < 0.01)

    # =========================== 9. hover over the dock =====================================
    # Ball held in the AIR directly over the dock centre (judged transiently, then
    # removed): the in_dock z-band rejects it, and neither docking nor crossing latches.
    env.reset(seed=71)
    step(5)
    dk = dock_p()
    place(scene.ball, float(dk[0]), float(dk[1]), c.plat_h + c.ball_r + 0.10,
          settle_steps=2)
    report("hover-dock")
    s, ok = judge()
    hover_ok = (not ok and not bool(scene.in_dock()[0])
                and float(scene.docked_latch[0]) == 0.0
                and float(scene.crossed_latch[0]) == 0.0)
    # remove it far from the dock before it can land inside
    place(scene.ball, -0.30, 0.0, c.plat_h + c.ball_r + 0.002, settle_steps=20)
    check("hover loophole: ball in the air over the dock centre — NOT success, "
          "docked and crossed latches stay 0 (z-band gates both)", hover_ok)

    # =========================== 10. bridge + ball mid-deck =================================
    # A REAL bridge (drop + settle through contact, exactly the solve's construction)
    # and the ball resting mid-deck: honest partial credit, NOT success.
    env.reset(seed=81)
    step(10)
    _near, far = scene._rims()
    dk = dock_p()
    x_c = float(far[0]) - 0.12
    place(scene.plank, x_c, float(dk[1]), c.plat_h + c.deck_t / 2 + 0.005,
          settle_steps=120)
    bridged_mid = bool(scene.bridged_now()[0])
    place(scene.ball, x_c, float(dk[1]), c.plat_h + c.deck_t + c.ball_r + 0.002,
          settle_steps=60)
    report("mid-deck")
    s_mid, ok = judge()
    check("bridge + ball resting mid-deck: bridged latched, yet NOT success and "
          "score <= 0.60 (no crossing, no docking)",
          bridged_mid and float(scene.bridged_latch[0]) == 1.0 and not ok
          and bool(scene.settled()[0]) and s_mid <= 0.60)

    # =========================== 11. mouth near-miss ========================================
    # Same episode: ball settled INSIDE the dock mouth but short of the accept plane
    # (dock-frame x below dock_x_lo). Crossing latches (it IS on the far platform),
    # but in_dock must stay False and success must not fire.
    place(scene.ball, float(dk[0]) + c.dock_x_lo - 0.017, float(dk[1]),
          c.plat_h + c.ball_r + 0.002, settle_steps=60)
    report("mouth-near-miss")
    s_near, ok = judge()
    d = scene._dock_rel()[0]
    check("mouth near-miss: ball settled inside the mouth but short of the accept "
          "plane — crossed latched, in_dock False, NOT success, score <= 0.85",
          float(d[0]) < c.dock_x_lo and bool(scene.settled()[0])
          and float(scene.crossed_latch[0]) == 1.0 and not bool(scene.in_dock()[0])
          and not ok and s_near <= 0.85)

    # =========================== 12. latched credit survives regression =====================
    place(scene.ball, -0.30, 0.0, c.plat_h + c.ball_r + 0.002, settle_steps=60)
    report("ball-removed")
    s_out, ok = judge()
    check("latched credit: removing the ball back to the near platform leaves the "
          f"latched score unchanged ({s_near:.3f} -> {s_out:.3f}) and success stays gone",
          s_near >= 0.55 and abs(s_out - s_near) < 0.02 and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(getattr(scene, nm).data.root_state_w).all())
              for nm in ("near_plat", "far_plat", "dock", "plank", "ball"))
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.moat_bridge")
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
