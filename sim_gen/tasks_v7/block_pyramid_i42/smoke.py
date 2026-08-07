"""Smoke / rubric-REJECTION battery for TiltLabyrinthScene (sim_gen task
`block_pyramid_i42`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — closed-loop tilt control that rolls the ball
through the serpentine and drops it through the green hole — is the acceptance
evidence that the rubric ACCEPTS a correct outcome). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome and asserts the rubric
REJECTS it; no probe in this battery ever reaches success(), and a final audit check
asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: ball at rolling height in the
                            start leg, tray SELF-LEVELLED (|u_xy| readback); score ~0;
  3-4. randomization      — READBACK over 6 seeded resets: apparatus xy + yaw and the
                            ball's tray-local start position all move;
  5.  null policy         — 300 idle steps: ball still in the start leg, score ~0;
  6.  SEED strategy       — the seed's whole plan is pick-and-place to the target zone:
                            ball hand-placed on the last-leg floor beside the goal hole.
                            It may even roll in — but the middle-corridor latch was
                            never earned, so NOT success, score <= 0.70;
  7.  direct-to-box       — ball teleported INSIDE the green catch box, settled:
                            physically in the box and still, yet NOT success (pathway
                            latches unearned), score ~0;
  8.  wrong place         — ball teleported into the RED trap box: NOT success, score ~0;
  9.  trap drop-through   — ball released directly over the red hole falls into the trap
                            box; crossing the floor plane INSIDE the hole opening must
                            not latch corridor credit -> score ~0, NOT success;
  10. sealed underside    — ball dropped from above onto the tray-rim gap lands ON the
                            rim cover (readback height) and never enters either box:
                            the only way in is through a hole in the tray;
  11. latched credit      — mid-corridor credit, once earned on the floor, survives the
                            ball being removed (and success is gone);
  12. monotonicity        — floor placements deeper along the route latch strictly more
                            credit (mid < north < goal-approach);
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.block_pyramid_i42.smoke --headless
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
    env = ENVS.get("simgen.tilt_labyrinth")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.95, -0.90, 0.85)) + o),
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

    def report(tag: str) -> None:
        p = scene.ball_tray_local()[0]
        pb = scene.ball_base_local()[0]
        u = scene.tray_up_base()[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | ball_tray=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):+.3f}) ball_base_z={float(pb[2]):.3f} "
              f"u=({float(u[0]):+.3f},{float(u[1]):+.3f}) "
              f"mid={bool(scene.lat_mid[0])} north={bool(scene.lat_north[0])} "
              f"near={bool(scene.lat_near[0])} goal={bool(scene.in_goal_box()[0])} "
              f"trapbox={bool(scene.in_trap_box()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def _local_to_world(body, lx: float, ly: float, lz: float) -> torch.Tensor:
        v = torch.tensor([lx, ly, lz], device=device).unsqueeze(0)
        return (body.data.root_pos_w[0] + quat_apply(body.data.root_quat_w, v)[0]
                - scene.env_origins[0])

    def place_ball_local(body, lx: float, ly: float, lz: float,
                         settle_steps: int = 30) -> None:
        """Kinematic probe placement in `body`'s local frame (instrumentation, not a
        solution) + REAL physics steps before judging (the zero-step trap)."""
        w = _local_to_world(body, lx, ly, lz)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = w
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.ball.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    z_roll = c.floor_top_local + c.ball_r  # tray-local rolling height (-0.030)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    p = scene.ball_tray_local()[0]
    u = scene.tray_up_base()[0]
    fin0 = bool(torch.isfinite(scene.base.data.root_state_w).all()
                and torch.isfinite(scene.frame.data.root_state_w).all()
                and torch.isfinite(scene.tray.data.root_state_w).all()
                and torch.isfinite(scene.ball.data.root_state_w).all())
    check("settle: states finite; ball at rolling height in the START leg and the tray "
          "self-levelled (|u_xy| readback)",
          fin0 and abs(float(p[2]) - z_roll) < 0.008 and float(p[1]) < -0.080
          and math.hypot(float(u[0]), float(u[1])) < 0.060 and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        bp = (scene.base.data.root_pos_w - scene.env_origins)[0]
        q = scene.base.data.root_quat_w[0]
        yaw = math.atan2(2 * float(q[0]) * float(q[3]), 1 - 2 * float(q[3]) ** 2)
        bl = scene.ball_tray_local()[0]
        reads.append((float(bp[0]), float(bp[1]), yaw, float(bl[0]), float(bl[1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (base_x, base_y, base_yaw, ball_tray_x, "
          f"ball_tray_y):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: apparatus xy + yaw vary across seeded resets (readback)",
          spread[0] > 0.005 and spread[1] > 0.005 and spread[2] > 0.5)
    check("randomization: ball tray-local start position varies across seeded resets "
          "(readback)", spread[3] > 0.02 and spread[4] > 0.005)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(300)
    report("null-policy")
    p = scene.ball_tray_local()[0]
    s, ok = judge()
    check("null policy: after 300 idle steps the ball is still in the START leg "
          "(readback y), score ~0, no success",
          float(p[1]) < -0.075 and s <= 0.02 and not ok)

    # =========================== 6. SEED strategy ===========================================
    # The seed's whole plan is pick-and-place to the target zone. Here that is: carry
    # the ball to the last leg and set it down beside the green hole. Gravity may even
    # roll it in — but the middle corridor was never rolled, so the pathway gate
    # (lat_mid) must refuse success.
    env.reset(seed=41)
    step(10)
    place_ball_local(scene.tray, 0.098, 0.150, z_roll + 0.003, settle_steps=240)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (pick-and-place to the target zone): ball set down beside the "
          "goal hole — mid-corridor latch never earned (verified), NOT success, "
          "score <= 0.70",
          not bool(scene.lat_mid[0]) and not ok and s <= 0.70)

    # =========================== 7. direct-to-box teleport ==================================
    env.reset(seed=51)
    step(10)
    place_ball_local(scene.base, c.goal_box_c[0], c.goal_box_c[1], 0.070, settle_steps=90)
    report("direct-to-box")
    s, ok = judge()
    check("direct-to-box teleport: ball verified physically IN the green box and "
          "settled, yet NOT success (pathway latches unearned), score <= 0.02",
          bool(scene.in_goal_box()[0]) and bool(scene.settled()[0]) and not ok
          and s <= 0.02)

    # =========================== 8. wrong place: trap box ===================================
    env.reset(seed=61)
    step(10)
    place_ball_local(scene.base, 0.0, 0.0, 0.070, settle_steps=90)
    report("trap-box")
    s, ok = judge()
    check("wrong place: ball teleported into the RED trap box (verified) — NOT success, "
          "score <= 0.02",
          bool(scene.in_trap_box()[0]) and not ok and s <= 0.02)

    # =========================== 9. trap drop-through =======================================
    # Released directly over the red hole: the ball crosses the maze-floor plane INSIDE
    # the hole opening while falling — that must not latch corridor credit.
    env.reset(seed=71)
    step(10)
    place_ball_local(scene.tray, 0.0, 0.0, z_roll + 0.015, settle_steps=150)
    report("trap-drop")
    s, ok = judge()
    check("trap drop-through: ball fell through the red hole into the trap box "
          "(verified), corridor latches all stayed False (hole-opening exclusion), "
          "NOT success, score <= 0.02",
          bool(scene.in_trap_box()[0]) and not bool(scene.lat_mid[0])
          and not bool(scene.lat_north[0]) and not ok and s <= 0.02)

    # =========================== 10. sealed underside =======================================
    # Drop the ball from above onto the ring gap between the tray edge and the guard
    # skirt: it must land ON the rim cover, never inside a catch box.
    env.reset(seed=81)
    step(10)
    place_ball_local(scene.base, 0.242, 0.0, 0.200, settle_steps=90)
    report("seal-drop")
    pb = scene.ball_base_local()[0]
    s, ok = judge()
    check("sealed underside: ball dropped onto the tray-rim gap rests ON the rim cover "
          "(readback z > 0.112) and is in NEITHER catch box — the only way in is "
          "through a hole in the tray",
          float(pb[2]) > 0.112 and not bool(scene.in_goal_box()[0])
          and not bool(scene.in_trap_box()[0]) and not ok)

    # =========================== 11. latched credit survives regression =====================
    env.reset(seed=91)
    step(10)
    place_ball_local(scene.tray, -0.120, 0.0, z_roll + 0.003, settle_steps=60)
    report("mid-placed")
    s_in, _ = judge()
    place_ball_local(scene.tray, 0.0, -0.150, z_roll + 0.003, settle_steps=60)
    report("ball-removed")
    s_out, ok = judge()
    check("latched credit: mid-corridor credit earned on the floor survives the ball "
          "being moved back to the start leg (and success is gone)",
          s_in >= 0.18 and abs(s_out - s_in) < 0.02 and not ok)

    # =========================== 12. monotonicity ===========================================
    # Same episode, deeper placements along the route latch strictly more credit.
    place_ball_local(scene.tray, -0.150, 0.140, z_roll + 0.003, settle_steps=40)
    s_north, _ = judge()
    place_ball_local(scene.tray, 0.080, 0.150, z_roll + 0.003, settle_steps=40)
    s_near, ok = judge()
    check("monotonicity: floor placements deeper along the route latch strictly more "
          f"credit (mid {s_in:.2f} < north {s_north:.2f} < approach {s_near:.2f}), "
          "still never success",
          s_in + 0.10 < s_north and s_north + 0.10 < s_near and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.base.data.root_state_w).all()
           and torch.isfinite(scene.frame.data.root_state_w).all()
           and torch.isfinite(scene.tray.data.root_state_w).all()
           and torch.isfinite(scene.ball.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.tilt_labyrinth")
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
