"""Smoke / rubric-REJECTION battery for BellHerdScene (sim_gen task
`track_bowl_i27`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — cage the red ball under the gravity-dropped
bell, force-drag the loaded bell to the pocket, park the bell clear — is the
acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport
here is instrumentation that CONSTRUCTS a wrong (or partial) outcome and asserts
the rubric REJECTS it; no probe in this battery ever reaches success(), and a
final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: both balls resting on the
                            play surface at sphere height, the bell grounded mouth-
                            down; score ~0 at rest, no success;
  3-4. randomization      — READBACK over 6 seeded resets: board xy + yaw move, the
                            RED ball takes BOTH y-sides, red/blue/bell local xy and
                            the progress baseline d0 all move;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's whole plan ("grasp the bowl-shaped object,
                            carry it along a path, put it at the goal") = the bell
                            itself transported and SET DOWN on the pocket: the bell
                            verifiably BRIDGES the hole (rim ring wider than the
                            pocket diagonal — readback), no ball moved -> NOT
                            success, score ~0;
  7.  near miss           — the red ball settled ON the surface hard against the
                            pocket edge: in_pocket() False (containment means below
                            the surface, not proximity), NOT success;
  8.  wrong object        — the BLUE ball dropped through the pocket plugs it; the
                            red ball dropped after it comes to rest ON TOP, still
                            above the surface: blue in_pocket, red NOT in_pocket,
                            NOT success (the wrong-ball failure is real and blocks
                            the right one);
  9.  covered delivery    — the red ball genuinely IN the pocket (gravity drop
                            through the hole) but the bell parked ON the pocket:
                            the bell_is_clear clause alone rejects success;
  10. latched credit      — the delivered red ball teleported back OUT of the
                            pocket: the latched pocket + progress score survives
                            unchanged (and still no success);
  11. cage != success     — the bell gravity-dropped over the red ball exactly as
                            the solution cages it: caged() True, yet score <= 0.22
                            and NOT success — capture alone is only partial credit;
  12. monotonicity        — placing the red ball closer to the pocket latches
                            strictly more progress credit than a farther placement;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.track_bowl_i27.smoke --headless
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
    env = ENVS.get("simgen.bell_herd")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.35, -0.95, 0.75)) + o),
                                tuple(np.array((0.40, 0.00, 0.10)) + o),
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

    def board_pose() -> tuple[torch.Tensor, float]:
        fp = (scene.board.data.root_pos_w - scene.env_origins)[0]
        q = scene.board.data.root_quat_w[0]
        return fp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def to_world(lx: float, ly: float) -> tuple[float, float]:
        fp, yaw = board_pose()
        cy, sy = math.cos(yaw), math.sin(yaw)
        return (float(fp[0]) + lx * cy - ly * sy, float(fp[1]) + lx * sy + ly * cy)

    def yaw_quat() -> tuple[float, float, float, float]:
        _fp, yaw = board_pose()
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def report(tag: str) -> None:
        rl = scene._board_local(scene.red.data.root_pos_w)[0]
        bl = scene._board_local(scene.blue.data.root_pos_w)[0]
        el = scene._board_local(scene.bell.data.root_pos_w)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | red=({float(rl[0]):+.3f},{float(rl[1]):+.3f},"
              f"{float(rl[2]):+.3f}) blue=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):+.3f}) bell=({float(el[0]):+.3f},{float(el[1]):+.3f},"
              f"{float(el[2]):+.3f}) caged={bool(scene.caged()[0])} "
              f"red_in={bool(scene.in_pocket(scene.red)[0])} "
              f"blue_in={bool(scene.in_pocket(scene.blue)[0])} "
              f"clear={bool(scene.bell_is_clear()[0])} settled={bool(scene.settled()[0])} "
              f"cage={float(scene.cage_latch[0]):.2f} prog={float(scene.prog_latch[0]):.2f} "
              f"pit={float(scene.pit_latch[0]):.2f} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_local(body, lx: float, ly: float, z: float,
                    quat=(1.0, 0.0, 0.0, 0.0), settle_steps: int = 30) -> None:
        """Kinematic probe placement in BOARD-LOCAL xy (instrumentation, not a
        solution) + REAL physics steps before judging (the zero-step trap)."""
        wx, wy = to_world(lx, ly)
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = wx, wy, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def fin_all() -> bool:
        return bool(torch.isfinite(scene.board.data.root_state_w).all()
                    and torch.isfinite(scene.bell.data.root_state_w).all()
                    and torch.isfinite(scene.red.data.root_state_w).all()
                    and torch.isfinite(scene.blue.data.root_state_w).all())

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    red_z = float(scene._board_local(scene.red.data.root_pos_w)[0][2])
    blue_z = float(scene._board_local(scene.blue.data.root_pos_w)[0][2])
    bell_z = float(scene._board_local(scene.bell.data.root_pos_w)[0][2])
    check("settle: states finite; both balls resting on the surface at sphere height, "
          "the bell grounded mouth-down (readback)",
          fin_all() and abs(red_z - c.ball_r) < 0.008 and abs(blue_z - c.ball_r) < 0.008
          and -0.010 < bell_z < 0.015 and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.03), no success", s <= 0.03 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        fp, fyaw = board_pose()
        rl = scene._board_local(scene.red.data.root_pos_w)[0]
        bl = scene._board_local(scene.blue.data.root_pos_w)[0]
        el = scene._board_local(scene.bell.data.root_pos_w)[0]
        reads.append((float(fp[0]), float(fp[1]), fyaw, float(scene.red_side[0]),
                      float(rl[0]), float(rl[1]), float(bl[0]), float(bl[1]),
                      float(el[0]), float(el[1]), float(scene.d0[0])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (board_x, board_y, board_yaw, red_side, "
          f"red_x, red_y, blue_x, blue_y, bell_x, bell_y, d0):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: board xy + yaw vary AND the RED ball takes both y-sides "
          "across seeded resets (readback)",
          spread[0] > 0.005 and spread[1] > 0.005 and spread[2] > 0.05
          and arr[:, 3].min() < -0.5 and arr[:, 3].max() > 0.5)
    check("randomization: red xy, blue xy, bell xy, and the progress baseline d0 vary "
          "across seeded resets (readback)",
          spread[4] > 0.01 and spread[5] > 0.01 and spread[6] > 0.01
          and spread[7] > 0.01 and spread[8] > 0.01 and spread[9] > 0.01
          and spread[10] > 0.005)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.03 and not ok)

    # =========================== 6. SEED strategy ===========================================
    # The seed's whole plan is "grasp the bowl-shaped object, carry it along a path,
    # and put it at the goal". Here that end state — the BELL itself transported and
    # SET DOWN on the pocket — must be worthless: no ball moved, and the bell cannot
    # even fall in (its rim ring out-spans the pocket diagonal — verified by readback).
    env.reset(seed=41)
    step(10)
    place_local(scene.bell, c.pit_x, 0.0, c.surface_h + 0.02, quat=yaw_quat(),
                settle_steps=90)
    report("seed-strategy")
    bell_z = float(scene._board_local(scene.bell.data.root_pos_w)[0][2])
    s, ok = judge()
    check("seed strategy (the bell itself carried and SET DOWN on the pocket): the "
          "bell BRIDGES the hole (rim ring wider than the pocket diagonal — readback), "
          "no ball moved -> NOT success, score <= 0.05",
          bell_z > -0.010 and not bool(scene.in_pocket(scene.red)[0])
          and not ok and s <= 0.05)

    # =========================== 7. near miss ===============================================
    env.reset(seed=51)
    step(10)
    place_local(scene.red, c.pit_x, c.pit_half + c.ball_r + 0.002,
                c.surface_h + c.ball_r + 0.002, settle_steps=90)
    report("near-miss")
    red_z = float(scene._board_local(scene.red.data.root_pos_w)[0][2])
    s, ok = judge()
    check("near miss: red ball settled ON the surface hard against the pocket edge — "
          "in_pocket() False (containment means below the surface, not proximity), "
          "NOT success, score <= 0.30",
          red_z > 0.5 * c.ball_r and not bool(scene.in_pocket(scene.red)[0])
          and not ok and s <= 0.30)

    # =========================== 8. wrong object ============================================
    # Drop the BLUE ball through the pocket: it plugs the hole (85 mm ball in a
    # 107 mm pocket sunk 60 mm — a second ball cannot pass). Then drop the red ball
    # onto the plugged pocket: it must come to rest ABOVE the surface, not inside.
    env.reset(seed=61)
    step(10)
    place_local(scene.blue, c.pit_x, 0.0, c.surface_h + 0.06, settle_steps=90)
    report("blue-plugged")
    blue_in = bool(scene.in_pocket(scene.blue)[0])
    place_local(scene.red, c.pit_x, 0.0, c.surface_h + 0.06, settle_steps=120)
    report("red-on-plug")
    s, ok = judge()
    check("wrong object: BLUE ball dropped through the pocket plugs it (blue "
          "in_pocket by readback); the red ball dropped after it rests on top, NOT "
          "in_pocket -> NOT success",
          blue_in and bool(scene.in_pocket(scene.blue)[0])
          and not bool(scene.in_pocket(scene.red)[0]) and not ok)

    # =========================== 9. covered delivery ========================================
    # A genuine delivery (red ball gravity-dropped through the open hole) that stays
    # COVERED by the bell parked on the pocket: the bell_is_clear clause alone must
    # reject success.
    env.reset(seed=71)
    step(10)
    place_local(scene.red, c.pit_x, 0.0, c.surface_h + 0.06, settle_steps=90)
    red_in = bool(scene.in_pocket(scene.red)[0])
    place_local(scene.bell, c.pit_x, 0.0, c.surface_h + 0.02, quat=yaw_quat(),
                settle_steps=90)
    report("covered")
    s_cov, ok = judge()
    check("covered delivery: red ball genuinely IN the pocket (gravity drop, readback) "
          "but the bell parked ON the pocket — bell_is_clear False, the reveal clause "
          "alone rejects success",
          red_in and bool(scene.in_pocket(scene.red)[0])
          and not bool(scene.bell_is_clear()[0]) and not ok)

    # =========================== 10. latched credit survives regression =====================
    place_local(scene.red, -0.20, 0.0, c.surface_h + c.ball_r + 0.002, settle_steps=60)
    report("red-removed")
    s_out, ok = judge()
    check("latched credit: teleporting the delivered red ball back OUT of the pocket "
          "leaves the latched pocket + progress score unchanged (and still no success)",
          s_cov >= 0.50 and abs(s_out - s_cov) < 0.02 and not ok)

    # =========================== 11. cage alone is not success ==============================
    # Exactly the solution's caging move: hover the bell centred on the red ball,
    # mouth 25 mm up, and let it fall. Capture must read back True — and still be
    # worth only its partial credit.
    env.reset(seed=81)
    step(30)
    rl = scene._board_local(scene.red.data.root_pos_w)[0]
    place_local(scene.bell, float(rl[0]), float(rl[1]), c.surface_h + 0.025,
                quat=yaw_quat(), settle_steps=60)
    report("caged-only")
    s, ok = judge()
    check("cage != success: the bell gravity-dropped over the red ball cages it "
          "(caged() True by readback), yet score <= 0.22 and NOT success — capture "
          "alone is only partial credit",
          bool(scene.caged()[0]) and s <= 0.22 and not ok)

    # =========================== 12. progress monotonicity ==================================
    env.reset(seed=91)
    step(5)
    place_local(scene.red, -0.10, 0.0, c.surface_h + c.ball_r + 0.002, settle_steps=3)
    p_far = float(scene.prog_latch[0])
    place_local(scene.red, 0.10, 0.0, c.surface_h + c.ball_r + 0.002, settle_steps=3)
    p_near = float(scene.prog_latch[0])
    check("monotonicity: placing the red ball closer to the pocket latches strictly "
          f"more progress credit ({p_far:.3f} < {p_near:.3f})", p_far + 0.10 < p_near)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", fin_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.bell_herd")
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
