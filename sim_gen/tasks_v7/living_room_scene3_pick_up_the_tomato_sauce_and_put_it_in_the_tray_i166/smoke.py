"""Smoke / rubric-REJECTION battery for SauceChuteScene (sim_gen task
`living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray_i166`) —
NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — dock the empty tray, fingertip-force nudge, the
gravity roll/drop/catch, the loaded-tray delivery — is the acceptance evidence that
the rubric ACCEPTS a correct outcome). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome as a settled state and asserts the rubric
REJECTS it; no probe in this battery ever reaches success(), and a final audit check
asserts exactly that.

  1-2.  settle/no-NaN     — reset layout settles finite: can seated captive in the
                            cradle, tray/decoy on the floor, everything still,
                            score ~0 at rest;
  3-4.  randomization     — READBACK over 8 seeded resets: tray / pad / decoy
                            positions vary; tray yaw and the can's cradle-seat x vary;
  5.   null policy        — 240 idle steps -> score ~0, no success, and the can never
                            leaves the cradle (captivity is real; no capsule creep);
  6.   seed strategy      — the seed's plan (pick the loose can, place it in the
                            tray) executed on the only loose can: WHITE decoy
                            hand-placed into the tray on the pad -> rejected, score ~0
                            (wrong object; the red can untouched);
  7.   dock near-miss     — tray settled 50 mm off the dock target (outside the 35 mm
                            tolerance) -> the docked latch never sets;
  8.   floored rule       — the RED can landing on open floor latches permanent
                            failure; the manual rescue (that same can hand-placed
                            into the tray on the pad) is still rejected — `caught`
                            can never set, score <= 0.31;
  9.   catch, no delivery — tray docked and the can contact-dropped into it: docked
                            and caught latch, but no success (tray not on the pad),
                            score <= 0.71;
  10.  pad near-miss      — the loaded tray moved 45 mm off the pad centre (outside
                            the 30 mm tolerance) -> not on the pad, no delivery, no
                            success;
  11.  decoy clause + cap — the decoy added INSIDE the tray, then the loaded tray
                            placed PERFECTLY on the pad: every stage latch earned,
                            score pinned at the 0.85 non-success cap, success still
                            False (the white can in the tray rejects);
  12.  latched credit     — the red can yanked back out to open floor: the latched
                            score holds (credit does not evaporate), still no success;
  13.  tray not presented — the tray INVERTED on the pad with the can resting on top:
                            not upright, can not contained, no success;
  14.  rejection audit    — success() was never True at ANY judged point;
  15.  final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray_i166.smoke --headless
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
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> the
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.sauce_chute")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.65, -1.15, 0.90)) + o),
                                tuple(np.array((0.40, 0.00, 0.15)) + o),
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

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        cp, tp, dp = rel(scene.can), rel(scene.tray), rel(scene.decoy)
        s, ok = judge()
        print(f"[smoke] {tag:16s} | can=({float(cp[0]):+.3f},{float(cp[1]):+.3f},"
              f"{float(cp[2]):.3f}) tray=({float(tp[0]):+.3f},{float(tp[1]):+.3f},"
              f"{float(tp[2]):.3f}) decoy=({float(dp[0]):+.3f},{float(dp[1]):+.3f}) "
              f"docked={bool(scene._docked[0])} exited={bool(scene._exited[0])} "
              f"caught={bool(scene._caught[0])} delivered={bool(scene._delivered[0])} "
              f"floored={bool(scene._floored[0])} "
              f"in_tray={bool(scene.in_tray(scene.can)[0])} "
              f"on_pad={bool(scene.tray_on_pad()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def move_rigid(bodies, dx: float, dy: float, dz_tray_target: float) -> None:
        """Move a group by ONE rigid delta computed from the tray (relative poses
        preserved, all velocities zeroed) — the loaded-carry constructor."""
        tray_p = scene.tray.data.root_state_w[all_ids, 0:3].clone()
        delta = torch.zeros(n, 3, device=device)
        delta[:, 0] = dx - (tray_p[:, 0] - scene.env_origins[:, 0])
        delta[:, 1] = dy - (tray_p[:, 1] - scene.env_origins[:, 1])
        delta[:, 2] = dz_tray_target - (tray_p[:, 2] - scene.env_origins[:, 2])
        for body in bodies:
            st = body.data.root_state_w[all_ids].clone()
            st[:, 0:3] += delta
            st[:, 7:13] = 0.0
            body.write_root_state_to_sim(st, all_ids)

    def yaw_deg(body) -> float:
        q = body.data.root_quat_w[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    pad_top = c.pad_size[2]

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.can.data.root_state_w).all()
            and torch.isfinite(scene.tray.data.root_state_w).all()
            and torch.isfinite(scene.decoy.data.root_state_w).all()
            and torch.isfinite(scene.pad.data.root_state_w).all()
            and torch.isfinite(scene.chute.data.root_state_w).all())
    cp = rel(scene.can)
    still = (float(scene.can.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.tray.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.decoy.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, can seated captive in the cradle, tray and decoy on "
          "the floor, everything still",
          bool(fin0) and abs(float(cp[1]) - c.cradle_y) < 0.010
          and abs(float(cp[2]) - c.can_rest_z) < 0.008 and still
          and not bool(scene.in_tray(scene.can)[0])
          and not bool(scene.in_tray(scene.decoy)[0])
          and float(rel(scene.tray)[2]) < 0.012 and float(rel(scene.decoy)[2]) < 0.040)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(2)
        tp, pp, dp, cq = rel(scene.tray), rel(scene.pad), rel(scene.decoy), rel(scene.can)
        reads.append([float(tp[0]), float(tp[1]), yaw_deg(scene.tray),
                      float(pp[0]), float(pp[1]), float(dp[0]), float(dp[1]),
                      float(cq[0])])
    arr = np.array(reads)
    print(f"[smoke] randomization readback (tx, ty, t_yaw, px, py, dx, dy, can_x):\n"
          f"{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: tray / pad / decoy positions vary across seeded resets "
          f"(xy spreads {spread[0]:.3f}/{spread[1]:.3f}, {spread[3]:.3f}/{spread[4]:.3f}, "
          f"{spread[5]:.3f}/{spread[6]:.3f})",
          min(spread[0], spread[1]) > 0.02 and min(spread[3], spread[4]) > 0.02
          and max(spread[5], spread[6]) > 0.01)
    check("randomization: tray yaw and the can's cradle-seat x vary (readback: yaw "
          f"spread {spread[2]:.1f} deg, can x spread {spread[7] * 1000:.1f} mm)",
          spread[2] > 20.0 and spread[7] > 0.0015)

    # =========================== 5. null policy fails (and captivity is real) ===============
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    cp = rel(scene.can)
    s, ok = judge()
    check("null policy: 240 idle steps — score ~0, no success, and the can never left "
          f"the cradle (y={float(cp[1]):.3f}; captivity real, no capsule creep)",
          s <= 0.02 and not ok and abs(float(cp[1]) - c.cradle_y) < 0.010
          and abs(float(cp[2]) - c.can_rest_z) < 0.008)

    # =========================== 6. seed strategy: pick the loose can, place it ============
    # The seed's plan — grasp the freestanding can, carry it, lower it into the tray —
    # is executable here only on the WHITE decoy (the one loose can). Constructed
    # settled: decoy inside the tray, tray on the pad. Rejected, score ~0.
    torch.manual_seed(41)
    env.reset()
    step(10)
    pp = rel(scene.pad)
    place(scene.tray, float(pp[0]), float(pp[1]), pad_top + 0.002)
    step(60)
    place(scene.decoy, float(pp[0]), float(pp[1]),
          pad_top + c.tray_floor_t + c.can_r + 0.020)
    step(90)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy / wrong object: WHITE decoy hand-placed into the tray on the "
          "pad, red can untouched — no success, score <= 0.02",
          bool(scene.in_tray(scene.decoy)[0]) and bool(scene.tray_on_pad()[0])
          and not ok and s <= 0.02)

    # =========================== 7. dock near-miss ==========================================
    torch.manual_seed(46)
    env.reset()
    step(10)
    place(scene.tray, c.dock_target[0], c.dock_target[1] - 0.050, 0.002)
    step(90)
    report("dock-miss")
    s, ok = judge()
    check("dock near-miss: tray settled 50 mm off the dock target (tol 35 mm) — the "
          "docked latch never sets, no success",
          not bool(scene._docked[0]) and not ok and s <= 0.02)

    # =========================== 8. the floored rule ========================================
    # The RED can dropped on open floor -> permanent failure latch. Then the manual
    # rescue — that same can hand-placed into the tray on the pad — still rejected:
    # `caught` can never set once floored.
    torch.manual_seed(61)
    env.reset()
    step(10)
    place(scene.can, 0.45, -0.30, c.can_r + 0.030)
    step(90)
    report("floored")
    floored_set = bool(scene._floored[0])
    pp = rel(scene.pad)
    place(scene.tray, float(pp[0]), float(pp[1]), pad_top + 0.002)
    step(60)
    place(scene.can, float(pp[0]), float(pp[1]),
          pad_top + c.tray_floor_t + c.can_r + 0.020)
    step(90)
    report("floor-rescue")
    s, ok = judge()
    check("floored rule: the red can landing on open floor latches permanent failure; "
          "the manual rescue into the tray on the pad is still rejected — caught never "
          f"sets, score <= 0.31 (s={s:.3f})",
          floored_set and bool(scene._floored[0]) and bool(scene.in_tray(scene.can)[0])
          and bool(scene.tray_on_pad()[0]) and not bool(scene._caught[0])
          and not ok and s <= 0.31)

    # =========================== 9. catch without delivery ==================================
    # Tray docked, can contact-dropped INTO it (inside the footprint the whole way —
    # floored never trips): docked + caught latch, but the tray never reaches the pad.
    torch.manual_seed(71)
    env.reset()
    step(10)
    place(scene.tray, c.dock_target[0], c.dock_target[1], 0.002)
    step(60)
    place(scene.can, c.dock_target[0], c.dock_target[1] - 0.050,
          c.tray_floor_t + c.can_r + 0.020)
    step(90)
    report("caught-no-pad")
    s, ok = judge()
    check("catch without delivery: tray docked and the can settled inside it — docked "
          f"and caught latched, but no success off the pad, score <= 0.71 (s={s:.3f})",
          bool(scene._docked[0]) and bool(scene._caught[0])
          and bool(scene.in_tray(scene.can)[0]) and not bool(scene._floored[0])
          and not ok and s <= 0.71)

    # =========================== 10. pad near-miss ==========================================
    pp = rel(scene.pad)
    move_rigid([scene.tray, scene.can], float(pp[0]) + 0.045, float(pp[1]),
               pad_top + 0.002)
    step(90)
    report("pad-miss")
    s, ok = judge()
    check("pad near-miss: the loaded tray set down 45 mm off the pad centre (tol "
          "30 mm) — not on the pad, no delivery, no success",
          not bool(scene.tray_on_pad()[0]) and not bool(scene._delivered[0])
          and bool(scene.in_tray(scene.can)[0]) and not ok)

    # =========================== 11. decoy clause at the score cap ==========================
    # The WHITE decoy added inside the tray (beside the can), then the loaded tray
    # placed PERFECTLY on the pad: every stage latch earns, the score pins at the 0.85
    # non-success cap — and success stays False on the decoy clause alone.
    tp = rel(scene.tray)
    place(scene.decoy, float(tp[0]), float(tp[1]) + 0.050,
          float(tp[2]) + c.tray_floor_t + c.can_r + 0.020)
    step(60)
    decoy_in_before = bool(scene.in_tray(scene.decoy)[0])
    move_rigid([scene.tray, scene.can, scene.decoy], float(pp[0]), float(pp[1]),
               pad_top + 0.002)
    step(90)
    report("decoy-clause")
    s, ok = judge()
    check("decoy clause + cap: decoy INSIDE the tray, everything else perfect on the "
          f"pad — every latch earned, score pinned at the 0.85 cap (s={s:.4f}), "
          "success still False",
          decoy_in_before and bool(scene.in_tray(scene.decoy)[0])
          and bool(scene.in_tray(scene.can)[0]) and bool(scene.tray_on_pad()[0])
          and bool(scene._delivered[0]) and not ok and 0.84 <= s <= 0.8501)

    # =========================== 12. latched credit survives regression =====================
    place(scene.can, 0.75, 0.20, c.can_r + 0.002)  # yank the red can back out
    step(60)
    report("regressed")
    s_a, ok_a = judge()
    step(60)
    report("regressed2")
    s_b, ok_b = judge()
    check("latched credit: every stage latch earned then the can yanked back out — "
          f"score holds ({s_a:.3f} -> {s_b:.3f}), still no success",
          bool(scene._caught[0]) and bool(scene._delivered[0]) and abs(s_a - s_b) < 1e-3
          and s_a >= 0.84 and not ok_a and not ok_b)

    # =========================== 13. tray not presented: inverted on the pad ================
    torch.manual_seed(81)
    env.reset()
    step(10)
    pp = rel(scene.pad)
    place(scene.tray, float(pp[0]), float(pp[1]),
          pad_top + c.tray_floor_t + c.tray_wall_h + 0.002, quat=(0.0, 1.0, 0.0, 0.0))
    step(30)
    tz = float(rel(scene.tray)[2])
    place(scene.can, float(pp[0]), float(pp[1]), tz + c.can_r + 0.004)
    step(90)
    report("inverted-tray")
    s, ok = judge()
    check("tray not presented: tray INVERTED on the pad, can resting on top — not "
          "upright, can not contained, no success",
          not bool(scene.tray_upright()[0]) and not bool(scene.in_tray(scene.can)[0])
          and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.can.data.root_state_w).all()
           and torch.isfinite(scene.tray.data.root_state_w).all()
           and torch.isfinite(scene.decoy.data.root_state_w).all()
           and torch.isfinite(scene.pad.data.root_state_w).all()
           and torch.isfinite(scene.chute.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.sauce_chute")
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
    except BaseException as exc:  # noqa: BLE001 — die loudly, never hang until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(exc).__name__}: {exc})", flush=True)
        os._exit(1)
