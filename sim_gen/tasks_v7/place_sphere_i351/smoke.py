"""Smoke / rubric-REJECTION battery for BallPumpScene (sim_gen task
`place_sphere_i351`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — whites hover-released into the intake funnel so
the shaft column drives the chain until the red ball tips into the exit tray, then
the red ball hover-released over the goal bin — is the acceptance evidence that the
rubric ACCEPTS a correct outcome). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome as a settled state and asserts the rubric
REJECTS it; no probe in this battery ever reaches success(), and a final audit
check asserts exactly that.

   1-2. settle/no-NaN     — reset layout settles finite: red ball at rest lodged at
                            its sampled depth x0 inside the flat passage, all 9
                            whites at rest in the supply tray, everything still,
                            score ~0, no success;
   3-4. randomization     — READBACK over 8 seeded resets: pump yaw spans a wide
                            arc + xy jitter real, tray side flips, x0 varies;
   5.  null policy        — 240 idle steps -> score ~0, no success;
   6.  roof perch         — red ball placed at rest ON TOP of the machine (the flat
                            roof EXTERIOR, in the trough between the side walls —
                            inside the conduit's xy footprint): the roofed-run
                            z-bound excludes it, progress stays 0, score 0;
   7.  wrong object       — a white ball settled in the GOAL BIN and another in the
                            EXIT TRAY: the rubric judges only the red ball -> 0;
   8.  exit-tray shortcut — red ball constructed at rest IN the exit tray without
                            ever traversing the conduit: eject_now reads True but
                            the eject latch is gated on FULL conduit progress ->
                            nothing latches, score ~0;
   9.  bin shortcut       — red ball dropped straight into the GOAL BIN (the seed's
                            own literal strategy: carry the sphere, place it on the
                            goal): deliver_now True but the deliver latch is gated
                            on the eject latch -> success False, score ~0;
  10.  feed non-vacuity   — 2 whites genuinely fed into the funnel: the chain
                            advances, progress credit rises WELL above baseline (the
                            zeros in 8/9 are gates, not a dead rubric), no success;
  11.  latched credit     — the mid-conduit red ball then yanked out to open
                            ground: the latched progress score HOLDS, no success;
  12.  eject control      — fresh episode fed to full ejection (genuine physics):
                            eject latch earned, score == the 0.70 non-success cap,
                            success still False (the red ball is never delivered —
                            this battery never constructs success);
  13.  rejection audit    — success() was never True at ANY judged point;
  14.  final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.place_sphere_i351.smoke --headless
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
    env = ENVS.get("simgen.ball_pump")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.40, -0.95, 0.90)) + o),
                                tuple(np.array((0.50, 0.03, 0.18)) + o),
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

    def pump_pose() -> tuple[torch.Tensor, float]:
        pp = (scene.pump.data.root_pos_w - scene.env_origins)[0]
        q = scene.pump.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return pp, yaw

    def report(tag: str) -> None:
        rl = scene._local(scene.red, scene.pump)[0]
        s, ok = judge()
        print(f"[smoke] {tag:18s} | red_l=({float(rl[0]):+.3f},{float(rl[1]):+.3f},"
              f"{float(rl[2]):.4f}) prog={float(scene._prog[0]):.3f} "
              f"eject_latch={bool(scene._eject_latch[0])} "
              f"deliver_latch={bool(scene._deliver_latch[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_fix_local(body, fix, lx: float, ly: float, z: float) -> None:
        """Zero-velocity root-state write at `fix`-fixture (lx, ly), world z."""
        fp = (fix.data.root_pos_w - scene.env_origins)[0]
        q = fix.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        wx = float(fp[0]) + math.cos(yaw) * lx - math.sin(yaw) * ly
        wy = float(fp[1]) + math.sin(yaw) * lx + math.cos(yaw) * ly
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = wx, wy, z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def place_world(body, x: float, y: float, z: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def feed_white(i: int) -> None:
        """GENUINE feed: hover-release white i above the funnel (airborne write),
        then let gravity + the chain physics act."""
        place_fix_local(scene.whites[i], scene.pump, c.shaft_cx, 0.0, 0.320)
        step(180)

    def red_local() -> torch.Tensor:
        return scene._local(scene.red, scene.pump)[0]

    def tray_local(body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - scene.tray.data.root_pos_w
        return quat_apply_inverse(scene.tray.data.root_quat_w, rel)[0]

    def in_supply_tray(body) -> bool:
        loc = tray_local(body)
        return (abs(float(loc[0])) < c.tray_half and abs(float(loc[1])) < c.tray_half
                and abs(float(loc[2]) - (0.008 + c.ball_r)) < 0.02)

    def states_finite() -> bool:
        ok = bool(torch.isfinite(scene.red.data.root_state_w).all()
                  and torch.isfinite(scene.pump.data.root_state_w).all()
                  and torch.isfinite(scene.tray.data.root_state_w).all()
                  and torch.isfinite(scene.bin.data.root_state_w).all())
        for wb in scene.whites:
            ok = ok and bool(torch.isfinite(wb.data.root_state_w).all())
        return ok

    def all_still() -> bool:
        ok = float(scene.red.data.root_lin_vel_w[0].norm()) < c.settle_speed
        for wb in scene.whites:
            ok = ok and float(wb.data.root_lin_vel_w[0].norm()) < c.settle_speed
        return ok

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    rl = red_local()
    x0 = float(scene._x0[0])
    whites_in_tray = all(in_supply_tray(wb) for wb in scene.whites)
    check("settle: states finite, red ball at rest lodged at its sampled depth x0 in "
          f"the flat passage (x={float(rl[0]):+.4f} vs x0={x0:+.4f}), all 9 whites at "
          "rest in the supply tray, everything still",
          states_finite() and abs(float(rl[0]) - x0) < 0.01 and float(rl[2]) < 0.05
          and whites_in_tray and all_still())
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(2)
        pp, pyaw = pump_pose()
        tp = (scene.tray.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(pp[0]), float(pp[1]), math.degrees(pyaw),
                      float(tp[1]) > 0.0, float(scene._x0[0])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (pump_x, pump_y, pump_yaw_deg, "
          f"tray_side_pos_y, x0):\n{arr}", flush=True)
    # yaw span: compare unit vectors (degrees wrap at +/-180)
    yaws = np.radians(arr[:, 2])
    yaw_spread = float(np.ptp(np.cos(yaws)) + np.ptp(np.sin(yaws)))
    check("randomization: pump yaw spans a wide arc, xy jitter real, tray side flips "
          "(readback)",
          yaw_spread > 0.8
          and float((arr[:, 0:2].max(axis=0) - arr[:, 0:2].min(axis=0)).max()) > 0.01
          and 0.0 < arr[:, 3].mean() < 1.0)
    check("randomization: red start depth x0 varies across seeded resets (span > 5 mm)",
          float(arr[:, 4].max() - arr[:, 4].min()) > 0.005)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. roof perch: on the machine, not in it ===================
    # Red ball placed at rest ON the flat-roof EXTERIOR, in the trough between the
    # side walls — inside the conduit's xy footprint (x ~ -0.023, |y| < cond_y) but
    # ABOVE the roofed run. The roofed-run z-bound must exclude it from conduit
    # membership so it earns no progress.
    torch.manual_seed(41)
    env.reset()
    step(10)
    place_fix_local(scene.red, scene.pump, -0.0225, 0.0, 0.0865)  # roof top 0.065 + r + eps
    step(120)
    report("roof-perch")
    s, ok = judge()
    rl = red_local()
    in_cond = bool(scene._in_conduit(scene._local(scene.red, scene.pump))[0])
    check("roof perch: red ball at rest ON TOP of the machine roof, inside the conduit "
          f"xy footprint (readback z={float(rl[2]):.4f}) — conduit membership False, "
          "progress stays 0, score ~0, no success",
          float(rl[2]) > 0.075 and -0.0395 < float(rl[0]) < -0.004 and not in_cond
          and float(scene._prog[0]) < 0.01 and s <= 0.02 and not ok)

    # =========================== 7. wrong object ============================================
    torch.manual_seed(51)
    env.reset()
    step(10)
    place_fix_local(scene.whites[0], scene.bin, 0.0, 0.0, 0.008 + c.ball_r + 0.05)
    place_fix_local(scene.whites[1], scene.pump, 0.15, 0.0, 0.010 + c.ball_r + 0.03)
    step(120)
    report("wrong-object")
    s, ok = judge()
    check("wrong object: a WHITE ball settled in the goal bin and another in the exit "
          "tray — the rubric judges only the red ball: nothing latches, score 0",
          not bool(scene._eject_latch[0]) and not bool(scene._deliver_latch[0])
          and s <= 0.02 and not ok)

    # =========================== 8. exit-tray shortcut ======================================
    # Red ball constructed at rest IN the exit tray without ever traversing the
    # conduit: eject_now reads True (it IS in the window) but the eject latch is
    # gated on full conduit progress — nothing latches.
    torch.manual_seed(61)
    env.reset()
    step(10)
    place_fix_local(scene.red, scene.pump, 0.15, 0.0, 0.010 + c.ball_r + 0.02)
    step(120)
    report("tray-shortcut")
    s, ok = judge()
    eject_now = bool(scene._eject_now()[0])
    check("exit-tray shortcut: red ball at rest in the exit tray WITHOUT conduit "
          f"passage — eject_now={eject_now} (in the window) yet the progress gate "
          "keeps the latch off: score ~0, no success",
          eject_now and not bool(scene._eject_latch[0])
          and float(scene._prog[0]) < 0.01 and s <= 0.02 and not ok)

    # =========================== 9. bin shortcut (the seed's own strategy) ==================
    # The seed's literal plan — carry the red sphere and place it on/at the goal —
    # executed directly: red dropped into the goal bin. deliver_now True, but the
    # deliver latch is gated on the eject latch (out THROUGH the machine).
    torch.manual_seed(71)
    env.reset()
    step(10)
    place_fix_local(scene.red, scene.bin, 0.0, 0.0, 0.008 + c.ball_r + 0.05)
    step(120)
    report("bin-shortcut")
    s, ok = judge()
    deliver_now = bool(scene._deliver_now()[0])
    check("bin shortcut (seed strategy): red ball dropped straight into the GOAL BIN "
          f"and settled — deliver_now={deliver_now} yet the eject gate keeps success "
          "False and score ~0",
          deliver_now and not bool(scene._deliver_latch[0])
          and not bool(scene._eject_latch[0]) and s <= 0.02 and not ok)

    # =========================== 10-11. feed non-vacuity + latched credit ===================
    # Two whites genuinely fed into the funnel: the chain advances and progress
    # credit rises well above the baseline — proving the zeros of checks 8/9 are
    # order gates, not a dead rubric. Then the mid-conduit red ball is yanked out
    # to open ground: the latched progress credit holds.
    torch.manual_seed(81)
    env.reset()
    step(30)
    feed_white(0)
    feed_white(1)
    step(60)
    report("fed-2")
    s_a, ok_a = judge()
    check("feed non-vacuity: 2 whites genuinely fed into the funnel advance the chain "
          f"— progress credit rises well above baseline (score {s_a:.3f} > 0.05), "
          "no success",
          s_a > 0.05 and not ok_a and float(scene._prog[0]) > 0.05)
    place_world(scene.red, 0.95, 0.0, c.ball_r + 0.002)  # yank to open ground
    step(60)
    report("yanked")
    s_b, ok_b = judge()
    check("latched credit: mid-conduit red ball yanked out to open ground — the "
          f"latched progress score holds ({s_a:.3f} -> {s_b:.3f}), no success",
          abs(s_a - s_b) < 1e-3 and not ok_b)

    # =========================== 12. eject control: full genuine run, capped ================
    # Fresh episode fed to full ejection by genuine physics (the solve's P1). The
    # eject latch is earned, the score sits exactly at the 0.70 non-success cap,
    # and success stays False — the red ball is never delivered (this battery
    # never constructs success).
    torch.manual_seed(91)
    env.reset()
    step(30)
    ejected = False
    for i in range(c.n_white):
        feed_white(i)
        rl = red_local()
        if float(rl[0]) > c.eject_x[0] and float(rl[2]) < c.eject_z[1]:
            ejected = True
            print(f"[smoke] red ball ejected after {i + 1} feeds", flush=True)
            break
    latched = False
    if ejected:
        for _ in range(30):
            step(20)
            if bool(scene._eject_latch[0]):
                latched = True
                break
    report("eject-control")
    s, ok = judge()
    check("eject control: full genuine feed run ejects the red ball, the eject latch "
          f"is earned, score == the 0.70 non-success cap (got {s:.4f}), success still "
          "False (never delivered)",
          ejected and latched and abs(s - 0.70) < 5e-3 and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", states_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ball_pump")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(okc for _nm, okc in checks)
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
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001 - die loudly, never idle to the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL (exception: {e})", flush=True)
        os._exit(1)
