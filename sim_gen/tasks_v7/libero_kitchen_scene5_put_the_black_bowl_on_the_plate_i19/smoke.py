"""Smoke / rubric-REJECTION battery for MoatCausewayScene (sim_gen task
`libero_kitchen_scene5_put_the_black_bowl_on_the_plate_i19`) — NullRobot, teleported
probe states, RECORDED.

This is NOT a solution (solve.py — lay the plank by gravity release, push the cube
across with a bounded force — is the acceptance evidence that the rubric ACCEPTS a
correct outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partial) outcome and asserts the rubric REJECTS it; no probe in this battery ever
reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: cube at rest on the start
                            deck (readback height), planks on the ground; score ~0;
  3-4. randomization      — READBACK over 8 seeded resets: layout centre xy + FREE
                            yaw + the moat gap all vary; cube deck xy, plank + decoy
                            ground poses vary and the plank spawns on BOTH sides;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's whole plan ("carry the object and set it down
                            at the goal") = the red cube WRITTEN directly onto the
                            yellow pad, settled: on_pad verified True, yet the
                            crossing pathway latch stays 0 -> NOT success, score ~0;
  7.  moat drop           — the cube dropped into the bare moat ends BELOW deck level
                            (the moat is deeper than the cube: unrecoverable), no
                            credit ever latched (falling through the crossing band
                            does not count: |v_z| gate + no live bridge);
  8.  decoy cannot span   — the WHITE decoy released over the moat centre falls to
                            the ground (shorter than every gap), span latch stays 0;
  9.  bridge-only         — honest plank lay (hover release + gravity): spanning
                            verified True, score ~0.25, no success;
  10. short lay           — plank released with one end short of the far rim:
                            cantilever tips into the moat, spanning False, score ~0;
  11. mid-bridge stop     — on a spanning bridge, the cube honestly dropped mid-span:
                            the crossing latch fires (supported over the void on a
                            live bridge) but the cube is not on the pad -> NOT
                            success, score <= 0.56;
  12. arrived-not-on-pad  — cube placed on the green deck just OUTSIDE the pad
                            rectangle: arrival latch fires, yet NOT success,
                            score <= 0.76;
  13. latched credit      — removing the cube to the ground leaves the latched
                            span + cross + arrive score unchanged (success gone);
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene5_put_the_black_bowl_on_the_plate_i19.smoke --headless
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
    env = ENVS.get("simgen.moat_causeway")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.10, -1.10, 0.90)) + o),
                                tuple(np.array((0.0, 0.0, 0.10)) + o),
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

    def place_mid(body, lx: float, ly: float, lz: float, settle_steps: int = 30) -> None:
        """Kinematic probe placement in the MOAT frame (instrumentation, not a
        solution) + REAL physics steps before judging (the zero-step trap)."""
        mp, mq = scene.mid_frame()
        off = torch.tensor([[lx, ly, lz]], device=device)
        w = mp[0:1] + quat_apply(mq[0:1], off)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = w
        st[:, 3:7] = mq[0:1]
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def place_tgt(body, lx: float, ly: float, lz: float, settle_steps: int = 30) -> None:
        """Probe placement in the TARGET-platform frame + real physics steps."""
        tp = scene.target_ped.data.root_pos_w[0:1]
        tq = scene.target_ped.data.root_quat_w[0:1]
        off = torch.tensor([[lx, ly, lz]], device=device)
        w = tp + quat_apply(tq, off)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = w
        st[:, 3:7] = tq
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def lay_bridge(settle_steps: int = 150) -> None:
        """The honest bridging move solve.py performs: hover the plank 30 mm above
        deck-top height over the moat centre, aligned with the crossing axis, then
        free fall + settle on the two rims."""
        place_mid(scene.plank, 0.0, 0.0, 0.030, settle_steps=settle_steps)

    def report(tag: str) -> None:
        ml = scene.mid_local(scene.cube.data.root_pos_w)[0]
        pl = scene.mid_local(scene.plank.data.root_pos_w)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | gap={float(scene.gap()[0]) * 1000:.0f}mm "
              f"cube_mid=({float(ml[0]):+.3f},{float(ml[1]):+.3f},{float(ml[2]):+.3f}) "
              f"plank_mid=({float(pl[0]):+.3f},{float(pl[1]):+.3f},{float(pl[2]):+.3f}) "
              f"span={bool(scene.spanning()[0])} on_pad={bool(scene.on_pad()[0])} "
              f"on_deck={bool(scene.on_target_deck()[0])} "
              f"span_l={float(scene.span_latch[0]):.2f} "
              f"cross_l={float(scene.cross_latch[0]):.2f} "
              f"arr_l={float(scene.arrive_latch[0]):.2f} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def states_finite() -> bool:
        return bool(torch.isfinite(scene.start_ped.data.root_state_w).all()
                    and torch.isfinite(scene.target_ped.data.root_state_w).all()
                    and torch.isfinite(scene.plank.data.root_state_w).all()
                    and torch.isfinite(scene.decoy.data.root_state_w).all()
                    and torch.isfinite(scene.cube.data.root_state_w).all())

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    ml = scene.mid_local(scene.cube.data.root_pos_w)[0]
    pl = scene.mid_local(scene.plank.data.root_pos_w)[0]
    check("settle: states finite; cube at rest ON the start deck (readback height, "
          "start side); planks on the ground",
          states_finite() and abs(float(ml[2]) - c.cube_s / 2) < 0.012
          and float(ml[0]) < -float(scene.gap()[0]) / 2 and bool(scene.settled()[0])
          and float(pl[2]) < -0.05)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(5)
        mp, _ = scene.mid_frame()
        ctr = (mp - scene.env_origins)[0]
        yw = float(scene._yaw_of(scene.target_ped)[0])
        g = float(scene.gap()[0])
        cm = scene.mid_local(scene.cube.data.root_pos_w)[0]
        pm = scene.mid_local(scene.plank.data.root_pos_w)[0]
        dm = scene.mid_local(scene.decoy.data.root_pos_w)[0]
        reads.append((float(ctr[0]), float(ctr[1]), yw, g, float(cm[0]), float(cm[1]),
                      float(pm[0]), float(pm[1]), float(dm[0]), float(dm[1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (ctr_x, ctr_y, yaw, gap, cube_x, cube_y, "
          f"plank_x, plank_y, decoy_x, decoy_y):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: layout centre xy + free yaw + the moat gap all vary across "
          "seeded resets (readback)",
          spread[0] > 0.01 and spread[1] > 0.01 and spread[2] > 0.5
          and spread[3] > 0.005)
    both_sides = bool((arr[:, 7] > 0).any() and (arr[:, 7] < 0).any())
    check("randomization: cube deck xy, plank + decoy ground poses vary and the plank "
          "spawns on BOTH sides of the moat across seeds (readback)",
          spread[4] > 0.01 and spread[5] > 0.03 and spread[6] > 0.02
          and spread[8] > 0.02 and both_sides)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy ===========================================
    # The seed's whole plan is "carry the object and set it down at the goal". Here that
    # is the red cube written directly onto the yellow pad. The rubric must refuse it:
    # the cube never crossed the moat on a bridge.
    env.reset(seed=41)
    step(10)
    place_tgt(scene.cube, c.pad_cx, 0.0, c.pad_t + c.cube_s / 2 + 0.003, settle_steps=60)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (cube set down directly ON the yellow pad, settled): on_pad "
          "verified True, yet the crossing pathway latch stays 0 -> NOT success, score ~0",
          bool(scene.on_pad()[0]) and bool(scene.settled()[0])
          and float(scene.cross_latch[0]) < 0.5 and not ok and s <= 0.02)

    # =========================== 7. moat drop ===============================================
    # The moat is impassable for the bare cube: dropped in, it ends BELOW deck level.
    # Falling through the crossing z band earns nothing (|v_z| gate + no live bridge).
    env.reset(seed=51)
    step(10)
    place_mid(scene.cube, 0.0, 0.0, 0.100, settle_steps=90)
    report("moat-drop")
    ml = scene.mid_local(scene.cube.data.root_pos_w)[0]
    s, ok = judge()
    check("moat drop: the cube dropped into the bare moat ends BELOW deck level "
          "(unrecoverable), no credit latched, NOT success",
          float(ml[2]) < -0.02 and float(scene.cross_latch[0]) < 0.5
          and not ok and s <= 0.02)

    # =========================== 8. decoy cannot span =======================================
    env.reset(seed=61)
    step(10)
    place_mid(scene.decoy, 0.0, 0.0, 0.030, settle_steps=90)
    report("decoy-span")
    dm = scene.mid_local(scene.decoy.data.root_pos_w)[0]
    s, ok = judge()
    check("decoy cannot span: the WHITE plank released over the moat centre falls to "
          "the ground (shorter than every gap), span latch stays 0",
          float(dm[2]) < -0.05 and float(scene.span_latch[0]) < 0.5
          and not ok and s <= 0.02)

    # =========================== 9. bridge-only =============================================
    env.reset(seed=71)
    step(10)
    lay_bridge()
    report("bridge-only")
    s, ok = judge()
    check("bridge-only: honest plank lay spans (verified live), score ~0.25, no success",
          bool(scene.spanning()[0]) and 0.24 <= s <= 0.26 and not ok)

    # =========================== 10. short lay ==============================================
    env.reset(seed=81)
    step(10)
    g = float(scene.gap()[0])
    # centre the plank over the start rim: the far end stops short of the target rim
    place_mid(scene.plank, -(g / 2) - 0.02, 0.0, 0.030, settle_steps=120)
    report("short-lay")
    s, ok = judge()
    check("short lay: plank released with one end short of the far rim tips — "
          "never spanning, score ~0",
          not bool(scene.spanning()[0]) and float(scene.span_latch[0]) < 0.5
          and not ok and s <= 0.02)

    # =========================== 11. mid-bridge stop ========================================
    env.reset(seed=91)
    step(10)
    lay_bridge()
    place_mid(scene.cube, 0.0, 0.0, 0.100, settle_steps=90)
    report("mid-bridge")
    ml = scene.mid_local(scene.cube.data.root_pos_w)[0]
    s, ok = judge()
    check("mid-bridge stop: cube dropped onto the spanning bridge rests over the void "
          "(crossing latch fires) but is NOT on the pad -> NOT success, score <= 0.56",
          bool(scene.spanning()[0]) and abs(float(ml[0])) < g / 2
          and float(scene.cross_latch[0]) > 0.5 and not bool(scene.on_pad()[0])
          and not ok and s <= 0.56)

    # =========================== 12. arrived but not on the pad =============================
    x_short = c.pad_cx - c.pad_l / 2 - 0.012  # on the green deck, outside the pad rect
    place_tgt(scene.cube, x_short, 0.0, c.cube_s / 2 + 0.003, settle_steps=60)
    report("short-of-pad")
    s, ok = judge()
    check("arrived-not-on-pad: cube on the green deck just outside the pad rectangle "
          "(arrival latch fires), yet NOT success, score <= 0.76",
          bool(scene.on_target_deck()[0]) and not bool(scene.on_pad()[0])
          and float(scene.arrive_latch[0]) > 0.5 and not ok and s <= 0.76)
    s_full = s

    # =========================== 13. latched credit survives regression =====================
    place_mid(scene.cube, 0.0, 0.60, 0.100 - c.ped_h, settle_steps=60)  # off to the ground
    report("cube-removed")
    s_out, ok = judge()
    check("latched credit: removing the cube to the ground leaves the latched "
          "span + cross + arrive score unchanged (and success is gone)",
          s_full >= 0.70 and abs(s_out - s_full) < 0.02 and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", states_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.moat_causeway")
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
