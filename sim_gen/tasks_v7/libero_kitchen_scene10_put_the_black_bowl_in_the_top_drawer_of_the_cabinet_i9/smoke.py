"""Smoke / rubric-REJECTION battery for CarouselAirlockScene (sim_gen task
`libero_kitchen_scene10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_i9`) —
NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — drop the ball through the window under gravity, then
drive the rotor with a bounded torque — is the acceptance evidence that the rubric
ACCEPTS a correct outcome). Every teleport here is instrumentation that CONSTRUCTS a
wrong (or partial) outcome and asserts the rubric REJECTS it; no probe in this battery
ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: rotor resting in the well, balls
                            on the floor; score ~0 at rest, no success;
  3-4. randomization      — READBACK over 6 seeded resets: housing xy + yaw, rotor (vane)
                            yaw, ball xy, decoy xy, and the approach baseline d0 all move;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's whole plan ("put the object into the fixture's
                            compartment, done") = ball dropped through the loading window
                            into the covered well and LEFT there: verified in the annulus,
                            load latch set, yet NOT success, score <= 0.45;
  7.  direct-to-basin     — ball teleported straight into the GREEN basin, settled:
                            verified physically in the basin and still, yet NOT success
                            (no load, no transit — the pathway latches), score <= 0.15;
  8.  up-the-chute cheat  — ball dropped onto the DISCHARGE CHUTE from outside (it rolls
                            down into the basin): it passes through the covered corridor
                            region, but the transit latch stays 0 (gated on sweep progress
                            INSIDE the annulus) -> NOT success, score <= 0.15;
  9.  wrong object        — the RED decoy dropped into the basin -> in_basin(decoy) True,
                            success remains False (identity matters);
  10. near-miss           — ball resting on the ground just OUTSIDE the basin's far wall
                            -> not in the basin, NOT success;
  11. near-miss           — ball resting ON TOP of the roof (above the covered well) ->
                            not in the annulus, no load credit, NOT success;
  12. latched credit      — loading the ball (honest drop) then removing it leaves the
                            latched approach + load score unchanged (and success is gone);
  13. monotonicity        — moving the ball closer to the loading window latches strictly
                            more approach credit than a farther placement;
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_i9.smoke --headless
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


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.carousel_airlock")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.05, -0.95, 0.80)) + o),
                                tuple(np.array((-0.10, 0.05, 0.05)) + o),
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

    def housing_yaw() -> float:
        return float(scene._yaw_of(scene.housing)[0])

    def local_to_world(lx: float, ly: float) -> tuple[float, float]:
        w = scene._housing_world_xy(lx, ly)[0]
        return (float(w[0]) - float(scene.env_origins[0, 0]),
                float(w[1]) - float(scene.env_origins[0, 1]))

    def report(tag: str) -> None:
        bl = scene._housing_local(scene.ball.data.root_pos_w)[0]
        r = math.hypot(float(bl[0]), float(bl[1]))
        s, ok = judge()
        print(f"[smoke] {tag:16s} | ball_local=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) r={r:.3f} ann={bool(scene.in_annulus(scene.ball)[0])} "
              f"basin={bool(scene.in_basin(scene.ball)[0])} "
              f"appr={float(scene.approach_latch[0]):.2f} "
              f"load={float(scene.load_latch[0]):.2f} sweep={float(scene.sweep_latch[0]):.2f} "
              f"transit={float(scene.transit_latch[0]):.2f} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_body(body, x: float, y: float, z: float, settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def drop_through_window(body) -> None:
        """The honest loading move solve.py performs: hover 150 mm up above the open
        window (azimuth chosen away from the parked vane), then free fall."""
        vane_rel = _wrap(float(scene._yaw_of(scene.rotor)[0]) - housing_yaw())
        cands = [0.0, math.radians(-28.0), math.radians(28.0)]
        phi = next(p for p in cands if abs(_wrap(vane_rel - p)) > math.radians(25.0))
        wx, wy = local_to_world(c.load_r * math.cos(phi), c.load_r * math.sin(phi))
        place_body(body, wx, wy, 0.150, settle_steps=120)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    ball_z = float((scene.ball.data.root_pos_w - scene.env_origins)[0, 2])
    rotor_z = float((scene.rotor.data.root_pos_w - scene.env_origins)[0, 2])
    fin0 = bool(torch.isfinite(scene.rotor.data.root_state_w).all()
                and torch.isfinite(scene.ball.data.root_state_w).all()
                and torch.isfinite(scene.decoy.data.root_state_w).all())
    check("settle: states finite; ball on the floor and rotor resting in the well "
          "(readback heights)",
          fin0 and abs(ball_z - c.ball_r) < 0.008 and abs(rotor_z - c.floor_t) < 0.010
          and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        hp = (scene.housing.data.root_pos_w - scene.env_origins)[0]
        hy = housing_yaw()
        ry = float(scene._yaw_of(scene.rotor)[0])
        b = (scene.ball.data.root_pos_w - scene.env_origins)[0]
        d = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(hp[0]), float(hp[1]), hy, ry, float(b[0]), float(b[1]),
                      float(d[0]), float(d[1]), float(scene.d0[0])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (housing_x, housing_y, housing_yaw, rotor_yaw, "
          f"ball_x, ball_y, decoy_x, decoy_y, d0):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: housing xy + yaw vary across seeded resets (readback)",
          spread[0] > 0.005 and spread[1] > 0.005 and spread[2] > 0.2)
    check("randomization: rotor (vane) yaw, ball xy, decoy xy, and the approach "
          "baseline d0 vary across seeded resets (readback)",
          spread[3] > 0.2 and spread[4] > 0.005 and spread[5] > 0.005
          and spread[6] > 0.005 and spread[7] > 0.005 and spread[8] > 0.005)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy ===========================================
    # The seed's whole plan is "put the object into the fixture's compartment — done".
    # Here that is exactly the loading step: ball dropped through the window into the
    # covered well and LEFT there. The rubric must refuse it as an end state.
    env.reset(seed=41)
    step(10)
    drop_through_window(scene.ball)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (ball into the fixture's compartment, nothing else): verified "
          "IN the covered annulus with the load latch set, yet NOT success, score <= 0.45",
          bool(scene.in_annulus(scene.ball)[0]) and float(scene.load_latch[0]) > 0.5
          and not ok and s <= 0.45)

    # =========================== 7. direct-to-basin teleport ================================
    env.reset(seed=51)
    step(10)
    bx, by = local_to_world(c.basin_cx, 0.0)
    place_body(scene.ball, bx, by, 0.120, settle_steps=90)
    report("direct-basin")
    s, ok = judge()
    check("direct-to-basin teleport: ball verified physically IN the basin and settled, "
          "yet NOT success (load + transit pathway latches unearned), score <= 0.15",
          bool(scene.in_basin(scene.ball)[0]) and bool(scene.settled()[0])
          and not ok and s <= 0.15)

    # =========================== 8. up-the-chute cheat ======================================
    # Skipping the machine: ball dropped onto the DISCHARGE CHUTE from outside. It
    # physically crosses the covered corridor region and rolls into the basin, but the
    # transit latch is gated on sweep progress INSIDE the annulus and must stay 0.
    env.reset(seed=61)
    step(10)
    cx, cy = local_to_world(-0.190, 0.0)  # over the chute, under the hood lip
    place_body(scene.ball, cx, cy, 0.080, settle_steps=10)
    in_corridor = bool(scene.in_transit(scene.ball)[0])
    step(170)  # rolls down the chute into the basin
    report("up-the-chute")
    s, ok = judge()
    check("up-the-chute shortcut: ball passed through the covered corridor region "
          "(verified) and ended in the basin, yet transit latch stayed 0 (sweep gate) "
          "-> NOT success, score <= 0.15",
          in_corridor and bool(scene.in_basin(scene.ball)[0])
          and float(scene.transit_latch[0]) < 0.5 and not ok and s <= 0.15)

    # =========================== 9. wrong object ============================================
    env.reset(seed=71)
    step(10)
    bx, by = local_to_world(c.basin_cx, 0.0)
    place_body(scene.decoy, bx, by, 0.120, settle_steps=90)
    report("wrong-object")
    s, ok = judge()
    check("wrong object: the RED decoy rests in the basin (verified) — success remains "
          "False (identity control)",
          bool(scene.in_basin(scene.decoy)[0]) and not ok)

    # =========================== 10. near-miss: beside the basin ============================
    env.reset(seed=81)
    step(10)
    nx, ny = local_to_world(c.basin_x_inner[0] - 0.060, 0.0)  # just beyond the far wall
    place_body(scene.ball, nx, ny, c.ball_r + 0.002, settle_steps=60)
    report("beside-basin")
    s, ok = judge()
    check("near-miss: ball resting on the ground just OUTSIDE the basin's far wall — "
          "not in the basin, NOT success",
          not bool(scene.in_basin(scene.ball)[0]) and not ok)

    # =========================== 11. near-miss: parked on the roof ==========================
    env.reset(seed=91)
    step(10)
    rx, ry = local_to_world(0.0, 0.100)  # above the roof annulus, away from the window
    place_body(scene.ball, rx, ry, c.ring_z1 + c.ball_r + 0.003, settle_steps=60)
    report("on-roof")
    s, ok = judge()
    bl = scene._housing_local(scene.ball.data.root_pos_w)[0]
    check("near-miss: ball parked ON TOP of the roof (verified above the roof plane) — "
          "not in the annulus, no load credit, NOT success",
          float(bl[2]) > c.ring_z1 - 0.005 and not bool(scene.in_annulus(scene.ball)[0])
          and float(scene.load_latch[0]) < 0.5 and not ok)

    # =========================== 12. latched credit survives regression =====================
    env.reset(seed=101)
    step(10)
    drop_through_window(scene.ball)
    report("loaded")
    s_in, _ = judge()
    place_body(scene.ball, 0.45, -0.35, c.ball_r + 0.002, settle_steps=60)
    report("ball-removed")
    s_out, ok = judge()
    check("latched credit: removing the loaded ball leaves the latched approach + load "
          "score unchanged (and success is gone)",
          s_in >= 0.28 and abs(s_out - s_in) < 0.02 and not ok
          and not bool(scene.in_annulus(scene.ball)[0]))

    # =========================== 13. approach monotonicity ==================================
    env.reset(seed=111)
    step(5)
    win = scene._housing_world_xy(c.load_r, 0.0)[0]
    wx = float(win[0]) - float(scene.env_origins[0, 0])
    wy = float(win[1]) - float(scene.env_origins[0, 1])
    b0 = (scene.ball.data.root_pos_w - scene.env_origins)[0]
    dx, dy = wx - float(b0[0]), wy - float(b0[1])
    place_body(scene.ball, float(b0[0]) + 0.5 * dx, float(b0[1]) + 0.5 * dy, 0.30,
               settle_steps=3)
    a_half = float(scene.approach_latch[0])
    place_body(scene.ball, float(b0[0]) + 0.85 * dx, float(b0[1]) + 0.85 * dy, 0.30,
               settle_steps=3)
    a_near = float(scene.approach_latch[0])
    check("monotonicity: moving the ball closer to the loading window latches strictly "
          f"more approach credit ({a_half:.3f} < {a_near:.3f})", a_half + 0.10 < a_near)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.housing.data.root_state_w).all()
           and torch.isfinite(scene.rotor.data.root_state_w).all()
           and torch.isfinite(scene.ball.data.root_state_w).all()
           and torch.isfinite(scene.decoy.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.carousel_airlock")
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
