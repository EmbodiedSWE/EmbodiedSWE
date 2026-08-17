"""Smoke / rubric-REJECTION battery for TunnelRelayScene (sim_gen task
`slide_block_to_target_i352`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — four contact-pushed feed cycles that eject the
cargo into the pen — is the acceptance evidence that the rubric ACCEPTS a correct
outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partial) outcome and asserts the rubric REJECTS it; no probe in this battery ever
reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN     — reset layout settles finite: cargo riding the deck top
                           inside the bore, feeders at rest on the floor, score ~0;
  3-4. randomization     — READBACK over 6 seeded resets: rig xy + yaw, cargo
                           depth, feeder scatter + free yaw all really vary;
  5.  null policy        — 240 idle steps -> score ~0, no success;
  6.  seed strategy      — the nearest executable version of the SEED's plan (push
                           the goal cube along a surface to the target): the cargo
                           settled ON THE FLOOR flush beside the pen wall -> NOT
                           success, score ~0 (the pen is entered only from above);
  7.  wrong object       — a FEEDER dropped into the pen, settled on its floor ->
                           NOT success (only the red cargo is judged);
  8.  stacked            — cargo resting ON TOP of a feeder inside the pen (centre
                           above the resting band) -> NOT success;
  9.  partial extraction — cargo settled INSIDE the bore just short of the cliff
                           (the 3-feeder outcome, constructed) -> NOT success,
                           score <= 0.42;
  10. backward removal   — cargo settled on the open APRON behind the mouth (the
                           only reachable opening) -> NOT success, score ~0;
  11. overshoot          — cargo settled on the floor BEHIND the pen's back wall
                           -> NOT success (containment walls are load-bearing);
  12. airborne           — cargo in free fall directly over the pen, judged
                           mid-air -> NOT success (resting-height gate; relocated
                           before it can land);
  13. latched credit     — in-bore progress credit survives the cargo being put
                           back at its spawn depth: latched score UNCHANGED, still
                           no success;
  14. THREE FEEDERS SHORT — the resource arithmetic, demonstrated with REAL
                           CONTACT PHYSICS: three feeders pushed in exactly like
                           solve.py, each to the mouth plane -> the cargo stalls
                           inside the bore short of the cliff, NOT success,
                           score <= 0.46;
  15. rejection audit    — success() was never True at ANY judged point;
  16. final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
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
# RTX recipe: kit mis-decodes some driver versions and silently rejects RTX -> the
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
    from . import scene as scene_mod  # noqa: F401  (registers simgen.tunnel_relay)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tunnel_relay")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    fl = c.feeder_size[0]
    b = c.cargo_size

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.85, -0.75, 0.65)) + o),
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
        sc, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return sc, ok

    def cargo_l() -> torch.Tensor:
        return scene.cargo_local()[0]

    def report(tag: str) -> None:
        p = cargo_l()
        sc, ok = judge()
        print(f"[smoke] {tag:18s} | cargo_l=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):.3f}) in_pen={bool(scene.in_pen()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"prog={float(scene._prog_max[0]):.3f} eject={bool(scene._eject_ever[0])} "
              f"score={sc:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def rig_quat0() -> torch.Tensor:
        yaw = float(scene._rig_yaw[0])
        return torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
                            device=device)

    def place(body, x: float, y: float, z: float) -> None:
        """Probe constructor: teleport a body to a RIG-LOCAL pose (rig yaw
        orientation), zero velocity."""
        pos_l = torch.tensor([[float(x), float(y), float(z)]], device=device)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.to_world(pos_l, all_ids) + scene.env_origins
        st[:, 3:7] = rig_quat0()
        body.write_root_state_to_sim(st, all_ids)

    def still() -> bool:
        return bool(scene.settled()[0])

    def finite_all() -> bool:
        ok = bool(torch.isfinite(scene.cargo.data.root_state_w).all())
        for f in scene.feeders:
            ok = ok and bool(torch.isfinite(f.data.root_state_w).all())
        return ok

    def push_feeder_flush(k: int) -> bool:
        """The solve's insertion move: stage feeder k on the apron, push it with a
        velocity-capped force along the bore axis until its rear reaches the mouth
        plane (the fingertip never enters the bore)."""
        from isaaclab.utils.math import quat_apply_inverse  # noqa: F401

        place(scene.feeders[k], -0.110, 0.0, c.deck_top + c.feeder_size[2] / 2 + 0.003)
        step(30)
        feeder = scene.feeders[k]
        yaw = float(scene._rig_yaw[0])
        d_w = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0], device=device)
        zero_w = torch.zeros(n, 1, 3, device=device)
        f_mag, moved = 1.5, False
        x_mark, mark_i = -1.0, 0
        for i in range(3000):
            pf = scene.feeder_local(k)[0]
            if float(pf[0]) - fl / 2 >= c.x_mouth - 0.006:
                moved = True
                break
            v_along = float(torch.dot(feeder.data.root_lin_vel_w[0], d_w))
            push = f_mag if v_along < 0.05 else 0.0
            f_w = (d_w * push).view(1, 1, 3).expand(n, 1, 3)
            feeder.set_external_force_and_torque(
                f_w.contiguous(), zero_w, env_ids=all_ids, is_global=True)
            env.step(no_action)
            x_now = float(pf[0])
            if x_now > x_mark + 0.004:
                x_mark, mark_i = x_now, i
            elif i - mark_i > 240 and f_mag < 12.0:
                f_mag += 1.5
                mark_i = i
        feeder.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
        step(60)
        return moved

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    report("reset")
    step(90)
    report("show")
    p = cargo_l()
    feeders_ok = all(
        abs(float(scene.feeder_local(i)[0][2]) - c.feeder_size[2] / 2) < 0.008
        for i in range(4))
    check("settle: states finite, cargo riding the deck top inside the bore, feeders "
          "at rest on the floor, all still",
          finite_all() and abs(float(p[2]) - (c.deck_top + b / 2)) < 0.008
          and abs(float(p[1])) < 0.02 and c.cargo_depth[0] - 0.01 < float(p[0]) <
          c.cargo_depth[1] + 0.01 and feeders_ok and still())
    sc, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", sc <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        f0 = scene.feeder_local(0)[0]
        q = scene.feeders[0].data.root_quat_w[0]
        f0_yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        rel_yaw = (f0_yaw - float(scene._rig_yaw[0]) + math.pi) % (2 * math.pi) - math.pi
        reads.append((float(scene._rig_xy[0, 0]), float(scene._rig_xy[0, 1]),
                      math.degrees(float(scene._rig_yaw[0])), float(scene._x0[0]),
                      float(f0[0]), float(f0[1]), math.degrees(rel_yaw)))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (rig_x, rig_y, rig_yaw_deg, cargo_x0, "
          f"f0_lx, f0_ly, f0_relyaw_deg):\n{arr}", flush=True)
    rig_spread = arr[:, 0:2].max(axis=0) - arr[:, 0:2].min(axis=0)
    yaw_spread = float(arr[:, 2].max() - arr[:, 2].min())
    x0_spread = float(arr[:, 3].max() - arr[:, 3].min())
    check("randomization: rig xy spread > 10 mm on both axes, rig yaw spread > 5 deg, "
          "cargo depth spread > 6 mm and inside its band (readback)",
          float(rig_spread.min()) > 0.010 and yaw_spread > 5.0 and x0_spread > 0.006
          and all(c.cargo_depth[0] - 1e-3 <= r[3] <= c.cargo_depth[1] + 1e-3
                  for r in reads))
    f0_spread = arr[:, 4:6].max(axis=0) - arr[:, 4:6].min(axis=0)
    relyaw_spread = float(arr[:, 6].max() - arr[:, 6].min())
    check("randomization: feeder scatter xy spread > 8 mm and free yaw spread > "
          "10 deg relative to the rig (readback)",
          float(f0_spread.max()) > 0.008 and relyaw_spread > 10.0)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    sc, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          sc <= 0.02 and not ok)

    # =========================== 6. seed strategy: slide to the target ======================
    # The seed's plan is one direct planar push of the goal cube to the target. The
    # cargo here is untouchable, and the pen is enclosed on all four sides at floor
    # level — the nearest executable end state is the cube slid across the floor to
    # rest flush against the pen's outer wall.
    env.reset(seed=41)
    step(30)
    place(scene.cargo, c.x_cliff + c.pen_len / 2, c.pen_half_w + c.pen_wall_t + b / 2
          + 0.003, b / 2 + 0.002)
    step(150)
    report("seed-strategy")
    p = cargo_l()
    sc, ok = judge()
    check("seed strategy: cargo slid on the floor flush against the pen's outer wall "
          "— NOT success and score <= 0.02 (the pen is entered only from above)",
          still() and float(p[2]) < 0.05 and abs(float(p[1])) > c.pen_half_w
          and not ok and sc <= 0.02)

    # =========================== 7. wrong object in the pen =================================
    env.reset(seed=51)
    step(30)
    place(scene.feeders[0], c.x_cliff + c.pen_len / 2, 0.0, 0.15)  # dropped in from above
    step(150)
    report("feeder-in-pen")
    pf = scene.feeder_local(0)[0]
    sc, ok = judge()
    check("wrong object: a FEEDER dropped into the pen and settled on its floor — "
          "NOT success, score ~0 (only the red cargo is judged)",
          float(pf[2]) < 0.05 and abs(float(pf[1])) < c.pen_half_w
          and c.x_cliff < float(pf[0]) < c.x_cliff + c.pen_len
          and not ok and sc <= 0.02)

    # =========================== 8. stacked on a feeder in the pen ==========================
    env.reset(seed=61)
    step(30)
    place(scene.feeders[1], c.x_cliff + 0.08, 0.0, c.feeder_size[2] / 2 + 0.002)
    step(60)
    place(scene.cargo, c.x_cliff + 0.08, 0.0, c.feeder_size[2] + b / 2 + 0.003)
    step(150)
    report("stacked")
    p = cargo_l()
    sc, ok = judge()
    check("stacked: cargo resting ON TOP of a feeder inside the pen — its centre "
          "sits above the resting band, NOT success",
          still() and float(p[2]) > c.pen_z[1] + 0.005
          and c.x_cliff < float(p[0]) < c.x_cliff + c.pen_len
          and not bool(scene.in_pen()[0]) and not ok and sc < 1.0)

    # =========================== 9. partial extraction (3-feeder outcome) ===================
    env.reset(seed=71)
    step(30)
    place(scene.cargo, 0.140, 0.0, c.deck_top + b / 2 + 0.002)
    step(120)
    report("partial-extract")
    p = cargo_l()
    sc, ok = judge()
    check("partial extraction: cargo settled INSIDE the bore just short of the cliff "
          "— NOT success, score <= 0.42",
          still() and float(p[0]) < c.x_cliff and float(p[2]) > c.deck_top
          and not ok and sc <= 0.42)

    # =========================== 10. backward removal onto the apron ========================
    env.reset(seed=81)
    step(30)
    place(scene.cargo, -0.100, 0.0, c.deck_top + b / 2 + 0.002)
    step(120)
    report("apron")
    p = cargo_l()
    sc, ok = judge()
    check("backward removal: cargo settled on the open APRON behind the mouth — NOT "
          "success, score ~0 (regressing out the only reachable opening earns "
          "nothing)",
          still() and float(p[0]) < c.x_mouth and float(p[2]) > c.deck_top
          and not ok and sc <= 0.02)

    # =========================== 11. overshoot behind the back wall =========================
    env.reset(seed=91)
    step(30)
    place(scene.cargo, c.x_cliff + c.pen_len + c.pen_wall_t + b / 2 + 0.04, 0.0,
          b / 2 + 0.002)
    step(150)
    report("overshoot")
    p = cargo_l()
    sc, ok = judge()
    check("overshoot: cargo settled on the floor BEHIND the pen's back wall — NOT "
          "success (the containment walls are load-bearing)",
          still() and float(p[0]) > c.x_cliff + c.pen_len + c.pen_wall_t
          and float(p[2]) < 0.05 and not bool(scene.in_pen()[0]) and not ok
          and sc <= 0.70 + 1e-3)

    # =========================== 12. airborne fly-through ===================================
    env.reset(seed=101)
    step(30)
    place(scene.cargo, c.x_cliff + 0.06, 0.0, 0.30)
    step(2)
    report("airborne")
    p = cargo_l()
    sc, ok = judge()
    air_ok = (float(p[2]) > c.pen_z[1] + 0.05 and abs(float(p[1])) < c.pen_y_tol
              and c.pen_x[0] < float(p[0]) < c.pen_x[1]
              and not bool(scene.in_pen()[0]) and not ok)
    place(scene.cargo, -0.35, 0.20, b / 2 + 0.002)  # relocate pre-landing
    step(60)
    check("airborne: cargo in free fall directly over the pen, judged mid-air — NOT "
          "success (the resting-height gate is load-bearing)", air_ok)

    # =========================== 13. latched credit survives regression =====================
    env.reset(seed=111)
    step(30)
    place(scene.cargo, 0.130, 0.0, c.deck_top + b / 2 + 0.002)
    step(90)
    report("bore-credit")
    sc13a, ok = judge()
    place(scene.cargo, float(scene._x0[0]), 0.0, c.deck_top + b / 2 + 0.002)
    step(90)
    report("regressed")
    sc13b, ok = judge()
    check("latched credit: returning the cargo to its spawn depth leaves the latched "
          f"score unchanged ({sc13a:.3f} -> {sc13b:.3f}), still no success",
          sc13a >= 0.15 and abs(sc13b - sc13a) < 1e-3 and not ok)

    # =========================== 14. THREE feeders are not enough (contact physics) =========
    env.reset(seed=121)
    step(30)
    x_start = float(cargo_l()[0])
    pushed = all(push_feeder_flush(k) for k in range(3))
    step(120)
    report("three-feeders")
    p = cargo_l()
    sc, ok = judge()
    check("THREE FEEDERS SHORT: three feeders contact-pushed to the mouth plane "
          "exactly like solve.py — the cargo advanced but stalls inside the bore "
          "short of the cliff, NOT success, score <= 0.46 (the fourth block is "
          "load-bearing)",
          pushed and float(p[0]) > x_start + 0.02 and float(p[0]) < c.x_cliff
          and float(p[2]) > c.deck_top and still() and not ok and sc <= 0.46)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.tunnel_relay")
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
    except Exception as exc:  # noqa: BLE001 — die fast, not at the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL (exception: {exc!r})", flush=True)
        os._exit(2)
