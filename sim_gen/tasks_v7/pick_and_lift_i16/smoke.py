"""Smoke / rubric-REJECTION battery for BallastLeverScene (sim_gen task
`pick_and_lift_i16`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — drop three steel blocks into the basket and let
real torque tip the beam — is the acceptance evidence that the rubric ACCEPTS a
correct outcome). Every teleport/force here is instrumentation that CONSTRUCTS a
wrong (or partial) outcome and asserts the rubric REJECTS it; no probe in this
battery ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: beam seated cargo-end DOWN on
                            its stop, cargo caged, five blocks flat on the floor;
                            score ~0 at rest, no success;
  3-4. randomization      — READBACK over 6 seeded resets: machine xy + yaw, block
                            positions (slot PERMUTATION moves the foam), the approach
                            baselines d0, and the cargo's in-cage jitter all move;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's whole plan is "grasp the red cube and lift it".
                            A steady 1.5x-weight upward pull on the cargo (a firm
                            grasp; strong enough to tip the neutral beam, gentle
                            enough not to rip the machine apart) for 2 s: the CAGE
                            HOLDS the cube, the machine stays seated, and although the
                            pull physically tips the beam to its raised stop, k=0
                            ballast -> NOT success, and the gated raise credit stays
                            ~0 (raising the beam by hand farms nothing);
  7.  nothing lasting     — pull released: the beam falls back cargo-end down (readback)
                            and the score is still ~0 — no latched credit from the hand;
  8.  foam decoy          — 2 steel + the FOAM block dropped into the basket (the foam
                            verified inside): the beam must NOT tip — discrimination is
                            by MASS, not by looks -> NOT success, score <= 0.45;
  9.  beside the basket   — 3 steel piled on the GROUND under the raised basket end:
                            visually "at" the basket but k=0, no tip, score <= 0.15;
  10. on-beam, outside    — 2 steel resting ON the beam plate outside the basket walls:
                            weight on the beam but not IN the basket -> k=0, no tip,
                            NOT success, score <= 0.15;
  11. off-cradle tableau  — the full success geometry (beam at raised pitch, 3 steel in
                            the basket, cargo in the cage) teleported OFF the cradle:
                            beam_seated() is False -> NOT success (the mechanism must
                            be operated where it stands);
  12. latched credit      — 2 steel dropped into the basket then removed to the floor:
                            the latched score is unchanged (and success stays absent);
  13. monotonicity        — hovering a block closer to the basket latches strictly more
                            approach credit than a farther hover;
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.pick_and_lift_i16.smoke --headless
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
    env = ENVS.get("simgen.ballast_lever_lift")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.95, -1.10, 0.80)) + o),
                                tuple(np.array((-0.15, 0.00, 0.10)) + o),
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
        s, ok = judge()
        print(f"[smoke] {tag:16s} | raised_sin={float(scene.raised_sin_now()[0]):+.3f} "
              f"k={int(scene.steel_in_basket()[0].sum())} "
              f"foam_in={bool(scene.foam_in_basket()[0])} "
              f"caged={bool(scene.cargo_in_cage()[0])} seated={bool(scene.beam_seated()[0])} "
              f"settled={bool(scene.settled()[0])} appr={float(scene.approach_latch[0]):.3f} "
              f"kL={float(scene.k_latch[0]):.1f} rL={float(scene.raise_latch[0]):.3f} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_body(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0),
                   settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def write_beam_local(body, loc_xyz, settle_steps: int = 30) -> None:
        """Probe placement expressed in the LIVE beam body frame (orientation matched)."""
        bq = scene.beam.data.root_quat_w
        bp = scene.beam.data.root_pos_w
        loc = torch.tensor(loc_xyz, device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = bp + quat_apply(bq, loc)
        st[:, 3:7] = bq
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def drop_into_basket(body, y_lane: float, settle_steps: int = 150) -> None:
        """The same honest load solve.py performs: free-space hover above the open
        basket top, then gravity + contact."""
        write_beam_local(
            body, (-c.arm, y_lane, c.tray_floor_z + c.wall_h + c.block_size / 2 + 0.015),
            settle_steps=settle_steps)

    def basket_xy_w() -> torch.Tensor:
        loc = torch.tensor([-c.arm, 0.0, 0.0], device=device).expand(n, 3)
        return (scene.beam.data.root_pos_w + quat_apply(scene.beam.data.root_quat_w, loc)
                - scene.env_origins)[0, :2]

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    fin0 = bool(torch.isfinite(scene.beam.data.root_state_w).all()
                and torch.isfinite(scene.cargo.data.root_state_w).all()
                and torch.isfinite(scene.foam.data.root_state_w).all()
                and all(torch.isfinite(b.data.root_state_w).all() for b in scene.steel))
    blocks_z = [float((b.data.root_pos_w - scene.env_origins)[0, 2]) for b in scene.steel]
    blocks_flat = all(abs(z - c.block_size / 2) < 0.010 for z in blocks_z)
    check("settle: states finite; beam seated cargo-end DOWN on its stop "
          "(raised_sin < -0.15), cargo caged, blocks flat on the floor",
          fin0 and float(scene.raised_sin_now()[0]) < -0.15
          and bool(scene.beam_seated()[0]) and bool(scene.cargo_in_cage()[0])
          and blocks_flat and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        cp = (scene.cradle.data.root_pos_w - scene.env_origins)[0]
        q = scene.cradle.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        s0 = (scene.steel[0].data.root_pos_w - scene.env_origins)[0]
        fp = (scene.foam.data.root_pos_w - scene.env_origins)[0]
        cargo_ly = float(scene._beam_local(scene.cargo.data.root_pos_w)[0, 1])
        reads.append((float(cp[0]), float(cp[1]), yaw, float(s0[0]), float(s0[1]),
                      float(fp[0]), float(fp[1]), float(scene.d0[0, 0]), cargo_ly))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (cradle_x, cradle_y, yaw, s0_x, s0_y, foam_x, "
          f"foam_y, d0_0, cargo_local_y):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: machine xy + yaw vary across seeded resets (readback)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.3)
    check("randomization: block positions (slot permutation moves steel_0 and the foam), "
          "approach baseline d0, and the cargo's in-cage jitter vary (readback)",
          spread[3] > 0.03 and spread[4] > 0.03 and spread[5] > 0.03 and spread[6] > 0.03
          and spread[7] > 0.03 and spread[8] > 0.004)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6-7. SEED strategy: grasp-and-lift the red cube ===========
    # The seed's whole plan is "grasp the red cube and raise it". Simulate a firm
    # grasp: a steady 1.5x-weight upward pull on the cargo for 2 s (net +0.5 mg on the
    # roof tips the neutral beam decisively, ~0.84 N.m, while staying far below the
    # ~3 N.m that would lever the axle out of its cradle). The cage must HOLD the cube,
    # the machine must stay seated, and although the pull physically tips the beam to
    # its raised stop, there is no ballast -> NOT success, and the raise credit (gated
    # on >= 3 steel IN the basket) stays ~0. Releasing must undo everything.
    env.reset(seed=41)
    step(30)
    pull = torch.zeros(n, 1, 3, device=device)
    pull[:, 0, 2] = 1.5 * c.cargo_mass * 9.81
    cage_held, max_raised = True, -1.0
    for i in range(240):
        scene.cargo.set_external_force_and_torque(pull, zero_wrench, env_ids=all_ids,
                                                  is_global=True)
        env.step(no_action)
        if i % 20 == 19:
            cage_held = cage_held and bool(scene.cargo_in_cage()[0])
            max_raised = max(max_raised, float(scene.raised_sin_now()[0]))
            judge()
    report("grasp-pull")
    s, ok = judge()
    check("seed strategy (grasp-and-lift): 1.5x-weight pull on the caged cargo — the "
          f"cage HOLDS the cube, the machine stays seated; the pull tips the beam "
          f"raised (max sin {max_raised:+.3f} >= {c.raised_sin:+.3f}) yet k=0 ballast "
          "-> NOT success, score <= 0.02 (gated raise credit farms nothing)",
          cage_held and bool(scene.cargo_in_cage()[0]) and bool(scene.beam_seated()[0])
          and max_raised >= c.raised_sin and not ok and s <= 0.02)
    scene.cargo.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    step(300)
    report("pull-released")
    s, ok = judge()
    check("nothing lasting: pull released -> the beam falls back cargo-end down "
          "(raised_sin < 0, readback), still seated, score still <= 0.02, no success",
          float(scene.raised_sin_now()[0]) < 0.0 and bool(scene.beam_seated()[0])
          and s <= 0.02 and not ok)

    # =========================== 8. foam decoy ==============================================
    env.reset(seed=51)
    step(30)
    drop_into_basket(scene.steel[0], -0.042)
    drop_into_basket(scene.steel[1], +0.042)
    drop_into_basket(scene.foam, 0.0, settle_steps=390)  # generous: give it every chance
    report("foam-decoy")
    s, ok = judge()
    check("foam decoy: 2 steel + the FOAM block verified in the basket — the beam must "
          "NOT tip (mass, not looks): still cargo-end down, NOT success, score <= 0.45",
          bool(scene.steel_in_basket()[0, 0]) and bool(scene.steel_in_basket()[0, 1])
          and bool(scene.foam_in_basket()[0]) and float(scene.raised_sin_now()[0]) < 0.0
          and not ok and s <= 0.45)

    # =========================== 9. beside the basket (on the ground) =======================
    env.reset(seed=61)
    step(30)
    bxy = basket_xy_w()
    bq = scene.beam.data.root_quat_w[0]
    byaw = 2.0 * math.atan2(float(bq[3]), float(bq[0]))
    ca, sa = math.cos(byaw), math.sin(byaw)
    for j, dy in enumerate((-0.09, 0.0, 0.09)):
        # ground spot just beyond the basket end, laid out across the beam's yaw
        gx = float(bxy[0]) + ca * (-0.13) - sa * dy
        gy = float(bxy[1]) + sa * (-0.13) + ca * dy
        place_body(scene.steel[j], gx, gy, c.block_size / 2 + 0.002,
                   quat=(math.cos(byaw / 2), 0.0, 0.0, math.sin(byaw / 2)),
                   settle_steps=30)
    step(90)
    report("ground-pile")
    s, ok = judge()
    check("beside the basket: 3 steel piled on the GROUND under the raised basket end — "
          "k=0, beam still cargo-end down, NOT success, score <= 0.15",
          int(scene.steel_in_basket()[0].sum()) == 0
          and float(scene.raised_sin_now()[0]) < 0.0 and not ok and s <= 0.15)

    # =========================== 10. on the beam, outside the basket ========================
    env.reset(seed=71)
    step(30)
    write_beam_local(scene.steel[0], (-0.17, 0.0, c.tray_floor_z + c.block_size / 2 + 0.003),
                     settle_steps=30)
    write_beam_local(scene.steel[1], (-0.11, 0.0, c.tray_floor_z + c.block_size / 2 + 0.003),
                     settle_steps=30)
    step(120)
    report("on-beam-outside")
    s, ok = judge()
    check("on-beam, outside the basket: 2 steel resting ON the beam plate (weight on the "
          "beam, not IN the basket) — k=0, no tip, NOT success, score <= 0.15",
          int(scene.steel_in_basket()[0].sum()) == 0
          and float(scene.raised_sin_now()[0]) < 0.0 and not ok and s <= 0.15)

    # =========================== 11. off-cradle tableau =====================================
    # The full success GEOMETRY — beam at raised pitch, 3 steel in the basket, cargo in
    # the cage — teleported to empty floor space away from the cradle. Every clause
    # reads True except beam_seated(): the mechanism must be operated where it stands.
    # (Transient probe: judged after 2 real steps, then discarded.)
    env.reset(seed=81)
    step(10)
    th = -math.asin(c.sin_stop - 0.01)  # raised pitch, just inside the stop
    qt = (math.cos(th / 2), 0.0, math.sin(th / 2), 0.0)
    qt_t = torch.tensor(qt, device=device)
    base = torch.tensor([0.90, 0.70, c.axle_h], device=device)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = base
    st[:, 3], st[:, 5] = qt[0], qt[2]
    st[:, 0:3] += scene.env_origins
    scene.beam.write_root_state_to_sim(st, all_ids)
    for j, dy in enumerate((-0.042, 0.0, 0.042)):
        loc = torch.tensor([-c.arm, dy, c.tray_floor_z + c.block_size / 2 + 0.001],
                           device=device)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = base + quat_apply(qt_t.expand(n, 4), loc.expand(n, 3))
        st[:, 3], st[:, 5] = qt[0], qt[2]
        st[:, 0:3] += scene.env_origins
        scene.steel[j].write_root_state_to_sim(st, all_ids)
    loc = torch.tensor([c.arm, 0.0, c.tray_floor_z + c.cargo_size / 2 + 0.0015], device=device)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = base + quat_apply(qt_t.expand(n, 4), loc.expand(n, 3))
    st[:, 3], st[:, 5] = qt[0], qt[2]
    st[:, 0:3] += scene.env_origins
    scene.cargo.write_root_state_to_sim(st, all_ids)
    step(2)
    report("off-cradle")
    s, ok = judge()
    check("off-cradle tableau: raised pitch + 3 steel in basket + cargo caged, but the "
          "beam is OFF its cradle — beam_seated False -> NOT success",
          float(scene.raised_sin_now()[0]) >= c.raised_sin
          and int(scene.steel_in_basket()[0].sum()) >= 3
          and bool(scene.cargo_in_cage()[0]) and not bool(scene.beam_seated()[0]) and not ok)

    # =========================== 12. latched credit survives regression =====================
    env.reset(seed=91)
    step(30)
    drop_into_basket(scene.steel[0], -0.042)
    drop_into_basket(scene.steel[1], +0.042)
    report("two-loaded")
    s_in, _ = judge()
    place_body(scene.steel[0], -0.75, -0.30, c.block_size / 2 + 0.002, settle_steps=30)
    place_body(scene.steel[1], -0.75, +0.30, c.block_size / 2 + 0.002, settle_steps=60)
    report("blocks-removed")
    s_out, ok = judge()
    check("latched credit: removing both loaded blocks leaves the latched score "
          f"unchanged ({s_in:.3f} -> {s_out:.3f}) and success stays absent",
          s_in >= 0.35 and abs(s_out - s_in) < 0.02 and not ok
          and int(scene.steel_in_basket()[0].sum()) == 0)

    # =========================== 13. approach monotonicity ==================================
    env.reset(seed=101)
    step(5)
    bxy = basket_xy_w()
    b0 = (scene.steel[0].data.root_pos_w - scene.env_origins)[0]
    dx, dy = float(bxy[0] - b0[0]), float(bxy[1] - b0[1])
    place_body(scene.steel[0], float(b0[0]) + 0.5 * dx, float(b0[1]) + 0.5 * dy, 0.30,
               settle_steps=3)
    a_half = float(scene.approach_latch[0])
    place_body(scene.steel[0], float(b0[0]) + 0.85 * dx, float(b0[1]) + 0.85 * dy, 0.30,
               settle_steps=3)
    a_near = float(scene.approach_latch[0])
    check("monotonicity: hovering the block closer to the basket latches strictly more "
          f"approach credit ({a_half:.3f} < {a_near:.3f})", a_half + 0.10 < a_near)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.cradle.data.root_state_w).all()
           and torch.isfinite(scene.beam.data.root_state_w).all()
           and torch.isfinite(scene.cargo.data.root_state_w).all()
           and torch.isfinite(scene.foam.data.root_state_w).all()
           and all(torch.isfinite(b.data.root_state_w).all() for b in scene.steel))
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ballast_lever_lift")
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
