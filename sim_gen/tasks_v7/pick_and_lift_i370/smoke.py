"""Smoke / rubric-REJECTION battery for RollerRelayScene (sim_gen task
`pick_and_lift_i370`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — lay the rollers, push the slab across the
rolling relay, park it on the goal dock — is the acceptance evidence that the
rubric ACCEPTS a correct outcome). Every teleport/force here is instrumentation
that CONSTRUCTS a wrong (or partial) outcome and asserts the rubric REJECTS it;
no probe in this battery ever reaches success(), and a final audit asserts that.

  1-2. settle/no-NaN      — reset layout settles finite: slab flat on the start dock
                            at dock-top height, rollers lying in the staging band;
                            score ~0 at rest, no success;
  3-4. randomization      — READBACK over 8 seeded resets: slab xy + yaw, roller
                            slot xy, and WHICH SIDE of the track the staging band is
                            on all move;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's whole plan is "grasp the object and lift it".
                            A steady 1.5x-weight (103 N!) vertical pull hoists the
                            slab off the dock (probe verified to MOVE it well clear),
                            then holds it aloft: airborne/off-track carriage earns
                            NOTHING (aboard + advance are gated on ride height over
                            the channel) -> score ~0, NOT success;
  7.  nothing lasting     — pull released: the slab falls back; still score ~0;
  8.  channel dead-end    — the slab placed down IN the rubber channel and shoved
                            with a steady 35 N for 2 s barely moves (<3 cm): dragging
                            is physically hopeless, and the dropped slab earns
                            nothing (ride-height gates) -> the declared irreversible
                            dead end is REAL;
  9.  probe validity      — the SAME 35 N applied to the slab on the slick dock
                            moves it >5 cm in 0.4 s: the drag probe force is amply
                            sufficient on low friction — the channel result above
                            measures friction, not a weak probe;
  10. near-miss xy        — slab flat AT dock height on the goal dock but one corner
                            row short of the dock edge -> NOT success;
  11. near-miss z         — slab flat, fully over the goal dock, but RIDING ON TWO
                            ROLLERS (~36 mm too high): the freight was never landed
                            -> NOT success;
  12. wrong object        — the three rollers delivered onto the goal dock, slab
                            still home on the start dock -> score ~0, NOT success;
  13. latched credit      — rollers honestly staged in the channel (score latches
                            ~0.10), then carried back out: latched score unchanged,
                            still no success;
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.pick_and_lift_i370.smoke --headless
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.roller_relay_freight")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.30, -1.15, 0.85)) + o),
                                tuple(np.array((0.23, 0.00, 0.02)) + o),
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
        p = scene.slab_pos()[0]
        print(f"[smoke] {tag:16s} | slab=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):.3f}) bz={float(scene.slab_bottom_z()[0]):.3f} "
              f"flat={bool(scene.slab_flat()[0])} corners={bool(scene.corners_on_goal()[0])} "
              f"staged={int(scene.rollers_staged()[0].sum())} "
              f"stagedL={float(scene.staged_latch[0].sum()):.1f} "
              f"aboardL={float(scene.aboard_latch[0]):.2f} "
              f"advL={float(scene.advance_latch[0]):.3f} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

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

    def push_slab(fx: float, steps: int) -> float:
        """Constant world-frame push at the slab centre; returns the displacement
        magnitude measured AT FORCE-OFF (the peak, before gravity/friction restores
        anything). Wrench cleared afterwards."""
        p0 = scene.slab_pos()[0, :2].clone()
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 0] = fx
        for _ in range(steps):
            scene.slab.set_external_force_and_torque(f, zero_wrench, env_ids=all_ids,
                                                     is_global=True)
            env.step(no_action)
        moved = float((scene.slab_pos()[0, :2] - p0).norm())
        scene.slab.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids, is_global=True)
        return moved

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    fin0 = bool(torch.isfinite(scene.slab.data.root_state_w).all()
                and all(torch.isfinite(r.data.root_state_w).all() for r in scene.rollers))
    bz = float(scene.slab_bottom_z()[0])
    rz = [float(z) for z in scene.roller_pos()[0, :, 2]]
    check("settle: states finite; slab flat on the start dock at dock-top height, "
          "rollers lying in the staging band",
          fin0 and bool(scene.slab_flat()[0]) and abs(bz - c.start_top) < 0.008
          and all(abs(z - c.roller_r) < 0.010 for z in rz) and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(5)
        p = scene.slab_pos()[0]
        q = scene.slab.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        r0 = scene.roller_pos()[0, 0]
        reads.append((float(p[0]), float(p[1]), yaw, float(r0[0]), abs(float(r0[1])),
                      float(scene.side[0])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (slab_x, slab_y, yaw, r0_x, |r0_y|, side):\n"
          f"{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: slab xy + yaw vary across seeded resets (readback)",
          spread[0] > 0.012 and spread[1] > 0.012 and spread[2] > 0.03)
    check("randomization: roller slot xy vary and the staging SIDE flips (readback)",
          spread[3] > 0.02 and spread[4] > 0.02 and spread[5] > 1.5)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6-7. SEED strategy: grasp-and-lift the slab ================
    # The seed's whole plan is "grasp the object and lift it". Simulate an impossibly
    # strong grasp: a steady 1.5x-weight vertical pull (103 N — no Franka does this)
    # until the slab is well airborne, then hold it aloft with a damped weight-hold.
    # Airborne carriage must earn NOTHING: aboard/advance credit is gated on riding at
    # roller height over the channel.
    env.reset(seed=41)
    step(30)
    w_n = c.slab_m * 9.81
    max_bz = -1.0
    f = torch.zeros(n, 1, 3, device=device)
    for i in range(300):
        bz = float(scene.slab_bottom_z()[0])
        max_bz = max(max_bz, bz)
        vz = float(scene.slab.data.root_lin_vel_w[0, 2])
        f[:, 0, 2] = 1.5 * w_n if bz < 0.20 else w_n - 15.0 * vz
        scene.slab.set_external_force_and_torque(f, zero_wrench, env_ids=all_ids,
                                                 is_global=True)
        env.step(no_action)
        if i % 25 == 24:
            judge()
    report("grasp-hoist")
    s, ok = judge()
    check("seed strategy (grasp-and-lift): a 1.5x-weight (103 N) pull verifiably hoists "
          f"the slab well off the dock (max underside {max_bz:.3f} m >= 0.15) and holds "
          "it aloft — airborne carriage earns NOTHING: score <= 0.02, NOT success",
          max_bz >= 0.15 and s <= 0.02 and not ok)
    scene.slab.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids,
                                             is_global=True)
    step(300)
    report("hoist-released")
    s, ok = judge()
    check("nothing lasting: pull released -> the slab falls back and settles; score "
          "still <= 0.02, no success", s <= 0.02 and not ok)

    # =========================== 8-9. the channel dead-end is REAL ==========================
    # Place the slab down IN the channel (it lands across the rubber curbs / mat — the
    # exact state a premature push produces) and shove it with a steady 35 N for 2 s:
    # rubber grip (~mu 1.1-1.2 against ~69 N of weight -> ~76+ N to slide) must hold it
    # to < 3 cm, and the dropped slab must earn nothing. Then prove the SAME 35 N is an
    # AMPLE force on the slick dock (> 5 cm in 0.4 s): the probe measures friction.
    env.reset(seed=51)
    step(30)
    place(scene.slab, 0.17, 0.0, c.curb_h + c.slab_t / 2 + 0.003, settle_steps=90)
    report("in-channel")
    moved_channel = push_slab(35.0, 240)
    step(60)
    report("channel-shoved")
    s, ok = judge()
    check("channel dead-end: slab dropped into the rubber channel, shoved with 35 N for "
          f"2 s — moved {moved_channel * 1000:.0f} mm < 30 mm (dragging is hopeless), "
          "score <= 0.02, NOT success: the premature-push dead end is irreversible",
          moved_channel < 0.030 and s <= 0.02 and not ok)
    env.reset(seed=52)
    step(30)
    moved_dock = push_slab(-35.0, 48)  # push toward the dock rear: pure slick sliding
    step(30)
    report("dock-shoved")
    check("probe validity: the SAME 35 N on the slick start dock moves the slab "
          f"{moved_dock * 1000:.0f} mm > 50 mm in 0.4 s — the dead-end probe force is "
          "amply sufficient on low friction", moved_dock > 0.050)

    # =========================== 10. near-miss xy ===========================================
    env.reset(seed=61)
    step(30)
    # flat, AT dock height, but the rear corner row ~2 cm short of the goal-dock edge
    place(scene.slab, c.goal_x0 + c.slab_l / 2 - 0.020, 0.0,
          c.goal_top + c.slab_t / 2 + 0.002, settle_steps=60)
    report("xy-short")
    s, ok = judge()
    check("near-miss xy: slab flat at dock height but rear corners short of the goal "
          "dock -> corners_on_goal False, NOT success",
          bool(scene.slab_flat()[0]) and not bool(scene.corners_on_goal()[0]) and not ok)

    # =========================== 11. near-miss z (still riding rollers) =====================
    env.reset(seed=71)
    step(30)
    rq = (1.0, 0.0, 0.0, 0.0)
    place(scene.rollers[0], 0.55, 0.0, c.goal_top + c.roller_r + 0.002, rq, settle_steps=20)
    place(scene.rollers[1], 0.66, 0.0, c.goal_top + c.roller_r + 0.002, rq, settle_steps=20)
    place(scene.slab, 0.605, 0.0,
          c.goal_top + 2 * c.roller_r + c.slab_t / 2 + 0.004, settle_steps=45)
    report("riding-on-dock")
    s, ok = judge()
    bz = float(scene.slab_bottom_z()[0])
    check("near-miss z: slab flat and fully over the goal dock but RIDING ON TWO ROLLERS "
          f"(underside {bz * 1000:.0f} mm, ~36 mm too high; probe verified still aloft) "
          "-> NOT success (the freight was never landed)",
          bool(scene.slab_flat()[0]) and bool(scene.corners_on_goal()[0])
          and bz > c.goal_top + 0.022 and not ok)

    # =========================== 12. wrong object ===========================================
    env.reset(seed=81)
    step(30)
    place(scene.rollers[0], 0.53, -0.05, c.goal_top + c.roller_r + 0.002, rq, settle_steps=10)
    place(scene.rollers[1], 0.62, 0.00, c.goal_top + c.roller_r + 0.002, rq, settle_steps=10)
    place(scene.rollers[2], 0.71, 0.05, c.goal_top + c.roller_r + 0.002, rq, settle_steps=60)
    report("wrong-object")
    s, ok = judge()
    check("wrong object: the three rollers delivered onto the goal dock while the slab "
          "sits home on the start dock -> score <= 0.02, NOT success",
          s <= 0.02 and not ok)

    # =========================== 13. latched staging credit =================================
    env.reset(seed=91)
    step(30)
    for i, x in enumerate((0.035, 0.125, 0.215)):
        place(scene.rollers[i], x, 0.0, c.mat_t + c.roller_r + 0.009, rq, settle_steps=50)
    step(60)
    report("staged")
    s_in, _ = judge()
    for i in range(3):
        place(scene.rollers[i], 0.10 + 0.09 * i, 0.32, c.roller_r + 0.006, rq,
              settle_steps=20)
    step(40)
    report("unstaged")
    s_out, ok = judge()
    check("latched credit: staging all three rollers latches ~0.10, and carrying them "
          f"back out leaves the latched score unchanged ({s_in:.3f} -> {s_out:.3f}), "
          "no success",
          s_in >= 0.095 and abs(s_out - s_in) < 0.01 and not ok
          and int(scene.rollers_staged()[0].sum()) == 0)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = bool(torch.isfinite(scene.slab.data.root_state_w).all()
               and all(torch.isfinite(r.data.root_state_w).all() for r in scene.rollers))
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.roller_relay_freight")
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
    main()
