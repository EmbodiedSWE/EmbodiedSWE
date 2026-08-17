"""Smoke / rubric-REJECTION battery for TeaBallTransferScene (sim_gen task
`approach_grasp_ceramic_teapot_i209`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — lid force-lifted out of the funnel, ball
force-lifted out and dropped through the destination mouth, lid wedge-seated by a
released drop — is the acceptance evidence that the rubric ACCEPTS a correct
outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partial) outcome as a settled state and asserts the rubric REJECTS it; no probe in
this battery ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN     — reset layout settles finite: lid seated on the source,
                           ball resting inside it, all still, score ~0;
  3-4. randomization     — READBACK over 8 seeded resets: per-canister xy jitter +
                           free yaw are real; the capped/loaded role flips
                           (Bernoulli); ball in-pot jitter is real;
  5.  null policy        — 240 idle steps -> score ~0, no success (the lid starts
                           seated; nothing fires by itself);
  6.  caged ball         — ORDER IS FORCED: with the lid seated, a velocity-capped
                           1.2 N lateral shove (rotated through 4 directions, 3 s)
                           rattles the ball measurably (> 8 mm, the probe is not
                           vacuous) but it NEVER rises past the sill, never leaves
                           the source, and the transfer latch never fires;
  7.  seed strategy      — the end state the SEED's plan produces (grasp the
                           object and end the episode holding/setting it aside):
                           lid parked on the floor, ball set down on open floor ->
                           only the opening credit (0.15), no success;
  8.  undo / wrong pot   — lid lifted then re-seated on the ORIGINAL canister,
                           ball still inside it: a sealed canister WITH the ball —
                           but the wrong one -> 0.15, no success;
  9.  missing seal       — ball transferred into the destination but the lid
                           re-seated on the SOURCE: both partial credits latch ->
                           exactly the 0.40 non-success cap, no success;
  10. near-miss seat     — ball in the destination, lid dropped 32 mm off-axis and
                           tilted 18 deg: it ends askew on the rim / off the seat
                           (fails centring/depth/tilt gates) -> 0.40, no success;
  11. near-miss payload  — lid perfectly seated in the DESTINATION but the ball
                           left on the floor beside it -> 0.15, no success;
  12. latched credit     — ball probe-dropped into the destination (transfer
                           credit 0.25, lid untouched so opening never fired);
                           ball then teleported back out to the floor -> latched
                           credit survives (still 0.25), NOT success;
  13. rejection audit    — success() was never True at ANY judged point;
  14. final no-NaN       — all task-object states finite at the end.

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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tea_ball_transfer")().build(num_envs=args.num_envs,
                                                       device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply, quat_mul

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.20, 0.85, 0.75)) + o),
                                tuple(np.array((0.42, 0.00, 0.05)) + o),
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
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def report(tag: str) -> None:
        ls = scene._local(scene.lid, True)[0]
        ld = scene._local(scene.lid, False)[0]
        bs = scene._local(scene.ball, True)[0]
        bd = scene._local(scene.ball, False)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | lid_src=({float(ls[0]):+.3f},{float(ls[1]):+.3f},"
              f"{float(ls[2]):.3f}) lid_dst=({float(ld[0]):+.3f},{float(ld[1]):+.3f},"
              f"{float(ld[2]):.3f}) ball_src=({float(bs[0]):+.3f},{float(bs[1]):+.3f},"
              f"{float(bs[2]):.3f}) ball_dst=({float(bd[0]):+.3f},{float(bd[1]):+.3f},"
              f"{float(bd[2]):.3f}) opened={bool(scene._opened_ever[0])} "
              f"transferred={bool(scene._transferred_ever[0])} "
              f"seated_src={bool(scene._lid_seated_on(True, require_still=False)[0])} "
              f"seated_dst={bool(scene._lid_seated_on(False, require_still=False)[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_pot_local(body, source: bool, x: float, y: float, z: float, *,
                        tilt_deg: float = 0.0) -> None:
        """Teleport `body` to a point in the source/destination canister's CURRENT
        frame (probe constructor), level or tilted about the canister's y-axis."""
        p_pos, p_quat = scene._pot_state(source)
        loc = torch.tensor([x, y, z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = p_pos + quat_apply(p_quat, loc)
        if tilt_deg:
            h = math.radians(tilt_deg) / 2
            qt = torch.tensor([math.cos(h), 0.0, math.sin(h), 0.0],
                              device=device).expand(n, 4)
            st[:, 3:7] = quat_mul(p_quat, qt)
        else:
            st[:, 3:7] = p_quat
        body.write_root_state_to_sim(st, all_ids)

    def place_world(body, x: float, y: float, z: float) -> None:
        """Teleport `body` to an env-origin-relative point, level, at rest."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins + torch.tensor([x, y, z],
                                                      device=device).expand(n, 3)
        st[:, 3] = 1.0
        body.write_root_state_to_sim(st, all_ids)

    def drop_ball_into_dest(max_iters: int = 480) -> bool:
        """Probe constructor: stage the ball over the destination mouth, release,
        wait for a settled interior rest (streak-gated)."""
        place_pot_local(scene.ball, False, 0.004, 0.0, c.rim_z + 0.06)
        quiet = 0
        for _ in range(max_iters):
            env.step(no_action)
            inside = bool(scene._ball_in(False)[0])
            still = bool(scene._ball_still()[0])
            quiet = quiet + 1 if (inside and still) else 0
            if quiet >= 30:
                return True
        return False

    def drop_lid_onto(source: bool, *, dx: float = 0.0, tilt_deg: float = 0.0,
                      max_iters: int = 480) -> bool:
        """Probe constructor: stage the lid over the given canister's mouth (level
        + centred unless offset/tilted), release, wait for a settled seat readback.
        Returns whether the seat readback confirmed within the budget."""
        place_pot_local(scene.lid, source, dx, 0.0, c.lid_seat_z + 0.10,
                        tilt_deg=tilt_deg)
        quiet = 0
        for _ in range(max_iters):
            env.step(no_action)
            quiet = quiet + 1 if bool(scene._lid_seated_on(source)[0]) else 0
            if quiet >= 30:
                return True
        return False

    def finite_all() -> bool:
        return bool(torch.isfinite(scene.pot_a.data.root_state_w).all()
                    and torch.isfinite(scene.pot_b.data.root_state_w).all()
                    and torch.isfinite(scene.lid.data.root_state_w).all()
                    and torch.isfinite(scene.ball.data.root_state_w).all())

    def pot_pose(body) -> tuple[float, float, float]:
        p = (body.data.root_pos_w - scene.env_origins)[0]
        q = body.data.root_quat_w[0]
        return (float(p[0]), float(p[1]),
                math.degrees(2.0 * math.atan2(float(q[3]), float(q[0]))))

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    still = (float(scene.lid.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.ball.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, lid seated on the source, ball resting inside it, "
          "all still",
          finite_all() and bool(scene._lid_seated_on(True)[0])
          and bool(scene._ball_in(True)[0]) and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        ax, ay, ayaw = pot_pose(scene.pot_a)
        bx, by, byaw = pot_pose(scene.pot_b)
        bl = scene._local(scene.ball, True)[0]
        reads.append((ax, ay, ayaw, bx, by, byaw,
                      1.0 if bool(scene.src_is_a[0]) else 0.0,
                      float(bl[0]), float(bl[1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (pot_a x,y,yaw | pot_b x,y,yaw | "
          f"src_is_a | ball_loc x,y):\n{arr}", flush=True)
    xy_spread = float((arr[:, [0, 1, 3, 4]].max(axis=0)
                       - arr[:, [0, 1, 3, 4]].min(axis=0)).max())
    yaw_spread = max(float(arr[:, 2].max() - arr[:, 2].min()),
                     float(arr[:, 5].max() - arr[:, 5].min()))
    check("randomization: per-canister xy jitter (> 8 mm spread) and free yaw "
          "(> 20 deg spread) are real (readback)",
          xy_spread > 0.008 and yaw_spread > 20.0)
    flags = arr[:, 6]
    ball_spread = float((arr[:, 7:9].max(axis=0) - arr[:, 7:9].min(axis=0)).max())
    check("randomization: the capped/loaded role flips across seeded resets "
          "(Bernoulli) and ball in-pot jitter (> 4 mm spread) is real (readback)",
          0.0 < flags.mean() < 1.0 and ball_spread > 0.004)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps (the lid "
          "starts seated; no latch fires by itself)", s <= 0.02 and not ok)

    # =========================== 6. caged ball: order is geometry-forced ====================
    # With the lid seated, shove the ball with a velocity-capped 1.2 N lateral force
    # rotated through 4 directions (3 s total). The probe must MOVE the ball
    # (non-vacuous) yet the ball must never rise past the sill, never leave the
    # source, and the transfer latch must never fire: unseal-first is forced.
    torch.manual_seed(41)
    env.reset()
    step(60)
    assert bool(scene._lid_seated_on(True)[0]) and bool(scene._ball_in(True)[0])
    start_xy = scene._local(scene.ball, True)[0, :2].clone()
    dirs = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))
    max_dev, max_z = 0.0, 0.0
    for i in range(360):
        d = dirs[(i // 90) % 4]
        v = scene.ball.data.root_lin_vel_w[0]
        along = float(v[0] * d[0] + v[1] * d[1])
        f_mag = 1.2 if along < 0.30 else 0.0
        f_w = torch.tensor([f_mag * d[0], f_mag * d[1], 0.0],
                           device=device).view(1, 1, 3).expand(n, 1, 3)
        scene.ball.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                                 env_ids=all_ids, is_global=True)
        env.step(no_action)
        bl = scene._local(scene.ball, True)[0]
        max_dev = max(max_dev, float((bl[:2] - start_xy).norm()))
        max_z = max(max_z, float(bl[2]))
    scene.ball.set_external_force_and_torque(zero_wrench, zero_wrench,
                                             env_ids=all_ids)
    step(120)
    report("caged-ball")
    print(f"[smoke] caged-ball probe: max_xy_dev={max_dev:.3f} m, "
          f"max_local_z={max_z:.3f} m (sill {c.sill_z:.3f})", flush=True)
    s, ok = judge()
    check("caged ball: a velocity-capped 1.2 N lateral shove moves the sealed-in "
          f"ball (> 8 mm, got {max_dev * 1000:.0f} mm — the probe is not vacuous) "
          f"but it never rises past the sill (max z {max_z:.3f} < {c.sill_z:.3f}), "
          "stays inside the source with the lid still seated, and the transfer "
          "latch never fires: the unseal-first order is geometry-forced",
          max_dev > 0.008 and max_z < c.sill_z
          and bool(scene._ball_in(True)[0])
          and bool(scene._lid_seated_on(True, require_still=False)[0])
          and not bool(scene._transferred_ever[0]) and s <= 0.02 and not ok)

    # =========================== 7. seed strategy: grasp-and-hold ===========================
    # The seed's plan — approach, grasp, end the episode holding the object — has
    # this settled analog: lid taken off (parked) and the payload set down on open
    # floor. Only the opening credit fires; holding/possessing the ball is worth 0.
    torch.manual_seed(51)
    env.reset()
    step(30)
    place_world(scene.lid, 0.12, 0.0, 0.03)
    step(90)
    place_world(scene.ball, 0.75, 0.0, c.ball_r + 0.002)
    step(180)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy: lid parked + ball grasped-and-set-down on open floor "
          f"earns ONLY the opening credit (~{c.w_open:.2f}, got {s:.3f}), no "
          "transfer, no success (the seed's hold-the-object outcome is worthless)",
          abs(s - c.w_open) < 0.01 and not bool(scene._transferred_ever[0])
          and not ok)

    # =========================== 8. undo / wrong pot re-sealed ==============================
    # Lid lifted clear then re-seated on the ORIGINAL canister with the ball still
    # inside: a sealed canister WITH the ball — but the wrong one.
    torch.manual_seed(61)
    env.reset()
    step(30)
    reseated = drop_lid_onto(True)
    report("wrong-pot")
    s, ok = judge()
    check("undo / wrong pot: lid re-seated on the ORIGINAL canister (seat readback "
          f"confirms) with the ball still inside -> only opening credit "
          f"(~{c.w_open:.2f}, got {s:.3f}), NO success (sealing the wrong canister "
          "is a failure end state)",
          reseated and bool(scene._ball_in(True)[0])
          and abs(s - c.w_open) < 0.01 and not ok)

    # =========================== 9. missing seal: the 0.40 cap ==============================
    # Ball genuinely transferred (probe drop through the destination funnel) but the
    # lid re-seated on the SOURCE: both partials latch -> exactly the non-success cap.
    torch.manual_seed(71)
    env.reset()
    step(30)
    transferred = drop_ball_into_dest()
    resealed_src = drop_lid_onto(True)
    report("missing-seal")
    s, ok = judge()
    check("missing seal: ball transferred into the destination but lid re-seated "
          f"on the SOURCE -> exactly the non-success cap (0.40, got {s:.3f}), no "
          "success (the seal of the DESTINATION is load-bearing)",
          transferred and resealed_src and abs(s - 0.40) < 0.01
          and not bool(scene._lid_seated_on(False, require_still=False)[0])
          and not ok)

    # =========================== 10. near-miss seat: askew on the rim =======================
    # Ball in the destination; lid dropped 32 mm off-axis (outside the ~14 mm funnel
    # capture) and tilted 18 deg: it must NOT read seated (centring/depth/tilt gates).
    torch.manual_seed(81)
    env.reset()
    step(30)
    transferred = drop_ball_into_dest()
    seated = drop_lid_onto(False, dx=0.032, tilt_deg=18.0, max_iters=360)
    report("askew-lid")
    ld = scene._local(scene.lid, False)[0]
    s, ok = judge()
    check("near-miss seat: lid dropped 32 mm off-axis + 18 deg tilt onto the "
          f"destination ends askew/off the seat (readback z={float(ld[2]):.3f} vs "
          f"seat {c.lid_seat_z:.3f}, xy={float(ld[:2].norm()) * 1000:.0f} mm) — NOT "
          f"seated, score stays at the cap ({s:.3f}), no success",
          transferred and not seated
          and not bool(scene._lid_seated_on(False, require_still=False)[0])
          and s <= 0.40 + 0.01 and not ok)

    # =========================== 11. near-miss payload: ball beside =========================
    # Lid PERFECTLY seated in the destination — but the ball left on the floor
    # beside it: the seal alone is not the goal.
    torch.manual_seed(91)
    env.reset()
    step(30)
    place_world(scene.ball, 0.75, 0.0, c.ball_r + 0.002)
    step(60)
    sealed_dst = drop_lid_onto(False)
    report("ball-beside")
    s, ok = judge()
    check("near-miss payload: lid PERFECTLY seated in the destination (readback "
          f"confirms) but ball on the floor beside -> only opening credit "
          f"(~{c.w_open:.2f}, got {s:.3f}), NO success (the payload transfer is "
          "load-bearing)",
          sealed_dst and abs(s - c.w_open) < 0.01
          and not bool(scene._ball_in(False)[0]) and not ok)

    # =========================== 12. latched credit survives regression =====================
    torch.manual_seed(101)
    env.reset()
    step(30)
    transferred = drop_ball_into_dest()
    s12a, ok = judge()
    got_credit = transferred and abs(s12a - c.w_transfer) < 0.01 and not ok
    place_world(scene.ball, 0.75, 0.0, c.ball_r + 0.002)
    step(60)
    report("regressed")
    s12b, ok = judge()
    check("latched credit: probe-dropped transfer earns exactly the transfer credit "
          f"({c.w_transfer:.2f}, got {s12a:.3f}, lid untouched so opening never "
          f"fired) with NO success; teleporting the ball back out leaves the "
          f"latched score unchanged ({s12a:.3f} -> {s12b:.3f}), still no success",
          got_credit and abs(s12b - s12a) < 1e-3
          and not bool(scene._ball_in(False)[0]) and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.tea_ball_transfer")
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
    except Exception as exc:  # noqa: BLE001 — die fast, Kit teardown hangs
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({exc})", flush=True)
        os._exit(1)
