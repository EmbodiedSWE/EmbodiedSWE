"""Smoke / rubric-REJECTION battery for GearTrainDialScene (sim_gen task
`libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf_i32`) — NullRobot,
teleported probe states, RECORDED.

This is NOT a solution (solve.py — seat the toothed idler under gravity, then crank the
driver with a bounded torque so the mesh turns the caged dial onto its mark — is the
acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome and asserts the rubric
REJECTS it; no probe in this battery ever reaches success(), and a final audit check
asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: driver + output seated on their
                            axles (readback), idler + decoy on the floor; score ~0, no
                            success;
  3-4. randomization      — READBACK over 6 seeded resets: frame xy + yaw, driver yaw,
                            the SIGNED initial dial error (direction changes!), idler xy,
                            decoy xy, and the approach baseline d0 all move;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's whole plan ("carry the object somewhere and set it
                            down") = idler carried onto the plate beside the empty axle
                            and LEFT there: approach credit only, NOT seated, NOT success,
                            score <= 0.15;
  7.  teleported dial     — idler honestly seated first (train complete, settled), then
                            the output's yaw WRITTEN directly onto the stripe: marker
                            reads within tol, every instantaneous predicate passes — yet
                            the gated error ACCOUNT never saw plausible motion, err_acc
                            stays >> tol -> NOT success (the account is the anti-teleport
                            contract);
  8.  trainless rotation  — WITHOUT the idler, the output is slow-torqued onto the stripe
                            (physically plausible per-substep motion): marker within tol,
                            but the gate (train incomplete) never credited it -> NOT
                            success;
  9.  decoy substitution  — the SMOOTH disc seated on the centre axle, then the driver
                            cranked hard (driver verified spinning): no tooth contact, no
                            transmission — the output does not align, NOT success,
                            score <= 0.15;
  10. near-miss seat      — idler resting flat on the plate 35 mm OFF the axle (outside
                            seat_xy_tol): not seated, no seat credit, NOT success;
  11. latched credit      — honestly seated idler then removed to the floor: the latched
                            approach + seat score survives (and success is gone);
  12. monotonicity        — placing the idler closer to the centre axle latches strictly
                            more approach credit than a farther placement;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf_i32.smoke --headless
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
    env = ENVS.get("simgen.gear_train_dial")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.95, -0.90, 0.80)) + o),
                                tuple(np.array((0.0, 0.0, 0.06)) + o),
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

    def frame_yaw() -> float:
        return float(scene._yaw_of(scene.frame)[0])

    def local_to_world(lx: float, ly: float) -> tuple[float, float]:
        w = scene._frame_world_xy(lx, ly)[0]
        return (float(w[0]) - float(scene.env_origins[0, 0]),
                float(w[1]) - float(scene.env_origins[0, 1]))

    def marker_err_deg() -> float:
        return math.degrees(float(scene.marker_err()[0]))

    def acc_deg() -> float:
        return math.degrees(float(scene.err_acc[0]))

    def report(tag: str) -> None:
        ip = (scene.idler.data.root_pos_w - scene.env_origins)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | idler=({float(ip[0]):+.3f},{float(ip[1]):+.3f},"
              f"{float(ip[2]):.3f}) seated={bool(scene.idler_seated()[0])} "
              f"decoy_seated={bool(scene.decoy_seated()[0])} "
              f"err={marker_err_deg():+.1f}deg acc={acc_deg():+.1f}deg "
              f"appr={float(scene.approach_latch[0]):.2f} "
              f"seat={float(scene.seat_latch[0]):.2f} "
              f"align={float(scene.align_latch[0]):.2f} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_body(body, x: float, y: float, z: float, yaw: float = 0.0,
                   settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def z_wrench(body, fz: float, tz: float) -> None:
        f = torch.tensor([0.0, 0.0, fz], device=device).view(1, 1, 3).expand(n, 1, 3)
        t = torch.tensor([0.0, 0.0, tz], device=device).view(1, 1, 3).expand(n, 1, 3)
        body.set_external_force_and_torque(
            f.contiguous(), t.contiguous(), env_ids=all_ids, is_global=True)

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def honest_seat(body) -> bool:
        """The honest installation move solve.py performs: hover the part above the
        centre axle, free-fall, and if it lands tooth-on-tooth, wiggle (alternating
        z-torque + light press) until it seats."""
        for yaw_off in (0.0, math.radians(22.5)):
            ax, ay = local_to_world(0.0, 0.0)
            place_body(body, ax, ay, c.plate_top + c.axle_h + 0.010,
                       yaw=frame_yaw() + yaw_off, settle_steps=90)
            seated_fn = scene.idler_seated if body is scene.idler else scene.decoy_seated
            for j in range(10):
                if bool(seated_fn()[0]):
                    break
                w_tgt = 2.0 if j % 2 == 0 else -2.0  # speed-GOVERNED jiggle
                for _ in range(45):
                    wz = float(body.data.root_ang_vel_w[0, 2])
                    z_wrench(body, -1.5, max(-0.06, min(0.06, 0.8 * (w_tgt - wz))))
                    env.step(no_action)
                    if bool(seated_fn()[0]):
                        break
                clear_wrench(body)
                step(25)
            clear_wrench(body)
            step(40)
            if bool(seated_fn()[0]):
                return True
        return False

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    idler_z = float((scene.idler.data.root_pos_w - scene.env_origins)[0, 2])
    fin0 = bool(torch.isfinite(scene.driver.data.root_state_w).all()
                and torch.isfinite(scene.output.data.root_state_w).all()
                and torch.isfinite(scene.idler.data.root_state_w).all()
                and torch.isfinite(scene.decoy.data.root_state_w).all())
    check("settle: states finite; driver + output seated on their axles (readback), "
          "idler flat on the floor, everything at rest",
          fin0 and bool(scene.driver_seated()[0])
          and bool(scene.seated(scene.output, c.spacing)[0])
          and not bool(scene.idler_seated()[0]) and idler_z < 0.02
          and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.03), no success", s <= 0.03 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        fp = (scene.frame.data.root_pos_w - scene.env_origins)[0]
        fy = frame_yaw()
        dy = _wrap(float(scene._yaw_of(scene.driver)[0]) - fy)
        rel = _wrap(float(scene._yaw_of(scene.output)[0]) - fy)
        delta0 = _wrap(c.stripe_az - rel)  # SIGNED initial dial error
        ip = (scene.idler.data.root_pos_w - scene.env_origins)[0]
        dp = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(fp[0]), float(fp[1]), fy, dy, delta0,
                      float(ip[0]), float(ip[1]), float(dp[0]), float(dp[1]),
                      float(scene.d0[0])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (frame_x, frame_y, frame_yaw, drv_yaw, "
          f"signed_delta0, idler_x, idler_y, decoy_x, decoy_y, d0):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: frame xy + yaw vary across seeded resets (readback)",
          spread[0] > 0.005 and spread[1] > 0.005 and spread[2] > 0.2)
    check("randomization: driver yaw, the SIGNED initial dial error (crank direction "
          "changes), idler xy, decoy xy, and the approach baseline d0 vary (readback)",
          spread[3] > 0.2 and spread[4] > 0.5 and spread[5] > 0.005
          and spread[6] > 0.005 and spread[7] > 0.005 and spread[8] > 0.005
          and spread[9] > 0.005)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.03 and not ok)

    # =========================== 6. SEED strategy ===========================================
    # The seed's whole plan is "carry the object somewhere and set it down — done". Here
    # that is: idler carried onto the plate BESIDE the empty axle and left. The rubric
    # must refuse it as an end state.
    env.reset(seed=41)
    step(10)
    sx, sy = local_to_world(0.0, 0.11)  # on the plate, beside the centre axle
    place_body(scene.idler, sx, sy, c.plate_top + 0.002, yaw=frame_yaw(), settle_steps=60)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (carry the part over and set it down): idler resting ON the "
          "plate near the axle — approach credit only, NOT seated, NOT success, "
          "score <= 0.15",
          not bool(scene.idler_seated()[0]) and float(scene.seat_latch[0]) < 0.5
          and not ok and s <= 0.15)

    # =========================== 7. teleported dial =========================================
    # Sharpest cheat: assemble the train HONESTLY, then WRITE the output's yaw onto the
    # stripe. Marker reads aligned, train complete, settled — every instantaneous
    # predicate passes — but the per-substep account saw a radians-scale jump (discarded)
    # so err_acc stays >> tol. The account is the anti-teleport contract.
    env.reset(seed=51)
    step(10)
    seated7 = honest_seat(scene.idler)
    report("honest-seat")
    acc_before = float(scene.err_acc[0])
    ox, oy = local_to_world(c.spacing, 0.0)
    place_body(scene.output, ox, oy, c.plate_top + 0.001,
               yaw=frame_yaw() + c.stripe_az, settle_steps=60)
    report("dial-written")
    s, ok = judge()
    check("teleported dial: train honestly complete and marker WRITTEN onto the stripe "
          "(reads within tol, settled) — yet err_acc kept the debt, NOT success, "
          "score <= 0.40",
          seated7 and bool(scene.idler_seated()[0]) and bool(scene.driver_seated()[0])
          and marker_err_deg() < c.tol_deg and bool(scene.settled()[0])
          and float(scene.err_acc[0]) > 0.8 * acc_before
          and float(scene.err_acc[0]) > c.tol and not ok and s <= 0.40)

    # =========================== 8. trainless rotation ======================================
    # Without the idler, slow-torque the OUTPUT onto the stripe: per-substep motion is
    # physically plausible (passes dpsi), but the gate (train incomplete) never credits
    # the decrease.
    env.reset(seed=61)
    step(10)
    for _ in range(2400):
        rel = _wrap(float(scene._yaw_of(scene.output)[0]) - frame_yaw())
        delta = _wrap(c.stripe_az - rel)
        if abs(delta) < 0.05:
            break
        direction = 1.0 if delta > 0 else -1.0
        wz = float(scene.output.data.root_ang_vel_w[0, 2])
        z_wrench(scene.output, 0.0, max(-0.08, min(0.08, 1.0 * (direction * 0.8 - wz))))
        env.step(no_action)
    for _ in range(90):
        wz = float(scene.output.data.root_ang_vel_w[0, 2])
        z_wrench(scene.output, 0.0, max(-0.08, min(0.08, -1.0 * wz)))
        env.step(no_action)
    clear_wrench(scene.output)
    step(60)
    report("trainless-spin")
    s, ok = judge()
    check("trainless rotation: output slow-torqued onto the stripe with the train "
          "INCOMPLETE — marker reads within tol yet the gate credited nothing "
          "(err_acc > tol), NOT success",
          marker_err_deg() < c.tol_deg and float(scene.err_acc[0]) > c.tol and not ok)

    # =========================== 9. decoy substitution ======================================
    # The SMOOTH disc seats on the centre axle, but its rim clears both neighbours'
    # teeth: cranking the driver spins the driver freely and transmits NOTHING.
    env.reset(seed=71)
    step(10)
    dec_seated = honest_seat(scene.decoy)
    err_before = marker_err_deg()
    w_max = 0.0
    for _ in range(1500):
        wz = float(scene.driver.data.root_ang_vel_w[0, 2])
        w_max = max(w_max, abs(wz))
        z_wrench(scene.driver, 0.0, max(-0.20, min(0.20, 2.0 * (1.2 - wz))))
        env.step(no_action)
    clear_wrench(scene.driver)
    step(90)
    report("decoy-cranked")
    s, ok = judge()
    check("decoy substitution: smooth disc seated on the centre axle (verified) and the "
          "driver cranked (spun > 0.8 rad/s) — no transmission: dial moved < 8 deg, "
          "NOT success, score <= 0.15",
          dec_seated and bool(scene.decoy_seated()[0]) and w_max > 0.8
          and abs(marker_err_deg() - err_before) < 8.0
          and not bool(scene.idler_seated()[0]) and not ok and s <= 0.15)

    # =========================== 10. near-miss seat =========================================
    env.reset(seed=81)
    step(10)
    nx, ny = local_to_world(0.035, 0.0)  # 35 mm off the axle: outside seat_xy_tol
    place_body(scene.idler, nx, ny, c.plate_top + c.axle_h + 0.02,
               yaw=frame_yaw(), settle_steps=90)
    report("near-miss-seat")
    s, ok = judge()
    check("near-miss: idler dropped 35 mm OFF the centre axle — lands on the plate but "
          "outside seat_xy_tol: not seated, no seat credit, NOT success",
          not bool(scene.idler_seated()[0]) and float(scene.seat_latch[0]) < 0.5 and not ok)

    # =========================== 11. latched credit survives regression =====================
    env.reset(seed=91)
    step(10)
    seated11 = honest_seat(scene.idler)
    report("seated")
    s_in, _ = judge()
    fx, fy_ = local_to_world(0.0, c.spawn_y + 0.10)
    place_body(scene.idler, fx, fy_, 0.02, settle_steps=60)
    report("idler-removed")
    s_out, ok = judge()
    check("latched credit: removing the seated idler leaves the latched approach + seat "
          "score unchanged (and success is gone)",
          seated11 and s_in >= 0.28 and abs(s_out - s_in) < 0.02 and not ok
          and not bool(scene.idler_seated()[0]))

    # =========================== 12. approach monotonicity ==================================
    env.reset(seed=101)
    step(5)
    ax, ay = local_to_world(0.0, 0.0)
    i0 = (scene.idler.data.root_pos_w - scene.env_origins)[0]
    dx, dy_ = ax - float(i0[0]), ay - float(i0[1])
    place_body(scene.idler, float(i0[0]) + 0.5 * dx, float(i0[1]) + 0.5 * dy_, 0.15,
               settle_steps=3)
    a_half = float(scene.approach_latch[0])
    place_body(scene.idler, float(i0[0]) + 0.9 * dx, float(i0[1]) + 0.9 * dy_, 0.15,
               settle_steps=3)
    a_near = float(scene.approach_latch[0])
    check("monotonicity: moving the idler closer to the centre axle latches strictly "
          f"more approach credit ({a_half:.3f} < {a_near:.3f})", a_half + 0.10 < a_near)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.frame.data.root_state_w).all()
           and torch.isfinite(scene.driver.data.root_state_w).all()
           and torch.isfinite(scene.output.data.root_state_w).all()
           and torch.isfinite(scene.idler.data.root_state_w).all()
           and torch.isfinite(scene.decoy.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.gear_train_dial")
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
