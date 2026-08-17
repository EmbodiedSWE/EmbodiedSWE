"""Smoke / rubric-REJECTION battery for TipFeederScene (sim_gen task `hockey_i185`)
— NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — ball placed into the hopper, hinge torque-tipped,
gravity delivery through the window, hopper released — is the acceptance evidence
that the rubric ACCEPTS a correct outcome). Every teleport here is instrumentation
that CONSTRUCTS a wrong (or partial) outcome as a settled state and asserts the
rubric REJECTS it; no probe in this battery ever reaches success(), and a final
audit check asserts exactly that.

  1.  settle/no-NaN      — reset layout settles finite: hopper gravity-seated at
                           -12 deg, ball at its side slot, all still; MASS READBACK
                           (root_physx_view.get_masses) proves the authored masses
                           took (custom spawners silently ignore cfg mass_props);
  2.  score ~0 at reset  — no credit for existing;
  3-4. randomization     — READBACK over 8 seeded resets: ball side flips
                           (Bernoulli), per-side xy jitter, apparatus yaw + offset
                           spreads are real;
  5.  null policy        — 240 idle steps -> score ~0, no success, hopper seated;
  6.  seed strategy      — the end state the SEED's plan produces here (propel the
                           ball at the goal): the ball is FIRED along the floor at
                           the box; it passes under the hopper, bounces off the
                           sealed lower wall and ends OUTSIDE -> score ~0, no
                           success (there is no floor-level opening);
  7.  drop-in (roof)     — ball released over the box rests ON THE ROOF -> ~0;
  8.  drop-in (hood)     — ball released right above the window lane is DEFLECTED
                           by the hood (onto the floor or into the hopper basin
                           below) and never enters the box -> no delivery credit;
  9.  load-only          — ball settled in the basin -> NOT success, score = the
                           0.20 load credit band only;
  10. near-miss sill     — ball resting ON the sill shelf inside the window tunnel
                           (past the lip plane but ABOVE the inside_z gate) -> NOT
                           success (containment is judged BELOW the aperture);
                           judged on a short contact-settle: the perch is
                           metastable and a long settle would roll INTO the goal;
  11. wrong place        — ball settled against the box's OUTSIDE side wall at
                           in-range height -> box-frame math rejects;
  12. held-tilted        — ball constructed inside AND the hopper HELD at +30 deg
                           by torque: every ball gate passes, the seat clause alone
                           rejects (the machine must be released) -> NOT success;
                           the ball is then removed BEFORE the hold is released;
  13. latched credit     — after the ball is teleported back out, the latched
                           credit is unchanged (does not evaporate), still no
                           success;
  14. rejection audit    — success() was never True at ANY judged point;
  15. final no-NaN       — all task-object states finite at the end.

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
# DAEMON so an exception in main() (handled below with os._exit) never leaves the
# process idling for the full watchdog interval.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tip_feeder")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -1.05, 0.85)) + o),
                                tuple(np.array((0.42, 0.00, 0.15)) + o),
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

    def theta_deg() -> float:
        return math.degrees(float(scene.hinge_angle()[0]))

    def report(tag: str) -> None:
        hl = scene._local(scene.hopper, scene.ball)[0]
        bl = scene._local(scene.box, scene.ball)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | theta={theta_deg():+6.1f}deg "
              f"ball_hop=({float(hl[0]):+.3f},{float(hl[1]):+.3f},{float(hl[2]):+.3f}) "
              f"ball_box=({float(bl[0]):+.3f},{float(bl[1]):+.3f},{float(bl[2]):+.3f}) "
              f"basin={bool(scene._in_basin()[0])} inside={bool(scene._inside()[0])} "
              f"load={bool(scene._load_ever[0])} tilt={bool(scene._tilt_ever[0])} "
              f"dlv={bool(scene._dlv_ever[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_box_local(x: float, y: float, z: float, vel_x: float = 0.0) -> None:
        """Teleport the ball to a box-frame point of the box's CURRENT pose (probe
        constructor: builds inside/on-top/beside relations directly), optionally
        with an initial velocity along the box axis (the seed's shot)."""
        b_pos = scene.box.data.root_pos_w
        b_quat = scene.box.data.root_quat_w
        loc = torch.tensor([x, y, z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = b_pos + quat_apply(b_quat, loc)
        st[:, 3] = 1.0
        if vel_x:
            axis = quat_apply(b_quat, torch.tensor([1.0, 0.0, 0.0],
                                                   device=device).expand(n, 3))
            st[:, 7:10] = axis * vel_x
        scene.ball.write_root_state_to_sim(st, all_ids)

    def place_hopper_local(x: float, y: float, z: float) -> None:
        h_pos = scene.hopper.data.root_pos_w
        h_quat = scene.hopper.data.root_quat_w
        loc = torch.tensor([x, y, z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = h_pos + quat_apply(h_quat, loc)
        st[:, 3] = 1.0
        scene.ball.write_root_state_to_sim(st, all_ids)

    def place_world(x: float, y: float, z: float, vel_x: float = 0.0) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 7] = vel_x
        st[:, 0:3] += scene.env_origins
        scene.ball.write_root_state_to_sim(st, all_ids)

    def hold_hinge(tau: float) -> None:
        trq = torch.zeros(n, 1, 3, device=device)
        trq[:, 0, 1] = tau
        scene.hopper.set_external_force_and_torque(zero_wrench, trq.contiguous(),
                                                   env_ids=all_ids, is_global=False)

    def ball_xy() -> tuple[float, float]:
        p = (scene.ball.data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1])

    def stand_yaw() -> float:
        q = scene.stand.data.root_quat_w[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def finite_all() -> bool:
        return bool(torch.isfinite(scene.box.data.root_state_w).all()
                    and torch.isfinite(scene.stand.data.root_state_w).all()
                    and torch.isfinite(scene.hopper.data.root_state_w).all()
                    and torch.isfinite(scene.ball.data.root_state_w).all())

    # =========================== 1-2. settle / no-NaN / masses ==============================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(120)
    report("show")
    masses_ok = True
    try:
        mh = float(scene.hopper.root_physx_view.get_masses()[0, 0])
        mb = float(scene.ball.root_physx_view.get_masses()[0, 0])
        ms = float(scene.stand.root_physx_view.get_masses()[0, 0])
        print(f"[smoke] mass readback: hopper={mh:.3f} ball={mb:.3f} stand={ms:.1f}",
              flush=True)
        masses_ok = (abs(mh - c.hopper_mass) < 0.05 * c.hopper_mass
                     and abs(mb - c.ball_mass) < 0.05 * c.ball_mass
                     and ms > 20.0)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] mass readback FAILED ({exc!r})", flush=True)
        masses_ok = False
    still = (float(scene.ball.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.hopper.data.root_ang_vel_w[0].norm()) < c.settle_omega)
    check("settle: states finite, hopper gravity-seated near -12 deg, ball at its "
          "slot, all still; authored masses verified by readback",
          finite_all() and abs(theta_deg() - c.seat_lim_deg) < 2.5 and still
          and not bool(scene._in_basin()[0]) and not bool(scene._inside()[0])
          and masses_ok)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        bx, by = ball_xy()
        spx = float((scene.stand.data.root_pos_w - scene.env_origins)[0, 0])
        spy = float((scene.stand.data.root_pos_w - scene.env_origins)[0, 1])
        reads.append((bx, by, 1.0 if by > 0 else 0.0, stand_yaw(), spx, spy))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (ball_x, ball_y, ball_left, stand_yaw_deg, "
          f"stand_x, stand_y):\n{arr}", flush=True)
    flags = arr[:, 2]
    check("randomization: ball side flips across seeded resets (Bernoulli readback)",
          0.0 < flags.mean() < 1.0)
    jit = 0.0
    for flag in (0.0, 1.0):
        grp = arr[flags == flag]
        if len(grp) >= 2:
            jit = max(jit, float((grp[:, 0:2].max(axis=0) - grp[:, 0:2].min(axis=0)).max()))
    yaw_spread = float(arr[:, 3].max() - arr[:, 3].min())
    off_spread = float((arr[:, 4:6].max(axis=0) - arr[:, 4:6].min(axis=0)).max())
    check("randomization: per-side ball jitter (> 4 mm), apparatus yaw spread "
          "(> 2 deg) and offset spread (> 4 mm) are real (readback)",
          jit > 0.004 and yaw_spread > 2.0 and off_spread > 0.004)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0, no success, hopper still seated after 240 idle steps",
          s <= 0.02 and not ok and theta_deg() < c.seat_deg)

    # =========================== 6. seed strategy: shoot the ball at the box ================
    # The seed's plan — propel the ball at the goal — executed here: fired along the
    # floor at the box it passes UNDER the seated hopper, bounces off the sealed
    # lower wall and never enters (there is no floor-level opening).
    torch.manual_seed(41)
    env.reset()
    step(30)
    place_world(0.10, 0.0, c.ball_r + 0.002, vel_x=1.5)
    x0, y0 = ball_xy()
    step(300)
    report("seed-strategy")
    x1, y1 = ball_xy()
    traveled = math.hypot(x1 - x0, y1 - y0)
    bl = scene._local(scene.box, scene.ball)[0]
    s, ok = judge()
    check("seed strategy: ball FIRED along the floor at the box travels, bounces off "
          "the sealed lower wall and ends OUTSIDE (below-sill wall is the interlock) "
          "— score ~0 (<= 0.02), no success",
          traveled > 0.10 and not bool(scene._inside()[0])
          and float(bl[2]) < c.sill_z and not ok and s <= 0.02)

    # =========================== 7. drop-in attempt: on the roof ============================
    torch.manual_seed(51)
    env.reset()
    step(30)
    place_box_local(0.0, 0.0, c.roof_z + c.wall_t + c.ball_r + 0.004)
    step(240)
    report("on-roof")
    s, ok = judge()
    check("drop-in (roof): ball released over the box rests ON THE ROOF (or rolls "
          "off outside) — never inside, score ~0 (<= 0.02), no success",
          not bool(scene._inside()[0]) and not ok and s <= 0.02)

    # =========================== 8. drop-in attempt: above the window lane ==================
    # Released right above the window tunnel: the HOOD TOP catches it — the overhang
    # is the drop-in interlock for the only opening.
    torch.manual_seed(61)
    env.reset()
    step(30)
    place_box_local(-c.in_hx - c.wall_t - 0.005, 0.0,
                    c.win_top + c.wall_t + c.ball_r + 0.004)
    step(240)
    report("on-hood")
    bl = scene._local(scene.box, scene.ball)[0]
    s, ok = judge()
    # The hood may DEFLECT the ball forward into the open hopper basin waiting
    # below (observed on the forge) — that is the interlock doing its job, and the
    # 0.20 load credit for a ball genuinely in the basin is honest. What must hold:
    # the ball never ENTERS the box.
    check("drop-in (hood): ball released right above the window lane is deflected "
          "by the HOOD (onto the floor or into the hopper basin below) and never "
          "enters the box — NOT success, no delivery credit (score <= 0.21)",
          not bool(scene._inside()[0]) and not bool(scene._dlv_ever[0])
          and not ok and s <= 0.21)

    # =========================== 9. load-only: ball parked in the basin =====================
    torch.manual_seed(71)
    env.reset()
    step(30)
    place_hopper_local(-0.100, 0.0, 0.040)
    step(180)
    report("load-only")
    s9, ok = judge()
    check("load-only: ball settled in the hopper basin is only the 0.20 load credit "
          "— NOT success, 0.19 <= score <= 0.21",
          bool(scene._in_basin()[0]) and not ok and 0.19 <= s9 <= 0.21)

    # =========================== 10. near-miss: perched on the sill =========================
    # Past the lip plane, at window height — but ABOVE the inside_z gate: containment
    # is judged BELOW the aperture, so a ball that stopped in the window tunnel (the
    # delivery's one honest near-miss) scores no delivery credit and never succeeds.
    # The sill perch is METASTABLE: on the forge the ball slowly rolled off the
    # shelf INTO the box (solver creep on the slick shelf) — a probe that settles
    # long would route THROUGH the goal. So: 0.5 mm set-down gap (no jolt), a
    # short 0.2 s contact-settle (creep moves the ball < 10 mm on the 37 mm
    # shelf), judged while it demonstrably rests in the tunnel above the gate.
    torch.manual_seed(81)
    env.reset()
    step(30)
    place_box_local(-c.in_hx - c.wall_t - c.apron_d / 2, 0.0,
                    c.sill_z + c.ball_r + 0.0005)
    step(24)
    report("near-miss-sill")
    bl = scene._local(scene.box, scene.ball)[0]
    on_sill = (float(bl[2]) > c.inside_z_max
               and float(scene.ball.data.root_lin_vel_w[0].norm()) < 3 * c.settle_speed)
    s, ok = judge()
    check("near-miss sill: ball resting ON the sill shelf in the window tunnel "
          "(past the lip plane but above the inside_z gate) — NOT success, no "
          "delivery credit (score <= 0.02): containment is judged BELOW the "
          "aperture",
          on_sill and not bool(scene._inside()[0]) and not bool(scene._dlv_ever[0])
          and not ok and s <= 0.02)

    # =========================== 11. wrong place: against the outside wall ==================
    torch.manual_seed(91)
    env.reset()
    step(30)
    place_box_local(0.0, c.in_hy + c.wall_t + c.ball_r + 0.003, c.ball_r + 0.002)
    step(150)
    report("beside-wall")
    bl = scene._local(scene.box, scene.ball)[0]
    s, ok = judge()
    check("wrong place: ball settled against the box's OUTSIDE side wall at in-range "
          "height — box-frame math rejects: NOT success, score ~0",
          abs(float(bl[1])) > c.in_hy and not bool(scene._inside()[0])
          and not ok and s <= 0.02)

    # =========================== 12. held-tilted: the seat clause ===========================
    # Ball constructed inside AND the hopper held at ~+30 deg by torque: every ball
    # gate passes; the seat clause alone must reject (the machine must be RELEASED).
    # The ball is removed BEFORE the hold is released so the reseated hopper never
    # completes a genuine success (protects the never-success audit).
    torch.manual_seed(101)
    env.reset()
    step(30)
    place_box_local(0.02, 0.0, c.ball_r + 0.004)
    # PD + gravity-ff servo with a stall-escalating clamp (the effectively-applied
    # external torque on the forge pods is a fraction of the commanded one — a
    # fixed sub-N*m hold stalls below the tilt-credit angle, as the first solve
    # run proved at +11.6 deg).
    tgt = math.radians(30.0)
    ff0 = c.hopper_mass * 9.81 * 0.090
    cap = 0.8
    th_prev = float(scene.hinge_angle()[0])
    ref = th_prev
    for i in range(720):
        th = float(scene.hinge_angle()[0])
        w = (th - th_prev) * 120.0
        th_prev = th
        tau = ff0 * math.cos(th) + 2.5 * (tgt - th) - 0.06 * w
        hold_hinge(max(-cap, min(cap, tau)))
        env.step(no_action)
        if i % 60 == 59:
            if th - ref < math.radians(0.5) and th < math.radians(26.0):
                cap = min(cap * 1.7, 8.0)
                print(f"[smoke] held-tilted: hinge stalled at "
                      f"{math.degrees(th):+.1f}deg — torque clamp -> {cap:.2f} N*m",
                      flush=True)
            ref = th
        if th > math.radians(28.0) and i >= 300:  # tilted AND ball had time to settle
            break
    report("held-tilted")  # judged while the hold torque is still applied
    s12, ok12 = judge()
    held_ok = (theta_deg() > c.tilt_credit_deg and bool(scene._inside()[0])
               and not ok12)
    # remove the ball, then release the hold
    place_world(0.20, -0.30, c.ball_r + 0.002)
    step(30)
    hold_hinge(0.0)
    step(240)
    report("released-empty")
    check("held-tilted: ball inside but hopper HELD at +30 deg — the seat clause "
          "alone rejects (NOT success); hopper gravity-reseats once released",
          held_ok and theta_deg() < c.seat_deg)

    # =========================== 13. latched credit survives regression =====================
    s13, ok = judge()
    check("latched credit: after the ball is teleported back OUT the latched score "
          f"is unchanged ({s12:.3f} -> {s13:.3f}, tilt+delivery credit kept, <= "
          "0.70 cap), still no success",
          abs(s13 - s12) < 1e-3 and s13 <= 0.701 and not bool(scene._inside()[0])
          and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.tip_feeder")
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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
