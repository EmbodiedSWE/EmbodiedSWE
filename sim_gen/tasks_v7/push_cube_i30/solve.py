"""Teleport solution for SiloScoopScene (sim_gen task `push_cube_i30`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY and always both starts and ends in FREE SPACE
(a non-contact hover); every load-bearing interaction goes through CONTACT DYNAMICS
via external forces on the SCOOP only — no force or teleport ever touches a ball:
  PROBE — this stack's `set_external_force_and_torque(is_global=True)` may apply the
  wrench rotated by the body's rotation since env.reset (pod/version dependent), so
  before any real work single-step force impulses on the free-hovering scoop measure
  the applied-frame matrix M directly from dv responses; the reference orientation is
  recovered from M and the resulting encoding is verified closed-loop (hover + 60 deg
  pitch) before use, with a drag-toggle fallback.
  Per TRIP (repeated while a present ball remains in the silo):
  T-a DIP: the scoop is teleported to an upright hover ABOVE the silo mouth over the
      guaranteed ball-free entry window (pan bottom above the rim — intersecting
      nothing), then force-servoed straight down through the mouth until the pan
      rests on the silo floor. The 7.5 mm per-side clearance is real: the descent is
      a genuinely constrained insertion.
  T-b SHOVEL: the pan is pressed down and stroked toward the +x wall at 6 cm/s; the
      ramp lip wedges under the balls and they ride up into the pan (dustpan-against-
      the-wall, pure contact; the chamfered corners funnel them into the mouth).
  T-c EXTRACT: the pan pitches nose-up (25 deg, then deepened in place to 45 deg)
      pivoting about its REAR bottom edge while the x carrot keeps pressing toward
      the wall (the servo holds the retracting tip against the wall, so forward
      escape stays blocked while the wedged ball rides the rising ramp and rolls
      back over the crest into the pan), then retreats, still tilted, and rises
      back out of the mouth.
  T-d CARRY: a slow eased glide through the air to a spot short of the tray centre —
      the balls ride LOOSE in the open pan the whole way (carry accel ~0.2 m/s^2,
      far below the friction cone; nothing holds them in but the pan walls).
  T-e DUMP: hovering low over the tray, the pan pitches nose-down; each ball rolls
      down the pan, over the lip, and FALLS into the tray under gravity (with a
      deeper-tilt wiggle fallback). Delivery is rolling contact + gravity only.
  STOW: the now-empty hovering scoop is teleported to open floor (free space, 6 mm
      up), dropped, and everything settles hands-off.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.push_cube_i30.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import math
import os
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# Servo gains: the position PD commands ACCELERATION per kg and the attitude PD
# commands ANGULAR acceleration mapped through the body's true world-frame inertia
# tensor (the scoop's inertia is wildly anisotropic — a long thin handle makes I_z
# ~50x smaller than I_x/I_y, so fixed torque gains would be discretely unstable
# about the handle axis). Both loops use the same ~critically-damped pair.
KP, KD = 150.0, 24.0  # m/s^2 per m, per (m/s)
# The attitude loop must be STIFF against contact couples: pressing the pan into
# the wall with ~4.5 N acts at the COM 0.13 m up the handle, a ~0.6 N.m pitch
# moment. At KRA=150 the pitch stiffness was only KRA*I_yy ~ 0.45 N.m/rad, so
# the scoop pitched, dug its tip and rode up over the ball (observed). KRA=2000
# gives ~6 N.m/rad (residual sag ~6 deg), and the known couple is additionally
# cancelled by a feedforward (see servo_step's pivot_b). Discrete stability at
# 120 Hz: w_n*dt = sqrt(2000)/120 ~ 0.37, zeta = 90/(2*sqrt(2000)) ~ 1.0.
KRA, KRDA = 2000.0, 90.0  # rad/s^2 per rad, per (rad/s)
# TAU_MAX sizing: a wrist gripping the handle resists pitch/roll geometrically
# (jaw purchase, not friction), so 2 N.m is embodiment-honest.
F_MAX, TAU_MAX = 8.0, 2.0


def main() -> None:  # noqa: PLR0915
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.silo_scoop")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import (matrix_from_quat, quat_apply, quat_apply_inverse,
                                     quat_from_matrix, quat_mul)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    origin0 = scene.env_origins[0]

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def qrot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return quat_apply(q.view(1, 4), v.view(1, 3)).view(3)

    def qinv(q: torch.Tensor) -> torch.Tensor:
        out = q.clone()
        out[1:] = -out[1:]
        return out

    def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return quat_mul(a.view(1, 4), b.view(1, 4)).view(4)

    def qz(yaw: float) -> torch.Tensor:
        return torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
                            device=device)

    def qy(pitch: float) -> torch.Tensor:
        return torch.tensor([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0],
                            device=device)

    def scoop_p() -> torch.Tensor:
        return scene.scoop.data.root_pos_w[0] - origin0

    def teleport_scoop(pos, quat: torch.Tensor) -> None:
        """TRANSPORT ONLY: rewrite the scoop's root state (zero velocity) at a
        free-space pose; forces are zeroed in the same step."""
        scene.scoop.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                  env_ids=all_ids)
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = float(pos[0]), float(pos[1]), float(pos[2])
        st[:, 3:7] = quat.view(1, 4)
        st[:, 0:3] += scene.env_origins
        scene.scoop.write_root_state_to_sim(st, all_ids)

    # ----- force-frame state (see module docstring: pod-dependent wrench frame) ------------
    # Known pod quirk: with is_global=True the stack applies
    #   applied_world = (R_now . R_ref^T) . given
    # to BOTH the force and the torque, where R_ref is the body's orientation at
    # env.reset (write_root_state does NOT re-anchor it). The impulse probe below
    # measures M = R_now.R_ref^T directly from dv responses at a known R_now, so
    # q_ref is recovered empirically (R_ref = M^T . R_probe) rather than assumed.
    # If M ~ I on this pod, drag stays False and wrenches pass through raw.
    frame = {"q_ref": torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
             "drag": False, "lever": False}

    # True body-frame inertia tensor + COM from PhysX (uniform-density from the
    # collision shapes): the attitude loop maps angular acceleration through the
    # inertia, and the COM lever compensates force-application torque if the stack
    # applies wrenches at the link origin rather than the COM (probed below).
    inertia_b = (scene.scoop.root_physx_view.get_inertias()
                 .view(-1, 3, 3)[0].to(device=device, dtype=torch.float32))
    com_b = (scene.scoop.root_physx_view.get_coms()
             .view(-1, 7)[0, :3].to(device=device, dtype=torch.float32))
    print(f"[solve] scoop inertia diag={inertia_b.diagonal().tolist()} "
          f"com_b={com_b.tolist()}", flush=True)

    def servo_step(pos_des: torch.Tensor, q_des: torch.Tensor,
                   f_down: float = 0.0, attitude: bool = True,
                   pivot_b: torch.Tensor | None = None) -> None:
        """One physics step under a 6-DoF wrench servo on the scoop: position PD +
        gravity feedforward (+ optional press-down bias), attitude PD on the
        axis-angle error, world wrench encoded per the probed force-frame mode.
        pivot_b: body-frame contact point; when set, feed forward the couple that
        the commanded HORIZONTAL force (applied at the COM, reacted quasi-
        statically at that contact) exerts about the COM, so sustained presses do
        not pitch the scoop:  tau_ff = (pivot - com) x f_h  (world)."""
        p, v = scoop_p(), scene.scoop.data.root_lin_vel_w[0]
        err = (pos_des - p).clamp(-0.20, 0.20)
        f_w = c.mass * (KP * err - KD * v)
        f_w[2] += c.mass * 9.81 - f_down
        fn = float(f_w.norm())
        if fn > F_MAX:
            f_w = f_w * (F_MAX / fn)
        q_now = scene.scoop.data.root_quat_w[0]
        rot = matrix_from_quat(q_now.view(1, 4))[0]
        if attitude:
            q_err = qmul(q_des, qinv(q_now))
            if float(q_err[0]) < 0.0:
                q_err = -q_err
            vlen = float(q_err[1:].norm())
            ang = 2.0 * math.atan2(vlen, float(q_err[0]))
            rotvec = q_err[1:] / max(vlen, 1e-9) * ang
            alpha = KRA * rotvec - KRDA * scene.scoop.data.root_ang_vel_w[0]
            tau_w = rot @ inertia_b @ rot.T @ alpha
        else:
            tau_w = torch.zeros(3, device=device)
        if pivot_b is not None:
            f_h = f_w.clone()
            f_h[2] = 0.0
            tau_w = tau_w + torch.linalg.cross(rot @ (pivot_b - com_b), f_h)
        tn = float(tau_w.norm())
        if tn > TAU_MAX:
            tau_w = tau_w * (TAU_MAX / tn)
        if frame["lever"]:
            # wrench applies at the link origin: cancel the (origin - com) x F couple
            tau_w = tau_w + torch.linalg.cross(rot @ com_b, f_w)
        if frame["drag"]:
            # pre-rotate so that (R_now.R_ref^T).given == desired world wrench
            q_corr = qmul(frame["q_ref"], qinv(q_now))
            f_arg, t_arg = qrot(q_corr, f_w), qrot(q_corr, tau_w)
        else:
            f_arg, t_arg = f_w, tau_w
        frame["last_f"], frame["last_tau"] = f_w, tau_w
        scene.scoop.set_external_force_and_torque(
            f_arg.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            t_arg.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            env_ids=all_ids, is_global=True)
        env.step(no_action)

    def report(tag: str) -> None:
        sil = scene.in_silo()[0]
        bas = scene.in_basin()[0]
        pres = scene.present[0]
        sp = scoop_p()
        print(f"[solve] {tag:12s} | present={pres.tolist()} in_silo={sil.tolist()} "
              f"in_basin={bas.tolist()} scoop=({float(sp[0]):+.3f},{float(sp[1]):+.3f},"
              f"{float(sp[2]):+.3f}) stowed={bool(scene.scoop_stowed()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(90)
    silo_xy = (scene.silo.data.root_pos_w[0, :2] - origin0[:2]).clone()
    sq = scene.silo.data.root_quat_w[0]
    silo_yaw = 2.0 * math.atan2(float(sq[3]), float(sq[0]))
    basin_xy = (scene.basin.data.root_pos_w[0, :2] - origin0[:2]).clone()
    cyw, syw = math.cos(silo_yaw), math.sin(silo_yaw)
    u_nose = torch.tensor([cyw, syw], device=device)  # silo-local +x in world

    def sl2w(lx: float, ly: float) -> tuple[float, float]:
        return (float(silo_xy[0]) + lx * cyw - ly * syw,
                float(silo_xy[1]) + lx * syw + ly * cyw)

    def ball_silo_x(i: int) -> float:
        loc = scene._local(scene.silo, scene.balls[i].data.root_pos_w)[0]
        return float(loc[0])

    def in_pan(i: int) -> bool:
        bp = scene.balls[i].data.root_pos_w[0]
        sp = scene.scoop.data.root_pos_w[0]
        sq_ = scene.scoop.data.root_quat_w[0]
        loc = quat_apply_inverse(sq_.view(1, 4), (bp - sp).view(1, 3)).view(3)
        return (-0.032 < float(loc[0]) < c.pan_len / 2 + 0.004
                and abs(float(loc[1])) < c.pan_w / 2 + 0.006
                and -0.010 < float(loc[2]) < c.side_h + 0.020)

    pres0 = scene.present[0]
    k_present = int(pres0.sum())
    print(f"[solve] layout readback (seed {args.seed}): "
          f"silo=({float(silo_xy[0]):+.3f},{float(silo_xy[1]):+.3f}) "
          f"yaw={math.degrees(silo_yaw):+.1f}deg "
          f"basin=({float(basin_xy[0]):+.3f},{float(basin_xy[1]):+.3f}) "
          f"present={pres0.tolist()} (k={k_present}) "
          f"ball_x_local={[round(ball_silo_x(i), 3) for i in range(c.n_balls)]}",
          flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: PROBE the wrench conventions in empty space -----------------
    # 1a: single-step IMPULSE experiments — ground truth for how this stack turns the
    # (force, torque) arguments into rigid-body wrenches. From a fresh free hover:
    # a known 1-step force impulse gives dv (force frame + scale) and dw (application
    # point: at the COM dw=0; at the link origin dw = I^-1 (o-c)xF dt ~ -0.7 rad/s
    # about y for 2 N in +x); a known 1-step torque impulse gives the torque scale;
    # zero-wrench steps in between expose persistence/accumulation semantics.
    probe_pos = torch.tensor([-0.40, -0.40, 0.55], device=device)

    def imp_step(fv, tv, tag: str) -> tuple[torch.Tensor, torch.Tensor]:
        fw = torch.zeros(n, 1, 3, device=device)
        tw = torch.zeros(n, 1, 3, device=device)
        fw[:, 0, :] = torch.tensor(fv, device=device)
        tw[:, 0, :] = torch.tensor(tv, device=device)
        v0 = scene.scoop.data.root_lin_vel_w[0].clone()
        w0 = scene.scoop.data.root_ang_vel_w[0].clone()
        scene.scoop.set_external_force_and_torque(fw, tw, env_ids=all_ids,
                                                  is_global=True)
        env.step(no_action)
        dv = scene.scoop.data.root_lin_vel_w[0] - v0
        dw = scene.scoop.data.root_ang_vel_w[0] - w0
        print(f"[solve]   imp {tag}: dv=({float(dv[0]):+.4f},{float(dv[1]):+.4f},"
              f"{float(dv[2]):+.4f}) dw=({float(dw[0]):+.3f},{float(dw[1]):+.3f},"
              f"{float(dw[2]):+.3f})", flush=True)
        return dv, dw

    teleport_scoop(probe_pos, qz(0.0))
    dv_0, _ = imp_step((0, 0, 0), (0, 0, 0), "zero (expect dv_z=-0.082)")
    dv_0 = dv_0.clone()
    dv_fx, dw_fx = imp_step((2.0, 0, 0), (0, 0, 0), "F=2x")
    dv_fx, dw_fx = dv_fx.clone(), dw_fx.clone()
    imp_step((0, 0, 0), (0, 0, 0), "zero (persistence check)")
    q_probe = scene.scoop.data.root_quat_w[0].clone()  # R_now for the y impulse
    dv_fy, _ = imp_step((0, 2.0, 0), (0, 0, 0), "F=2y")
    dv_fy = dv_fy.clone()
    imp_step((0, 0, 0), (0, 0, 0), "zero")
    dv_t, dw_t = imp_step((0, 0, 0), (0, 0.05, 0), "T=0.05y")
    imp_step((0, 0, 0), (0, 0, 0), "zero (persistence check)")

    # Application point from the measured force->spin coupling (COM => dw ~ 0).
    lever_meas = float(-dw_fx[1]) * float(inertia_b[1, 1]) / (2.0 / 120.0)
    frame["lever"] = abs(lever_meas) > 0.03
    # Build the applied-frame matrix M column by column from the dv responses
    # (gravity-baseline subtracted, then orthonormalised): applied = M . given.
    c1 = dv_fx - dv_0
    c1 = c1 / c1.norm()
    c2 = dv_fy - dv_0
    c2 = c2 - (c2 @ c1) * c1
    c2 = c2 / c2.norm()
    c3 = torch.linalg.cross(c1, c2)
    M = torch.stack((c1, c2, c3), dim=1)
    drag_err = float((M - torch.eye(3, device=device)).norm())
    frame["drag"] = drag_err > 0.20
    if frame["drag"]:
        # M = R_probe . R_ref^T  =>  R_ref = M^T . R_probe (q_probe ~ identity here,
        # but use the measured quat rather than assuming the teleport stuck exactly).
        r_ref = M.T @ matrix_from_quat(q_probe.view(1, 4))[0]
        frame["q_ref"] = quat_from_matrix(r_ref.view(1, 3, 3)).view(4).clone()
    print(f"[solve] probe 1a: lever ~{lever_meas:+.3f} m -> "
          f"{'ON' if frame['lever'] else 'OFF'}; |M-I|={drag_err:.3f} -> drag "
          f"{'ON' if frame['drag'] else 'OFF'} "
          f"(M x_hat->({float(c1[0]):+.2f},{float(c1[1]):+.2f},{float(c1[2]):+.2f})); "
          f"q_ref={[round(float(x), 3) for x in frame['q_ref']]}; "
          f"torque dw=({float(dw_t[0]):+.3f},{float(dw_t[1]):+.3f},"
          f"{float(dw_t[2]):+.3f})", flush=True)

    # 1b: closed-loop verification of the selected encoding — hover then pitch to
    # 60 deg under the full 6-DoF servo. Dense telemetry so any residual instability
    # is attributable from stdout alone. If tracking fails, toggle drag and retry.
    def att_probe(tag: str, steps: int, tilt_deg: float,
                  attitude: bool = True) -> tuple[float, float]:
        teleport_scoop(probe_pos, qz(0.0))
        max_err, a_e = 0.0, 0.0
        for i in range(steps):
            fr = min(1.0, i / max(steps * 0.6, 1.0))
            q_des = qy(math.radians(tilt_deg) * fr)
            servo_step(probe_pos, q_des, attitude=attitude)
            pe_v = scoop_p() - probe_pos
            pe = float(pe_v.norm())
            max_err = max(max_err, pe)
            q_now = scene.scoop.data.root_quat_w[0]
            q_e = qmul(q_des, qinv(q_now))
            a_e = 2.0 * math.atan2(float(q_e[1:].norm()), abs(float(q_e[0])))
            if i % 15 == 14:
                w = scene.scoop.data.root_ang_vel_w[0]
                lf, lt = frame["last_f"], frame["last_tau"]
                print(f"[solve]   {tag} i={i + 1}: perr=({float(pe_v[0]):+.3f},"
                      f"{float(pe_v[1]):+.3f},{float(pe_v[2]):+.3f}) "
                      f"att={math.degrees(a_e):5.1f}deg "
                      f"w=({float(w[0]):+.2f},{float(w[1]):+.2f},{float(w[2]):+.2f}) "
                      f"f=({float(lf[0]):+.2f},{float(lf[1]):+.2f},{float(lf[2]):+.2f}) "
                      f"tau=({float(lt[0]):+.3f},{float(lt[1]):+.3f},"
                      f"{float(lt[2]):+.3f})", flush=True)
            if pe > 0.30:
                break
        print(f"[solve] {tag}: max_pos_err={max_err:.3f} "
              f"att_err={math.degrees(a_e):.1f}deg", flush=True)
        return max_err, math.degrees(a_e)

    pe_v_, ae_v = att_probe("V1 hover", 90, 0.0)
    pe_t, ae_t = att_probe("V2 tilt60", 210, 60.0)
    if pe_t > 0.10 or ae_t > 15.0:  # selected encoding does not track: toggle drag
        print(f"[solve] verification failed (perr={pe_t:.3f} att={ae_t:.1f}deg); "
              f"toggling drag -> {not frame['drag']}", flush=True)
        frame["drag"] = not frame["drag"]
        pe_t, ae_t = att_probe("V3 tilt60 (toggled)", 210, 60.0)
    assert pe_t < 0.10 and ae_t < 15.0, (
        f"wrench servo cannot track under either encoding "
        f"(perr={pe_t:.3f} att={ae_t:.1f}deg)")
    print(f"[solve] wrench conventions selected: drag={frame['drag']} "
          f"lever={frame['lever']} (verify perr={pe_t:.3f} att={ae_t:.1f}deg)",
          flush=True)
    s1 = print_score("P1 wrench-convention probe")
    assert s1 >= s0 - 1e-6, "score decreased across the probe phase"

    # ---------------- trips: dip -> shovel -> extract -> carry -> dump ---------------------
    z_floor = c.floor_t + c.pan_t + 0.001  # scoop origin height, pan resting on silo floor
    x_end = c.in_x / 2 - c.nose_x - 0.002  # stroke end: nose 2 mm short of the +x wall
    tip_b = torch.tensor([c.nose_x, 0.0, -c.pan_t], device=device)  # nose tip (body)
    dt = 1.0 / 120.0

    def glide(p0: torch.Tensor, p1: torch.Tensor, q_of, vmax: float,
              amax: float, f_down: float = 0.0) -> None:
        """Eased (cosine) straight-line carrot from p0 to p1; q_of(frac) gives the
        attitude carrot. Peak vel/accel bounded by vmax/amax."""
        length = float((p1 - p0).norm())
        T = max(math.pi * length / (2 * vmax),
                math.sqrt(math.pi ** 2 * length / (2 * amax)), 0.2)
        steps = int(T / dt) + 1
        for i in range(steps):
            fr = 0.5 * (1.0 - math.cos(math.pi * min(1.0, i / (T / dt))))
            servo_step(p0 + (p1 - p0) * fr, q_of(fr), f_down=f_down)

    def hold(pos: torch.Tensor, q_des: torch.Tensor, k: int,
             f_down: float = 0.0) -> None:
        for _ in range(k):
            servo_step(pos, q_des, f_down=f_down)

    def do_trip(t: int, rem: list[int]) -> None:
        q_up = qz(silo_yaw)
        # wait for the balls to stop rolling, then plan the dip from LIVE positions
        # (a previous trip's retreat can leave a ball still rolling across the floor)
        for _ in range(8):
            if all(float(scene.balls[i].data.root_lin_vel_w[0].norm()) < 0.02
                   for i in rem):
                break
            step(30)
        min_x = min(ball_silo_x(i) for i in rem)
        ox = min_x - c.ball_r - 0.006 - c.nose_x
        ox = max(min(ox, 0.0), -c.in_x / 2 - c.back_x + 0.004)
        print(f"[solve] trip {t}: rem={rem} min_ball_x={min_x:+.3f} dip_x={ox:+.3f}",
              flush=True)

        def snap(tag: str) -> None:
            sp_l = scene._local(scene.silo, scene.scoop.data.root_pos_w)[0]
            q_e = qmul(q_up, qinv(scene.scoop.data.root_quat_w[0]))
            tilt_now = math.degrees(
                2.0 * math.atan2(float(q_e[1:].norm()), abs(float(q_e[0]))))
            parts = []
            for i in rem:
                bl = scene._local(scene.silo, scene.balls[i].data.root_pos_w)[0]
                bp = scene.balls[i].data.root_pos_w[0]
                sp = scene.scoop.data.root_pos_w[0]
                sq_ = scene.scoop.data.root_quat_w[0]
                lo = quat_apply_inverse(sq_.view(1, 4), (bp - sp).view(1, 3)).view(3)
                parts.append(f"b{i} silo=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
                             f"{float(bl[2]):+.3f}) scoop=({float(lo[0]):+.3f},"
                             f"{float(lo[1]):+.3f},{float(lo[2]):+.3f})")
            print(f"[solve]   t{t} {tag}: scoop_silo=({float(sp_l[0]):+.3f},"
                  f"{float(sp_l[1]):+.3f},{float(sp_l[2]):+.3f}) "
                  f"tilt={tilt_now:.1f}deg " + " | ".join(parts), flush=True)

        # T-a DIP: hover above the rim (free space), then descend through the mouth.
        hx, hy = sl2w(ox, 0.0)
        top = torch.tensor([hx, hy, c.rim_z + 0.05], device=device)
        teleport_scoop(top, q_up)
        hold(top, q_up, 30)
        bot = torch.tensor([hx, hy, z_floor], device=device)
        glide(top, bot, lambda fr: q_up, 0.25, 1.5)
        hold(bot, q_up, 40)
        snap("dip-bottom")

        # T-b SHOVEL: press down and stroke to the +x wall (balls ride up the ramp);
        # the final 25 mm approach runs at half speed so the wall-pin is quasi-static.
        mx, my = sl2w(x_end - 0.025, 0.0)
        mid = torch.tensor([mx, my, z_floor - 0.003], device=device)
        ex, ey = sl2w(x_end, 0.0)
        press = torch.tensor([ex, ey, z_floor - 0.003], device=device)
        glide(bot, mid, lambda fr: q_up, 0.06, 0.5, f_down=0.30)
        snap("stroke-mid")
        glide(mid, press, lambda fr: q_up, 0.03, 0.5, f_down=0.30)
        hold(press, q_up, 40, f_down=0.30)
        snap("stroke-end")

        # T-b2 PRESS: the carrot spring is soft (mass*KP = 22.5 N/m), so the stroke
        # stalls at ~0.4 N on the wall-pinned ball — far below the ~1-2 N the
        # ramp-edge/wall wedge needs to pop the 0.15 N ball up the lip. Ramp the x
        # carrot to 130 mm past the wall (~3 N quasi-static, F_MAX-clamped) while
        # pinning the scoop DOWN with 2.5 N (without the pin the 150 g scoop rides
        # up over the ball instead — observed). The nose closes to the wall, which
        # geometrically forces the ball fully onto the ramp (a wall-touching ball's
        # centre is 12.5 mm behind the tip).
        for j in range(300):
            fb = min(1.0, j / 120.0) * 0.130
            bx, by = sl2w(x_end + fb, 0.0)
            servo_step(torch.tensor([bx, by, z_floor - 0.003], device=device),
                       q_up, f_down=2.5, pivot_b=tip_b)
        snap("press")

        # T-c CAPTURE-TILT + EXTRACT: pitch the pan nose-up pivoting about its REAR
        # bottom edge (the only floor contact that can act as the pivot: any nose-up
        # rotation lifts the tip, so the z carrot must follow the HEEL kinematics —
        # origin rises by only (pan_len/2)*sin, keeping the heel pressed down; a
        # tip-anchored z carrot lifts the whole scoop and opens a floor gap the
        # ball escapes under (observed)). The x carrot stays biased past the wall
        # so the wall-pinned ball rides the rising ramp; the tip-wall gap grows by
        # just ~2+7 mm at 25 deg — far below the 25 mm ball diameter, so forward
        # escape stays blocked — and once the tilt passes the ramp's ~11 deg
        # incline the ball rolls back over the crest into the pan.
        # Then retreat from the wall, still tilted, and rise out of the mouth.
        tdeg = -25.0
        px_, py_ = sl2w(x_end + 0.060, 0.0)  # ~1.4 N sustained press during the tilt
        tsteps = 300
        for i in range(tsteps):
            fr = min(1.0, i / (tsteps * 0.8))
            ang = math.radians(tdeg) * fr  # negative = nose-up
            zc = z_floor - 0.003 + (c.pan_len / 2) * math.sin(-ang)
            servo_step(torch.tensor([px_, py_, zc], device=device),
                       qmul(q_up, qy(ang)), f_down=0.5, pivot_b=tip_b)
        snap("tilt-done")
        # At 25 deg the wedged ball has demonstrably climbed the tip (observed:
        # silo z 0.024 -> 0.040) but sits ON the crest, not behind it — a retreat
        # here flings it back to the floor (observed). DEEPEN the tilt in place to
        # 45 deg so the tip edge rotates under the wall-pinned ball and it rolls
        # backward over the crest into the pan. Critically, the far x bias must
        # NOT be kept during this: a ~1.2 N press braces the tip into the
        # wall/floor corner, the jammed scoop slides UP the wall (observed: z
        # +21 mm, pitch stalled at 31 deg) and the ball escapes under the lip.
        # Instead the x carrot tracks the wall GEOMETRICALLY — the origin
        # advances by exactly the tip retraction nose_x*(1-cos) plus a light
        # ~0.45 N bias — while f_down=2.5 pins the heel to the floor.
        tdeg2 = -45.0
        dsteps = 360
        for i in range(dsteps):
            fr = min(1.0, i / (dsteps * 0.8))
            ang = math.radians(tdeg + (tdeg2 - tdeg) * fr)  # negative = nose-up
            xc = x_end + c.nose_x * (1.0 - math.cos(ang)) + 0.020
            wx, wy = sl2w(xc, 0.0)
            zc = z_floor - 0.003 + (c.pan_len / 2) * math.sin(-ang)
            servo_step(torch.tensor([wx, wy, zc], device=device),
                       qmul(q_up, qy(ang)), f_down=2.5, pivot_b=tip_b)
        snap("tilt-deep")
        tdeg = tdeg2
        q_tilt = qmul(q_up, qy(math.radians(tdeg)))
        px_, py_ = sl2w(x_end + c.nose_x * (1.0 - math.cos(math.radians(tdeg)))
                        + 0.020, 0.0)
        tipz = z_floor - 0.003 + (c.pan_len / 2) * math.sin(math.radians(-tdeg))
        rx, ry = sl2w(x_end - 0.030, 0.0)
        lift0 = torch.tensor([rx, ry, tipz + 0.008], device=device)
        glide(torch.tensor([px_, py_, tipz], device=device), lift0,
              lambda fr: q_tilt, 0.04, 0.5)
        hold(lift0, q_tilt, 30)
        snap("retreat")
        got = [i for i in rem if in_pan(i)]
        print(f"[solve] trip {t}: captured={got} (of {rem})", flush=True)
        high = torch.tensor([rx, ry, c.rim_z + 0.11], device=device)
        glide(lift0, high, lambda fr: q_tilt, 0.22, 1.0)
        hold(high, q_tilt, 30)
        if not got:  # nothing riding in the pan: skip the trip's transfer legs
            print(f"[solve] trip {t}: empty pan after the stroke; retrying",
                  flush=True)
            return

        # T-d CARRY: eased glide to short of the tray centre, balls riding loose.
        drop_xy = basin_xy - u_nose * 0.08
        over = torch.tensor([float(drop_xy[0]), float(drop_xy[1]), c.rim_z + 0.11],
                            device=device)
        glide(high, over, lambda fr: q_tilt, 0.25, 0.5)
        hold(over, q_tilt, 40)
        low = torch.tensor([float(drop_xy[0]), float(drop_xy[1]), 0.16], device=device)
        glide(over, low, lambda fr: q_tilt, 0.20, 0.8)
        hold(low, q_tilt, 30)
        still = [i for i in got if in_pan(i)]
        if not still:  # everything fell out during the carry: retry the trip
            print(f"[solve] trip {t}: cargo lost in transit ({got}); retrying",
                  flush=True)
            park0 = torch.tensor([float(drop_xy[0]), float(drop_xy[1]), 0.35],
                                 device=device)
            glide(low, park0, lambda fr: q_up, 0.25, 1.0)
            return
        got = still

        # T-e DUMP: slow nose-down pour; balls roll out and fall into the tray.
        def poured() -> bool:
            return all(not in_pan(i) for i in got) if got else True

        cur_deg = tdeg
        for stage_deg, steps in ((35.0, 240), (55.0, 200)):
            start_deg = cur_deg
            for i in range(steps):
                fr = min(1.0, i / (steps * 0.75))
                cur_deg = start_deg + (stage_deg - start_deg) * fr
                servo_step(low, qmul(q_up, qy(math.radians(cur_deg))))
            hold(low, qmul(q_up, qy(math.radians(cur_deg))), 60)
            if poured():
                break
        if not poured():  # wiggle fallback: deeper tilt + small fore-aft shake
            for i in range(300):
                cur_deg = 60.0 + 8.0 * math.sin(i / 10.0)
                sh = 0.010 * math.sin(i / 7.0)
                pos = low + torch.tensor([float(u_nose[0]) * sh,
                                          float(u_nose[1]) * sh, 0.0], device=device)
                servo_step(pos, qmul(q_up, qy(math.radians(cur_deg))))
                if poured():
                    break
        step_in = [i for i in got if bool(scene.in_basin()[0, i])]
        print(f"[solve] trip {t}: poured -> in_basin now {step_in} (of {got})",
              flush=True)

        # level out and rise to a free hover (ready to teleport away)
        park = torch.tensor([float(drop_xy[0]), float(drop_xy[1]), 0.35], device=device)
        end_deg = cur_deg
        glide(low, park, lambda fr: qmul(q_up, qy(math.radians(end_deg * (1 - fr)))),
              0.25, 1.0)
        hold(park, q_up, 30)

    scores = [s1]
    for t in range(4):
        rem = [i for i in range(c.n_balls)
               if bool(scene.present[0, i]) and bool(scene.in_silo()[0, i])
               and not bool(scene.in_basin()[0, i])]
        if not rem:
            break
        do_trip(t, rem)
        report(f"trip{t}")
        s = print_score(f"P2.{t} trip {t} (dip+shovel+extract+carry+dump)")
        assert s >= scores[-1] - 1e-6, "score decreased across a trip"
        scores.append(s)

    delivered = bool((scene.in_basin() | ~scene.present).all(dim=1)[0])
    if not delivered:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (present balls not all delivered to the tray)",
              flush=True)
        os._exit(1)

    # ---------------- phase 3: STOW — park the empty scoop on open floor -------------------
    # The scoop hovers free above the tray; teleport it (transport, free space: 6 mm
    # above its lying rest height) to open floor clear of both fixtures, handle
    # pointing away from everything, and let it drop.
    q_stow = qmul(qz(0.0), qy(math.radians(-90.0)))
    stow = torch.tensor([-0.35, 0.10, -c.back_x + 0.010], device=device)
    teleport_scoop(stow, q_stow)
    step(180)
    report("stowed")
    s3 = print_score("P3 scoop stowed on open floor")
    assert s3 >= scores[-1] - 1e-6, "score decreased across the stow phase"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after stow)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, hands-off) -----------
    hold_ok = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold_ok = hold_ok and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = hold_ok and bool(scene.success()[0]) and s4 >= s3 - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
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
