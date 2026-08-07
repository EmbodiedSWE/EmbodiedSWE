"""Solution for BendGalleryScene (sim_gen task `pick_up_cup_i76`) — the task's
legitimacy certificate.

This solve uses NO teleports at all: the entire trajectory — extraction from the
dead-end tunnel, the corner rotation in the plaza, the threading of the exit tunnel,
and the final placement on the pad — is driven by applied forces and a z-torque at the
dipper's centre of mass (the wrench of a fingertip pushing on the handle / cup wall),
against real wall, roof and floor contact. Nothing is ever lifted: the roofs make
lifting impossible inside, and the finish is a floor slide onto the flush pad marking.

Controller: a waypoint servo in the GALLERY frame. Waypoints come from the offline
config-space BFS that validated the geometry (rod footprint inflated 16 mm still
passes the corner). Per step: desired planar velocity toward the waypoint and desired
yaw rate toward the waypoint heading -> force = kv*(v_des - v) with a stiction floor
and a hard cap (3.5 N ~ a light fingertip push), torque_z = kw*(w_des - w) with its
own stiction floor and cap. Stall watch bumps the stiction floors and, if needed,
backs off toward the previous waypoint and retries. Pod force-frame quirk: some pods
rotate an applied wrench by the body's rotation since its reference orientation — the
rod YAWS ~90 deg in this task, so the mode is PROBED from measured displacement at the
start and the force is pre-encoded per `encode_force` every step ((0,0,tau) is
yaw-invariant and needs no encoding).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched): P0 settle, P1 tail cleared (latch, mid-turn), P2 cup entered the
exit tunnel, P3 cup emerged in the yard, P4 full extraction + cup on the pad, then
holds HANDS-OFF for >= 3 simulated seconds and prints `SIM_GEN_SOLVE: SUCCESS` only
if success() still holds.

Run (forge): python -u -m simgen_tasks.pick_up_cup_i76.solve --headless [--seed N]
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
    from .scene import _qapply, _qinv, encode_force
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import _qapply, _qinv, encode_force

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bend_gallery")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_forces() -> None:
        scene.dipper.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                   env_ids=all_ids)

    # ----- gallery-frame readouts (env 0) ---------------------------------------------------
    def gal_q() -> torch.Tensor:
        return scene.gallery.data.root_quat_w

    def center_local() -> torch.Tensor:
        return scene._rod_pts_local()[0, 2]

    def theta_local() -> float:
        h = scene.heading_local()[0]
        return math.atan2(float(h[1]), float(h[0]))

    def vel_local():
        v = _qapply(_qinv(gal_q()), scene.dipper.data.root_lin_vel_w)[0]
        return v

    def report(tag: str) -> None:
        cl = center_local()
        pts = scene._rod_pts_local()[0]
        print(f"[solve] {tag:12s} | c=({float(cl[0]):+.3f},{float(cl[1]):+.3f}) "
              f"th={math.degrees(theta_local()):+6.1f} "
              f"tail=({float(pts[0, 0]):+.3f},{float(pts[0, 1]):+.3f}) "
              f"cup=({float(pts[4, 0]):+.3f},{float(pts[4, 1]):+.3f}) "
              f"up_z={float(scene.rod_up_z()[0]):+.3f} "
              f"clr={bool(scene._cleared[0])} ent={bool(scene._entered[0])} "
              f"emg={bool(scene._emerged[0])} ext={bool(scene.extracted()[0])} "
              f"pad={bool(scene.cup_on_pad()[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    prev_score = [0.0]

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        assert s >= prev_score[0] - 1e-6, f"score decreased {prev_score[0]} -> {s}"
        prev_score[0] = s
        return s

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(120)
    gp = (scene.gallery.data.root_pos_w - scene.env_origins)[0]
    gh = scene.heading_local()[0]
    pts0 = scene._rod_pts_local()[0]
    pad_l = scene._gal_local(scene.pad.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"gallery=({float(gp[0]):+.3f},{float(gp[1]):+.3f}) "
          f"tail_x={float(pts0[0, 0]):+.3f} (depth in tunnel1) "
          f"rod_th={math.degrees(theta_local()):+.1f} deg "
          f"pad_local=({float(pad_l[0]):+.3f},{float(pad_l[1]):+.3f})", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert float(pts0[0, 0]) < -0.14, "handle tail must start deep in tunnel1"
    assert abs(math.degrees(theta_local())) < 6.0, "rod must start along the tunnel axis"
    assert float(pts0[4, 0]) > 0.01, "cup head must start exposed in the plaza"
    assert not bool(scene.extracted()[0]), "nothing may start extracted"
    s0 = print_score("P0 reset+settle (dipper racked in the dead-end tunnel)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # Force-frame reference orientation for `encode_force` (readback at reset).
    # At reset the body has not rotated, so both modes coincide and no probe can
    # distinguish them YET; the divergence watch inside the servo is the real probe —
    # it flips the mode the moment the applied force visibly disagrees with progress
    # (which can only happen once the rod has yawed). Default to mode 1: this pod
    # family has been measured to rotate applied wrenches by the body's rotation
    # since its reference orientation.
    q_ref = scene.dipper.data.root_quat_w.clone()
    mode = [1]

    def apply_wrench(f_local_xy, tau_z: float) -> None:
        """f_local_xy: desired planar force in the GALLERY frame."""
        f3 = torch.zeros(n, 3, device=device)
        f3[0, 0], f3[0, 1] = float(f_local_xy[0]), float(f_local_xy[1])
        f_world = _qapply(gal_q(), f3)
        f_arg = encode_force(mode[0], q_ref, scene.dipper.data.root_quat_w, f_world)
        t3 = torch.zeros(n, 1, 3, device=device)
        t3[0, 0, 2] = float(tau_z)
        scene.dipper.set_external_force_and_torque(
            f_arg.view(n, 1, 3), t3, env_ids=all_ids, is_global=True)

    # ---------------- stiction sanity check (gentle +x_local push) --------------------------
    # Confirms the wrench pathway moves the rod at all before the gauntlet (frame
    # mode is settled later, by the in-flight divergence watch). Stops at 4 mm so the
    # cup never picks up speed toward the far plaza wall.
    x_before = float(center_local()[0])
    dx = 0.0
    for probe_f in (2.0, 2.8):
        for _ in range(90):
            apply_wrench((probe_f, 0.0), 0.0)
            env.step(no_action)
            dx = float(center_local()[0]) - x_before
            if abs(dx) >= 0.004:
                break
        clear_forces()
        step(60)
        dx = float(center_local()[0]) - x_before
        if abs(dx) >= 0.003:
            break
    print(f"[solve] stiction check: dx={dx:+.4f} m (mode {mode[0]})", flush=True)
    assert abs(dx) >= 0.003, "probe push produced no measurable motion"

    # ---------------- the gauntlet: waypoint servo through the gallery ----------------------
    # (cx, cy, theta_deg, pos_tol, ang_tol_deg) in the gallery frame; from the
    # config-space BFS reference path (corner pass validated at >= 16 mm clearance).
    pad_xy = (float(pad_l[0]), float(pad_l[1]))
    gauntlet = [
        (0.02, 0.170, 0.0, 0.020, 14.0),      # advance out of the dead end
        (0.060, 0.155, -20.0, 0.022, 14.0),   # begin the corner rotation
        (0.080, 0.115, -33.0, 0.022, 12.0),
        (0.095, 0.085, -45.0, 0.022, 12.0),
        (0.110, 0.070, -56.0, 0.022, 12.0),
        (0.125, 0.055, -66.0, 0.022, 12.0),   # tail clears tunnel1 around here
        (0.150, 0.045, -78.0, 0.022, 12.0),
        (0.168, 0.010, -88.0, 0.018, 8.0),    # aligned with the exit tunnel
        (0.170, -0.070, -90.0, 0.020, 8.0),   # threading (cup enters tunnel2)
        (0.170, -0.190, -90.0, 0.020, 8.0),   # cup emerges into the yard
        (0.170, -0.310, -90.0, 0.020, 10.0),
        (0.170, -0.380, -90.0, 0.018, 10.0),  # tail clears the exit: fully extracted
        (0.170, -0.380, 0.0, 0.025, 10.0),    # yard turn (free space, stays clear)
        (pad_xy[0] - (c.rod_len / 2 - c.cup_r), pad_xy[1], 0.0, 0.010, 10.0),  # cup -> pad
    ]

    def wrap(a: float) -> float:
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    latch_names = ["cleared", "entered", "emerged"]
    latch_printed = {k: False for k in latch_names}

    def check_latch_prints() -> None:
        if not latch_printed["cleared"] and bool(scene._cleared[0]):
            latch_printed["cleared"] = True
            report("cleared")
            print_score("P1 handle tail cleared the dead-end tunnel (mid-turn)")
        if not latch_printed["entered"] and bool(scene._entered[0]):
            latch_printed["entered"] = True
            report("entered")
            print_score("P2 cup entered the exit tunnel (corner rotation done)")
        if not latch_printed["emerged"] and bool(scene._emerged[0]):
            latch_printed["emerged"] = True
            report("emerged")
            print_score("P3 cup emerged into the yard")

    def drive_to(wp, *, budget: int, tag: str, final: bool = False) -> bool:
        """Servo to one waypoint. Returns True when position AND heading converge."""
        tx, ty, tdeg, ptol, atol = wp
        tth = math.radians(tdeg)
        f_floor, t_floor = 1.8, 0.14
        win_i = 0
        win_err = 1e9
        for i in range(budget):
            cl = center_local()
            dx, dy = tx - float(cl[0]), ty - float(cl[1])
            dist = math.hypot(dx, dy)
            e_ang = wrap(tth - theta_local())
            if final:
                d_pad = float((scene.cup_pos_w()[0, :2]
                               - scene.pad.data.root_pos_w[0, :2]).norm())
                if d_pad < 0.015 and abs(e_ang) < math.radians(atol):
                    clear_forces()
                    return True
            elif dist < ptol and abs(e_ang) < math.radians(atol):
                clear_forces()
                return True
            # tilt guard: never fight a rocking rod
            if float(scene.rod_up_z()[0]) < 0.95:
                clear_forces()
                step(30)
                continue
            # velocity servo (gallery frame)
            ux, uy = (dx / dist, dy / dist) if dist > 1e-6 else (0.0, 0.0)
            sp = min(0.10, 2.0 * dist)
            v = vel_local()
            fx = 8.0 * (sp * ux - float(v[0]))
            fy = 8.0 * (sp * uy - float(v[1]))
            f_along = fx * ux + fy * uy
            if math.hypot(float(v[0]), float(v[1])) < 0.02 and dist > ptol \
                    and f_along < f_floor:
                fx += ux * (f_floor - f_along)
                fy += uy * (f_floor - f_along)
            fn = math.hypot(fx, fy)
            if fn > 3.5:
                fx, fy = fx * 3.5 / fn, fy * 3.5 / fn
            # yaw servo
            w_des = max(-0.9, min(0.9, 3.0 * e_ang))
            w_z = float(scene.dipper.data.root_ang_vel_w[0, 2])
            tau = 0.6 * (w_des - w_z)
            if abs(w_z) < 0.06 and abs(e_ang) > math.radians(5.0) and abs(tau) < t_floor:
                tau = math.copysign(t_floor, e_ang)
            tau = max(-0.30, min(0.30, tau))
            apply_wrench((fx, fy), tau)
            env.step(no_action)
            check_latch_prints()
            # stall watch every 90 steps
            err = dist + 0.15 * abs(e_ang)
            if i - win_i >= 90:
                if err > win_err + 0.02:
                    mode[0] = 1 - mode[0]
                    print(f"[solve] {tag}: diverging (err {win_err:.3f} -> {err:.3f}); "
                          f"force-frame mode -> {mode[0]}", flush=True)
                elif err > win_err - 0.006:
                    f_floor = min(f_floor + 0.3, 3.0)
                    t_floor = min(t_floor + 0.03, 0.24)
                    print(f"[solve] {tag}: stalled (err {err:.3f}); floors -> "
                          f"{f_floor:.1f} N / {t_floor:.2f} Nm", flush=True)
                win_i, win_err = i, err
        clear_forces()
        print(f"[solve] {tag}: budget exhausted (err {err:.3f})", flush=True)
        return False

    ok = True
    for j, wp in enumerate(gauntlet):
        final = j == len(gauntlet) - 1
        ok = drive_to(wp, budget=1400, tag=f"wp{j}", final=final)
        if not ok and j > 0:
            # one backoff-and-retry: ease toward the previous waypoint, then re-run
            print(f"[solve] wp{j}: backoff and retry", flush=True)
            drive_to(gauntlet[j - 1], budget=350, tag=f"wp{j}-backoff")
            ok = drive_to(wp, budget=1400, tag=f"wp{j}-retry", final=final)
        if not ok:
            break
        report(f"wp{j}")
    if not ok:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (gauntlet did not converge)", flush=True)
        os._exit(1)

    # ---------------- phase 4: hands-off settle on the pad ----------------------------------
    clear_forces()
    step(180)
    report("placed")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (final state is not success)", flush=True)
        os._exit(1)
    s4 = print_score("P4 dipper fully extracted, cup upright on the pad, settled")
    assert s4 >= 1.0 - 1e-6, f"success must score 1.0, got {s4}"

    # ---------------- phase 5: persistence (>= 3 simulated seconds, hands-off) --------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s5 >= s4 - 1e-6
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
