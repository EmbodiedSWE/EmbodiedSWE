"""Teleport solution for CarouselDispatchScene (sim_gen task `reach_and_drag_i111`) —
the task's legitimacy certificate.

There is NO transport teleport in this solve: both load-bearing interactions are
executed through contact dynamics, end to end.

1. ROTATION UNDER CONTACT (applied torque + bearing friction): the platter is turned
   with a small z-torque at its CoM — the same twist a finger-push on the rim or the
   yellow handle peg produces. A bang-bang velocity servo (|w| capped at 0.6 rad/s —
   GENTLE, far below the ~6 rad/s centrifugal-ejection limit) drives the signed bay->
   red-garage azimuth error to < 2.5 deg; dry bearing friction (deck resting on the
   well-wall tops) brakes and holds it there. The torque escalates on measured stall
   (start 0.8 N.m, cap 3.5) because the bearing's break-away friction is a physical
   unknown. The platter is NEVER teleported after reset.
2. EJECTION UNDER CONTACT (applied horizontal force + friction): the cargo cube is
   pushed at its CoM radially outward (the direction a fingertip push through the bay
   opening produces), velocity-regulated to <= 0.15 m/s, force capped at 2.2 N (below
   the m*g = 2.45 N tipping bound for a CoM-height push on a 90 mm cube). It slides
   out of the bay, over the deck edge, across the 10 mm gap, through the red mouth,
   and onto the garage floor. The force-frame mode (some pods rotate applied wrenches
   by the body's rotation since reset) is PROBED from measured progress and toggled
   if the cube moves away — never assumed.
3. ORDER: align FIRST, then eject — the task's forced order (ejecting early throws
   the cube at a wall, a blank plinth face, or the open floor; the covered garages
   accept the cube only through their mouths at deck height).
4. The GREEN and BLUE garages are never touched; the red garage is chosen by reading
   back the per-episode color permutation through the scene's own red-dock frame.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
partial credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.reach_and_drag_i111.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401  (registers the scene)
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cargo_carousel")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

    def encode_force(mode: int, q_ref: torch.Tensor, q_now: torch.Tensor,
                     f_world: torch.Tensor) -> torch.Tensor:
        """Pre-encode a desired WORLD force. mode 0: pass through. mode 1: premultiply
        by R_ref * R_now^T for pods that rotate wrenches by rotation-since-reset."""
        if mode == 0:
            return f_world
        return quat_apply(quat_mul(q_ref, quat_conjugate(q_now)), f_world)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_forces() -> None:
        scene.platter.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                    env_ids=all_ids)
        scene.cargo.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                  env_ids=all_ids)

    def err_deg() -> float:
        return math.degrees(float(scene.align_err_rad()[0]))

    def w_z() -> float:
        return float(scene.platter.data.root_ang_vel_w[0, 2])

    def cargo_red_local() -> torch.Tensor:
        return scene._dock_local("red", scene.cargo.data.root_pos_w)[0]

    def report(tag: str) -> None:
        cl = cargo_red_local()
        print(f"[solve] {tag:14s} | err={err_deg():+7.2f}deg w_z={w_z():+6.3f} "
              f"cargo_red_loc=({float(cl[0]):+.3f},{float(cl[1]):+.3f},"
              f"{float(cl[2]):+.3f}) in_bay={bool(scene.cargo_in_bay()[0])} "
              f"aligned={bool(scene._aligned[0])} delivered={bool(scene._delivered[0])} "
              f"resting={bool(scene.resting_in_red()[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(180)  # platter seats on its bearing, cargo seats in the bay
    bp = (scene.pedestal.data.root_pos_w - scene.env_origins)[0]
    bq = scene.pedestal.data.root_quat_w[0]
    byaw = math.degrees(2.0 * math.atan2(float(bq[3]), float(bq[0])))
    pz = float((scene.platter.data.root_pos_w - scene.env_origins)[0, 2])
    pxy_off = float((scene.platter.data.root_pos_w[0, :2]
                     - scene.pedestal.data.root_pos_w[0, :2]).norm())
    rslot = int(scene.red_slot[0])
    print(f"[solve] layout readback (seed {args.seed}): "
          f"base=({float(bp[0]):+.3f},{float(bp[1]):+.3f}) yaw={byaw:+.1f}deg "
          f"red_slot={rslot} (az {c.slot_az_deg[rslot]:.0f}deg base-local) "
          f"platter_z={pz:.4f} centre_off={pxy_off * 1000:.1f}mm "
          f"start_err={err_deg():+.1f}deg", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert abs(pz - c.z_platter) < 0.006, f"platter must rest on its bearing, z={pz:.4f}"
    assert pxy_off < 0.012, f"platter must be centred by the well, off={pxy_off:.4f}"
    assert bool(scene.cargo_in_bay()[0]), "cargo must start in the bay"
    assert abs(err_deg()) > c.start_err_min_deg - 8.0, \
        f"start error {err_deg():.1f} deg should be >= ~{c.start_err_min_deg} deg"
    s0 = print_score("P0 reset+settle (cargo aboard, bay away from the red garage)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # Force-frame reference orientation for the cargo push (readback at reset).
    q_cargo_ref = scene.cargo.data.root_quat_w.clone()

    # ---------------- phase 1: rotate the platter under contact ----------------------------
    # Bang-bang velocity servo: torque about z (yaw frame-drag cannot touch a z-torque),
    # |w| capped GENTLE, torque escalated on measured stall (bearing break-away unknown).
    tau, stall, aligned_done = 0.8, 0, False
    for i in range(3600):
        e = float(scene.align_err_rad()[0])
        w = w_z()
        if abs(e) < math.radians(2.5) and abs(w) < 0.08:
            aligned_done = True
            break
        if not bool(scene.cargo_in_bay()[0]):
            clear_forces()
            report("FAIL-state")
            print("SIM_GEN_SOLVE: FAIL (cargo left the bay during rotation)", flush=True)
            os._exit(1)
        w_des = 0.0 if abs(e) < math.radians(2.5) else max(-0.6, min(0.6, 1.8 * e))
        dw = w_des - w
        t = torch.zeros(n, 1, 3, device=device)
        if abs(dw) > 0.04:
            t[:, 0, 2] = tau if dw > 0 else -tau
        scene.platter.set_external_force_and_torque(zero_wrench, t, env_ids=all_ids,
                                                    is_global=True)
        env.step(no_action)
        if abs(w) < 0.02 and abs(e) > math.radians(2.5):
            stall += 1
            if stall >= 120:
                tau = min(tau + 0.5, 3.5)
                stall = 0
                print(f"[solve] platter stalled at err={math.degrees(e):+.1f}deg; "
                      f"torque -> {tau:.1f} N.m", flush=True)
        else:
            stall = 0
        if i % 300 == 299:
            print(f"[solve] rotating: err={math.degrees(e):+.1f}deg w={w:+.3f} "
                  f"tau={tau:.1f}", flush=True)
    clear_forces()
    step(150)  # friction brake + full settle, hands-off
    report("aligned")
    if not aligned_done:
        print("SIM_GEN_SOLVE: FAIL (rotation servo timed out)", flush=True)
        os._exit(1)
    assert abs(err_deg()) < c.align_tol_deg, f"final align err {err_deg():.2f} deg"
    assert bool(scene.cargo_in_bay()[0]), "cargo must still be aboard after rotation"
    assert bool(scene._aligned[0]), "aligned latch must have fired"
    assert not bool(scene.success()[0]), "cannot be success with the cargo aboard"
    s1 = print_score("P1 bay aligned with the red garage (rotation under contact)")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_aligned - 1e-6, \
        f"P1 score {s1} (expect aligned={c.w_aligned})"

    # ---------------- phase 2: eject the cargo radially under contact ----------------------
    # Push direction: horizontal platter-axis -> red-garage ray (the docks are kinematic;
    # computed once). Velocity-regulated CoM push, force capped below the tipping bound.
    u = (scene.docks["red"].data.root_pos_w[0, :2]
         - scene.platter.data.root_pos_w[0, :2])
    u = u / u.norm().clamp(min=1e-9)
    mode = 0  # force-frame mode, probed from measured progress
    floor_f = 1.4  # stiction floor (~mu*m*g = 1.2 N for the 0.25 kg cube)
    win_i, win_x = 0, float(cargo_red_local()[0])
    ejected = False
    for i in range(3600):
        x_loc = float(cargo_red_local()[0])
        if x_loc > 0.0:  # centre 45 mm past the full-inside plane, 25 mm short of the back
            ejected = True
            break
        v = scene.cargo.data.root_lin_vel_w[0, :2]
        v_des = u * (0.15 if x_loc < -0.10 else 0.07)
        f_xy = 6.0 * (v_des - v)
        f_along = float((f_xy * u).sum())
        if float(v.norm()) < 0.02 and f_along < floor_f:
            f_xy = f_xy + u * (floor_f - f_along)  # break stiction
        fn = float(f_xy.norm())
        if fn > 2.2:
            f_xy = f_xy * (2.2 / fn)  # stay below the m*g tipping bound
        f_world = torch.zeros(n, 3, device=device)
        f_world[0, :2] = f_xy
        f_arg = encode_force(mode, q_cargo_ref, scene.cargo.data.root_quat_w, f_world)
        scene.cargo.set_external_force_and_torque(f_arg.view(n, 1, 3), zero_wrench,
                                                  env_ids=all_ids, is_global=True)
        env.step(no_action)
        # progress probe every 45 steps: wrong force-frame mode moves the cube AWAY
        if i - win_i >= 45:
            if x_loc < win_x - 0.008:
                mode = 1 - mode
                print(f"[solve] eject: moving away (x_loc {win_x:+.3f} -> {x_loc:+.3f}); "
                      f"force-frame mode -> {mode}", flush=True)
            elif x_loc < win_x + 0.004:
                floor_f = min(floor_f + 0.3, 2.1)
                print(f"[solve] eject: stalled at x_loc={x_loc:+.3f}; "
                      f"stiction floor -> {floor_f:.1f} N", flush=True)
            win_i, win_x = i, x_loc
    clear_forces()
    step(180)  # slide-out + full settle, hands-off
    report("ejected")
    if not ejected:
        print("SIM_GEN_SOLVE: FAIL (ejection push never got the cube inside)", flush=True)
        os._exit(1)
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the cube entered the garage)",
              flush=True)
        os._exit(1)
    s2 = print_score("P2 cargo pushed through the red mouth, at rest inside")
    assert s2 >= s1 - 1e-6 and s2 >= 0.999, f"P2 score {s2} (expect success=1.0)"

    # ---------------- phase 3: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s3 >= s2 - 1e-6
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
