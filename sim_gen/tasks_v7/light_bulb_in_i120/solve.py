"""Teleport solution for LampTurnstileScene (sim_gen task `light_bulb_in_i120`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY. Every load-bearing interaction goes through
contact/constraint dynamics:
  1. TURN OUT (dynamics): the turnstile is rotated half a turn to the service index by
     a rate-cascade torque servo about its own free revolute joint (the stand-in for
     pushing a red handle peg along its arc). Nothing is teleported.
  2. LOAD (transport + dynamics): the bulb is teleported from the tray to free space
     55 mm ABOVE the now-exposed cradle well — outside the cabinet, above an open
     well, pure transport across reachable air — then RELEASED. Gravity + contact
     with the brass walls and platter seat it; the seating is never spawned.
  3. TURN IN (dynamics): the same torque servo rotates the LOADED turnstile half a
     turn back to the lit index. The mechanism — not a teleport — carries the bulb
     through the window into the sealed cabinet.

Servo notes (external-wrench plant recipe): the physx flag
`enable_external_forces_every_iteration` is set in the scene; the hinge rate is
finite-differenced from yaw (root_ang_vel_w is phantom under external wrenches); the
carousel's diagonal inertia is authored (Iz = 0.0026), so the inner-loop gain is
discretely stable: kw*dt/Iz = 0.08/(120*0.0026) ~ 0.26 < 1. Release only when slow
(|w| < 0.05 rad/s -> damping-1.0 coast ~ 3 deg, inside both index tolerances); on
stall the GAIN escalates, not the torque cap.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.5 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.light_bulb_in_i120.solve --headless [--seed N]
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


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.lamp_turnstile")().build(num_envs=args.num_envs,
                                                    device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_rows = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def torque_z(tau: float) -> None:
        """World z torque on the carousel, encoded in its current body frame (the
        house convention: the engine treats the given wrench as body-frame). For a
        pure-z-rotated body the encoding is the identity, but do it anyway."""
        from isaaclab.utils.math import quat_apply_inverse

        t_w = torch.zeros(n, 3, device=device)
        t_w[:, 2] = tau
        scene.carousel.set_external_force_and_torque(
            zero_rows,
            quat_apply_inverse(scene.carousel.data.root_link_quat_w, t_w).unsqueeze(1),
            env_ids=all_ids)

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        loc = scene.bulb_local()[0]
        print(f"[solve] {tag:14s} | yaw={math.degrees(float(scene.yaw()[0])):+7.1f}deg"
              f" latches=({float(scene.out_latch[0]):.0f},"
              f"{float(scene.seat_latch[0]):.0f},{float(scene.carry_latch[0]):.0f})"
              f" bulb_loc=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
              f"{float(loc[2]):+.3f})"
              f" in_ring={bool(scene.in_ring()[0])}"
              f" inside={bool(scene.bulb_inside()[0])}"
              f" settled={bool(scene.settled()[0])}"
              f" success={bool(scene.success()[0])}"
              f" score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def rotate_to(theta_t: float, tag: str, rate_cap: float = 1.2) -> float:
        """Rate-cascade servo on the turnstile's free vertical DOF: outer loop
        commands a rate toward `theta_t`, inner loop torques toward that rate using
        the FINITE-DIFFERENCE yaw rate (root_ang_vel_w is phantom under external
        wrenches). Releases only near the target AND slow; damping parks the rest."""
        kth, kw, tau_cap = 3.0, 0.08, 0.25
        yaw_prev = float(scene.yaw()[0])
        last_check = abs(wrap(theta_t - yaw_prev))
        for i in range(2400):
            yaw = float(scene.yaw()[0])
            w_fd = wrap(yaw - yaw_prev) * 120.0
            yaw_prev = yaw
            err = wrap(theta_t - yaw)
            if abs(err) < 0.05 and abs(w_fd) < 0.05:
                break
            w_des = max(min(kth * err, rate_cap), -rate_cap)
            tau = max(min(kw * (w_des - w_fd), tau_cap), -tau_cap)
            torque_z(tau)
            env.step(no_action)
            if (i + 1) % 180 == 0:
                if last_check - abs(err) < 0.03:  # stalled: escalate GAIN, not cap
                    kw = min(kw * 1.5, 0.35)
                    print(f"[solve] rotate {tag}: stall at |err|="
                          f"{math.degrees(abs(err)):.1f}deg, kw->{kw:.3f}", flush=True)
                last_check = abs(err)
        torque_z(0.0)
        for _ in range(12):  # hands-off: damping coasts ~w/1.0 rad and parks it
            step(30)
            if float(scene.carousel.data.root_ang_vel_w[0].norm()) < 0.05:
                break
        final = abs(wrap(theta_t - float(scene.yaw()[0])))
        print(f"[solve] rotate {tag}: parked {math.degrees(final):.1f}deg off target",
              flush=True)
        return final

    def seat_bulb() -> None:
        """TRANSPORT the bulb across free space to 55 mm above the exposed cradle
        well (outside the cabinet at the service index), zero velocity, release —
        gravity and brass/platter contact make the seating."""
        from isaaclab.utils.math import quat_apply

        off = torch.tensor([c.ring_c, 0.0, c.plat_t / 2 + c.bulb_r + 0.055],
                           device=device).expand(n, 3)
        tgt = scene.carousel.data.root_pos_w + quat_apply(
            scene.carousel.data.root_quat_w, off)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = tgt
        st[:, 3] = 1.0
        scene.bulb.write_root_state_to_sim(st, all_ids)
        step(240)  # free fall ~55 mm + contact settling inside the well

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    print(f"[solve] layout readback (seed {args.seed}): "
          f"yaw={math.degrees(float(scene.yaw()[0])):+.1f}deg "
          f"bulb=({float(rel(scene.bulb)[0]):+.3f},{float(rel(scene.bulb)[1]):+.3f},"
          f"{float(rel(scene.bulb)[2]):.3f}) "
          f"tray=({float(rel(scene.tray)[0]):+.3f},{float(rel(scene.tray)[1]):+.3f})",
          flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"

    # ---------------- phase 1: turn the turnstile OUT (dynamics) ---------------------------
    miss = rotate_to(math.pi, "out")
    report("turned out")
    assert miss < math.radians(c.out_tol_deg) - 0.06, \
        f"turnstile not parked at service ({math.degrees(miss):.1f}deg off)"
    assert bool(scene.service_indexed()[0]), "service index not read back"
    s1 = print_score("P1 turnstile at service index (joint dynamics)")
    assert s1 >= s0 - 1e-6 and s1 >= 0.20 - 1e-6

    # ---------------- phase 2: load the cradle (transport + drop) --------------------------
    seat_bulb()
    report("bulb seated")
    assert bool(scene.in_ring()[0]), "bulb did not seat in the cradle well"
    s2 = print_score("P2 bulb dropped into the cradle (gravity + contact)")
    assert s2 >= s1 - 1e-6 and s2 >= 0.50 - 1e-6

    # ---------------- phase 3: turn the LOADED turnstile IN (dynamics) ---------------------
    miss = rotate_to(0.0, "in", rate_cap=0.9)
    report("turned in")
    assert miss < math.radians(c.lit_tol_deg) - 0.06, \
        f"turnstile not parked at lit ({math.degrees(miss):.1f}deg off)"
    s3 = print_score("P3 loaded cradle carried inside (joint dynamics)")
    assert s3 >= s2 - 1e-6

    # ---------------- phase 4: settle to success -------------------------------------------
    for _ in range(16):  # up to 4 s of hands-off settling
        if bool(scene.success()[0]):
            break
        step(30)
    report("settled")
    s4 = print_score("P4 all settled")
    assert s4 >= s3 - 1e-6, "score decreased across settling"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after out+load+in+settle)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3.5 simulated seconds, hands-off) -----------
    hold, flickers = True, 0
    for i in range(420):  # 420 substeps = 3.5 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                print(f"[solve] persist flicker @step {i}: "
                      f"in_ring={bool(scene.in_ring()[0])} "
                      f"lit={bool(scene.lit_indexed()[0])} "
                      f"settled={bool(scene.settled()[0])} "
                      f"blin={float(scene.bulb.data.root_lin_vel_w[0].norm()):.4f} "
                      f"bang={float(scene.bulb.data.root_ang_vel_w[0].norm()):.4f} "
                      f"cang={float(scene.carousel.data.root_ang_vel_w[0].norm()):.4f}",
                      flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/420 steps", flush=True)
    report("persist")
    s5 = print_score("P5 persistence 3.5 s")
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
    try:
        main()
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
