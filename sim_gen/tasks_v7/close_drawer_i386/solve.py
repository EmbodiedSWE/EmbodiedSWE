"""Teleport solution for BayonetDrawerScene (sim_gen task `close_drawer_i386`) — the
task's legitimacy certificate.

This solve uses NO teleports at all: every interaction is a velocity-regulated
external wrench (re-set every step, zeroed before judging), exactly like a hand on
the red T-handle. Gains respect the one-substep wrench delay (KV*dt/m ~= 0.17,
KD*dt/I ~= 0.17, both << 1). Forces/torques are rotated into the BODY frame every
step (the external-force frame-drag trap); the rotor torque acts about the body x
axis (== the revolute axis).

PLAN (read-only, from scene.describe()): the pitched cabinet glides its drawer open
on its own; only the bayonet holds it shut:
  P0 settle + readback: drawer has glided fully open, rotor spawns misaligned;
     score ~0,
  P1 SQUARE the rotor (torque servo -> 0 deg) so the lug bar can pass the bezel slot,
  P2 PRESS the drawer to the closed stop by force (gravity-along-rail feedforward +
     velocity servo), holding the rotor square during the slot transit,
  P3 TWIST the rotor to ~92 deg while the press holds the drawer seated (the bar
     turns vertical BEHIND the plate); locked latch fires,
  P4 RELEASE everything: the drawer creeps ~7 mm until the bar catches the plate's
     back face and rests there — the dead-man moment; ring down to success,
  P5 hands-off persistence >= 3.3 simulated seconds.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then `SIM_GEN_SOLVE: SUCCESS` only if success() still holds
after the hands-off hold.

Run (forge): python -u -m simgen_tasks.close_drawer_i386.solve --headless [--seed N]
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

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:  # older isaaclab name
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

G = 9.81
# Rotor twist servo (I_xx = 2e-4: KD*dt/I = 0.004/(120*2e-4) ~= 0.17).
R_KP = 0.020    # position loop (N m / rad)
R_KD = 0.004    # rate loop (N m s / rad)
R_TMAX = 0.05   # torque clamp (N m)
R_RAMP = 1.5    # theta_ref ramp rate (rad/s)
# Drawer press servo (mass 1.5: KV*dt/m = 30/(120*1.5) ~= 0.17).
D_KP = 3.0      # outer position loop -> desired velocity (1/s)
D_VCAP = 0.12   # desired-velocity cap (m/s)
D_KV = 30.0     # velocity loop -> force (N s/m)
D_FMIN, D_FMAX = -12.0, 5.0


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.twistlock_drawer_i386")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)
    lock_ref = math.radians(92.0)
    lock_min = math.radians(c.lock_min_deg)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def q0() -> float:
        return float(scene.drawer_q()[0])

    def th0() -> float:
        return float(scene.rotor_theta()[0])

    def report(tag: str) -> None:
        print(f"[solve] {tag:12s} | q={q0():+.4f} theta={math.degrees(th0()):+.1f}deg "
              f"| latches c/p/l={float(scene.closure_latch[0]):.2f}/"
              f"{float(scene.pressed_latch[0]):.0f}/{float(scene.locked_latch[0]):.0f} "
              f"| settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def wrenches_off() -> None:
        scene.drawer.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
        scene.rotor.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)

    def rotor_torque(theta_ref: torch.Tensor) -> None:
        """One step of the rotor twist servo (torque about the BODY x axis == the
        revolute axis; the rotor is balanced, so no gravity feedforward)."""
        th = scene.rotor_theta()
        w_body = quat_apply_inverse(scene.rotor.data.root_quat_w,
                                    scene.rotor.data.root_ang_vel_w)
        tau = (R_KP * (theta_ref - th) - R_KD * w_body[:, 0]).clamp(-R_TMAX, R_TMAX)
        t = torch.zeros(n, 1, 3, device=device)
        t[:, 0, 0] = tau
        scene.rotor.set_external_force_and_torque(zero_w, t, env_ids=all_ids)

    def drawer_press(q_ref: float) -> None:
        """One step of the drawer press servo: body-frame x force with a
        feedforward cancelling the downhill gravity component."""
        q = scene.drawer_q()
        v_body = quat_apply_inverse(scene.drawer.data.root_quat_w,
                                    scene.drawer.data.root_lin_vel_w)
        v_des = (D_KP * (q_ref - q)).clamp(-D_VCAP, D_VCAP)
        fx = (-c.drawer_mass * G * math.sin(c.pitch)
              + D_KV * (v_des - v_body[:, 0])).clamp(D_FMIN, D_FMAX)
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 0] = fx
        scene.drawer.set_external_force_and_torque(f, zero_w, env_ids=all_ids)

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(120)  # the drawer glides fully open on its own (dead-man demo)
    masses = scene.drawer.root_physx_view.get_masses().flatten()
    assert abs(float(masses[0]) - c.drawer_mass) <= 0.2 * c.drawer_mass, \
        f"authored drawer mass missing (got {float(masses[0]):.3f})"
    p_st = (scene.station.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] readback (seed {args.seed}): station=({float(p_st[0]):+.3f},"
          f"{float(p_st[1]):+.3f},{float(p_st[2]):+.3f}) q={q0():+.4f} "
          f"(open stop {c.stroke:.3f}) theta0={math.degrees(th0()):+.1f}deg "
          f"(pass angle {c.pass_deg:.1f}deg) q_rest={c.q_rest:.4f} "
          f"q_trap={c.q_trap:.4f}", flush=True)
    assert q0() >= c.stroke - 0.010, "drawer must have glided fully open on its own"
    assert abs(th0()) >= math.radians(3.0), "rotor must spawn visibly misaligned"
    report("reset")
    s0 = print_score("P0 reset+settle (drawer glided open)")
    assert s0 < 0.02, "score must start ~0"

    # ---------------- phase 1: SQUARE the rotor --------------------------------------------
    done = 0
    for i in range(600):
        rotor_torque(torch.zeros(n, device=device))
        env.step(no_action)
        w = float(scene.rotor.data.root_ang_vel_w[0].norm())
        done = done + 1 if (abs(th0()) < math.radians(0.8) and w < 0.2) else 0
        if done >= 20:
            break
    print(f"[solve] P1: rotor squared to {math.degrees(th0()):+.2f}deg "
          f"in {i + 1} servo steps", flush=True)
    assert abs(th0()) < math.radians(2.0), "rotor must be square before the press"
    s1 = print_score("P1 rotor squared")
    assert s1 >= s0 - 1e-6, "score decreased across P1"

    # ---------------- phase 2: PRESS the drawer to the closed stop -------------------------
    done = 0
    for i in range(1200):
        drawer_press(0.0005)
        rotor_torque(torch.zeros(n, device=device))  # hold square through the slot
        env.step(no_action)
        v = float(scene.drawer.data.root_lin_vel_w[0].norm())
        done = done + 1 if (q0() <= 0.003 and v < 0.03) else 0
        if done >= 20:
            break
        if (i + 1) % 300 == 0:
            print(f"[solve] P2 telemetry @{i + 1}: q={q0():+.4f} "
                  f"theta={math.degrees(th0()):+.2f}deg v={v:+.4f}", flush=True)
    print(f"[solve] P2: drawer pressed to q={q0():+.4f} in {i + 1} servo steps "
          f"(theta {math.degrees(th0()):+.2f}deg)", flush=True)
    assert q0() <= c.closed_tol - 0.005, "drawer must be seated at the closed stop"
    assert float(scene.pressed_latch[0]) > 0.5, "pressed latch must have fired"
    s2 = print_score("P2 drawer pressed home (bar behind the plate)")
    assert s2 >= max(s1, 0.29), "closure+pressed credit missing"

    # ---------------- phase 3: TWIST to lock while the press holds -------------------------
    theta_ref = torch.full((n,), th0(), device=device)
    done = 0
    for i in range(900):
        theta_ref = (theta_ref + R_RAMP / 120.0).clamp(max=lock_ref)
        drawer_press(0.0005)
        rotor_torque(theta_ref)
        env.step(no_action)
        w = float(scene.rotor.data.root_ang_vel_w[0].norm())
        done = done + 1 if (th0() >= lock_min + math.radians(10.0) and w < 0.3) else 0
        if done >= 20:
            break
        if (i + 1) % 300 == 0:
            print(f"[solve] P3 telemetry @{i + 1}: theta={math.degrees(th0()):+.1f}deg "
                  f"q={q0():+.4f} w={w:+.3f}", flush=True)
    print(f"[solve] P3: rotor locked at {math.degrees(th0()):+.1f}deg "
          f"in {i + 1} servo steps (q={q0():+.4f})", flush=True)
    assert th0() >= lock_min + math.radians(5.0), "rotor must be twisted well past lock_min"
    assert float(scene.locked_latch[0]) > 0.5, "locked latch must have fired"
    s3 = print_score("P3 bayonet twisted home")
    assert s3 >= max(s2, 0.59), "locked credit missing"

    # ---------------- phase 4: RELEASE — the dead-man moment -------------------------------
    wrenches_off()
    step(120)  # the drawer creeps out until the bar catches the plate's back face
    print(f"[solve] P4 released: q={q0():+.4f} (locked rest {c.q_rest:+.4f}), "
          f"theta={math.degrees(th0()):+.1f}deg — wrenches OFF", flush=True)
    assert q0() <= c.closed_tol, "released drawer must be caught by the bayonet"
    # Ring-down: wait for success() to hold CONTINUOUSLY for 1 s.
    consec = 0
    for _ in range(2400):  # up to 20 s
        step(1)
        if bool(scene.success()[0]):
            consec += 1
            if consec >= 120:
                break
        else:
            consec = 0
    report("P4-ringdown")
    s4 = print_score("P4 released — bayonet holds, at rest")
    assert s4 >= s3 - 1e-6, "score decreased across P4"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after release + ring-down)", flush=True)
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3.3 simulated seconds, hands off) -----------
    hold, flickers = True, 0
    for i in range(400):  # 400 steps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                dv = float(scene.drawer.data.root_lin_vel_w[0].norm())
                rw = float(scene.rotor.data.root_ang_vel_w[0].norm())
                print(f"[solve] persist flicker @step {i}: q={q0():+.4f} "
                      f"theta={math.degrees(th0()):+.1f}deg drawer_v={dv:.4f} "
                      f"rotor_w={rw:.4f} settled={bool(scene.settled()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
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
    try:
        main()
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
