"""Teleport solution for CamGateCabinetScene (sim_gen task
`libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_i98`) — the task's
legitimacy certificate.

This solve uses ZERO teleports after reset: every judged outcome is produced by
contact dynamics under one applied wrench. The single actuation is a torque-servo
on the GATE about the world vertical (what a hand pushing the gate's panel or its
yellow handle through the arc does, torque-limited to 1.2 N*m):

1. PERCEPTION: station pose/yaw, the drawer's initial opening q0 and the gate's
   initial angle theta0 are read back from the episode state — never hard-coded.
   The gate angle at which the cam FIRST TOUCHES the roller is computed by
   inverting the scene's own cam relation at the MEASURED opening.
2. APPROACH (applied torque): the servo swings the gate from theta0 down to just
   above the touch angle. The drawer has not moved — only gate credit is earned.
3. CAM DRIVE (applied torque): the servo keeps swinging through touch to a target
   just past flush. The gate's inner face presses the red roller and the CAM —
   sliding contact on the slick material — converts the swing into the drawer's
   closing translation. The judged "drawer closes" outcome is delivered by the
   machine's transmission, never by a wrench on the drawer (the drawer is never
   touched by the solver at all).
4. RELEASE: the wrench is dropped; the gate rests flush on its stop post pressing
   the drawer seated; success() is judged LIVE on the settled state.

A pure z-torque is frame-encoding invariant for a yaw-only body (every known
set_external_force_and_torque mode maps (0,0,tau) to itself), so no runtime
force-frame probe is needed.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the rubric
latches), then holds HANDS-OFF >= 3 simulated seconds after success() first turns
True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_i98.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cam_gate_cabinet")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    from isaaclab.utils.math import quat_apply_inverse

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        print(f"[solve] {tag:12s} | q={float(scene.drawer_q()[0]):+.4f} "
              f"theta={math.degrees(float(scene.gate_theta()[0])):+.1f}deg "
              f"gate_closed={bool(scene.gate_closed()[0])} "
              f"drawer_closed={bool(scene.drawer_closed()[0])} "
              f"on_hinge={bool(scene.gate_on_hinge()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    zero = torch.zeros(n, 1, 3, device=device)

    def swing(theta_end: float, *, steps: int, rate: float = 0.30, kp: float = 2.5,
              kd: float = 0.5, clamp: float = 1.2, hold: int = 0, done=None,
              label: str = "") -> None:
        """Torque-servo the gate about the world vertical: a ramped angle target
        (so the swing is slow and regulated, ~rate rad/s) with PD torque, clamped
        to `clamp` N*m — a hand's push on the panel, force-limited. The wrench
        goes through the GATE only; the drawer is never touched."""
        tgt = scene.gate_theta().clone()
        end = torch.full_like(tgt, theta_end)
        dt = 1.0 / 120.0
        i = 0
        for i in range(steps):
            tgt = torch.maximum(tgt - rate * dt, end)
            th = scene.gate_theta()
            w = scene.gate.data.root_ang_vel_w[:, 2]
            tau = (kp * (tgt - th) - kd * w).clamp(-clamp, clamp)
            t_w = torch.zeros(n, 3, device=device)
            t_w[:, 2] = tau
            t_b = quat_apply_inverse(scene.gate.data.root_quat_w, t_w)
            scene.gate.set_external_force_and_torque(zero, t_b.reshape(n, 1, 3))
            env.step(no_action)
            if done is not None and bool(tgt[0] <= theta_end + 1e-6) and done():
                break
        for _ in range(hold):  # keep pressing at the end target
            th = scene.gate_theta()
            w = scene.gate.data.root_ang_vel_w[:, 2]
            tau = (kp * (end - th) - kd * w).clamp(-clamp, clamp)
            t_w = torch.zeros(n, 3, device=device)
            t_w[:, 2] = tau
            t_b = quat_apply_inverse(scene.gate.data.root_quat_w, t_w)
            scene.gate.set_external_force_and_torque(zero, t_b.reshape(n, 1, 3))
            env.step(no_action)
        scene.gate.set_external_force_and_torque(zero, zero)
        print(f"[solve] swing {label}: theta="
              f"{math.degrees(float(scene.gate_theta()[0])):+.2f}deg "
              f"q={float(scene.drawer_q()[0]):+.4f} after <= {i + 1}+{hold} steps",
              flush=True)

    # ---------------- phase 0: settle, layout readback, baseline ---------------------------------
    step(150)   # gate ring drops 1 mm onto its collar; drawer seats on the slab
    sp0 = (scene.station.data.root_pos_w - scene.env_origins)[0]
    sq0 = scene.station.data.root_quat_w[0]
    yaw = math.degrees(2.0 * math.atan2(float(sq0[3]), float(sq0[0])))
    q0 = float(scene.q0[0])
    th0 = float(scene.theta0[0])
    q_meas = float(scene.drawer_q()[0])
    th_meas = float(scene.gate_theta()[0])
    print(f"[solve] layout readback (seed {args.seed}): "
          f"station=({float(sp0[0]):+.3f},{float(sp0[1]):+.3f}) yaw~{yaw:+.1f}deg "
          f"q0={q0:+.4f} (measured {q_meas:+.4f}) "
          f"theta0={math.degrees(th0):+.1f}deg (measured "
          f"{math.degrees(th_meas):+.1f}deg)", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert bool(scene.gate_on_hinge()[0]), "gate must hang on its pin"
    assert bool(scene.drawer_in_channel()[0]), "drawer must ride in its channel"
    assert not bool(scene.drawer_closed()[0]), "drawer must start open"
    assert not bool(scene.gate_closed()[0]), "gate must start open"
    assert abs(q_meas - q0) < 0.010, "drawer must rest at its sampled opening"
    assert abs(th_meas - th0) < math.radians(4.0), \
        "gate must hang at its sampled angle"
    s_prev = print_score("P0 reset+settle (gate standing open, drawer out)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s_prev <= 0.05, f"baseline score should be ~0, got {s_prev}"

    # Invert the scene's cam relation at the MEASURED opening: the gate angle at
    # which its inner face first touches the roller (bisection; cam_q monotone).
    lo, hi = 0.0, th_meas
    for _ in range(60):
        mid = (lo + hi) / 2
        if c.cam_q(mid) > q_meas:
            hi = mid
        else:
            lo = mid
    th_touch = (lo + hi) / 2
    print(f"[solve] cam touch angle at q={q_meas:+.4f}: "
          f"{math.degrees(th_touch):+.2f}deg", flush=True)

    # ---------------- phase 1: approach — swing to just above the touch angle --------------------
    th_pre = th_touch + 0.04
    swing(th_pre, steps=600, hold=30, label="approach")
    q_after = float(scene.drawer_q()[0])
    assert abs(q_after - q_meas) < 0.008, \
        f"drawer must not move during the approach (q {q_meas:+.4f} -> {q_after:+.4f})"
    assert abs(float(scene.gate_theta()[0]) - th_pre) < math.radians(6.0), \
        "gate must track the servo to the pre-touch angle"
    report("approach")
    s = print_score("P1 gate swung to the cam touch angle (drawer untouched)")
    assert s >= s_prev - 1e-6, "score decreased across P1"
    assert s >= 0.08, f"P1 must earn real gate credit, got {s:.3f}"
    assert s <= 0.55, f"P1 score {s:.3f} too high — drawer credit leaked?"
    s_prev = s

    # ---------------- phase 2: cam drive — swing through touch to flush --------------------------
    # The gate's inner face presses the red roller; the cam converts the regulated
    # swing into the drawer's closing translation. Target just past flush so the
    # gate presses its stop post firmly (the post IS the flush rest).
    swing(-0.06, steps=900, rate=0.25, hold=150,
          done=lambda: bool(scene.gate_closed()[0]) and bool(scene.drawer_closed()[0]),
          label="cam drive")
    report("cam-drive")
    # hands off; the gate rests on its post, the drawer stays seated (no spring)
    step(180)
    report("released")
    assert bool(scene.gate_closed()[0]), \
        f"gate failed to seat flush: theta={math.degrees(float(scene.gate_theta()[0])):+.2f}deg"
    assert bool(scene.drawer_closed()[0]), \
        f"cam failed to seat the drawer: q={float(scene.drawer_q()[0]):+.4f}"
    assert bool(scene.success()[0]), "success() must hold on the settled flush state"
    s = print_score("P2 cam drive: gate flush on its post, drawer seated, hands off")
    assert s >= s_prev - 1e-6, "score decreased across P2"
    assert s >= 1.0 - 1e-6, "success must score 1.0"
    s_prev = s

    # ---------------- persistence (>= 3 simulated seconds, hands-off) ----------------------------
    hold_ok = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold_ok = hold_ok and bool(scene.success()[0])
    report("persist")
    s_final = print_score("P-final persistence 3.3 s")
    ok = hold_ok and bool(scene.success()[0]) and s_final >= s_prev - 1e-6
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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
