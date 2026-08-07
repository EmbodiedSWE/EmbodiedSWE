"""Teleport solution for CompassDialsScene (sim_gen task `play_jenga_i31`) — the
task's legitimacy certificate.

NOTHING is teleported after reset: every arrow already sits captive on its pivot pin,
and the entire task is executed through CONTACT DYNAMICS. For each dial in turn, a
floating-hand velocity-servoed TORQUE about world +z (the tangential nudge a
fingertip on the arrow's shaft applies) spins the arrow about its pin — the pin-hole
contact is what constrains the swing, and bench friction is what the servo works
against and what finally holds the set heading. The torque is cut inside the
tolerance and the arrow settles on real friction before the next dial is touched.
The applied torque is along +z, the same axis the arrow rotates about, so it is
immune to the actuator's rotate-with-the-body wrench frame drag.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.play_jenga_i31.solve --headless [--seed N]
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


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.compass_dials")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        err = scene.heading_err()[0]
        on = scene.on_pin()[0]
        al = scene.aligned()[0]
        print(f"[solve] {tag:14s} | err_deg=("
              + ",".join(f"{math.degrees(float(e)):6.1f}" for e in err)
              + ") on_pin=(" + ",".join(str(bool(v))[0] for v in on)
              + ") aligned=(" + ",".join(str(bool(v))[0] for v in al)
              + f") settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear(k: int) -> None:
        scene.arrows[k].set_external_force_and_torque(zero, zero, env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    az = scene.target_az[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"targets_deg=(" + ",".join(f"{math.degrees(float(a)):+7.1f}" for a in az)
          + ") spawn_err_deg=("
          + ",".join(f"{math.degrees(float(e)):6.1f}" for e in scene.heading_err()[0])
          + ")", flush=True)
    for k, nm in enumerate(scene.names):
        p = (scene.pins[k].data.root_pos_w - scene.env_origins)[0]
        q = (scene.posts[k].data.root_pos_w - scene.env_origins)[0]
        print(f"[solve]   dial {nm:5s}: pin=({float(p[0]):+.3f},{float(p[1]):+.3f}) "
              f"post=({float(q[0]):+.3f},{float(q[1]):+.3f})", flush=True)
    report("reset")
    s_prev = print_score("P0 reset+settle")

    # ---------------- phases 1..3: swing each dial through pin-hole contact ----------------
    # The motion regime is friction-dominated (light bar flat on the bench), so the
    # right controller is PULSE-AND-COAST creep, not a stiff velocity servo: apply a
    # drive torque about world +z only while the swing speed is below an error-scaled
    # cap, coast (zero torque, bench friction brakes almost instantly) when above it,
    # and escalate the pulse torque only if the arrow provably is not creeping (static
    # friction unbroken). Inside the stop band the controller brakes, cuts the wrench,
    # and lets friction alone hold the heading. The +z torque axis coincides with the
    # swing axis, so the actuator's rotate-with-the-body wrench drag cannot rotate it.
    tol = math.radians(c.align_tol_deg)
    for k, nm in enumerate(scene.names):
        target = float(scene.target_az[0, k])
        done = False
        tau_a = 0.02  # pulse torque (N m); escalates until the arrow actually creeps
        for round_i in range(4):
            still_ct = 0
            for i in range(2500):
                err = float(scene._wrap(scene.headings()[:, k] - scene.target_az[:, k])[0])
                omega = float(scene.arrows[k].data.root_ang_vel_w[0, 2])
                if abs(err) < 0.02 and abs(omega) < 0.05:
                    break
                sgn = -1.0 if err > 0.0 else 1.0  # swing the heading toward the target
                if abs(err) < 0.02:  # stop band: brake only
                    tau = (-tau_a if omega > 0.0 else tau_a) if abs(omega) > 0.05 else 0.0
                else:
                    w_cap = min(1.0, max(0.15, 2.0 * abs(err)))
                    tau = sgn * tau_a if sgn * omega < w_cap else 0.0
                    if abs(omega) < 0.02:  # provably not creeping under this pulse
                        still_ct += 1
                        if still_ct > 40:
                            tau_a = min(tau_a * 1.5, 0.30)
                            still_ct = 0
                            print(f"[solve] dial {nm}: static at err="
                                  f"{math.degrees(err):+.1f} deg, tau_a={tau_a:.3f}",
                                  flush=True)
                    else:
                        still_ct = 0
                tq = torch.zeros(n, 1, 3, device=device)
                tq[:, 0, 2] = tau
                scene.arrows[k].set_external_force_and_torque(
                    zero, tq, env_ids=all_ids, is_global=True)
                env.step(no_action)
            clear(k)
            step(90)  # friction settle, hands off
            err_now = float(scene.heading_err()[0, k])
            if err_now <= tol * 0.6 and bool(scene.on_pin()[0, k]):
                done = True
                break
            print(f"[solve] dial {nm}: settled at "
                  f"{math.degrees(err_now):.1f} deg (round {round_i}), retrying", flush=True)
        report(f"dial-{nm}")
        s_now = print_score(f"P{k + 1} dial {nm} set + settle")
        assert s_now >= s_prev - 1e-6, f"score decreased across dial {nm}"
        s_prev = s_now
        if not done:
            print(f"SIM_GEN_SOLVE: FAIL (dial {nm} did not align: target {target:+.3f})",
                  flush=True)
            os._exit(1)

    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after all dials set)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, no intervention) -----
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s_prev - 1e-6
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
