"""Teleport solution for MicrowaveCarouselScene (sim_gen task
`libero_kitchen_scene7_open_the_microwave_i106`) — the task's legitimacy certificate.

NOTHING is teleported. The entire solution is one contact-mediated interaction: a
bounded tangential torque on the platter body (the proxy for a fingertip pushing the
rim / the white pegs sideways), PD-servoed on the RED CUP's signed bearing error from
the doorway, so the servo takes the short way around on every seed automatically. The
cargo is never touched — it rides the turntable on friction alone, which is exactly
the transport mechanism the rubric grades (slot invariance + uprightness). Torque is
cut before alignment completes; damping brings the platter to rest inside the window.

Plant numbers (why this converges): total yaw inertia ~= 6.6e-3 kg m^2 (platter
0.5*m*r^2 = 4.5e-3 + cargo m*r^2), torque cap 0.012 N m -> peak alpha ~= 1.8 rad/s^2,
rim acceleration alpha*r ~= 0.15 m/s^2 and centripetal w^2*r <= 0.06 m/s^2, both
~100x below the friction budget mu*g ~= 8 m/s^2 — the cargo cannot slip or tip under
this drive. Linearized closed loop: zeta ~= 1.9 (overdamped), slow pole ~0.56/s.
Servo gain bound K*dt/I ~= 0.04 << 1 (wrenches act one substep late on this stack).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
milestone credit is latched in post_step), then holds HANDS-OFF for >= 3.3 simulated
seconds after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene7_open_the_microwave_i106.solve --headless [--seed N]
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

# ----- servo constants (see plant numbers in the module docstring) ------------------------------
K_ANG = 0.03    # N m / rad  proportional gain on the cup's signed bearing error
D_ANG = 0.04    # N m s      derivative gain on platter spin
TAU_MAX = 0.012  # N m       torque cap (a fingertip-scale tangential push at the rim)
ALIGN_ERR = math.radians(6.0)   # cut the drive inside this (tol is 20 deg)
ALIGN_SPIN = 0.08                # ... and slower than this (coast ~= w/3 rad after cut)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.microwave_carousel")().build(num_envs=args.num_envs,
                                                        device=device)
    scene = env.scene
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def platter_torque(tau: float) -> None:
        """Apply a pure z-torque to the platter. The house convention wants BODY-frame
        wrenches (`is_global=True` silently drops torques on this stack); the platter
        is yaw-only, so a z-torque is the SAME vector in body and world frame — no
        transform needed. Re-set every step while driving."""
        t = torch.zeros(n, 1, 3, device=device)
        t[:, 0, 2] = tau
        scene.platter.set_external_force_and_torque(zero_w, t, env_ids=all_ids)

    def report(tag: str) -> None:
        e = math.degrees(float(scene.cup_bearing_err()[0]))
        psi = math.degrees(float(scene.platter_yaw()[0]))
        w = float(scene.platter.data.root_ang_vel_w[0, 2])
        se_c = float(scene._slot_err(scene.cup, scene.cup_slot)[0])
        se_b = float(scene._slot_err(scene.bottle, scene.bottle_slot)[0])
        print(f"[solve] {tag:16s} | err={e:+7.1f}deg psi={psi:+7.1f}deg "
              f"w={w:+.3f} slot=(cup {se_c * 1000:.1f}mm, bot {se_b * 1000:.1f}mm) "
              f"riding={bool(scene.riding_ok()[0])} settled={bool(scene.settled()[0])} "
              f"m1={float(scene.mile1_latch[0]):.0f} m2={float(scene.mile2_latch[0]):.0f} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(60)  # cargo drops its 2 mm spawn float and everything comes to rest
    e0 = float(scene.cup_bearing_err()[0])
    bot_d = scene.bottle.data.root_pos_w[0, 0:2] - scene._center_xy()[0]
    bot_th = math.degrees(math.atan2(float(bot_d[1]), float(bot_d[0])))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"cup err={math.degrees(e0):+.1f}deg "
          f"platter psi={math.degrees(float(scene.platter_yaw()[0])):+.1f}deg "
          f"bottle world bearing={bot_th:+.1f}deg", flush=True)
    report("reset")
    assert bool(scene.riding_ok()[0]), "cargo not riding intact at reset"
    assert abs(math.degrees(e0)) >= 90.0, \
        f"cup spawned too close to the doorway ({math.degrees(e0):+.1f} deg)"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"

    # ---------------- phase 1: rotate the loaded carousel (contact dynamics) ---------------
    # PD torque servo on the platter, error = the cup's signed bearing from the
    # doorway. The wrapped error makes the SHORT way the descent direction on every
    # seed; friction carries the cargo. Milestones latch inside the scene's post_step.
    m1_seen = m2_seen = False
    s_prev = s0
    slip_warned = 0
    for i in range(4200):  # 35 s hard budget; the plant needs ~8-12 s
        e = float(scene.cup_bearing_err()[0])
        w = float(scene.platter.data.root_ang_vel_w[0, 2])
        if abs(e) < ALIGN_ERR and abs(w) < ALIGN_SPIN:
            break
        tau = max(min(-K_ANG * e - D_ANG * w, TAU_MAX), -TAU_MAX)
        platter_torque(tau)
        env.step(no_action)
        if not bool(scene.riding_ok()[0]) and slip_warned < 5:
            slip_warned += 1
            report(f"SLIP @ {i}")
        if not m1_seen and float(scene.mile1_latch[0]) > 0.5:
            m1_seen = True
            report(f"mile1 @ {i}")
            s1a = print_score("P1a riding cup within 65 deg (friction transport)")
            assert s1a >= s_prev - 1e-6
            s_prev = s1a
        if not m2_seen and float(scene.mile2_latch[0]) > 0.5:
            m2_seen = True
            report(f"mile2 @ {i}")
            s1b = print_score("P1b riding cup within 40 deg (friction transport)")
            assert s1b >= s_prev - 1e-6
            s_prev = s1b
    platter_torque(0.0)
    report("drive cut")
    assert m1_seen and m2_seen, "milestones did not latch during the rotation"
    assert abs(math.degrees(float(scene.cup_bearing_err()[0]))) < 12.0, \
        "servo did not align the cup"

    # ---------------- phase 2: hands off, settle to success --------------------------------
    for _ in range(16):  # up to 4 s of hands-off settling (damping stops the platter)
        if bool(scene.success()[0]):
            break
        step(30)
    report("settled")
    s2 = print_score("P2 drive cut, all settled")
    assert s2 >= s_prev - 1e-6, "score decreased across settling"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after rotate+settle)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3 simulated seconds, no intervention) -------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                print(f"[solve] persist flicker @step {i}: "
                      f"err={math.degrees(float(scene.cup_bearing_err()[0])):+.1f} "
                      f"riding={bool(scene.riding_ok()[0])} "
                      f"settled={bool(scene.settled()[0])} "
                      f"w={float(scene.platter.data.root_ang_vel_w[0, 2]):+.4f}",
                      flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
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
    try:
        main()
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
