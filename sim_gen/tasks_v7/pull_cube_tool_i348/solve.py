"""Solution certificate for BoomCorralScene — and the task's legitimacy proof.

NullRobot scene-level env. This solve uses ZERO teleports on the success path: the cube
and the pen are both permanently out of reach, and the ONLY thing that ever moves the
cube is the boom's far blade, driven through the scene's `boom_drive` plant buffer with
a bounded yaw torque (|tau| <= 1.5 N*m ~ 3.6 N at the 0.42 m handle — a one-hand push
on the handle post). Every phase is contact dynamics.

Phases (each boundary prints `SIM_GEN_SCORE`, non-decreasing):
  0. reset(seed), settle — cube resting far-field on the blade circle, blade behind it
     on the anti-pen side; assert score ~0.
  1. HERD: read the pen side, closed-loop yaw-rate servo (omega_des = side * 0.25 rad/s,
     tau = k*(omega_des - omega) clamped to +-drive_cap) sweeps the blade into the cube
     and pushes it along the arc through the pen mouth until its pen-frame depth
     >= 0.07 m; assert the `swept` and `entered` latches               -> score ~0.5+
  2. PARK: reverse the servo briefly to back the blade out of the mouth, zero the
     drive, let everything settle; the cube rests inside the pen        -> score 1.00
  3. PERSIST: >= 3.5 s of pure simulation, no intervention; success at every poll and
     at the end -> print `SIM_GEN_SOLVE: SUCCESS`.

Run (forge): python -u -m simgen_tasks.pull_cube_tool_i348.solve --headless [--seed N]
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

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

try:
    from .scene import BoomCorralScene  # registers "boom_corral" + env
except ImportError:  # direct-file fallback
    from scene import BoomCorralScene

assert BoomCorralScene is not None


def hard_exit(code: int) -> None:
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        app.close()
    finally:
        os._exit(code)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.boom_corral")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        cube = (scene.cube.data.root_pos_w[0] - scene.env_origins[0]).tolist()
        uw = scene.cube_pen_frame()[0].tolist()
        print(f"[solve] {tag:10s} | cube=({cube[0]:.3f},{cube[1]:.3f},{cube[2]:.3f}) "
              f"az={math.degrees(float(scene.cube_azimuth()[0])):6.1f}deg "
              f"boom={math.degrees(float(scene.boom_yaw()[0])):6.1f}deg "
              f"arc={math.degrees(float(scene.arc_done[0])):6.1f}/"
              f"{math.degrees(float(scene.arc_req[0])):.1f}deg "
              f"pen_uw=({uw[0]:+.3f},{uw[1]:+.3f}) "
              f"swept={bool(scene.swept[0])} entered={bool(scene.entered[0])} "
              f"in_pen={bool(scene.in_pen()[0])} score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])}", flush=True)

    def score_line() -> float:
        s = float(scene.score()[0])
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ================= phase 0: reset + settle ================================================
    env.reset(seed=args.seed)
    step(60)
    report("settled")
    s0 = score_line()
    rel = scene.cube_rel()[0]
    r0 = float(rel.norm())
    z0 = float(scene.cube.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    side = float(scene.side[0])
    if not (c.band_r_min < r0 < c.band_r_max and abs(z0 - c.rest_z) < 0.01
            and not bool(scene.swept[0]) and not bool(scene.entered[0]) and s0 < 0.05):
        print("SIM_GEN_SOLVE: FAILURE (bad initial state)", flush=True)
        hard_exit(1)

    # ================= phase 1: HERD (bounded yaw torque, blade pushes the cube) ==============
    # Yaw-rate servo on a FINITE-DIFFERENCE boom rate (the velocity readback can report
    # phantom rates in sustained contact, which would zero a naive servo): the blade
    # closes the 13-18 deg start gap, pockets the cube, and pushes it along the arc
    # through the pen mouth until its pen-frame depth >= 0.07 m. A slowly escalating
    # feedforward bias (velocity-triggered reset) rides through contact stiction; the
    # total command never exceeds the honest drive cap. Gain sized for the one-substep
    # wrench delay (k*dt/I ~ 0.45 < 1).
    k_om, om_des, dt = 4.0, side * 0.25, 1.0 / 120.0
    bias = 0.0
    yaw_prev = float(scene.boom_yaw()[0])
    done_herd = False
    for i in range(3000):
        uw = scene.cube_pen_frame()[0]
        if float(uw[0]) >= 0.070 and abs(float(uw[1])) < c.in_pen_w:
            done_herd = True
            break
        yaw = float(scene.boom_yaw()[0])
        om_fd = ((yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi) / dt
        yaw_prev = yaw
        if side * om_fd < 0.06:  # no real progress -> lean on the drive
            bias = min(bias + 0.01, 1.0)
        elif side * om_fd > 0.5 * abs(om_des):  # moving again -> relax
            bias = max(bias - 0.05, 0.0)
        tau = k_om * (om_des - om_fd) + side * bias
        scene.boom_drive[:] = max(-c.drive_cap, min(c.drive_cap, tau))
        env.step(no_action)
        if i % 240 == 0:
            report(f"herd{i:5d}")
            print(f"[solve]   servo: om_fd={om_fd:+.3f} "
                  f"om_rb={float(scene.boom_rate()[0]):+.3f} bias={bias:.2f} "
                  f"tau={tau:+.2f}", flush=True)
    report("herded")
    s1 = score_line()
    if not (done_herd and bool(scene.swept[0]) and bool(scene.entered[0])):
        print("SIM_GEN_SOLVE: FAILURE (herd did not complete or latch)", flush=True)
        hard_exit(1)
    if s1 < s0 - 1e-6:
        print("SIM_GEN_SOLVE: FAILURE (score decreased across the herd)", flush=True)
        hard_exit(1)

    # ================= phase 2: PARK (back the blade out, let the cube settle) ================
    yaw_prev = float(scene.boom_yaw()[0])
    for _ in range(110):  # ~12 deg reverse at 0.22 rad/s — blade clears the mouth
        yaw = float(scene.boom_yaw()[0])
        om_fd = ((yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi) / dt
        yaw_prev = yaw
        scene.boom_drive[:] = max(-c.drive_cap, min(c.drive_cap,
                                                    k_om * (-side * 0.22 - om_fd)))
        env.step(no_action)
    scene.boom_drive[:] = 0.0
    step(180)  # boom spins down under viscous friction; cube settles in the pen
    report("parked")
    s2 = score_line()
    if not bool(scene.success()[0]):
        print("SIM_GEN_SOLVE: FAILURE (cube did not settle inside the pen)", flush=True)
        hard_exit(1)
    if s2 < s1 - 1e-6 or abs(s2 - 1.0) > 1e-3:
        print("SIM_GEN_SOLVE: FAILURE (score not 1.0 at success)", flush=True)
        hard_exit(1)

    # ================= phase 3: PERSIST (>= 3.5 s, no intervention) ============================
    ok = True
    for _ in range(14):  # 14 * 30 = 420 steps = 3.5 s
        step(30)
        ok = ok and bool(scene.success()[0])
    report("persist")
    s3 = score_line()
    if ok and bool(scene.success()[0]) and s3 >= s2 - 1e-6:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        hard_exit(0)
    print("SIM_GEN_SOLVE: FAILURE (success did not persist)", flush=True)
    hard_exit(1)


if __name__ == "__main__":
    main()
