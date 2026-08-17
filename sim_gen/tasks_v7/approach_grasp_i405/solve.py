"""Teleport solution for FlywheelInterlockScene — the task's legitimacy certificate.

NullRobot scene-level env. The parcel is teleported ONCE, for TRANSPORT ONLY (deck
spawn -> staging pose on the same deck, both far outside the machine); every
load-bearing interaction is real physics:
  - ARREST / ALIGN / RESTART act through the scene's `wheel_drive` plant buffer — the
    bounded torque of a hand dragging the rim / yellow peg about the axle;
  - POST slides the parcel by a bounded external CoM force (a fingertip push): the
    parcel crosses the deck, bridges the gap through the wheel's parked open sector,
    passes the mouth, and GRAVITY drops it into the sealed vault — the rubric's
    `deposited` latch fires only from this continuous, physical path;
  - the wheel is never teleported, and the parcel is never teleported inside (the
    vault is roofed: the mouth is its only aperture).

Phases (each boundary prints `SIM_GEN_SCORE`, non-decreasing):
  0. reset(seed), settle; verify the wheel is LIVE (readback vs the episode draw)  ~0
  1. ARREST: brake to a standstill; hold until `calmed` latches               -> 0.15
  2. ALIGN: jog the stopped wheel until the open sector faces the mouth       -> 0.15
  3. STAGE (teleport, transport only) + POST: force-push through mouth        -> 0.50
  4. RESTART: spin back up; `respun` latches; success() goes live             -> 1.00
  5. PERSIST: >= 3 s of pure simulation, no intervention; success must hold at
     every poll and at the end -> print `SIM_GEN_SOLVE: SUCCESS`.

Run (forge): python -u -m simgen_tasks.approach_grasp_i405.solve --headless [--seed N]
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
    from .scene import FlywheelInterlockScene  # registers "flywheel_interlock" + env
except ImportError:  # direct-file fallback
    from scene import FlywheelInterlockScene

assert FlywheelInterlockScene is not None


def hard_exit(code: int) -> None:
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        app.close()
    finally:
        os._exit(code)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.flywheel_interlock")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    from isaaclab.utils.math import quat_apply_inverse  # noqa: PLC0415

    no_action = torch.empty(0, device=device)
    zero3 = torch.zeros(1, 1, 3, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def wrap(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def hold_drive() -> None:
        """Park servo: keep the stopped wheel's open sector at the mouth."""
        th = wrap(float(scene.wheel_yaw()[0]))
        w = float(scene.wheel_w[0])
        scene.wheel_drive[:] = max(-0.22, min(0.22, -0.6 * th - 0.12 * w))

    def parcel_wrench(fx: float, fy: float, fz: float, tz: float) -> None:
        q = scene.parcel.data.root_quat_w
        f_b = quat_apply_inverse(q, torch.tensor([[fx, fy, fz]], device=device))
        t_b = quat_apply_inverse(q, torch.tensor([[0.0, 0.0, tz]], device=device))
        scene.parcel.set_external_force_and_torque(
            f_b.reshape(1, 1, 3), t_b.reshape(1, 1, 3))

    def report(tag: str) -> None:
        p = scene._local(scene.parcel)[0].tolist()
        print(f"[solve] {tag:10s} | th={math.degrees(wrap(float(scene.wheel_yaw()[0]))):+7.1f}deg "
              f"w={float(scene.wheel_w[0]):+.3f} "
              f"parcel=({p[0]:+.3f},{p[1]:+.3f},{p[2]:.3f}) "
              f"calm={bool(scene.calmed[0])} dep={bool(scene.deposited[0])} "
              f"respun={bool(scene.respun[0])} foul={bool(scene.fouled[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])}",
              flush=True)

    def score_line() -> float:
        s = float(scene.score()[0])
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ================= phase 0: reset + settle (the machine is LIVE) ==========================
    env.reset(seed=args.seed)
    step(30)
    report("settled")
    s0 = score_line()
    w0 = float(scene.w0[0])
    w_fd = float(scene.wheel_w[0])
    p = scene._local(scene.parcel)[0]
    if not (abs(w_fd - w0) < 0.15 * abs(w0) + 0.10 and abs(w_fd) >= c.w0_lo * 0.8):
        print(f"SIM_GEN_SOLVE: FAILURE (wheel not live after reset: fd={w_fd:.2f} "
              f"draw={w0:.2f})", flush=True)
        hard_exit(1)
    if not (abs(float(p[2]) - c.deck_rest_z) < 0.006 and float(p[1]) < -(c.slab_y + 0.04)):
        print("SIM_GEN_SOLVE: FAILURE (parcel not resting on the deck after reset)",
              flush=True)
        hard_exit(1)
    if bool(scene.fouled[0]) or bool(scene.calmed[0]):
        print("SIM_GEN_SOLVE: FAILURE (latches dirty at reset)", flush=True)
        hard_exit(1)

    # ================= phase 1: ARREST (brake the live wheel; earn `calmed`) ==================
    for i in range(1200):
        w = float(scene.wheel_w[0])
        scene.wheel_drive[:] = max(-0.27, min(0.27, -0.20 * w))
        env.step(no_action)
        if bool(scene.calmed[0]):
            break
        if i % 240 == 120:
            report(f"brake{i:4d}")
    report("calmed")
    if not bool(scene.calmed[0]):
        print("SIM_GEN_SOLVE: FAILURE (wheel never verifiably stopped)", flush=True)
        hard_exit(1)
    s1 = score_line()
    if s1 < s0 - 1e-6 or s1 < c.w_calmed - 1e-3:
        print("SIM_GEN_SOLVE: FAILURE (calm credit missing)", flush=True)
        hard_exit(1)

    # ================= phase 2: ALIGN (jog the open sector over the mouth) ====================
    ok_n = 0
    for i in range(1800):
        hold_drive()
        env.step(no_action)
        th = wrap(float(scene.wheel_yaw()[0]))
        if abs(th) < 0.06 and abs(float(scene.wheel_w[0])) < 0.10:
            ok_n += 1
            if ok_n >= 30:
                break
        else:
            ok_n = 0
        if i % 240 == 120:
            report(f"align{i:4d}")
    report("aligned")
    th = wrap(float(scene.wheel_yaw()[0]))
    if not (abs(th) < c.align_tol * 0.5 and bool(scene.gap_aligned()[0])):
        print("SIM_GEN_SOLVE: FAILURE (open sector never parked over the mouth)", flush=True)
        hard_exit(1)
    s2 = score_line()
    if s2 < s1 - 1e-6:
        print("SIM_GEN_SOLVE: FAILURE (score decreased during alignment)", flush=True)
        hard_exit(1)

    # ================= phase 3: STAGE (transport teleport) + POST (force push) ================
    st = scene.parcel.data.root_state_w[0:1].clone()
    st[:, 0] = scene.env_origins[0, 0] + 0.0
    st[:, 1] = scene.env_origins[0, 1] - 0.075
    st[:, 2] = scene.env_origins[0, 2] + c.deck_rest_z + 0.004
    st[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    st[:, 7:13] = 0.0
    scene.parcel.write_root_state_to_sim(st, torch.tensor([0], device=device))
    step(30)  # settle the staged parcel; continuity re-arms
    report("staged")

    pushed_clear = False
    for i in range(1500):
        hold_drive()  # keep the wheel parked against push grazes
        p = scene._local(scene.parcel)[0]
        py, px, pz = float(p[1]), float(p[0]), float(p[2])
        # clear = trailing face past the wall back face (y=0.024), or already dropping
        if py > 0.050 or pz < c.deck_rest_z - 0.020:
            pushed_clear = True
            break
        vy = float(scene.parcel.data.root_lin_vel_w[0, 1])
        fy = max(0.0, min(0.55, 6.0 * (0.08 - vy)))
        fx = max(-0.20, min(0.20, -8.0 * px))
        wz = float(scene.parcel.data.root_ang_vel_w[0, 2])
        tz = max(-0.02, min(0.02, -0.01 * wz))
        parcel_wrench(fx, fy, 0.0, tz)
        env.step(no_action)
        if i % 240 == 120:
            report(f"push{i:5d}")
    parcel_wrench(0.0, 0.0, 0.0, 0.0)  # fingertip off; gravity finishes the job
    if not pushed_clear:
        print("SIM_GEN_SOLVE: FAILURE (parcel never crossed the sill under push)", flush=True)
        hard_exit(1)
    for _ in range(240):  # 2 s: drop + settle inside the vault
        hold_drive()
        env.step(no_action)
    report("deposited")
    if not bool(scene.deposited[0]) or bool(scene.fouled[0]):
        print("SIM_GEN_SOLVE: FAILURE (parcel not deposited via the mouth)", flush=True)
        hard_exit(1)
    if not bool(scene.in_vault()[0]):
        print("SIM_GEN_SOLVE: FAILURE (parcel not resting in the vault)", flush=True)
        hard_exit(1)
    s3 = score_line()
    if s3 < c.w_calmed + c.w_dep - 1e-3:
        print("SIM_GEN_SOLVE: FAILURE (deposit credit missing)", flush=True)
        hard_exit(1)

    # ================= phase 4: RESTART (spin the machine back up) ============================
    w_t = 1.6  # target coast speed (> w_run 1.2 > w_hold 1.0)
    for i in range(1200):
        w = float(scene.wheel_w[0])
        scene.wheel_drive[:] = max(-0.27, min(0.27, 0.5 * (w_t - w)))
        env.step(no_action)
        if bool(scene.respun[0]):
            break
        if i % 240 == 120:
            report(f"spin{i:5d}")
    scene.wheel_drive[:] = 0.0  # hands off: the machine coasts
    step(30)
    report("respun")
    if not bool(scene.respun[0]):
        print("SIM_GEN_SOLVE: FAILURE (wheel never sustained the restart speed)", flush=True)
        hard_exit(1)
    if not bool(scene.success()[0]):
        print("SIM_GEN_SOLVE: FAILURE (no live success after restart)", flush=True)
        hard_exit(1)
    s4 = score_line()
    if s4 < s3 - 1e-6 or abs(s4 - 1.0) > 1e-3:
        print("SIM_GEN_SOLVE: FAILURE (score not 1.0 at success)", flush=True)
        hard_exit(1)

    # ================= phase 5: PERSIST (>= 3 s, no intervention) =============================
    ok = True
    for _ in range(14):  # 14 * 30 = 420 steps = 3.5 s
        step(30)
        ok = ok and bool(scene.success()[0])
    report("persist")
    s5 = score_line()
    if ok and bool(scene.success()[0]) and s5 >= s4 - 1e-6:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        hard_exit(0)
    print("SIM_GEN_SOLVE: FAILURE (success did not persist)", flush=True)
    hard_exit(1)


if __name__ == "__main__":
    main()
