"""Teleport solution for CrankEjectorScene — the task's legitimacy certificate.

NullRobot scene-level env. NO teleport of any object is ever needed (or used): the
whole solution is the same indirect actuation a robot would perform — a bounded torque
written to the scene's `crank_drive` plant buffer (the wrench of a hand turning the
YELLOW crank handle about the axle), while the scotch yoke, the ram and gravity do all
the work on the cubes:
  - CRANK: read the episode's cargo side, then servo the crank toward that side's stop;
    the drive peg pushes the ram fork, the ram plows the red cargo cube down the
    tunnel, over the drop lip, and it falls into the catch pocket — the rubric's
    `ejected`/`seated` latches fire only from this continuous, physical path;
  - RELEASE: zero drive; the crank settles at the stop, the cargo rests in the pocket,
    the grey decoy never moves — success() requires exactly this quiet end state.

Phases (each boundary prints `SIM_GEN_SCORE`, non-decreasing):
  0. reset(seed), settle                                     -> score ~0
  1. CRANK: servo toward the cargo side until `ejected`      -> score ~0.45
  2. FINISH: keep the stroke until the cargo `seated`        -> score 0.70
  3. RELEASE: zero drive, everything settles                 -> score 1.00 (success)
  4. PERSIST: >= 3 s of pure simulation, no intervention; success must hold at every
     poll and at the end -> print `SIM_GEN_SOLVE: SUCCESS`.

Run (forge): python -u -m simgen_tasks.approach_grasp_banana_i372.solve --headless [--seed N]
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
    from .scene import CrankEjectorScene  # registers "crank_ejector" + env
except ImportError:  # direct-file fallback
    from scene import CrankEjectorScene

assert CrankEjectorScene is not None


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
    env = ENVS.get("simgen.crank_ejector")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        pc = scene._local(scene.cargo)[0].tolist()
        print(f"[solve] {tag:10s} | th={math.degrees(float(scene.crank_theta()[0])):6.1f}deg "
              f"ram_x={float(scene.ram_x()[0]):+.3f} side={float(scene.side[0]):+.0f} "
              f"cargo=({pc[0]:+.3f},{pc[1]:+.3f},{pc[2]:.3f}) "
              f"prog={float(scene.prog[0]):.2f} ej={bool(scene.ejected[0])} "
              f"seat={bool(scene.seated[0])} foul={bool(scene.fouled[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])}",
              flush=True)

    def score_line() -> float:
        s = float(scene.score()[0])
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ================= phase 0: reset + settle ================================================
    env.reset(seed=args.seed)
    step(60)
    report("settled")
    s0 = score_line()
    pc = scene._local(scene.cargo)[0]
    pd = scene._local(scene.decoy)[0]
    if not (abs(float(pc[0])) < c.tunnel_hl and abs(float(pc[2]) - c.cube_rest_z) < 0.008
            and abs(float(pd[0])) < c.tunnel_hl and abs(float(pd[2]) - c.cube_rest_z) < 0.008):
        print("SIM_GEN_SOLVE: FAILURE (cubes not resting in the tunnel after reset)",
              flush=True)
        hard_exit(1)

    # ================= phases 1+2: CRANK the cargo side until ejected, then seated ============
    # Velocity servo with escalating feed-forward (stall-proof): tau = dir*ff
    # + k*(dir*w_des - w_fd), clamped inside drive_max; hard speed guard. The drive
    # peg's lever does the rotary->linear conversion; gravity finishes the job.
    side = float(scene.side[0])
    w_des, k, ff = 1.2, 0.25, 0.06
    cap = 0.9 * c.drive_max
    th_mark, i_mark = float(scene.crank_theta()[0]), 0

    def servo_step() -> None:
        w = float(scene.crank_w[0])
        tau = side * ff + k * (side * w_des - w)
        if abs(w) > 2.5 * w_des:
            tau = 0.0
        scene.crank_drive[:] = max(-cap, min(cap, tau))
        env.step(no_action)

    done_eject = False
    for i in range(2400):
        if bool(scene.ejected[0]) and not done_eject:
            done_eject = True
            report("ejected")
            s1 = score_line()
            if s1 < s0 - 1e-6:
                print("SIM_GEN_SOLVE: FAILURE (score decreased at ejection)", flush=True)
                hard_exit(1)
        if bool(scene.seated[0]):
            break
        servo_step()
        if i % 240 == 120:
            report(f"crank{i:4d}")
        if i - i_mark >= 240:  # stall watch: escalate ff/k if the crank stops advancing
            th = float(scene.crank_theta()[0])
            if side * (th - th_mark) < math.radians(2.0):
                ff = min(ff * 1.6, 0.40)
                k = min(k * 1.3, 0.60)
                print(f"[solve] stall escalation: ff={ff:.3f} k={k:.3f}", flush=True)
            th_mark, i_mark = th, i
    report("seated")
    if not done_eject or not bool(scene.seated[0]):
        print("SIM_GEN_SOLVE: FAILURE (cargo never ejected+seated in the goal pocket)",
              flush=True)
        hard_exit(1)
    if bool(scene.fouled[0]):
        print("SIM_GEN_SOLVE: FAILURE (decoy fouled during the correct stroke)", flush=True)
        hard_exit(1)
    s2 = score_line()
    if s2 < 0.70 - 1e-3:
        print("SIM_GEN_SOLVE: FAILURE (latched credit below 0.70 after seating)", flush=True)
        hard_exit(1)

    # ================= phase 3: RELEASE (let go of the crank; everything settles) =============
    scene.crank_drive[:] = 0.0
    step(300)  # 2.5 s
    report("released")
    s3 = score_line()
    if not bool(scene.success()[0]):
        print("SIM_GEN_SOLVE: FAILURE (no settled success after release)", flush=True)
        hard_exit(1)
    if s3 < s2 - 1e-6 or abs(s3 - 1.0) > 1e-3:
        print("SIM_GEN_SOLVE: FAILURE (score not 1.0 at success)", flush=True)
        hard_exit(1)

    # ================= phase 4: PERSIST (>= 3 s, no intervention) =============================
    ok = True
    for _ in range(14):  # 14 * 30 = 420 steps = 3.5 s
        step(30)
        ok = ok and bool(scene.success()[0])
    report("persist")
    s4 = score_line()
    if ok and bool(scene.success()[0]) and s4 >= s3 - 1e-6:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        hard_exit(0)
    print("SIM_GEN_SOLVE: FAILURE (success did not persist)", flush=True)
    hard_exit(1)


if __name__ == "__main__":
    main()
