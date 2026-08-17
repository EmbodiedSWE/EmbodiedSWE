"""Teleport solution for WeaveCloseCrateScene — the task's legitimacy certificate.

NullRobot scene-level env. Teleportation is used for TRANSPORT ONLY (lifting each
item from the open floor and moving it across free space to above the open crate's
mouth); every load-bearing interaction goes through contact dynamics:
  - the PACKING: each item is released ABOVE the rim and falls in through the open
    mouth, impacts the crate floor and settles — never spawned seated inside.
  - the CLOSING: each flap is flipped shut by a bounded torque about its hinge
    (the moment a fingertip push on the plate produces, applied through the scene's
    `tuck_tau`/`main_tau` plant buffers) — a gravity-feedforward rate servo swings
    the flap up from its open rest, over vertical, and lowers it rate-limited onto
    its stop; the final seating impacts, the main flap LANDING ON the seated tuck
    flap (the weave) and every ring-down are resolved by contact dynamics. The
    flaps are never teleported.

Phases (each boundary prints `SIM_GEN_SCORE`, non-decreasing):
  0. reset(seed), settle; assert flaps at open rest + items standing  -> score ~0
  1. PACK can: teleport to hover above the mouth (crate-frame x -5 cm), drop,
     settle                                                           -> score 0.20
  2. PACK candle: same at crate-frame x +5 cm                         -> score 0.40
  3. TUCK: servo the short red flap shut (cap 0.10 N m); it seats flat on its
     0-degree stop; wait for the `tuck_set` latch                     -> score 0.60
  4. MAIN: servo the long blue flap shut (cap 0.55 N m); its tip lands ON the
     seated tuck flap — the weave; wait for `lid_set` + success       -> score 1.00
  5. PERSIST: >= 3.5 s of pure simulation, success at every poll
     -> print `SIM_GEN_SOLVE: SUCCESS`.

Run (forge): python -u -m simgen_tasks.box_task_replay_i318.solve --headless [--seed N]
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
    from .scene import WeaveCloseCrateScene  # registers "weave_close_crate" + env
except ImportError:  # direct-file fallback
    from scene import WeaveCloseCrateScene

assert WeaveCloseCrateScene is not None


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
    env = ENVS.get("simgen.weave_close_crate")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    one = torch.arange(1, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        p = scene.crate_pos()[0].tolist()
        loc = scene.items_local()[0]
        print(f"[solve] {tag:10s} | crate=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
              f"tuck={math.degrees(float(scene.tuck_open()[0])):+7.1f} "
              f"main={math.degrees(float(scene.main_open()[0])):+7.1f} "
              f"weave={float(scene.weave_delta()[0]):+.4f} "
              f"can=({float(loc[0, 0]):+.3f},{float(loc[0, 1]):+.3f},{float(loc[0, 2]):+.3f}) "
              f"cnd=({float(loc[1, 0]):+.3f},{float(loc[1, 1]):+.3f},{float(loc[1, 2]):+.3f}) "
              f"Pc={bool(scene.packed_can[0])} Pd={bool(scene.packed_candle[0])} "
              f"T={bool(scene.tuck_set[0])} L={bool(scene.lid_set[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])}",
              flush=True)

    def score_line() -> float:
        s = float(scene.score()[0])
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ================= phase 0: reset + settle ================================================
    env.reset(seed=args.seed)
    step(90)
    report("settled")
    s0 = score_line()
    if not (float(scene.tuck_open()[0]) > math.radians(150.0)
            and float(scene.main_open()[0]) > math.radians(150.0)):
        print("SIM_GEN_SOLVE: FAILURE (flaps not at the open rest after reset)", flush=True)
        hard_exit(1)
    can_z = float(scene.can.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    cnd_z = float(scene.candle.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    if abs(can_z - c.can_h / 2) > 0.02 or abs(cnd_z - c.candle_h / 2) > 0.02:
        print("SIM_GEN_SOLVE: FAILURE (items not standing on the floor at reset)", flush=True)
        hard_exit(1)
    if s0 > 0.05:
        print("SIM_GEN_SOLVE: FAILURE (nonzero score before doing anything)", flush=True)
        hard_exit(1)

    # ================= phases 1+2: PACK (transport teleport + physical drop-in) ===============
    # Transport only: each item is lifted from the open floor and moved to hover
    # ABOVE the crate mouth (crate-frame x offset so the two land side by side),
    # then released — entry through the mouth, floor impact and settling are all
    # contact dynamics.
    from isaaclab.utils.math import quat_apply

    def drop_item(body, dx: float, h: float) -> None:
        q = scene.crate.data.root_quat_w[0:1]
        off = torch.tensor([[dx, 0.0, c.rim_z_local + c.drop_clear + h / 2]], device=device)
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = scene.crate.data.root_pos_w[0] + quat_apply(q, off)[0]
        st[0, 3:7] = q[0]  # yaw-aligned with the crate (upright)
        body.write_root_state_to_sim(st, one)

    for name, body, dx, h, latch, floor_s in (
            ("can", scene.can, c.drop_dx[0], c.can_h, "packed_can", 0.20),
            ("candle", scene.candle, c.drop_dx[1], c.candle_h, "packed_candle", 0.40)):
        drop_item(body, dx, h)
        ok = False
        for _ in range(480):
            env.step(no_action)
            if bool(getattr(scene, latch)[0]):
                ok = True
                break
        step(30)
        report(f"{name}_in")
        s = score_line()
        if not ok:
            print(f"SIM_GEN_SOLVE: FAILURE ({name} did not settle inside the crate)", flush=True)
            hard_exit(1)
        if s < floor_s - 1e-6:
            print(f"SIM_GEN_SOLVE: FAILURE ({name} credit not latched)", flush=True)
            hard_exit(1)

    # ================= phases 3+4: CLOSE (torque servo per flap, tuck first) ==================
    # Gravity-feedforward rate servo about the hinge: tau_close =
    # clamp(-mgd*cos(a) + k*(w_des_eff - w_close), -cap, cap); w_des tapers near
    # closed so the flap lands rate-limited; torque cut at `cut` and gravity seats
    # it on the stop (gain bounds: k*dt/I_hinge = 0.36 tuck / 0.32 main, < 1).
    def close_flap(tag: str, buf: torch.Tensor, angle_fn, col: int,
                   mgd: float, k: float, w_des: float, cap: float) -> bool:
        cut = math.radians(8.0)
        best_a, best_i = float("inf"), 0
        done = False
        for i in range(1800):  # up to 15 s
            a = float(angle_fn()[0])
            if a < cut:
                done = True
                break
            if a < best_a - 0.01:
                best_a, best_i = a, i
            if i - best_i > 720:  # 6 s with no progress: report, don't grind
                print(f"[solve] STALL: {tag} stuck at {math.degrees(a):+.1f} deg", flush=True)
                break
            w_close = -float(scene.flap_rate[0, col])
            w_eff = w_des * min(1.0, max(0.15, a / 0.7))
            tau = -mgd * math.cos(a) + k * (w_eff - w_close)
            buf[:] = max(-cap, min(cap, tau))
            env.step(no_action)
            if i % 240 == 0:
                report(f"{tag}{i:4d}")
        buf[:] = 0.0
        return done

    if not close_flap("tuck", scene.tuck_tau, scene.tuck_open, 0,
                      c.tuck_mgd, 0.004, 2.5, c.tuck_tau_max):
        print("SIM_GEN_SOLVE: FAILURE (tuck flap never reached the closed band)", flush=True)
        hard_exit(1)
    ok_t = False
    for i in range(900):
        env.step(no_action)
        if bool(scene.tuck_set[0]):
            ok_t = True
            break
        if i % 240 == 0:
            report(f"tseat{i:4d}")
    step(30)
    report("tuck_set")
    s3 = score_line()
    if not ok_t:
        print("SIM_GEN_SOLVE: FAILURE (tuck flap did not seat/latch)", flush=True)
        hard_exit(1)
    if s3 < 0.60 - 1e-6:
        print("SIM_GEN_SOLVE: FAILURE (tuck credit not latched)", flush=True)
        hard_exit(1)

    if not close_flap("main", scene.main_tau, scene.main_open, 1,
                      c.main_mgd, 0.10, 2.0, c.main_tau_max):
        print("SIM_GEN_SOLVE: FAILURE (main flap never reached the closed band)", flush=True)
        hard_exit(1)
    ok_m = False
    for i in range(900):
        env.step(no_action)
        if bool(scene.lid_set[0]) and bool(scene.success()[0]):
            ok_m = True
            break
        if i % 240 == 0:
            report(f"mseat{i:4d}")
    step(30)
    report("lid_set")
    s4 = score_line()
    if not ok_m or not bool(scene.success()[0]):
        print("SIM_GEN_SOLVE: FAILURE (main flap did not weave onto the tuck / no success)",
              flush=True)
        hard_exit(1)
    if s4 < s3 - 1e-6 or abs(s4 - 1.0) > 1e-3:
        print("SIM_GEN_SOLVE: FAILURE (score not 1.0 at success)", flush=True)
        hard_exit(1)

    # ================= phase 5: PERSIST (>= 3.5 s, no intervention) ===========================
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
