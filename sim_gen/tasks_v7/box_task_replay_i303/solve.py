"""Teleport solution for CartonFlipPackScene — the task's legitimacy certificate.

NullRobot scene-level env. Teleportation is used for TRANSPORT ONLY (lifting each item
from the open floor and moving it across free space to above the upright carton's
mouth); every load-bearing interaction goes through contact dynamics:
  - the RIGHTING: a bounded torque about the carton's body-x (ridge) axis — the moment
    a ~4.5 N fingertip push near the top edge produces, applied through the scene's
    `roll_tau` plant buffer — tips the mouth-down carton over a floor edge onto its
    side, then over the next edge onto its base: two chained edge-pivots, with the
    tipping thresholds, the falls, the landing impacts and the final rock-to-rest all
    resolved by contact dynamics. The carton is never teleported.
  - the LOADING: each item is released ABOVE the rim of the now-upright carton and
    falls in through the mouth, impacts the carton floor and settles — never spawned
    seated inside.

Phases (each boundary prints `SIM_GEN_SCORE`, non-decreasing):
  0. reset(seed), settle; assert mouth-down + items standing     -> score ~0
  1. RIGHT: pick the roll direction whose corridor is farthest from both items;
     bang-bang rate-governed roll torque (drive 0.32 N m while s*w < 0.9 rad/s,
     brake while s*w > 1.4, cut for good at up_z >= 0.70 — past the second balance
     point); wait for the upright rest + `righted` latch          -> score 0.30
  2. PACK can: teleport the can to hover above the mouth (body-x offset +4 cm),
     drop, settle                                                 -> score 0.55
  3. PACK candle: same at body-x offset -4 cm                     -> score 1.00
  4. PERSIST: >= 3.5 s of pure simulation, success at every poll
     -> print `SIM_GEN_SOLVE: SUCCESS`.

Run (forge): python -u -m simgen_tasks.box_task_replay_i303.solve --headless [--seed N]
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
    from .scene import CartonFlipPackScene  # registers "carton_flip_pack" + env
except ImportError:  # direct-file fallback
    from scene import CartonFlipPackScene

assert CartonFlipPackScene is not None


def hard_exit(code: int) -> None:
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        app.close()
    finally:
        os._exit(code)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.carton_flip_pack")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        p = scene.carton_pos()[0].tolist()
        loc = scene.items_local()[0]
        print(f"[solve] {tag:10s} | carton=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
              f"up_z={float(scene.up_z()[0]):+.3f} w={float(scene.roll_rate()[0]):+.2f} "
              f"can_loc=({float(loc[0, 0]):+.3f},{float(loc[0, 1]):+.3f},{float(loc[0, 2]):+.3f}) "
              f"cnd_loc=({float(loc[1, 0]):+.3f},{float(loc[1, 1]):+.3f},{float(loc[1, 2]):+.3f}) "
              f"R={bool(scene.righted[0])} Pc={bool(scene.packed_can[0])} "
              f"Pd={bool(scene.packed_candle[0])} "
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
    if not (float(scene.up_z()[0]) < -0.95):
        print("SIM_GEN_SOLVE: FAILURE (carton not mouth-down at reset)", flush=True)
        hard_exit(1)
    can_z = float(scene.can.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    cnd_z = float(scene.candle.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    if abs(can_z - c.can_h / 2) > 0.02 or abs(cnd_z - c.candle_h / 2) > 0.02:
        print("SIM_GEN_SOLVE: FAILURE (items not standing on the floor at reset)", flush=True)
        hard_exit(1)
    if s0 > 0.05:
        print("SIM_GEN_SOLVE: FAILURE (nonzero score before doing anything)", flush=True)
        hard_exit(1)

    # ================= phase 1: RIGHT (roll torque; two edge-pivots) ==========================
    # Roll direction: driving torque s about body-x moves the carton along
    # t_w(s) = -s * (z_hat x a_w). Both +-body-y corridors are item-free by
    # construction; pick the sign whose 30 cm travel corridor is FARTHEST from both
    # items (max of min point-to-segment distance).
    a = scene.roll_axis_w()[0]
    axy = a[:2] / a[:2].norm().clamp_min(1e-6)
    perp = torch.tensor([-float(axy[1]), float(axy[0])], device=device)  # z_hat x a_w
    c_xy = scene.carton_pos()[0, :2]
    item_xy = torch.stack([
        scene.can.data.root_pos_w[0, :2] - scene.env_origins[0, :2],
        scene.candle.data.root_pos_w[0, :2] - scene.env_origins[0, :2]])

    def corridor_clearance(s: float) -> float:
        t = -s * perp
        rel = item_xy - c_xy  # (2, 2)
        lam = (rel @ t).clamp(0.0, 0.30)
        d = (rel - lam.unsqueeze(-1) * t).norm(dim=-1)
        return float(d.min())

    s = 1.0 if corridor_clearance(1.0) >= corridor_clearance(-1.0) else -1.0
    print(f"[solve] roll sign s={s:+.0f} "
          f"(clear +1: {corridor_clearance(1.0):.3f} m, -1: {corridor_clearance(-1.0):.3f} m)",
          flush=True)

    # Smooth ONE-SIDED rate servo (drive only; gravity is the brake): a chattering
    # bang-bang excites a rock-and-scoot limit cycle on the light carton, so instead
    # tau = s * clamp(tau_ff + k * (w_des - s*w), 0, 0.42). Start torque 0.42 N m
    # > tau_tip1 (0.206) with 2x margin; gain within the wrench-delay stability
    # bound (k * dt / I_x = 0.25 / (120 * 0.00104 + m r^2 ~ 0.0043) ~ 0.5 < 1).
    tau_ff, k_w, w_des, tau_cap = 0.20, 0.25, 1.0, 0.42
    start_xy = scene.carton_pos()[0, :2].clone()
    done_roll = False
    best_uz, best_i = -1.0, 0
    for i in range(1800):  # up to 15 s
        uz = float(scene.up_z()[0])
        if uz >= 0.70:  # past the second balance point (up_z ~ 0.48)
            done_roll = True
            break
        if uz > best_uz + 0.02:
            best_uz, best_i = uz, i
        if i - best_i > 720:  # 6 s with no roll progress: report, don't grind
            print(f"[solve] STALL: up_z stuck at {uz:+.3f} (best {best_uz:+.3f})", flush=True)
            break
        if float((scene.carton_pos()[0, :2] - start_xy).norm()) > 0.45:
            print("[solve] WALK: carton scooted instead of pivoting", flush=True)
            break
        w = s * float(scene.roll_rate()[0])
        scene.roll_tau[:] = s * min(tau_cap, max(0.0, tau_ff + k_w * (w_des - w)))
        env.step(no_action)
        if i % 240 == 0:
            report(f"roll{i:4d}")
    scene.roll_tau[:] = 0.0
    report("roll_cut")
    if not done_roll:
        print("SIM_GEN_SOLVE: FAILURE (carton never rolled past the second balance point)",
              flush=True)
        hard_exit(1)

    # Free fall onto the base + rock to rest; wait for the righted latch.
    ok_r = False
    for i in range(900):
        env.step(no_action)
        if bool(scene.righted[0]) and bool(scene.carton_upright()[0]) \
                and bool(scene.carton_settled()[0]):
            ok_r = True
            break
        if i % 240 == 0:
            report(f"rest{i:4d}")
    step(30)
    report("righted")
    s1 = score_line()
    if not ok_r:
        print("SIM_GEN_SOLVE: FAILURE (carton did not settle upright)", flush=True)
        hard_exit(1)
    if s1 < s0 - 1e-6 or s1 < 0.30 - 1e-6:
        print("SIM_GEN_SOLVE: FAILURE (righted credit not latched)", flush=True)
        hard_exit(1)

    # ================= phases 2+3: PACK (transport teleport + physical drop-in) ===============
    # Transport only: each item is lifted from the open floor and moved across free
    # space to hover ABOVE the carton mouth (offset along the carton's body-x so the
    # two items land side by side), then released — entry through the mouth, the
    # impact on the carton floor and the settling are all contact dynamics.
    from isaaclab.utils.math import quat_apply

    one = torch.arange(1, device=device)

    def drop_item(body, dx: float, h: float) -> None:
        q = scene.carton.data.root_quat_w[0:1]
        off = quat_apply(q, torch.tensor([[dx, 0.0, 0.0]], device=device))[0]
        cp = scene.carton.data.root_pos_w[0]
        st = torch.zeros(1, 13, device=device)
        st[0, 0] = cp[0] + off[0]
        st[0, 1] = cp[1] + off[1]
        st[0, 2] = cp[2] + c.rim_z_local + c.drop_clear + h / 2  # bottom 2 cm above the rim
        st[0, 3] = 1.0  # upright (the arm reorients the item in hand)
        body.write_root_state_to_sim(st, one)

    drop_item(scene.can, c.drop_dx[0], c.can_h)
    ok_can = False
    for i in range(480):
        env.step(no_action)
        if bool(scene.packed_can[0]) and bool(scene.contained()[0, 0]):
            ok_can = True
            break
    step(30)
    report("can_in")
    s2 = score_line()
    if not ok_can:
        print("SIM_GEN_SOLVE: FAILURE (can did not settle inside the carton)", flush=True)
        hard_exit(1)
    if s2 < s1 - 1e-6 or s2 < 0.55 - 1e-6:
        print("SIM_GEN_SOLVE: FAILURE (can credit not latched)", flush=True)
        hard_exit(1)

    drop_item(scene.candle, c.drop_dx[1], c.candle_h)
    ok_cnd = False
    for i in range(480):
        env.step(no_action)
        if bool(scene.packed_candle[0]) and bool(scene.success()[0]):
            ok_cnd = True
            break
    step(30)
    report("candle_in")
    s3 = score_line()
    if not ok_cnd or not bool(scene.success()[0]):
        print("SIM_GEN_SOLVE: FAILURE (candle did not settle inside / success not reached)",
              flush=True)
        hard_exit(1)
    if s3 < s2 - 1e-6 or abs(s3 - 1.0) > 1e-3:
        print("SIM_GEN_SOLVE: FAILURE (score not 1.0 at success)", flush=True)
        hard_exit(1)

    # ================= phase 4: PERSIST (>= 3.5 s, no intervention) ===========================
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
