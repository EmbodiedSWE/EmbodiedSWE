"""Teleport-solution for `open_washing_machine_i393` (scene `coinop_washer`).

Run: python -m simgen_tasks.open_washing_machine_i393.solve --headless [--seed N]

Phases (teleports = TRANSPORT ONLY; every load-bearing interaction is contact
dynamics or an applied wrench through the scene's post_step slots):
  1. PAY    — the brass token is teleported to FREE AIR ~30 mm above the coin
              tray pocket (readback of the live rocker pose), then FALLS in; its
              weight tips the rocker to the paid stop by gravity alone.
  2. OPEN   — a velocity-cascade force servo on the door body (door_f slot,
              KV*dt/m ~ 0.25 against the one-substep wrench delay) slides the
              live prismatic joint open; released slow near the stop.
  3. OUT    — a velocity servo on the ball (ball_f slot) rolls it through the
              exposed doorway window and off the sill onto the table.
  4. DELIVER— the ball (now out in the open) is teleported to free air above the
              basket (readback), falls in and settles.
  5. PERSIST— >= 3 simulated seconds hands-off (all wrench slots asserted zero),
              success re-checked, then `SIM_GEN_SOLVE: SUCCESS`.

Prints `SIM_GEN_SCORE <s>` at each phase boundary (non-decreasing by design of
the latch ladder).
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
    env = ENVS.get("simgen.coinop_washer")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply

    env.reset(seed=args.seed)  # seed AFTER build
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        bp = scene.ball.data.root_pos_w[0] - scene.env_origins[0]
        print(f"[solve] {tag:10s} | door={float(scene.door_open()[0]) * 1000:6.1f}mm "
              f"rocker={math.degrees(float(scene.rocker_angle()[0])):+6.1f}deg "
              f"tok_tray={bool(scene.token_in_tray()[0])} paid={bool(scene.paid_now()[0])} "
              f"ball=({float(bp[0]):+.3f},{float(bp[1]):+.3f},{float(bp[2]):+.3f}) "
              f"in_basket={bool(scene.ball_in_basket()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def teleport(body, pos_w: torch.Tensor) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3] = 1.0
        body.write_root_state_to_sim(st, all_ids)

    step(60)  # clean settle
    report("settled")
    print_score("baseline")

    # ---- phase 1: PAY — drop the token into the live tray (gravity does the rest) ----
    tray_local = torch.tensor([[0.0, c.tray_by, c.tray_floor_top]], device=device).repeat(n, 1)
    tray_w = scene.rocker.data.root_pos_w + quat_apply(scene.rocker.data.root_quat_w, tray_local)
    drop_pos = tray_w.clone()
    drop_pos[:, 2] += 0.030
    # free-air assert: the drop point is beside the housing, above every tray wall
    assert float(drop_pos[0, 1] - scene.env_origins[0, 1]) > c.housing_width / 2, "drop not clear"
    teleport(scene.token, drop_pos)
    step(30)
    # wait for the rocker to tip and ring down on the paid stop
    for _ in range(20):
        step(24)
        if bool(scene.paid_now()[0]) and \
                float(scene.rocker.data.root_ang_vel_w.norm(dim=-1)[0]) < 0.3:
            break
    report("paid")
    if not bool(scene.paid_now()[0]) or not bool(scene.token_in_tray()[0]):
        print("SIM_GEN_SOLVE: FAIL (token did not pay the lever)", flush=True)
        os._exit(1)
    print_score("paid")

    # ---- phase 2: OPEN — velocity-cascade force servo on the door joint ----
    kv, v_max, f_cap = 12.0, 0.10, 8.0  # kv*dt/m = 12/(120*0.4) = 0.25
    target = c.door_stroke - 0.008
    for i in range(900):
        q = scene.door_open()
        v = scene.door.data.root_lin_vel_w[:, 1]
        v_des = torch.clamp(6.0 * (target - q), -v_max, v_max)
        scene.door_f = torch.clamp(kv * (v_des - v), -f_cap, f_cap)
        step(1)
        if float(q[0]) > target - 0.004 and abs(float(v[0])) < 0.03:
            break
    scene.door_f = torch.zeros(n, device=device)
    step(30)
    report("opened")
    if float(scene.door_open()[0]) < c.open_thresh:
        print("SIM_GEN_SOLVE: FAIL (door did not open)", flush=True)
        os._exit(1)
    print_score("opened")

    # ---- phase 3: OUT — roll the ball through the doorway window onto the table ----
    wy = 0.5 * (c.window_y[0] + c.window_y[1])  # centre of the exposed window
    waypoints = [(c.housing_pos[0] + 0.04, wy), (c.face_x + 0.10, wy)]
    kv_b, sp, f_cap_b = 3.0, 0.18, 1.2  # kv*dt/m = 3/(120*0.06) = 0.42
    for wx, wyy in waypoints:
        tgt = torch.tensor([[wx, wyy]], device=device).repeat(n, 1)
        for i in range(1200):
            p = scene.ball.data.root_pos_w[:, 0:2] - scene.env_origins[:, 0:2]
            d = tgt - p
            dist = d.norm(dim=-1, keepdim=True)
            if float(dist[0]) < 0.020:
                break
            v_des = sp * d / dist.clamp(min=1e-6)
            v = scene.ball.data.root_lin_vel_w[:, 0:2]
            f = torch.clamp(kv_b * (v_des - v), -f_cap_b, f_cap_b)
            scene.ball_f = torch.cat([f, torch.zeros(n, 1, device=device)], dim=1)
            step(1)
    # active brake: a rolling sphere is nearly undamped — servo v_des=0 to a stop
    for _ in range(360):
        v = scene.ball.data.root_lin_vel_w[:, 0:2]
        if float(v.norm(dim=-1)[0]) < 0.04:
            break
        f = torch.clamp(kv_b * (-v), -f_cap_b, f_cap_b)
        scene.ball_f = torch.cat([f, torch.zeros(n, 1, device=device)], dim=1)
        step(1)
    scene.ball_f = torch.zeros(n, 3, device=device)
    step(30)
    report("ball out")
    if not bool(scene.ball_out()[0]):
        print("SIM_GEN_SOLVE: FAIL (ball never left the chamber)", flush=True)
        os._exit(1)
    print_score("ball out")

    # ---- phase 4: DELIVER — free-air drop into the basket (readback) ----
    bk = scene.basket.data.root_pos_w.clone()
    bk[:, 2] += c.basket_wall_t + c.basket_wall_h + c.ball_r + 0.020
    # free-air assert: drop point is over the open basket, far from the housing
    assert float(bk[0, 0] - scene.env_origins[0, 0]) > c.face_x + 0.05, "drop not clear"
    teleport(scene.ball, bk)
    for _ in range(20):
        step(24)
        if bool(scene.success()[0]):
            break
    report("delivered")
    print_score("delivered")
    if not bool(scene.success()[0]):
        print("SIM_GEN_SOLVE: FAIL (no success after delivery)", flush=True)
        os._exit(1)

    # ---- phase 5: PERSIST — hands-off >= 3 simulated seconds ----
    assert float(scene.door_f.abs().max()) == 0.0
    assert float(scene.ball_f.abs().max()) == 0.0
    assert float(scene.rocker_tau.abs().max()) == 0.0
    held = True
    for _ in range(10):
        step(40)  # 10 * 40 / 120 = 3.33 s
        held = held and bool(scene.success()[0])
    report("persist")
    print_score("persist")
    if held and bool(scene.success()[0]):
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        code = 0
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)
        code = 1

    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        app.close()
    except Exception:
        pass
    os._exit(code)


if __name__ == "__main__":
    main()
