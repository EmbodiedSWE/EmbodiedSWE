"""Teleport solution for BellHerdScene (sim_gen task `track_bowl_i27`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY, always ending in FREE SPACE or a non-contact
hover; every load-bearing interaction goes through CONTACT DYNAMICS:
  P1 — CAGE: the bell is carried from its spawn to a hover CENTRED over the red ball
  (mouth 25 mm above the surface — the 75 mm mouth clears the 42.5 mm ball radius,
  so the hover intersects nothing) and then simply DROPS under gravity onto the
  board, enclosing the ball. Nothing is bypassed: caging is geometric envelopment,
  and everything that follows is contact.
  P2 — HERD: a floating-hand drag. External forces servo the bell's own body along a
  straight board-frame path from the caging point to the pocket centre; the bell's
  skirt pushes the caged, rolling ball across the surface through friction contact.
  (Physics-API note: on this stack external wrenches are applied in a rotating body
  frame, so the desired WORLD force is pre-rotated by q_ref * q_now^-1 each step;
  the bell is also yaw/tilt-stable — low COM, damped — so the correction stays tiny.)
  When the cage crosses the pocket the ball falls through the hole under gravity —
  the delivery is pure contact + gravity; the ball is NEVER teleported into or even
  toward the pocket, and no force ever touches either ball.
  P3 — REVEAL: with the ball settled in the pocket, forces stop; the now-empty bell
  is teleported straight up off the board (free space) and parked on the far side,
  clear of the pocket, then everything settles.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.track_bowl_i27.solve --headless [--seed N]
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
    env = ENVS.get("simgen.bell_herd")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def board_pose() -> tuple[torch.Tensor, float]:
        fp = (scene.board.data.root_pos_w - scene.env_origins)[0]
        q = scene.board.data.root_quat_w[0]
        return fp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def to_world(lx: float, ly: float) -> tuple[float, float]:
        fp, yaw = board_pose()
        cy, sy = math.cos(yaw), math.sin(yaw)
        return (float(fp[0]) + lx * cy - ly * sy, float(fp[1]) + lx * sy + ly * cy)

    def report(tag: str) -> None:
        bl = scene._board_local(scene.red.data.root_pos_w)[0]
        el = scene._board_local(scene.bell.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} | red_local=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):+.3f}) bell_local=({float(el[0]):+.3f},{float(el[1]):+.3f},"
              f"{float(el[2]):+.3f}) caged={bool(scene.caged()[0])} "
              f"in_pocket={bool(scene.in_pocket(scene.red)[0])} "
              f"blue_in={bool(scene.in_pocket(scene.blue)[0])} "
              f"bell_clear={bool(scene.bell_is_clear()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_wrench() -> None:
        scene.bell.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def quat_rotate_single(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        return quat_apply(q.view(1, 4), v.view(1, 3)).view(3)

    def quat_conj(q: torch.Tensor) -> torch.Tensor:
        out = q.clone()
        out[1:] = -out[1:]
        return out

    def quat_mul_single(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_mul

        return quat_mul(a.view(1, 4), b.view(1, 4)).view(4)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    fp, fyaw = board_pose()
    side = float(scene.red_side[0])
    red0 = scene._board_local(scene.red.data.root_pos_w)[0]
    blue0 = scene._board_local(scene.blue.data.root_pos_w)[0]
    bell0 = scene._board_local(scene.bell.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): board=({float(fp[0]):+.3f},"
          f"{float(fp[1]):+.3f}) yaw={math.degrees(fyaw):+.1f}deg side={side:+.0f} "
          f"red_local=({float(red0[0]):+.3f},{float(red0[1]):+.3f}) "
          f"blue_local=({float(blue0[0]):+.3f},{float(blue0[1]):+.3f}) "
          f"bell_local=({float(bell0[0]):+.3f},{float(bell0[1]):+.3f}) "
          f"d0={float(scene.d0[0]):.3f}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: bell TRANSPORT + gravity drop = CAGE ------------------------
    # Hover the bell centred on the red ball, mouth 25 mm above the surface: the open
    # 150 mm mouth around an 85 mm ball intersects nothing. Then let it fall. The cage
    # closes by gravity + rim contact with the board.
    ball_loc = scene._board_local(scene.red.data.root_pos_w)[0]
    wx, wy = to_world(float(ball_loc[0]), float(ball_loc[1]))
    _fp, byaw = board_pose()
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1], st[:, 2] = wx, wy, c.surface_h + 0.025
    st[:, 3], st[:, 6] = math.cos(byaw / 2), math.sin(byaw / 2)
    st[:, 0:3] += scene.env_origins
    scene.bell.write_root_state_to_sim(st, all_ids)
    step(60)  # free fall 25 mm + rim impact + settle
    report("caged")
    s1 = print_score("P1 bell transport + gravity cage")
    assert s1 >= s0 - 1e-6, "score decreased across the caging phase"
    if not bool(scene.caged()[0]):
        print("SIM_GEN_SOLVE: FAIL (cage did not close)", flush=True)
        os._exit(1)

    # ---------------- phase 2: HERD — force-drag the loaded bell to the pocket -------------
    # Straight board-frame path from the caging point to the pocket centre, tracked by
    # a moving carrot at ~0.10 m/s. Forces act on the BELL only; the ball is pushed by
    # the skirt through contact. World-frame force is pre-rotated by q_ref * q_now^-1
    # (rotating-body-frame wrench API), and a weak upright torque servo keeps the
    # damped bell from heeling while it slides.
    q_ref = scene.bell.data.root_quat_w[0].clone()
    start = scene._board_local(scene.bell.data.root_pos_w)[0][:2].clone()
    # Aim PAST the pocket centre: sliding friction gives the PD a steady-state lag
    # (~mu*g/kp), and the ball leaning into the hole rests against the bell's FRONT
    # inner wall — the bell must advance beyond centre for that wall to release it.
    # Overshoot is safe: the loop breaks the moment the ball is in the pocket, and
    # the bell's rim can never fall in.
    goal = torch.tensor([c.pit_x + 0.05, 0.0], device=device)
    path = goal - start
    path_len = float(path.norm())
    path_dir = path / max(path_len, 1e-6)
    v_carrot = 0.10
    dt = 1.0 / 120.0
    m_bell = c.bell_mass
    kp, kd = 240.0, 20.0  # per-kg gains; steady-state friction lag ~mu*g/kp ~ 1.6 cm
    kr, krd = 0.8, 0.08  # upright torque servo
    delivered_at = None
    ez = torch.tensor([0.0, 0.0, 1.0], device=device)
    fp2, byaw2 = board_pose()
    cy2, sy2 = math.cos(byaw2), math.sin(byaw2)

    def local_dir_to_world(d: torch.Tensor) -> torch.Tensor:
        return torch.tensor([float(d[0]) * cy2 - float(d[1]) * sy2,
                             float(d[0]) * sy2 + float(d[1]) * cy2, 0.0], device=device)

    for i in range(1400):
        if bool(scene.in_pocket(scene.red)[0]):
            delivered_at = i
            break
        prog = min(path_len, v_carrot * i * dt)
        carrot = start + path_dir * prog
        bell_loc = scene._board_local(scene.bell.data.root_pos_w)[0]
        err_l = carrot - bell_loc[:2]
        v_w = scene.bell.data.root_lin_vel_w[0]
        f_w = local_dir_to_world(err_l) * kp * m_bell
        f_w = f_w - v_w * kd * m_bell
        f_w[2] = 0.0
        f_w = f_w.clamp(-8.0, 8.0)
        # upright servo torque: tau = kr * (u x ez) - krd * w  (world frame)
        q_now = scene.bell.data.root_quat_w[0]
        u = quat_rotate_single(q_now, ez)
        tau_w = kr * torch.linalg.cross(u, ez) - krd * scene.bell.data.root_ang_vel_w[0]
        # pre-rotate world wrench into the rotating wrench frame: q_ref * q_now^-1
        q_corr = quat_mul_single(q_ref, quat_conj(q_now))
        f_arg = quat_rotate_single(q_corr, f_w)
        t_arg = quat_rotate_single(q_corr, tau_w)
        scene.bell.set_external_force_and_torque(
            f_arg.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            t_arg.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            env_ids=all_ids, is_global=True)
        env.step(no_action)
    clear_wrench()
    print(f"[solve] herd loop ended (delivered_at={delivered_at})", flush=True)
    step(90)  # ball falls to the pocket floor and settles; bell stops
    report("delivered")
    s2 = print_score("P2 herd + gravity delivery")
    assert s2 >= s1 - 1e-6, "score decreased across the herding phase"
    if not bool(scene.in_pocket(scene.red)[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (ball not delivered to the pocket)", flush=True)
        os._exit(1)

    # ---------------- phase 3: REVEAL — park the empty bell clear --------------------------
    # The ball rests in the pocket; nothing touches the bell's interior any more.
    # Lift the empty bell straight up off the board (free space) and set it down on
    # the far side of the field, clear of the pocket; then hands off.
    # Park on the RED ball's original side (+side): that half is now vacated, whereas
    # the blue ball still sits near (-0.16, -0.13*side) and must stay untouched.
    wx, wy = to_world(-0.22, 0.20 * side)
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1], st[:, 2] = wx, wy, c.surface_h + 0.06
    _fp3, byaw3 = board_pose()
    st[:, 3], st[:, 6] = math.cos(byaw3 / 2), math.sin(byaw3 / 2)
    st[:, 0:3] += scene.env_origins
    scene.bell.write_root_state_to_sim(st, all_ids)
    step(120)  # drop 60 mm, settle
    report("revealed")
    s3 = print_score("P3 bell parked clear")
    assert s3 >= s2 - 1e-6, "score decreased across the reveal phase"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the bell was parked)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, no intervention) -----
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s3 - 1e-6
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
