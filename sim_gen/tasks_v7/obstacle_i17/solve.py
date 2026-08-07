"""Teleport solution for SkittleGalleryScene (sim_gen task `obstacle_i17`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY, in one write, ending in FREE SPACE:
  P1 — the blue ball is carried from its scatter spawn to a STAGING point on the RED
  lane's axis, 420 mm in front of the wall — open floor, outside the funnels, clear
  of the decoy. Everything after that goes through CONTACT DYNAMICS.
  P2 — a floating-hand throw: the hand-forces first LIFT the ball a few cm off the
  floor (vertical PD — a ball held in a hand does not spin), then accelerate it along
  the lane axis to ~1.2 m/s with a lateral PD holding the lane line, and are CUT
  ~180 mm before the wall, before the funnel throat. From there the shot is
  ballistic: the ball lands short of the mouth, funnels into the tunnel, crosses the
  cell under its own momentum, and topples the RED pin by impact. (The airborne
  carry is also a physics-API necessity: external wrenches are applied in a frozen
  body frame on this stack, so forcing a ROLLING ball precesses the push — a held,
  non-spinning ball keeps the force world-true.) The ball is never teleported into
  the tunnel or the cell; no force ever touches a pin; the white pin and the decoy
  are never touched at all.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.obstacle_i17.solve --headless [--seed N]
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
    env = ENVS.get("simgen.skittle_gallery")().build(num_envs=args.num_envs, device=device)
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

    def fix_pose() -> tuple[torch.Tensor, float]:
        fp = (scene.gallery.data.root_pos_w - scene.env_origins)[0]
        q = scene.gallery.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return fp, yaw

    def to_world(local_xy: tuple) -> tuple[float, float]:
        fp, yaw = fix_pose()
        cy, sy = math.cos(yaw), math.sin(yaw)
        return (float(fp[0]) + local_xy[0] * cy - local_xy[1] * sy,
                float(fp[1]) + local_xy[0] * sy + local_xy[1] * cy)

    def report(tag: str) -> None:
        bl = scene._fix_local(scene.ball.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} | ball_local=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) red_upz={float(scene._up_z(scene.red)[0]):+.3f} "
              f"red_z={float(scene._z_rel(scene.red)[0]):.3f} "
              f"white_upz={float(scene._up_z(scene.white)[0]):+.3f} "
              f"down={bool(scene.red_down()[0])} stand={bool(scene.white_standing()[0])} "
              f"in_cell={bool(scene.ball_in_red_cell()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_wrench() -> None:
        scene.ball.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    fp, fyaw = fix_pose()
    side = float(scene.red_side[0])
    ball0 = scene._fix_local(scene.ball.data.root_pos_w)[0]
    dec0 = scene._fix_local(scene.decoy.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): gallery=({float(fp[0]):+.3f},"
          f"{float(fp[1]):+.3f}) yaw={math.degrees(fyaw):+.1f}deg red_side={side:+.0f} "
          f"ball_local=({float(ball0[0]):+.3f},{float(ball0[1]):+.3f}) "
          f"decoy_local=({float(dec0[0]):+.3f},{float(dec0[1]):+.3f}) "
          f"d0={float(scene.d0[0]):.3f}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: ball TRANSPORT (teleport to staging) ------------------------
    # Staging: on the RED lane's axis, 420 mm in front of the outer face — open floor,
    # outside the funnel splays (which start at -140 mm), clear of the decoy spawn zone.
    stage_local = (side * c.lane_off, -0.42)
    wx, wy = to_world(stage_local)
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1], st[:, 2] = wx, wy, c.ball_r + 0.002
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins
    scene.ball.write_root_state_to_sim(st, all_ids)
    step(30)
    report("staged")
    s1 = print_score("P1 ball transport to staging")
    assert s1 >= s0 - 1e-6, "score decreased across ball transport"

    # ---------------- phase 2: throw (forces), free flight, impact -------------------------
    # Floating-hand throw. IMPORTANT: on this stack external wrenches are applied in a
    # frozen body frame, so a force on a ROLLING ball precesses with the roll. A hand
    # holds the ball, so we do the same: LIFT it off the floor first (vertical-only PD
    # while grounded — no torque, no spin), then, airborne with a frozen orientation,
    # accelerate along the lane axis with a velocity-regulated horizontal force +
    # lateral PD, and CUT at 180 mm before the wall (before the funnel throat) — from
    # there it is ballistic: land short of the mouth, roll in, strike.
    m_ball, g, v_des = c.ball_mass, 9.81, 1.2
    z_des = 0.042  # hover height ~ bore centre; well inside the 82 mm tunnel section
    fp, fyaw = fix_pose()
    cy, sy = math.cos(fyaw), math.sin(fyaw)
    fwd = torch.tensor([-sy, cy, 0.0], device=device)  # gallery-local +y in world
    lat = torch.tensor([cy, sy, 0.0], device=device)  # gallery-local +x in world
    launched = False
    for i in range(500):
        bl = scene._fix_local(scene.ball.data.root_pos_w)[0]
        if float(bl[1]) > -0.18:
            launched = True
            break
        z = float(scene._z_rel(scene.ball)[0])
        v_w = scene.ball.data.root_lin_vel_w[0]
        airborne = z > c.ball_r + 0.004
        # vertical: PD hold at the hover height (a support force, never downward)
        fz = m_ball * (g + 30.0 * (z_des - z) - 10.0 * float(v_w[2]))
        fz = max(0.0, min(3.0 * m_ball * g, fz))
        if airborne:
            v_fwd = float(torch.dot(v_w, fwd))
            v_lat = float(torch.dot(v_w, lat))
            e_lat = side * c.lane_off - float(bl[0])
            f_fwd = max(0.0, min(4.0, m_ball * 60.0 * (v_des - v_fwd)))
            f_lat = max(-1.5, min(1.5, m_ball * (60.0 * e_lat - 12.0 * v_lat)))
        else:
            f_fwd, f_lat = 0.0, 0.0  # grounded: lift only, keep the ball spin-free
        f_world = f_fwd * fwd + f_lat * lat
        f_world[2] = fz
        scene.ball.set_external_force_and_torque(
            f_world.view(1, 1, 3).expand(n, 1, 3).contiguous(), zero_wrench,
            env_ids=all_ids, is_global=True)
        env.step(no_action)
    clear_wrench()
    v_rel = float(scene.ball.data.root_lin_vel_w[0].norm())
    print(f"[solve] released (launched={launched}) at v={v_rel:.2f} m/s", flush=True)
    step(420)  # 3.5 s: funnel, tunnel, cell crossing, impact, topple, settle
    report("post-shot")
    s2 = print_score("P2 launch + ballistic strike + settle")
    assert s2 >= s1 - 1e-6, "score decreased across the shot"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the shot settled)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3.3 simulated seconds, no intervention) -----
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
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
    main()
