"""Teleport solution for GateGoalScene (sim_gen task `hockey_i325`) — the task's
legitimacy certificate.

The task's load-bearing interactions and how they are executed:

1. EXTRACT THE GATE (contact dynamics — never teleported): the yellow board is CAPTIVE
   in its vertical channel (post pairs fore/aft, end caps sideways); the only way out
   is a ~13 cm straight vertical slide. A velocity-limited vertical force at the
   board's CoM (the applied-wrench emulation of the arm's pinch-and-lift on the
   board's exposed top edge) slides it up THROUGH the channel under contact — every
   millimetre of the extraction, including any lean-against-the-posts friction, is
   physics. Only once the board's bottom edge has verifiably cleared the channel posts
   (readback) is it a free body in open air; ONE pose write then parks it flat on the
   floor well clear of the goal — pure free-space transport of an already-freed
   object. Nothing is bypassed: the channel was exited through the channel.
2. TRANSPORT (teleport): one pose write stages the white ball 16 cm OUTSIDE the mouth
   plane on the goal axis, on the floor (verifiably outside the chamber — asserted).
   Staging satisfies no rubric clause beyond the mouth-approach ramp.
3. ROLL IN THROUGH THE MOUTH (contact dynamics — never teleported): a velocity-limited
   horizontal force at the ball's CoM (the fingertip-push emulation) rolls the ball
   across the floor, between the channel posts and through the mouth. The force is CUT
   the moment the ball's CoM passes the "inside" line (mouth plane + r + 15 mm); it
   coasts, meets the back wall if it gets there, and settles hands-off. From the force
   cut to the verdict nothing touches the ball.

The black distractor ball is never touched. Prints `SIM_GEN_SCORE <score>` at each
phase boundary (non-decreasing: the scene's credit is latched), then holds HANDS-OFF
for >= 3.3 simulated seconds after success() first turns True and prints
`SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.<task>.solve --headless [--seed N]
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
    env = ENVS.get("simgen.gate_goal")().build(num_envs=args.num_envs, device=device)
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

    from isaaclab.utils.math import quat_apply

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def w_loc() -> torch.Tensor:
        return scene._goal_local(scene.white)[0]

    def b_loc() -> torch.Tensor:
        return scene._goal_local(scene.board)[0]

    def report(tag: str) -> None:
        wl, bl = w_loc(), b_loc()
        print(f"[solve] {tag:12s} | white_loc=({float(wl[0]):+.3f},{float(wl[1]):+.3f},"
              f"{float(wl[2]):.3f}) board_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) in_gate={bool(scene._board_in_gate()[0])} "
              f"gate_out={bool(scene._gate_out[0])} app={float(scene._app_max[0]):.3f} "
              f"inside={bool(scene._inside(scene.white)[0])} "
              f"black_in={bool(scene._inside(scene.black)[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    g = (scene.goal.data.root_pos_w - scene.env_origins)[0]
    gq = scene.goal.data.root_quat_w[0]
    g_yaw = 2.0 * math.atan2(float(gq[3]), float(gq[0]))
    w0 = (scene.white.data.root_pos_w - scene.env_origins)[0]
    k0 = (scene.black.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"goal=({float(g[0]):+.3f},{float(g[1]):+.3f}) yaw={math.degrees(g_yaw):+.1f}deg "
          f"white=({float(w0[0]):+.3f},{float(w0[1]):+.3f}) "
          f"black=({float(k0[0]):+.3f},{float(k0[1]):+.3f})", flush=True)
    report("reset")
    assert bool(scene._board_in_gate()[0]), "board did not spawn seated in the gate"
    assert not bool(scene._inside(scene.white)[0]), "white ball spawned inside"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- phase 1: EXTRACT THE GATE (contact dynamics) --------------------------
    # Velocity-limited vertical CoM force slides the captive board up its channel; the
    # force is the wrench emulation of the arm's pinch on the exposed top edge. The
    # extraction is DONE only when the readback shows the bottom edge above the posts.
    up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
    grav = c.board_mass * 9.81
    f_lift, v_des = grav + 1.6, 0.30
    clear_z = c.board_h / 2 + c.post_h + 0.012  # CoM height when the bottom edge clears
    extracted = False
    for i in range(1200):
        bl = b_loc()
        if float(bl[2]) > clear_z:
            extracted = True
            break
        vz = float(scene.board.data.root_lin_vel_w[0, 2])
        fz = f_lift if vz < v_des else grav
        scene.board.set_external_force_and_torque((up * fz).view(n, 1, 3).contiguous(),
                                                  zero_wrench, env_ids=all_ids,
                                                  is_global=True)
        env.step(no_action)
        if i and i % 360 == 0:  # jammed against the channel: pull a little harder
            f_lift = min(f_lift + 0.8, grav + 5.0)
            print(f"[solve] slow extraction at z={float(bl[2]):.3f} -> "
                  f"f_lift={f_lift:.1f} N", flush=True)
    scene.board.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    report("extracted")
    assert extracted, "vertical pull never carried the board's bottom edge clear of the channel"
    assert bool(scene._gate_out[0]), "extraction did not latch gate_out"

    # The board is now a free body in open air above the channel: ONE pose write parks
    # it flat on the floor at the free patch (free-space transport, nothing bypassed).
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1] = c.park_spot[0], c.park_spot[1]
    st[:, 2] = c.board_t / 2 + 0.004
    h = math.pi / 4  # lying flat: rotate 90 deg about y (local z -> world x)
    st[:, 3], st[:, 5] = math.cos(h), math.sin(h)
    st[:, 0:3] += scene.env_origins
    scene.board.write_root_state_to_sim(st, all_ids)
    step(60)
    report("parked")
    assert not bool(scene._board_in_gate()[0]), "parked board still reads as in the gate"
    s1 = print_score("P1 gate extracted through the channel, parked clear")
    assert s1 >= s0 - 1e-6, "score decreased across extraction"

    # ---------------- phase 2: TRANSPORT to the staging pose (teleport, outside) ------------
    # One pose write: white ball to 16 cm outside the mouth plane on the goal axis.
    g_pos = scene.goal.data.root_pos_w
    g_quat = scene.goal.data.root_quat_w
    stage_loc = torch.tensor([-0.16, 0.0, c.ball_r + 0.002], device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = g_pos + quat_apply(g_quat, stage_loc)
    st[:, 3] = 1.0
    scene.white.write_root_state_to_sim(st, all_ids)
    step(30)
    report("staged")
    assert not bool(scene._inside(scene.white)[0]), \
        "staging pose is already inside the chamber (teleport must stay outside)"
    s2 = print_score("P2 white ball transported to the staging pose (outside the mouth)")
    assert s2 >= s1 - 1e-6, "score decreased across transport"

    # ---------------- phase 3: ROLL IN THROUGH THE MOUTH (contact dynamics) -----------------
    # Velocity-limited CoM force along the goal axis + a small lateral centering term;
    # cut the force the moment the CoM passes the inside line — coasting and the back
    # wall finish the job.
    push_dir = quat_apply(g_quat, torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
    lat_dir = quat_apply(g_quat, torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3))
    f_push, v_des = 0.9, 0.22
    best_u, last_bump = -1.0, 0
    entered = False
    for i in range(1500):
        wl = w_loc()
        u, v = float(wl[0]), float(wl[1])
        if u > c.in_u_min + 0.012:
            entered = True
            break
        vel = scene.white.data.root_lin_vel_w[0]
        u_vel = float((vel * push_dir[0]).sum())
        v_vel = float((vel * lat_dir[0]).sum())
        f_axis = f_push if u_vel < v_des else 0.0
        f_lat = max(-0.5, min(0.5, -5.0 * v - 1.2 * v_vel))
        f_w = (push_dir * f_axis + lat_dir * f_lat).view(n, 1, 3)
        scene.white.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                                 env_ids=all_ids, is_global=True)
        env.step(no_action)
        if u > best_u + 0.003:
            best_u, last_bump = u, i
        elif i - last_bump > 240:  # stalled: push harder
            f_push = min(f_push + 0.4, 4.0)
            last_bump = i
            print(f"[solve] stall at u={u:+.3f} -> f_push={f_push:.1f} N", flush=True)
    scene.white.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    report("entered")
    assert entered, "push never carried the white ball past the inside line"

    # hands-off: coast and settle inside
    quiet = 0
    for _ in range(900):
        env.step(no_action)
        still = (float(scene.white.data.root_lin_vel_w[0].norm()) < 0.03
                 and float(scene.white.data.root_ang_vel_w[0].norm()) < 0.4)
        quiet = quiet + 1 if (still and bool(scene._inside(scene.white)[0])) else 0
        if quiet >= 30:
            break
    report("settled")
    assert bool(scene._inside(scene.white)[0]), "white ball did not settle inside the chamber"
    assert not bool(scene._inside(scene.black)[0]), "black ball ended up inside (untouched?!)"
    s3 = print_score("P3 rolled through the mouth, settled inside")
    assert s3 >= s2 - 1e-6, "score decreased across the roll-in"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after settling)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, no intervention) ------
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
