"""Teleport solution for BladeGuardScene (sim_gen task
`put_knife_in_knife_block_i312`) — the task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): ONE pose write moves the GREEN sleeve across free space from
   its table slot to a hover with its bottom face 2 cm ABOVE the blade tip, centered
   on the blade axis and yaw-matched to the vise's randomized yaw — touching nothing,
   the blade NOT inside the channel (the solve asserts tip-entry is False and no
   seat/success gate is satisfied by the teleport). The knife/vise is NEVER teleported
   outside reset re-pose, and the decoy is never touched.
2. THREADING (contact dynamics — the core interaction; teleporting the sleeve seated
   would bypass the task and is never done): a regulated vertical force (velocity-
   servoed at -0.25 m/s, |F| <= 2.5 N) lowers the sleeve so the fixed blade enters the
   descending channel and guides it through REAL contact for ~8.5 cm of overlap; the
   force is CUT ~2.5 cm above the seat, and the final drop onto the vise shoulders
   and settling are pure gravity + contact. The solve asserts the tip ENTERED the
   channel during descent (contact threading demonstrably happened) and that the
   sleeve settled seated with the blade fully contained.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.put_knife_in_knife_block_i312.solve --headless [--seed N]
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
    env = ENVS.get("simgen.blade_guard")().build(num_envs=args.num_envs, device=device)
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

    def loc(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        p_slv, p_dcy = loc(scene.sleeve), loc(scene.decoy)
        tip = scene._tip_local()[0]
        print(f"[solve] {tag:12s} | sleeve=({float(p_slv[0]):+.3f},{float(p_slv[1]):+.3f},"
              f"{float(p_slv[2]):.3f}) decoy=({float(p_dcy[0]):+.3f},"
              f"{float(p_dcy[1]):+.3f},{float(p_dcy[2]):.3f}) "
              f"tip_local=({float(tip[0]):+.3f},{float(tip[1]):+.3f},{float(tip[2]):+.3f}) "
              f"in_ch={bool(scene.tip_in_channel()[0])} "
              f"frac={float(scene.insertion_frac()[0]):.2f} "
              f"yaw_err={float(scene.yaw_err_deg()[0]):.1f}deg "
              f"seated={bool(scene.seated_config()[0])} "
              f"lat=[l{int(scene._lifted[0])} e{int(scene._entered[0])} "
              f"d{float(scene._depth_max[0]):.2f} s{int(scene._seated[0])}] "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    p_slv, p_dcy = loc(scene.sleeve), loc(scene.decoy)
    vx, vy = float(scene._vise_xy[0, 0]), float(scene._vise_xy[0, 1])
    vyaw = float(scene._vise_yaw[0])
    print(f"[solve] layout readback (seed {args.seed}): sleeve=({float(p_slv[0]):+.3f},"
          f"{float(p_slv[1]):+.3f}) decoy=({float(p_dcy[0]):+.3f},{float(p_dcy[1]):+.3f}) "
          f"vise=({vx:+.3f},{vy:+.3f}) vise_yaw={math.degrees(vyaw):+.1f}deg "
          f"tip_z={c.tip_z:.3f}", flush=True)
    report("reset")
    assert float(p_slv[2]) < 0.03 and float(p_dcy[2]) < 0.03, \
        "sleeve/decoy did not settle standing on the table"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.05, "score not ~0 at reset"

    # ---------------- phase 1: TRANSPORT the sleeve (one free-space teleport) ---------------
    # Endpoint: bottom face 2 cm ABOVE the blade tip, centered on the blade axis and
    # yaw-matched — free space, touching nothing; the blade is NOT in the channel and
    # no seat gate is satisfied (the teleport does none of the task's insertion).
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1] = vx, vy
    st[:, 2] = c.tip_z + 0.020
    st[:, 3] = math.cos(vyaw / 2)
    st[:, 6] = math.sin(vyaw / 2)
    st[:, 0:3] += scene.env_origins
    scene.sleeve.write_root_state_to_sim(st, all_ids)
    report("hover")
    assert not bool(scene.tip_in_channel()[0]), \
        "hover must leave the blade OUTSIDE the channel (teleport is transport only)"
    assert not bool(scene.seated_config()[0]) and not bool(scene.success()[0]), \
        "hover must not satisfy the seat gate or success"
    s1 = print_score("P1 transport: sleeve hovering above the blade tip")
    assert s1 >= s0 - 1e-6, "score decreased across transport"

    # ---------------- phase 2: THREADING (contact dynamics) ---------------------------------
    # Velocity-servoed descent at -0.25 m/s (|F| <= 2.5 N): the blade enters the
    # channel and guides the sleeve through real contact. Cut ~2.5 cm above the seat;
    # the final drop onto the shoulders and settling are pure gravity + contact.
    mg = c.sleeve_mass * 9.81
    entered_during_descent = False
    for _ in range(900):
        z = float(loc(scene.sleeve)[2])
        if z <= c.shoulder_z + 0.025:
            break
        vz = float(scene.sleeve.data.root_lin_vel_w[0, 2])
        fz = max(-2.5, min(2.5, mg + 8.0 * (-0.25 - vz)))
        f_w = torch.tensor([0.0, 0.0, fz], device=device).view(1, 1, 3).expand(n, 1, 3)
        scene.sleeve.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                                   env_ids=all_ids, is_global=True)
        env.step(no_action)
        entered_during_descent = entered_during_descent or bool(scene.tip_in_channel()[0])
    scene.sleeve.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    report("descended")
    assert entered_during_descent, \
        "blade tip never entered the channel during the servoed descent"
    quiet = 0
    for _ in range(600):
        env.step(no_action)
        ok_now = bool(scene.seated_config()[0]) and bool(scene._still()[0])
        quiet = quiet + 1 if ok_now else 0
        if quiet >= 30:
            break
    report("seated")
    assert bool(scene.seated_config()[0]) and bool(scene._still()[0]), \
        "sleeve did not settle seated on the shoulders with the blade contained"
    s2 = print_score("P2 blade threaded through the channel, sleeve seated (contact)")
    assert s2 >= s1 - 1e-6, "score decreased across the threading"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after seating)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3 simulated seconds, no intervention) --------
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
