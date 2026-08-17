"""Teleport solution for PanDomeEggScene (sim_gen task
`libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf_i244`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY, in free space:
  - P1 carries the egg from the bowl to a hover 15 mm above the green pad's ring and
    RELEASES it: the seating itself is a free fall into the lip ring, retained by real
    contact (retried from hover if a bounce ever left it unseated).
  - P2 carries the pan to an inverted hover ~50 mm above its rim-down rest, centred on
    the pad, handle out the open front, and then LOWERS it with a velocity-regulated
    world-z support force (never a pose write): the rim meets the floor, contact takes
    the load, the force is cut and the pan settles rim-down over the egg on real
    contact dynamics. Nothing is ever written into a scoring pose: both objects reach
    their judged states by falling/being lowered under gravity onto real contacts.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf_i244.solve --headless [--seed N]
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
    env = ENVS.get("simgen.pan_dome_egg")().build(num_envs=args.num_envs, device=device)
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

    def yaw_of(q: torch.Tensor) -> float:
        return 2.0 * math.atan2(float(q[3]), float(q[0]))

    def report(tag: str) -> None:
        green = (scene.green_pad_w() - scene.env_origins)[0]
        egg = (scene.egg.data.root_pos_w - scene.env_origins)[0]
        pan = (scene.pan.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] {tag:12s} | egg=({float(egg[0]):+.3f},{float(egg[1]):+.3f},"
              f"{float(egg[2]):.3f}) pan_z={float(pan[2]):.3f} "
              f"up_z={float(scene._pan_up_z()[0]):+.2f} "
              f"on_pad={bool(scene.egg_on_pad()[0])} dome={bool(scene.pan_dome()[0])} "
              f"covered={bool(scene.covered()[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f} "
              f"| green=({float(green[0]):+.3f},{float(green[1]):+.3f})", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_force() -> None:
        scene.pan.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(60)
    sp = (scene.shelf.data.root_pos_w - scene.env_origins)[0]
    syaw = yaw_of(scene.shelf.data.root_quat_w[0])
    pyaw = yaw_of(scene.pads.data.root_quat_w[0])
    green = (scene.green_pad_w() - scene.env_origins)[0]
    egg0 = (scene.egg.data.root_pos_w - scene.env_origins)[0]
    tom0 = (scene.tomato.data.root_pos_w - scene.env_origins)[0]
    pan0 = (scene.pan.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): shelf=({float(sp[0]):+.3f},"
          f"{float(sp[1]):+.3f}) yaw={math.degrees(syaw):+.1f}deg "
          f"pads_yaw={math.degrees(pyaw):+.1f}deg "
          f"green=({float(green[0]):+.3f},{float(green[1]):+.3f}) "
          f"egg=({float(egg0[0]):+.3f},{float(egg0[1]):+.3f}) "
          f"tomato=({float(tom0[0]):+.3f},{float(tom0[1]):+.3f}) "
          f"pan=({float(pan0[0]):+.3f},{float(pan0[1]):+.3f}) "
          f"d0={float(scene.d0[0]):.3f}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: egg — teleport to hover, DROP into the ring -----------------
    # Transport only: the egg is placed in FREE AIR 15 mm above its seat height, centred
    # on the green pad; seating is the free fall into the lip ring, retained by contact.
    for attempt in range(4):
        green = scene.green_pad_w() - scene.env_origins
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = green[:, 0:2]
        st[:, 2] = c.egg_seat_z + 0.015
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.egg.write_root_state_to_sim(st, all_ids)
        step(90)  # free fall + real settle inside the ring (latches need a 20-streak)
        if bool(scene.egg_on_pad()[0]):
            break
        print(f"[solve] egg drop attempt {attempt} did not seat, retrying", flush=True)
    assert bool(scene.egg_on_pad()[0]), "egg failed to seat in the ring"
    step(30)
    assert bool(scene.seat_latch[0]), "seat latch did not take"
    report("egg-seated")
    s1 = print_score("P1 egg dropped into the ring")
    assert s1 >= s0 - 1e-6, "score decreased across egg seating"

    # ---------------- phase 2: pan — inverted hover, regulated contact lowering ------------
    # Transport only: one pose write puts the pan UPSIDE-DOWN in free air ~50 mm above
    # its rim-down rest height, centred over the pad, handle out the open front
    # (pan yaw = shelf yaw - pi/2). The descent is a velocity-regulated world-z support
    # force (never a pose write): rim meets floor, contact carries the load, force cut.
    green = scene.green_pad_w() - scene.env_origins
    pan_yaw = syaw - math.pi / 2  # handle world dir = (cos, sin) of this -> shelf -y (out)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:2] = green[:, 0:2]
    st[:, 2] = c.rim_rest_z + 0.050
    # q = qz(pan_yaw) x qx(pi) = (0, cos(yaw/2), sin(yaw/2), 0): cavity down.
    st[:, 4] = math.cos(pan_yaw / 2)
    st[:, 5] = math.sin(pan_yaw / 2)
    st[:, 0:3] += scene.env_origins
    scene.pan.write_root_state_to_sim(st, all_ids)
    step(2)
    report("pan-hover")

    m_pan, g = c.pan_mass, 9.81
    v_des = -0.05  # slow, controlled descent
    lowered = False
    for i in range(1200):
        z = float((scene.pan.data.root_pos_w - scene.env_origins)[0, 2])
        vz = float(scene.pan.data.root_lin_vel_w[0, 2])
        if z <= c.rim_rest_z + 0.003 and abs(vz) < 0.03:
            lowered = True
            break
        # world-z support force: gravity feedforward + P on vertical speed
        f_up = m_pan * (g + 4.0 * (v_des - vz))
        f_up = max(0.2 * m_pan * g, min(1.8 * m_pan * g, f_up))
        f_w = torch.tensor([0.0, 0.0, f_up], device=device).view(1, 1, 3).expand(n, 1, 3)
        scene.pan.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                                env_ids=all_ids, is_global=True)
        env.step(no_action)
    clear_force()
    print(f"[solve] lowering {'complete' if lowered else 'timed out (force cut)'}",
          flush=True)
    step(120)  # hands-off settle on real contacts
    report("pan-lowered")
    s2 = print_score("P2 pan lowered rim-down over the egg")
    assert s2 >= s1 - 1e-6, "score decreased across pan lowering"

    # ---------------- phase 3: hands off, settle, judge ------------------------------------
    step(120)
    report("settled")
    s3 = print_score("P3 settle")
    assert s3 >= s2 - 1e-6, "score decreased across settle"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after settle)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) -------
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
