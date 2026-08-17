"""Teleport solution for WeighStationScene (sim_gen task `handover_i124`) — the task's
legitimacy certificate.

Teleports are TRANSPORT ONLY (relocating a body the solver is already holding in free
air, always released into free air above open-top receptacles). Every load-bearing
interaction is gravity + contact:

1. PERCEPTION: the station pose, WHICH parcel stands on the green shipment pad, and
   therefore its size class — the mass in units, k in {1,2,3} — are read back from the
   episode state (station-local pad test), never hard-coded. The readback is asserted
   against the scene's own k.
2. DELIVER (gravity + contact): the shipment parcel is teleported from the pad to free
   air 3 cm above the RED cradle's rest pose and RELEASED with the beam's orientation.
   Gravity drops it in; the cradle floor and walls seat it. The unbalanced beam keels
   toward its stop — expected, and exactly why carrying alone cannot succeed.
3. BALLAST (gravity + contact): one 0.15 kg unit counterweight per required unit is
   teleported to free air just above its BLUE rack pocket (release point = live
   pocket-centre WORLD position + 12 mm straight up, faces beam-aligned, so the
   world-vertical fall lands centred in the tilted pocket; wall clearance at the
   ±20 deg stop stays positive). Each drop is seated by contact; the beam swings back
   toward level as the mass arithmetic closes. Overdamped pendulum (slow pole
   K/b ~ 0.5 /s) — settling is waited out, never forced.
4. VERIFY: hands off, success() (cradled + LEVEL within 6 deg + exactly k in rack +
   clean beam + settled + finite) must hold and keep holding >= 3.3 more simulated
   seconds before SIM_GEN_SOLVE: SUCCESS prints.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the rubric
latches: 0.30 parcel-ever-cradled + 0.30 required-ballast fraction, 1.0 iff success).

Run (forge): python -u -m simgen_tasks.handover_i124.solve --headless [--seed N]
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
    env = ENVS.get("simgen.weigh_station")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        in_rack, on_beam = scene._weight_flags()
        print(f"[solve] {tag:16s} | tilt={math.degrees(float(scene.tilt()[0])):+6.1f}deg "
              f"rate={float(scene.rate_fd[0]):+.3f} "
              f"in_rack={int(in_rack[0].sum())}/{int(scene.k_units[0])} "
              f"cradled={bool(scene.in_cradle()[0])} "
              f"level={bool(scene.level()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ----- phase 0: settle + perception ----------------------------------------------------------
    step(120)
    report("settled start")
    # perception: WHICH parcel stands on the green pad (station-local box test).
    # Size class IS the mass in units — that is the whole point of the task.
    st_p = scene.station.data.root_pos_w
    st_q = scene.station.data.root_quat_w
    px, py = c.pad_center
    ship = torch.full((n,), -1, dtype=torch.long, device=device)
    for j in range(3):
        loc = quat_apply_inverse(st_q, scene.parcels[j].data.root_pos_w - st_p)
        on_pad = ((loc[:, 0] - px).abs() < 0.06) & ((loc[:, 1] - py).abs() < 0.06) \
            & (loc[:, 2] > 0.0) & (loc[:, 2] < 0.10)
        ship = torch.where(on_pad, torch.full_like(ship, j), ship)
    assert bool((ship >= 0).all()), "exactly one parcel must stand on the pad"
    k_perceived = ship + 1  # 46/58/66 mm -> 1/2/3 units (one shared density)
    assert bool((k_perceived == scene.k_units).all()), "pad readback must match episode k"
    print(f"[solve] perception: shipment = parcel_{int(ship[0])} "
          f"({1000 * c.parcel_sizes[int(ship[0])]:.0f} mm -> k = {int(k_perceived[0])} "
          f"unit(s) of ballast)", flush=True)
    s0 = print_score("start")
    assert s0 < 0.05, f"fresh episode must score ~0, got {s0}"

    # ----- phase 1: DELIVER — drop the shipment into the red cradle (beam level) -----------------
    beam_p = scene.beam.data.root_pos_w
    beam_q = scene.beam.data.root_quat_w
    for j in range(3):
        ids = torch.nonzero(ship == j, as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            continue
        rest = c.floor_z1 + c.parcel_sizes[j] / 2
        local = torch.tensor([c.arm_r, 0.0, rest], device=device).expand(ids.numel(), 3)
        st = torch.zeros(ids.numel(), 13, device=device)
        st[:, 0:3] = beam_p[ids] + quat_apply(beam_q[ids], local)
        st[:, 2] += 0.030  # release 3 cm straight up (world) — free air above the cradle
        st[:, 3:7] = beam_q[ids]  # faces parallel to the cradle walls
        scene.parcels[j].write_root_state_to_sim(st, ids)
    step(300)  # drop + the unballasted beam keels onto its stop (overdamped swing)
    report("delivered")
    assert bool(scene.in_cradle().all()), "shipment must be seated in the cradle"
    s1 = print_score("parcel in cradle")
    assert s1 >= 0.29, f"cradle latch must pay, got {s1}"
    assert s1 >= s0, "score must not decrease"

    # ----- phase 2: BALLAST — seat exactly k unit weights in the rack pockets --------------------
    # Release point = live pocket-centre WORLD position + 12 mm straight up: the
    # world-vertical fall lands centred in the (tilted) pocket, and the beam-local
    # entry offset h*sin(20deg) ~ 4 mm stays inside the ±8 mm pocket slop.
    for u in range(3):
        ids = torch.nonzero(scene.k_units > u, as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            continue
        beam_p = scene.beam.data.root_pos_w
        beam_q = scene.beam.data.root_quat_w
        rest = c.floor_z1 + c.weight_size / 2
        local = torch.tensor([-c.arm_r, c.pocket_y[u], rest],
                             device=device).expand(ids.numel(), 3)
        st = torch.zeros(ids.numel(), 13, device=device)
        st[:, 0:3] = beam_p[ids] + quat_apply(beam_q[ids], local)
        st[:, 2] += 0.012
        st[:, 3:7] = beam_q[ids]  # faces parallel to the pocket walls
        scene.weights[u].write_root_state_to_sim(st, ids)
        step(150)  # seat + let the beam swing toward its new equilibrium
        report(f"ballast {u + 1}")
    step(300)  # final approach to level (slow pole ~0.5/s)
    report("ballasted")
    in_rack, _ = scene._weight_flags()
    assert bool((in_rack.sum(dim=1) == scene.k_units).all()), \
        "exactly k weights must be seated in the rack"
    s2 = print_score("ballast seated")
    assert s2 >= 0.59, f"both latches must pay, got {s2}"
    assert s2 >= s1, "score must not decrease"

    # ----- phase 3: VERIFY — hands off, success must hold and keep holding -----------------------
    for i in range(1200):
        if bool(scene.success().all()):
            break
        env.step(no_action)
        if i % 300 == 299:
            report(f"levelling {i + 1}")
    report("level")
    assert bool(scene.success().all()), "success() must hold before the persistence window"
    s3 = print_score("success reached")
    assert s3 >= 0.999, f"live success must score 1.0, got {s3}"

    step(400)  # >= 3.3 simulated seconds, hands off
    report("persistence")
    ok = scene.success()
    s4 = print_score("after persistence")
    if bool(ok.all()) and s4 >= 0.999:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        os._exit(0)
    print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)
    os._exit(1)


try:
    main()
except Exception as e:  # noqa: BLE001
    print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
    os._exit(1)
