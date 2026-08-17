"""Teleport solution for EscrowClampScene (sim_gen task `handover_i339`) — the task's
legitimacy certificate.

Teleports are TRANSPORT ONLY (one relocation of the spool, in free air, released into
free air above the open-top ledge/prong channel). Every load-bearing interaction is
gravity + contact + the two sanctioned slide drives (scene.jaw_drive /
scene.ledge_drive — plant-side forces along the authored prismatic axes; the scene's
post_step owns both external-wrench slots).

1. PERCEPTION: the station pose, WHICH apron slot holds the waisted spool (vs the
   identical-looking decoy), and both slide seats are read back from the episode state
   (station-local box tests + q readbacks), never hard-coded. Readbacks are asserted
   against the scene's own swap / q_jaw0 / q_ledge0.
2. PRESENT (gravity + contact): the spool is teleported to free air 25 mm above its
   standing pose on the escrow ledge, released upright with the station's heading.
   Gravity stands it on the shelf between the OPEN prongs. `SIM_GEN_SCORE ~0.25`.
3. CLAMP (sanctioned drive): a velocity servo on the jaw slide (K = 15 N·s/m,
   v_des = 0.08 m/s; K*dt/m = 0.25 << 1; the jaw closes in -q, so the negative-
   feedback law is K*(v_des + q̇)) pushes the jaw closed until q_jaw <= 4 mm;
   the notch flanks the waist between the flanges and the tip-to-anvil gap shrinks
   below the waist diameter. `SIM_GEN_SCORE ~0.55`.
4. RELEASE (sanctioned drive): the same servo law on the ledge slide withdraws the
   shelf from under the standing spool. It drops 9 mm and HANGS by its top flange on
   the prong tops, suspended over the reject basin — the receiver alone bears it.
5. VERIFY: hands off (both drives zero), success() (jaw closed + ledge out + spool
   upright in the capture cell at hang height + settled + finite) must hold and keep
   holding >= 3.3 more simulated seconds before SIM_GEN_SOLVE: SUCCESS prints.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the rubric
latches 0.25 presented + 0.30 clamped, capped 0.55; exactly 1.0 iff live success()).

Run (forge): python -u -m simgen_tasks.handover_i339.solve --headless [--seed N]
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.escrow_clamp")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    from isaaclab.utils.math import quat_apply

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
        loc = scene.spool_local()
        print(f"[solve] {tag:16s} | q_jaw={1000 * float(scene.q_jaw()[0]):+6.1f}mm "
              f"q_ledge={1000 * float(scene.q_ledge()[0]):+6.1f}mm "
              f"spool=({float(loc[0, 0]):+.3f},{float(loc[0, 1]):+.3f},{float(loc[0, 2]):+.3f}) "
              f"up={bool(scene.spool_upright()[0])} "
              f"cell={bool(scene.spool_in_cell()[0])} "
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
    # perception: WHICH apron slot holds the SPOOL (station-local box test on the two
    # slots) — the waist is the only difference between the two green cylinders, and
    # only the waisted one can ever be clamped. Asserted against the episode's swap.
    loc_sp = scene._station_local(scene.spool.data.root_pos_w)
    in_slot = []
    for sx, sy in c.slot_xy:
        in_slot.append(((loc_sp[:, 0] - sx).abs() < 0.06)
                       & ((loc_sp[:, 1] - sy).abs() < 0.06)
                       & (loc_sp[:, 2] > 0.0) & (loc_sp[:, 2] < 0.10))
    assert bool((in_slot[0] ^ in_slot[1]).all()), "spool must stand in exactly one slot"
    swap_perceived = in_slot[1]
    assert bool((swap_perceived == scene.swap).all()), "slot readback must match episode swap"
    qj, ql = scene.q_jaw(), scene.q_ledge()
    assert bool(((qj - scene.q_jaw0).abs() < 0.006).all()), "jaw seat readback"
    assert bool(((ql - scene.q_ledge0).abs() < 0.006).all()), "ledge seat readback"
    assert bool((qj > c.jaw_closed_q + 0.05).all()), "jaw must start OPEN"
    print(f"[solve] perception: spool in slot {int(swap_perceived[0])} "
          f"(q_jaw0={1000 * float(qj[0]):.1f}mm q_ledge0={1000 * float(ql[0]):.1f}mm)",
          flush=True)
    s0 = print_score("start")
    assert s0 < 0.05, f"fresh episode must score ~0, got {s0}"

    # ----- phase 1: PRESENT — stand the spool on the escrow ledge between the prongs -------------
    stand_z = c.ledge_anchor[2] + c.shelf_hz + c.spool_hh
    st_p = scene.station.data.root_pos_w
    st_q = scene.station.data.root_quat_w
    local = torch.tensor([c.present_xy[0], c.present_xy[1], stand_z + 0.025],
                         device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = st_p + quat_apply(st_q, local)  # free air above the open channel
    st[:, 3:7] = st_q  # upright, station heading
    scene.spool.write_root_state_to_sim(st)
    step(240)  # drop 25 mm + settle standing on the shelf
    report("presented")
    assert bool(scene._presented.all()), "spool must be PRESENTED on the ledge"
    loc = scene.spool_local()
    assert bool(((loc[:, 2] - stand_z).abs() < 0.006).all()), "spool must STAND on the shelf"
    s1 = print_score("presented")
    assert s1 >= 0.24, f"present latch must pay, got {s1}"
    assert s1 >= s0, "score must not decrease"

    # ----- phase 2: CLAMP — servo the jaw closed around the waist --------------------------------
    # Velocity servo (K*dt/m = 15/(120*0.5) = 0.25 << 1); coast onto the q=0 stop is
    # damped out by the authored 6 /s linear damping.
    for i in range(900):
        done = scene.q_jaw() <= 0.004
        if bool(done.all()):
            break
        # jaw closes in the -q direction: drive = K * (v_des + q̇) is the negative-
        # feedback law (q̇ < 0 while closing; overspeed q̇ < -v_des brakes)
        scene.jaw_drive = torch.where(
            done, torch.zeros(n, device=device), 15.0 * (0.08 + scene.rate_jaw))
        env.step(no_action)
        if i % 120 == 119:
            report(f"closing {i + 1}")
    scene.jaw_drive = torch.zeros(n, device=device)
    step(120)  # settle on the stop
    report("clamped")
    assert bool(scene.jaw_closed().all()), "jaw must be CLOSED around the waist"
    assert bool(scene._clamped.all()), "clamp latch must have fired"
    s2 = print_score("clamped")
    assert s2 >= 0.54, f"both latches must pay, got {s2}"
    assert s2 >= s1, "score must not decrease"

    # ----- phase 3: RELEASE — withdraw the ledge; the jaw alone bears the spool ------------------
    for i in range(900):
        done = scene.q_ledge() >= 0.100
        if bool(done.all()):
            break
        scene.ledge_drive = torch.where(
            done, torch.zeros(n, device=device), 15.0 * (0.06 - scene.rate_ledge))
        env.step(no_action)
        if i % 120 == 119:
            report(f"pulling {i + 1}")
    scene.ledge_drive = torch.zeros(n, device=device)
    report("released")
    assert bool(scene.ledge_out().all()), "ledge must be OUT"

    # ----- phase 4: VERIFY — hands off, success must hold and keep holding -----------------------
    for i in range(1200):
        if bool(scene.success().all()):
            break
        env.step(no_action)
        if i % 300 == 299:
            report(f"settling {i + 1}")
    report("handed over")
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
