"""Teleport solution for SwingTopBottleScene (sim_gen task `open_wine_bottle_i99`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. UNLATCH (applied hinge torque, hands-off finish): the bail is opened by a small
   torque about the hinge axis (`scene.bail_drive`, the scene-owned wrench plant) —
   the finger-push on the crossbar. The drive is CUT just past the over-center apex
   (~51 deg): the bail crosses dead-centre under drive but reaches the open limit
   (125 deg) purely under gravity — the snap is the mechanism's own dynamics, and
   the `open` latch requires the bail AT REST past 105 deg, which no transient can
   fake.
2. EXTRACT (applied force): with the bail open, a vertical world-frame pull
   (`scene.stopper_pull`) lifts the stopper out of the neck bore. The `out` credit
   is gated on the open latch — with the bail closed this exact pull provably jams
   (the crossbar sits past top-dead-centre, so the rising cap torques the bail INTO
   its lower limit; smoke.py demonstrates the rejection).
3. TRANSPORT (teleport): one root-state write carries the freed, airborne stopper
   to a hover 20 mm above the coaster — exactly the carry a gripper performs. It
   satisfies no rubric clause by itself: `placed` and success() require the stopper
   RESTING on the coaster, which the subsequent hands-off drop + settle produces by
   gravity and contact.
4. RESTRAINT: the amber decoy bottle is never touched; success() checks live that
   its bail is closed, its stopper seated, and both bottles upright.

The order unlatch -> extract is physically forced (the interlock), not scripted.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.open_wine_bottle_i99.solve --headless [--seed N]
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

import os
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers the scene)
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.swing_top_bottle")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] instruction: " + env.scene.instruction(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        ang = scene.bail_angle_deg()[0]
        loc = scene.stopper_local()[0]
        print(f"[solve] {tag:12s} | bail(t/d)=({float(ang[0]):+7.1f},{float(ang[1]):+7.1f})deg "
              f"stop_z(t/d)=({float(loc[0, 2]):.3f},{float(loc[1, 2]):.3f}) "
              f"seated_d={bool(scene.seated()[0, 1])} "
              f"on_coaster={bool(scene.on_coaster()[0])} "
              f"up={bool(scene.bottle_up()[0].all())} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # everything seats (bottles ground, bails at the closed limit, stoppers on rims)
    swap = bool(scene.swap[0])
    orig = scene.env_origins[0]
    def _xy(body):
        p = body.data.root_pos_w[0] - orig
        return f"({float(p[0]):+.3f},{float(p[1]):+.3f})"
    yaw_t = scene._qref_bail[0, 0]
    import math as _m
    yaw_deg = _m.degrees(2.0 * _m.atan2(float(yaw_t[3]), float(yaw_t[0])))
    print(f"[solve] layout readback (seed {args.seed}): swap={swap} "
          f"target={_xy(scene.bottles[0])} yaw_t={yaw_deg:+.1f}deg "
          f"decoy={_xy(scene.bottles[1])} coaster={_xy(scene.coaster)}", flush=True)
    report("reset")
    ang0 = scene.bail_angle_deg()[0]
    assert bool(scene._finite()[0]), "NaN/inf after the reset settle"
    assert float(ang0.abs().max()) < 6.0, f"bails must settle closed, got {ang0.tolist()}"
    assert bool(scene.seated()[0].all()), "both stoppers must start seated"
    assert bool(scene.bottle_up()[0].all()), "both bottles must start upright"
    s0 = print_score("P0 reset+settle (both bottles sealed)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: snap the target bail open -----------------------------------
    # Drive torque ~2x the static gravity torque at closed (m g com_y ~ 0.0044 N*m);
    # cut past the apex (~51 deg) at 65 deg — gravity finishes the snap to the
    # 125 deg limit. Escalate TORQUE on retry (a stalled push needs more push, not
    # more time).
    tau = 0.009
    opened = False
    for attempt in range(3):
        scene.bail_drive[:, 0] = tau
        for _ in range(300):
            env.step(no_action)
            if float(scene.bail_angle_deg()[0, 0]) > 65.0:
                break
        scene.bail_drive[:, 0] = 0.0
        for _ in range(480):
            env.step(no_action)
            a = float(scene.bail_angle_deg()[0, 0])
            w = float(scene.bails[0].data.root_ang_vel_w.norm(dim=-1)[0])
            if a > c.open_min_deg and w < 0.3:
                opened = True
                break
        if opened:
            break
        print(f"[solve] bail not open after attempt {attempt + 1} "
              f"(angle {float(scene.bail_angle_deg()[0, 0]):.1f} deg); "
              f"escalating torque {tau:.4f} -> {tau * 1.6:.4f}", flush=True)
        tau *= 1.6
    step(60)  # hands-off settle at the open limit
    report("unlatch")
    if not (opened and bool(scene._open[0])):
        print("SIM_GEN_SOLVE: FAIL (bail never latched open)", flush=True)
        os._exit(1)
    assert not bool(scene.success()[0]), "cannot be success with the stopper seated"
    s1 = print_score("P1 target bail snapped open (rest past the open limit)")
    assert s1 >= s0 - 1e-6 and s1 >= 0.34, f"P1 score {s1} (expect moved+open=0.35)"

    # ---------------- phase 2: pull the freed stopper out of the bore ----------------------
    # Vertical world pull ~1.9x the stopper weight (0.030 kg): the plug rises out of
    # the bore under force + contact. With the bail closed this same pull jams
    # (smoke.py proves it); here the open latch has already been earned physically.
    pull = 0.55
    freed = False
    for attempt in range(3):
        scene.stopper_pull[:, 0, :] = 0.0
        scene.stopper_pull[:, 0, 2] = pull
        for _ in range(360):
            env.step(no_action)
            if bool(scene.plug_clear()[0, 0]):
                freed = True
                break
        scene.stopper_pull[:, 0, :] = 0.0
        if freed:
            break
        print(f"[solve] plug not clear after attempt {attempt + 1}; "
              f"escalating pull {pull:.2f} -> {pull * 1.5:.2f} N", flush=True)
        pull *= 1.5
        step(120)  # let it re-seat before the next try
    report("extract")
    if not (freed and bool(scene._out[0])):
        print("SIM_GEN_SOLVE: FAIL (stopper never cleared the bore)", flush=True)
        os._exit(1)
    s2 = print_score("P2 stopper pulled clear of the neck bore (gated on open)")
    assert s2 >= s1 - 1e-6 and s2 >= 0.59, f"P2 score {s2} (expect +out=0.60)"

    # ---------------- phase 3: stand the stopper on the coaster ----------------------------
    # TRANSPORT: one write carries the airborne stopper to a free-space hover 20 mm
    # above the coaster top (upright, zero velocity); the landing and rest are pure
    # gravity + contact (CoM at the plug bottom -> it stands on its plug end).
    placed = False
    for attempt in range(3):
        cp = scene.coaster.data.root_pos_w
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = cp[:, 0:2]
        st[:, 2] = cp[:, 2] + c.coaster_h / 2 + 0.020
        st[:, 3] = 1.0
        scene.stoppers[0].write_root_state_to_sim(st, all_ids)
        for i in range(240):
            env.step(no_action)
            if i > 30 and float(scene.stoppers[0].data.root_lin_vel_w.norm(dim=-1)[0]) < c.settle_speed:
                break
        step(60)  # extra hands-off settle
        if bool(scene.on_coaster()[0]) and bool(scene.success()[0]):
            placed = True
            break
        print(f"[solve] stopper not standing on the coaster after attempt "
              f"{attempt + 1}; re-dropping", flush=True)
        report("place-retry")
    report("place")
    if not placed:
        print("SIM_GEN_SOLVE: FAIL (stopper never rested on the coaster)", flush=True)
        os._exit(1)
    assert bool(scene._placed[0]), "placed latch did not set"
    s3 = print_score("P3 stopper standing on the coaster: full success")
    assert s3 >= 0.99, f"P3 score {s3} (expect success=1.0)"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -------------
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
