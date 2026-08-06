"""Teleport solution for LetterboxDepositScene (sim_gen task `put_item_in_drawer_i296`)
— the task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): one pose write carries the green parcel from its spawn slot
   across free space to a hold pose hovering just OUTSIDE the slot mouth, nose-on. This
   bypasses nothing: the parcel is outside the box at both endpoints, the flap is shut,
   and hovering satisfies no rubric clause beyond the approach-distance term (which
   teleports cannot cap at more than its 0.15 weight and which any real carry earns
   identically).
2. PUSH-THROUGH (contact dynamics — the core interaction; no teleport can produce it
   without bypassing the task): the parcel is pose-HELD each step (the kinematic-hold
   emulation of a rigid grasp — exactly what a closed jaw does to a held parcel,
   gravity-compensated vz=+g*dt) and advanced slowly along the slot axis. The FLAP is
   NEVER pose-written: the parcel's nose presses it, it swings inward about its real
   hinge against gravity (asserted to pass the 25 deg latch threshold), and the parcel
   rides the slot's bottom lip into the cavity mouth.
3. RELEASE + DROP (contact dynamics): the hold stops with the parcel's centre of mass
   ~20 mm past the lip's inner edge — still perched at slot height, NOT in the scoring
   band (asserted: the freshly-released state does not satisfy containment). Gravity
   tips it over the lip; it falls ~15 cm to the cavity floor, settles through real
   impacts, and the flap swings shut behind it under its own gravity. success() first
   turns True only here, judged on settled poses.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.put_item_in_drawer_i296.solve --headless [--seed N]
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

DT = 1.0 / 120.0
G_DT = 9.81 * DT  # gravity-compensated kinematic hold: write vz=+g*dt so net vz ~ 0


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.letterbox_deposit")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def flap_deg() -> float:
        return math.degrees(float(scene.flap_open()[0]))

    def report(tag: str) -> None:
        p = (scene.parcel.data.root_pos_w - scene.env_origins)[0]
        d = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] {tag:12s} | parcel=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):.3f}) decoy=({float(d[0]):+.3f},{float(d[1]):+.3f},"
              f"{float(d[2]):.3f}) flap={flap_deg():+.1f}deg "
              f"in={bool(scene._contained(scene.parcel)[0])} "
              f"app={float(scene._app_max[0]):.3f} "
              f"pushed={bool(scene._flap_pushed[0])} transit={bool(scene._in[0])} "
              f"dep={bool(scene._dep[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def hold_parcel(pos: torch.Tensor, vx: float) -> None:
        """One kinematic-hold write: pose imposed nose-on to the slot (identity quat),
        vz=+g*dt cancels the gravity kick, vx is the push speed (consistent with the
        trajectory so contacts see the true relative velocity)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3] = 1.0
        st[:, 7] = vx
        st[:, 9] = G_DT
        scene.parcel.write_root_state_to_sim(st, all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    p0 = (scene.parcel.data.root_pos_w - scene.env_origins)[0]
    d0 = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): parcel=({float(p0[0]):+.3f},"
          f"{float(p0[1]):+.3f}) decoy=({float(d0[0]):+.3f},{float(d0[1]):+.3f}) "
          f"box=({c.box_pos[0]:+.3f},{c.box_pos[1]:+.3f}) flap={flap_deg():+.1f}deg",
          flush=True)
    report("reset")
    assert abs(flap_deg()) <= c.flap_closed_deg, "flap did not fall shut at reset"
    assert not bool(scene._contained(scene.parcel)[0]), "parcel spawned inside the box"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"baseline score not ~0 ({s0:.3f})"

    # ---------------- phase 1: TRANSPORT (teleport to the slot mouth) -----------------------
    # One pose write carries the parcel to a hold pose nose-on, 20 mm outside the front
    # face, centred on the slot, bottom 4 mm above the slot's lip. Outside the box, flap
    # untouched: no containment/flap/transit clause is satisfied by this state.
    box_w = scene.box.data.root_pos_w[0]
    front_x = float(box_w[0]) - c.in_d / 2 - c.t  # front face plane (world x)
    hold_z = float(box_w[2]) + c.slot_z0 + c.parcel_h / 2 + 0.004
    pre = torch.zeros(n, 3, device=device)
    pre[:, 0] = front_x - c.parcel_d / 2 - 0.020
    pre[:, 1] = box_w[1]
    pre[:, 2] = hold_z
    hold_parcel(pre, 0.0)
    for _ in range(20):
        hold_parcel(pre, 0.0)
        env.step(no_action)
    report("transported")
    assert abs(flap_deg()) <= c.flap_closed_deg, "flap disturbed by the hover pose"
    assert not bool(scene._contained(scene.parcel)[0]), "hover pose already contained"
    s1 = print_score("P1 transport to the slot mouth (outside, flap shut)")
    assert s1 >= s0 - 1e-6, "score decreased across transport"

    # ---------------- phase 2: PUSH THROUGH the flap (contact dynamics) ---------------------
    # Kinematic-hold advance at 4 cm/s along +x. The flap is never written: the parcel's
    # nose presses it open about its real hinge (against gravity), and the parcel rides
    # the slot's bottom lip. Stop with the CoM 20 mm past the lip's inner edge.
    speed = 0.04
    stop_x = float(box_w[0]) - c.in_d / 2 + 0.020  # CoM target: 20 mm inside the inner face
    pos = pre.clone()
    max_open = 0.0
    for i in range(1500):
        if float(pos[0, 0]) >= stop_x:
            break
        pos[:, 0] += speed * DT
        hold_parcel(pos, speed)
        env.step(no_action)
        max_open = max(max_open, flap_deg())
    report("pushed")
    print(f"[solve] max flap opening during the push: {max_open:.1f} deg", flush=True)
    assert max_open >= c.flap_open_min_deg, \
        f"flap never displaced past the {c.flap_open_min_deg} deg latch — no real contact?"
    assert bool(scene._flap_pushed[0]), "flap_pushed latch not set"
    s2 = print_score("P2 contact-dynamics push through the flap")
    assert s2 >= s1 - 1e-6, "score decreased across the push"

    # ---------------- phase 3: RELEASE -> tip over the lip, drop, flap re-closes ------------
    # Stop holding. The freshly-released parcel is perched at slot height — asserted NOT
    # in the scoring band (containment requires z below deposit_z_max, far under the
    # slot). Gravity tips it over the lip; it falls to the cavity floor and settles; the
    # flap swings shut behind it under its own gravity + hinge damping.
    hold_parcel(pos, speed)  # final write, then hands off
    assert not bool(scene._contained(scene.parcel)[0]), \
        "released state already satisfies containment (release band too low)"
    settled = False
    for i in range(600):
        env.step(no_action)
        if i > 60:
            pz = float((scene.parcel.data.root_pos_w - scene.env_origins)[0, 2])
            still = float(scene.parcel.data.root_lin_vel_w[0].norm()) < 0.03
            if still and pz < 0.12:
                settled = True
                break
    report("dropped")
    assert settled, "parcel did not settle on the cavity floor after release"
    assert bool(scene._contained(scene.parcel)[0]), "parcel not contained after the drop"
    # give the flap time to finish its pendulum return if it is still moving
    for _ in range(240):
        if abs(flap_deg()) <= c.flap_closed_deg * 0.5:
            break
        env.step(no_action)
    step(60)
    report("flap-closed")
    s3 = print_score("P3 release: lip tip-over, drop to the floor, flap re-closed")
    assert s3 >= s2 - 1e-6, "score decreased across the release"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the deposit)", flush=True)
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
