"""Teleport solution for UtensilBalanceScene (sim_gen task `track_spoon_i160`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY, always ending in FREE SPACE or a non-contact
hover; every load-bearing interaction goes through CONTACT DYNAMICS and the passive
balance mechanism:
  P0 — SETTLE: after reset the hidden counterweight (resting in one hanging pan, placed
  by the scene) tips the beam to its hard stop under gravity: the reset state is a
  visibly loaded scale. Nothing is touched.
  P1 — PROBE (the discovery demo): a deliberately WRONG utensil is carried to a hover
  30 mm above the EMPTY pan's tray and DROPPED. The load transfers through contact,
  the hanging pan carries it, and the beam — the mechanism's readout — settles clearly
  OFF level (every wrong pairing out-torques the keel and pins the beam at a stop).
  The teleport bypasses nothing: the judged quantity is the beam's settled RESPONSE to
  the utensil's true mass, which only physics produces.
  P2 — SWAP: the wrong utensil is lifted out (transport to a hover over its floor slot,
  then dropped to the floor) and the MATCHING utensil is dropped into the empty pan the
  same way. With equal masses in the two hanging pans, the keel is the only net torque
  and the beam settles LEVEL — again, pure mechanism response.
  P3 — VERIFY + persistence: hands off for >= 3.3 simulated seconds after success()
  first turns True; `SIM_GEN_SOLVE: SUCCESS` only if it still holds.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched).

Run (forge): python -u -m simgen_tasks.track_spoon_i160.solve --headless [--seed N]
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
    env = ENVS.get("simgen.utensil_balance")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def deg(x: float) -> float:
        return math.degrees(x)

    def report(tag: str) -> None:
        wl, wr = scene.wt_in_pans()
        ml, mr = scene.match_in_pans()
        print(f"[solve] {tag:14s} | pitch={deg(float(scene.beam_pitch()[0])):+6.2f}deg "
              f"wt_in=({bool(wl[0])},{bool(wr[0])}) match_in=({bool(ml[0])},{bool(mr[0])}) "
              f"arr={bool(scene.arrangement()[0])} level={bool(scene.level()[0])} "
              f"floor={bool(scene.others_on_floor()[0])} parked={bool(scene.parked_far()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def stand_yaw_quat() -> tuple[float, float, float, float]:
        yaw = float(scene._stand_yaw[0])
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def place(body, wx: float, wy: float, z: float, quat) -> None:
        """Transport-only teleport to a world-frame pose (ends in a free hover)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = wx, wy, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def drop_into_pan(ute, pan_k: int, hover: float = 0.030) -> None:
        """Hover the utensil centred over pan k's tray (aligned with the stand yaw so
        it fits the tray's x-span), 30 mm above the tray floor — a non-contact hover
        inside the open tray airspace (the pan hangs PLUMB, so there is no tilted-wall
        drift) — then let gravity load the pan."""
        pan = scene.pans[pan_k]
        pp = (pan.data.root_pos_w - scene.env_origins)[0]
        place(ute, float(pp[0]), float(pp[1]),
              float(pp[2]) - c.hang_depth + hover, stand_yaw_quat())
        step(60)  # free fall + tray impact

    def drop_to_floor(ute, wx: float, wy: float) -> None:
        place(ute, wx, wy, 0.050, stand_yaw_quat())
        step(90)

    # ---------------- phase 0: reset, settle, readback -------------------------------------
    step(600)  # the counterweight tips the beam to its stop (~5 s with damping)
    match = int(scene._match[0])
    side = float(scene._side[0])
    free_pan = 0 if side > 0 else 1  # weight sits in PanR (+x) when side=+1
    pitch0 = float(scene.beam_pitch()[0])
    slots = [tuple(float(v) for v in (u.data.root_pos_w - scene.env_origins)[0][:2])
             for u in scene.utes]
    print(f"[solve] layout readback (seed {args.seed}): match={c.ute_names[match]} "
          f"(m={c.ute_masses[match] * 1000:.0f} g) side={side:+.0f} free_pan={free_pan} "
          f"pitch0={deg(pitch0):+.2f}deg stand=({float(scene._stand_xy[0, 0]):+.3f},"
          f"{float(scene._stand_xy[0, 1]):+.3f}) yaw={deg(float(scene._stand_yaw[0])):+.1f}deg "
          f"slots={[(round(x, 3), round(y, 3)) for x, y in slots]}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    if not (abs(deg(pitch0)) >= 20.0 and pitch0 * side > 0):
        print("SIM_GEN_SOLVE: FAIL (weight did not pin the beam at its stop)", flush=True)
        os._exit(1)

    # ---------------- phase 1: PROBE — a wrong utensil, read the beam ----------------------
    wrong = (match + 1) % 3
    print(f"[solve] probing with WRONG utensil: {c.ute_names[wrong]} "
          f"({c.ute_masses[wrong] * 1000:.0f} g vs weight "
          f"{c.ute_masses[match] * 1000:.0f} g)", flush=True)
    drop_into_pan(scene.utes[wrong], free_pan)
    step(840)  # the beam answers: wrong mass -> pinned off level (~7 s to settle)
    report("probe-wrong")
    pitch1 = float(scene.beam_pitch()[0])
    s1 = print_score("P1 wrong utensil probed")
    assert s1 >= s0 - 1e-6, "score decreased across the probe phase"
    if abs(deg(pitch1)) <= c.level_deg + 2.0:
        print("SIM_GEN_SOLVE: FAIL (wrong utensil read as level)", flush=True)
        os._exit(1)
    if not bool(scene._engaged[0]):
        print("SIM_GEN_SOLVE: FAIL (probe did not latch engagement)", flush=True)
        os._exit(1)

    # ---------------- phase 2: SWAP — wrong out to the floor, match in ---------------------
    drop_to_floor(scene.utes[wrong], *slots[wrong])
    print(f"[solve] placing MATCH utensil: {c.ute_names[match]}", flush=True)
    drop_into_pan(scene.utes[match], free_pan)
    # equal masses: the keel levels the beam; wait for the swing to die out
    for _ in range(30):
        step(60)
        if bool(scene.success()[0]):
            break
    report("match-in")
    s2 = print_score("P2 swap to the matching utensil")
    assert s2 >= s1 - 1e-6, "score decreased across the swap phase"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the match settled)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3.3 simulated seconds, hands off) -----------
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
