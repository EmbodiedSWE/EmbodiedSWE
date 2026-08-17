"""Teleport solution for BeamBalanceScene (sim_gen task `draw_triangle_i107`) —
the task's legitimacy certificate.

Teleportation is TRANSPORT ONLY: a brass weight is teleported to a hover a few
millimetres ABOVE the live tray floor (never into contact) and dropped, or
teleported back to an empty parking spot on the bench front. Everything
load-bearing happens through CONTACT DYNAMICS: the weight falls the last
millimetres onto the tray floor, gravity torques the beam through the pivot, and
the keel's restoring moment decides — level float for the exact combination,
hard stop for every wrong one. NO external forces are applied at any time; the
final state is fully HANDS-OFF.

The solve does NOT read the hidden stone mass to pick the answer. It runs the
intended closed-loop measurement: greedy binary weighing with the 4/2/1-unit
brass weights — seat the largest, read the settled beam ('over' = beam now tips
toward the brass side), remove if over, keep if not, then the next weight. After
the 3 trials the seated total equals the stone's units for every k in 1..7 (the
classic balance-scale binary search). The oracle values (`active_idx`,
`side_idx`) are read ONLY to cross-check the measurement in the log.

Phases print SIM_GEN_SCORE (non-decreasing: the scene's seating credit is
latched) and `SIM_GEN_SOLVE: SUCCESS` only if success() still holds after
>= 3.3 simulated seconds of hands-off persistence.

Run (forge): python -u -m simgen_tasks.draw_triangle_i107.solve --headless [--seed N]
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
# Daemon: must not keep a *crashed* process alive until the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

# Bench-corner parking spots for weights ruled OUT by a trial (one per weight
# index; all > exclusion_r from the stand under worst-case stand jitter, clear
# of the staging slots and of each other, on the bench).
PARK_XY = ((0.45, -0.35), (-0.45, -0.35), (0.45, 0.35))
# Tray-local xy of each weight's drop spot (side-by-side, >= 2 mm wall/neighbor
# clearance inside the 47.5 mm inner half-width): index 0 = w1, 1 = w2, 2 = w4.
TRAY_SPOT = ((0.026, -0.022), (0.026, 0.020), (-0.023, 0.0))
OVER_DEG = 7.0          # settled 'over' threshold: between level_tol (4) and stop (12)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.beam_balance")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import quat_apply

    def step(kk: int) -> None:
        for _ in range(kk):
            env.step(no_action)

    def tilt_deg() -> float:
        return math.degrees(float(scene.tilt()[0]))

    def report(tag: str) -> None:
        print(f"[solve] {tag:16s} | tilt={tilt_deg():+7.2f} deg  "
              f"in_tray={scene.weight_in_tray()[0].tolist()} "
              f"stone_home={bool(scene.stone_home()[0])} "
              f"clean={bool(scene.weights_clean()[0])} "
              f"stand_home={bool(scene.stand_home()[0])} "
              f"still={bool(scene.still()[0])} (n={int(scene.still_count[0])}) "
              f"seat=({float(scene.seat1_latch[0]):.0f},{float(scene.seat2_latch[0]):.0f}) "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def vel_debug(tag: str) -> None:
        names = (["pan_p", "pan_n"] + [f"stone_{i}" for i in range(c.units_max)]
                 + [nm for nm, *_ in c.weight_specs])
        bodies = scene.pans + scene.stones + scene.weights
        vels = [float(b.data.root_lin_vel_w.norm(dim=-1)[0]) for b in bodies]
        print(f"[solve] veldbg {tag}: beam_avel="
              f"{float(scene.beam.data.root_ang_vel_w.norm(dim=-1)[0]):.4f} "
              f"stand_lin={float(scene.stand.data.root_lin_vel_w.norm(dim=-1)[0]):.4f}",
              flush=True)
        print("[solve] veldbg " + " ".join(f"{n}={v:.4f}" for n, v in zip(names, vels)),
              flush=True)

    s_prev = -1.0

    def print_score(tag: str) -> None:
        nonlocal s_prev
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        assert s >= s_prev - 1e-6, f"score decreased across {tag}"
        s_prev = s

    def settle(min_steps: int = 240, extra: int = 720) -> None:
        """Let the beam swing out and settle: fixed transient, then poll until
        the beam is slow (damping 3.0 settles the float in a few swings)."""
        step(min_steps)
        for _ in range(extra // 30):
            if float(scene.beam.data.root_ang_vel_w.norm(dim=-1)[0]) < 0.15:
                break
            step(30)

    def drop_into_tray(widx: int, pan) -> None:
        """Teleport weight `widx` to a hover 12 mm above its tray spot in the
        LIVE pan frame (zero velocity, yaw-aligned with the pan) and let it
        fall the last millimetres onto the floor plate — pure transport."""
        h = c.weight_specs[widx][3]
        loc = torch.tensor([TRAY_SPOT[widx][0], TRAY_SPOT[widx][1],
                            -c.stem_len + h / 2 + 0.012], device=device).expand(n, 3)
        pos = pan.data.root_pos_w + quat_apply(pan.data.root_quat_w, loc)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = pan.data.root_quat_w
        scene.weights[widx].write_root_state_to_sim(st, torch.arange(n, device=device))

    def park(widx: int) -> None:
        """Teleport weight `widx` back to its empty parking spot (transport)."""
        h = c.weight_specs[widx][3]
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = scene.env_origins[:, 0] + PARK_XY[widx][0]
        st[:, 1] = scene.env_origins[:, 1] + PARK_XY[widx][1]
        st[:, 2] = scene.env_origins[:, 2] + c.bench_top + h / 2 + 0.003
        st[:, 3] = 1.0
        scene.weights[widx].write_root_state_to_sim(st, torch.arange(n, device=device))

    # ---------------- phase 0: premise — the stone parks the beam on a stop -----------------
    settle(min_steps=180)
    t0 = tilt_deg()
    k_true = int(scene.active_idx[0]) + 1
    side_true = int(scene.side_idx[0])         # 0 = stone in PanP (+x), 1 = PanN
    print(f"[solve] readback (seed {args.seed}): stone = {k_true} units "
          f"({k_true * c.unit_mass * 1000:.0f} g) in {'PanP' if side_true == 0 else 'PanN'}, "
          f"settled tilt {t0:+.2f} deg", flush=True)
    assert abs(t0) > 9.0, "premise: the stone must park the beam on a stop"
    # tilt() is positive when the +x (PanP) end is HIGH -> stone in PanP => negative
    assert (t0 < 0) == (side_true == 0), "premise: tilt sign must point at the stone"
    assert bool(scene.stone_home()[0]) and bool(scene.stand_home()[0])
    assert float(scene.score()[0]) < 1e-6, "premise: null score must be 0"
    report("reset")
    vel_debug("reset")
    print_score("P0 reset (beam pinned on a stop by the stone)")

    # The stone side is OBSERVED from the tilt sign (the oracle side_true is only
    # the cross-check above): counter tray = the other pan.
    stone_sign = -1.0 if t0 < 0 else 1.0       # sign of tilt when the stone is heavier
    counter_pan = scene.pans[1] if t0 < 0 else scene.pans[0]

    # ---------------- phases 1-3: greedy binary weighing (4, then 2, then 1) ----------------
    total = 0
    for widx, units in ((2, 4), (1, 2), (0, 1)):
        name = c.weight_specs[widx][0]
        for attempt in range(3):
            drop_into_tray(widx, counter_pan)
            settle()
            if bool(scene.weight_in_tray()[0, widx]):
                break
            print(f"[solve] {name}: bounced out of the tray (attempt {attempt}), "
                  f"re-dropping", flush=True)
        else:
            report("FAIL-seat")
            print(f"SIM_GEN_SOLVE: FAIL ({name} would not seat)", flush=True)
            os._exit(1)
        m = tilt_deg() * stone_sign            # >0: stone side still heavier
        over = m < -OVER_DEG
        print(f"[solve] trial {name} (+{units}u -> {total + units}u vs {k_true}u): "
              f"settled tilt {tilt_deg():+.2f} deg -> "
              f"{'OVER, remove' if over else 'keep'}", flush=True)
        if over:
            park(widx)
        else:
            total += units
        report(f"trial-{name}")
        print_score(f"P-{name} trial done (seated total {total}u)")

    print(f"[solve] measured stone mass: {total} units (oracle: {k_true})", flush=True)
    if total != k_true:
        print("SIM_GEN_SOLVE: FAIL (measurement disagrees with oracle)", flush=True)
        os._exit(1)

    # ---------------- final: hands-off — wait for level + sustained still -------------------
    for it in range(40):
        if bool(scene.success()[0]):
            break
        step(30)
        if it % 10 == 9:
            vel_debug(f"wait-{it}")
    report("balanced")
    print_score("P4 level + sustained still")
    if not bool(scene.success()[0]):
        print("SIM_GEN_SOLVE: FAIL (no success after exact load)", flush=True)
        os._exit(1)

    # ---------------- persistence (>= 3.3 simulated seconds, no intervention) ---------------
    hold = True
    for _ in range(10):       # 10 x 40 steps = 400 steps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    print_score("P5 persistence 3.3 s")
    ok = hold and bool(scene.success()[0])
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
    try:
        main()
    except Exception:  # noqa: BLE001 - fail fast and loud, never hang on teardown
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(2)
