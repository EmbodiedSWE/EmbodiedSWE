"""Teleport solution for MilkCarouselScene — the task's legitimacy certificate.

Teleportation moves the carton across FREE SPACE only; every load-bearing interaction
goes through contact dynamics:

  ROTATE — the carousel is driven by torque about its spindle (the same external-wrench
           channel a fingertip pushing the crank peg exercises), against the joint's
           viscous friction, with the milk carton and both distractors CARRIED by their
           bay walls through real contact until the milk's bay is aligned with the open
           roof sector (closed-loop on the measured milk azimuth). No pose of the rotor
           or of any bay item is ever written after reset.
  LIFT   — the carton is raised out of its bay by an applied vertical force at its CoM,
           rising through the open sector past the roof plane under contact physics
           (the smoke battery proves the same force CANNOT free it from a covered bay).
  PLACE  — with the carton verified airborne above the roof plane (free space), one
           teleport carries it to 25 mm above the delivery pad; the landing is a real
           gravity drop that must settle upright inside the pad tolerance.

After success() first turns True the sim keeps running with no intervention for 3+
simulated seconds; only if success() still holds is `SIM_GEN_SOLVE: SUCCESS` printed.
`SIM_GEN_SCORE <v>` is printed at every phase boundary and asserted non-decreasing.

Run: python -m simgen_tasks.libero_pick_milk_i3.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()

try:
    from . import scene as task_scene  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene  # noqa: F401

_ENV = None
_SCORES: list[float] = []


def _hard_exit(code: int) -> None:
    threading.Timer(20.0, lambda: os._exit(code)).start()
    try:
        _ENV.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


def _step(k: int = 1) -> None:
    no_action = torch.empty(0, device=_ENV.device)
    for _ in range(k):
        _ENV.step(no_action, render=False)


def _refresh() -> None:
    _ENV.iscene.update(0.0)


def _settle_until(pred, max_steps: int = 300, poll: int = 15) -> bool:
    _step(poll)  # ALWAYS force real steps first (zero-step teleport trap)
    if pred():
        return True
    waited = poll
    while waited < max_steps:
        _step(poll)
        waited += poll
        if pred():
            return True
    return False


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    err = math.degrees(float(scene.milk_sector_err()[0]))
    mp = scene.milk.data.root_pos_w[0]
    print(f"[solve] {tag:16s} | sector_err={err:6.1f}deg in_disp="
          f"{bool(scene.in_dispenser()[0])} aligned={bool(scene.aligned_now()[0])} "
          f"freed={bool(scene._freed[0])} milk_z={float(mp[2]):.3f} "
          f"delivered={bool(scene.delivered()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])}",
          flush=True)


def _mark_score(phase: str) -> None:
    _refresh()
    v = float(_ENV.scene.score()[0])
    if _SCORES and v < _SCORES[-1] - 1e-6:
        print(f"[solve] FATAL: score DECREASED {_SCORES[-1]:.3f} -> {v:.3f} at {phase}",
              flush=True)
        _hard_exit(1)
    _SCORES.append(v)
    print(f"SIM_GEN_SCORE {v:.4f}", flush=True)


def _signed_sector_err() -> float:
    """Signed wrap(sector_az - azimuth(milk)) in rad, env 0."""
    scene = _ENV.scene
    _refresh()
    rel = scene.milk.data.root_pos_w[0, :2] - scene._hub_xy()[0]
    phi = math.atan2(float(rel[1]), float(rel[0]))
    d = (float(scene.sector_az[0]) - phi + math.pi) % (2 * math.pi) - math.pi
    return d


def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.milk_carousel")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg

    torch.manual_seed(1000 * args.seed + 7)
    env.reset()
    _step(60)  # settle the dealt layout

    # layout readback — proves seeds differ (fable dress-rehearsal lesson)
    _refresh()
    print(f"[solve] seed={args.seed} layout: milk_bay={int(scene.milk_bay[0])} "
          f"sector_az={math.degrees(float(scene.sector_az[0])):.1f}deg "
          f"start_err={math.degrees(abs(_signed_sector_err())):.1f}deg "
          f"pad={scene.pad.data.root_pos_w[0, :2].tolist()}", flush=True)
    _report("start")
    if float(scene.score()[0]) > 0.05 or bool(scene.success()[0]):
        print("[solve] FATAL: nonzero score / success at reset", flush=True)
        _hard_exit(1)
    _mark_score("start")

    # ---- phase 1: ROTATE — drive the spindle until the milk bay is in the open sector -----
    stop_err = math.radians(8.0)
    ok = False
    for it in range(400):  # 400 * 4 = 1600 steps ~ 13 s sim, generous
        err = _signed_sector_err()
        omega = float(scene.rotor.data.root_ang_vel_w[0, 2])
        if abs(err) < stop_err and abs(omega) < 0.05:
            ok = True
            break
        scene.rotor_drive[:] = max(-0.28, min(0.28, 0.9 * err))
        _step(4)
        if it % 50 == 0:
            print(f"[solve]   rotate it={it} err={math.degrees(err):6.1f}deg "
                  f"omega={omega:5.2f}", flush=True)
    scene.rotor_drive[:] = 0.0
    _step(90)  # coast to rest on viscous friction
    _report("rotated")
    if not ok or not bool(scene.aligned_now()[0]):
        print("[solve] FATAL: rotation phase failed to align the milk bay", flush=True)
        _hard_exit(1)
    _mark_score("rotated")

    # ---- phase 2: LIFT — applied vertical force raises the carton through the opening -----
    lift = 1.6 * c.milk_mass * 9.81
    scene.milk_force[:, 2] = lift
    lifted = False
    for _ in range(300):
        _step(2)
        _refresh()
        if float(scene.milk.data.root_pos_w[0, 2]) > 0.35:
            lifted = True
            break
    scene.milk_force[:] = 0.0
    _refresh()
    if not lifted or not bool(scene.freed_now()[0] | scene._freed[0]):
        print("[solve] FATAL: lift phase failed to free the carton through the opening",
              flush=True)
        _hard_exit(1)
    _report("lifted")
    _mark_score("lifted")

    # ---- phase 3: PLACE — one free-space teleport to 25 mm above the pad, gravity drop ----
    # (the carton is verified airborne ABOVE the roof plane; both endpoints are free space)
    pad_top = scene.pad.data.root_pos_w[0].clone()
    st = torch.zeros(env.num_envs, 13, device=env.device)
    st[:, 0] = pad_top[0]
    st[:, 1] = pad_top[1]
    st[:, 2] = pad_top[2] + c.pad_t / 2 + c.milk_h / 2 + 0.025
    st[:, 3] = 1.0  # upright, zero velocity
    scene.milk.write_root_state_to_sim(st, torch.arange(env.num_envs, device=env.device))
    _refresh()
    settled = _settle_until(lambda: bool(scene.delivered()[0]), max_steps=360)
    _report("placed")
    if not settled:
        print("[solve] FATAL: carton did not settle upright on the pad", flush=True)
        _hard_exit(1)
    _mark_score("placed")

    # ---- persistence: 3+ simulated seconds hands-off, success must still hold -------------
    _step(400)  # 400 / 120 Hz = 3.33 s, no intervention
    _report("persist")
    _mark_score("persist")
    if not bool(scene.success()[0]) or float(scene.score()[0]) != 1.0:
        print("[solve] FATAL: success did not persist the 3 s hands-off window", flush=True)
        _hard_exit(1)

    print(f"[solve] score trajectory: {[round(v, 3) for v in _SCORES]}", flush=True)
    print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    _hard_exit(0)


if __name__ == "__main__":
    main()
