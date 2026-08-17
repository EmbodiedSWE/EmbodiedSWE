"""Teleport solution for MilkShuntYardScene — the task's legitimacy certificate.

NOTHING is teleported after reset: the whole solution is planar CoM pushes (the same
external-wrench channel a fingertip on the proud crate tops exercises) plus gravity:

  SHUNT   — a velocity-regulated +y (yard frame) CoM push on the orange blocker
            slides it out of the junction straight into the narrow siding pocket
            (which refuses the crate) and parks it there.
  COMMIT  — the same push channel drives the crate along the cross-lane toward the
            goal (green-tab) side, with a small south bias to ride the south wall.
  DELIVER — the push stops the moment the crate leaves the deck: its CoM crosses the
            goal aperture edge, it tips through under gravity and free-falls into the
            basket inside the enclosed under-deck cell. The landing is real contact.

After success() first turns True the sim keeps running with no intervention for 3+
simulated seconds; only if success() still holds is `SIM_GEN_SOLVE: SUCCESS` printed.
`SIM_GEN_SCORE <v>` is printed at every phase boundary and asserted non-decreasing.

Run: python -m simgen_tasks.libero_pick_milk_i359.solve --headless [--seed N]
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
    t = threading.Timer(20.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
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


def _yard_yaw() -> float:
    q = _ENV.scene.yard.data.root_quat_w[0]
    return 2.0 * math.atan2(float(q[3]), float(q[0]))


def _crate_local():
    scene = _ENV.scene
    return scene._to_yard(scene.crate.data.root_pos_w)[0]


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    p = _crate_local()
    b = scene._to_yard(scene.blocker.data.root_pos_w)[0]
    print(f"[solve] {tag:10s} | crate=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):.3f}) "
          f"blocker=({float(b[0]):+.3f},{float(b[1]):+.3f}) "
          f"shunt={bool(scene._shunted[0])} commit={bool(scene._committed[0])} "
          f"deliv={bool(scene._delivered[0])} trap={bool(scene._trapped[0])} "
          f"in_basket={bool(scene.in_basket(scene.crate)[0])} "
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


def _push(body, buf, axis, sign, track, pred, *, force: float, vmax: float, iters: int,
          tag: str, stop_off_deck: bool = False) -> bool:
    """Bang-bang velocity-regulated planar CoM push on `body` (force written into the
    scene-owned buffer `buf`) along yard-local `axis` (0=x, 1=y) with sign `sign`,
    while a lateral P-correction steers the OTHER coordinate toward `track` (keeps
    the body centered in its channel — a fixed direction lets it drift and corner-
    catch wall ends). Cuts the force the instant the body leaves the deck (gravity
    finishes a delivery). Returns pred() at exit."""
    scene = _ENV.scene
    c = scene.cfg
    yaw = _yard_yaw()
    cy, sy = math.cos(yaw), math.sin(yaw)
    lat = 1 - axis
    ok = False
    for it in range(iters):
        _refresh()
        if pred():
            ok = True
            break
        p = scene._to_yard(body.data.root_pos_w)[0]
        if stop_off_deck and float(p[2]) < c.deck_z - 0.005:
            break  # body is falling — hands off
        d = [0.0, 0.0]
        d[axis] = float(sign)
        d[lat] = max(-0.5, min(0.5, 25.0 * (track - float(p[lat]))))
        nrm = math.hypot(d[0], d[1])
        wx = (cy * d[0] - sy * d[1]) / nrm
        wy = (sy * d[0] + cy * d[1]) / nrm
        v = body.data.root_lin_vel_w[0]
        along = float(v[0]) * wx + float(v[1]) * wy
        f = 0.0 if along > vmax else force
        buf[0, 0] = f * wx
        buf[0, 1] = f * wy
        _step(2)
        if it % 100 == 0:
            print(f"[solve]   push[{tag}] it={it} p=({float(p[0]):+.3f},{float(p[1]):+.3f},"
                  f"{float(p[2]):.3f}) v={along:+.3f}", flush=True)
    buf[:] = 0.0
    _refresh()
    return ok or pred()


def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.milk_shunt_yard")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg

    torch.manual_seed(1000 * args.seed + 7)
    env.reset()
    _step(60)  # settle the dealt layout

    _refresh()
    gs = float(scene.goal_side[0])
    yp = scene.yard.data.root_pos_w[0]
    print(f"[solve] seed={args.seed} layout: goal_side={gs:+.0f} "
          f"yard=({float(yp[0]):.3f},{float(yp[1]):.3f}) yaw={math.degrees(_yard_yaw()):+.1f}deg "
          f"crate={[round(float(v), 3) for v in _crate_local()]}", flush=True)
    _report("start")
    if float(scene.score()[0]) > 0.05 or bool(scene.success()[0]):
        print("[solve] FATAL: nonzero score / success at reset", flush=True)
        _hard_exit(1)
    _mark_score("start")

    # ---- phase 1: SHUNT — push the blocker straight north into the siding ------------------
    # Direct CoM push (its top face is proud of the walls): no yaw build-up, so it
    # threads the 94 mm pocket cleanly. Park it well past the latch line.
    def _blocker_y() -> float:
        return float(scene._to_yard(scene.blocker.data.root_pos_w)[0][1])

    ok = _push(scene.blocker, scene.blocker_force, 1, +1.0, 0.0,
               lambda: _blocker_y() > 0.115,
               force=1.8, vmax=0.06, iters=900, tag="blocker")
    _step(30)  # let it settle
    if not ok or not (bool(scene.shunted_now()[0]) or bool(scene._shunted[0])):
        print("[solve] FATAL: blocker did not shunt into the siding", flush=True)
        _hard_exit(1)
    _report("shunted")
    _mark_score("shunted")

    # ---- phase 2: ENTER — push the crate north out of the bay into the lane ----------------
    ok = _push(scene.crate, scene.crate_force, 1, +1.0, 0.0,
               lambda: float(_crate_local()[1]) > -0.035,
               force=3.0, vmax=0.10, iters=900, tag="crate")
    if not ok:
        print("[solve] FATAL: crate did not enter the cross-lane", flush=True)
        _hard_exit(1)
    _report("entered")
    _mark_score("entered")

    # ---- phase 3: COMMIT — push the crate along the lane toward the green side -------------
    # Tracking y=-0.035 keeps it off the siding mouth / north wall while it turns.
    ok = _push(scene.crate, scene.crate_force, 0, gs, -0.035,
               lambda: float(_crate_local()[0]) * gs > c.commit_x_min + 0.02,
               force=3.0, vmax=0.15, iters=900, tag="crate", stop_off_deck=True)
    if not ok:
        print("[solve] FATAL: crate did not commit onto the goal side", flush=True)
        _hard_exit(1)
    _report("committed")
    _mark_score("committed")

    # ---- phase 4: DELIVER — push to the aperture edge; gravity does the rest ---------------
    _push(scene.crate, scene.crate_force, 0, gs, -0.020,
          lambda: bool(scene._delivered[0]),
          force=3.0, vmax=0.15, iters=900, tag="crate", stop_off_deck=True)
    ok = _settle_until(
        lambda: bool(scene._delivered[0]) and bool(scene.in_basket(scene.crate)[0])
        and bool(scene.settled(scene.crate)[0]),
        max_steps=480)
    _report("delivered")
    if not ok:
        if bool(scene._trapped[0]) or bool(scene.in_trap()[0]):
            print("[solve] FATAL: crate fell into the TRAP cell", flush=True)
        else:
            print("[solve] FATAL: crate did not land settled in the basket", flush=True)
        _hard_exit(1)
    if not bool(scene.success()[0]):
        print("[solve] FATAL: success() not true after delivery", flush=True)
        _hard_exit(1)
    _mark_score("delivered")

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
