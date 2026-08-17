"""Teleport solution for ButterDispenserScene — the task's legitimacy certificate.

Teleportation moves the BASKET across free space only; every load-bearing interaction
goes through contact dynamics:

  STAGE    — one free-space teleport carries the basket from its dealt spawn to 20 mm
             above the green catch mat, upright, zero velocity; the landing is a real
             gravity drop that must settle upright on the mat inside the tolerance.
  DISPENSE — the pusher blade is driven by a velocity-regulated x-force (the same
             external-wrench channel a fingertip pushing the red paddle exercises)
             through one full stroke. The bottom butter block is extruded through the
             outlet slot under real contact (the stack riding on it, the retained
             block dragged against the front wall) and free-falls off the lip into the
             staged basket. No block pose is ever written after reset.
  RETRACT  — the blade is driven back home the same way; the retained block drops onto
             the chamber floor and becomes the new bottom block, still inside the tower.

After success() first turns True the sim keeps running with no intervention for 3+
simulated seconds; only if success() still holds is `SIM_GEN_SOLVE: SUCCESS` printed.
`SIM_GEN_SCORE <v>` is printed at every phase boundary and asserted non-decreasing.

Run: python -m simgen_tasks.libero_pick_butter_i282.solve --headless [--seed N]
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
    ib = scene.in_basket()[0] & scene.present[0]
    print(f"[solve] {tag:12s} | disp={float(scene.blade_disp()[0]) * 1000:6.1f}mm "
          f"staged={bool(scene._staged[0])} act={bool(scene._actuated[0])} "
          f"deliv={bool(scene._delivered[0])} in_basket={int(ib.sum())} "
          f"in_tower={int((scene.in_tower()[0] & scene.present[0]).sum())} "
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


def _drive_blade(target_disp: float, force: float, vmax: float, iters: int) -> None:
    """Bang-bang velocity-regulated x-force on the blade until it passes target_disp.
    force < 0 extends (pushes the stroke), force > 0 retracts."""
    scene = _ENV.scene
    sgn = -1.0 if force < 0 else 1.0  # extension = -x travel
    for it in range(iters):
        _refresh()
        disp = float(scene.blade_disp()[0])
        if (sgn < 0 and disp >= target_disp) or (sgn > 0 and disp <= target_disp):
            break
        vx = float(scene.blade.data.root_lin_vel_w[0, 0])
        moving = (vx < -vmax) if sgn < 0 else (vx > vmax)
        scene.blade_force[:] = 0.0 if moving else force
        _step(2)
        if it % 100 == 0:
            print(f"[solve]   blade it={it} disp={disp * 1000:6.1f}mm vx={vx:6.3f}",
                  flush=True)
    scene.blade_force[:] = 0.0


def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.butter_dispenser")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg

    torch.manual_seed(1000 * args.seed + 7)
    env.reset()
    _step(60)  # settle the dealt layout

    # layout readback — proves seeds differ
    _refresh()
    bp = scene.basket.data.root_pos_w[0]
    print(f"[solve] seed={args.seed} layout: n_blocks={int(scene.present[0].sum())} "
          f"basket=({float(bp[0]):.3f}, {float(bp[1]):.3f}) "
          f"block0_xy={scene.blocks[0].data.root_pos_w[0, :2].tolist()}", flush=True)
    _report("start")
    if float(scene.score()[0]) > 0.05 or bool(scene.success()[0]):
        print("[solve] FATAL: nonzero score / success at reset", flush=True)
        _hard_exit(1)
    _mark_score("start")

    # ---- phase 1: STAGE — free-space teleport above the mat, real gravity set-down --------
    cxy = scene._catch_xy()[0]
    st = torch.zeros(env.num_envs, 13, device=env.device)
    st[:, 0] = cxy[0]
    st[:, 1] = cxy[1]
    st[:, 2] = scene.env_origins[0, 2] + c.mat_t + 0.020
    st[:, 3] = 1.0  # upright, zero velocity
    scene.basket.write_root_state_to_sim(st, torch.arange(env.num_envs, device=env.device))
    _refresh()
    if not _settle_until(lambda: bool(scene.staged_now()[0]), max_steps=360):
        print("[solve] FATAL: basket did not settle staged on the mat", flush=True)
        _hard_exit(1)
    _report("staged")
    _mark_score("staged")

    # ---- phase 2: DISPENSE — drive the blade through one full stroke ----------------------
    # First to 65% (past the actuated gate), mark, then to the full stroke: the bottom
    # block's CoM crosses the lip and it free-falls into the staged basket.
    _drive_blade(0.65 * c.stroke, force=-10.0, vmax=0.12, iters=500)
    _refresh()
    if not bool(scene._actuated[0]):
        print("[solve] FATAL: blade failed to reach the actuated gate", flush=True)
        _hard_exit(1)
    _report("actuated")
    _mark_score("actuated")

    _drive_blade(0.97 * c.stroke, force=-10.0, vmax=0.20, iters=500)
    ok = _settle_until(
        lambda: bool(scene._delivered[0])
        and int((scene.in_basket()[0] & scene.present[0]).sum()) == 1
        and bool(scene.blocks_settled()[0, 0] | ~scene.present[0, 0]),
        max_steps=480)
    _report("dispensed")
    if not ok:
        print("[solve] FATAL: dispense did not deliver one block into the basket", flush=True)
        _hard_exit(1)
    _mark_score("dispensed")

    # ---- phase 3: RETRACT — drive the blade home; the retained stack drops to the floor ---
    _drive_blade(0.005, force=8.0, vmax=0.12, iters=500)
    if not _settle_until(lambda: bool(scene.success()[0]), max_steps=360):
        print("[solve] FATAL: success() not reached after retract", flush=True)
        _hard_exit(1)
    _report("retracted")
    _mark_score("retracted")

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
