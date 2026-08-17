"""Teleport solution for DressingFerryDepotScene — the task's legitimacy certificate.

Teleport is TRANSPORT ONLY: the amber bottle is carried once through free air to a
hover pose above the extracted cart's flared well mouth. Every load-bearing
interaction is real contact dynamics:

  EXTRACT — a velocity-regulated depot-local -x CoM pull on the cart (the same
            channel a gripper on the protruding knob exercises) slides it out of
            the garage along the rail apron until the well is clear of the roof.
            While docked the roof-block interlock makes loading impossible.
  LOAD    — the bottle is teleported (transport) to a hover ABOVE the flare mouth,
            released, and falls in under gravity; the 45-deg flare funnels it and
            the snug well squares it upright. If it perches, a small downward press
            (bottle_force) finishes the insertion — contact, not teleport.
  FERRY   — the same pull channel, reversed (+x), pushes the LOADED cart back
            through the doorway (the seated bottle clears the roof by 20 mm) until
            it stops against the back wall = docked. The stop is a real collision.

After success() first turns True the sim keeps running with no intervention for 3+
simulated seconds; only if success() still holds is `SIM_GEN_SOLVE: SUCCESS` printed.
`SIM_GEN_SCORE <v>` is printed at every phase boundary and asserted non-decreasing.

Run: python -m simgen_tasks.libero_pick_salad_dressing_i368.solve --headless [--seed N]
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


def _depot_yaw() -> float:
    q = _ENV.scene.depot.data.root_quat_w[0]
    return 2.0 * math.atan2(float(q[3]), float(q[0]))


def _cart_local():
    scene = _ENV.scene
    return scene._cart_local()[0]


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    p = _cart_local()
    tb = scene._to_depot(scene.target.data.root_pos_w)[0]
    print(f"[solve] {tag:9s} | cart=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):.3f}) "
          f"bottle=({float(tb[0]):+.3f},{float(tb[1]):+.3f},{float(tb[2]):.3f}) "
          f"extr={bool(scene._extracted[0])} load={bool(scene._loaded[0])} "
          f"carry={float(scene._carry[0]):.2f} "
          f"seat={bool(scene.seated_now(scene.target)[0])} "
          f"dock={bool(scene.docked_now()[0])} "
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


def _push_cart(sign: float, pred, *, force: float, vmax: float, iters: int, tag: str) -> bool:
    """Bang-bang velocity-regulated CoM push/pull on the cart along the DEPOT x-axis
    (force written into the scene-owned buffer), with a lateral P-correction steering
    depot-local y toward 0 (keeps the slab centered between the rails). Returns
    pred() at exit; the buffer is always zeroed."""
    scene = _ENV.scene
    buf = scene.cart_force
    ok = False
    for it in range(iters):
        _refresh()
        if pred():
            ok = True
            break
        yaw = _depot_yaw()
        cy, sy = math.cos(yaw), math.sin(yaw)
        p = _cart_local()
        d = [float(sign), max(-0.5, min(0.5, 25.0 * (0.0 - float(p[1]))))]
        nrm = math.hypot(d[0], d[1])
        wx = (cy * d[0] - sy * d[1]) / nrm
        wy = (sy * d[0] + cy * d[1]) / nrm
        v = scene.cart.data.root_lin_vel_w[0]
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
    env = ENVS.get("simgen.dressing_ferry_depot")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg

    torch.manual_seed(1000 * args.seed + 7)
    env.reset()
    _step(60)  # settle the dealt layout

    _refresh()
    dp = scene.depot.data.root_pos_w[0]
    tp = scene.target.data.root_pos_w[0]
    print(f"[solve] seed={args.seed} layout: depot=({float(dp[0]):.3f},{float(dp[1]):.3f}) "
          f"yaw={math.degrees(_depot_yaw()):+.1f}deg swap={bool(scene.swap[0])} "
          f"target=({float(tp[0]):.3f},{float(tp[1]):+.3f}) "
          f"cart={[round(float(v), 3) for v in _cart_local()]}", flush=True)
    _report("start")
    if float(scene.score()[0]) > 0.05 or bool(scene.success()[0]):
        print("[solve] FATAL: nonzero score / success at reset", flush=True)
        _hard_exit(1)
    _mark_score("start")

    # ---- phase 1: EXTRACT — pull the cart out of the garage along the apron ----------------
    ok = _push_cart(-1.0, lambda: float(_cart_local()[0]) <= c.x_pull,
                    force=4.0, vmax=0.10, iters=900, tag="pull")
    _step(30)  # settle
    _refresh()
    if not ok or not bool(scene._extracted[0]):
        print("[solve] FATAL: cart did not extract clear of the roof", flush=True)
        _hard_exit(1)
    _report("extracted")
    _mark_score("extracted")

    # ---- phase 2: LOAD — transport-teleport the bottle above the flare mouth, drop ---------
    # Hover: bottle bottom `drop` above the flare top; upright, zero velocity. The
    # fall through the flare into the snug well is real contact dynamics.
    flare_top = c.slab_t / 2 + c.well_h + c.flare_s * math.sqrt(0.5)
    drop = 0.015
    from isaaclab.utils.math import quat_apply  # noqa: E402, PLC0415

    seated = False
    for attempt in range(3):
        _refresh()
        cq = scene.cart.data.root_quat_w
        off = torch.tensor([[c.well_x, 0.0, 0.0]], device=env.device)
        well_w = scene.cart.data.root_pos_w + quat_apply(cq, off)
        st = torch.zeros(1, 13, device=env.device)
        st[0, 0] = well_w[0, 0]
        st[0, 1] = well_w[0, 1]
        st[0, 2] = scene.cart.data.root_pos_w[0, 2] + flare_top + drop + c.bot_off
        st[0, 3] = 1.0  # upright, identity quat, zero velocity
        scene.target.write_root_state_to_sim(st, torch.tensor([0], device=env.device))
        seated = _settle_until(
            lambda: bool(scene.seated_now(scene.target)[0])
            and bool(scene.settled(scene.target)[0]), max_steps=240)
        if seated:
            break
        # perched on the flare: a gentle downward press finishes the insertion
        print(f"[solve]   load attempt {attempt}: perched, pressing down", flush=True)
        scene.bottle_force[0, 2] = -2.0
        _step(60)
        scene.bottle_force[:] = 0.0
        seated = _settle_until(
            lambda: bool(scene.seated_now(scene.target)[0])
            and bool(scene.settled(scene.target)[0]), max_steps=180)
        if seated:
            break
    if not seated or not bool(scene._loaded[0]):
        print("[solve] FATAL: bottle did not seat in the well", flush=True)
        _hard_exit(1)
    _report("loaded")
    _mark_score("loaded")

    # ---- phase 3: FERRY — push the loaded cart back through the doorway to the stop --------
    # Slow cap: the seated bottle must ride, not slosh; the back wall is the stop.
    ok = _push_cart(+1.0, lambda: bool(scene.docked_now()[0]),
                    force=6.0, vmax=0.06, iters=1400, tag="ferry")
    ok = _settle_until(
        lambda: bool(scene.docked_now()[0]) and bool(scene.settled(scene.cart)[0])
        and bool(scene.seated_now(scene.target)[0])
        and bool(scene.settled(scene.target)[0]), max_steps=360) and ok
    _report("docked")
    if not ok:
        print("[solve] FATAL: loaded cart did not dock settled against the stop", flush=True)
        _hard_exit(1)
    if not bool(scene.success()[0]):
        print("[solve] FATAL: success() not true after docking", flush=True)
        _hard_exit(1)
    _mark_score("docked")

    # ---- persistence: 3+ simulated seconds hands-off, success must still hold --------------
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
