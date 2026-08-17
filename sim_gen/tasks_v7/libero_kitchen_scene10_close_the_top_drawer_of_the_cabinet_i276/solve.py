"""Teleport solution for RamChuteCabinetScene (sim_gen task
`libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_i276`) — the task's
legitimacy certificate.

This solve applies ZERO forces and ZERO wrenches — ever. Its ONE teleport is pure
TRANSPORT: the ram ball is lifted out of its dock tray and released (zero velocity)
just above the chute's upper mouth, exactly what a hand does when it picks the ball
up, carries it over, and lets go. Everything judged happens through contact
dynamics under gravity:

1. PERCEPTION: station pose/yaw and the drawer's sampled opening q0 are read back
   from the episode state — never hard-coded. The release point is the scene's own
   `drop_point()`, mapped through the MEASURED station pose.
2. TRANSPORT (the single teleport): ball moved from the dock tray to the release
   point above the drop station, velocity zero. The drawer has not moved — only
   delivery credit is earned.
3. GRAVITY RAM (hands off, no actuation at all): the ball falls ~12 mm onto the
   chute, rolls down the 15 deg incline, shoots across the runway through the
   guard tunnel, strikes the drawer's front face, and its momentum drives the
   drawer to its rear hard stop. Restitution 0 everywhere: the ball ends parked
   at rest against the closed drawer face. The judged "drawer closes" outcome is
   delivered ENTIRELY by the machine — no wrench ever touches anything.
4. The settled end state is judged LIVE: drawer seated in its channel AND ball
   parked against it.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the rubric
latches), then holds HANDS-OFF >= 3 simulated seconds after success() first turns
True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_i276.solve --headless [--seed N]
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
    env = ENVS.get("simgen.ram_chute_cabinet")().build(num_envs=args.num_envs, device=device)
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

    def ball_local():
        return scene._station_local(scene.ball.data.root_pos_w)[0]

    def report(tag: str) -> None:
        bl = ball_local()
        print(f"[solve] {tag:12s} | q={float(scene.drawer_q()[0]):+.4f} "
              f"ball_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f},{float(bl[2]):+.3f}) "
              f"|v_ball|={float(scene.ball.data.root_lin_vel_w[0].norm()):.3f} "
              f"delivered={bool(scene.ball_delivered_now()[0])} "
              f"parked={bool(scene.ball_parked()[0])} "
              f"drawer_closed={bool(scene.drawer_closed()[0])} "
              f"in_channel={bool(scene.drawer_in_channel()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: settle, layout readback, baseline ---------------------------------
    step(150)   # drawer seats on the slab; ball rests in its dock tray
    sp0 = (scene.station.data.root_pos_w - scene.env_origins)[0]
    sq0 = scene.station.data.root_quat_w[0]
    yaw = math.degrees(2.0 * math.atan2(float(sq0[3]), float(sq0[0])))
    q0 = float(scene.q0[0])
    q_meas = float(scene.drawer_q()[0])
    bl = ball_local()
    print(f"[solve] layout readback (seed {args.seed}): "
          f"station=({float(sp0[0]):+.3f},{float(sp0[1]):+.3f}) yaw~{yaw:+.1f}deg "
          f"q0={q0:+.4f} (measured {q_meas:+.4f}) "
          f"ball_dock=({float(bl[0]):+.3f},{float(bl[1]):+.3f})", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert bool(scene.drawer_in_channel()[0]), "drawer must ride in its channel"
    assert not bool(scene.drawer_closed()[0]), "drawer must start open"
    assert not bool(scene.ball_delivered_now()[0]), "ball must start in its dock"
    assert not bool(scene.ball_parked()[0]), "ball must start away from the drawer"
    assert abs(q_meas - q0) < 0.010, "drawer must rest at its sampled opening"
    assert abs(float(bl[0]) - c.dock_x) < c.dock_jitter + 0.010 and \
        abs(float(bl[1]) - c.dock_y) < c.dock_jitter + 0.010, \
        "ball must rest in its jittered dock tray"
    s_prev = print_score("P0 reset+settle (ball docked, drawer out)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s_prev <= 0.05, f"baseline score should be ~0, got {s_prev}"

    # ---------------- phase 1: TRANSPORT — the single teleport (pick, carry, release) ------------
    # Ball centre moved to the scene's drop point above the chute's upper mouth,
    # mapped through the MEASURED station pose. Velocity zero: a hand letting go.
    dp_loc = torch.tensor([c.drop_point()], device=device).repeat(n, 1)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.station.data.root_pos_w + quat_apply(scene.station.data.root_quat_w, dp_loc)
    st[:, 3] = 1.0
    scene.ball.write_root_state_to_sim(st)
    step(10)    # ~0.08 s: the ball has landed on the chute surface and started rolling
    report("released")
    q_after = float(scene.drawer_q()[0])
    assert abs(q_after - q_meas) < 0.005, \
        f"drawer must not have moved at release (q {q_meas:+.4f} -> {q_after:+.4f})"
    assert bool(scene.ball_delivered_now()[0]), "ball must be in the chute corridor"
    s = print_score("P1 ball released into the chute mouth (drawer untouched)")
    assert s >= s_prev - 1e-6, "score decreased across P1"
    assert 0.28 <= s <= 0.40, f"P1 must earn delivery credit only, got {s:.3f}"
    s_prev = s

    # ---------------- phase 2: GRAVITY RAM — hands off, physics closes the drawer ----------------
    # The ball rolls down the incline, crosses the runway, strikes the drawer front
    # and drives it to the rear hard stop. Poll; no actuation of any kind.
    closed_at = -1
    for i in range(720):    # up to 6 s — the descent+impact takes ~1.5 s
        env.step(no_action)
        if closed_at < 0 and bool(scene.drawer_closed()[0]):
            closed_at = i
        if closed_at >= 0 and bool(scene.settled()[0]):
            break
    print(f"[solve] ram: drawer first closed at step {closed_at}", flush=True)
    report("ram")
    assert closed_at >= 0, \
        f"gravity ram failed to close the drawer (q={float(scene.drawer_q()[0]):+.4f})"
    step(120)   # hands-off settle: everything comes to rest
    report("settled")
    assert bool(scene.drawer_closed()[0]), \
        f"drawer did not stay seated: q={float(scene.drawer_q()[0]):+.4f}"
    assert bool(scene.ball_parked()[0]), \
        "ball must end parked against the closed drawer face"
    assert bool(scene.success()[0]), "success() must hold on the settled end state"
    s = print_score("P2 gravity ram: drawer seated, ball parked, all hands off")
    assert s >= s_prev - 1e-6, "score decreased across P2"
    assert s >= 1.0 - 1e-6, "success must score 1.0"
    s_prev = s

    # ---------------- persistence (>= 3 simulated seconds, hands-off) ----------------------------
    hold_ok = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold_ok = hold_ok and bool(scene.success()[0])
    report("persist")
    s_final = print_score("P-final persistence 3.3 s")
    ok = hold_ok and bool(scene.success()[0]) and s_final >= s_prev - 1e-6
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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
