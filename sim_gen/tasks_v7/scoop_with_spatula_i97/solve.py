"""Teleport solution for WeighbridgeScene (sim_gen task `scoop_with_spatula_i97`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, one cube at a time): a single root-state write carries an
   IRON cube from its ground scatter slot to a free-space HOVER pose a few
   centimetres above the open pan cavity (computed in the BEAM'S BODY FRAME, so the
   hover tracks the tilted pan). The write sets zero velocity and satisfies no
   rubric clause by itself — it is exactly the carry a gripper performs.
2. LOADING (gravity + contact, hands-off): from the hover the cube FALLS into the
   real pan tray and settles against its floor and walls. Nothing is ever written
   into the pan.
3. ACTUATION (pure lever physics, never touched): with ONE iron cube aboard the
   counterweight still wins and the beam stays on its rest stop — verified and
   scored mid-way. The SECOND iron cube's weight tips the balance: the beam swings
   through horizontal to the opposite hard stop carrying the load with it. The
   swing, the stop, and the hold are 100 % contact/joint dynamics; the beam is
   never teleported or forced.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.scoop_with_spatula_i97.solve --headless [--seed N]
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

_qy, _qz = scene_mod._qy, scene_mod._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.weighbridge")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

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
        th = float(scene.theta_deg()[0])
        cnt = int(scene.iron_in_pan()[0])
        print(f"[solve] {tag:12s} | theta={th:+6.2f}deg iron_in_pan={cnt} "
              f"appr={bool(scene._appr[0])} i1={bool(scene._iron1[0])} "
              f"i2={bool(scene._iron2[0])} cross={bool(scene._cross[0])} "
              f"high={bool(scene._high[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def hover_pose(y_off: float):
        """World hover pose for a cube: in the BEAM frame, above the pan cavity
        centre — clear of the walls, tracking the pan's current tilt."""
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = c.pan_x
        loc[:, 1] = y_off
        loc[:, 2] = c.arm_t / 2 + c.pan_floor_t + c.pan_wall_h + c.cube_size / 2 + 0.030
        pos = scene.beam.data.root_pos_w + quat_apply(scene.beam.data.root_quat_w, loc)
        return pos

    def settle(max_steps: int = 600, tail: int = 60) -> None:
        for i in range(max_steps):
            env.step(no_action)
            if i > 30 and bool(scene.settled()[0]):
                break
        step(tail)  # extra hands-off settle

    def load_iron(name: str, y_off: float, tag: str) -> None:
        """TRANSPORT the named iron cube to the free-space hover above the pan
        (zero velocity), then hands-off fall + settle. Up to 3 attempts (a bounce
        out of the tray is re-carried — the final state is still contact-made)."""
        want = int(scene.iron_in_pan()[0]) + 1
        for attempt in range(3):
            st = torch.zeros(n, 13, device=device)
            st[:, 0:3] = hover_pose(y_off)
            st[:, 3] = 1.0
            scene.cubes[name].write_root_state_to_sim(st, all_ids)
            settle()
            if int(scene.iron_in_pan()[0]) >= want:
                return
            print(f"[solve] {tag}: cube not in the pan after attempt {attempt + 1}; "
                  f"re-carrying", flush=True)
            report(f"{tag}-retry")
        report(f"{tag}-FAIL")
        print(f"SIM_GEN_SOLVE: FAIL ({tag} never landed in the pan)", flush=True)
        os._exit(1)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # the beam drops onto its rest stop; cubes seat on the ground
    slot_of = scene.slot_of[0].tolist()
    cubes_str = " ".join(
        f"{nm}=({float(scene.cubes[nm].data.root_pos_w[0, 0] - scene.env_origins[0, 0]):+.3f},"
        f"{float(scene.cubes[nm].data.root_pos_w[0, 1] - scene.env_origins[0, 1]):+.3f})"
        for nm in scene.CUBE_NAMES)
    print(f"[solve] layout readback (seed {args.seed}): slot_of={slot_of} {cubes_str}",
          flush=True)
    report("reset")
    th0 = float(scene.theta_deg()[0])
    assert th0 < -(c.limit_deg - 3.0), \
        f"beam must rest counterweight-down near -{c.limit_deg} deg, got {th0:.2f}"
    pos, _v = scene._cube_tensors(scene.CUBE_NAMES)
    assert torch.isfinite(pos).all(), "NaN/inf in cube states after settle"
    hz = pos[0, :, 2] - scene.env_origins[0, 2]
    assert bool((hz > 0.005).all() and (hz < 0.05).all()), \
        f"cubes must lie on the ground, z={hz.tolist()}"
    s0 = print_score("P0 reset+settle (beam at rest, cubes scattered on the ground)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: first iron cube into the pan --------------------------------
    # One iron cube is NOT enough: the counterweight must still win. This midpoint
    # is itself part of the certificate (the accumulation claim).
    load_iron("iron_0", -0.024, "iron-A")
    report("iron-A")
    assert int(scene.iron_in_pan()[0]) >= 1, "first iron cube must rest in the pan"
    th1 = float(scene.theta_deg()[0])
    assert th1 < 0.0, f"one iron cube must not tip the beam, theta={th1:.2f}"
    assert not bool(scene.success()[0]), "cannot be success with one iron cube"
    s1 = print_score("P1 one iron cube in the pan (beam still counterweight-down)")
    assert s1 >= s0 - 1e-6 and s1 >= 0.24, f"P1 score {s1} (expect appr+iron1=0.25)"

    # ---------------- phase 2: second iron cube -> the beam tips ----------------------------
    load_iron("iron_1", +0.024, "iron-B")
    settle(max_steps=900, tail=120)  # the swing + stop chatter takes a moment
    report("iron-B")
    assert int(scene.iron_in_pan()[0]) >= c.need_iron, \
        "both iron cubes must ride inside the pan"
    th2 = float(scene.theta_deg()[0])
    assert th2 >= c.tip_deg, \
        f"two iron cubes must tip the beam to the stop, theta={th2:.2f}"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the second iron cube)", flush=True)
        os._exit(1)
    s2 = print_score("P2 second iron cube aboard: the beam swung to the pan-side stop")
    assert s2 >= s1 - 1e-6, "score decreased across the tip"

    # ---------------- phase 3: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.3 s (load holds the beam on its stop)")
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
