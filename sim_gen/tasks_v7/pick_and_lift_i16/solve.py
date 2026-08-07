"""Teleport solution for BallastLeverScene (sim_gen task `pick_and_lift_i16`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY, in three writes, each ending in FREE SPACE:
  P1/P2/P3 — steel block k is carried from its floor spawn to a hover pose ~15 mm
  ABOVE the open top of the yellow basket (computed in the BEAM'S BODY FRAME from the
  live beam pose, one y-lane per block so three fit side by side), orientation matched
  to the beam so it lands flat. Everything load-bearing then happens through CONTACT
  DYNAMICS: the block falls the last few centimetres under gravity onto the tilted
  basket floor (friction holds it — the tilt is well inside the friction cone), and
  after the third block the accumulated BALLAST TORQUE — real weight on real contact
  at a real lever arm — overcomes the caged cargo's torque and swings the beam onto
  its raised stop, hoisting the cage. No block is ever teleported into the basket, the
  beam is never pushed, pulled, or posed by hand, and the red cargo is never touched.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.pick_and_lift_i16.solve --headless [--seed N]
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
    env = ENVS.get("simgen.ballast_lever_lift")().build(num_envs=args.num_envs, device=device)
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

    def report(tag: str) -> None:
        rs = float(scene.raised_sin_now()[0])
        k_now = int(scene.steel_in_basket()[0].sum())
        print(f"[solve] {tag:12s} | raised_sin={rs:+.3f} (need {c.raised_sin:+.3f}, "
              f"stop {c.sin_stop:+.3f}) k_in_basket={k_now} "
              f"cargo_caged={bool(scene.cargo_in_cage()[0])} "
              f"seated={bool(scene.beam_seated()[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def drop_block(j: int, y_lane: float) -> None:
        """TRANSPORT teleport: steel block j to a free-space hover above the open
        basket top (beam frame, live pose), orientation = beam orientation, zero
        velocity. The load happens by gravity + contact after this write."""
        from isaaclab.utils.math import quat_apply

        bq = scene.beam.data.root_quat_w
        bp = scene.beam.data.root_pos_w
        # local z: wall top (tray_floor + wall_h) + block half + 15 mm clearance
        loc = torch.tensor(
            [-c.arm, y_lane, c.tray_floor_z + c.wall_h + c.block_size / 2 + 0.015],
            device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = bp + quat_apply(bq, loc)  # root_pos_w is already world (origin in)
        st[:, 3:7] = bq
        scene.steel[j].write_root_state_to_sim(st, all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    cp = (scene.cradle.data.root_pos_w - scene.env_origins)[0]
    q = scene.cradle.data.root_quat_w[0]
    yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
    blocks = " ".join(
        f"s{j}=({float(p[0]):+.3f},{float(p[1]):+.3f})"
        for j, p in enumerate((scene.steel_pos_w() - scene.env_origins[:, None, :])[0]))
    fp = (scene.foam.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): cradle=({float(cp[0]):+.3f},"
          f"{float(cp[1]):+.3f}) yaw={math.degrees(yaw):+.1f}deg {blocks} "
          f"foam=({float(fp[0]):+.3f},{float(fp[1]):+.3f}) "
          f"d0={[round(float(v), 3) for v in scene.d0[0]]}", flush=True)
    report("reset")
    prev = print_score("P0 reset+settle")

    # ---------------- phases 1-3: ballast blocks, one per lane -----------------------------
    # Lanes across the basket's wide axis (inner y 130 mm, blocks 40 mm): three fit
    # side by side. Each drop is ~50 mm of free fall onto the tilted basket floor;
    # the third one supplies the torque that tips the beam, so give it a long
    # hands-off settle while the swing plays out on real contact.
    lanes = (-0.042, 0.0, 0.042)
    for k, y_lane in enumerate(lanes):
        drop_block(k, y_lane)
        step(150 if k < 2 else 480)  # 1.25 s; 4 s for the tipping swing
        report(f"ballast-{k + 1}")
        s = print_score(f"P{k + 1} steel block {k + 1} dropped into basket")
        assert s >= prev - 1e-6, f"score decreased across ballast drop {k + 1}"
        prev = s
        if not bool(scene.steel_in_basket()[0, k]):
            print(f"SIM_GEN_SOLVE: FAIL (block {k} not in basket)", flush=True)
            os._exit(1)

    if not bool(scene.success()[0]):
        step(240)  # grace: swing may still be finishing
        report("grace")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (beam did not tip to raised after 3 blocks)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, no intervention) -----
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= prev - 1e-6
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
