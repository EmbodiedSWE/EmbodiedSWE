"""Smoke test for NutThreadAssemblyScene — visual or headless, no real robot.

One linear run with a NullRobot. It can't pick-and-place, so it cheats the staging by teleport:
  1. show   — leave the reset layout as is (nuts resting flat on the table beside the fixed bolts);
  2. stage  — by direct sim write, lift each nut up and centre it upright just above its bolt's threads;
  3. screw  — press + twist each nut down its bolt until it seats;
  4. settle — let the nut(s) come to rest, then read out seating.

Bodies are placed / driven straight through the scene handles (env.scene.nuts / .bolts); the
NullRobot applies nothing. Watch with --livestream; --headless runs the same motion non-visually.

python -m robobench.suites.assembly.smokes.nut_thread_assembly_smoke --livestream 2
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
livestream_on = args.livestream > 0

app = AppLauncher(args).app

from typing import TYPE_CHECKING

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402
from robobench.suites.assembly.smokes import close_and_exit  # noqa: E402

if TYPE_CHECKING:
    from robobench.suites.assembly.scenes import NutThreadAssemblyScene

# Assemble: press down (-z) + twist clockwise about the bolt axis, capping the spin rate so the nut
# threads down steadily. TARGET_WZ is the spin-rate cap: twist only while the nut spins slower than this.
PRESS, TWIST, TARGET_WZ = -2.5, -0.15, -3.0
# Height above each bolt's origin (its base) to drop the nut in when staging it (m) — just above the
# threads so they engage on the first press.
STAGE_GAP = 0.026
# Phase boundaries, cumulative sim steps: show -> stage(teleport) -> screw -> settle.
SHOW_END, ASSEMBLE_END, END = 150, 1500, 1600


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Build the registered config (scene + NullRobot + the scene's own sim), then drive the scene
    # directly (teleport each nut onto its bolt, screw it down) and read out seating.
    robobench.discover()  # populate the registries (robots + suites) before resolving by name
    env = ENVS.get("assembly.nut_thread")().build(num_envs=args.num_envs, device=device)
    scene: NutThreadAssemblyScene = env.scene  # type: ignore[assignment]
    render = (not args.headless) or livestream_on
    n, npairs = env.num_envs, scene.cfg.num_pairs
    no_action = torch.empty(0, device=device)  # NullRobot ignores it
    all_ids = torch.arange(n, device=device)

    env.reset()

    for i in range(1, END + 1):
        if i == SHOW_END:  # stage each nut upright just above its bolt's threads
            for k, nut in enumerate(scene.nuts):
                bolt_pos = scene.bolts[k].data.root_pos_w  # (n, 3), already world (incl. env origin)
                st = torch.zeros(n, 13, device=device)
                st[:, 0:3] = bolt_pos
                st[:, 2] += STAGE_GAP  # bolt origin is at its base, so this lands just above the threads
                st[:, 3] = 1.0  # identity quat -> screw axis up, aligned with the bolt
                nut.write_root_state_to_sim(st, all_ids)
        elif SHOW_END < i < ASSEMBLE_END:  # press + twist (capped spin) to screw each nut down its bolt
            f = torch.zeros(n, 3, device=device)
            f[:, 2] = PRESS
            for nut in scene.nuts:
                t = torch.zeros(n, 3, device=device)
                t[nut.data.root_ang_vel_w[:, 2] > TARGET_WZ, 2] = TWIST
                nut.set_external_force_and_torque(f.unsqueeze(1), t.unsqueeze(1))

        env.step(no_action, render=render)  # NullRobot ignores the action

        if i % 150 == 0:  # progress: nut0 height above its bolt across ALL envs (min/mean/max, mm)
            h = (scene.nuts[0].data.root_pos_w - scene.bolts[0].data.root_pos_w)[:, 2] * 1e3
            print(f"  step {i:4d} | height mm: min={h.min():+.1f} mean={h.mean():+.1f} max={h.max():+.1f}", flush=True)

    # Verdict across ALL envs: how many fully seated, and the spread of final nut0 heights above the
    # bolt base (consistent threading -> all near the seated depth; flaky -> a wide spread / some high).
    seated = scene.seated()  # (num_envs, num_pairs)
    h0 = (scene.nuts[0].data.root_pos_w - scene.bolts[0].data.root_pos_w)[:, 2] * 1e3
    print(f"NUT-THREAD | seated {int(seated.all(dim=1).sum())}/{n} envs | seat_z={scene.cfg.seat_z * 1e3:.0f}mm "
          f"| nut0 height mm: min={h0.min():+.1f} mean={h0.mean():+.1f} max={h0.max():+.1f}", flush=True)
    close_and_exit(env, app)


if __name__ == "__main__":
    main()
