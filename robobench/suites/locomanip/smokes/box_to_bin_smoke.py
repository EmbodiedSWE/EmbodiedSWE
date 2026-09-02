"""Smoke test for BoxToBinScene — visual or headless, no real robot.

One linear run with a NullRobot. It can neither pick nor walk, so it cheats the whole task by
teleport, which is exactly what makes this a test of the SCENE and nothing else:
  1. rest    — leave the reset layout as is and let the box settle on its shelf board;
  2. lift    — by direct sim write, raise the box clear of the board (it is a free rigid body, so
               this also proves it is not stuck in the shelf's mesh collider, and that the board
               above leaves room to lift);
  3. transit — sweep it along the start -> bin line at carry height, one step at a time, so the
               radial `carried_fraction` readout and the `transited` latch are exercised on a
               moving box rather than a jump (the line exits through the shelf's front opening);
  4. drop    — hold it over the bin's interior centre just above its resting height, then release;
  5. settle  — let it fall in and come to rest, then read out `success()` / `score()`.

What this actually checks: the extracted shelf spawns with working collision at the measured board
height; the scaled bin on the z-scaled table catches and holds the box at the measured interior
height; and the scene's `placed()` / `settled()` / `dropped()` / `carried_fraction()` readouts and
the journey latches agree with what the box is visibly doing. Negative controls: the box must not
read as placed on its board; `transited` must not latch before the crossing; a box held INSIDE the
bin's xy bands but above `place_max_h` (rim height — the "straddling the rim" impostor) must not
read as placed; and a state teleported straight into the bin must read `placed()` but NOT
`success()` (the journey latches are the proof of work). It also prints the box's resting height
above the board and above the bin floor — use those to re-calibrate `BoxToBinSceneCfg.pick_board_z`
/ `BIN_FLOOR_Z` if an asset ever changes.

python -m robobench.suites.locomanip.smokes.box_to_bin_smoke --livestream 2
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
from robobench.suites.locomanip.smokes import close_and_exit  # noqa: E402

if TYPE_CHECKING:
    from robobench.suites.locomanip.scenes import BoxToBinScene

# Phase boundaries, cumulative sim steps (the scene runs at dt=1/200, and a NullRobot's control
# period is one physics step, so 200 steps = 1 s).
REST_END, LIFT_END, TRANSIT_END, RIM_END, DROP_END, END = 400, 600, 1400, 1600, 1900, 2700
LIFT_H = 0.30  # teleport lift above the box's resting height (m) — the board above leaves 0.39
CARRY_Z = 0.95  # carry height for the transit sweep (through the shelf's front opening)
DROP_H = 0.04  # release height above the box's in-bin resting height (m)
RIM_HOLD = 0.16  # held height above the table top for the rim negative control (> place_max_h)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()  # populate the registries before resolving by name
    env = ENVS.get("locomanip.box_to_bin")().build(num_envs=args.num_envs, device=device)
    scene: BoxToBinScene = env.scene  # type: ignore[assignment]
    c = scene.cfg
    render = (not args.headless) or livestream_on
    n = env.num_envs
    no_action = torch.empty(0, device=device)  # NullRobot ignores it
    all_ids = torch.arange(n, device=device)

    box0 = c.box_init_world()
    (ix, iy), _, bin_floor, bin_rim = c.bin_interior()
    rest_in_bin = bin_floor + c.BOX_HALF  # where the box's origin sits once seated in the bin
    print(f"BOX-TO-BIN | board top {c.pick_board_z:.3f}, box start {box0} | table top "
          f"{c.table_top_z:.3f}, bin floor {bin_floor:.3f} rim {bin_rim:.3f} at {c.bin_pos} | "
          f"interior x {ix} y {iy} | target box {c.place_box_world()}", flush=True)

    env.reset()

    def teleport(xy: tuple[float, float], z: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins + torch.tensor((xy[0], xy[1], z), device=device)
        st[:, 3] = 1.0  # identity quat -> flat, as it started
        scene.box.write_root_state_to_sim(st, all_ids)

    fails = []
    board_rest_mm = None
    for i in range(1, END + 1):
        if i == REST_END:
            # Calibration: where the box actually rests on the board (re-derive box_init_world).
            pos, _ = scene.box_pose()
            board_rest_mm = (pos[:, 2] - c.BOX_HALF - c.pick_board_z) * 1e3
            # NEGATIVE CONTROLS on the untouched layout.
            if int(scene.placed().sum()):
                fails.append("placed() true while the box is still on its shelf board")
            if int(scene._lifted.sum()) or int(scene._transited.sum()):
                fails.append("journey latches fired before the box moved")
            # A state teleported straight into the bin must be placed but NOT a success —
            # exercised on a snapshot so the main run's latches stay clean.
            st = env.get_states()
            teleport(c.bin_pos, rest_in_bin + 0.001)
            env.step(no_action, render=False)
            for _ in range(60):  # let it seat and settle
                env.step(no_action, render=False)
            if not int(scene.placed().sum()):
                fails.append("placed() false for a box seated in the bin (bands or heights wrong)")
            if int(scene.success().sum()):
                fails.append("success() true for a box that never made the journey (latches leak)")
            env.set_states(st)
            teleport((box0[0], box0[1]), box0[2] + LIFT_H)  # begin the lift, over its start xy
        elif LIFT_END <= i < TRANSIT_END:  # sweep along the start -> bin line at carry height
            f = (i - LIFT_END) / (TRANSIT_END - LIFT_END)
            teleport((box0[0] + f * (c.bin_pos[0] - box0[0]), box0[1] + f * (c.bin_pos[1] - box0[1])),
                     CARRY_Z)
        elif TRANSIT_END <= i < RIM_END:  # NEGATIVE CONTROL: inside the bin's xy, held at rim height
            teleport(c.bin_pos, c.table_top_z + RIM_HOLD)
        elif i == RIM_END:
            if int(scene.placed().sum()):
                fails.append("placed() true for a box held at rim height (place_max_h too loose)")
            teleport(c.bin_pos, rest_in_bin + DROP_H)
        elif RIM_END < i < DROP_END:  # hold it just above its seat, then let go
            teleport(c.bin_pos, rest_in_bin + DROP_H)

        env.step(no_action, render=render)  # NullRobot ignores the action

        if i % 250 == 0:
            pos, _ = scene.box_pose()
            print(f"  step {i:4d} | box ({pos[0, 0]:+.3f},{pos[0, 1]:+.3f},{pos[0, 2]:+.3f}) | "
                  f"carried {float(scene.carried_fraction().min()):.2f} | "
                  f"lifted {int(scene._lifted.sum())}/{n} transited {int(scene._transited.sum())}/{n} "
                  f"| placed {int(scene.placed().sum())}/{n} | score {float(scene.score().min()):.0f}",
                  flush=True)

    pos, _ = scene.box_pose()
    rest_h = (pos[:, 2] - c.BOX_HALF - bin_floor) * 1e3
    ok = int(scene.success().sum())
    if ok != n:
        fails.append(f"success() {ok}/{n} after a staged, teleported run — the scene cannot be solved")
    if int(scene.dropped().sum()):
        fails.append("dropped() true with the box resting in the bin")

    # get_state/set_state round-trip, latches included: read, perturb, restore, re-read.
    st = env.get_states()
    teleport((box0[0], box0[1]), box0[2] + LIFT_H)
    env.step(no_action, render=False)
    env.set_states(st)
    env.step(no_action, render=False)
    if int(scene.success().sum()) != n:
        fails.append("set_states() did not restore a solved state (latches or box pose lost)")

    print(f"BOX-TO-BIN | success {ok}/{n} | score {float(scene.score().min()):.0f} | dropped "
          f"{int(scene.dropped().sum())}/{n} | box rest above board mm: min={board_rest_mm.min():+.1f} "
          f"max={board_rest_mm.max():+.1f} | above bin floor mm: min={rest_h.min():+.1f} "
          f"mean={rest_h.mean():+.1f} max={rest_h.max():+.1f}", flush=True)
    print("BOX-TO-BIN SMOKE:", "FAIL — " + "; ".join(fails) if fails else "PASS", flush=True)
    close_and_exit(env, app)


if __name__ == "__main__":
    main()
