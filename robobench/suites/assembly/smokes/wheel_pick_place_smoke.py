"""Smoke test for WheelPickPlaceScene — visual or headless, no real robot.

One linear run with a NullRobot. It can't pick and place, so it cheats the transport by teleport:
  1. rest   — leave the reset layout as is and let the wheel settle on the table top;
  2. lift   — by direct sim write, raise the wheel clear of the table (it is a free rigid body, so
              this also proves it is not stuck in the table's mesh collider);
  3. drop   — set it flat above the basket's interior centre and release it;
  4. settle — let it fall in and come to rest, then read out `placed()`.

What this actually checks: the packing-table set piece spawns with working collision (the wheel rests
instead of sinking), the `container_h20` basket that came with it catches and holds the wheel (its
five box colliders are live), and the scene's `placed()` / `settled()` / `dropped()` readouts agree
with what the wheel is visibly doing. It also prints the wheel's resting height above the table top —
use it to re-calibrate `WheelPickPlaceSceneCfg.wheel_init_z` if the asset ever changes.

python -m robobench.suites.assembly.smokes.wheel_pick_place_smoke --livestream 2
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
    from robobench.suites.assembly.scenes import WheelPickPlaceScene

# Phase boundaries, cumulative sim steps (the scene runs at dt=1/200, and a NullRobot's control
# period is one physics step, so 200 steps = 1 s).
REST_END, LIFT_END, DROP_END, END = 400, 700, 1000, 1800
LIFT_H = 0.35  # how far above the table top the teleport lifts the wheel (m)
DROP_H = 0.04  # release height above the basket's interior floor (m)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()  # populate the registries before resolving by name
    env = ENVS.get("assembly.wheel_pick_place")().build(num_envs=args.num_envs, device=device)
    scene: WheelPickPlaceScene = env.scene  # type: ignore[assignment]
    c = scene.cfg
    render = (not args.headless) or livestream_on
    n = env.num_envs
    no_action = torch.empty(0, device=device)  # NullRobot ignores it
    all_ids = torch.arange(n, device=device)

    # Where the basket's interior centre and floor are, in env-local coords — derived from the
    # scene's own basket constants, so this follows the table if it is ever moved.
    wx, wy = c.workbench_pos
    basket_xy = (wx + sum(c.BASKET_INNER_X) / 2, wy + sum(c.BASKET_INNER_Y) / 2)
    basket_floor = c.surface_z + (c.BASKET_FLOOR_Z - c.TABLE_TOP_Z)
    print(f"WHEEL-PICK-PLACE | table top {c.surface_z:.3f} | basket floor {basket_floor:.3f} "
          f"at ({basket_xy[0]:.3f}, {basket_xy[1]:.3f}) | target box {c.place_box_world()}", flush=True)

    env.reset()

    def teleport(xy: tuple[float, float], z: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins + torch.tensor((xy[0], xy[1], z), device=device)
        st[:, 3] = 1.0  # identity quat -> flat, as it started
        scene.wheel.write_root_state_to_sim(st, all_ids)

    for i in range(1, END + 1):
        if i == REST_END:  # lift clear of the table, still over its start xy
            teleport(c.wheel_init_xy, c.surface_z + LIFT_H)
        elif i == LIFT_END:  # hold it flat above the basket mouth
            teleport(basket_xy, basket_floor + DROP_H)
        elif LIFT_END < i < DROP_END:  # keep it hovering (no gravity fight — just re-write the pose)
            teleport(basket_xy, basket_floor + DROP_H)

        env.step(no_action, render=render)  # NullRobot ignores the action

        if i % 200 == 0:
            pos, _ = scene.wheel_pose()
            h = (pos[:, 2] - c.surface_z) * 1e3
            print(f"  step {i:4d} | wheel xy ({pos[0, 0]:+.3f},{pos[0, 1]:+.3f}) | height above table mm: "
                  f"min={h.min():+.1f} mean={h.mean():+.1f} max={h.max():+.1f} | placed "
                  f"{int(scene.placed().sum())}/{n} | settled {int(scene.settled().sum())}/{n}", flush=True)

    pos, _ = scene.wheel_pose()
    rest_h = (pos[:, 2] - basket_floor) * 1e3
    print(f"WHEEL-PICK-PLACE | placed {int(scene.placed().sum())}/{n} envs | dropped "
          f"{int(scene.dropped().sum())}/{n} | wheel above basket floor mm: min={rest_h.min():+.1f} "
          f"mean={rest_h.mean():+.1f} max={rest_h.max():+.1f}", flush=True)
    close_and_exit(env, app)


if __name__ == "__main__":
    main()
