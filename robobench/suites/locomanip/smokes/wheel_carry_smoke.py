"""Smoke test for WheelCarryScene — visual or headless, no real robot.

One linear run with a NullRobot. It can neither pick nor walk, so it cheats the whole task by
teleport, which is exactly what makes this a test of the SCENE and nothing else:
  1. rest    — leave the reset layout as is and let the wheel settle on the PICK table top;
  2. lift    — by direct sim write, raise the wheel clear of that table (it is a free rigid body, so
               this also proves it is not stuck in the table's mesh collider);
  3. transit — sweep it across the `carry_dx` gap at carry height, one step at a time, so the
               `carried_fraction` readout and the `transited` latch are exercised on a moving wheel
               rather than a jump;
  4. drop    — set it flat above the basket's interior centre and release it;
  5. settle  — let it fall in and come to rest, then read out `success()` / `score()`.

What this actually checks: BOTH table stations spawn with working collision at the same height
(the two USDs are authored at different scales, so this is a real question, not a formality); the
`container_h20` basket on the PLACE table catches and holds the wheel; and the scene's
`placed()` / `settled()` / `dropped()` / `carried_fraction()` readouts and the journey latches agree
with what the wheel is visibly doing. Negative controls run first: the wheel is checked NOT to read
as placed while it sits on the pick table, and the `transited` latch is checked not to fire until
the wheel has actually crossed. It also prints the wheel's resting height above each surface — use
those to re-calibrate `WheelCarrySceneCfg.wheel_init_z` if an asset ever changes.

python -m robobench.suites.locomanip.smokes.wheel_carry_smoke --livestream 2
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
    from robobench.suites.locomanip.scenes import WheelCarryScene

# Phase boundaries, cumulative sim steps (the scene runs at dt=1/200, and a NullRobot's control
# period is one physics step, so 200 steps = 1 s).
REST_END, LIFT_END, TRANSIT_END, DROP_END, END = 400, 600, 1400, 1700, 2500
LIFT_H = 0.35  # how far above the table top the teleport lifts the wheel (m)
DROP_H = 0.04  # release height above the basket's interior floor (m)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()  # populate the registries before resolving by name
    env = ENVS.get("locomanip.wheel_carry")().build(num_envs=args.num_envs, device=device)
    scene: WheelCarryScene = env.scene  # type: ignore[assignment]
    c = scene.cfg
    render = (not args.headless) or livestream_on
    n = env.num_envs
    no_action = torch.empty(0, device=device)  # NullRobot ignores it
    all_ids = torch.arange(n, device=device)

    # Where the basket's interior centre and floor are, in env-local coords — derived from the
    # scene's own basket constants, so this follows the PLACE table if it is ever moved.
    qx, qy = c.place_pos
    basket_xy = (qx + sum(c.BASKET_INNER_X) / 2, qy + sum(c.BASKET_INNER_Y) / 2)
    basket_floor = c.surface_z + (c.BASKET_FLOOR_Z - c.TABLE_TOP_Z)
    wheel0 = c.wheel_init_world()
    print(f"WHEEL-CARRY | pick {c.pick_pos} -> place {c.place_pos} (carry_dx {c.carry_dx:.2f} m) | "
          f"tops {c.surface_z:.3f} | basket floor {basket_floor:.3f} at "
          f"({basket_xy[0]:.3f}, {basket_xy[1]:.3f}) | target box {c.place_box_world()}", flush=True)

    env.reset()

    def teleport(xy: tuple[float, float], z: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins + torch.tensor((xy[0], xy[1], z), device=device)
        st[:, 3] = 1.0  # identity quat -> flat, as it started
        scene.wheel.write_root_state_to_sim(st, all_ids)

    fails = []
    for i in range(1, END + 1):
        if i == REST_END:  # NEGATIVE CONTROL: on the pick table, nothing should have latched
            if int(scene.placed().sum()):
                fails.append("placed() true while the wheel is still on the pick table")
            if int(scene._transited.sum()):
                fails.append("transited latched before the wheel moved")
            teleport((wheel0[0], wheel0[1]), c.surface_z + LIFT_H)  # lift clear, over its start xy
        elif LIFT_END <= i < TRANSIT_END:  # sweep it across the gap at carry height
            f = (i - LIFT_END) / (TRANSIT_END - LIFT_END)
            teleport((wheel0[0] + f * c.carry_dx, wheel0[1] + f * (basket_xy[1] - wheel0[1])),
                     c.surface_z + LIFT_H)
        elif TRANSIT_END <= i < DROP_END:  # hold it flat above the basket mouth
            teleport(basket_xy, basket_floor + DROP_H)

        env.step(no_action, render=render)  # NullRobot ignores the action

        if i % 250 == 0:
            pos, _ = scene.wheel_pose()
            h = (pos[:, 2] - c.surface_z) * 1e3
            print(f"  step {i:4d} | wheel xy ({pos[0, 0]:+.3f},{pos[0, 1]:+.3f}) | above tops mm: "
                  f"min={h.min():+.1f} max={h.max():+.1f} | carried "
                  f"{float(scene.carried_fraction().min()):.2f} | lifted {int(scene._lifted.sum())}/{n} "
                  f"transited {int(scene._transited.sum())}/{n} | placed {int(scene.placed().sum())}/{n} "
                  f"| score {float(scene.score().min()):.0f}", flush=True)

    pos, _ = scene.wheel_pose()
    rest_h = (pos[:, 2] - basket_floor) * 1e3
    ok = int(scene.success().sum())
    if ok != n:
        fails.append(f"success() {ok}/{n} after a staged, teleported run — the scene cannot be solved")
    if int(scene.dropped().sum()):
        fails.append("dropped() true with the wheel resting in the basket")

    # get_state/set_state round-trip, latches included: read, perturb, restore, re-read.
    st = env.get_states()
    teleport((wheel0[0], wheel0[1]), c.surface_z + LIFT_H)
    env.step(no_action, render=False)
    env.set_states(st)
    env.step(no_action, render=False)
    if int(scene.success().sum()) != n:
        fails.append("set_states() did not restore a solved state (latches or wheel pose lost)")

    print(f"WHEEL-CARRY | success {ok}/{n} | score {float(scene.score().min()):.0f} | dropped "
          f"{int(scene.dropped().sum())}/{n} | wheel above basket floor mm: min={rest_h.min():+.1f} "
          f"mean={rest_h.mean():+.1f} max={rest_h.max():+.1f}", flush=True)
    print("WHEEL-CARRY SMOKE:", "FAIL — " + "; ".join(fails) if fails else "PASS", flush=True)
    close_and_exit(env, app)


if __name__ == "__main__":
    main()
