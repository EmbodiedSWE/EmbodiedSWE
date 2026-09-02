"""Canonical runnable env configs for the locomanip suite — registered in `ENVS` by name.

Each binds the suite's scene to an embodiment + control mode + sim, so a run or smoke test loads one
by name, instead of wiring scene/robot/mode by hand. `register_env` derives the canonical name by the
convention ``suite.scene[.robot[.control_mode]]`` (segments dropped from the right when
default/absent), so names stay consistent as scenes/robots multiply. Factories (a fresh `EnvCfg` per
call) keep one build from mutating another's cfg.

These are the *baseline* bindings; curriculum/debug variants are cheap `dataclasses.replace(cfg, ...)`
derivations the harness or agent can make — nothing here is locked.
"""

from __future__ import annotations

from robobench.core import EnvCfg, register_env
from robobench.robots import G1RobotCfg
from robobench.suites.locomanip.scenes import (
    BoxToBinSceneCfg,
    FruitDeliverySceneCfg,
    WheelCarrySceneCfg,
)

SUITE = "locomanip"


# ---- Steering-wheel carry (a wheel off one table into a basket on another, 3 m away) -------------
# Scene physics only (NullRobot: both tables, the wheel, the basket). -> "locomanip.wheel_carry"
#
# `env_spacing` = 8.0 is not the usual "just clear the furniture": the two 2.47 m tables already span
# about 5.5 m in x, and a walking robot leaves the footprint its scene furniture defines. Neighbouring
# envs must not be somewhere this robot can walk into.
register_env(SUITE, lambda: EnvCfg(scene="wheel_carry", robot="null", env_spacing=8.0))

# MOBILE G1, one binding per loco control mode. `fixed_base=False` is what makes this a different
# robot from every other G1 preset in the tree — the pelvis is free and the legs are driven by the
# frozen locomotion policy (see robots/g1.py and controllers/loco_policy.py). Passing a fixed-base
# G1 here raises at build rather than quietly standing still, so the pairing cannot be got wrong.
#
# No placement override: `G1RobotCfg`'s own default pelvis pose (0, 0, 0.75) facing +y puts the robot
# squarely at the PICK station's `stand_xy()` — the scene inherits `assembly.wheel_pick_place`'s
# geometry precisely so that this keeps working. Where the robot goes after that is its problem.
#   - "locomanip.wheel_carry.g1.loco_pink_ik" — arm+waist by whole-body Pink IK + a base command
#     (action = 2 wrist poses 14 + 14 hand joints + 4 base = 32)
#   - "locomanip.wheel_carry.g1.loco_joint"   — the same hardware by direct joint targets (31 + 4 = 35)
for _mode in ("loco_pink_ik", "loco_joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="wheel_carry",
                robot="g1",
                control_mode=mode,
                robot_cfg=G1RobotCfg(fixed_base=False),
                scene_cfg=WheelCarrySceneCfg(),
                env_spacing=8.0,
            )
        ),
    )


# ---- Box to bin (a cardboard box off a shelf board into a sorting bin, around a turn) ------------
# Scene physics only (NullRobot: shelf, table, bin, box). -> "locomanip.box_to_bin"
#
# `env_spacing` = 8.0 for the same reason as wheel_carry: a walking robot leaves the footprint its
# furniture defines, and neighbouring envs must not be somewhere this robot can walk into.
register_env(SUITE, lambda: EnvCfg(scene="box_to_bin", robot="null", env_spacing=8.0))

# MOBILE G1 (see the wheel_carry block above for why `fixed_base=False` is the whole point). No
# placement override here either: the scene is laid out around `G1RobotCfg`'s default pelvis pose
# (0, 0, 0.75) facing +y — the shelf face 0.46 m ahead is exactly the Arena task's stand-off.
#   - "locomanip.box_to_bin.g1.loco_pink_ik" — arm+waist by whole-body Pink IK + a base command
#     (action = 2 wrist poses 14 + 14 hand joints + 4 base = 32)
#   - "locomanip.box_to_bin.g1.loco_joint"   — the same hardware by direct joint targets (31 + 4 = 35)
for _mode in ("loco_pink_ik", "loco_joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="box_to_bin",
                robot="g1",
                control_mode=mode,
                robot_cfg=G1RobotCfg(fixed_base=False),
                scene_cfg=BoxToBinSceneCfg(),
                env_spacing=8.0,
            )
        ),
    )


# ---- Fruit delivery (fruits_on_plate stretched across a table no arm can span) ------------------
# Scene physics only (NullRobot oracle). -> "locomanip.fruit_delivery"
register_env(SUITE, lambda: EnvCfg(scene="fruit_delivery", robot="null", env_spacing=6.0))

# MOBILE G1. Same mobile pairing as wheel_carry (fixed_base=False + the frozen loco policy).
# The pelvis starts at (0, -0.85, 0.75) facing +y: clear of the kitchen table's near edge
# (top half-depth 0.50 + torso), at the pick side, with the fruit line 0.28 m ahead of the
# shoulder once it steps up to the working stance — the fixed-base fruits tier's calibrated
# grasp distance. Arm/hand PD keeps that tier's measured retune (see packing/configs/envs.py):
# the grasp findings carry over only if the actuator behaviour does.
#   - "locomanip.fruit_delivery.g1.loco_joint"   — 31 joint targets + 4 base = 35
#   - "locomanip.fruit_delivery.g1.loco_pink_ik" — 2 wrist poses + 14 hand + 4 base = 32
for _mode in ("loco_joint", "loco_pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="fruit_delivery",
                robot="g1",
                control_mode=mode,
                robot_cfg=G1RobotCfg(
                    fixed_base=False,
                    base_pos=(0.0, -0.85, 0.75),
                    arm_stiffness=1500.0,
                    arm_damping=90.0,
                    hand_stiffness=400.0,
                    hand_damping=16.0,
                ),
                scene_cfg=FruitDeliverySceneCfg(),
                env_spacing=6.0,
            )
        ),
    )
