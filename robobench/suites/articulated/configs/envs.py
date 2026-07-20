"""Canonical runnable env configs for the articulated suite — registered in `ENVS` by name.

Scene-physics-only first (NullRobot smoke/oracle), embodiments after the physics is
proven. Robot bindings raise the safe onto a bench and face its door (-y) toward the
robot, dial at comfortable fingertip height.
"""

from __future__ import annotations

from robobench.core import EnvCfg, register_env
from robobench.robots import (
    BimanualFrankaCfg,
    FrankaRobotCfg,
    G1RobotCfg,
    GR1T2RobotCfg,
    PiperRobotCfg,
    WxaiRobotCfg,
)
from robobench.suites.articulated.scenes import BalanceScaleSceneCfg, CombinationSafeSceneCfg

SUITE = "articulated"

# Scene physics only (NullRobot). -> "articulated.safe"
register_env(SUITE, lambda: EnvCfg(scene="safe", robot="null", env_spacing=3))


def _robot_scene_cfg() -> CombinationSafeSceneCfg:
    """Bench-height safe for fixed-base humanoids: dial ends up ~1.0 m high, door front
    (-y side) toward the robot."""
    return CombinationSafeSceneCfg(surface_z=0.7, safe_pos=(0.0, 0.22))


# Fixed-base G1. -> "articulated.safe.g1.{joint,pink_ik}"
for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="safe",
                scene_cfg=_robot_scene_cfg(),
                robot="g1",
                control_mode=mode,
                robot_cfg=G1RobotCfg(base_pos=(0.0, -0.55, 0.75)),
                env_spacing=3,
            )
        ),
    )

# GR1-T2 (fine dial work is its defining variant). -> "articulated.safe.gr1t2.{joint,pink_ik}"
for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="safe",
                scene_cfg=_robot_scene_cfg(),
                robot="gr1t2",
                control_mode=mode,
                robot_cfg=GR1T2RobotCfg(base_pos=(0.0, -0.50, 0.95),
                                        base_rot=(0.7071, 0.0, 0.0, 0.7071)),
                env_spacing=3,
            )
        ),
    )


# ---- balance scale (weigh-and-sort; the beam is the only instrument) ----
register_env(SUITE, lambda: EnvCfg(scene="scale", robot="null", env_spacing=3))


def _scale_robot_scene_cfg() -> BalanceScaleSceneCfg:
    """Bench-height scale for fixed-base humanoids (robot at -y; boxes row nearest)."""
    # bench shrunk to the crate-verified 0.9 depth (the default 1.1-deep bench put
    # its front edge at y=-0.55, INSIDE the humanoid bases at -0.50 -> boot chaos);
    # shelf pulled to 0.40 so it stays on the bench top.
    return BalanceScaleSceneCfg(surface_z=0.7, scale_pos=(0.0, 0.14),
                                shelf_pos=(0.0, 0.40), box_row_y=-0.16,
                                bench_size=(1.2, 0.9))


for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="scale",
                scene_cfg=_scale_robot_scene_cfg(),
                robot="gr1t2",
                control_mode=mode,
                robot_cfg=GR1T2RobotCfg(base_pos=(0.0, -0.50, 0.95),
                                        base_rot=(0.7071, 0.0, 0.0, 0.7071)),
                env_spacing=3,
            )
        ),
    )

for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="scale",
                scene_cfg=_scale_robot_scene_cfg(),
                robot="g1",
                control_mode=mode,
                robot_cfg=G1RobotCfg(base_pos=(0.0, -0.50, 0.75)),
                env_spacing=3,
            )
        ),
    )


# ---- Franka bindings ----
# Table-mounted arm at ground level (scene surface_z=0, the null layout), base BEHIND
# the work facing +y (base_rot = +90 deg about z; the Franka's home faces +x).
_FRANKA_ROT = (0.7071068, 0.0, 0.0, 0.7071068)

for _mode in ("osc", "joint", "impedance"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="safe",
                robot="franka",
                control_mode=mode,
                robot_cfg=FrankaRobotCfg(base_pos=(0.0, -0.45, 0.0), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )


def _scale_franka_scene_cfg() -> BalanceScaleSceneCfg:
    """Compact, gripper-sized layout: 6 cm boxes (the parallel gripper opens ~8 cm;
    the humanoid 9 cm boxes cannot be grasped), everything inside the ~0.75 m reach."""
    return BalanceScaleSceneCfg(scale_pos=(0.0, 0.10), shelf_pos=(0.0, 0.34),
                                box_row_y=-0.18, box_size=(0.06, 0.06, 0.05),
                                box_gap=0.11)


for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="scale",
                scene_cfg=_scale_franka_scene_cfg(),
                robot="franka",
                control_mode=mode,
                robot_cfg=FrankaRobotCfg(base_pos=(0.0, -0.45, 0.0), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )


# ---- Small-arm + bimanual scale bindings ----
# PiPER (0.6 m reach) gets an even more compact, embodiment-fitted layout (the layout
# lives in the scene cfg; physics/rules untouched). Bimanual Franka reuses the franka
# layout with side-by-side bases so each arm owns its half of the bench (kills the
# far-slot reach margin problem).
def _scale_piper_scene_cfg() -> BalanceScaleSceneCfg:
    """PiPER-sized layout: 5 cm boxes, row pulled to y=-0.08, shelf at 0.28 with
    narrower slots — every target inside the ~0.6 m reach from base (0, -0.22)."""
    return BalanceScaleSceneCfg(scale_pos=(0.0, 0.12), shelf_pos=(0.0, 0.28),
                                box_row_y=-0.08, box_size=(0.05, 0.05, 0.045),
                                box_gap=0.09, shelf_slot_w=0.105)


for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="scale",
                scene_cfg=_scale_piper_scene_cfg(),
                robot="piper",
                control_mode=mode,
                robot_cfg=PiperRobotCfg(base_pos=(0.0, -0.22, 0.0), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )

for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="scale",
                scene_cfg=_scale_franka_scene_cfg(),
                robot="bimanual_franka",
                control_mode=mode,
                robot_cfg=BimanualFrankaCfg(robots={
                    "left": ("franka", FrankaRobotCfg(base_pos=(-0.26, -0.42, 0.0), base_rot=_FRANKA_ROT)),
                    "right": ("franka", FrankaRobotCfg(base_pos=(0.26, -0.42, 0.0), base_rot=_FRANKA_ROT)),
                }),
                env_spacing=3,
            )
        ),
    )


# ---- Single-arm cobot bindings for the safe ----
# Base poses picked so (a) the door-swing arc
# (radius 0.42 around the hinge at (-0.21,-0.043)) clears the base column by >=5 cm and
# (b) the EE workspace covers the dial orbit, the handle and the prize-tumble press
# points (the EE never needs to reach the prize CENTER — the fingertip does).
for _mode in ("osc", "joint", "impedance"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="safe",
                robot="piper",
                control_mode=mode,
                robot_cfg=PiperRobotCfg(base_pos=(0.10, -0.40, 0.20), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )

for _mode in ("osc", "joint", "impedance"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="safe",
                robot="wxai",
                control_mode=mode,
                robot_cfg=WxaiRobotCfg(base_pos=(0.14, -0.36, 0.20), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )
