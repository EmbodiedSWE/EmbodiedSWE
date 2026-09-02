"""Canonical runnable env configs for the cutting suite — registered in `ENVS` by name.

`register_env` derives the name by the convention ``suite.scene[.robot[.control_mode]]``.
Baseline bindings: `cutting.slice` / `cutting.dice` (scene physics only, NullRobot) and the
Franka binding the slice solution was developed on. Variants (another food) are cheap cfg
copies — nothing here is locked.
"""

from __future__ import annotations

import math

from robobench.core import EnvCfg, register_env
from robobench.robots import FrankaRobotCfg

SUITE = "cutting"


def _franka() -> FrankaRobotCfg:
    # Franka Panda on the arm stand behind the island, facing +y toward the board. Base
    # 0.56 m behind the cut: held edge-level, the pinched handle sits ~15 cm toward the base
    # and only ~7 cm above the edge, and closer bases put the pressing hand in the elbow-fold
    # dead zone. Friction pinch only — a strong squeeze on the knife's high-friction handle
    # patch.
    return FrankaRobotCfg(
        base_pos=(0.0, -0.58, 0.858),
        base_rot=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
        gripper_stiffness=10000.0, gripper_damping=200.0, gripper_effort_limit=160.0,
    )


# Pre-split carrot on the chopping board, knife on its rest; scene physics only.
# -> "cutting.slice"
register_env(SUITE, lambda: EnvCfg(scene="slice", robot="null", env_spacing=3))

# Pre-split tomato (3 x 3 grid: x-planes, then the blade yaws 90 deg for the y-planes).
# -> "cutting.dice"
register_env(SUITE, lambda: EnvCfg(scene="dice", robot="null", env_spacing=3))

# Joint mode (the driver runs its own IK). -> "cutting.slice.franka.joint" / "cutting.dice.franka.joint"
register_env(SUITE, lambda: EnvCfg(scene="slice", robot="franka", control_mode="joint",
                                   env_spacing=3, robot_cfg=_franka()))
register_env(SUITE, lambda: EnvCfg(scene="dice", robot="franka", control_mode="joint",
                                   env_spacing=3, robot_cfg=_franka()))
