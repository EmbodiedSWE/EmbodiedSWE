"""Canonical runnable env configs for the cutting suite — registered in `ENVS` by name.

`register_env` derives the name by the convention ``suite.scene[.robot[.control_mode]]``.
Baseline bindings: `cutting.slice` (scene physics only, NullRobot) and the Franka binding the
slice solution was developed on. Variants (another food) are cheap cfg copies — nothing here
is locked.
"""

from __future__ import annotations

import math

from robobench.core import EnvCfg, register_env
from robobench.robots import FrankaRobotCfg

SUITE = "cutting"

# Pre-split carrot on the chopping board, knife on its rest; scene physics only.
# -> "cutting.slice"
register_env(SUITE, lambda: EnvCfg(scene="slice", robot="null", env_spacing=3))

# Franka Panda on the arm stand behind the island, facing +y toward the board, joint mode
# (the driver runs its own IK). Base 0.56 m behind the cut: held edge-level, the pinched
# handle sits ~15 cm toward the base and only ~7 cm above the edge, and closer bases put
# the pressing hand in the elbow-fold dead zone. Friction pinch only — a strong squeeze on
# the knife's high-friction handle patch. -> "cutting.slice.franka.joint"
register_env(SUITE, lambda: EnvCfg(
    scene="slice", robot="franka", control_mode="joint", env_spacing=3,
    robot_cfg=FrankaRobotCfg(
        base_pos=(0.0, -0.58, 0.858),
        base_rot=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
        gripper_stiffness=10000.0, gripper_damping=200.0, gripper_effort_limit=160.0,
    ),
))
