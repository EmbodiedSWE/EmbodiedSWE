"""Canonical runnable env configs for the folding suite — registered in `ENVS` by name.

Each binds the suite's scene to an embodiment + control mode + sim, so a run or smoke test loads
one by name. `register_env` derives the canonical ``suite.scene[.robot[.control_mode]]`` name.
These run ONLY under the Newton venv (`env_newton`) — building them needs isaaclab develop's
Newton backend; registration itself stays app-free in any venv.
"""

from __future__ import annotations

from robobench.core import EnvCfg, register_env
from robobench.robots import FrankaRobotCfg

SUITE = "folding"

# T-shirt on the box table, scene physics only (cloth settle / parameter sweeps). No articulation
# in this binding -> pure VBD solver (the coupled MJWarp manager needs >= 1 joint).
# -> "folding.tshirt"
register_env(SUITE, lambda: EnvCfg(scene="tshirt", robot="null", env_spacing=3, sim_overrides={"coupled": False}))

# Franka based 0.5 m to -x/-y of the table center, arm by direct joint position targets — the
# fold smoke computes those targets with differential IK from its key-pose script.
# -> "folding.tshirt.franka.joint"
register_env(
    SUITE,
    lambda: EnvCfg(
        scene="tshirt",
        robot="franka",
        control_mode="joint",
        robot_cfg=FrankaRobotCfg(
            base_pos=(-0.5, -0.5, 0.0),
            # isaaclab develop uses **xyzw** quats: identity = (0,0,0,1). The robobench default
            # (1,0,0,0) is the 2.x wxyz identity — under develop it reads as a 180° X-flip and
            # mounts the arm upside-down beneath the world.
            base_rot=(0.0, 0.0, 0.0, 1.0),
            default_dof_pos=(0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307, 0.0),  # elbow-bent ready pose over the table
            gripper_effort_limit=500.0,  # strong pinch for cloth
            arm_effort_limit=300.0,  # scripted joint tracking must not crawl at the real Panda limits
            gravity_compensation=1.0,  # weightless arm — the position servo must not sag at extended reach
        ),
        env_spacing=3,
    ),
)
