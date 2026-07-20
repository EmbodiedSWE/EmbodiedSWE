"""Canonical runnable env configs for the pouring suite — registered in `ENVS` by name.

These run ONLY under the Newton venv (`env_newton`) — building them needs isaaclab develop's
Newton backend; registration itself stays app-free in any venv. Single-env for now: the MPM
solver uses one fixed grid spanning the scene, so keep `num_envs=1` when building.
"""

from __future__ import annotations

from robobench.core import EnvCfg, register_env
from robobench.robots import FrankaRobotCfg
from robobench.robots.multi import BimanualFrankaCfg

SUITE = "pouring"

# Coffee cup + kinematic milk cup + two MPM liquids, scene physics only: the pour smoke (or an
# agent) drives the milk cup by writing its root pose. No articulation in this binding — the MPM
# manager treats rigids as colliders and is not a rigid-dynamics solver (a robot embodiment needs
# the future coupled MJWarp+MPM manager).
# -> "pouring.latte"
register_env(SUITE, lambda: EnvCfg(scene="latte", robot="null", env_spacing=3))

# Two Frankas behind the table (left holds the mug handle, right pours the milk cup). KINEMATIC
# embodiment: under the MPM manager the arms have no dynamics — smokes drive them by writing
# joint state (the manager runs FK, so every link is a live MPM collider); actuators/controllers
# are inert. Quats are develop-xyzw: identity = (0,0,0,1); bases face +y via Rz(+90).
# -> "pouring.latte.bimanual_franka.joint"
_RZ90_XYZW = (0.0, 0.0, 0.70710678, 0.70710678)
_HOME = (0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307, 0.0)
register_env(
    SUITE,
    lambda: EnvCfg(
        scene="latte",
        robot="bimanual_franka",
        control_mode="joint",
        env_spacing=3,
        robot_cfg=BimanualFrankaCfg(
            robots={
                "left": ("franka", FrankaRobotCfg(base_pos=(-0.25, -0.42, 0.0), base_rot=_RZ90_XYZW, default_dof_pos=_HOME)),
                "right": ("franka", FrankaRobotCfg(base_pos=(0.25, -0.42, 0.0), base_rot=_RZ90_XYZW, default_dof_pos=_HOME)),
            }
        ),
    ),
)

# The same bimanual latte on the COUPLED MJWarp+MPM substrate (scene "latte_dyn") —
# the arms are DYNAMIC: actuator PD tracks joint-position actions under real gravity, with
# MuJoCo-internal rigid contacts (arm-table, arm-arm, arm-floor). The dynamic-Franka knobs are
# the folding suite's proven set: gravcomp=1.0 (a kp=400 servo sags ~0.1 rad at reach without
# it — PhysX disable_gravity is IGNORED by the Newton pipeline), arm effort 300 N*m so scripted
# tracking never crawls at the real Panda limits, gripper effort 500 N for firm pinches.
# -> "pouring.latte_dyn.bimanual_franka.joint"
def _dyn_franka(base_pos: tuple[float, float, float]) -> FrankaRobotCfg:
    return FrankaRobotCfg(
        base_pos=base_pos,
        base_rot=_RZ90_XYZW,
        default_dof_pos=_HOME,
        gravity_compensation=1.0,
        arm_effort_limit=300.0,
        gripper_effort_limit=500.0,
    )


register_env(
    SUITE,
    lambda: EnvCfg(
        scene="latte_dyn",
        robot="bimanual_franka",
        control_mode="joint",
        env_spacing=3,
        robot_cfg=BimanualFrankaCfg(
            robots={
                "left": ("franka", _dyn_franka((-0.25, -0.42, 0.0))),
                "right": ("franka", _dyn_franka((0.25, -0.42, 0.0))),
            }
        ),
    ),
)

# Same bimanual rig on scene "latte_weld" — DYNAMIC vessels (free joints, authored
# mass, concave rigid proxies) carried via weld-at-grasp (scene.weld_vessel). The arms feel the
# real vessel mass; vessel-table and vessel-vessel contacts are live MuJoCo contacts.
# -> "pouring.latte_weld.bimanual_franka.joint"
register_env(
    SUITE,
    lambda: EnvCfg(
        scene="latte_weld",
        robot="bimanual_franka",
        control_mode="joint",
        env_spacing=3,
        robot_cfg=BimanualFrankaCfg(
            robots={
                "left": ("franka", _dyn_franka((-0.25, -0.42, 0.0))),
                "right": ("franka", _dyn_franka((0.25, -0.42, 0.0))),
            }
        ),
    ),
)


# AGENT BENCHMARK: scene "latte_auto" — the latte_weld substrate with AUTOMATIC weld grasping:
# a gripper that closes within auto_weld_dist of a handle bar grabs the vessel; opening releases
# it. No scripted weld calls anywhere — the mechanic lives in the scene's post_step, so any
# agent policy (or script) gets real carried-mass dynamics through a real-gripper contract.
# -> "pouring.latte_auto.bimanual_franka.joint"
register_env(
    SUITE,
    lambda: EnvCfg(
        scene="latte_auto",
        robot="bimanual_franka",
        control_mode="joint",
        env_spacing=3,
        robot_cfg=BimanualFrankaCfg(
            robots={
                "left": ("franka", _dyn_franka((-0.25, -0.42, 0.0))),
                "right": ("franka", _dyn_franka((0.25, -0.42, 0.0))),
            }
        ),
    ),
)


# "latte_feed" — latte_auto + 1.5-way liquid feedback (vessels weigh what they
# hold). -> "pouring.latte_feed.bimanual_franka.joint"
register_env(
    SUITE,
    lambda: EnvCfg(
        scene="latte_feed",
        robot="bimanual_franka",
        control_mode="joint",
        env_spacing=3,
        robot_cfg=BimanualFrankaCfg(
            robots={
                "left": ("franka", _dyn_franka((-0.25, -0.42, 0.0))),
                "right": ("franka", _dyn_franka((0.25, -0.42, 0.0))),
            }
        ),
    ),
)


# FORCE CLOSURE on scene "latte_grip" (same substrate as latte_weld; new name = the variant
# slot): the vessels hang from real friction pinches on the handle bars, so the gripper PD is
# pinch-grade (kp 20000 -> ~130 N at a blocked full-close command). See README for status.
def _grip_franka(base_pos: tuple[float, float, float]) -> FrankaRobotCfg:
    cfg = _dyn_franka(base_pos)
    cfg.gripper_stiffness = 20000.0
    cfg.gripper_damping = 200.0
    return cfg


# -> "pouring.latte_grip.bimanual_franka.joint"
register_env(
    SUITE,
    lambda: EnvCfg(
        scene="latte_grip",
        robot="bimanual_franka",
        control_mode="joint",
        env_spacing=3,
        robot_cfg=BimanualFrankaCfg(
            robots={
                "left": ("franka", _grip_franka((-0.25, -0.42, 0.0))),
                "right": ("franka", _grip_franka((0.25, -0.42, 0.0))),
            }
        ),
    ),
)
