"""Canonical runnable env configs for the pouring suite — registered in `ENVS` by name.

One env, one scene, one cfg: `pouring.latte.bimanual_franka.joint` — the full-physics latte
benchmark on the single registered `latte` scene; every run shares that one scene cfg. Runs
ONLY under the Newton venv (`env_newton`) — building needs isaaclab develop's Newton backend;
registration itself stays app-free in any venv. Single-env only: the MPM solver uses one fixed
grid spanning the scene, so keep `num_envs=1` when building.
"""

from __future__ import annotations

from robobench.core import EnvCfg, register_env
from robobench.robots import FrankaRobotCfg
from robobench.robots.multi import BimanualFrankaCfg

SUITE = "pouring"

# Two dynamic Frankas behind the table (left carries the mug, right pours the milk pitcher) on
# the COUPLED MJWarp+MPM substrate, scene "latte":
#   - DYNAMIC arms: actuator PD under real gravity, MuJoCo rigid contacts (arm-table, arm-arm,
#     arm-floor).
#   - DYNAMIC vessels: free rigid bodies (authored mass, concave rigid proxies) with live
#     vessel-table / vessel-vessel contacts.
#   - AUTO-GRASP contract: a gripper that closes within reach of a handle bar welds the vessel
#     on; opening releases it (scene post_step — no scripted attach calls anywhere).
#   - 1.5-WAY LIQUID FEEDBACK: the MPM collider impulses are applied back onto the rigid
#     bodies, so vessels weigh what they hold and lighten as they pour.
# Franka knobs are the folding suite's proven dynamic set: gravcomp=1.0 (a kp=400 servo sags
# ~0.1 rad at reach without it — PhysX disable_gravity is IGNORED by the Newton pipeline), arm
# effort 300 N*m so tracking never crawls at the real Panda limits, gripper effort 500 N for
# firm pinches. Quats are develop-xyzw: identity = (0,0,0,1); bases face +y via Rz(+90).
# -> "pouring.latte.bimanual_franka.joint"
_RZ90_XYZW = (0.0, 0.0, 0.70710678, 0.70710678)
_HOME = (0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307, 0.0)


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
        scene="latte",
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

# There is deliberately no second env / scene tier: every run builds this same `latte`
# scene, so exactly one scene registration and one scene cfg are ever in play.
