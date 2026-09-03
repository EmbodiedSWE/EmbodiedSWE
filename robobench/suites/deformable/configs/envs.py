"""Canonical runnable env configs for the deformable suite — registered in `ENVS` by name.

Each binds one of the suite's scenes to an embodiment + control mode + sim, so a run or smoke
test loads one by name. `register_env` derives the canonical ``suite.scene[.robot[.control_mode]]``
name. These run ONLY under the Newton venv (`env_newton`) — building them needs isaaclab
develop's Newton backend; registration itself stays app-free in any venv. Single-env only for
the MPM scenes (latte, dumpling): the solver uses one fixed grid spanning the scene, so keep
`num_envs=1` when building.

Presets (and the legacy alias each one also answers to — the four scenes were the `folding`,
`pouring`, `dough` and `shoe_tying` suites before they were merged here):

  deformable.tshirt                        folding.tshirt
  deformable.tshirt.franka.joint           folding.tshirt.franka.joint
  deformable.latte.bimanual_franka.joint   pouring.latte.bimanual_franka.joint
  deformable.dumpling                      dough.dumpling
  deformable.knot                          shoe_tying.knot
"""

from __future__ import annotations

from collections.abc import Callable

from robobench.core import EnvCfg, register_env
from robobench.robots import FrankaRobotCfg
from robobench.robots.multi import BimanualFrankaCfg
from robobench.suites.deformable.scenes import DumplingSceneCfg

SUITE = "deformable"

# Suite prefix each scene was registered under before the merge. Kept as aliases so existing
# solutions, harness task lists and run artifacts keyed on the old names keep resolving.
LEGACY_SUITE = {"tshirt": "folding", "latte": "pouring", "dumpling": "dough", "knot": "shoe_tying"}


def _register(factory: Callable[[], EnvCfg]) -> str:
    """Register under `deformable.` and under the scene's legacy suite prefix; returns the canonical name."""
    name = register_env(SUITE, factory)
    register_env(LEGACY_SUITE[name.split(".")[1]], factory)
    return name


# ============================== tshirt (VBD cloth) ===============================================
# T-shirt on the box table, scene physics only (cloth settle / parameter sweeps). No articulation
# in this binding -> pure VBD solver (the coupled MJWarp manager needs >= 1 joint).
# -> "deformable.tshirt"
_register(lambda: EnvCfg(scene="tshirt", robot="null", env_spacing=3, sim_overrides={"coupled": False}))

# Franka based 0.5 m to -x/-y of the table center, arm by direct joint position targets — the
# fold smoke computes those targets with differential IK from its key-pose script.
# -> "deformable.tshirt.franka.joint"
_register(
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


# ============================== latte (implicit-MPM liquids, coupled substrate) ===================
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
# Franka knobs are the tshirt binding's proven dynamic set: gravcomp=1.0 (a kp=400 servo sags
# ~0.1 rad at reach without it — PhysX disable_gravity is IGNORED by the Newton pipeline), arm
# effort 300 N*m so tracking never crawls at the real Panda limits, gripper effort 500 N for
# firm pinches. Quats are develop-xyzw: identity = (0,0,0,1); bases face +y via Rz(+90).
# -> "deformable.latte.bimanual_franka.joint"
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


_register(
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

# There is deliberately no second latte tier: every run builds this same `latte` scene, so
# exactly one scene registration and one scene cfg are ever in play.


# ============================== dumpling (elastoplastic MPM dough) ================================
# Robot-less physics-tuning env on the PURE-MPM substrate (MuJoCo rejects 0-joint models — the
# tshirt null binding's pattern); the pin spawns KINEMATIC so the pure-MPM manager ghosts it into
# an infinite-mass scripted collider. Iterate the material dials here — same dough physics, no
# arm. The suite ships the TASK only: the scene (with its auto-weld grasp contract) plus this
# registration. Robot bindings and solutions live outside the benchmark tree — an experiment
# builds its own `EnvCfg(scene="dumpling", robot=..., ...)` on the coupled substrate.
# -> "deformable.dumpling"
_register(
    lambda: EnvCfg(
        scene="dumpling",
        robot="null",
        env_spacing=3,
        scene_cfg=DumplingSceneCfg(pin_dynamic=False),
        sim_overrides={"coupled": False},
    ),
)


# ============================== knot (Newton rods) ================================================
# Two separate laces rooted at the sneaker's top eyelets, ends free; tie a self-holding half
# knot. Scene physics only: the rod ends are kinematic handles driven straight through the
# Newton manager (no robot in the loop yet), the way the standalone scripts drove them.
# Robot-less -> standalone VBD manager (cable joints are not MuJoCo-convertible).
# -> "deformable.knot"
_register(lambda: EnvCfg(scene="knot", robot="null", env_spacing=2.0))
