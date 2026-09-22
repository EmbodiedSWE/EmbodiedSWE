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
  deformable.dumpling.franka.joint         dough.dumpling.franka.joint
  deformable.knot                          shoe_tying.knot
  deformable.knot.aloha.joint              shoe_tying.knot.aloha.joint
"""

from __future__ import annotations

from copy import deepcopy

from collections.abc import Callable

from pathlib import Path
from robobench.core.assets import asset_path

from robobench.core import EnvCfg, register_env
from robobench.robots import AlohaCfg, FrankaRobotCfg
from robobench.robots.wx250s import Wx250sRobotCfg
from robobench.robots.multi import BimanualFrankaCfg
from robobench.suites.deformable.scenes import DumplingSceneCfg
from robobench.suites.deformable.scenes.shoe_knot import ShoeKnotSceneCfg


# Default room settings belong to these task configurations.

_TSHIRT_ROOM = {'backend': 'newton',
 'room': {'id': 'simple_room',
          'room_floor_z': -0.7696,
          'anchor': [-0.5, -0.353],
          'yaw': -90.0,
          'floor_world_z': -0.5686},
 'hide': ['/World/ground.*', '/World/envs/env_\\d+/Table'],
 'pedestals': [{'pos': [-0.541, -0.5], 'top': 0.0, 'size': [0.18, 0.18]}],
 'camera': [[1.15, -1.05, 0.85], [0.0, -0.5, 0.24]]}

_LATTE_ROOM = {'backend': 'newton',
 'room': {'room_floor_z': 0.0,
          'hide': ['Kitchen_Disk002',
                   'Kitchen_Orange001',
                   'Kitchen_Orange001_01',
                   'Kitchen_Orange001_02',
                   'Kitchen_Orange001_03',
                   'Kitchen_Orange002',
                   'Kitchen_Orange002_01',
                   'Kitchen_Flowers001',
                   'Plane'],
          'attrs': [['DomeLight_01', 'inputs:texture:file', ''],
                    ['DomeLight_01', 'inputs:intensity', 1200.0]],
          'anchor': [0.135, 0.09],
          'yaw': 0.0,
          'floor_world_z': -0.818,
          'id': 'kitchen'},
 'hide': ['/World/ground.*', '/World/envs/env_\\d+/Table', '/World/envs/env_\\d+/Floor'],
 'pedestals': [{'pos': [-0.25, -0.461], 'size': [0.18, 0.18], 'top': 0.0},
               {'pos': [0.25, -0.461], 'size': [0.18, 0.18], 'top': 0.0}],
 'camera': [[0.75, 1.25, 0.85], [0.08, -0.08, 0.14]]}

_KITCHEN_ROOM = {'backend': 'physx',
 'room': {'room_floor_z': 0.0,
          'hide': ['Kitchen_Disk002',
                   'Kitchen_Orange001',
                   'Kitchen_Orange001_01',
                   'Kitchen_Orange001_02',
                   'Kitchen_Orange001_03',
                   'Kitchen_Orange002',
                   'Kitchen_Orange002_01',
                   'Kitchen_Flowers001',
                   'Plane'],
          'attrs': [['DomeLight_01', 'inputs:texture:file', ''],
                    ['DomeLight_01', 'inputs:intensity', 1200.0]],
          'anchor': [0.215, 0.225],
          'yaw': 0.0,
          'floor_world_z': 0.0,
          'id': 'kitchen'},
 'hide': ['/World/ground.*', '/World/envs/env_\\d+/Island'],
 'camera': [[0.72, -0.62, 1.25], [-0.08, 0.08, 0.95]]}

_KNOT_ROOM = {'backend': 'physx',
 'room': {'room_floor_z': -0.7696,
          'anchor': [-0.37, 0.1],
          'yaw': 0.0,
          'floor_world_z': -0.7686,
          'id': 'simple_room'},
 'hide': ['/World/ground.*', '/World/envs/env_\\d+/Table.*'],
 'camera': [[1.22, -1.2, 1.08], [0.3, -0.02, 0.2]]}

def _tshirt_room(cfg, scene, robot):
    spec = deepcopy(_TSHIRT_ROOM)
    c = scene.cfg
    top = c.table_pos[2] + c.table_size[2] / 2
    spec["room"]["floor_world_z"] = top + .001 - .7696
    spec["room"]["anchor"] = [c.table_pos[1], -(.353+c.table_pos[0])]
    if cfg.robot == "null":
        spec.pop("pedestals", None)
    return spec


def _latte_room(cfg, scene, robot):
    return deepcopy(_LATTE_ROOM)


def _dumpling_room(cfg, scene, robot):
    spec = deepcopy(_KITCHEN_ROOM)
    spec["backend"] = "newton"
    spec["room"]["hide"].append("Kitchen_InsularShelf_01")
    spec["room"]["floor_world_z"] = -1.05
    spec["hide"] = [r"/World/ground.*"]
    spec["camera"] = [[1.1, -1.1, 1.0], [0., 0., scene.cfg.surface_z]]
    return spec


def _knot_room(cfg, scene, robot):
    spec = deepcopy(_KNOT_ROOM)
    spec["backend"] = "newton"
    spec["room"]["hide"] = ["table_low_327"]
    spec["room"]["floor_world_z"] = scene.cfg.ground_z
    spec["hide"] = [r"/World/ground.*"]
    spec["camera"] = [[1.0, -1.2, 1.0], [0., 0., scene.cfg.surface_z]]
    return spec

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
_register(lambda: EnvCfg(room=_tshirt_room, scene="tshirt", robot="null", env_spacing=3, sim_overrides={"coupled": False}))

# Franka based 0.5 m to -x/-y of the table center, arm by direct joint position targets — the
# fold smoke computes those targets with differential IK from its key-pose script.
# -> "deformable.tshirt.franka.joint"
_register(
    lambda: EnvCfg(
        room=_tshirt_room,
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
        room=_latte_room,
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
# arm.
# -> "deformable.dumpling"
_register(
    lambda: EnvCfg(
        room=_dumpling_room,
        scene="dumpling",
        robot="null",
        env_spacing=3,
        scene_cfg=DumplingSceneCfg(pin_dynamic=False),
        sim_overrides={"coupled": False},
    ),
)

# Franka on the COUPLED MJWarp+MPM substrate: dynamic arm and dynamic pin, arm by direct joint
# position targets. Base 0.15 m outside the table's -x edge, facing +x (develop-xyzw quats:
# identity = (0,0,0,1)). Franka knobs are the tshirt/latte bindings' proven dynamic set:
# gravcomp=1.0 (a kp=400 servo sags ~0.1 rad at reach without it — PhysX disable_gravity is
# IGNORED by the Newton pipeline), arm effort 300 N*m so tracking never crawls at the real Panda
# limits, gripper effort 500 N for firm pinches. No `sim_overrides`: the scene's `sim_cfg()`
# already selects the coupled substrate and carries its grasp-contract weld row.
# -> "deformable.dumpling.franka.joint"
_register(
    lambda: EnvCfg(
        room=_dumpling_room,
        scene="dumpling",
        robot="franka",
        control_mode="joint",
        env_spacing=3,
        robot_cfg=FrankaRobotCfg(
            base_pos=(-0.50, 0.0, 0.0),
            base_rot=(0.0, 0.0, 0.0, 1.0),
            default_dof_pos=_HOME,
            gravity_compensation=1.0,
            arm_effort_limit=300.0,
            gripper_effort_limit=500.0,
        ),
    ),
)


# ============================== knot (Newton rods) ================================================
# Two separate laces rooted at the sneaker's top eyelets, ends free; tie a self-holding half
# knot. Scene physics only: the rod ends are kinematic handles driven straight through the
# Newton manager (no robot in the loop yet), the way the standalone scripts drove them.
# Robot-less -> standalone VBD manager (cable joints are not MuJoCo-convertible).
# -> "deformable.knot"
_register(lambda: EnvCfg(room=_knot_room, scene="knot", robot="null", env_spacing=2.0))

# The nut-thread task's lab-table workbench, reused as the aloha binding's work surface.
# MONOREPO-ONLY cross-suite asset reference (73 MB — deliberately not duplicated into this
# suite's assets/; copy it under deformable/assets/props/ if this binding ever ships through
# eval/envbuild/extract.py, which only carries the target suite's tree).
_LAB_TABLE_USD = str(
    asset_path(Path(__file__).resolve().parents[3] / "suites" / "assembly" / "assets") / "props" / "lab_table" / "table_instanceable.usd"
)

# Bimanual WidowX 250 6DOF (the classic Interbotix ALOHA arms; official menagerie model
# converted to USD — see `robobench/robots/wx250s.py`) flanking the shoe on the nut-thread
# task's lab-table workbench, arms by direct joint position targets. Runs on the suite's
# PROXY-COUPLED rod substrate (`newton/lace_coupled_manager.py`): SolverMuJoCo owns the arms,
# SolverVBD the rods, and the finger bodies are proxied into the rod solve (finger-lace
# contact is real contact). The scene keeps only the eyelet roots anchored
# (`kinematic_ends=False`) — the free ends are dynamic rod.
# Work surface: the lab table spawned so its 1.28 x 0.91 m top is centered at env (0,0) with
# the top at surface_z (USD origin sits at the tabletop, 0.394 m off-center in x, top plane
# at -0.003); the physics twin box matches the top, and the ground sinks to the table feet
# (surface_z - 1.04). Placement: bases ON the table top (z = surface_z, +0.5 mm so the base
# plate doesn't start intersecting the tabletop collider) at x = ±0.42; the left arm at -x
# is nearest lace 1's end (left flank), the right arm at +x lace 2's (right).
# isaaclab develop uses **xyzw** quats: identity = (0,0,0,1); the right arm is yawed 180 deg.
# Both fingers are actuated (the menagerie mimic equality does not survive conversion); the
# right finger's coordinate mirrors the left — see the robot module.
# -> "deformable.knot.aloha.joint"
_register(
    lambda: EnvCfg(
        room=_knot_room,
        scene="knot",
        # free ends dynamic; the tails rest splayed outward on open table
        scene_cfg=ShoeKnotSceneCfg(
            kinematic_ends=False,
            end_splay=0.065,
            table_usd=_LAB_TABLE_USD,
            # raw table geometry: top x in [-0.455, 0.455], y in [-0.484, 0.796]; the VISUAL
            # top plane sits at raw z=0.0 (the collision cube's top is 3 mm lower — aligning
            # to that sinks the robot bases and the shoe into the visual top). The asset's
            # authored root transform is replaced by the spawner; yaw 90 deg puts the 1.28 m
            # axis along x (under the arms). x +0.016 slides the table 14 cm off the centered
            # +0.156 so the fixture plate + cutout (world x ~-0.46 when centered) sit clear
            # of the left arm's base.
            table_usd_offset=(0.016, 0.0, 0.0),
            table_usd_rot=(0.0, 0.0, 0.70711, 0.70711),  # xyzw: yaw +90 deg
            table_size=(1.28, 0.91, 0.2),
            ground_z=0.2 - 1.04,
        ),
        robot="aloha",
        control_mode="joint",
        robot_cfg=AlohaCfg(robots={
            # ready pose: shoulder 0.4 + elbow 0.2 lean the arm over the table, wrist_angle
            # 0.97 (= pi/2 - 0.6) points the gripper straight down
            "left": ("wx250s", Wx250sRobotCfg(
                base_pos=(-0.42, 0.0, 0.2005),
                base_rot=(0.0, 0.0, 0.0, 1.0),  # xyzw identity: faces +x
                default_dof_pos=(0.0, 0.4, 0.2, 0.0, 0.97, 0.0),
                gripper_stiffness=6000.0,
                gripper_damping=200.0,
            )),
            "right": ("wx250s", Wx250sRobotCfg(
                base_pos=(0.42, 0.0, 0.2005),
                base_rot=(0.0, 0.0, 1.0, 0.0),  # xyzw yaw 180 deg: faces -x
                default_dof_pos=(0.0, 0.4, 0.2, 0.0, 0.97, 0.0),
                gripper_stiffness=6000.0,
                gripper_damping=200.0,
            )),
        }),
        env_spacing=2.0,
        sim_overrides={
            "coupled": True,
            # WX250s gripper proxies: the finger links (their pad + fingertip-sphere shapes
            # are the model's only gripper colliders)
            "coupling": {"proxy_body_keywords": ("finger_link",)},
            # Rod-entry contact recipe for the coupled substrate: the upstream franka+cable
            # example's AVBD penalty ramp plus contact history (the manager's proxy pipeline
            # then uses "latest" contact matching — see lace_coupled_manager.py). The
            # standalone knot binding keeps the scene's history+sticky recipe.
            "vbd": {
                "iterations": 20,
                "rigid_contact_history": True,
                "rigid_avbd_beta": 1.0e2,
                "rigid_contact_k_start": 1.0e3,
                "rigid_body_contact_buffer_size": 512,
            },
        },
    ),
)
