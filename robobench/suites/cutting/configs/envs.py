"""Canonical runnable env configs for the cutting suite — registered in `ENVS` by name.

`register_env` derives the name by the convention ``suite.scene[_variant][.robot[.control_mode]]``.
Two scenes, each with food variants that are CFG ONLY (`scene_cfg=...Cfg(food=...)`, the scene
class untouched) and named through `EnvCfg.variant`: `slice` (carrot) / `slice_<food>` for every
name in `SLICE_FOODS` (e.g. `slice_banana`, `slice_cucumber_3`, `slice_baguette_1`) and `dice`
(tomato) / `dice_potato`. Baseline bindings per (scene, food): `cutting.<scene>` (scene physics
only, NullRobot) and the Franka on the arm stand in all five of its control modes, like the bulb
bindings:

  - "cutting.<scene>.franka.osc"       — operational-space control (default)
  - "cutting.<scene>.franka.impedance" — Jacobian-transpose task-space impedance
  - "cutting.<scene>.franka.diff_ik"   — differential IK (joint position targets)
  - "cutting.<scene>.franka.pink_ik"   — Pink QP IK (joint position targets)
  - "cutting.<scene>.franka.joint"     — direct joint position targets (the first slice
                                          solution ran its own IK on top of this)

Nothing here is locked — a variant is a cheap cfg copy.
"""

from __future__ import annotations

from copy import deepcopy

import math

from robobench.core import EnvCfg, register_env
from robobench.robots import FrankaRobotCfg
from robobench.suites.cutting.scenes import DiceFoodSceneCfg, SliceFoodSceneCfg


# Default room settings belong to these task configurations.

_CUTTING_ROOM = {'backend': 'physx',
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

def _cutting_room(cfg, scene, robot):
    # Slice and dice share the island and cutting-board layout for every food variant.
    return deepcopy(_CUTTING_ROOM)

SUITE = "cutting"
# Sliceable foods = `assets/<food>/` dirs (baked welded piece chains). Several scans of one
# species are indexed (`banana_1`..`banana_5`, `cucumber_1`..`cucumber_7`); a distinct variety
# gets a suffix instead (`carrot_purple`, `zucchini_yellow`, `cucumber_pickling`). The two
# originals keep their plain names (`carrot` = default, `banana`).
SLICE_FOODS = (
    "banana", "banana_1", "banana_2", "banana_3", "banana_4", "banana_5",
    "carrot_1", "carrot_2", "carrot_purple",
    "cucumber_1", "cucumber_2", "cucumber_3", "cucumber_4", "cucumber_5", "cucumber_6", "cucumber_7",
    "cucumber_pickling",
    "zucchini_1", "zucchini_2", "zucchini_yellow",
    "eggplant_1", "eggplant_2",
    "parsnip", "sweet_potato", "lotus_root", "corn", "chili", "okra",
    "baguette_1", "baguette_2", "bread_loaf", "garlic_bread",
)
# The scene's FOOD_PRESETS are keyed by the exact food name, so the indexed scans carry their
# species' conditions here (cfg-only): bananas lie arch-horizontal, firm straight foods release
# at the flesh centre.
_BANANA = dict(depth_past_center=0.0, food_rot=(math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0))
_FIRM = dict(depth_past_center=0.0)
FOOD_KW = {
    **{f"banana_{i}": _BANANA for i in range(1, 6)},
    "carrot_1": _FIRM, "carrot_2": _FIRM, "carrot_purple": _FIRM,
}
# (scene, variant, scene cfg): the default food carries no variant segment
FOODS = (
    ("slice", "", lambda: SliceFoodSceneCfg()),  # carrot
    *(("slice", f, lambda food=f: SliceFoodSceneCfg(food=food, **FOOD_KW.get(food, {}))) for f in SLICE_FOODS),
    ("dice", "", lambda: DiceFoodSceneCfg()),  # tomato
    ("dice", "potato", lambda: DiceFoodSceneCfg(food="potato")),
)
FRANKA_MODES = ("osc", "impedance", "diff_ik", "pink_ik", "joint")


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


for _scene, _variant, _scene_cfg in FOODS:
    # scene physics only -> "cutting.slice", "cutting.slice_<food>", "cutting.dice", "cutting.dice_potato"
    register_env(SUITE, lambda scene=_scene, variant=_variant, sc=_scene_cfg: EnvCfg(
        room=_cutting_room,
        scene=scene, variant=variant, scene_cfg=sc(), robot="null", env_spacing=3))
    # the Franka binding in every control mode -> "cutting.<scene>[_<food>].franka.<mode>"
    for _mode in FRANKA_MODES:
        register_env(SUITE, lambda scene=_scene, variant=_variant, sc=_scene_cfg, mode=_mode: EnvCfg(
            room=_cutting_room,
            scene=scene, variant=variant, scene_cfg=sc(), robot="franka", control_mode=mode,
            env_spacing=3, robot_cfg=_franka()))
