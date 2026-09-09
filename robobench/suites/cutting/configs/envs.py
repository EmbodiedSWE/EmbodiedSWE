"""Canonical runnable env configs for the cutting suite — registered in `ENVS` by name.

`register_env` derives the name by the convention ``suite.scene[_variant][.robot[.control_mode]]``.
Two scenes, each with a food variant that is CFG ONLY (`scene_cfg=...Cfg(food=...)`, the scene
class untouched) and named through `EnvCfg.variant`: `slice` (carrot) / `slice_banana` and `dice`
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

import math

from robobench.core import EnvCfg, register_env
from robobench.robots import FrankaRobotCfg
from robobench.suites.cutting.scenes import DiceFoodSceneCfg, SliceFoodSceneCfg

SUITE = "cutting"
# (scene, variant, scene cfg): the default food carries no variant segment
FOODS = (
    ("slice", "", lambda: SliceFoodSceneCfg()),  # carrot
    ("slice", "banana", lambda: SliceFoodSceneCfg(food="banana")),
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
    # scene physics only -> "cutting.slice", "cutting.slice_banana", "cutting.dice", "cutting.dice_potato"
    register_env(SUITE, lambda scene=_scene, variant=_variant, sc=_scene_cfg: EnvCfg(
        scene=scene, variant=variant, scene_cfg=sc(), robot="null", env_spacing=3))
    # the Franka binding in every control mode -> "cutting.<scene>[_<food>].franka.<mode>"
    for _mode in FRANKA_MODES:
        register_env(SUITE, lambda scene=_scene, variant=_variant, sc=_scene_cfg, mode=_mode: EnvCfg(
            scene=scene, variant=variant, scene_cfg=sc(), robot="franka", control_mode=mode,
            env_spacing=3, robot_cfg=_franka()))
