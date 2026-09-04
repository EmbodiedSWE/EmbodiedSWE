"""Registered env configs for the splat suite (name = suite.scene[.robot[.mode]]):

  - "splat.table"                       scene only (null robot)
  - "splat.table.franka_robotiq.joint"  Franka + Robotiq 2F-85 (ships a splat model)
  - "splat.table.franka.joint"          vendored Panda + panda hand (no robot splat; sim-only fallback)

Robot base at the origin (the scene's frame), so the robots' default `base_pos=(0,0,0)` is kept.
"""
from __future__ import annotations

from robobench.core import EnvCfg, register_env

SUITE = "splat"

register_env(SUITE, lambda: EnvCfg(scene="table", robot="null", env_spacing=3))

for _robot in ("franka_robotiq", "franka"):
    register_env(
        SUITE,
        lambda robot=_robot: EnvCfg(scene="table", robot=robot, control_mode="joint", env_spacing=3),
    )
