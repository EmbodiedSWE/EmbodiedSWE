"""Canonical runnable env configs for the dough suite — registered in `ENVS` by name.

One scene, ONE registered binding:

  - "dough.dumpling"  robot-less physics-tuning env on the PURE-MPM substrate (MuJoCo rejects
                      0-joint models — the folding suite's pattern); the pin spawns KINEMATIC
                      so the pure-MPM manager ghosts it into an infinite-mass scripted
                      collider. Iterate the material dials here — same dough physics, no arm.

The suite ships the TASK only: the scene (with its auto-weld grasp contract) plus this
registration. Robot bindings and solutions live outside the benchmark tree — an experiment
builds its own `EnvCfg(scene="dumpling", robot=..., ...)` on the coupled substrate.

Runs ONLY under the Newton venv (`env_newton`) — building needs isaaclab develop's Newton
backend; registration itself stays app-free in any venv. Single-env only: the MPM solver uses
one fixed grid spanning the scene, so keep `num_envs=1` when building.
"""

from __future__ import annotations

from robobench.core import EnvCfg, register_env
from robobench.suites.dough.scenes import DumplingSceneCfg

SUITE = "dough"

# -> "dough.dumpling"
register_env(
    SUITE,
    lambda: EnvCfg(
        scene="dumpling",
        robot="null",
        env_spacing=3,
        scene_cfg=DumplingSceneCfg(pin_dynamic=False),
        sim_overrides={"coupled": False},
    ),
)

# There is deliberately no second scene tier: every run builds this same `dumpling` scene, so
# exactly one scene registration and one scene cfg are ever in play.
