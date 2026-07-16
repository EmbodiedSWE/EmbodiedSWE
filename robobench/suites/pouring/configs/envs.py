"""Canonical runnable env configs for the pouring suite — registered in `ENVS` by name.

These run ONLY under the Newton venv (`env_newton`) — building them needs isaaclab develop's
Newton backend; registration itself stays app-free in any venv. Single-env for now: the MPM
solver uses one fixed grid spanning the scene, so keep `num_envs=1` when building.
"""

from __future__ import annotations

from robobench.core import EnvCfg, register_env

SUITE = "pouring"

# Coffee cup + kinematic milk cup + two MPM liquids, scene physics only: the pour smoke (or an
# agent) drives the milk cup by writing its root pose. No articulation in this binding — the MPM
# manager treats rigids as colliders and is not a rigid-dynamics solver (a robot embodiment needs
# the future coupled MJWarp+MPM manager).
# -> "pouring.latte"
register_env(SUITE, lambda: EnvCfg(scene="latte", robot="null", env_spacing=3))
