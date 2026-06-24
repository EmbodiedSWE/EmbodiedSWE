"""Core machinery shared across all robobench suites.

Base abstractions (import-light — safe without AppLauncher):
  BaseScene · BaseRobot · BaseController · BaseVerifier   (contracts concrete suites/robots implement)
  BaseRobotCfg                     (thin robot-cfg base carrying `control_mode`)
  BaseEnv                          (the open env shell; its build needs AppLauncher)
  config.py — the whole config layer (one module):
    BaseCfg · tunable · info       (scene/robot cfg base: tunable difficulty dials vs fixed info)
    SimCfg                         (sim substrate dt+PhysX; declared by the scene, patchable per env)
    EnvCfg · register_env          (binds scene+robot+mode+sim; loaded by name from ENVS; `.build()`)
  describe_stage · usd_text        (generic USD introspection — works on any stage)
  SCENES · ROBOTS · CONTROLLERS · ENVS  (string→factory registries; no TASKS — a scene *is* a task)

Still TODO (filled later): a structured verifier result type (verify() returns Any for now),
a Sim wrapper, sensors, gym_wrapper, a verifier registry.
"""

from .config import BaseCfg, EnvCfg, SimCfg, info, register_env, tunable
from .controller import BaseController, BaseControllerCfg
from .env import BaseEnv
from .introspection import describe_stage, usd_text
from .registries import CONTROLLERS, ENVS, ROBOTS, SCENES
from .robot import BaseRobot, BaseRobotCfg
from .scene import BaseScene
from .verifier import BaseVerifier

__all__ = [
    "BaseScene",
    "BaseRobot",
    "BaseRobotCfg",
    "BaseController",
    "BaseControllerCfg",
    "BaseVerifier",
    "BaseEnv",
    "EnvCfg",
    "register_env",
    "SimCfg",
    "BaseCfg",
    "tunable",
    "info",
    "describe_stage",
    "usd_text",
    "SCENES",
    "ROBOTS",
    "CONTROLLERS",
    "ENVS",
]
