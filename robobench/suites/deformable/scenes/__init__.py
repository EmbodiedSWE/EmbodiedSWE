"""Scenes for the deformable suite — each scene *is* a task (carries its own goal).

Importing this package registers every scene into `robobench.core.SCENES`. Import-light: scene
modules defer isaaclab imports, so registration is safe without a running app (and even without
the Newton venv — building the env is what needs `env_newton`).
"""

from . import shoe_knot  # noqa: F401
from .dumpling import DumplingScene, DumplingSceneCfg
from .latte import LatteScene, LatteSceneCfg
from .tshirt import TshirtFoldingScene, TshirtFoldingSceneCfg

__all__ = [
    "TshirtFoldingScene",
    "TshirtFoldingSceneCfg",
    "LatteScene",
    "LatteSceneCfg",
    "DumplingScene",
    "DumplingSceneCfg",
]
