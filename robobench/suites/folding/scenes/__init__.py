"""Scenes for the folding suite — each scene *is* a task (carries its own goal).

Importing this package registers every scene into `robobench.core.SCENES`. Import-light: scene
modules defer isaaclab imports, so registration is safe without a running app (and even without
the Newton venv — building the env is what needs `env_newton`).
"""

from .tshirt import TshirtFoldingScene, TshirtFoldingSceneCfg

__all__ = [
    "TshirtFoldingScene",
    "TshirtFoldingSceneCfg",
]
