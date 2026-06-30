"""Scenes for the assembly suite — each scene *is* a task (carries its own goal + optional verifier).

Importing this package registers every scene into `robobench.core.SCENES`. Import-light: scene
modules defer isaaclab/pxr, so registration is safe without a running app.
"""

from .bulb_assembly import BulbAssemblyScene, BulbAssemblySceneCfg
from .ikea_table_assembly import IkeaTableAssemblyScene, IkeaTableAssemblySceneCfg
from .nut_thread_assembly import NutThreadAssemblyScene, NutThreadAssemblySceneCfg

__all__ = [
    "IkeaTableAssemblyScene",
    "IkeaTableAssemblySceneCfg",
    "NutThreadAssemblyScene",
    "NutThreadAssemblySceneCfg",
    "BulbAssemblyScene",
    "BulbAssemblySceneCfg",
]
