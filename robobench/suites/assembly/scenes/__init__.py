"""Scenes for the assembly suite — each scene *is* a task (carries its own goal + optional verifier).

Importing this package registers every scene into `robobench.core.SCENES`. Import-light: scene
modules defer isaaclab/pxr, so registration is safe without a running app.
"""

from .ikea_table_assembly import IkeaTableAssemblyScene, IkeaTableAssemblySceneCfg

__all__ = ["IkeaTableAssemblyScene", "IkeaTableAssemblySceneCfg"]
