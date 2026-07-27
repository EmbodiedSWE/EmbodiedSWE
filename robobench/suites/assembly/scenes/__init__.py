"""Scenes for the assembly suite — each scene *is* a task (carries its own goal + optional grader).

Importing this package registers every scene into `robobench.core.SCENES`. Import-light: scene
modules defer isaaclab/pxr, so registration is safe without a running app.
"""

from .allen_bolt_assembly import AllenBoltAssemblyScene, AllenBoltAssemblySceneCfg
from .bulb_assembly import BulbAssemblyScene, BulbAssemblySceneCfg
from .ikea_table_assembly import IkeaTableAssemblyScene, IkeaTableAssemblySceneCfg
from .nut_thread_assembly import NutThreadAssemblyScene, NutThreadAssemblySceneCfg
from .pc_gpu_assembly import PcGpuAssemblyScene, PcGpuAssemblySceneCfg
from .pc_motherboard_assembly import PcMotherboardAssemblyScene, PcMotherboardAssemblySceneCfg
from .so101_assembly import SO101AssemblyScene, SO101SceneCfg

__all__ = [
    "IkeaTableAssemblyScene",
    "IkeaTableAssemblySceneCfg",
    "NutThreadAssemblyScene",
    "NutThreadAssemblySceneCfg",
    "BulbAssemblyScene",
    "BulbAssemblySceneCfg",
    "AllenBoltAssemblyScene",
    "AllenBoltAssemblySceneCfg",
    "PcGpuAssemblyScene",
    "PcGpuAssemblySceneCfg",
    "PcMotherboardAssemblyScene",
    "PcMotherboardAssemblySceneCfg",
    "SO101AssemblyScene",
    "SO101SceneCfg",
]
