"""Privileged graders for the assembly suite — never shipped to agents.

One module per scene, mirroring `scenes/`. The eval extractor drops this
folder from every agent-facing tree; only grading processes import it.
Import-light (torch deferred), and NOT imported by the suite's `__init__` —
graders are opt-in, constructed by the harness.

A new grader is a small `BaseGrader` subclass: docstring + SCENE (the scene
class it grades) + weighted RUBRIC + one method per stage + setup() +
check_success() — all required, anything missing or mismatched fails at
construction. Register it in GRADERS.

Weight convention (shared by every suite): identical parts share one k/N
fraction stage; every part is worth the same; within a part the rungs
grasped (once) -> aligned (live) -> seated (live, alignment gates x linear
depth ramp between the scene's own cfg datums) carry equal weight; progress
reaches 1.0 iff the scene's own success holds.
"""

from .allen_bolt_assembly import AllenBoltAssemblyGrader
from .bulb_assembly import BulbAssemblyGrader
from .ikea_table_assembly import IkeaTableAssemblyGrader
from .nut_thread_assembly import NutThreadAssemblyGrader
from .pc_gpu_assembly import PcGpuAssemblyGrader
from .pc_gpu_ram_assembly import PcGpuRamAssemblyGrader
from .pc_motherboard_assembly import PcMotherboardAssemblyGrader
from .pc_ram_assembly import PcRamAssemblyGrader
from .so101_assembly import SO101AssemblyGrader

GRADERS = {
    "allen_bolt": AllenBoltAssemblyGrader,
    "bulb": BulbAssemblyGrader,
    "ikea_table": IkeaTableAssemblyGrader,
    "nut_thread": NutThreadAssemblyGrader,
    "pc_gpu": PcGpuAssemblyGrader,
    "pc_gpu_ram": PcGpuRamAssemblyGrader,
    "pc_motherboard": PcMotherboardAssemblyGrader,
    "pc_ram": PcRamAssemblyGrader,
    "so101": SO101AssemblyGrader,
}

__all__ = [*sorted(c.__name__ for c in GRADERS.values()), "GRADERS"]
