"""Privileged graders for the assembly suite — never shipped to agents.

One module per scene, mirroring `scenes/`. The eval extractor drops this
folder from every agent-facing tree; only grading processes import it.
Import-light (torch deferred), and NOT imported by the suite's `__init__` —
graders are opt-in, constructed by the harness.

A new grader is a small `BaseGrader` subclass: docstring + SCENE (the scene
class it grades) + weighted RUBRIC + one method per stage + setup() +
check_success() — all required, anything missing or mismatched fails at
construction. Register it in GRADERS.
"""

from .bulb_assembly import BulbAssemblyGrader
from .pc_ram_assembly import PcRamAssemblyGrader

GRADERS = {"bulb": BulbAssemblyGrader, "pc_ram": PcRamAssemblyGrader}

__all__ = ["BulbAssemblyGrader", "PcRamAssemblyGrader", "GRADERS"]
