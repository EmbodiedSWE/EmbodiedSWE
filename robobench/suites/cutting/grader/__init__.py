"""Privileged graders for the cutting suite — never shipped to agents.

One module per scene, mirroring `scenes/`. Import-light (torch deferred), and NOT imported
by the suite's `__init__` — graders are opt-in, constructed by the harness.
"""

from .dice_food import DiceFoodGrader
from .slice_food import SliceFoodGrader

GRADERS = {"slice": SliceFoodGrader, "dice": DiceFoodGrader}

__all__ = ["SliceFoodGrader", "DiceFoodGrader", "GRADERS"]
