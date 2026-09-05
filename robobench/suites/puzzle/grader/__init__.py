"""Privileged graders for the puzzle suite — never shipped to agents.

One module per scene, mirroring `scenes/`. Import-light (torch deferred), and NOT imported
by the suite's `__init__` — graders are opt-in, constructed by the harness.
"""

from .classify_objects import ClassifyObjectsGrader
from .coffee_service import CoffeeServiceGrader
from .push_shapes import PushShapesGrader
from .spatula_flip_serve import SpatulaFlipServeGrader
from .stack_blocks import StackBlocksGrader
from .syringe_dosing import SyringeDosingGrader

GRADERS = {
    "classify_objects": ClassifyObjectsGrader,
    "coffee": CoffeeServiceGrader,
    "push_shapes": PushShapesGrader,
    "spatula": SpatulaFlipServeGrader,
    "stack_blocks": StackBlocksGrader,
    "syringe": SyringeDosingGrader,
}

__all__ = [*sorted(c.__name__ for c in GRADERS.values()), "GRADERS"]
