"""Privileged graders for the packing suite — never shipped to agents.

One module per scene, mirroring `scenes/`. The eval extractor drops this
folder from every agent-facing tree; only grading processes import it.
Import-light (torch deferred), and NOT imported by the suite's `__init__` —
graders are opt-in, constructed by the harness.

A new grader is a small `BaseGrader` subclass: docstring + SCENE (the scene
class it grades) + weighted RUBRIC + one method per stage + setup() +
check_success() — all required, anything missing or mismatched fails at
construction. Register it in GRADERS.
"""

from .clear_organic_objects import ClearOrganicObjectsGrader
from .egg_carton import EggCartonGrader
from .pen_holder import PenHolderGrader
from .tool_packing import ToolPackingGrader

GRADERS = {
    "clear_organic_objects": ClearOrganicObjectsGrader,
    "egg_carton": EggCartonGrader,
    "pen_holder": PenHolderGrader,
    "tool_packing": ToolPackingGrader,
}

__all__ = [*sorted(c.__name__ for c in GRADERS.values()), "GRADERS"]
