"""Privileged graders for the locomanip suite — never shipped to agents.

One module per scene, mirroring `scenes/`. The eval extractor drops this
folder from every agent-facing tree; only grading processes import it.
Import-light (torch deferred), and NOT imported by the suite's `__init__` —
graders are opt-in, constructed by the harness.

A new grader is a small `BaseGrader` subclass: docstring + SCENE (the scene
class it grades) + weighted RUBRIC + one method per stage + setup() +
check_success() — all required, anything missing or mismatched fails at
construction. Register it in GRADERS.
"""

from .box_to_bin import BoxToBinGrader
from .wheel_carry import WheelCarryGrader

GRADERS = {"box_to_bin": BoxToBinGrader, "wheel_carry": WheelCarryGrader}

__all__ = ["BoxToBinGrader", "WheelCarryGrader", "GRADERS"]
