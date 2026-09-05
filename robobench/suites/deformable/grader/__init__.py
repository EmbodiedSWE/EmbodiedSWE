"""Privileged graders for the deformable suite — never shipped to agents.

One module per scene, mirroring `scenes/`: `tshirt.py` (TshirtFoldingGrader), `latte.py`
(LatteGrader), `dumpling.py` (DumplingGrader), `shoe_knot.py` (ShoeKnotGrader). Each has
programmatic rungs from the scene's own metrics and cfg thresholds plus ONE final VLM
plausibility gate (`vlm_judge.py` in this package — kept out of `robobench.core`, which ships to agents), the last rung of the ladder — it can only take
credit away, never add it. Import-light (torch deferred) and NOT imported by the suite's
`__init__`; the harness constructs graders through `GRADERS`.

The gate renders one frame per env at verdict time, so the grading app must run with cameras
enabled (`eval/grader/grade.py --render`); without a frame the gate records the error and fails.
"""

from .dumpling import DumplingGrader
from .latte import LatteGrader
from .shoe_knot import ShoeKnotGrader
from .tshirt import TshirtFoldingGrader

GRADERS = {
    "dumpling": DumplingGrader,
    "knot": ShoeKnotGrader,
    "latte": LatteGrader,
    "tshirt": TshirtFoldingGrader,
}

__all__ = [*sorted(c.__name__ for c in GRADERS.values()), "GRADERS"]
