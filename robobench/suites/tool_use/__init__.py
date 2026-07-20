"""tool_use suite — tool-mediated manipulation with quantitative outcomes.

Standard category in the manipulation literature (simtoolreal/DexToolBench, tool-use
benchmarks): the robot holds a TOOL and the score is what the tool did — an amount
moved, a mark made — not where an object ended up.

Tasks:
  - `syringe` (SyringeDosingScene) — draw a full syringe and dispense exactly one third
    into each of three wells.
  - `whiteboard` (WhiteboardWordScene) — write the episode's word with a marker, then
    erase and redo the worst letter.
Structure mirrors the assembly suite: `scenes/`, `configs/`, `smokes/`.
"""

from . import scenes  # noqa: F401  (registers the suite's scenes into robobench.core.SCENES)
from . import configs  # noqa: F401  (registers the suite's named env configs into robobench.core.ENVS)
