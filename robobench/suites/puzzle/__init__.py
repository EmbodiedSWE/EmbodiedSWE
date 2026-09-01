"""puzzle suite — multi-step appliance / tool puzzles with quantitative outcomes.

The category merges the former `articulated` (operate a jointed mechanism's state
machine) and `tool_use` (tool-mediated manipulation scored by what the tool did)
suites under the benchmark's meta-naming (2026-08-13): the common thread is a
PUZZLE-like task graph — preconditions, ordering, and metered outcomes — rather
than free-object relocation.

Tasks:
  - `push_t` (PushTScene) — push a T-shaped block onto its matching target pad.
  - `push_shapes` (PushShapesScene) — push three shaped blocks (T/X/L) onto their own
    matching pads, correcting each block's yaw; multi-stage successor to `push_t`.
  - `coffee` (CoffeeServiceScene, the microwave port's Franka-native successor) —
    operate a capsule coffee machine's state machine (closed-cover + pod + cup
    preconditions, start/stop toggle, abort-on-early-open, spill abort, brew
    timer, run lamp + visible stream): brew one capsule coffee and serve the
    filled mug on the tray.
  - `syringe` (SyringeDosingScene) — draw a full syringe and dispense exactly one
    third into each of three wells.
  - `spatula` (SpatulaFlipServeScene) — flip a patty with a spatula and serve it
    onto a plate (non-prehensile payload control).
Structure mirrors the assembly suite: `scenes/`, `configs/`, `smokes/`.
"""

from . import scenes  # noqa: F401  (registers the suite's scenes into robobench.core.SCENES)
from . import configs  # noqa: F401  (registers the suite's named env configs into robobench.core.ENVS)
