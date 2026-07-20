"""articulated suite — articulated-object manipulation (operate jointed mechanisms).

Standard category in the manipulation literature (doors/drawers/knobs in robocasa,
dishwashers/articulated fixtures in BEHAVIOR): the work is OPERATING a mechanism's
joints, not relocating free objects.

Tasks:
  - `safe` (CombinationSafeScene) — a safe with a rotary dial whose 3-number
    combination is random each episode and never revealed. Detents (brief brake pulses
    on the dial at the combination angles) are the only clue: the robot feels them as
    hitches in its commanded-vs-actual dial rotation.
  - `scale` (BalanceScaleScene) — sort five identical-looking boxes by hidden mass
    using a two-tray beam balance (arrest/release discipline; centred placement).
Structure mirrors the assembly suite: `scenes/`, `configs/`, `smokes/`.
"""

from . import scenes  # noqa: F401  (registers the suite's scenes into robobench.core.SCENES)
from . import configs  # noqa: F401  (registers the suite's named env configs into robobench.core.ENVS)
