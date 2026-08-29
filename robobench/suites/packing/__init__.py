"""packing suite — fit-things-into-containers tasks judged by physics.

Tasks:
  - `egg_carton` (EggCartonScene) — choose one of four side-lying eggs, reorient it, and
    seat it upright in any physical carton pocket.
  - `pen_holder` (PenHolderScene) — collect scattered pens into the holder cup.
  - `tool_packing` (ToolPackingScene) — stow tools into the right drawers of an
    articulated tool chest (2 glass doors + 3 drawers, real scanned assets) and
    close it up.

Structure mirrors the assembly suite: `scenes/` (each scene IS one task), `configs/`
(named ENVS bindings), `smokes/` (NullRobot smoke/oracle runs).
"""

from . import scenes  # noqa: F401  (registers the suite's scenes into robobench.core.SCENES)
from . import configs  # noqa: F401  (registers the suite's named env configs into robobench.core.ENVS)
