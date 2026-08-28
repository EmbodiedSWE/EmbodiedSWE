"""packing suite — fit-things-into-containers tasks judged by physics.

Tasks:
  - `clear_organic_objects` (ClearOrganicObjectsScene) — sort the organic fruits & vegetables out of a
    cluttered table into the bin, leaving the non-food clutter (a port of RoboLab's
    ClearOrganicObjectsTask, on the RoboLab scanned assets).
  - `pen_holder` (PenHolderScene) — collect scattered pens into the holder cup.
  - `tool_packing` (ToolPackingScene) — stow tools into the right drawers of an
    articulated tool chest (2 glass doors + 3 drawers, real scanned assets) and
    close it up.

Structure mirrors the assembly suite: `scenes/` (each scene IS one task), `configs/`
(named ENVS bindings), `smokes/` (NullRobot smoke/oracle runs).
"""

from . import scenes  # noqa: F401  (registers the suite's scenes into robobench.core.SCENES)
from . import configs  # noqa: F401  (registers the suite's named env configs into robobench.core.ENVS)
