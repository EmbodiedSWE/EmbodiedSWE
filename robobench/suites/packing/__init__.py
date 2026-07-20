"""packing suite — fit-things-into-containers tasks judged by physics.

Tasks:
  - `crate` (CratePackingScene) — pack a manifest of odd-shaped parts into a shipping
    crate tight enough that the lid rests flush on the rim. The lid is the judge: any
    part proud of the rim physically blocks it. (Design decisions worth keeping from
    the brief: the crate interior is derived from the reference packing times an
    `oversize` dial, so required precision is a knob, not an accident; the manifest
    only fits LAYERED — a flat single-layer footprint exceeds the floor.)

Structure mirrors the assembly suite: `scenes/` (each scene IS one task), `configs/`
(named ENVS bindings), `smokes/` (NullRobot smoke/oracle runs).
"""

from . import scenes  # noqa: F401  (registers the suite's scenes into robobench.core.SCENES)
from . import configs  # noqa: F401  (registers the suite's named env configs into robobench.core.ENVS)
