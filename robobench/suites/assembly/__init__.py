"""assembly suite — IKEA-style furniture screw-assembly (a table + four threaded legs).

The first robobench suite. Holds its `scenes/` (each scene *is* a task — it carries its own goal
and an optional verifier; e.g. `assemble_table`, `screw_one_leg`), the furniture `assets/`, and
concrete `configs/`; it reuses `robobench.core` + `robobench.robots`. Built by lifting working
components out of `../../../legacy/four_leg_env.py` (the weld lifecycle) and `build_assets.py` /
`table.usd`.
"""

from . import scenes  # noqa: F401  (registers the suite's scenes into robobench.core.SCENES)
from . import configs  # noqa: F401  (registers the suite's named env configs into robobench.core.ENVS)
