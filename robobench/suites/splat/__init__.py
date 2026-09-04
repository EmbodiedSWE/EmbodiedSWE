"""splat suite — sim proxies of Gaussian-splat captures (GSWorld-style workcells).

Each scene is a plain physics proxy of a captured workspace (kinematic table box + ground + light)
carrying the capture's layout — robot base at the origin, table top at a known height — so a
rollout can be rendered photoreal by `gsworld` (robot link splats re-posed from the sim, the
scanned table/scene splats static in the same frame). Structure mirrors the assembly suite:
`scenes/`, `configs/`.
"""

from . import scenes  # noqa: F401  (registers scenes into robobench.core.SCENES)
from . import configs  # noqa: F401  (registers named env configs into robobench.core.ENVS)
