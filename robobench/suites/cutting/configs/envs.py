"""Canonical runnable env configs for the cutting suite — registered in `ENVS` by name.

Baseline binding, scene physics only (a robot is bound by the driver, like the other
suites): `cutting.slice` = transverse slicing, default carrot. Variants (another food) are
cheap cfg copies — nothing here is locked.
"""

from __future__ import annotations

from robobench.core import EnvCfg, register_env

SUITE = "cutting"

# Pre-split carrot on the chopping board; knife presses release the scored-plane welds.
# -> "cutting.slice"
register_env(SUITE, lambda: EnvCfg(scene="slice", robot="null", env_spacing=3))
