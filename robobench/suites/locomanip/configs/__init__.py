"""Concrete configs for the locomanip suite.

Importing this package registers the suite's canonical runnable env configs into
`robobench.core.ENVS` (see `envs.py`) — so a run/test loads one by name (e.g.
`locomanip.wheel_carry.g1.loco_pink_ik`); names follow `suite.scene[.robot[.control_mode]]`.
Import-light: `EnvCfg` is app-free; only `EnvCfg.build()` needs the app.
"""

from . import envs  # noqa: F401  (registers the suite's EnvCfgs into robobench.core.ENVS)
