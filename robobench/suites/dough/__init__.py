"""dough — elastoplastic dough manipulation on Newton implicit MPM (coupled MJWarp+MPM).

One scene = one task: `dumpling` (roll the dough ball out into a wrapper with the rolling
pin). Importing this package registers the scene and env as a side effect (`robobench`
discovery contract); everything stays app-free until an env is BUILT — building requires the
Newton venv (`env_newton`, see the README) and `num_envs=1` (the MPM fixed grid spans the
whole scene).

Layout:
  newton_sim.py       DoughSimCfg — the coupled MJWarp+MPM substrate cfg
  coupled_manager.py  the coupled MJWarp+MPM manager — a verbatim twin of the pouring
                      suite's (suites must not import each other; keep the twins in sync)
  scenes/             DumplingScene + DumplingSceneCfg (the scene IS the task)
  configs/            registered env preset ("dough.dumpling", the robot-less tuning binding)
"""

from . import configs, scenes  # noqa: F401  (registration side effects)
