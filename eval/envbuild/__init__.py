"""envbuild — construct experiment environments from robobench's registries.

Pipeline (build.py is the only module that knows this order):
    resolve CLI intent -> registered presets (the feasibility gate)
    -> extract a minimal benchmark tree per stage
    -> patch ablations into the tree (by construction, from the condition's `features:`)
    -> boot-validate the REGISTERED preset from the tree + harvest describe()
    -> write the resolved receipt (re-runnable provenance)

condition.py is the single reader of a condition yaml (rules / skills / tools / features);
every other module takes the loaded `Condition` rather than parsing config itself.

prompts.py and tools.py are NOT part of the build pipeline: they assemble the per-run /task
folder (instructions + task + rules + skills + tool docs) and are invoked by the run launcher,
so one built world can host many prompt conditions.

Every module is data-in/data-out and usable alone. Architecture overview and
diagrams: eval/README.html.
"""

from .build import StageSpec, build_experiment
from .condition import Condition, load as load_condition
from .resolve import list_envs, resolve_preset

__all__ = [
    "Condition",
    "StageSpec",
    "build_experiment",
    "list_envs",
    "load_condition",
    "resolve_preset",
]
