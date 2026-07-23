"""envbuild — construct experiment environments from robobench's registries.

Pipeline (build.py is the only module that knows this order):
    resolve CLI intent -> registered presets (the feasibility gate)
    -> extract a minimal benchmark tree per stage
    -> patch ablation arms into the tree (by construction)
    -> boot-validate the REGISTERED preset from the tree + harvest describe()
    -> write the resolved receipt (re-runnable provenance)

Task-prompt rendering (prompts.py) is currently unplugged from the pipeline
while the prompt design is revisited.

Every module is data-in/data-out and usable alone. Architecture overview and
diagrams: eval/README.html.
"""

from .build import StageSpec, build_experiment
from .resolve import list_envs, resolve_preset

__all__ = ["StageSpec", "build_experiment", "list_envs", "resolve_preset"]
