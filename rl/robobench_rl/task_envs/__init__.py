"""Task envs: subclasses of RoboBenchEnv. `load_task_env_cls(name)` -> class (RoboBenchEnv when name is None)."""
from __future__ import annotations

import importlib

from ..vec_env import RoboBenchEnv

REGISTRY = {"bulb_tuned": ("bulb_tuned", "BulbTunedEnv")}


def load_task_env_cls(name: str | None) -> type[RoboBenchEnv]:
    if not name:
        return RoboBenchEnv
    if name not in REGISTRY:
        raise KeyError(f"unknown task_env {name!r}; have {sorted(REGISTRY)}")
    mod, cls = REGISTRY[name]
    return getattr(importlib.import_module(f"{__package__}.{mod}"), cls)
