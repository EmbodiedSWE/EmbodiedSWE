"""String -> factory registries for the swappable layers (scene / robot).

Register concrete classes at import time, then a run resolves them by name
from its `EnvCfg`. This decouples "what exists" from "what a run uses", so adding a scene or robot
never edits the other. Import-light on purpose (no isaaclab), so it's safe to import
without `AppLauncher` (e.g. to list what's available, or in tests).

There is deliberately no `TASKS` registry: there is no task layer. Each scene *is* one task — it
carries its own goal (`scene.describe()`); success criteria live in an optional, hidden Grader.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, overload

_F = TypeVar("_F")


class _Registry:
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._factories: dict[str, Callable[..., Any]] = {}

    # Overloads so the decorator form is identity-typed: `@REG.register("x")` on a class returns the
    # class unchanged (not the internal lambda), so `MyClass(...)` type-checks normally downstream.
    @overload
    def register(self, name: str, factory: None = None) -> Callable[[_F], _F]: ...
    @overload
    def register(self, name: str, factory: _F) -> _F: ...

    def register(self, name: str, factory: Any = None) -> Any:
        """Register `factory` under `name`. Usable directly or as a decorator:

            ROBOTS.register("franka", FrankaCfg)
            @SCENES.register("assemble_table")
            class AssembleTableScene: ...
        """
        if factory is None:  # decorator form
            return lambda f: self.register(name, f)
        if name in self._factories:
            raise KeyError(f"{self._kind} '{name}' already registered")
        self._factories[name] = factory
        return factory

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._factories:
            raise KeyError(f"{self._kind} '{name}' not registered. Known: {self.list()}")
        return self._factories[name]

    def list(self) -> list[str]:
        return sorted(self._factories)


SCENES = _Registry("scene")
ROBOTS = _Registry("robot")
CONTROLLERS = _Registry("controller")
ENVS = _Registry("env")
