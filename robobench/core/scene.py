"""BaseScene — the object world (embodiment-agnostic).

Declares its object assets, describes its layout (+ the goal), and reads/writes its own state.
Knows nothing about the robot. Children must implement every abstract method below. Heavy imports
are deferred so this module imports without AppLauncher.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .config import SimCfg

if TYPE_CHECKING:
    import torch

    from .env import BaseEnv


class BaseScene(ABC):
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._env: BaseEnv | None = None

    @abstractmethod
    def assets(self) -> dict[str, Any]:
        """`{name: asset_cfg}` for the objects to spawn (no robot)."""

    def sim_cfg(self) -> SimCfg:
        """The sim substrate this scene needs (dt + PhysX). Lives here because the scene's contact
        geometry drives the requirements (SDF threads -> big GPU buffers / small dt). Default is a
        plain config; **override** for contact-rich scenes. An `EnvCfg` may patch the result
        (`sim_overrides`) for a specific embodiment. May read `self.cfg`."""
        return SimCfg()

    @abstractmethod
    def reset(self, env_ids: torch.Tensor) -> None:
        """Re-place the objects for `env_ids` (defaults + randomization from cfg)."""

    @abstractmethod
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        """The scene's full restorable state (object poses/vels, weld flags, …)."""

    @abstractmethod
    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        """Restore what `get_state` returned."""

    @abstractmethod
    def describe(self) -> str:
        """**Natural-language** description of the scene **and the goal** (the goal lives here since
        there's no separate task layer), for the agent. E.g. "A table with four threaded studs at
        the corners (±0.25 m) and four loose legs nearby. Goal: screw all four legs into the table."
        (Numeric/low-level geometry is the env's `describe_stage()`.)"""

    @property
    def env(self) -> BaseEnv:
        """The env this scene is bound to. Available after `bind()` (i.e. inside every method except
        `assets()`); raises if accessed before binding."""
        if self._env is None:
            raise RuntimeError("scene is not bound to an env yet (call happens before bind())")
        return self._env

    def bind(self, env: BaseEnv) -> None:
        """Cache the env back-ref + asset handles, once, after the scene is built. Override to grab
        handles (call `super().bind(env)`)."""
        self._env = env

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Step-coupled scene mechanics, run by the env once per `step()` after the sim advances.
        Default no-op. Override for things that must react to the new physics state every step
        (e.g. auto-welding a part the instant it seats). Not for the agent to call."""
