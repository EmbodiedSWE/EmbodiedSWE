"""BaseScene — the object world (embodiment-agnostic).

Declares its object assets, describes its layout (+ the goal), and reads/writes its own state.
Knows nothing about the robot. Children must implement every abstract method below. Heavy imports
are deferred so this module imports without AppLauncher.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from .config import SimCfg

if TYPE_CHECKING:
    import torch

    from .env import BaseEnv


class BaseScene(ABC):
    #: The scene's L4 physics dials: cfg fields it can re-apply PER ENV after the build (via
    #: `apply_physical_params` — post-build view-writable physics: material frictions, masses, joint
    #: drives), each mapped to a pre-baked sampling band (dist spec, see data_engine's sampler) or
    #: None = appliable but not sampled (bind still applies the nominal). The data engine samples
    #: these automatically, one world per env; the cfg default is always the nominal. Empty on the
    #: base; a concrete scene that implements the hook declares its fields + bands here.
    #: SCOPE: physical world parameters ONLY — friction, mass, material/drive constants. Never
    #: sim/solver settings (dt, iterations), never grading thresholds or controller gains, and
    #: never start-pose jitter (initial conditions belong to the scene's own reset and the phase
    #: reset conditions, not to physics sampling).
    PHYSICAL_PARAMS: ClassVar[dict[str, dict | None]] = {}

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

    def apply_physical_params(self, env: BaseEnv, values: dict[str, list]) -> None:
        """Write world-physics values PER ENV through the sim views (post-build) — `values[name]`
        holds one value per env, for names from `PHYSICAL_PARAMS`. A scene that declares
        `PHYSICAL_PARAMS` implements this and routes its own nominal application in `bind()` through
        it (uniform values), so the nominal path and per-env sampling share one code path. The
        base supports no fields: any request here is a caller bug — fail loudly, never silently
        skip (silently unapplied values would mislabel recorded data)."""
        if values:
            raise ValueError(f"{type(self).__name__} declares no PHYSICAL_PARAMS; cannot apply {sorted(values)}")

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Step-coupled scene mechanics, run by the env **once per physics substep** (after the sim
        advances), so they react at sim rate even under control decimation. Default no-op. Override for
        things that must react to the new physics state every step (e.g. auto-welding a part the instant
        it seats). Not for the agent to call."""
