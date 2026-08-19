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
    #: Sampling declarations for the data engine (see data_engine/engine/sampler.py for the
    #: distribution grammar): each entry maps a cfg field (its default = the nominal) to a band.
    #: PHYSICAL_PARAMS varies world physics per env at generation; VISUAL_PARAMS varies the LOOK
    #: stage-wide per render pass at replay (lights/materials are shared prims — never per env).
    PHYSICAL_PARAMS: ClassVar[dict[str, dict | None]] = {}
    VISUAL_PARAMS: ClassVar[dict[str, dict | None]] = {}

    #: Named EXTERNAL viewpoints for visual replay (data_engine render.py) — where to stand to see
    #: THIS scene's geometry, authored next to it. Each entry: {"eye": (x,y,z), "target": (x,y,z),
    #: "focal": mm} — env-origin-relative on the work surface — plus optional "bands": per-episode
    #: pose randomization with the sampler grammar on {eye,target}_{x,y,z} (nominal = the declared
    #: value). Ego (robot-mounted) views live on the ROBOT's `CAMERAS` instead.
    CAMERAS: ClassVar[dict[str, dict]] = {}

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
        """Step-coupled scene mechanics, run by the env **once per physics substep** (after the sim
        advances), so they react at sim rate even under control decimation. Default no-op. Override for
        things that must react to the new physics state every step (e.g. auto-welding a part the instant
        it seats). Not for the agent to call."""

    def apply_physical_params(self, env: BaseEnv, values: dict[str, list]) -> None:
        """Write a `PHYSICAL_PARAMS` draw PER ENV — `values[name]` is one value per env slot,
        written through the PhysX views (frictions, masses, ...). Unlike the visual hook below
        there is no harmless default: a scene that declares bands MUST override this, or sampled
        batches would silently stay nominal — so the default fails loudly instead."""
        if values:
            raise NotImplementedError(
                f"{type(self).__name__} declares PHYSICAL_PARAMS but does not implement "
                "apply_physical_params")

    def apply_visual_params(self, env: BaseEnv, values: dict[str, Any]) -> None:
        """Apply the LIVE subset of a `VISUAL_PARAMS` draw — attribute writes (light intensity, a
        texture file), material rebinds — one stage-wide value per knob. The engine already wrote the
        whole draw onto the scene cfg BEFORE the build, so build-consumed knobs (fields `assets()`
        reads, like a table preset) need no branch here; override only for knobs that must be written
        onto live prims. Default no-op: every knob is build-consumed."""
