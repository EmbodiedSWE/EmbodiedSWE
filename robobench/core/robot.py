"""BaseRobot — the actor (embodiment).

A robot declares its assets + the control modes it supports, and how to read/write its own state.
The per-step action→sim pipeline is handled **generically through a controller**: `bind()`
orchestrates the lifecycle — bind the robot to the env, acquire its sim handles, then build + bind a
controller *to the robot* — and `apply_action` / `action_dim` delegate to that controller. A concrete
robot only fills in **hooks**: `on_bind` (grab handles), `build_controller` (the controller for the
active `control_mode`), and — unless it's a single articulation — `actuator_sink` / `actuator_limits`.
It does not re-implement the lifecycle. Controller-less robots (NullRobot, force-driven debug actors)
just leave `build_controller` returning None.

The env is open, so an agent can drop in its own controller at runtime with `set_controller(...)`
(or override `apply_action` for a fully custom path). Each controller writes its own command to sim
through the `actuator_sink` it captured at bind, so a composite can mix command types per joint group.

Heavy imports are deferred so this module imports without AppLauncher.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .config import BaseCfg

if TYPE_CHECKING:
    import torch

    from .controller import BaseController
    from .env import BaseEnv


@dataclass
class BaseRobotCfg(BaseCfg):
    """Thin shared base for robot configs: carries only the one universal selectable, the
    `control_mode`. Concrete robots subclass this and add their own asset / partial body variant (e.g. hands) /
    `fix_root_link` / ee-frame / gains — morphologies differ too much for a fat shared cfg."""

    #: Which of the robot's `control_modes` to use; "" -> the first one the robot declares. The agent
    #: may switch it (a new actuation of the same hardware).
    control_mode: str = ""  # active control mode; "" selects the robot's first

    #: The robot's scene namespace: its asset key in `env.iscene` and (capitalized) its prim name —
    #: "robot" -> `iscene["robot"]` at `{ENV_REGEX_NS}/Robot`. Single-robot envs keep the default; a
    #: composite parent (`MultiRobot`) stamps each child's name ("left", "right", ...) so several
    #: robots coexist in one scene without key/prim collisions. Concrete robots read it through
    #: `BaseRobot.name` / `BaseRobot.prim_name` rather than hardcoding "robot".
    name: str = field(default="robot", kw_only=True)  # scene-asset key / prim namespace for this robot


class BaseRobot(ABC):
    #: Control modes this embodiment supports (e.g. ("joint", "ee_pose", "osc_impedance")).
    #: Declared per concrete robot; the active one is `self.control_mode`.
    control_modes: tuple[str, ...] = ()

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._env: BaseEnv | None = None
        self.control_mode: str | None = getattr(cfg, "control_mode", None) or (
            self.control_modes[0] if self.control_modes else None
        )
        #: The active controller, built in `bind()` (None for a controller-less robot). The agent may
        #: swap it at runtime via `set_controller()`.
        self.controller: BaseController | None = None
        #: The robot's main articulation handle — set in `on_bind` for the common single-articulation
        #: case (the default `actuator_sink` / `actuator_limits` use it). None until bound.
        self.articulation: Any = None

    # ----- naming (the robot's scene namespace; see BaseRobotCfg.name) --------------------------
    @property
    def name(self) -> str:
        """This robot's scene-asset key (`cfg.name`; "robot" for a plain single-robot env). Concrete
        robots use it in `assets()` and `on_bind` (`env.iscene[self.name]`) so a composite parent can
        re-namespace them just by stamping `cfg.name`."""
        return getattr(self.cfg, "name", None) or "robot"

    @property
    def prim_name(self) -> str:
        """USD prim name under `{ENV_REGEX_NS}`: `cfg.name` with the first letter upper-cased
        ("robot" -> "Robot", "left" -> "Left"), so default single-robot prim paths stay unchanged."""
        n = self.name
        return n[:1].upper() + n[1:]

    # ----- assets / state / description (the robot-specific contract) ---------------------------
    @abstractmethod
    def assets(self) -> dict[str, Any]:
        """`{name: cfg}` for the robot prim(s) (ArticulationCfg / RigidObjectCfg / …)."""

    @abstractmethod
    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset the robot to its home configuration for `env_ids` (also reset `self.controller`)."""

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        """Full restorable state, uniform across single-articulation robots: the articulation's
        sim-resident state (root, joint pos/vel, and the actuator setpoints) plus the controller's own
        state (delegated, so it travels with whatever controller is bound).

        To ADD robot-specific state, extend rather than replace — call `super().get_state(env_ids)` for
        this uniform part, add your own keys, and restore them after `super().set_state(...)`:
            def get_state(self, ids):  s = super().get_state(ids); s["tool"] = ...; return s
            def set_state(self, s, ids):  super().set_state(s, ids); self._tool.write_(s["tool"])
        `set_state` reads only the base keys, so it ignores any extra keys a subclass adds (no clash).
        Override fully only if the robot isn't a single articulation (e.g. multi-body)."""
        state: dict[str, Any] = {}
        art = self.articulation
        if art is not None:
            d = art.data
            state["root"] = d.root_state_w[env_ids].clone()
            state["joint_pos"] = d.joint_pos[env_ids].clone()
            state["joint_vel"] = d.joint_vel[env_ids].clone()
            state["joint_pos_target"] = d.joint_pos_target[env_ids].clone()
            state["joint_effort_target"] = d.joint_effort_target[env_ids].clone()
        if self.controller is not None:
            state["controller"] = self.controller.get_state(env_ids)
        return state

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        """Restore what `get_state` returned: write the articulation's root + joint state and re-apply
        the actuator setpoints (so a restored robot resumes its commanded motion), then restore the
        controller's state."""
        art = self.articulation
        if art is not None:
            art.write_root_state_to_sim(state["root"], env_ids)
            art.write_joint_state_to_sim(state["joint_pos"], state["joint_vel"], env_ids=env_ids)
            art.set_joint_position_target(state["joint_pos_target"], env_ids=env_ids)
            art.set_joint_effort_target(state["joint_effort_target"], env_ids=env_ids)
        if self.controller is not None and "controller" in state:
            self.controller.set_state(state["controller"], env_ids)

    @abstractmethod
    def describe(self) -> str:
        """**Natural-language** description of the embodiment + its control, for the agent."""

    # ----- lifecycle: env -> robot -> controller (concrete; override the HOOKS, not bind) -------
    def bind(self, env: BaseEnv) -> None:
        """Bind the robot to `env`, then build + bind its controller **to the robot**. The chain:
        the env builds the scene, calls this once → it caches the env, `on_bind` grabs sim handles,
        `build_controller` makes the controller for the active mode, and `set_controller` binds it to
        this robot. Override the hooks (`on_bind` / `build_controller`, or `actuator_sink` /
        `actuator_limits` for non-articulation robots), not this."""
        self._env = env
        self.on_bind(env)
        controller = self.build_controller()
        if controller is not None:
            self.set_controller(controller)

    def on_bind(self, env: BaseEnv) -> None:
        """Hook: grab sim handles once after build (e.g. `self.articulation = env.iscene["robot"]`).
        Runs before the controller is built, so the controller can resolve its joints against the
        handle. No-op by default."""

    def build_controller(self) -> BaseController | None:
        """Hook: the controller for the active `control_mode` (its `joint_ids` / solver are resolved
        when it is bound, in `set_controller`). Return None for a controller-less robot (default)."""
        return None

    def set_controller(self, controller: BaseController) -> None:
        """Bind `controller` to this robot and install it as the active one. The **supported seam**
        for an agent to drop in its own controller at runtime —
        `env.robot.set_controller(MyController(...))` — which takes effect on the next `step()` (the
        env dispatches through `self.robot`). Bypasses `control_modes`: a custom controller need not
        be in the menu."""
        controller.bind(self)
        self.controller = controller

    # ----- action + control rate (generic, via the active controller) ---------------------------
    @property
    def action_dim(self) -> int:
        """Width of the action vector = the active controller's (0 if there is none)."""
        return self.controller.action_dim if self.controller is not None else 0

    @property
    def control_period(self) -> int:
        """Physics substeps the env runs per `env.step` = the active controller's `control_period`
        (1 if there is none, i.e. control at sim rate). For a composite this is its slowest leaf."""
        return self.controller.control_period if self.controller is not None else 1

    def control_dt(self, controller: BaseController) -> float | None:
        """Hook: a robot-level control period (s) that OVERRIDES `controller.cfg.dt`; None -> defer to
        the controller. Takes the controller so a robot can pin per-leaf rates of a composite. Read at
        bind and turned into an integer `control_period`."""
        return None

    def apply_action(self, action: torch.Tensor, substep: int = 0) -> None:
        """Run the active controller, which computes **and writes** its command (no-op without one).
        `substep` is the physics-step index in the `env.step` window; the controller fires only on its
        own subdivision (see `BaseController.apply`). Override for a fully custom action path.

        `action` may be any array-like (numpy / list / torch): controllers assume a float32 torch
        tensor on the sim device, so it is coerced here — the one choke point every path shares
        (`env.step`, direct calls). Agent-written code passes numpy, which used to surface as a
        baffling numpy↔cuda-tensor TypeError deep in the controller's EMA math. A no-op (same
        object back) when the input is already a float32 tensor on the sim device."""
        if self.controller is not None:
            import torch

            action = torch.as_tensor(action, dtype=torch.float32, device=self.env.device)
            if action.dim() == 1:  # tolerate a single unbatched action vector
                action = action.unsqueeze(0)
            self.controller.apply(action, substep)

    def actuator_sink(self, command_type: str):
        """The **write path** for a controller's `command_type`: a callable `(command, joint_ids) ->
        None` that sends `command` to sim through the matching joint-target setter. A controller
        captures this once at bind and writes through it itself — so a `composite` can mix command
        types (effort arms + position hands), each leaf writing its own group. Default uses
        `self.articulation`; override for multi-body or non-articulation robots."""
        art = self.articulation
        setter = {
            "position": art.set_joint_position_target,
            "velocity": art.set_joint_velocity_target,
            "effort": art.set_joint_effort_target,
        }[command_type]
        return lambda command, joint_ids: setter(command, joint_ids=joint_ids)

    def actuator_limits(self, joint_ids: Any) -> dict[str, Any]:
        """Per-joint actuation bounds for `joint_ids`, handed to a controller at bind (`self.limits`)
        so it can clamp / scale / validate — most controllers ignore them and stay general. A snapshot
        from the articulation (re-query for live values). Keys: ``pos`` (n, k, 2 = lower/upper),
        ``vel`` (n, k), ``effort`` (n, k). Override for non-articulation robots."""
        d = self.articulation.data
        return {
            "pos": d.joint_pos_limits[:, joint_ids, :],
            "vel": d.joint_vel_limits[:, joint_ids],
            "effort": d.joint_effort_limits[:, joint_ids],
        }

    @property
    def env(self) -> BaseEnv:
        """The env this robot is bound to. Available after `bind()`; raises if accessed before."""
        if self._env is None:
            raise RuntimeError("robot is not bound to an env yet (call happens before bind())")
        return self._env

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Step-coupled robot bookkeeping, run by the env **once per physics substep** (after the sim
        advances), so it tracks every physics step even under control decimation. Default no-op.
        Override for things that must track the new state every step (e.g. advancing a controller's
        internal target/integrator). Not for the agent to call."""
