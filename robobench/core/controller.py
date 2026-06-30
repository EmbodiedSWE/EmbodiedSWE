"""BaseController — the shared, pluggable mid-level control layer (the "System 1" of the robot).

A controller maps an agent action to a joint command for the subset of a robot's DOFs it owns, and
**writes it to sim itself** through a *sink* captured at bind. A robot's active `control_mode`
selects a controller, and the robot delegates `action_dim` / `apply` to it. Controllers are
**reusable across robots** (one Diff-IK serves any arm, Pink IK any humanoid) and registered in
`CONTROLLERS`; binding to a robot resolves joint indices, captures the write path, and builds any
solver. Only genuinely robot-specific controllers live with the robot.

Two responsibilities, kept separate:
  - `compute(action) -> command` — **pure math** for the DOFs it owns (`joint_ids`), in `command_type`
    units. Stays testable and is what wrappers (residual, policy) stack on.
  - `apply(action)` — compute **then write** through the sink. Default = `sink(compute(action))`.

`command_type` ∈ {`"position"`, `"velocity"`, `"effort"`} is **leaf-level** — it picks which
`set_joint_{position,velocity,effort}_target` the leaf's sink routes to:
  - **position / velocity** — the robot's actuators run the PD loop (gains in the `RobotCfg`).
  - **effort** — the controller computes torque itself; those joints' actuators must be in a
    pass-through / zero-PD mode (a `RobotCfg` fact).
Because each leaf writes its **own** group through its **own** sink, a `composite` can MIX types
(effort arms + position hands) — there is no single command_type to agree on. The consistency rule is
per-group: a leaf's joints must have actuators matching its `command_type` (a robot-cfg concern; the
per-joint limits the robot passes at bind, `self.limits`, are there to help — most leaves ignore them).

Heavy imports (isaaclab / pxr / solvers) are deferred so this module — and registering a controller
— stays app-free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .config import BaseCfg, info

if TYPE_CHECKING:
    import torch

    from .robot import BaseRobot


@dataclass
class BaseControllerCfg(BaseCfg):
    """Shared base for controller configs (a `BaseCfg`, so fields use `tunable()` / `info()`). The
    split is meaningful here: **`info` = robot-supplied structural wiring** (joint names, ee-frames,
    URDF — set by the robot in `build_controller`, since only it knows its kinematics); **`tunable` =
    knobs a higher layer (curriculum / env-register, robot-mediated) may dial** (gains, costs, scale).
    Thin base — concrete controllers add their own fields. Not every controller needs a cfg (e.g.
    `composite` takes a list of sub-controllers)."""

    #: Control period in seconds; `None` -> every physics step. Resolved to an integer `control_period`
    #: at `bind`. Keyword-only so it doesn't shift subclasses' positional args; the robot may override it.
    dt: float | None = info(None, doc="control period (s); None -> every physics step", kw_only=True)


class BaseController(ABC):
    """Map `(num_envs, action_dim)` actions to a joint command for `joint_ids` and write it to sim.
    Subclass and register with `@CONTROLLERS.register("name")`. `cfg` carries the per-robot wiring
    (joint names, ee-frame, gains); binding resolves it against the robot. A *leaf* overrides
    `_resolve_joints` + `compute` (and inherits `bind`/`apply`); a composite overrides `bind`/`apply`."""

    def __init__(self, cfg: Any = None, *, command_type: str | None = None) -> None:
        self.cfg = cfg
        #: Which actuator setter this (leaf) controller writes through — "position" / "velocity" /
        #: "effort". **No default**: it's set explicitly, either by a fixed-type subclass (OSC ->
        #: "effort") or passed by the caller for a flexible one (JointController can be position OR
        #: effort). The robot's `build_controller` — which knows its actuator config — chooses it, and
        #: it must match that config (see module docstring). A composite leaves it None (no single
        #: type; it fans out). A writing controller with None command_type errors at `bind`.
        self.command_type: str | None = command_type
        self._robot: BaseRobot | None = None
        #: Articulation DOF indices this controller drives — resolved in `bind()`.
        self.joint_ids: Any = None
        #: Write path captured at bind: `sink(command, joint_ids) -> None` for this `command_type`.
        self._sink: Any = None
        #: Per-joint limits for `joint_ids` (pos/vel/effort), passed by the robot at bind for the
        #: controller to clamp/scale/validate against if it wants. Most controllers ignore them.
        self.limits: Any = None
        #: Physics substeps between this controller's updates (1 = every step). Resolved in `bind()`
        #: from the effective control dt and the sim dt; see `control_period`.
        self._control_period: int = 1

    @property
    def robot(self) -> BaseRobot:
        """The robot this controller is bound to. Available after `bind()`; raises before."""
        if self._robot is None:
            raise RuntimeError("controller is not bound to a robot yet (call happens before bind())")
        return self._robot

    @property
    def dt(self) -> float | None:
        """This controller's preferred control period in seconds (`cfg.dt`); None -> every physics
        step. The robot may override it at bind (`robot.control_dt`)."""
        return getattr(self.cfg, "dt", None) if self.cfg is not None else None

    @property
    def control_period(self) -> int:
        """Physics substeps between this controller's updates (1 = every step), resolved at `bind`."""
        return self._control_period

    def bind(self, robot: BaseRobot) -> None:
        """Bind to `robot`: resolve `joint_ids`, capture the actuator sink for this `command_type`,
        snapshot the joint limits for those DOFs, and resolve `control_period` (robot override ->
        `cfg.dt` -> sim rate, against the sim dt). Override `_resolve_joints` (not this) for a leaf;
        a composite overrides `bind` to bind its sub-controllers instead."""
        self._robot = robot
        self.joint_ids = self._resolve_joints(robot)
        if self.command_type is None:
            raise ValueError(
                f"{type(self).__name__} has no command_type — pass command_type='position'|'velocity'|"
                "'effort' at construction (or set it on a fixed-type subclass)."
            )
        self._sink = robot.actuator_sink(self.command_type)
        self.limits = robot.actuator_limits(self.joint_ids)
        dt = robot.control_dt(self)  # robot override wins...
        if dt is None:
            dt = self.dt  # ...else this controller's own cfg.dt...
        self._control_period = 1 if dt is None else max(1, round(dt / robot.env.dt))  # ...else every step

    def _resolve_joints(self, robot: BaseRobot) -> Any:
        """Which articulation DOFs this controller drives. Default: all of them. Override to select a
        subset (by name, or an IK chain's joints)."""
        return list(range(robot.articulation.num_joints))

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Width of the action slice this controller consumes (depends on the control mode)."""

    @abstractmethod
    def compute(self, action: torch.Tensor) -> torch.Tensor:
        """**Pure**: map a `(num_envs, action_dim)` action to `(num_envs, len(joint_ids))` joint
        commands for this controller's DOFs, in `command_type` units. No sim writes (see `apply`)."""

    def apply(self, action: torch.Tensor, substep: int = 0) -> None:
        """Compute and write the command through the sink, but only on this controller's subdivision
        (`substep` a multiple of `control_period`); in between, the previously-written command persists
        (actuator PD holds a position target; a torque stays applied). `substep` is the physics-step
        index within the `env.step` window. A composite overrides this to fan out to its sub-controllers."""
        if substep % self._control_period == 0:
            self._sink(self.compute(action), self.joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset any internal state (IK integrator, policy hidden state). Default no-op; override for
        stateful controllers."""

    def get_state(self, env_ids: torch.Tensor | None = None) -> dict[str, Any]:
        """The controller's own restorable state (smoothing buffer, IK warm-start, ...). Default empty
        (stateless); the robot's `get_state` delegates here so state travels with the controller."""
        return {}

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor | None = None) -> None:
        """Restore what `get_state` returned. Default no-op (stateless)."""
