"""JointController — pass-through: the action *is* the joint targets (optionally shaped).

The simplest control mode (and a handy debug/teleop baseline): the action vector is written to the
controlled joints, optionally through an affine map `target = scale * action + offset` (default
identity = raw pass-through). Controls `cfg.joint_names` (default: all of the robot's joints). It is
**type-flexible** — the same pass-through works for position *or* effort targets — so `command_type`
is **required** at construction (no default): `JointController(cfg, command_type="position")` writes
position targets; `command_type="effort"` writes torques. Whoever builds it (the robot's
`build_controller`) picks the type to match the actuators of those joints.

Note `scale`/`offset` are **action shaping**, not a PD gain — for position control the PD gain lives in
the robot's actuators (`RobotCfg`), not here. Use `offset` e.g. to centre actions on the home pose
(action = delta) and `scale` to map a normalized action to joint range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from robobench.core import CONTROLLERS, BaseController, BaseControllerCfg

if TYPE_CHECKING:
    import torch


@dataclass
class JointControllerCfg(BaseControllerCfg):
    """Config for `JointController`: which joints, and the affine action→target shaping.
    `scale`/`offset` may be a scalar or anything that broadcasts against the
    `(num_envs, len(joint_ids))` action (e.g. a per-joint tensor on the sim device); identity by
    default (raw pass-through)."""

    joint_names: tuple[str, ...] | None = None  # None -> all of the robot's joints
    scale: Any = 1.0  # action multiplier; 1.0 = none
    offset: Any = 0.0  # added after scaling; 0.0 = none (e.g. the home pose for delta control)


@CONTROLLERS.register("joint")
class JointController(BaseController):
    """`target = scale * action + offset` for `joint_ids` (position or effort, per `command_type`);
    `action_dim` = number of those joints. (`bind` / `apply` are inherited; only the joint selection
    + the affine math are local.)"""

    def __init__(self, cfg: JointControllerCfg | None = None, *, command_type: str) -> None:
        super().__init__(cfg, command_type=command_type)  # command_type required: no silent default
        # Cache the shaping so compute() stays allocation-light; identity by default.
        self.scale: Any = getattr(cfg, "scale", 1.0) if cfg is not None else 1.0
        self.offset: Any = getattr(cfg, "offset", 0.0) if cfg is not None else 0.0

    def _resolve_joints(self, robot: Any) -> Any:
        names = getattr(self.cfg, "joint_names", None) if self.cfg is not None else None
        # `articulation` is the concrete robot's Isaac handle; find_joints -> (indices, names).
        return robot.articulation.find_joints(names)[0] if names else list(range(robot.articulation.num_joints))

    @property
    def action_dim(self) -> int:
        return len(self.joint_ids)

    def compute(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.scale + self.offset  # identity (1.0, 0.0) -> the action itself
