"""CompositeController — run several controllers over disjoint DOF groups, each writing its own.

For modes that mix controllers: G1 upper-body = arms/waist by Pink IK **+** hands by direct joint
targets; loco-manip = arms by IK **+** legs by a frozen policy. The action is split into consecutive
slices — one per sub-controller, sized by each `action_dim` — and each sub **applies its own slice**
(computes and writes its own joints through its own sink). Because each leaf writes itself, the
sub-controllers may use **different `command_type`s** — e.g. effort arms + position hands — with no
single command to agree on. `action_dim` = sum of the parts; `joint_ids` = their union (introspection).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robobench.core import CONTROLLERS, BaseController

if TYPE_CHECKING:
    import torch


@CONTROLLERS.register("composite")
class CompositeController(BaseController):
    """Compose sub-controllers over disjoint joint groups, each writing its own command (possibly a
    different `command_type`). Built directly with the sub-controllers (the robot knows its modes),
    not from a registry name."""

    def __init__(self, controllers: list[BaseController]) -> None:
        super().__init__(cfg=None)
        self.controllers = controllers

    def bind(self, robot: Any) -> None:
        # Bind each sub (each resolves its own joint_ids + captures its own sink/limits for its own
        # command_type). The composite itself has no single sink — it never writes directly.
        self._robot = robot
        for c in self.controllers:
            c.bind(robot)
        self.joint_ids = [j for c in self.controllers for j in c.joint_ids]  # union, for introspection

    @property
    def action_dim(self) -> int:
        return sum(c.action_dim for c in self.controllers)

    def compute(self, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "CompositeController is heterogeneous (sub-controllers may differ in command_type) — there "
            "is no single command to return. Use apply(), which fans out to each sub-controller."
        )

    def apply(self, action: torch.Tensor) -> None:
        i = 0
        for c in self.controllers:
            c.apply(action[:, i : i + c.action_dim])
            i += c.action_dim

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        for c in self.controllers:
            c.reset(env_ids)

    def get_state(self, env_ids: torch.Tensor | None = None) -> dict[str, Any]:
        return {str(i): c.get_state(env_ids) for i, c in enumerate(self.controllers)}

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor | None = None) -> None:
        for i, c in enumerate(self.controllers):
            c.set_state(state[str(i)], env_ids)
