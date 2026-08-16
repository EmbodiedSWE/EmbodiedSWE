"""NullRobot — the empty embodiment: no assets, no actions.

`BaseEnv` is `scene + robot`, so even to bring a *scene* to life you need a robot in the slot. The
NullRobot fills it with nothing: it spawns no prims and applies no action, so the scene runs under
gravity and its own mechanics (e.g. auto-weld) while no actor interferes. Use it for scene smoke
tests, settling/inspection, and as the "no embodiment" baseline. To poke the scene anyway, drive its
bodies directly through their handles, e.g. `env.scene.legs[k].set_external_force_and_torque(...)`.

Import-light (no isaaclab) — there is nothing to spawn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robobench.core import ROBOTS, BaseRobot

if TYPE_CHECKING:
    import torch


@ROBOTS.register("null")
class NullRobot(BaseRobot):
    """An actor that does nothing. No controller, so `action_dim == 0` and `apply_action` is a no-op
    (both inherited from `BaseRobot`); state I/O are no-ops too."""

    control_modes: tuple[str, ...] = ()

    def __init__(self, cfg: Any = None) -> None:
        super().__init__(cfg)

    def assets(self) -> dict[str, Any]:
        return {}

    def reset(self, _env_ids: torch.Tensor) -> None:
        pass

    # get_state / set_state inherited from BaseRobot: no articulation + no controller -> {} / no-op.

    def describe(self) -> str:
        return "No robot (null embodiment): the scene runs under physics alone; no actor applies actions."
