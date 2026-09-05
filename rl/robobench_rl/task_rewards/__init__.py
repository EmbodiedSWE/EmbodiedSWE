"""Hand-designed shaped rewards, one module per task in the RL test subset.

Each task reward is a POTENTIAL phi(state) in [0, 1]: a weighted sum of dense stage terms a
practitioner would write from the scene's public accessors (part poses, seated(), cfg tolerances)
and the robot's hand pose — never from the private grader. The potential is paid every step
(see robobench_rl/reward.py), the same algebra as the grader-progress reward, so the two
conditions differ only in WHO wrote the potential: the task author's rubric, or a dense shaping
written for RL.

Add a task: subclass TaskReward in a new module, register it in `load_task_reward` under the
scene's registry name (EnvCfg.scene), e.g. "bulb".

NOT suitable for RL as-is: any grasp-weld scene (pc_ram, so101, ...). The weld contract's joint
pool is finite per env for the life of the process (PhysX latches joint frames on first enable),
so an env can weld each part only GRASP_POOL (8) times across ALL its training episodes."""
from __future__ import annotations

import torch


class TaskReward:
    """Per-task potential. `terms()` returns {name: (num_envs,) in [0,1]}; WEIGHTS sums them."""

    WEIGHTS: dict[str, float] = {}

    def __init__(self, env) -> None:
        self.env = env
        self.scene = env.scene
        self.device = env.device
        self.reset(torch.arange(env.num_envs, device=env.device))

    def reset(self, env_ids: torch.Tensor) -> None:  # capture per-env references (start heights, ...)
        pass

    def terms(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def potential(self, terms: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        t = self.terms() if terms is None else terms
        total = sum(self.WEIGHTS.values())
        return sum(w * t[k].clamp(0.0, 1.0) for k, w in self.WEIGHTS.items()) / total

    def describe(self) -> str:
        return f"{type(self).__name__}: " + " · ".join(f"{k} ×{w:g}" for k, w in self.WEIGHTS.items())


def load_task_reward(scene_name: str) -> type[TaskReward]:
    from .bulb import BulbShapedReward
    from .nut_thread import NutThreadShapedReward
    from .pen_holder import PenHolderShapedReward
    from .slice import SliceShapedReward
    from .tool_packing import ToolPackingShapedReward

    table: dict[str, type[TaskReward]] = {"bulb": BulbShapedReward, "nut_thread": NutThreadShapedReward,
                                          "pen_holder": PenHolderShapedReward, "tool_packing": ToolPackingShapedReward,
                                          "slice": SliceShapedReward}
    if scene_name not in table:
        raise KeyError(f"no shaped reward for scene {scene_name!r}; have {sorted(table)}")
    return table[scene_name]
