"""One module per task, `tasks/<scene>.py`, holding the task's reward(s) and, when the task has one, its tuned env:

  <Scene>DenseReward   the DENSE condition: a full-task potential proposed ONCE, up front, from the scene's public
                       accessors (part poses, seated(), cfg tolerances) and the hand pose — never iterated, never run
                       against the grader in a loop. `load_task_reward(scene)` returns it.
  <Scene>TunedReward   the TUNED condition: the expert-iterated first-stage reward (measured grasp geometry, per-finger
                       grasp, settled lift baseline), selected by the tuned env's `reward_cls`.
  <Scene>TunedEnv      a RoboBenchEnv subclass baking the tuned env changes (horizon, finger PD, action map, warm-start
                       curriculum, termination). `load_task_env_cls(name)` returns it (yaml `task.task_env`).

Every reward is a POTENTIAL phi(state) in [0, 1]: a weighted sum of stage terms, paid every step (robobench_rl/reward.py),
the same algebra as the grader-progress reward — the conditions differ only in WHO wrote the potential.

NOT suitable for RL as-is: any grasp-weld scene (pc_ram, so101, ...): the weld joint pool is finite per env for the
life of the process, so an env can weld each part only GRASP_POOL (8) times across ALL its training episodes."""
from __future__ import annotations

import importlib

import torch


class TaskReward:
    """Per-task potential. `terms()` returns {name: (num_envs,) in [0,1]}; WEIGHTS sums them."""

    WEIGHTS: dict[str, float] = {}

    def __init__(self, env, weights: dict[str, float] | None = None) -> None:
        self.env = env
        self.scene = env.scene
        self.device = env.device
        if weights:  # debugging aid: `reward.weights: {reach: 1.0}` isolates terms; unknown names fail loudly
            bad = set(weights) - set(self.WEIGHTS)
            if bad:
                raise KeyError(f"{type(self).__name__} has no terms {sorted(bad)}; have {sorted(self.WEIGHTS)}")
            self.WEIGHTS = {k: float(v) for k, v in weights.items()}
        self.reset(torch.arange(env.num_envs, device=env.device))

    def reset(self, env_ids: torch.Tensor) -> None:  # capture per-env references (start heights, ...)
        pass

    def terms(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def hover_target(self) -> torch.Tensor | None:
        """(n, 3) world position above the grasp point for the warm-start curriculum, or None."""
        return None

    def potential(self, terms: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        t = self.terms() if terms is None else terms
        total = sum(self.WEIGHTS.values())
        return sum(w * t[k].clamp(0.0, 1.0) for k, w in self.WEIGHTS.items()) / total

    def describe(self) -> str:
        return f"{type(self).__name__}: " + " · ".join(f"{k} ×{w:g}" for k, w in self.WEIGHTS.items())


DENSE = {"bulb": "BulbDenseReward", "nut_thread": "NutThreadDenseReward", "pen_holder": "PenHolderDenseReward",
         "slice": "SliceDenseReward", "tool_packing": "ToolPackingDenseReward"}
TUNED_ENVS = {"bulb_tuned": ("bulb", "BulbTunedEnv"), "nut_tuned": ("nut_thread", "NutTunedEnv"),
              "slice_tuned": ("slice", "SliceTunedEnv"), "pen_tuned": ("pen_holder", "PenTunedEnv")}


def load_task_reward(scene_name: str) -> type[TaskReward]:
    if scene_name not in DENSE:
        raise KeyError(f"no dense reward for scene {scene_name!r}; have {sorted(DENSE)}")
    return getattr(importlib.import_module(f"{__package__}.{scene_name}"), DENSE[scene_name])


def load_task_env_cls(name: str | None):
    """The env class for yaml `task.task_env` (RoboBenchEnv itself when unset)."""
    if not name:
        from ..vec_env import RoboBenchEnv

        return RoboBenchEnv
    if name not in TUNED_ENVS:
        raise KeyError(f"unknown task_env {name!r}; have {sorted(TUNED_ENVS)}")
    mod, cls = TUNED_ENVS[name]
    return getattr(importlib.import_module(f"{__package__}.{mod}"), cls)
