"""Reward = a potential in [0, 1] paid every step (Isaac Lab style) + a success bonus:

    r_t = (P_t + bonus * success_t) * step_dt

  progress  P = the grader's weighted rubric progress — PRIVILEGED (agents never see the grader)
  shaped    P = the task's hand-designed potential (task_rewards/<scene>.py)

Success and the logged stage curve always come from the grader. Graders are single-trajectory objects
(setup() captures start poses once, "once" milestones keep a best-ever), so `reset(env_ids)` re-runs
setup() and restores the un-reset envs' rows from a snapshot of the grader's per-env tensors."""
from __future__ import annotations

import importlib

import torch

from .task_rewards import load_task_reward


def load_grader_cls(preset: str, scene_name: str):
    """`assembly.bulb.franka.osc` -> robobench.suites.assembly.grader.GRADERS['bulb']; graders are owned by
    the benchmark team and a scene without one cannot be trained or graded here yet."""
    suite = preset.split(".")[0]
    try:
        mod = importlib.import_module(f"robobench.suites.{suite}.grader")
    except ModuleNotFoundError as e:
        raise KeyError(f"suite {suite!r} has no grader package yet (needed for scene {scene_name!r})") from e
    graders = getattr(mod, "GRADERS", {})
    if scene_name not in graders:
        raise KeyError(f"no grader for scene {scene_name!r} in suite {suite!r} yet: have {sorted(graders)}")
    return graders[scene_name]


class GraderReward:
    def __init__(self, env, grader_cls, scene_name: str, *, mode: str = "progress", success_bonus: float = 1.0,
                 weights: dict | None = None) -> None:
        if mode not in ("progress", "shaped"):
            raise ValueError(f"reward mode must be progress|shaped, got {mode!r}")
        self.env, self.mode, self.bonus = env, mode, success_bonus
        self.step_dt = env.dt * env.robot.control_period
        self.device = env.device
        self.grader = grader_cls(env)  # runs setup() on the just-reset env
        self.task = load_task_reward(scene_name)(env, weights) if mode == "shaped" else None

    def measure(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        vals = self.grader.measure()
        return self.grader.progress(vals).to(self.device), {k: v.to(self.device) for k, v in vals.items()}

    def success(self) -> torch.Tensor:
        return self.grader._per_env(self.grader.check_success(), "check_success").bool().to(self.device)

    def compute(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        p, stages = self.measure()
        succ = self.success()
        stages["progress"] = p
        pot = p
        if self.task is not None:
            t = self.task.terms()
            pot = self.task.potential(t).to(self.device)
            stages.update({f"shaped/{k}": v.to(self.device) for k, v in t.items()})
            stages["shaped/potential"] = pot
        return (pot + self.bonus * succ.float()) * self.step_dt, succ, stages

    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor) -> None:
        g = self.grader
        n = g.num_envs
        ids = env_ids.to("cpu")
        if len(ids) >= n:
            g.setup()
        else:
            snap = {k: v.clone() for k, v in vars(g).items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n}
            g.setup()  # recaptures ALL envs from live state ...
            keep = torch.ones(n, dtype=torch.bool)
            keep[ids] = False
            for k, old in snap.items():  # ... so put the un-reset envs' rows back
                new = getattr(g, k, None)
                if torch.is_tensor(new) and new.shape == old.shape:
                    km = keep.to(new.device)
                    new[km] = old.to(new.device)[km]
        for name in g._best:
            g._best[name][ids] = 0.0
        if self.task is not None:
            self.task.reset(env_ids.to(self.device))

    def describe(self) -> str:
        s = f"{self.mode}: grader {self.grader.describe()}"
        if self.task is not None:
            s += f"\n        shaped {self.task.describe()}"
        return s
