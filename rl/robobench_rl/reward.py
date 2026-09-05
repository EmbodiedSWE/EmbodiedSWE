"""Reward = delta of a potential + success bonus. Two reward files select the potential's author:

  progress  P = the grader's weighted rubric progress (the task author's decomposition).
            "once" milestones are monotone; live stages dip when state is lost.
            PRIVILEGED — the coding agent never sees the grader. State this in any report.
  shaped    P = a hand-designed dense potential written for RL from the scene's public accessors
            (robobench_rl/task_rewards/<scene>.py), one per task in the test subset.

Two forms of turning the potential into a per-step reward (`reward.form`):
  delta   r_t = scale * (P_t - P_{t-1}) + bonus * success_t   — return = potential gained + bonus;
          unbiased under success termination (the default)
  level   r_t = scale * P_t * step_dt + bonus * success_t     — Isaac Lab style dense reward; the
          env then does NOT terminate on success (a raw level reward would punish finishing early)
Both subtract penalty * |a_t - a_{t-1}|^2 (an action-RATE penalty: jitter, not motion; off by default). Success (termination/bonus) and the logged stage curve always come
from the grader, in both modes, so training logs are comparable and the terminal signal is the same.

Graders are single-trajectory objects (setup() captures start poses for all envs once; "once"
milestones keep a best-ever per env). Training needs per-env auto-reset, so `reset(env_ids)`
re-runs setup() and restores the rows of every un-reset env from a snapshot of the grader's
per-env tensors, then zeroes the reset envs' milestones. Generic over any BaseGrader whose
per-env state lives in tensor attributes with shape[0] == num_envs (all shipped graders)."""
from __future__ import annotations

import importlib

import torch

from .task_rewards import load_task_reward


def load_grader_cls(preset: str, scene_name: str):
    """`assembly.bulb.franka.osc` -> robobench.suites.assembly.grader.GRADERS['bulb']. Graders are
    owned by the benchmark team; a scene without one cannot be trained or graded here yet."""
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
    def __init__(self, env, grader_cls, scene_name: str, *, mode: str = "progress", form: str = "delta",
                 progress_scale: float = 1.0, success_bonus: float = 1.0, action_penalty: float = 0.0) -> None:
        if mode not in ("progress", "shaped"):
            raise ValueError(f"reward mode must be progress|shaped, got {mode!r}")
        if form not in ("delta", "level"):
            raise ValueError(f"reward form must be delta|level, got {form!r}")
        self.env, self.mode, self.form = env, mode, form
        self.step_dt = env.dt * env.robot.control_period
        self.scale, self.bonus, self.penalty = progress_scale, success_bonus, action_penalty
        self.grader = grader_cls(env)  # runs setup() on the just-reset env
        self.device = env.device
        self.task = load_task_reward(scene_name)(env) if mode == "shaped" else None
        self.prev = self._potential()
        self.prev_action = torch.zeros(env.num_envs, env.robot.action_dim, device=self.device)
        self.stage_names = [n for n, _, _ in self.grader._stages]

    def _potential(self) -> torch.Tensor:
        if self.task is not None:
            return self.task.potential().to(self.device)
        return self.grader.progress().to(self.device)

    def measure(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        vals = self.grader.measure()
        p = self.grader.progress(vals).to(self.device)
        return p, {k: v.to(self.device) for k, v in vals.items()}

    def success(self) -> torch.Tensor:
        return self.grader._per_env(self.grader.check_success(), "check_success").bool().to(self.device)

    def compute(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        p, stages = self.measure()  # grader rubric: always measured (logs + success), reward in `progress` mode
        succ = self.success()
        stages["progress"] = p
        if self.task is not None:
            t = self.task.terms()
            pot = self.task.potential(t).to(self.device)
            stages.update({f"shaped/{k}": v.to(self.device) for k, v in t.items()})
            stages["shaped/potential"] = pot
        else:
            pot = p
        if self.form == "delta":
            r = self.scale * (pot - self.prev) + self.bonus * succ.float()
        else:
            r = self.scale * pot * self.step_dt + self.bonus * succ.float()
        if self.penalty > 0:
            r = r - self.penalty * (action - self.prev_action).pow(2).sum(dim=-1)
        self.prev_action = action.clone()
        self.prev = pot
        return r, succ, stages

    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor) -> None:
        g = self.grader
        n = g.num_envs
        ids = env_ids.to("cpu")
        if len(ids) >= n:
            g.setup()
        else:
            snap = {k: v.clone() for k, v in vars(g).items()
                    if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n}
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
        self.prev_action[env_ids.to(self.device)] = 0.0
        self.prev = self._potential()  # baseline for the delta after the reset

    def describe(self) -> str:
        s = f"{self.mode} ({self.form}): grader {self.grader.describe()}"
        if self.task is not None:
            s += f"\n        shaped {self.task.describe()}"
        return s
