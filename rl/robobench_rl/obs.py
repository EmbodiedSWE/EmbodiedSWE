"""The observation rule — one rule for every task, no per-task authoring.

obs = flatten(env.get_states())  in env-local coordinates
    + end-effector pose/velocity  (one universal addition: saves the policy from learning FK)
    + last action                  (replaces the dropped controller setpoints)

`get_states()` is the full restorable state by definition, so nothing task-relevant is missing.
Three fixes when flattening: (1) root/pose tensors carry the env origin — subtract it, or every
vectorized env sees different numbers for the same situation; (2) drop the actuator setpoints and
controller state — they are the policy's own previous command; (3) sorted-key traversal so the
layout is deterministic. Value ranges (metres, radians, rad/s) are left to PPO's running
observation normalization."""
from __future__ import annotations

from typing import Any, Iterator

import torch

ROOT_STATE_W = 13  # Isaac root state: pos3 quat4 linvel3 angvel3
POSE_W = 7  # pos3 quat4


def _walk(d: dict, prefix: str = "") -> Iterator[tuple[str, Any]]:
    for k in sorted(d):
        v = d[k]
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from _walk(v, name + ".")
        else:
            yield name, v


class StateObs:
    def __init__(self, env, *, drop_keys=("joint_pos_target", "joint_effort_target", "controller"),
                 env_local: bool = True, ee_pose: bool = True, last_action: bool = True) -> None:
        self.env = env
        self.drop = tuple(drop_keys)
        self.env_local, self.with_ee, self.with_last_action = env_local, ee_pose, last_action
        self.origins = env.iscene.env_origins  # (n, 3)
        self._ee_idx = self._resolve_ee()
        self.layout: list[tuple[str, int, int]] = []
        self.dim = 0
        self.compute(torch.zeros(env.num_envs, env.robot.action_dim, device=env.device))  # builds layout

    def _resolve_ee(self) -> int | None:
        robot = self.env.robot
        art = getattr(robot, "articulation", None)
        if not self.with_ee or art is None:
            return None
        names = list(art.data.body_names)
        cands = []
        if getattr(robot, "EE_BODY", None):
            cands.append(robot.EE_BODY)
        cands += list(getattr(robot, "EE_BODIES", ()) or ())
        for c in cands:
            if c in names:
                return names.index(c)
        return None

    def _dropped(self, name: str) -> bool:
        parts = name.split(".")
        return any(p in self.drop for p in parts) or name in self.drop

    def _localize(self, v: torch.Tensor) -> torch.Tensor:
        if not self.env_local or not v.is_floating_point() or v.dim() < 2:
            return v
        if v.shape[-1] in (ROOT_STATE_W, POSE_W) and v.shape[0] == self.env.num_envs:
            v = v.clone()
            o = self.origins.to(v.device).reshape(v.shape[0], *([1] * (v.dim() - 2)), 3)
            v[..., 0:3] -= o
        return v

    def compute(self, last_action: torch.Tensor) -> torch.Tensor:
        env = self.env
        n = env.num_envs
        parts: list[tuple[str, torch.Tensor]] = []
        for name, v in _walk(env.get_states()):
            if self._dropped(name) or not torch.is_tensor(v):
                continue
            if v.dim() == 0 or v.shape[0] != n:
                continue
            parts.append((name, self._localize(v).reshape(n, -1).float()))
        if self._ee_idx is not None:
            d = env.robot.articulation.data
            i = self._ee_idx
            ee_pos = d.body_pos_w[:, i] - self.origins.to(d.body_pos_w.device)
            parts.append(("ee.pos", ee_pos))
            parts.append(("ee.quat", d.body_quat_w[:, i]))
            parts.append(("ee.lin_vel", d.body_lin_vel_w[:, i]))
            parts.append(("ee.ang_vel", d.body_ang_vel_w[:, i]))
        if self.with_last_action:
            parts.append(("last_action", last_action.reshape(n, -1).float()))
        if not self.layout:
            start = 0
            for name, t in parts:
                self.layout.append((name, start, start + t.shape[1]))
                start += t.shape[1]
            self.dim = start
        obs = torch.cat([t.to(env.device) for _, t in parts], dim=1)
        return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def describe(self) -> str:
        rows = [f"  {name:32s} [{a:4d}:{b:4d})  {b - a}" for name, a, b in self.layout]
        return f"obs dim {self.dim}\n" + "\n".join(rows)
