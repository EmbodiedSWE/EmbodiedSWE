"""Shaped potential for the nut-thread scene (scene accessors only, no grader).

    reach      finger pads at the nut
    lift       nut raised off the table
    transport  nut xy over the (nearest) bolt axis
    upright    nut screw axis parallel to the bolt axis
    approach   when over the axis and upright: nut lowered to the thread start
    thread     when engaged: depth from the thread start to the seat
Per nut, averaged over an env's nuts."""
from __future__ import annotations

import math

import torch

from . import TaskReward
from .common import kernel, local_axis, pinch_point, reach_kernel

ENGAGE_Z = 0.035  # nut-origin height above the bolt origin at the thread start [CALIBRATE, = grader's engage_z]
LIFT_FULL = 0.06


class NutThreadShapedReward(TaskReward):
    WEIGHTS = {"reach": 0.1, "lift": 0.1, "transport": 0.2, "upright": 0.1, "approach": 0.2, "thread": 0.3}

    def reset(self, env_ids: torch.Tensor) -> None:
        z = torch.stack([b.data.root_pos_w[:, 2] for b in self.scene.nuts], dim=1)
        if not hasattr(self, "_z0"):
            self._z0 = z.clone()
        self._z0[env_ids] = z[env_ids]

    def terms(self) -> dict[str, torch.Tensor]:
        from isaaclab.utils.math import quat_apply_inverse

        sc, c = self.scene, self.scene.cfg
        pinch = pinch_point(self.env)
        pos = torch.stack([b.data.root_pos_w for b in sc.nuts], dim=1)  # (n,N,3)
        quat = torch.stack([b.data.root_quat_w for b in sc.nuts], dim=1)
        n, N = pos.shape[:2]
        reach = reach_kernel((pos - pinch[:, None, :]).norm(dim=-1))
        lift = ((pos[..., 2] - self._z0) / LIFT_FULL).clamp(0.0, 1.0)
        bp = torch.stack([b.data.root_pos_w for b in sc.bolts], dim=1)  # (n,B,3)
        bq = torch.stack([b.data.root_quat_w for b in sc.bolts], dim=1)
        b_up = torch.stack([local_axis(b.data.root_quat_w, 2) for b in sc.bolts], dim=1)  # (n,B,3)
        xy, depth, cos = [], [], []
        for k in range(N):
            rel = quat_apply_inverse(bq, pos[:, k, None, :] - bp)  # (n,B,3)
            d_xy, near = rel[..., :2].norm(dim=-1).min(dim=-1)
            xy.append(d_xy)
            depth.append(torch.gather(rel[..., 2], 1, near[:, None]).squeeze(1))
            nut_up = local_axis(quat[:, k], 2)
            near_up = torch.gather(b_up, 1, near[:, None, None].expand(-1, 1, 3)).squeeze(1)
            cos.append((nut_up * near_up).sum(-1))
        xy, depth, cos = torch.stack(xy, 1), torch.stack(depth, 1), torch.stack(cos, 1)
        transport = kernel(xy, 0.10)
        upright = cos.clamp(0.0, 1.0)
        aligned = (xy <= 2 * c.align_xy) & (cos >= math.cos(math.radians(2 * c.align_axis_deg)))
        approach = aligned.float() * kernel((depth - ENGAGE_Z).clamp(min=0.0), 0.05)
        engaged = (xy <= c.align_xy) & (depth <= ENGAGE_Z)
        thread = engaged.float() * ((ENGAGE_Z - depth) / (ENGAGE_Z - c.seat_z)).clamp(0.0, 1.0)
        return {k: v.mean(dim=1) for k, v in dict(reach=reach, lift=lift, transport=transport, upright=upright,
                                                  approach=approach, thread=thread).items()}
