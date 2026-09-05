"""Shaped potential for the bulb-threading scene (public accessors only).

Stages a practitioner would write for pick -> carry -> upright -> engage -> thread:
    reach      finger pads close to the bulb's glass belly
    lift       bulb raised off its start height
    transport  bulb xy over the socket axis
    upright    bulb screw axis parallel to the socket axis (cap down)
    approach   when over the axis and upright: bulb lowered to the bore mouth
    thread     when engaged in the bore: depth from free-rest to seat (the scene's own glow band)
All per-bulb, averaged over an env's bulbs. Success itself is left to the grader/bonus."""
from __future__ import annotations

import math

import torch

from . import TaskReward
from .common import kernel, local_axis, pinch_point, reach_kernel

GRASP_Z = 0.055  # m above the cap bottom (bulb origin) — the glass belly a gripper pinches; calibrate
LIFT_FULL = 0.08  # m of lift that counts as fully lifted


class BulbShapedReward(TaskReward):
    WEIGHTS = {"reach": 0.1, "lift": 0.1, "transport": 0.2, "upright": 0.1, "approach": 0.2, "thread": 0.3}

    def reset(self, env_ids: torch.Tensor) -> None:
        z = torch.stack([b.data.root_pos_w[:, 2] for b in self.scene.bulbs], dim=1)  # (n, N)
        if not hasattr(self, "_z0"):
            self._z0 = z.clone()
        self._z0[env_ids] = z[env_ids]

    def _offsets(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per bulb vs its NEAREST socket: xy distance (n,N), depth z in socket frame (n,N), axis cos (n,N)."""
        from isaaclab.utils.math import quat_apply_inverse

        sp = torch.stack([s.data.root_pos_w for s in self.scene.sockets], dim=1)  # (n,B,3)
        sq = torch.stack([s.data.root_quat_w for s in self.scene.sockets], dim=1)  # (n,B,4)
        s_up = torch.stack([local_axis(s.data.root_quat_w, 2) for s in self.scene.sockets], dim=1)  # (n,B,3)
        xy, depth, cos = [], [], []
        for b in self.scene.bulbs:
            rel = quat_apply_inverse(sq, b.data.root_pos_w[:, None, :] - sp)  # (n,B,3)
            d_xy, near = rel[..., :2].norm(dim=-1).min(dim=-1)  # (n,)
            xy.append(d_xy)
            depth.append(torch.gather(rel[..., 2], 1, near[:, None]).squeeze(1))
            b_up = local_axis(b.data.root_quat_w, 2)  # (n,3)
            near_up = torch.gather(s_up, 1, near[:, None, None].expand(-1, 1, 3)).squeeze(1)
            cos.append((b_up * near_up).sum(-1))
        return torch.stack(xy, 1), torch.stack(depth, 1), torch.stack(cos, 1)

    def terms(self) -> dict[str, torch.Tensor]:
        from isaaclab.utils.math import quat_apply

        c = self.scene.cfg
        pinch = pinch_point(self.env)  # (n,3)
        pos = torch.stack([b.data.root_pos_w for b in self.scene.bulbs], dim=1)  # (n,N,3)
        quat = torch.stack([b.data.root_quat_w for b in self.scene.bulbs], dim=1)  # (n,N,4)
        n, N = pos.shape[:2]
        gz = torch.tensor([0.0, 0.0, GRASP_Z], device=pos.device).expand(n * N, 3)
        grasp_pt = pos + quat_apply(quat.reshape(-1, 4), gz).reshape(n, N, 3)
        reach = reach_kernel((grasp_pt - pinch[:, None, :]).norm(dim=-1))
        lift = ((pos[..., 2] - self._z0) / LIFT_FULL).clamp(0.0, 1.0)
        xy, depth, cos = self._offsets()
        transport = kernel(xy, 0.10)
        upright = cos.clamp(0.0, 1.0)
        aligned = (xy <= 2 * c.align_xy) & (cos >= math.cos(math.radians(2 * c.align_axis_deg)))
        approach = aligned.float() * kernel((depth - c.light_start_z).clamp(min=0.0), 0.05)
        engaged = (xy <= c.align_xy) & (depth <= c.socket_opening_z)
        thread = engaged.float() * ((c.light_start_z - depth) / (c.light_start_z - c.seat_z)).clamp(0.0, 1.0)
        return {k: v.mean(dim=1) for k, v in dict(reach=reach, lift=lift, transport=transport, upright=upright,
                                                  approach=approach, thread=thread).items()}
