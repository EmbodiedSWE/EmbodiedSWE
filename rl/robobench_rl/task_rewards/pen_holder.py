"""Shaped potential for the pen-holder scene.

    reach    finger pads at the nearest not-yet-inserted present pen
    lift     that pen raised off the table
    carry    that pen's bottom end over the holder axis (holder frame)
    tip_up   that pen's axis along the holder axis, tip up
    insert   pen bottom lowered below the rim toward depth_min (when near the axis)
    all_in   fraction of present pens counted (the scene's own rubric term)
"Nearest uninserted pen" focuses the dense terms on one pen at a time; `all_in` keeps credit
for the ones already done."""
from __future__ import annotations

import torch

from . import TaskReward
from .common import kernel, pinch_point, reach_kernel

LIFT_FULL = 0.10


class PenHolderShapedReward(TaskReward):
    WEIGHTS = {"reach": 0.1, "lift": 0.1, "carry": 0.15, "tip_up": 0.1, "insert": 0.25, "all_in": 0.3}

    def reset(self, env_ids: torch.Tensor) -> None:
        pos, _, _ = self.scene._pen_tensors()
        if not hasattr(self, "_z0"):
            self._z0 = pos[..., 2].clone()
        self._z0[env_ids] = pos[env_ids, :, 2]

    def terms(self) -> dict[str, torch.Tensor]:
        sc, c = self.scene, self.scene.cfg
        pos, _quat, _vel = sc._pen_tensors()  # (n,P,3)
        b_loc, t_loc = sc._pen_ends_local()  # holder frame (n,P,3)
        present = sc.present
        counted = sc.counted() & present
        pinch = pinch_point(self.env)
        n, P = pos.shape[:2]
        # focus: the nearest present, not-yet-counted pen (fallback: any present pen)
        d = (pos - pinch[:, None, :]).norm(dim=-1)
        d_focus = torch.where(present & ~counted, d, torch.full_like(d, 1e3))
        d_focus = torch.where(present.any(dim=1, keepdim=True) & (d_focus >= 1e3).all(dim=1, keepdim=True),
                              torch.where(present, d, torch.full_like(d, 1e3)), d_focus)
        k = d_focus.argmin(dim=1)  # (n,)
        idx = torch.arange(n, device=pos.device)
        done_all = counted.sum(1) >= present.sum(1)
        reach = reach_kernel(d[idx, k])
        lift = ((pos[idx, k, 2] - self._z0[idx, k]) / LIFT_FULL).clamp(0.0, 1.0)
        bxy = b_loc[idx, k, :2].norm(dim=-1)
        carry = kernel(bxy, 0.10)
        axis = t_loc[idx, k] - b_loc[idx, k]
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        tip_up = axis[:, 2].clamp(0.0, 1.0)
        depth = c.holder_h / 2 - b_loc[idx, k, 2]  # below the rim
        insert = (bxy <= 2 * c.xy_tol).float() * (depth / c.depth_min).clamp(0.0, 1.0)
        # once every present pen is in, the focused pen is a done one: hold the dense terms at 1
        one = torch.ones_like(reach)
        reach, lift, carry, tip_up, insert = [torch.where(done_all, one, t) for t in (reach, lift, carry, tip_up, insert)]
        all_in = counted.sum(1).float() / present.sum(1).clamp(min=1).float()
        return dict(reach=reach, lift=lift, carry=carry, tip_up=tip_up, insert=insert, all_in=all_in)
