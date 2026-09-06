"""Shaped potential for the tool-packing scene.

    reach     finger pads at the nearest not-yet-stowed item
    lift      that item raised off the table
    carry     that item's origin over its assigned tray centre (drawer body frame)
    stowed    fraction of items inside their trays (scene's `_item_in_tray`)
    drawers   mean drawer shut-ness (1 = shut, 0 = fully out)
    doors     mean door shut-ness"""
from __future__ import annotations

import math

import torch

from . import TaskReward
from .common import kernel, pinch_point, reach_kernel

LIFT_FULL = 0.08


class ToolPackingShapedReward(TaskReward):
    WEIGHTS = {"reach": 0.1, "lift": 0.1, "carry": 0.2, "stowed": 0.3, "drawers": 0.2, "doors": 0.1}

    def reset(self, env_ids: torch.Tensor) -> None:
        pos = torch.stack([b.data.root_pos_w for b in self.scene.items.values()], dim=1)
        if not hasattr(self, "_z0"):
            self._z0 = pos[..., 2].clone()
        self._z0[env_ids] = pos[env_ids, :, 2]

    def terms(self) -> dict[str, torch.Tensor]:
        from isaaclab.utils.math import quat_apply_inverse
        # the part settles after reset (measured: the bulb drops 1.9 cm) — track the lowest height seen
        # since reset as the lift baseline, or lift credit only starts above the unsettled spawn height
        self._z0 = torch.minimum(self._z0, torch.stack([b.data.root_pos_w for b in self.scene.items.values()], dim=1)[..., 2])

        sc, c = self.scene, self.scene.cfg
        pos = torch.stack([b.data.root_pos_w for b in sc.items.values()], dim=1)  # (n,I,3)
        n, I = pos.shape[:2]
        in_tray = sc._item_in_tray()  # (n,I)
        pinch = pinch_point(self.env)
        d = (pos - pinch[:, None, :]).norm(dim=-1)
        d_focus = torch.where(~in_tray, d, torch.full_like(d, 1e3))
        k = d_focus.argmin(dim=1)
        idx = torch.arange(n, device=pos.device)
        done_all = in_tray.all(dim=1)
        # tray-centre offset of the focused item, in its drawer body frame
        bpos, bquat = sc.box.data.body_pos_w, sc.box.data.body_quat_w
        offs = []
        for i in range(I):
            b = sc._drawer_b[sc._assigned[i]]
            loc = quat_apply_inverse(bquat[:, b], pos[:, i] - bpos[:, b]) - sc._tray_c
            offs.append(loc)
        off = torch.stack(offs, dim=1)[idx, k]  # (n,3)
        reach = reach_kernel(d[idx, k])
        lift = ((pos[idx, k, 2] - self._z0[idx, k]) / LIFT_FULL).clamp(0.0, 1.0)
        carry = kernel(off.norm(dim=-1), 0.15)
        one = torch.ones_like(reach)
        reach, lift, carry = [torch.where(done_all, one, t) for t in (reach, lift, carry)]
        stowed = in_tray.float().mean(dim=1)
        travel = float(c.drawer_travel or 0.2)
        drawers = (1.0 - sc.drawer_pos().abs() / travel).clamp(0.0, 1.0).mean(dim=1)
        door_max = math.radians(90.0)
        doors = (1.0 - sc.door_pos().abs() / door_max).clamp(0.0, 1.0).mean(dim=1) if sc.door_pos().shape[1] else one
        return dict(reach=reach, lift=lift, carry=carry, stowed=stowed, drawers=drawers, doors=doors)
