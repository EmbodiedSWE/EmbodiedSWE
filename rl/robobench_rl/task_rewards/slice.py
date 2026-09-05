"""Shaped potential for the slice scene (knife on a rest, scored food on a board).

    reach   finger pads at the knife handle
    taken   knife lifted off its rest
    carry   the blade's edge centre over the food (food frame xy)
    align   blade plane normal parallel to the cut axis (food x)
    press   for the next uncut plane: edge sample nearest the plane within the plane and
            pressed toward the board (food-frame depth)
    cut     fraction of planes released (the scene's live weld mask)"""
from __future__ import annotations

import torch

from . import TaskReward
from .common import kernel, local_axis, pinch_point, reach_kernel

HANDLE_LOCAL = (-0.09, 0.0, 0.0)  # knife frame: the handle grip point [CALIBRATE; edge runs x 0..0.22]
LIFT_FULL = 0.06


class SliceShapedReward(TaskReward):
    WEIGHTS = {"reach": 0.1, "taken": 0.1, "carry": 0.15, "align": 0.15, "press": 0.2, "cut": 0.3}

    def reset(self, env_ids: torch.Tensor) -> None:
        z = self.scene.knife.data.root_pos_w[:, 2]
        if not hasattr(self, "_z0"):
            self._z0 = z.clone()
            dev = z.device
            self._edge = torch.tensor(self.scene.knife_edge(), device=dev, dtype=torch.float32)  # (E,3)
            b0, b1 = self.scene.manifest["bounds"]
            self._food_c = torch.tensor([(a + b) / 2 for a, b in zip(b0, b1)], device=dev)
            self._food_zmin = float(b0[2])
            self._planes_x = torch.tensor([p for _, p in self.scene.planes], device=dev)
        self._z0[env_ids] = z[env_ids]

    def terms(self) -> dict[str, torch.Tensor]:
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        sc = self.scene
        n = self.env.num_envs
        kp, kq = sc.knife.data.root_pos_w, sc.knife.data.root_quat_w
        pinch = pinch_point(self.env)
        handle = kp + quat_apply(kq, torch.tensor(HANDLE_LOCAL, device=kp.device).expand(n, 3))
        reach = reach_kernel((handle - pinch).norm(dim=-1))
        taken = ((kp[:, 2] - self._z0) / LIFT_FULL).clamp(0.0, 1.0)
        # edge samples into the food frame (anchored on the mid piece, as the scene does)
        ref = sc.pieces[len(sc.pieces) // 2]
        rp, rq = ref.data.root_pos_w, ref.data.root_quat_w
        ref_cent = torch.tensor(sc._cents[len(sc.pieces) // 2], device=kp.device)
        E = self._edge.shape[0]
        edge_w = kp[:, None, :] + quat_apply(kq[:, None, :].expand(n, E, 4).reshape(-1, 4),
                                             self._edge[None].expand(n, E, 3).reshape(-1, 3)).reshape(n, E, 3)
        edge_f = quat_apply_inverse(rq[:, None, :].expand(n, E, 4).reshape(-1, 4),
                                    (edge_w - rp[:, None, :]).reshape(-1, 3)).reshape(n, E, 3) + ref_cent
        centre = edge_f.mean(dim=1)  # (n,3)
        carry = kernel((centre[:, :2] - self._food_c[:2]).norm(dim=-1), 0.10)
        bn_f = quat_apply_inverse(rq, local_axis(kq, 2))  # blade normal in the food frame
        align = bn_f[:, 0].abs().clamp(0.0, 1.0)
        # next uncut plane per env (all cut -> hold press at 1)
        cut = sc.cut  # (n,K) bool
        big = torch.full_like(self._planes_x[None].expand(n, -1), 1e3)
        order = torch.where(cut, big, torch.arange(cut.shape[1], device=kp.device, dtype=torch.float32)[None].expand(n, -1))
        k = order.argmin(dim=1)
        px = self._planes_x[k]  # (n,)
        dx = (edge_f[..., 0] - px[:, None]).abs()  # (n,E)
        e = dx.argmin(dim=1)
        pt = edge_f[torch.arange(n, device=kp.device), e]  # (n,3)
        in_plane = kernel((pt[:, 0] - px).abs(), 0.01)
        depth = ((self._food_c[2] - pt[:, 2]) / (self._food_c[2] - self._food_zmin)).clamp(0.0, 1.0)
        press = in_plane * (0.3 + 0.7 * depth)
        press = torch.where(cut.all(dim=1), torch.ones_like(press), press)
        return dict(reach=reach, taken=taken, carry=carry, align=align, press=press, cut=cut.float().mean(dim=1))
