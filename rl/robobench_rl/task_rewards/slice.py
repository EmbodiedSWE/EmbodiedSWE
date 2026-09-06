"""Shaped potential for the slice scene (knife on a rest, scored food on a board).

    reach   hand at the nominal grasp POSE: pads at the handle centre, approaching from above, fingers
            opening across the plate (keypoint distance, so orientation counts — a side-on pinch
            of the handle butt with the hand lying on the island was the measured local optimum)
    grasp   pads at the handle AND fingers closed to the handle width (a pinch, not a bat)
    taken   knife lifted off its rest while it stays at the pads (held, whatever the fingers read):
            gating lift on the aperture window made holding a pinch strictly safer than trying to
            lift (a slip drops the grasp term) — measured: grasp 0.76, lift 0.00 after 300 iterations
    carry   the blade's edge centre over the food (food frame xy)
    align   blade plane normal parallel to the cut axis (food x)
    press   for the next uncut plane: edge sample nearest the plane within the plane and
            pressed toward the board (food-frame depth)
    cut     fraction of planes released (the scene's live weld mask)"""
from __future__ import annotations

import torch

from . import TaskReward
from .common import (finger_positions, grasp_term_fingers, hand_target_from_axes, kernel, keypoint_distance,
                     local_axis, pinch_point, reach_kernel)

# Measured from knife22r/piece_0.usd: a flat 4 mm plate; the handle spans x [-0.08, 0], y [0, 0.05]
# in the knife frame (blade edge runs toward -y, tip at x 0.22). Grip the handle's centre; the
# pinch closes across the plate thickness.
HANDLE_LOCAL = (-0.05, 0.03, 0.0)
HANDLE_WIDTH = 0.005  # m pad-to-pad when stalled on the 4 mm plate (finger-joint sum ~ pad gap)
LIFT_FULL = 0.04  # m of lift that counts as fully taken (the rest notches are ~1-2 cm deep)
HELD_NEAR = 0.06  # m: knife handle point within this of the pads = "held" for the lift term (loose on purpose)


class SliceShapedReward(TaskReward):
    WEIGHTS = {"reach": 0.1, "grasp": 0.1, "taken": 0.1, "carry": 0.1, "align": 0.1, "press": 0.2, "cut": 0.3}

    def reset(self, env_ids: torch.Tensor) -> None:
        z = self.scene.knife.data.root_pos_w[:, 2]
        if not hasattr(self, "_z0"):
            self._z0 = z.clone()
            art = self.env.robot.articulation
            self._hand_idx = list(art.data.body_names).index(self.env.robot.EE_BODY)
            dev = z.device
            self._edge = torch.tensor(self.scene.knife_edge(), device=dev, dtype=torch.float32)  # (E,3)
            b0, b1 = self.scene.manifest["bounds"]
            self._food_c = torch.tensor([(a + b) / 2 for a, b in zip(b0, b1)], device=dev)
            self._food_zmin = float(b0[2])
            self._planes_x = torch.tensor([p for _, p in self.scene.planes], device=dev)
        self._z0[env_ids] = z[env_ids]

    def terms(self) -> dict[str, torch.Tensor]:
        from isaaclab.utils.math import quat_apply, quat_apply_inverse
        # the part settles after reset (measured: the bulb drops 1.9 cm) — track the lowest height seen
        # since reset as the lift baseline, or lift credit only starts above the unsettled spawn height
        self._z0 = torch.minimum(self._z0, self.scene.knife.data.root_pos_w[:, 2])

        sc = self.scene
        n = self.env.num_envs
        kp, kq = sc.knife.data.root_pos_w, sc.knife.data.root_quat_w
        pinch = pinch_point(self.env)
        handle = kp + quat_apply(kq, torch.tensor(HANDLE_LOCAL, device=kp.device).expand(n, 3))
        d_handle = (handle - pinch).norm(dim=-1)
        # nominal grasp pose: approach straight down, fingers opening along the plate normal (knife +z)
        down = torch.tensor([0.0, 0.0, -1.0], device=kp.device).expand(n, 3)
        plate_n = local_axis(kq, 2)
        hand_y = local_axis(self.env.robot.articulation.data.body_quat_w[:, self._hand_idx], 1)  # finger-opening axis
        plate_n = torch.where((plate_n * hand_y).sum(-1, keepdim=True) < 0, -plate_n, plate_n)  # nearer of the 2 grasps
        t_pos, t_quat = hand_target_from_axes(handle, down, plate_n)
        reach = reach_kernel(keypoint_distance(self.env, t_pos, t_quat))
        grasp = grasp_term_fingers(d_handle, finger_positions(self.env), HANDLE_WIDTH)
        held = (d_handle <= HELD_NEAR).float()
        taken = held * ((kp[:, 2] - self._z0) / LIFT_FULL).clamp(0.0, 1.0)
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
        return dict(reach=reach, grasp=grasp, taken=taken, carry=carry, align=align, press=press,
                    cut=cut.float().mean(dim=1))
