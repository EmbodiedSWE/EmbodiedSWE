"""Dense potential for the nut-thread scene (scene accessors only, no grader).

    reach      hand at the nominal grasp pose: pads at the nut centre, approaching from above (keypoints;
               the finger-opening yaw is left free — the hex is close enough to round for the pads)
    grasp      pads at the nut AND both fingers closed onto its width (a pinch, not a bat)
    lift       nut raised off its settled rest while it stays at the pads: half a ramp, half a bonus
               at the grader's lifted height (bolt_height above the bolt origin)
    transport  nut xy over the (nearest) bolt axis
    upright    nut screw axis parallel to the bolt axis
    approach   when over the axis and upright: nut lowered to the thread start
    thread     when engaged: depth from the thread start to the seat
Per nut, averaged over an env's nuts."""
from __future__ import annotations

import math

import torch

from . import TaskReward
from .common import (finger_positions, grasp_term_fingers, hand_target_from_axes, kernel, keypoint_distance,
                     local_axis, pinch_point, reach_kernel)

ENGAGE_Z = 0.035  # nut-origin height above the bolt origin at the thread start [CALIBRATE]
NUT_WIDTH = 0.024  # m, M16 across flats — what the pads close onto
GRASP_DZ = 0.008  # m above the nut centre: pinch the UPPER half of the ~13 mm nut (pads at the centre hit the table)
PINCH_OFFSET = 0.103  # m, panda_hand -> pad TIPS (the band centre at 0.099 sits ~1 cm above the tips)
LIFT_FULL = 0.06
LIFT_BONUS_Z = 0.03  # m above the settled rest ~ the grader's `lifted` (origin at bolt_height 0.025 over the bolt origin)
HELD_NEAR = 0.06


class NutThreadDenseReward(TaskReward):
    WEIGHTS = {"reach": 0.1, "grasp": 0.1, "lift": 0.1, "transport": 0.15, "upright": 0.1, "approach": 0.15, "thread": 0.3}

    def reset(self, env_ids: torch.Tensor) -> None:
        z = torch.stack([b.data.root_pos_w[:, 2] for b in self.scene.nuts], dim=1)
        if not hasattr(self, "_z0"):
            self._z0 = z.clone()
            art = self.env.robot.articulation
            self._hand_idx = list(art.data.body_names).index(self.env.robot.EE_BODY)
        self._z0[env_ids] = z[env_ids]

    def grasp_points(self) -> torch.Tensor:
        pos = torch.stack([b.data.root_pos_w for b in self.scene.nuts], dim=1)  # (n,N,3): the nut centre
        return pos + torch.tensor([0.0, 0.0, GRASP_DZ], device=pos.device)

    def hover_target(self) -> torch.Tensor:
        return self.grasp_points()[:, 0] + torch.tensor([0.0, 0.0, 0.10], device=self.device)

    def terms(self) -> dict[str, torch.Tensor]:
        from isaaclab.utils.math import quat_apply_inverse
        # the part settles after reset (measured: the bulb drops 1.9 cm) — track the lowest height seen
        # since reset as the lift baseline, or lift credit only starts above the unsettled spawn height
        self._z0 = torch.minimum(self._z0, torch.stack([b.data.root_pos_w[:, 2] for b in self.scene.nuts], dim=1))

        sc, c = self.scene, self.scene.cfg
        pinch = pinch_point(self.env, PINCH_OFFSET)
        pos = torch.stack([b.data.root_pos_w for b in sc.nuts], dim=1)  # (n,N,3)
        quat = torch.stack([b.data.root_quat_w for b in sc.nuts], dim=1)
        n, N = pos.shape[:2]
        gpt = self.grasp_points()
        d_grasp = (gpt - pinch[:, None, :]).norm(dim=-1)
        # nominal grasp pose: approach straight down onto the nut; opening axis = the hand's current finger
        # axis (+y) flattened (hex ~ round for the pads)
        down = torch.tensor([0.0, 0.0, -1.0], device=pos.device).expand(n * N, 3)
        hand_y = local_axis(self.env.robot.articulation.data.body_quat_w[:, self._hand_idx], 1)
        hand_y = hand_y * torch.tensor([1.0, 1.0, 0.0], device=pos.device)
        hand_y = torch.where(hand_y.norm(dim=-1, keepdim=True) < 1e-3, torch.tensor([0.0, 1.0, 0.0], device=pos.device).expand(n, 3), hand_y)
        t_pos, t_quat = hand_target_from_axes(gpt.reshape(-1, 3), down, hand_y[:, None, :].expand(n, N, 3).reshape(-1, 3),
                                              pinch_offset=PINCH_OFFSET)
        reach = reach_kernel(torch.stack([keypoint_distance(self.env, t_pos.reshape(n, N, 3)[:, k], t_quat.reshape(n, N, 4)[:, k])
                                          for k in range(N)], dim=1))
        fq = finger_positions(self.env)
        grasp = torch.stack([grasp_term_fingers(d_grasp[:, k], fq, NUT_WIDTH) for k in range(N)], dim=1)
        held = (d_grasp <= HELD_NEAR).float()
        dz = pos[..., 2] - self._z0
        lift = held * (0.5 * (dz / LIFT_FULL).clamp(0.0, 1.0) + 0.5 * (dz > LIFT_BONUS_Z).float())
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
        out = {k: v.mean(dim=1) for k, v in dict(reach=reach, grasp=grasp, lift=lift, transport=transport, upright=upright,
                                                 approach=approach, thread=thread).items()}
        out["dist_m"] = d_grasp.mean(dim=1)  # diagnostic only
        return out
