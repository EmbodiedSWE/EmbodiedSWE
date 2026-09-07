"""nut_thread: DENSE reward = the up-front full-task proposal (never iterated); TUNED reward + env = the expert-iterated
first-stage recipe (measured grasp geometry, per-finger grasp, settled lift baseline, curriculum, finger PD)."""
from __future__ import annotations

import math

import torch

from . import TaskReward
from .common import (finger_positions, grasp_term_fingers, hand_target_from_axes, kernel, keypoint_distance,
                     local_axis, pinch_point, reach_kernel)
from ..vec_env import RoboBenchEnv


class NutThreadDenseReward(TaskReward):
    """Dense potential for the nut-thread scene (scene accessors only, no grader).

    reach      finger pads at the nut
    lift       nut raised off the table
    transport  nut xy over the (nearest) bolt axis
    upright    nut screw axis parallel to the bolt axis
    approach   when over the axis and upright: nut lowered to the thread start
    thread     when engaged: depth from the thread start to the seat
    Per nut, averaged over an env's nuts."""

    ENGAGE_Z = 0.035  # nut-origin height above the bolt origin at the thread start [CALIBRATE, = grader's engage_z]
    LIFT_FULL = 0.06

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
        lift = ((pos[..., 2] - self._z0) / self.LIFT_FULL).clamp(0.0, 1.0)
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
        approach = aligned.float() * kernel((depth - self.ENGAGE_Z).clamp(min=0.0), 0.05)
        engaged = (xy <= c.align_xy) & (depth <= self.ENGAGE_Z)
        thread = engaged.float() * ((self.ENGAGE_Z - depth) / (self.ENGAGE_Z - c.seat_z)).clamp(0.0, 1.0)
        return {k: v.mean(dim=1) for k, v in dict(reach=reach, lift=lift, transport=transport, upright=upright,
                                                  approach=approach, thread=thread).items()}


# ----- TUNED condition: expert-iterated first-stage reward + env (task_env: <scene>_tuned) -----


class NutThreadTunedReward(TaskReward):
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

    ENGAGE_Z = 0.035  # nut-origin height above the bolt origin at the thread start [CALIBRATE]
    NUT_WIDTH = 0.024  # m, M16 across flats — what the pads close onto
    GRASP_DZ = 0.008  # m above the nut centre: pinch the UPPER half of the ~13 mm nut (pads at the centre hit the table)
    PINCH_OFFSET = 0.103  # m, panda_hand -> pad TIPS (the band centre at 0.099 sits ~1 cm above the tips)
    LIFT_FULL = 0.06
    LIFT_BONUS_Z = 0.03  # m above the settled rest ~ the grader's `lifted` (origin at bolt_height 0.025 over the bolt origin)
    HELD_NEAR = 0.06

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
        return pos + torch.tensor([0.0, 0.0, self.GRASP_DZ], device=pos.device)

    def hover_target(self) -> torch.Tensor:
        return self.grasp_points()[:, 0] + torch.tensor([0.0, 0.0, 0.10], device=self.device)

    def terms(self) -> dict[str, torch.Tensor]:
        from isaaclab.utils.math import quat_apply_inverse
        # the part settles after reset (measured: the bulb drops 1.9 cm) — track the lowest height seen
        # since reset as the lift baseline, or lift credit only starts above the unsettled spawn height
        self._z0 = torch.minimum(self._z0, torch.stack([b.data.root_pos_w[:, 2] for b in self.scene.nuts], dim=1))

        sc, c = self.scene, self.scene.cfg
        pinch = pinch_point(self.env, self.PINCH_OFFSET)
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
                                              pinch_offset=self.PINCH_OFFSET)
        reach = reach_kernel(torch.stack([keypoint_distance(self.env, t_pos.reshape(n, N, 3)[:, k], t_quat.reshape(n, N, 4)[:, k])
                                          for k in range(N)], dim=1))
        fq = finger_positions(self.env)
        grasp = torch.stack([grasp_term_fingers(d_grasp[:, k], fq, self.NUT_WIDTH) for k in range(N)], dim=1)
        held = (d_grasp <= self.HELD_NEAR).float()
        dz = pos[..., 2] - self._z0
        lift = held * (0.5 * (dz / self.LIFT_FULL).clamp(0.0, 1.0) + 0.5 * (dz > self.LIFT_BONUS_Z).float())
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
        approach = aligned.float() * kernel((depth - self.ENGAGE_Z).clamp(min=0.0), 0.05)
        engaged = (xy <= c.align_xy) & (depth <= self.ENGAGE_Z)
        thread = engaged.float() * ((self.ENGAGE_Z - depth) / (self.ENGAGE_Z - c.seat_z)).clamp(0.0, 1.0)
        out = {k: v.mean(dim=1) for k, v in dict(reach=reach, grasp=grasp, lift=lift, transport=transport, upright=upright,
                                                 approach=approach, thread=thread).items()}
        out["dist_m"] = d_grasp.mean(dim=1)  # diagnostic only
        return out


class NutTunedEnv(RoboBenchEnv):
    """Nut-thread TUNED env (first-stage target: the grader's `lifted` — nut origin raised to the bolt-top height).
    Same recipe as bulb_tuned: pose reach onto the nut centre, per-finger grasp on the 24 mm hex, lift gated on
    proximity from the settled height, finger PD 8000, warm-start curriculum above the nut."""
    name = "nut_tuned"
    reward_cls = NutThreadTunedReward

    @classmethod
    def defaults(cls):
        return {
            "task": {"episode_seconds": 14},
            "robot": {"gripper_stiffness": 8000},
            "action": {"affine": [{"dims": [6, 7], "lo": 0.0, "hi": 0.04}]},
            "curriculum": {"hover_start_frac": 0.5, "hover_steps": 90, "hover_jitter": 0.03},
            "reward": {"weights": {"reach": 0.2, "grasp": 0.2, "lift": 0.6}},
            "ppo": {"num_steps_per_env": 32, "algorithm": {"entropy_coef": 0.006}},
        }

    def _nut(self) -> torch.Tensor:
        return self.env.scene.nuts[0].data.root_pos_w

    def _get_terminated(self):
        c = self.env.scene.cfg
        return self._nut()[:, 2] < (c.surface_z + c.nut_init_z - 0.10)

    def _hover_target(self):
        return self._nut() + torch.tensor([0.0, 0.0, 0.10], device=self.device)
