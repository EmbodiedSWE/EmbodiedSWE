"""pen_holder: DENSE reward = the up-front full-task proposal (never iterated); TUNED reward + env = the expert-iterated
first-stage recipe (measured grasp geometry, per-finger grasp, settled lift baseline, curriculum, finger PD)."""
from __future__ import annotations

import torch

from . import TaskReward
from .common import (finger_positions, grasp_term_fingers, hand_target_from_axes, kernel, keypoint_distance,
                     local_axis, pinch_point, reach_kernel)
from ..vec_env import RoboBenchEnv


class PenHolderDenseReward(TaskReward):
    """Dense potential for the pen-holder scene.

    reach    finger pads at the nearest not-yet-inserted present pen
    lift     that pen raised off the table
    carry    that pen's bottom end over the holder axis (holder frame)
    tip_up   that pen's axis along the holder axis, tip up
    insert   pen bottom lowered below the rim toward depth_min (when near the axis)
    all_in   fraction of present pens counted (the scene's own rubric term)
    "Nearest uninserted pen" focuses the dense terms on one pen at a time; `all_in` keeps credit
    for the ones already done."""

    LIFT_FULL = 0.10

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
        lift = ((pos[idx, k, 2] - self._z0[idx, k]) / self.LIFT_FULL).clamp(0.0, 1.0)
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


# ----- TUNED condition: expert-iterated first-stage reward + env (task_env: <scene>_tuned) -----


class PenHolderTunedReward(TaskReward):
    """Dense potential for the pen-holder scene.

    reach    finger pads at the nearest not-yet-inserted present pen
    lift     that pen raised off the table
    carry    that pen's bottom end over the holder axis (holder frame)
    tip_up   that pen's axis along the holder axis, tip up
    insert   pen bottom lowered below the rim toward depth_min (when near the axis)
    all_in   fraction of present pens counted (the scene's own rubric term)
    "Nearest uninserted pen" focuses the dense terms on one pen at a time; `all_in` keeps credit
    for the ones already done."""

    LIFT_FULL = 0.10
    LIFT_BONUS_Z = 0.04
    PEN_WIDTH = 0.0116  # m, the vendored pencil barrel Ø (2 x 5.8 mm radius)
    PINCH_OFFSET = 0.099
    HELD_NEAR = 0.06

    WEIGHTS = {"reach": 0.1, "grasp": 0.1, "lift": 0.1, "carry": 0.15, "tip_up": 0.1, "insert": 0.2, "all_in": 0.25}

    def reset(self, env_ids: torch.Tensor) -> None:
        pos, _, _ = self.scene._pen_tensors()
        if not hasattr(self, "_z0"):
            self._z0 = pos[..., 2].clone()
            art = self.env.robot.articulation
            self._hand_idx = list(art.data.body_names).index(self.env.robot.EE_BODY)
        self._z0[env_ids] = pos[env_ids, :, 2]

    def focus(self) -> torch.Tensor:
        """(n,) the pen the dense terms track: nearest present, not-yet-counted (fallback: any present)."""
        sc = self.scene
        pos, _, _ = sc._pen_tensors()
        present, counted = sc.present, sc.counted() & sc.present
        d = (pos - pinch_point(self.env, self.PINCH_OFFSET)[:, None, :]).norm(dim=-1)
        d_focus = torch.where(present & ~counted, d, torch.full_like(d, 1e3))
        d_focus = torch.where(present.any(dim=1, keepdim=True) & (d_focus >= 1e3).all(dim=1, keepdim=True),
                              torch.where(present, d, torch.full_like(d, 1e3)), d_focus)
        return d_focus.argmin(dim=1)

    def hover_target(self) -> torch.Tensor:
        pos, _, _ = self.scene._pen_tensors()
        k = self.focus()
        return pos[torch.arange(pos.shape[0], device=pos.device), k] + torch.tensor([0.0, 0.0, 0.10], device=self.device)

    def terms(self) -> dict[str, torch.Tensor]:
        # the part settles after reset (measured: the bulb drops 1.9 cm) — track the lowest height seen
        # since reset as the lift baseline, or lift credit only starts above the unsettled spawn height
        self._z0 = torch.minimum(self._z0, self.scene._pen_tensors()[0][..., 2])
        sc, c = self.scene, self.scene.cfg
        pos, _quat, _vel = sc._pen_tensors()  # (n,P,3)
        b_loc, t_loc = sc._pen_ends_local()  # holder frame (n,P,3)
        present = sc.present
        counted = sc.counted() & present
        pinch = pinch_point(self.env, self.PINCH_OFFSET)
        n, P = pos.shape[:2]
        k = self.focus()
        idx = torch.arange(n, device=pos.device)
        done_all = counted.sum(1) >= present.sum(1)
        d = (pos - pinch[:, None, :]).norm(dim=-1)
        pk, qk = pos[idx, k], _quat[idx, k]
        # nominal grasp pose on the focused pen: approach down, fingers opening ACROSS the pen axis
        down = torch.tensor([0.0, 0.0, -1.0], device=pos.device).expand(n, 3)
        axis_w = local_axis(qk, 2)
        across = torch.cross(axis_w, down, dim=-1) * torch.tensor([1.0, 1.0, 0.0], device=pos.device)
        across = torch.where(across.norm(dim=-1, keepdim=True) < 1e-3, torch.tensor([0.0, 1.0, 0.0], device=pos.device).expand(n, 3), across)
        hand_y = local_axis(self.env.robot.articulation.data.body_quat_w[:, self._hand_idx], 1)
        across = torch.where((across * hand_y).sum(-1, keepdim=True) < 0, -across, across)
        t_pos, t_quat = hand_target_from_axes(pk, down, across, pinch_offset=self.PINCH_OFFSET)
        reach = reach_kernel(keypoint_distance(self.env, t_pos, t_quat))
        grasp = grasp_term_fingers(d[idx, k], finger_positions(self.env), self.PEN_WIDTH)
        held = (d[idx, k] <= self.HELD_NEAR).float()
        dz = pk[:, 2] - self._z0[idx, k]
        lift = held * (0.5 * (dz / self.LIFT_FULL).clamp(0.0, 1.0) + 0.5 * (dz > self.LIFT_BONUS_Z).float())
        bxy = b_loc[idx, k, :2].norm(dim=-1)
        carry = held * kernel(bxy, 0.10)  # carry / tip-up / insert count only while HELD (else: shoving exploit)
        axis = t_loc[idx, k] - b_loc[idx, k]
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        tip_up = held * axis[:, 2].clamp(0.0, 1.0)
        depth = c.holder_h / 2 - b_loc[idx, k, 2]  # below the rim
        insert = held * (bxy <= 2 * c.xy_tol).float() * (depth / c.depth_min).clamp(0.0, 1.0)
        # once every present pen is in, the focused pen is a done one: hold the dense terms at 1
        one = torch.ones_like(reach)
        reach, grasp, lift, carry, tip_up, insert = [torch.where(done_all, one, t) for t in (reach, grasp, lift, carry, tip_up, insert)]
        all_in = counted.sum(1).float() / present.sum(1).clamp(min=1).float()
        return dict(reach=reach, grasp=grasp, lift=lift, carry=carry, tip_up=tip_up, insert=insert, all_in=all_in, dist_m=d[idx, k])


class PenTunedEnv(RoboBenchEnv):
    """Pen-holder TUNED env. The grader's first stage (`pens_in`) already needs a pen inserted tip-up in the cup, so
    the tuned reward keeps all rungs (reach, per-finger grasp, lift, carry, tip-up, insert) and the curriculum
    warm-starts above the focused pen."""
    name = "pen_tuned"
    reward_cls = PenHolderTunedReward

    @classmethod
    def defaults(cls):
        return {
            "task": {"episode_seconds": 20},
            "robot": {"gripper_stiffness": 8000},
            "action": {"affine": [{"dims": [6, 7], "lo": 0.0, "hi": 0.04}]},
            "curriculum": {"hover_start_frac": 0.5, "hover_steps": 90, "hover_jitter": 0.03},
            "ppo": {"num_steps_per_env": 32, "algorithm": {"entropy_coef": 0.006}},
        }

    def _get_terminated(self):
        c = self.env.scene.cfg
        pos, _, _ = self.env.scene._pen_tensors()
        return ((pos[..., 2] < c.surface_z - 0.10) & self.env.scene.present).any(dim=1)
