"""bulb: DENSE reward = the up-front full-task proposal (never iterated); TUNED reward + env = the expert-iterated
first-stage recipe (measured grasp geometry, per-finger grasp, settled lift baseline, curriculum, finger PD)."""
from __future__ import annotations

import math

import torch

from . import TaskReward
from .common import (finger_positions, grasp_term_fingers, hand_target_from_axes, kernel, keypoint_distance,
                     local_axis, pinch_point, reach_kernel)
from ..vec_env import RoboBenchEnv


class BulbDenseReward(TaskReward):
    """Dense potential for the bulb-threading scene (public accessors only).

    Stages a practitioner would write for pick -> carry -> upright -> engage -> thread:
    reach      finger pads close to the bulb's glass belly
    lift       bulb raised off its start height
    transport  bulb xy over the socket axis
    upright    bulb screw axis parallel to the socket axis (cap down)
    approach   when over the axis and upright: bulb lowered to the bore mouth
    thread     when engaged in the bore: depth from free-rest to seat (the scene's own glow band)
    All per-bulb, averaged over an env's bulbs. Success itself is left to the grader/bonus."""

    GRASP_Z = 0.055  # m above the cap bottom (bulb origin) — the glass belly a gripper pinches; calibrate
    LIFT_FULL = 0.08  # m of lift that counts as fully lifted

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
        gz = torch.tensor([0.0, 0.0, self.GRASP_Z], device=pos.device).expand(n * N, 3)
        grasp_pt = pos + quat_apply(quat.reshape(-1, 4), gz).reshape(n, N, 3)
        reach = reach_kernel((grasp_pt - pinch[:, None, :]).norm(dim=-1))
        lift = ((pos[..., 2] - self._z0) / self.LIFT_FULL).clamp(0.0, 1.0)
        xy, depth, cos = self._offsets()
        transport = kernel(xy, 0.10)
        upright = cos.clamp(0.0, 1.0)
        aligned = (xy <= 2 * c.align_xy) & (cos >= math.cos(math.radians(2 * c.align_axis_deg)))
        approach = aligned.float() * kernel((depth - c.light_start_z).clamp(min=0.0), 0.05)
        engaged = (xy <= c.align_xy) & (depth <= c.socket_opening_z)
        thread = engaged.float() * ((c.light_start_z - depth) / (c.light_start_z - c.seat_z)).clamp(0.0, 1.0)
        return {k: v.mean(dim=1) for k, v in dict(reach=reach, lift=lift, transport=transport, upright=upright,
                                                  approach=approach, thread=thread).items()}


# ----- TUNED condition: expert-iterated first-stage reward + env (task_env: <scene>_tuned) -----


class BulbTunedReward(TaskReward):
    """Dense potential for the bulb-threading scene (public accessors only).

    Stages a practitioner would write for pick -> carry -> upright -> engage -> thread:
    reach      hand at the nominal grasp pose: pads at the neck waist, approaching from above, fingers
               opening across the bulb axis (keypoint distance)
    grasp      pads at the waist AND fingers closed to the waist width (a pinch)
    lift       bulb raised off its start height while the waist stays at the pads (held): half a
               ramp to LIFT_FULL, half a threshold bonus at LIFT_BONUS_Z (Isaac Lab lift-cube style —
               only a real grasp can lift, so lift is the grasp signal that needs no contact sensing)
    transport  bulb xy over the socket axis
    upright    bulb screw axis parallel to the socket axis (cap down)
    approach   when over the axis and upright: bulb lowered to the bore mouth
    thread     when engaged in the bore: depth from free-rest to seat (the scene's own glow band)
    All per-bulb, averaged over an env's bulbs. Success itself is left to the grader/bonus."""

    GRASP_Z = 0.025  # m along the bulb axis from the cap end: the neck waist
    BELLY_WIDTH = 0.020  # m, the waist diameter the pads close onto (the Ø20 cap end)
    PINCH_OFFSET = 0.099  # m, panda_hand -> pad contact-band centre (solver's GRASP_DROP)
    HELD_NEAR = 0.06  # m: a real pinch measured 3.0-3.2 cm from the neck point (pad geometry); 3 cm gated lift out
    LIFT_FULL = 0.08  # m of lift that counts as fully lifted
    LIFT_BONUS_Z = 0.04  # m: the grader's pick_lift — above this the threshold half of `lift` pays in full

    WEIGHTS = {"reach": 0.1, "grasp": 0.1, "lift": 0.1, "transport": 0.15, "upright": 0.1, "approach": 0.15, "thread": 0.3}

    def reset(self, env_ids: torch.Tensor) -> None:
        z = torch.stack([b.data.root_pos_w[:, 2] for b in self.scene.bulbs], dim=1)  # (n, N)
        if not hasattr(self, "_z0"):
            self._z0 = z.clone()
            art = self.env.robot.articulation
            self._hand_idx = list(art.data.body_names).index(self.env.robot.EE_BODY)
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

    def grasp_points(self) -> torch.Tensor:
        """(n, N, 3) world: the neck-waist point of every bulb."""
        from isaaclab.utils.math import quat_apply

        pos = torch.stack([b.data.root_pos_w for b in self.scene.bulbs], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.scene.bulbs], dim=1)
        n, N = pos.shape[:2]
        gz = torch.tensor([0.0, 0.0, self.GRASP_Z], device=pos.device).expand(n * N, 3)
        return pos + quat_apply(quat.reshape(-1, 4), gz).reshape(n, N, 3)

    def hover_target(self) -> torch.Tensor:
        return self.grasp_points()[:, 0] + torch.tensor([0.0, 0.0, 0.10], device=self.device)

    def terms(self) -> dict[str, torch.Tensor]:
        from isaaclab.utils.math import quat_apply
        # the part settles after reset (measured: the bulb drops 1.9 cm) — track the lowest height seen
        # since reset as the lift baseline, or lift credit only starts above the unsettled spawn height
        self._z0 = torch.minimum(self._z0, torch.stack([b.data.root_pos_w[:, 2] for b in self.scene.bulbs], dim=1))

        c = self.scene.cfg
        pinch = pinch_point(self.env, self.PINCH_OFFSET)  # (n,3)
        pos = torch.stack([b.data.root_pos_w for b in self.scene.bulbs], dim=1)  # (n,N,3)
        quat = torch.stack([b.data.root_quat_w for b in self.scene.bulbs], dim=1)  # (n,N,4)
        n, N = pos.shape[:2]
        gz = torch.tensor([0.0, 0.0, self.GRASP_Z], device=pos.device).expand(n * N, 3)
        grasp_pt = pos + quat_apply(quat.reshape(-1, 4), gz).reshape(n, N, 3)
        d_grasp = (grasp_pt - pinch[:, None, :]).norm(dim=-1)  # (n,N)
        # nominal grasp pose per bulb: approach straight down onto the belly, fingers opening ACROSS the
        # bulb's axis (open axis = axis x down, flattened), signed to the nearer of the two grasps by the
        # hand's current finger axis (+y). Leaving the yaw free let the pads land along the bulb.
        down = torch.tensor([0.0, 0.0, -1.0], device=pos.device).expand(n * N, 3)
        axis_w = local_axis(quat.reshape(-1, 4), 2).reshape(n, N, 3)  # bulb screw axis, world
        across = torch.cross(axis_w, down.reshape(n, N, 3), dim=-1)
        across = across * torch.tensor([1.0, 1.0, 0.0], device=pos.device)
        across = torch.where(across.norm(dim=-1, keepdim=True) < 1e-3,
                             torch.tensor([0.0, 1.0, 0.0], device=pos.device).expand(n, N, 3), across)
        hand_y = local_axis(self.env.robot.articulation.data.body_quat_w[:, self._hand_idx], 1)[:, None, :]
        across = torch.where((across * hand_y).sum(-1, keepdim=True) < 0, -across, across)
        open_w = across.reshape(-1, 3)
        t_pos, t_quat = hand_target_from_axes(grasp_pt.reshape(-1, 3), down, open_w, pinch_offset=self.PINCH_OFFSET)
        reach = torch.stack([keypoint_distance(self.env, t_pos.reshape(n, N, 3)[:, k], t_quat.reshape(n, N, 4)[:, k])
                             for k in range(N)], dim=1)
        reach = reach_kernel(reach)
        fq = finger_positions(self.env)  # (n, F)
        grasp = torch.stack([grasp_term_fingers(d_grasp[:, k], fq, self.BELLY_WIDTH) for k in range(N)], dim=1)
        held = (d_grasp <= self.HELD_NEAR).float()  # part at the pads = held (see slice.py on why not the aperture)
        dz = pos[..., 2] - self._z0
        lift = held * (0.5 * (dz / self.LIFT_FULL).clamp(0.0, 1.0) + 0.5 * (dz > self.LIFT_BONUS_Z).float())
        xy, depth, cos = self._offsets()
        transport = kernel(xy, 0.10)
        upright = cos.clamp(0.0, 1.0)
        aligned = (xy <= 2 * c.align_xy) & (cos >= math.cos(math.radians(2 * c.align_axis_deg)))
        approach = aligned.float() * kernel((depth - c.light_start_z).clamp(min=0.0), 0.05)
        engaged = (xy <= c.align_xy) & (depth <= c.socket_opening_z)
        thread = engaged.float() * ((c.light_start_z - depth) / (c.light_start_z - c.seat_z)).clamp(0.0, 1.0)
        out = {k: v.mean(dim=1) for k, v in dict(reach=reach, grasp=grasp, lift=lift, transport=transport,
                                                 upright=upright, approach=approach, thread=thread).items()}
        out["dist_m"] = d_grasp.mean(dim=1)  # diagnostic only (not in WEIGHTS): pinch point -> belly, metres
        return out


class BulbTunedEnv(RoboBenchEnv):
    """Bulb TUNED env: everything the task-tuned condition changes, in one class — horizon, finger PD, action map, warm-start target,
    the dense reward (neck-waist grasp pose, per-finger grasp,
    settled lift baseline)."""
    name = "bulb_tuned"
    reward_cls = BulbTunedReward

    @classmethod
    def defaults(cls):
        return {
            "task": {"episode_seconds": 14},           # approach ~5-7 s at the learned pace, then pinch + lift
            "robot": {"gripper_stiffness": 8000},      # the scripted solver's finger PD (preset: 2000)
            "action": {"affine": [{"dims": [6, 7], "lo": 0.0, "hi": 0.04}]},  # fingers in metres, independent
            "curriculum": {"hover_start_frac": 0.5, "hover_steps": 90, "hover_jitter": 0.03},
            "reward": {"weights": {"reach": 0.2, "grasp": 0.2, "lift": 0.6}},
            "ppo": {"num_steps_per_env": 32, "algorithm": {"entropy_coef": 0.006}},
        }

    def _neck(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        b = self.env.scene.bulbs[0]
        return b.data.root_pos_w + quat_apply(b.data.root_quat_w, torch.tensor([0.0, 0.0, BulbTunedReward.GRASP_Z], device=self.device).expand(self.num_envs, 3))


    def _hover_target(self):
        return self._neck() + torch.tensor([0.0, 0.0, 0.10], device=self.device)
