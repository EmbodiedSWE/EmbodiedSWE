"""Bulb TUNED env: everything the task-tuned condition changes, in one class — horizon, finger PD, action map, warm-start target,
extra observation, early termination and the dense reward (neck-waist grasp pose, per-finger grasp,
settled lift baseline)."""
from __future__ import annotations

import torch

from ..task_rewards.bulb import GRASP_Z, BulbDenseReward
from ..vec_env import RoboBenchEnv


class BulbTunedEnv(RoboBenchEnv):
    name = "bulb_tuned"
    reward_cls = BulbDenseReward

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
        return b.data.root_pos_w + quat_apply(b.data.root_quat_w, torch.tensor([0.0, 0.0, GRASP_Z], device=self.device).expand(self.num_envs, 3))

    def _get_extra_obs(self):
        """Neck-waist point of bulb 0 relative to the hand, and the bulb axis: what the pinch needs."""
        from isaaclab.utils.math import quat_apply

        art = self.env.robot.articulation
        ee = list(art.data.body_names).index(self.env.robot.EE_BODY)
        axis = quat_apply(self.env.scene.bulbs[0].data.root_quat_w, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3))
        return torch.cat([self._neck() - art.data.body_pos_w[:, ee], axis], dim=1)

    def _get_terminated(self):
        """Bulb knocked off the work surface (fell > 10 cm below its spawn height): end the episode."""
        c = self.env.scene.cfg
        return self.env.scene.bulbs[0].data.root_pos_w[:, 2] < (c.surface_z + c.bulb_init_z - 0.10)

    def _hover_target(self):
        return self._neck() + torch.tensor([0.0, 0.0, 0.10], device=self.device)
