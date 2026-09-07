"""Nut-thread TUNED env (first-stage target: the grader's `lifted` — nut origin raised to the bolt-top height).
Same recipe as bulb_tuned: pose reach onto the nut centre, per-finger grasp on the 24 mm hex, lift gated on
proximity from the settled height, finger PD 8000, warm-start curriculum above the nut."""
from __future__ import annotations

import torch

from ..task_rewards.nut_thread import NutThreadDenseReward
from ..vec_env import RoboBenchEnv


class NutTunedEnv(RoboBenchEnv):
    name = "nut_tuned"
    reward_cls = NutThreadDenseReward

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
