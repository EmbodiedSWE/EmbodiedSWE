"""Smoke test: launch Isaac Sim headless, create the AutoMate assembly env, step it a few times.

Verifies asset downloads, env construction, and offscreen rendering work before
writing the full scripted-insertion controller.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg("Isaac-AutoMate-Assembly-Direct-v0", device="cuda:0", num_envs=4)
    env = gym.make("Isaac-AutoMate-Assembly-Direct-v0", cfg=env_cfg, render_mode="rgb_array")
    print("[SMOKE] env created OK")
    obs, info = env.reset()
    print("[SMOKE] reset OK, obs keys:", obs.keys() if isinstance(obs, dict) else type(obs))

    uenv = env.unwrapped
    print("[SMOKE] fixed_pos[0]:", uenv.fixed_pos[0].cpu().numpy())
    print("[SMOKE] held_pos[0]:", uenv.held_pos[0].cpu().numpy())
    print("[SMOKE] fingertip_pos[0]:", uenv.fingertip_midpoint_pos[0].cpu().numpy())
    print("[SMOKE] gripper_goal_pos[0]:", uenv.gripper_goal_pos[0].cpu().numpy())
    print("[SMOKE] disassembly_dists:", uenv.disassembly_dists[:4].cpu().numpy())
    print("[SMOKE] curriculum_height_bound[0]:", uenv.curriculum_height_bound[0].cpu().numpy())
    print("[SMOKE] max_episode_length:", uenv.max_episode_length)

    for i in range(5):
        action = torch.zeros((uenv.num_envs, 6), device=uenv.device)
        obs, rew, terminated, truncated, info = env.step(action)
    frame = env.render()
    print("[SMOKE] render frame:", None if frame is None else frame.shape)
    print("[SMOKE] 5 steps OK")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
