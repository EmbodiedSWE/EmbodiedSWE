# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted controller that solves the ``Isaac-AutoMate-Assembly-Direct-v0`` task.

The task spawns a Franka robot that has already grasped a plug; the plug starts a
few centimeters above (disassembled from) its socket.  The goal is to insert the
plug into the socket.

This script does *not* use a trained RL policy.  Instead it drives the robot with
an analytic, two-phase task-space controller that exploits the structure of the
environment:

* ``env.gripper_goal_pos`` is the world position the fingertip-midpoint must reach
  for the grasped plug to be fully seated in the socket (it already accounts for
  the socket pose and the grasp pose).
* The 6-D action is interpreted by the env as ``[dx, dy, dz, d_rotx, d_roty,
  d_rotz]`` where the position part is scaled by ``pos_threshold`` (0.1 m) and
  added to the current fingertip position to form an impedance-control target.

Controller:
    Phase 1 (align): command the fingertip to hover a small clearance directly
        above the goal so the plug's lateral (x, y) error is driven to ~0 over the
        socket opening.
    Phase 2 (insert): once aligned, descend straight down onto the goal.  The
        socket lead-in chamfer + impedance control absorb the residual <1 mm
        misalignment and seat the plug.

Run (from the repo root so the AutoMate asset files resolve in ./):

    ./isaaclab.sh -p Opus/AutoMate-Assembly-Direct-v0/insert_plug.py --headless --video

The recorded video is written to Opus/AutoMate-Assembly-Direct-v0/.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Scripted plug insertion for AutoMate assembly.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
parser.add_argument("--num_steps", type=int, default=150, help="Number of control steps to run.")
parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible resets.")
parser.add_argument("--cam_env", type=int, default=0, help="Env index the recording camera focuses on.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the trajectory.")
parser.add_argument("--video_length", type=int, default=None, help="Video length in steps (default: num_steps).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Recording needs cameras enabled.
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# ---------------------------------------------------------------------------
# Launch Isaac Sim
# ---------------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after the simulator is up."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401  (registers the gym tasks)
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Isaac-AutoMate-Assembly-Direct-v0"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    num_steps = args_cli.num_steps
    video_length = args_cli.video_length if args_cli.video_length is not None else num_steps

    # ---- configure the environment ----
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    # Turn OFF the sampling-based curriculum so *every* plug is initialized fully
    # outside the socket -> the controller must perform a real insertion.
    env_cfg.tasks["insertion"].if_sbc = False
    env_cfg.tasks["insertion"].if_logging_eval = False
    # Long enough episode that the env does not auto-reset mid-trajectory.
    env_cfg.episode_length_s = 30.0

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(TASK, cfg=env_cfg, render_mode=render_mode)

    if args_cli.video:
        video_kwargs = {
            "video_folder": OUT_DIR,
            "step_trigger": lambda step: step == 0,
            "video_length": video_length,
            "disable_logger": True,
            "name_prefix": "automate_insertion",
        }
        print(f"[INFO] Recording video ({video_length} steps) to: {OUT_DIR}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    e = env.unwrapped
    device = e.device

    # ---- reset (this grasps the plug and positions it above the socket) ----
    env.reset()

    # Point the recording camera at the focus env's socket.
    cam_idx = min(args_cli.cam_env, e.num_envs - 1)
    socket_w = (e.fixed_pos[cam_idx] + e.scene.env_origins[cam_idx]).detach().cpu().numpy()
    eye = (float(socket_w[0]) + 0.22, float(socket_w[1]) + 0.22, float(socket_w[2]) + 0.16)
    target = (float(socket_w[0]), float(socket_w[1]), float(socket_w[2]) + 0.03)
    e.sim.set_camera_view(eye, target)

    # ---- controller constants ----
    pos_thresh = e.pos_threshold            # (N, 3), = 0.1 m
    hover_clearance = 0.015                 # hover 1.5 cm above goal while aligning
    align_tol = 0.0025                      # x-y error (m) required before descending
    hover_z_tol = 0.004                     # z tolerance (m) for "reached hover point"

    descending = torch.zeros(e.num_envs, dtype=torch.bool, device=device)
    ever_success = torch.zeros(e.num_envs, dtype=torch.bool, device=device)

    print(f"[INFO] Running scripted insertion for {num_steps} steps on {e.num_envs} envs.")
    for step in range(num_steps):
        p = e.fingertip_midpoint_pos          # (N, 3) world fingertip position
        g = e.gripper_goal_pos                # (N, 3) world target for full insertion

        xy_err = torch.norm(g[:, :2] - p[:, :2], dim=-1)
        hover_z = g[:, 2] + hover_clearance

        # Latch into the descent phase once aligned over the opening at hover height.
        reached_hover = (xy_err < align_tol) & (torch.abs(p[:, 2] - hover_z) < hover_z_tol)
        descending = descending | reached_hover

        # Build the position target: hover above goal until aligned, then seat onto goal.
        target_pos = g.clone()
        target_pos[:, 2] = torch.where(descending, g[:, 2], hover_z)

        pos_err = target_pos - p
        action = torch.zeros((e.num_envs, 6), device=device)
        action[:, 0:3] = torch.clamp(pos_err / pos_thresh, -1.0, 1.0)
        # Keep orientation fixed (round peg -> yaw irrelevant; env forces upright).

        env.step(action)

        ever_success |= e.ep_succeeded.bool()

        if step % 15 == 0 or step == num_steps - 1:
            n_desc = int(descending.sum().item())
            n_succ = int(e.ep_succeeded.bool().sum().item())
            print(
                f"  step {step:3d} | descending {n_desc:3d}/{e.num_envs} | "
                f"mean xy_err {xy_err.mean().item()*1000:5.2f} mm | "
                f"success now {n_succ:3d}/{e.num_envs}"
            )

    n_succ = int(ever_success.sum().item())
    print(f"\n[RESULT] Insertion success: {n_succ}/{e.num_envs} envs "
          f"({100.0 * n_succ / e.num_envs:.1f}%).")
    print(f"[RESULT] Focus env {cam_idx} success: {bool(ever_success[cam_idx].item())}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
