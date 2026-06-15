"""Probe the AutoMate assembly env reset pipeline stage by stage.

Monkeypatches logging into the reset methods to find where the grasp fails:
IK convergence, gripper close, or post-gravity plug retention.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--sbc", type=int, default=0, help="1 = if_sbc True (stock train), 0 = False (eval)")
parser.add_argument("--render_interval", type=int, default=1, help="stock is 1")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.automate.assembly_env import AssemblyEnv
from isaaclab_tasks.utils import parse_env_cfg


def fmt(t, n=3):
    return [round(float(x), 4) for x in t.cpu().flatten()[: n * 4]]


# --- monkeypatch logging into reset stages ---
_orig_ik = AssemblyEnv.set_pos_inverse_kinematics
_orig_grasp = AssemblyEnv._move_gripper_to_grasp_pose
_orig_held = AssemblyEnv.randomize_held_initial_state
_orig_rand = AssemblyEnv.randomize_initial_state


def patched_ik(self, env_ids):
    pos_err, aa_err = _orig_ik(self, env_ids)
    print(f"[PROBE] IK done: pos_err_norm(mm)={[round(float(x)*1e3,2) for x in pos_err.norm(dim=-1)]} "
          f"aa_err_norm(rad)={[round(float(x),4) for x in aa_err.norm(dim=-1)]}", flush=True)
    return pos_err, aa_err


def patched_grasp(self, env_ids):
    print(f"[PROBE] pre-grasp : held_z={[round(float(x),4) for x in self.held_pos[:,2]]} "
          f"fingertip={fmt(self.fingertip_midpoint_pos[0])}", flush=True)
    _orig_grasp(self, env_ids)
    err = (self.ctrl_target_fingertip_midpoint_pos - self.fingertip_midpoint_pos).norm(dim=-1)
    print(f"[PROBE] post-grasp-move: fingertip_target_err(mm)={[round(float(x)*1e3,2) for x in err]} "
          f"fingers={[round(float(x),4) for x in self.joint_pos[0,7:9]]}", flush=True)


def patched_held(self, env_ids, pre_grasp):
    _orig_held(self, env_ids, pre_grasp)
    print(f"[PROBE] held placed (pre_grasp={pre_grasp}): disp={[round(float(x),4) for x in self.curriculum_disp]} "
          f"held-socket_xy(mm)={[round(float(x)*1e3,1) for x in (self.held_pos[:,:2]-self.fixed_pos[:,:2]).norm(dim=-1)]} "
          f"held_z-socket_z(mm)={[round(float(x)*1e3,1) for x in (self.held_pos[:,2]-self.fixed_pos[:,2])]}", flush=True)


def patched_rand(self, env_ids):
    print(f"[PROBE] === randomize_initial_state: gripper_open_width={self.gripper_open_width}", flush=True)
    _orig_rand(self, env_ids)
    print(f"[PROBE] === reset complete (gravity restored):", flush=True)
    print(f"[PROBE] fingers={[round(float(x),4) for x in self.joint_pos[0,7:9]]} (env0)", flush=True)
    print(f"[PROBE] held-socket_xy(mm)={[round(float(x)*1e3,1) for x in (self.held_pos[:,:2]-self.fixed_pos[:,:2]).norm(dim=-1)]}", flush=True)
    print(f"[PROBE] held_z-socket_z(mm)={[round(float(x)*1e3,1) for x in (self.held_pos[:,2]-self.fixed_pos[:,2])]}", flush=True)
    ft_to_plug = (self.fingertip_midpoint_pos - self.held_pos).norm(dim=-1)
    print(f"[PROBE] fingertip-plug dist(mm)={[round(float(x)*1e3,1) for x in ft_to_plug]}", flush=True)


AssemblyEnv.set_pos_inverse_kinematics = patched_ik
AssemblyEnv._move_gripper_to_grasp_pose = patched_grasp
AssemblyEnv.randomize_held_initial_state = patched_held
AssemblyEnv.randomize_initial_state = patched_rand


def main():
    env_cfg = parse_env_cfg("Isaac-AutoMate-Assembly-Direct-v0", device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.tasks["insertion"].if_sbc = bool(args_cli.sbc)
    env_cfg.sim.render_interval = args_cli.render_interval
    env = gym.make("Isaac-AutoMate-Assembly-Direct-v0", cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped

    print("[PROBE] ##### RESET 1 #####", flush=True)
    env.reset()

    # let physics settle with zero actions (gravity on) and watch plug retention
    for i in range(10):
        with torch.inference_mode():
            env.step(torch.zeros((uenv.num_envs, 6), device=uenv.device))
        if i % 3 == 0:
            xy = (uenv.held_pos[:, :2] - uenv.fixed_pos[:, :2]).norm(dim=-1)
            print(f"[PROBE] step {i}: held-socket_xy(mm)={[round(float(x)*1e3,1) for x in xy]} "
                  f"held_z-socket_z(mm)={[round(float(x)*1e3,1) for x in (uenv.held_pos[:,2]-uenv.fixed_pos[:,2])]} "
                  f"ft-plug(mm)={[round(float(x)*1e3,1) for x in (uenv.fingertip_midpoint_pos-uenv.held_pos).norm(dim=-1)]}",
                  flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
