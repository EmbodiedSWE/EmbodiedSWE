"""Smoke test: build Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0, reset, introspect.

Prints the action-manager layout, scene entities, robot links/joints, and key poses.
Does NOT step with nonzero actions (absolute IK targets need a sane layout first).
"""

import argparse

# Import pinocchio before AppLauncher to force the use of the pip-installed version
# over the one bundled with Isaac Sim (boost-python type registration conflict).
import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
# makes AppLauncher patch pxr.Gf.Matrix4d for pinocchio compatibility
args.enable_pinocchio = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

# pick_place is blacklisted from the bulk import scan (pinocchio compat TODO);
# the task only registers when this subpackage is imported explicitly.
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0"


def main():
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    env = gym.make(TASK, cfg=env_cfg)
    print("[SMOKE] env created OK", flush=True)
    obs, info = env.reset()
    print("[SMOKE] reset OK", flush=True)

    uenv = env.unwrapped
    am = uenv.action_manager
    print("[SMOKE] total_action_dim:", am.total_action_dim, flush=True)
    print("[SMOKE] action terms:", am.active_terms, flush=True)
    for name in am.active_terms:
        term = am.get_term(name)
        print(f"[SMOKE] term '{name}': dim={term.action_dim} class={type(term).__name__}", flush=True)

    robot = uenv.scene["robot"]
    print("[SMOKE] robot body names:", robot.body_names, flush=True)
    print("[SMOKE] robot joint names:", robot.joint_names, flush=True)
    print("[SMOKE] robot root pos:", robot.data.root_pos_w[0].cpu().numpy(), flush=True)

    obj = uenv.scene["object"]
    print("[SMOKE] object pos:", obj.data.root_pos_w[0].cpu().numpy(), flush=True)
    print("[SMOKE] object quat:", obj.data.root_quat_w[0].cpu().numpy(), flush=True)

    li = robot.body_names.index("left_wrist_yaw_link")
    ri = robot.body_names.index("right_wrist_yaw_link")
    print("[SMOKE] left eef pos:", robot.data.body_pos_w[0, li].cpu().numpy(),
          "quat:", robot.data.body_quat_w[0, li].cpu().numpy(), flush=True)
    print("[SMOKE] right eef pos:", robot.data.body_pos_w[0, ri].cpu().numpy(),
          "quat:", robot.data.body_quat_w[0, ri].cpu().numpy(), flush=True)

    print("[SMOKE] obs policy keys:", list(obs["policy"].keys()) if isinstance(obs["policy"], dict) else obs["policy"].shape, flush=True)

    # hand joint limits (for open/close targets)
    hand_names = [
        "left_hand_index_0_joint", "left_hand_middle_0_joint", "left_hand_thumb_0_joint",
        "right_hand_index_0_joint", "right_hand_middle_0_joint", "right_hand_thumb_0_joint",
        "left_hand_index_1_joint", "left_hand_middle_1_joint", "left_hand_thumb_1_joint",
        "right_hand_index_1_joint", "right_hand_middle_1_joint", "right_hand_thumb_1_joint",
        "left_hand_thumb_2_joint", "right_hand_thumb_2_joint",
    ]
    limits = robot.data.joint_pos_limits[0]
    for n in hand_names:
        j = robot.joint_names.index(n)
        lo, hi = limits[j, 0].item(), limits[j, 1].item()
        print(f"[SMOKE] limit {n}: [{lo:.3f}, {hi:.3f}] default={robot.data.default_joint_pos[0, j].item():.3f}", flush=True)

    # finger/hand link offsets in the left wrist frame (hand geometry)
    import isaaclab.utils.math as math_utils
    lw_pos = robot.data.body_pos_w[0, li]
    lw_quat = robot.data.body_quat_w[0, li]
    for bn in robot.body_names:
        if "left_hand" in bn or "left_wrist" in bn:
            bi = robot.body_names.index(bn)
            rel = math_utils.quat_apply(math_utils.quat_inv(lw_quat), robot.data.body_pos_w[0, bi] - lw_pos)
            print(f"[SMOKE] left-wrist-frame offset of {bn}: {rel.cpu().numpy().round(4)}", flush=True)

    print("[SMOKE] object mass:", obj.root_physx_view.get_masses().cpu().numpy(), flush=True)

    # hold-pose stepping check: command current eef poses + default hand joints.
    # if the action layout is right, the robot should barely move.
    action = torch.zeros((1, 28), device=uenv.device)
    action[0, 0:3] = robot.data.body_pos_w[0, li] - uenv.scene.env_origins[0]
    action[0, 3:7] = robot.data.body_quat_w[0, li]
    action[0, 7:10] = robot.data.body_pos_w[0, ri] - uenv.scene.env_origins[0]
    action[0, 10:14] = robot.data.body_quat_w[0, ri]
    for n_i, n in enumerate(hand_names):
        j = robot.joint_names.index(n)
        action[0, 14 + n_i] = robot.data.default_joint_pos[0, j]
    for i in range(25):
        obs, rew, terminated, truncated, info = env.step(action)
    lw_err = (robot.data.body_pos_w[0, li] - lw_pos).norm().item()
    print(f"[SMOKE] after 25 hold steps: left wrist moved {lw_err*1000:.1f} mm "
          f"(small => action layout correct)", flush=True)
    print("[SMOKE] left eef pos now:", (robot.data.body_pos_w[0, li] - uenv.scene.env_origins[0]).cpu().numpy(), flush=True)
    print("[SMOKE] right eef pos now:", (robot.data.body_pos_w[0, ri] - uenv.scene.env_origins[0]).cpu().numpy(), flush=True)
    print("[SMOKE] object pos now:", (obj.data.root_pos_w[0] - uenv.scene.env_origins[0]).cpu().numpy(), flush=True)
    print("[SMOKE] terminated/truncated:", bool(terminated[0]), bool(truncated[0]), flush=True)

    env.close()
    print("[SMOKE] done", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
