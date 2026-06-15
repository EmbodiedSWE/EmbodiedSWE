# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Empirical probe for Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0.

Boots the env once and:
  * dumps static facts (object pose, wrist home poses, body names, hand-joint order, object AABB),
  * confirms the action pose frame (command home pose, read achieved wrist pose),
  * maps reachability of candidate pick/place waypoints for BOTH hands (the Pink IK stalls
    silently on infeasible targets, so we measure commanded-vs-achieved error to detect reach).
"""

import sys

if sys.platform != "win32":
    import pinocchio  # noqa: F401  (force IsaacLab's pinocchio before the app)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = "Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0"
HOME_Q = [0.7071, 0.0, 0.0, 0.7071]  # known-good wrist orientation (wxyz)


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    # Avoid auto-resets while probing (we don't move the object).
    for term in ("time_out", "object_dropping", "success"):
        if hasattr(env_cfg.terminations, term):
            setattr(env_cfg.terminations, term, None)

    env = gym.make(TASK, cfg=env_cfg).unwrapped
    obs, _ = env.reset()

    device = env.device
    scene = env.scene
    robot = scene["robot"]
    obj = scene["object"]
    body_names = robot.data.body_names
    li = body_names.index("left_wrist_yaw_link")
    ri = body_names.index("right_wrist_yaw_link")
    origin = scene.env_origins[0]

    action_term = env.action_manager.get_term("upper_body_ik")
    hand_names = action_term._hand_joint_names
    n_hand = len(hand_names)
    print("\n================ STATIC FACTS ================")
    print("action dim:", env.action_manager.total_action_dim)
    print("num_hand_joints:", n_hand)
    print("hand joint order (action[14:]):")
    for k, nm in enumerate(hand_names):
        print(f"   {k:2d} -> {nm}")
    print("env origin:", origin.cpu().numpy())

    def wpos(idx):
        return (robot.data.body_pos_w[0, idx] - origin).cpu().numpy()

    def wquat(idx):
        return robot.data.body_quat_w[0, idx].cpu().numpy()

    print("pelvis idx:", body_names.index("pelvis"), "pos:", wpos(body_names.index("pelvis")))
    print("left_wrist_yaw_link  home pos:", wpos(li), "quat:", wquat(li))
    print("right_wrist_yaw_link home pos:", wpos(ri), "quat:", wquat(ri))
    objp = (obj.data.root_pos_w[0] - origin).cpu().numpy()
    print("object pos:", objp, "quat:", obj.data.root_quat_w[0].cpu().numpy())

    # Object AABB via USD
    try:
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath("/World/envs/env_0/Object")
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = np.array(rng.GetMin()), np.array(rng.GetMax())
        print("object world AABB min:", mn, "max:", mx, "size:", mx - mn)
    except Exception as e:
        print("AABB failed:", e)

    def make_action(lp, lq, rp, rq, hand=None):
        a = np.zeros(14 + 14, dtype=np.float32)
        a[0:3] = lp
        a[3:7] = lq
        a[7:10] = rp
        a[10:14] = rq
        if hand is not None:
            a[14:] = hand
        return torch.tensor(a, device=device).unsqueeze(0)

    def quat_to_mat(q):
        from isaaclab.utils.math import matrix_from_quat

        return matrix_from_quat(torch.tensor(q, device=device).unsqueeze(0))[0].cpu().numpy()

    def hold(lp, rp, n=60, lq=HOME_Q, rq=HOME_Q):
        act = make_action(lp, lq, rp, rq)
        for _ in range(n):
            env.step(act)
        lp_a, rp_a = wpos(li), wpos(ri)
        lq_a, rq_a = wquat(li), wquat(ri)
        lx = quat_to_mat(lq_a)[:, 0]  # local +X (finger pointing dir) in world
        rx = quat_to_mat(rq_a)[:, 0]
        return lp_a, rp_a, lx, rx, np.linalg.norm(lp_a - lp), np.linalg.norm(rp_a - rp)

    HOME_L = np.array([-0.18, 0.1, 0.8])
    HOME_R = np.array([0.18, 0.1, 0.8])

    # (label, left_target, right_target)
    probes = [
        ("home", HOME_L, HOME_R),
        ("L_over_object", np.array([-0.35, 0.45, 0.85]), HOME_R),
        ("L_at_object", np.array([-0.35, 0.45, 0.72]), HOME_R),
        ("L_carry_mid", np.array([0.05, 0.45, 0.95]), HOME_R),
        ("L_place_x0.45", np.array([0.45, 0.45, 0.85]), HOME_R),
        ("L_place_x0.55", np.array([0.55, 0.475, 0.85]), HOME_R),
        ("R_over_object_xbody", HOME_L, np.array([-0.35, 0.45, 0.85])),
        ("R_place_x0.45", HOME_L, np.array([0.45, 0.45, 0.85])),
        ("R_place_x0.55", HOME_L, np.array([0.55, 0.475, 0.80])),
        ("R_retract", HOME_L, np.array([0.15, 0.1, 0.80])),
    ]

    print("\n================ REACHABILITY PROBES (home orientation) ================")
    print("(err = |commanded - achieved| wrist pos, meters; lower = reachable)")
    for label, lp, rp in probes:
        lp_a, rp_a, lx, rx, lerr, rerr = hold(lp, rp, n=60)
        print(f"\n[{label}]")
        print(f"  L cmd {np.round(lp,3)} -> ach {np.round(lp_a,3)}  err={lerr:.3f}  fingerX_world={np.round(lx,2)}")
        print(f"  R cmd {np.round(rp,3)} -> ach {np.round(rp_a,3)}  err={rerr:.3f}  fingerX_world={np.round(rx,2)}")

    print("\n================ ORIENTATION CANDIDATES for LEFT hand above object ================")
    # Find an orientation that points fingers (local +X) DOWN (-Z world) for a top grasp.
    cand = {
        "home_q": [0.7071, 0.0, 0.0, 0.7071],
        "rotX-90": [0.7071, -0.7071, 0.0, 0.0],
        "rotY+90": [0.7071, 0.0, 0.7071, 0.0],
        "rotY-90": [0.7071, 0.0, -0.7071, 0.0],
        "identity": [1.0, 0.0, 0.0, 0.0],
        "flipZ": [0.0, 0.0, 0.0, 1.0],
    }
    lp = np.array([-0.35, 0.45, 0.9])
    for name, q in cand.items():
        act = make_action(lp, q, HOME_R, HOME_Q)
        for _ in range(50):
            env.step(act)
        lq_a = wquat(li)
        m = quat_to_mat(lq_a)
        print(f"  {name:9s} cmdq={q} -> fingerX_world(local+X)={np.round(m[:,0],2)} localZ={np.round(m[:,2],2)} achpos={np.round(wpos(li),3)}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
