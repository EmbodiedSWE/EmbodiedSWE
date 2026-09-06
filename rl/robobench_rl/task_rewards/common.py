"""Shared geometry helpers for the shaped rewards (public robot/scene data only)."""
from __future__ import annotations

import torch

PANDA_PINCH_OFFSET = 0.1034  # hand origin -> finger-pad centre along the hand's approach (+z) axis


def kernel(d: torch.Tensor, scale: float) -> torch.Tensor:
    """1 at d=0, ~0 beyond a few `scale`: 1 - tanh(d / scale)."""
    return 1.0 - torch.tanh(d / scale)


def reach_kernel(d: torch.Tensor) -> torch.Tensor:
    """Two-scale reach shaping (Factory-style coarse + fine): the coarse 0.3 m kernel still has
    gradient at the ~0.4 m the hand starts from, the fine 0.05 m kernel rewards the last few cm.
    A single 5 cm kernel reads 0.000 from the home pose (measured), i.e. no signal to approach."""
    return 0.5 * kernel(d, 0.30) + 0.5 * kernel(d, 0.05)


def hand_pose(env) -> tuple[torch.Tensor, torch.Tensor]:
    """World pose of the robot's control-frame body (EE_BODY), (n,3) and (n,4) wxyz."""
    robot = env.robot
    art = robot.articulation
    idx = art.data.body_names.index(robot.EE_BODY)
    return art.data.body_pos_w[:, idx], art.data.body_quat_w[:, idx]


def pinch_point(env, offset: float | None = None) -> torch.Tensor:
    """Finger-pad centre in world: hand origin pushed along the hand's local +z."""
    from isaaclab.utils.math import quat_apply

    pos, quat = hand_pose(env)
    off = offset if offset is not None else getattr(env.scene, "GRASP_PINCH_OFFSET", PANDA_PINCH_OFFSET)
    z = torch.tensor([0.0, 0.0, float(off)], device=pos.device).expand(pos.shape[0], 3)
    return pos + quat_apply(quat, z)


def local_axis(quat: torch.Tensor, axis: int) -> torch.Tensor:
    from isaaclab.utils.math import quat_apply

    e = torch.zeros(3, device=quat.device)
    e[axis] = 1.0
    return quat_apply(quat, e.expand(quat.shape[0], 3))



def finger_positions(env) -> torch.Tensor:
    """Per-finger joint positions (n, F) of the robot's declared GRIPPER_JOINTS (panda: 2 x [0, 0.04] m)."""
    robot = env.robot
    art = getattr(robot, "articulation", None)
    pats = getattr(robot, "GRIPPER_JOINTS", None)
    if art is None or not pats:
        return torch.zeros(env.num_envs, 1, device=env.device)
    ids = art.find_joints(list(pats))[0]
    return art.data.joint_pos[:, ids]


def grasp_term_fingers(pinch_dist: torch.Tensor, fingers: torch.Tensor, part_width: float, near: float = 0.03,
                       open_each: float = 0.04) -> torch.Tensor:
    """Like `grasp_term` but PER FINGER: every finger must have closed onto the part (each joint from
    `open_each` down to part_width / F) and none may have shut past it. With independent finger
    actions the summed-aperture version was satisfied by ONE finger closed and one open — no pinch
    at all (measured on bulb: grasp 0.25, lift 0.00, probe showed aperture = half the command)."""
    F = fingers.shape[1]
    each = part_width / F
    closed = ((open_each - fingers) / max(open_each - each, 1e-3)).clamp(0.0, 1.0).amin(dim=1)
    not_through = ((fingers - (each - 0.003)) / 0.003).clamp(0.0, 1.0).amin(dim=1)
    return kernel(pinch_dist, near) * closed * not_through



# hand frame keypoints: origin, +z (approach, toward the fingertips), +y (the panda's finger-opening
# axis — the fingers slide along hand-local Y; the grasp-weld contract's pinch_axis is (0, 1, 0))
KEYPOINT_OFFSETS = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.05), (0.0, 0.05, 0.0))


def hand_target_from_axes(grasp_point: torch.Tensor, approach_w: torch.Tensor, open_w: torch.Tensor,
                          pinch_offset: float = PANDA_PINCH_OFFSET) -> tuple[torch.Tensor, torch.Tensor]:
    """Nominal hand pose (pos (n,3), quat wxyz (n,4)) that puts the pinch point on `grasp_point` with the
    hand's approach axis (+z) along `approach_w` and its finger-opening axis (+y, panda) along `open_w`
    (both world, (n,3); `open_w` is re-orthogonalised against the approach)."""
    from isaaclab.utils.math import quat_from_matrix

    z = approach_w / approach_w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    y = open_w - (open_w * z).sum(-1, keepdim=True) * z
    y = y / y.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    x = torch.cross(y, z, dim=-1)  # right-handed: x = y × z
    R = torch.stack([x, y, z], dim=-1)  # columns = hand axes in world
    quat = quat_from_matrix(R)
    pos = grasp_point - z * pinch_offset
    return pos, quat


def keypoint_distance(env, target_pos: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
    """Mean distance (n,) between the hand's keypoints (origin + approach + opening axis points) and the
    same keypoints of a target hand pose — position AND orientation error in one metre-valued number
    (Factory-style). Sign-ambiguous axes are not special-cased: the target should already pick the
    nearer of two equivalent grasps if the part is symmetric."""
    from isaaclab.utils.math import quat_apply

    pos, quat = hand_pose(env)
    n = pos.shape[0]
    offs = torch.tensor(KEYPOINT_OFFSETS, device=pos.device)  # (3,3)
    K = offs.shape[0]
    q_h = quat[:, None, :].expand(n, K, 4).reshape(-1, 4)
    q_t = target_quat[:, None, :].expand(n, K, 4).reshape(-1, 4)
    o = offs[None].expand(n, K, 3).reshape(-1, 3)
    kp_h = pos[:, None, :] + quat_apply(q_h, o).reshape(n, K, 3)
    kp_t = target_pos[:, None, :] + quat_apply(q_t, o).reshape(n, K, 3)
    return (kp_h - kp_t).norm(dim=-1).mean(dim=1)
