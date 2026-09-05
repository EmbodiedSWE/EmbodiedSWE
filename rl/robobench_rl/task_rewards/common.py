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
    off = PANDA_PINCH_OFFSET if offset is None else offset
    off = getattr(env.scene, "GRASP_PINCH_OFFSET", off)
    z = torch.tensor([0.0, 0.0, float(off)], device=pos.device).expand(pos.shape[0], 3)
    return pos + quat_apply(quat, z)


def local_axis(quat: torch.Tensor, axis: int) -> torch.Tensor:
    from isaaclab.utils.math import quat_apply

    e = torch.zeros(3, device=quat.device)
    e[axis] = 1.0
    return quat_apply(quat, e.expand(quat.shape[0], 3))
