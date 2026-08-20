"""Label conventions — pure projections Episode -> (state, action, named parts).

Every convention resamples the SAME canonical timeline to `rate_hz` (which must
divide the episode's video fps): the state is taken at tick t, the action is the
generic tracking target for t -> t + 1/rate. At a fixed rate, position and
velocity carry the same information (v = (q_next - q) * rate), so these are
dialects, not different data:

    joint_pos   state [q, grip], action [q_next, grip_next]      (GR00T / LeRobot school)
    joint_vel   state [q, grip], action [(q_next-q)*rate, grip_next]  (pi-DROID: 15 Hz)
    raw_cmd     state [q, grip], action = the recorded controller command, verbatim
                (the sim-only matched-controller benchmark arm; needs ep.raw_action)

Whatever the convention, sim episodes also carry the verbatim command as an extra
`raw_command` column — force intent (press = sustained offset) survives every
projection, and later relabelings never need the raw files again.

Named parts (`{name: (start, end)}`) describe the concatenated vectors; convert.py
turns them into GR00T's modality.json and the dataset's feature names.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from episode import Episode


@dataclass
class Projected:
    video_ticks: list[int]           # video frame k per dataset tick
    state: np.ndarray                # (N, S) float32
    action: np.ndarray               # (N, A) float32
    state_names: list[str]
    action_names: list[str]
    state_parts: dict[str, tuple[int, int]]
    action_parts: dict[str, tuple[int, int]]
    raw_command: np.ndarray | None   # (N, Araw) the verbatim command at each tick


def _ticks(ep: Episode, rate_hz: float) -> tuple[list[int], list[int], list[int]]:
    """(video frame k, timeline row t, next-tick row t2) per dataset tick."""
    stride = ep.fps / rate_hz
    if abs(stride - round(stride)) > 1e-6:
        raise SystemExit(f"rate {rate_hz} Hz does not divide the video fps {ep.fps} — "
                         f"pick a divisor")
    stride = round(stride)
    ks = list(range(0, len(ep.frame_indices), stride))
    ts = [ep.frame_indices[k] for k in ks]
    t_next = ts[1:] + [ep.frame_indices[-1]]  # last tick holds (action = stay)
    return ks, ts, t_next


def _state(ep: Episode, ts: list[int]):
    s = np.concatenate([ep.q[ts], ep.gripper[ts, None]], axis=1)
    names = [*ep.arm_joints, "gripper"]
    parts = {"arm_qpos": (0, len(ep.arm_joints)),
             "gripper_qpos": (len(ep.arm_joints), len(ep.arm_joints) + 1)}
    return s.astype(np.float32), names, parts


def joint_pos(ep: Episode, rate_hz: float) -> Projected:
    ks, ts, t2 = _ticks(ep, rate_hz)
    state, s_names, s_parts = _state(ep, ts)
    action = np.concatenate([ep.q[t2], ep.gripper[t2, None]], axis=1).astype(np.float32)
    return Projected(ks, state, action, s_names, [*ep.arm_joints, "gripper"],
                     s_parts, dict(s_parts),
                     ep.raw_action[ts] if ep.raw_action is not None else None)


def joint_vel(ep: Episode, rate_hz: float) -> Projected:
    ks, ts, t2 = _ticks(ep, rate_hz)
    state, s_names, s_parts = _state(ep, ts)
    vel = (ep.q[t2] - ep.q[ts]) * rate_hz
    action = np.concatenate([vel, ep.gripper[t2, None]], axis=1).astype(np.float32)
    n = len(ep.arm_joints)
    return Projected(ks, state, action, s_names,
                     [f"{j}_vel" for j in ep.arm_joints] + ["gripper"],
                     s_parts, {"arm_qvel": (0, n), "gripper_qpos": (n, n + 1)},
                     ep.raw_action[ts] if ep.raw_action is not None else None)


def raw_cmd(ep: Episode, rate_hz: float) -> Projected:
    if ep.raw_action is None:
        raise SystemExit(f"{ep.ep_dir}: raw_cmd needs the recorded command (sim episodes only)")
    ks, ts, _ = _ticks(ep, rate_hz)
    state, s_names, s_parts = _state(ep, ts)
    action = ep.raw_action[ts]
    return Projected(ks, state, action, s_names,
                     [f"cmd_{i}" for i in range(action.shape[1])],
                     s_parts, {"raw_cmd": (0, action.shape[1])}, None)


CONVENTIONS = {"joint_pos": joint_pos, "joint_vel": joint_vel, "raw_cmd": raw_cmd}
