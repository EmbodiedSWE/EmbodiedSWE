"""Tick filters — post-process a projection before it is written (opt-in).

Idle filtering drops the DEAD ticks of scripted waits: a tick survives unless,
between it and the next dataset tick, the robot is static (arm + gripper), every
scene object is static (pose delta AND recorded velocities ~ zero), and the raw
command shows no intent. The intent guard is what keeps contact presses — the
robot and objects are static during the socket press, but the command keeps
pushing (|EE offset| >= CMD_GUARD), so those ticks are never dropped. Active
settling (an object still rocking, the arm damping out) has nonzero velocities
and is kept — only the converged tail of a wait goes.

The first tick of every idle run is kept so one "stay" remains as the
arrive-and-settle cue; the timeline simply skips the rest (timestamps stay
uniform in the dataset — BC does not consume wall-clock gaps).
"""

from __future__ import annotations

import numpy as np

from conventions import Projected
from episode import Episode

TOL_Q = 2e-3        # rad, max |dq| between ticks
TOL_GRIP = 2e-3     # closedness units
TOL_OBJ_POS = 5e-4  # m, max object position delta
TOL_OBJ_VEL = 1e-2  # m/s and rad/s, recorded object velocities
CMD_GUARD = 0.1     # keep any tick whose raw EE-offset intent exceeds this


def idle_keep_mask(ep: Episode, proj: Projected) -> np.ndarray:
    """(N,) bool — True = keep. Vectorized over the projection's ticks."""
    ts = np.asarray([ep.frame_indices[k] for k in proj.video_ticks])
    t2 = np.append(ts[1:], ts[-1])
    static = (np.abs(ep.q[t2] - ep.q[ts]).max(axis=1) < TOL_Q) \
        & (np.abs(ep.gripper[t2] - ep.gripper[ts]) < TOL_GRIP)
    if ep.objects:
        for arr in ep.objects.values():  # (T, K, 13): pos 0:3, quat 3:7, vels 7:13
            dpos = np.abs(arr[t2, :, 0:3] - arr[ts, :, 0:3]).max(axis=(1, 2))
            vel = np.abs(arr[ts, :, 7:13]).max(axis=(1, 2))
            static &= (dpos < TOL_OBJ_POS) & (vel < TOL_OBJ_VEL)
    if ep.raw_action is not None:  # intent guard: a static press is NOT idle
        static &= np.abs(ep.raw_action[ts, :6]).max(axis=1) < CMD_GUARD
    keep = ~static
    keep[np.flatnonzero(static[1:] & ~static[:-1]) + 1] = True  # first tick of each run
    keep[0] = True
    return keep


def apply_mask(proj: Projected, keep: np.ndarray) -> Projected:
    return Projected(
        video_ticks=[k for k, m in zip(proj.video_ticks, keep) if m],
        state=proj.state[keep], action=proj.action[keep],
        state_names=proj.state_names, action_names=proj.action_names,
        state_parts=proj.state_parts, action_parts=proj.action_parts,
        raw_command=None if proj.raw_command is None else proj.raw_command[keep],
    )
