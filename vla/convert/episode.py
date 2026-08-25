"""The canonical episode — the uniform form every raw source converts through.

Sim or real, an Episode is the same object: a joint-space timeline (achieved arm
joint positions + a normalized gripper closedness), per-view video handles, a
language task, and provenance. Conventions (conventions.py) are pure projections
of this form; convert.py streams it into a LeRobotDataset. A real-robot log
becomes co-trainable by writing ONE reader that returns this same object —
nothing downstream changes.

Gripper convention: CLOSEDNESS in [0, 1] (0 = fully open, 1 = fully closed),
normalized by each finger joint's travel. Unknown gripper joints must be added to
FINGER_TRAVEL — guessing a limit would silently corrupt every label built on it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# per-joint travel (rad or m) of known gripper joints; 0 assumed fully open
FINGER_TRAVEL = {
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}
_FINGER_MARKERS = ("finger", "gripper")


@dataclass
class Episode:
    ep_dir: Path
    origin: str                 # "sim" | "real"
    robot_type: str             # e.g. the sim preset, or a real-robot id
    rate_hz: float              # native timeline rate (one row per control latch)
    arm_joints: list[str]
    q: np.ndarray               # (T, n_arm) achieved arm joint positions
    gripper: np.ndarray         # (T,) closedness in [0, 1]
    videos: dict[str, Path]     # view -> mp4 path
    frame_indices: list[int]    # video frame k -> timeline row t
    fps: float                  # video frame rate
    size: tuple[int, int]       # (W, H) of every view
    task: str
    success: bool
    raw_action: np.ndarray | None    # (T, A) the recorded controller command (sim)
    objects: dict[str, np.ndarray] | None = None  # {name: (T, K, 13) pos+quat+vels} scene bodies
    controller: dict | None = None   # the recorded control law (meta.json block), if stamped
    #: COMMANDED arm joint targets (T, n_arm) — controller intent, recorded from the applied
    #: actuator targets (sim, position-mode arms). None on episodes recorded before the channel
    #: existed, on torque-mode arms (the stamp's law says), and on real logs without it.
    joint_target: np.ndarray | None = None
    #: COMMANDED gripper closedness (T,) — same normalization as `gripper` but UNCLAMPED:
    #: a squeeze is a target past the measured width, so values may exceed 1 (that overshoot
    #: IS the force intent; clamping it re-flattens what this channel exists to keep).
    gripper_target: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


def _goal_sentence(desc: str) -> str:
    tail = desc.split("Goal:", 1)
    return tail[1].strip() if len(tail) == 2 else desc.strip()


def _split_gripper(joint_names: list[str], joint_pos: np.ndarray, clip: bool = True):
    """(arm_joints, q, closedness): fingers found by name marker, normalized by travel.
    `clip=False` for COMMANDED targets: a squeeze command sits past the measured width, so
    commanded closedness legitimately exceeds 1 — that overshoot is the force intent."""
    fingers = [i for i, n in enumerate(joint_names) if any(m in n.lower() for m in _FINGER_MARKERS)]
    arm = [i for i in range(len(joint_names)) if i not in fingers]
    if not fingers:
        return [joint_names[i] for i in arm], joint_pos[:, arm], np.zeros(len(joint_pos), np.float32)
    travel = []
    for i in fingers:
        if joint_names[i] not in FINGER_TRAVEL:
            raise SystemExit(f"gripper joint '{joint_names[i]}' not in episode.FINGER_TRAVEL — "
                             f"add its travel before converting")
        travel.append(FINGER_TRAVEL[joint_names[i]])
    frac = joint_pos[:, fingers] / np.asarray(travel, np.float32)
    closed = 1.0 - (np.clip(frac.mean(axis=1), 0.0, 1.0) if clip else frac.mean(axis=1))
    return [joint_names[i] for i in arm], joint_pos[:, arm], closed.astype(np.float32)


def read_sim_episode(ep_dir: Path, cams: list[str] | None = None) -> Episode:
    """Build the canonical form from a rendered sim episode
    (`traj.npz` + `imgs/<view>.mp4` + `imgs/render_<view>.json` + `meta.json`)."""
    ep_dir = Path(ep_dir)
    meta = json.loads((ep_dir / "meta.json").read_text())
    views = sorted(cams) if cams else sorted(
        p.stem.removeprefix("render_") for p in (ep_dir / "imgs").glob("render_*.json"))
    if not views:
        raise SystemExit(f"{ep_dir}: no rendered views — run data_engine/scripts/render.py first")
    contracts = {v: json.loads((ep_dir / "imgs" / f"render_{v}.json").read_text()) for v in views}
    r0 = contracts[views[0]]
    for v, r in contracts.items():
        if (r["fps"], r["size"], r["joint_names"], r["frame_indices"]) != \
           (r0["fps"], r0["size"], r0["joint_names"], r0["frame_indices"]):
            raise SystemExit(f"{ep_dir}: view '{v}' was rendered separately from "
                             f"'{views[0]}' — re-render the views together")
    traj = np.load(ep_dir / "traj.npz")
    arm_joints, q, closed = _split_gripper(r0["joint_names"],
                                           traj["robot/joint_pos"].astype(np.float32))
    jt, gt = None, None
    if "robot/joint_target" in traj:  # commanded-target channel (same joint order as joint_pos)
        _, jt, gt = _split_gripper(r0["joint_names"],
                                   traj["robot/joint_target"].astype(np.float32), clip=False)
    return Episode(
        ep_dir=ep_dir, origin="sim", robot_type=meta.get("preset", "unknown"),
        rate_hz=1.0 / (meta["sim_dt"] * meta.get("decimation", 1)),
        arm_joints=arm_joints, q=q, gripper=closed,
        videos={v: ep_dir / "imgs" / contracts[v].get("video", f"{v}.mp4") for v in views},
        frame_indices=r0["frame_indices"], fps=r0["fps"], size=tuple(r0["size"]),
        task=_goal_sentence(r0.get("scene_description", "")),
        success=bool(meta.get("success")),
        raw_action=traj["action"].astype(np.float32) if "action" in traj else None,
        objects={k.removeprefix("scene/"): traj[k].astype(np.float32)
                 for k in traj.files if k.startswith("scene/")} or None,
        controller=meta.get("controller"), joint_target=jt, gripper_target=gt, meta=meta,
    )
