"""Shipped gsworld assets and their lookup.

    gsworld/assets/
    ├── models/<object>/                        the model library: one folder per object —
    │   ├── franka_robotiq/<name>.ply + _poses.json  a robot (labels in the PLY, bodies' scan poses in json)
    │   └── table/table.ply + table_poses.json       any object: same pair, one body
    └── scenes/<workcell>/scene.json            compositions: env + which models + camera

Robot models are keyed by the robobench robot registry name; ``robot_model_for(env)`` also falls
back to the Franka+Robotiq model for any Panda-armed robot (the arm links match, the hand stays
raytraced). robobench itself knows nothing about these files; producing models is external to the repo.
"""
from __future__ import annotations

from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"


def robot_model(name: str) -> tuple[Path, Path] | None:
    """(model .ply, its _poses.json) for a robobench robot name, or None if not shipped."""
    ply, poses = ASSETS / "models" / name / f"{name}.ply", ASSETS / "models" / name / f"{name}_poses.json"
    return (ply, poses) if ply.exists() and poses.exists() else None


def robot_model_for(env) -> tuple[Path, Path] | None:
    """The model for a built env's robot: by registry name, else the Franka+Robotiq model for any
    Panda-armed robot (partial: arm only)."""
    from robobench.core import ROBOTS

    for name in ROBOTS.list():
        if ROBOTS.get(name) is type(env.robot):
            m = robot_model(name)
            if m is not None:
                return m
    if "panda_link0" in env.robot.articulation.body_names:
        return robot_model("franka_robotiq")
    return None


def scene(name: str) -> Path | None:
    """Path to a shipped scene config, or None."""
    p = ASSETS / "scenes" / name / "scene.json"
    return p if p.exists() else None


def models() -> list[str]:
    """Names of the shipped models (one folder each under assets/models/)."""
    root = ASSETS / "models"
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []


def scenes() -> list[str]:
    """Names of the shipped scene configs (one folder each under assets/scenes/)."""
    root = ASSETS / "scenes"
    return sorted(p.parent.name for p in root.glob("*/scene.json")) if root.exists() else []
