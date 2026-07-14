"""Reusable embodiments, shared across all suites.

Importing this package registers every robot into `robobench.core.ROBOTS`. The concrete control
pipeline (joint PD / EE-IK / OSC, gains, end-effector body, kinematics) lives in each robot here,
not in `BaseRobot`. Import-light where possible (isaaclab deferred), so registration is app-free.
"""

from .franka import FrankaRobot, FrankaRobotCfg
from .g1 import G1Robot, G1RobotCfg
from .gr1t2 import GR1T2Robot, GR1T2RobotCfg
from .multi import MultiRobot, MultiRobotCfg
from .null_robot import NullRobot

__all__ = [
    "NullRobot",
    "FrankaRobot",
    "FrankaRobotCfg",
    "G1Robot",
    "G1RobotCfg",
    "GR1T2Robot",
    "GR1T2RobotCfg",
    "MultiRobot",
    "MultiRobotCfg",
]
