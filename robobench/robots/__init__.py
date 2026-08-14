"""Reusable embodiments, shared across all suites.

Importing this package registers every robot into `robobench.core.ROBOTS`. The concrete control
pipeline (joint PD / EE-IK / OSC, gains, end-effector body, kinematics) lives in each robot here,
not in `BaseRobot`. Import-light where possible (isaaclab deferred), so registration is app-free.
"""

from .attached import (
    AttachedArmRobotCfg,
    CRX10iAL2F85Robot,
    Festo2F85Robot,
    Gen3N72F85Robot,
    Rizon42F85Robot,
    SawyerEGU50Robot,
    TM122F85Robot,
    Z1Lite6GRobot,
)
from .cobotta import CobottaPro1300Robot, CobottaPro1300RobotCfg
from .franka import FrankaRobot, FrankaRobotCfg
from .g1 import G1Robot, G1RobotCfg
from .gr1t2 import GR1T2Robot, GR1T2RobotCfg
from .jaco2 import Jaco2N7Robot, Jaco2N7RobotCfg
from .multi import Aloha, AlohaCfg, BimanualFranka, BimanualFrankaCfg, BimanualPiper, BimanualPiperCfg, MultiRobot, MultiRobotCfg
from .null_robot import NullRobot
from .piper import PiperRobot, PiperRobotCfg
from .wxai import WxaiRobot, WxaiRobotCfg
from .xarm7 import XArm7Robot, XArm7RobotCfg

__all__ = [
    "NullRobot",
    "AttachedArmRobotCfg",
    "Aloha",
    "AlohaCfg",
    "BimanualFranka",
    "BimanualFrankaCfg",
    "BimanualPiper",
    "BimanualPiperCfg",
    "CobottaPro1300Robot",
    "CobottaPro1300RobotCfg",
    "FrankaRobot",
    "FrankaRobotCfg",
    "G1Robot",
    "G1RobotCfg",
    "GR1T2Robot",
    "GR1T2RobotCfg",
    "Jaco2N7Robot",
    "Jaco2N7RobotCfg",
    "MultiRobot",
    "MultiRobotCfg",
    "PiperRobot",
    "PiperRobotCfg",
    "WxaiRobot",
    "WxaiRobotCfg",
    "XArm7Robot",
    "XArm7RobotCfg",
]
