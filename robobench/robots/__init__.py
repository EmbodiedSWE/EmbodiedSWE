"""Reusable embodiments, shared across all suites.

Importing this package registers every robot into `robobench.core.ROBOTS`. The concrete control
pipeline (joint PD / EE-IK / OSC, gains, end-effector body, kinematics) lives in each robot here,
not in `BaseRobot`. Import-light where possible (isaaclab deferred), so registration is app-free.
"""

from .attached import (
    AttachedArmRobotCfg,
    FestoPandaRobot,
    Gen3N7PandaRobot,
    Rizon4PandaRobot,
    SawyerEGK25Robot,
    SawyerPandaRobot,
    Z1Lite6GRobot,
)
from .cobotta import CobottaPro1300Robot, CobottaPro1300RobotCfg
from .franka import FrankaRobot, FrankaRobotCfg
from .franka_robotiq import FrankaRobotiqRobot, FrankaRobotiqRobotCfg
from .g1 import G1Robot, G1RobotCfg
from .gr1t2 import GR1T2Robot, GR1T2RobotCfg
from .jaco2 import Jaco2N7Robot, Jaco2N7RobotCfg
from .multi import Aloha, AlohaCfg, BimanualFranka, BimanualFrankaCfg, BimanualPiper, BimanualPiperCfg, MultiRobot, MultiRobotCfg
from .null_robot import NullRobot
from .piper import PiperRobot, PiperRobotCfg
from .wx250s import Wx250sRobot, Wx250sRobotCfg
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

    "FrankaRobotiqRobot",

    "FrankaRobotiqRobotCfg",
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
    "Wx250sRobot",
    "Wx250sRobotCfg",
    "WxaiRobot",
    "WxaiRobotCfg",
    "XArm7Robot",
    "XArm7RobotCfg",
]
