from __future__ import annotations

from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class PsihandRightCfg(RobotCfg):
    """Configuration for PsiHand Right (Psi-SynHand).

    NOTE: This robot has known compatibility issues with IsaacGym:
    - URDF mode: Joint positions become NaN, fingers not visible
    - USD mode: DOF force tensor initialization fails
    Recommended to use with MuJoCo, Genesis, or other simulators instead.
    """

    name: str = "psihand_right"
    num_joints: int = 21
    fix_base_link: bool = True
    urdf_path: str = "roboverse_data/robots/psihand/urdf/psihand_right.urdf"
    mjcf_path: str = "roboverse_data/robots/psihand/mjcf/psihand_right.xml"
    usd_path: str = "roboverse_data/robots/psihand/usd/psihand_right.usd"
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False

    actuators: dict[str, BaseActuatorCfg] = {
        # Thumb (5 joints)
        "joint_1_1": BaseActuatorCfg(stiffness=1000, damping=100),
        "joint_1_2": BaseActuatorCfg(stiffness=800, damping=80),
        "joint_1_3": BaseActuatorCfg(stiffness=600, damping=60),
        "joint_1_4": BaseActuatorCfg(stiffness=500, damping=50),
        "joint_1_5": BaseActuatorCfg(stiffness=500, damping=50),
        # Index finger (4 joints)
        "joint_2_1": BaseActuatorCfg(stiffness=800, damping=80),
        "joint_2_2": BaseActuatorCfg(stiffness=700, damping=70),
        "joint_2_3": BaseActuatorCfg(stiffness=600, damping=60),
        "joint_2_4": BaseActuatorCfg(stiffness=500, damping=50),
        # Middle finger (4 joints)
        "joint_3_1": BaseActuatorCfg(stiffness=800, damping=80),
        "joint_3_2": BaseActuatorCfg(stiffness=700, damping=70),
        "joint_3_3": BaseActuatorCfg(stiffness=600, damping=60),
        "joint_3_4": BaseActuatorCfg(stiffness=500, damping=50),
        # Ring finger (4 joints)
        "joint_4_1": BaseActuatorCfg(stiffness=800, damping=80),
        "joint_4_2": BaseActuatorCfg(stiffness=700, damping=70),
        "joint_4_3": BaseActuatorCfg(stiffness=600, damping=60),
        "joint_4_4": BaseActuatorCfg(stiffness=500, damping=50),
        # Pinky finger (4 joints)
        "joint_5_1": BaseActuatorCfg(stiffness=800, damping=80),
        "joint_5_2": BaseActuatorCfg(stiffness=700, damping=70),
        "joint_5_3": BaseActuatorCfg(stiffness=600, damping=60),
        "joint_5_4": BaseActuatorCfg(stiffness=500, damping=50),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        # Thumb
        "joint_1_1": (-0.205, 0.75),
        "joint_1_2": (0.0, 1.57),
        "joint_1_3": (0.0, 0.628),
        "joint_1_4": (0.0, 1.05),
        "joint_1_5": (0.0, 1.1),
        # Index finger
        "joint_2_1": (-0.349, 0.349),
        "joint_2_2": (0.0, 1.57),
        "joint_2_3": (0.0, 1.27),
        "joint_2_4": (0.0, 1.29),
        # Middle finger
        "joint_3_1": (-0.349, 0.349),
        "joint_3_2": (0.0, 1.57),
        "joint_3_3": (0.0, 1.27),
        "joint_3_4": (0.0, 1.29),
        # Ring finger
        "joint_4_1": (-0.349, 0.349),
        "joint_4_2": (0.0, 1.57),
        "joint_4_3": (0.0, 1.27),
        "joint_4_4": (0.0, 1.29),
        # Pinky finger
        "joint_5_1": (-0.349, 0.349),
        "joint_5_2": (0.0, 1.57),
        "joint_5_3": (0.0, 1.27),
        "joint_5_4": (0.0, 1.29),
    }

    default_joint_positions: dict[str, float] = {
        # Thumb - slightly opened
        "joint_1_1": 0.0,
        "joint_1_2": 0.2,
        "joint_1_3": 0.1,
        "joint_1_4": 0.1,
        "joint_1_5": 0.1,
        # Index finger - slightly opened
        "joint_2_1": 0.0,
        "joint_2_2": 0.2,
        "joint_2_3": 0.2,
        "joint_2_4": 0.2,
        # Middle finger - slightly opened
        "joint_3_1": 0.0,
        "joint_3_2": 0.2,
        "joint_3_3": 0.2,
        "joint_3_4": 0.2,
        # Ring finger - slightly opened
        "joint_4_1": 0.0,
        "joint_4_2": 0.2,
        "joint_4_3": 0.2,
        "joint_4_4": 0.2,
        # Pinky finger - slightly opened
        "joint_5_1": 0.0,
        "joint_5_2": 0.2,
        "joint_5_3": 0.2,
        "joint_5_4": 0.2,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        # All joints use position control
        "joint_1_1": "position",
        "joint_1_2": "position",
        "joint_1_3": "position",
        "joint_1_4": "position",
        "joint_1_5": "position",
        "joint_2_1": "position",
        "joint_2_2": "position",
        "joint_2_3": "position",
        "joint_2_4": "position",
        "joint_3_1": "position",
        "joint_3_2": "position",
        "joint_3_3": "position",
        "joint_3_4": "position",
        "joint_4_1": "position",
        "joint_4_2": "position",
        "joint_4_3": "position",
        "joint_4_4": "position",
        "joint_5_1": "position",
        "joint_5_2": "position",
        "joint_5_3": "position",
        "joint_5_4": "position",
    }
