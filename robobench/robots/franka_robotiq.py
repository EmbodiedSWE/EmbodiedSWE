"""FrankaRobotiqRobot — a Franka Emika Panda arm + Robotiq 2F-85 gripper.

The same arm as `FrankaRobot` with a one-joint parallelogram-linkage gripper instead of the panda
hand, spawned from the vendored `assets/franka_robotiq/franka_robotiq.usd`. It subclasses
`FrankaRobot`, so the arm control modes are inherited unchanged (the task-space laws only need
`EE_BODY` / `ARM_JOINTS` and the cfg gains):

  - "osc" / "impedance" -> torque task-space control; action = 6 EE pose deltas + 1 gripper = 7.
  - "diff_ik"           -> differential IK to joint position targets; same 7-D action.
  - "joint"             -> direct joint position targets; action = 7 arm + 1 gripper = 8.
  ("pink_ik" is not offered: the vendored kinematics URDF describes the panda hand.)

The gripper has ONE actuated joint (`finger_joint`: 0 = open ~85 mm -> pi/4 rad = closed); its
five follower joints ride `PhysxMimicJointAPI` constraints authored in the asset (reference =
finger_joint, gearing +/-1), so they get a zero-gain actuator group and are never commanded.
Heavy imports are deferred so registration stays app-free. (A Gaussian-splat model of this rig
ships with the optional `gsworld` package, which renders it photoreal.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from robobench.core import ROBOTS
from robobench.robots.franka import FrankaRobot, FrankaRobotCfg

_ASSETS = Path(__file__).resolve().parent / "assets" / "franka_robotiq"


@dataclass
class FrankaRobotiqRobotCfg(FrankaRobotCfg):
    """`FrankaRobotCfg` dials (placement, arm gains, nullspace posture, task gains) plus the Robotiq
    drive. Differences from the panda-hand Franka: the gripper PD acts on one revolute drive joint
    [N*m/rad], and the arm is spawned weightless (PhysX `disable_arm_gravity`) by default — the
    task-space torque law has no gravity term and the real arm gravity-compensates."""

    gripper_stiffness: float = 100.0  # `finger_joint` PD [N*m/rad, N*m*s/rad] (authored 100 / 2e-4)
    gripper_damping: float = 10.0
    gripper_effort_limit: float | None = None  # [N*m]; None -> the authored 16.5
    default_gripper_pos: float = 0.0  # `finger_joint` home: 0 = open. MUST stay at the followers'
    # authored rest — a mimic linkage spawned constraint-violating detonates. Close by command (pi/4).
    disable_arm_gravity: bool = True
    franka_robotiq_usd: str = ""  # "" -> the vendored robots/assets/franka_robotiq/franka_robotiq.usd

    def __post_init__(self) -> None:
        super().__post_init__()
        self.franka_robotiq_usd = self.franka_robotiq_usd or str(_ASSETS / "franka_robotiq.usd")


@ROBOTS.register("franka_robotiq")
class FrankaRobotiqRobot(FrankaRobot):
    """Franka Panda arm + Robotiq 2F-85 gripper. Controllers, binding and reset come from `FrankaRobot`;
    only the articulation (a different USD and gripper actuator groups) and the gripper conventions
    differ."""

    control_modes: tuple[str, ...] = ("osc", "impedance", "diff_ik", "joint")
    cfg: FrankaRobotiqRobotCfg

    GRIPPER_JOINTS: tuple[str, ...] = ("finger_joint",)  # the DRIVEN joint only
    #: PhysxMimicJointAPI followers of finger_joint (gearing -1, -1, +1, +1, +1) — never commanded.
    PASSIVE_JOINTS: tuple[str, ...] = (
        "right_outer_knuckle_joint",
        "right_inner_finger_joint",
        "right_inner_finger_knuckle_joint",
        "left_inner_finger_knuckle_joint",
        "left_inner_finger_joint",
    )
    EE_BODY: str = "base_link"  # the Robotiq base, welded to panda_link8 — the task-space control frame
    GRIPPER_OPEN: float = 0.0
    GRIPPER_CLOSED: float = math.pi / 4

    def __init__(self, cfg: FrankaRobotiqRobotCfg | None = None) -> None:
        super().__init__(cfg or FrankaRobotiqRobotCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """The Franka+Robotiq articulation, authored from scratch over the vendored USD. Arm groups mirror
        `FRANKA_PANDA_HIGH_PD_CFG` (shoulder/forearm split, 87/12 N*m, 2.175/2.61 rad/s) with the same
        per-mode gain handling as `FrankaRobot` (torque modes zero the arm PD); the gripper drive is PD;
        the mimic followers stay passive (zero gains)."""
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg

        c = self.cfg
        torque_mode = self.control_mode in ("impedance", "osc")
        arm_kp = 0.0 if torque_mode else c.arm_stiffness
        arm_kd = 0.0 if torque_mode else c.arm_damping
        actuators: dict[str, Any] = {
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"], stiffness=arm_kp, damping=arm_kd,
                effort_limit_sim=87.0 if c.arm_effort_limit is None else c.arm_effort_limit,
                velocity_limit_sim=2.175,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"], stiffness=arm_kp, damping=arm_kd,
                effort_limit_sim=12.0 if c.arm_effort_limit is None else c.arm_effort_limit,
                velocity_limit_sim=2.61,
            ),
            "robotiq_drive": ImplicitActuatorCfg(
                joint_names_expr=list(self.GRIPPER_JOINTS), stiffness=c.gripper_stiffness,
                damping=c.gripper_damping, effort_limit_sim=c.gripper_effort_limit,  # None -> authored 16.5
            ),
            "robotiq_passive": ImplicitActuatorCfg(
                joint_names_expr=list(self.PASSIVE_JOINTS), stiffness=0.0, damping=0.0,  # mimic-driven
            ),
        }
        return {
            self.name: ArticulationCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{self.prim_name}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.franka_robotiq_usd,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=bool(c.disable_arm_gravity), max_depenetration_velocity=5.0),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False,  # the linkage fingers interleave
                        solver_position_iteration_count=8,  # raise (up to 64) if the linkage chatters
                        solver_velocity_iteration_count=0,
                        fix_root_link=c.fixed_base,  # toggles the asset's own world joint
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=c.base_pos, rot=c.base_rot,
                    joint_pos={
                        **{f"panda_joint{i + 1}": float(q) for i, q in enumerate(c.default_dof_pos)},
                        **{gj: c.default_gripper_pos for gj in self.GRIPPER_JOINTS},
                        **{pj: 0.0 for pj in self.PASSIVE_JOINTS},  # the followers' authored rest
                    },
                ),
                actuators=actuators,
                soft_joint_pos_limit_factor=1.0,
            )
        }

    # NOTE: build_controller / on_bind / reset / apply_action are inherited from FrankaRobot.

    # ----- gripper convenience ----------------------------------------------------------------
    def gripper_open_closed(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Position targets for the gripper controller leaf: (open, closed), shape (1, 1)."""
        return torch.tensor([[self.GRIPPER_OPEN]]), torch.tensor([[self.GRIPPER_CLOSED]])

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        arm = {"osc": "6 EE pose deltas (operational-space torque control)",
               "impedance": "6 EE pose deltas (task-space impedance)",
               "diff_ik": "6 EE pose deltas (differential IK)",
               "joint": "7 arm joint position targets"}[self.control_mode]
        return (
            f"A Franka Emika Panda arm with a Robotiq 2F-85 gripper, fixed to the table. Control mode "
            f"'{self.control_mode}': {arm}, plus 1 gripper joint position target (finger_joint, rad: 0 = open "
            f"~85 mm, {math.pi / 4:.3f} = closed; the finger linkage follows by mimic constraint). "
            f"Action dim {self.action_dim}."
        )
