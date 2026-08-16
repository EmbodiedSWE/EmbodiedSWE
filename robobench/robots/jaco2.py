"""Jaco2N7Robot — a Kinova Jaco2 7-DOF arm with its integrated 3-finger hand, for table-top
manipulation.

A fixed-base 7-DOF assistive-class arm (~0.90 m reach, carbon-fiber links) whose hand is part of
the articulation: three curved fingers, each a proximal + tip joint (0.2 rad = open, ~1.2 rad =
closed — all six actuated; the real hand's underactuation is emulated by commanding tips with the
proximals). Built over `isaaclab_assets`' `KINOVA_JACO2_N7S300_CFG` preset (actuator ratings +
home pose), with the USD swapped to the vendored copy (`assets/jaco2_n7/`, from the Isaac Sim 5.1
asset library's `Robots/Kinova/Jaco2/J2N7S300`, unmodified).

Control modes mirror `FrankaRobot` (the controllers are embodiment-generic; only names/gains are
per-robot):

  - "osc"       -> arm by operational-space control (torque); action = 6 EE pose deltas + 6 finger = 12.
  - "impedance" -> arm by Jacobian-transpose task-space impedance (torque); same 12-D action.
  - "joint"     -> arm by direct joint position targets; action = 7 arm + 6 finger = 13.

The finger order in the gripper action is [finger_1, finger_2, finger_3, tip_1, tip_2, tip_3].
Heavy imports are deferred so registration stays app-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from robobench.controllers import (
    CompositeController,
    JointController,
    JointControllerCfg,
    OperationalSpaceController,
    TaskSpaceControllerCfg,
    TaskSpaceImpedanceController,
)
from robobench.core import ROBOTS, BaseRobot, BaseRobotCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from robobench.core import BaseEnv


@dataclass
class Jaco2N7RobotCfg(BaseRobotCfg):
    """Config for `Jaco2N7Robot`. `control_mode` is inherited from `BaseRobotCfg` ("" -> first
    declared, i.e. "osc"). Arm PD gains are used in "joint" mode only (the torque modes zero them);
    the fingers are always position-controlled."""

    fixed_base: bool = True  # weld the base to the world (a table-mounted arm)
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)  # base at the table level
    base_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz; faces +x
    # Arm position-PD gains — used in "joint" mode only. The preset's 40/15 stiffness is too soft
    # for tracking against gravity; these mirror the sibling arms' joint-mode gains.
    arm_stiffness: float = 400.0
    arm_damping: float = 80.0
    # Arm actuator effort cap [N*m]; None -> the preset's per-joint ratings (80/80/40/40/20/20/20).
    arm_effort_limit: float | None = None
    # Finger PD gains (always position-controlled). The preset ships 1.2/0.01 (a limp grip good
    # only for visuals); the default here holds a commanded pinch against light contact. The
    # benchmark's weld-on-closure contract carries the load once a grasp verifies.
    gripper_stiffness: float = 20.0
    gripper_damping: float = 1.0
    gripper_effort_limit: float | None = 10.0  # preset's 2.0 barely curls a finger
    # Home posture of the 7 arm joints — the isaaclab_assets preset's home (joints 2/4 have limits
    # [30, 330] deg, so zeros are out of range). Retune per task.
    default_dof_pos: tuple[float, ...] = (0.0, 2.76, 0.0, 2.0, 2.0, 0.0, 0.0)
    # Finger home (rad, applied to all three proximal + all three tip joints): 0.2 = open, ~1.2 = closed.
    default_gripper_pos: float = 0.2
    # Posture the task-space nullspace pulls toward; () -> use default_dof_pos.
    nullspace_dof_pos: tuple[float, ...] = ()
    # Arm gravity compensation (PhysX: spawn the robot's bodies with gravity disabled — the
    # analog of the franka cfg's Newton-only `gravcomp`). The real Jaco2 is an assistive arm
    # whose controller actively gravity-compensates; without it, gravity-blind torque-mode
    # control cannot support this light arm's own weight within its effort ratings.
    gravity_compensation: bool = False
    jaco2_usd: str = ""  # "" -> the vendored robots/assets/jaco2_n7/j2n7s300_instanceable.usd

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parent / "assets" / "jaco2_n7"
        self.jaco2_usd = self.jaco2_usd or str(assets / "j2n7s300_instanceable.usd")


@ROBOTS.register("jaco2_n7")
class Jaco2N7Robot(BaseRobot):
    """Kinova Jaco2 7-DOF arm + integrated 3-finger hand. `apply_action` delegates to the controller
    for `control_mode`; the 6 finger joints are always direct position targets, the 7 arm joints by
    the mode's arm controller (torque-mode task-space control, or position JointController)."""

    # Scene grasp-weld contract keys: the curling 3-finger hand has no aperture window — a
    # caging WRAP certified by the proximal stall (a free-air curl reaches its command; only
    # pressing a part stops the fingers early) + thumb/pair flanking the grip band. Hand axes
    # in the link_7 frame.
    GRASP_IFACE = dict(
        hand_body="j2n7s300_link_7",
        finger_joints="j2n7s300_joint_finger.*",  # proximals then tips (articulation order)
        approach=(0.001, -0.121, -0.993), pinch_offset=0.188,
        closure=("wrap", 0.85, 1.30, 0.12),  # prox stall band (rad) + squeeze margin
        pad_bodies=("j2n7s300_link_finger_tip_1", "j2n7s300_link_finger_tip_2",
                    "j2n7s300_link_finger_tip_3"),
        pinch_axis=(0.003, 0.993, -0.121),
        wrap_off=(0.025, 0.050),  # site closure window -> tip-origin proxy band
        prox_release=0.15,
        band_dist=0.035,  # a 3-finger squeeze displaces the wrist, riding the pinch centre
        # off the band; the flank + stall gates still anchor the grasp to the part
    )

    control_modes: tuple[str, ...] = ("osc", "impedance", "joint")  # osc default

    cfg: Jaco2N7RobotCfg

    ARM_JOINTS: tuple[str, ...] = ("j2n7s300_joint_[1-7]",)
    GRIPPER_JOINTS: tuple[str, ...] = (  # proximals first, then tips (the action order)
        "j2n7s300_joint_finger_[1-3]",
        "j2n7s300_joint_finger_tip_[1-3]",
    )
    # The OSC control frame. NOT the asset's `j2n7s300_end_effector` tool frame: that body is
    # massless (1e-7 kg) and PhysX merges massless fixed children, leaving its jacobian row
    # unreliable — the torque law computes near-zero commands and the arm freezes.
    EE_BODY: str = "j2n7s300_link_7"

    # Action/target rate (s) — same scheme as FrankaRobot.
    TORQUE_CONTROL_DT: float = 1.0 / 15.0
    JOINT_CONTROL_DT: float = 0.02

    def __init__(self, cfg: Jaco2N7RobotCfg | None = None) -> None:
        super().__init__(cfg or Jaco2N7RobotCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """The Jaco2 articulation (`KINOVA_JACO2_N7S300_CFG`), base fixed per `cfg.fixed_base`,
        spawned at the configured pose over the vendored USD. In the torque modes the arm actuators
        are set to torque mode (zero stiffness/damping); the fingers are always PD."""
        from isaaclab_assets.robots.kinova import KINOVA_JACO2_N7S300_CFG

        c = self.cfg
        robot = KINOVA_JACO2_N7S300_CFG.copy()  # type: ignore[attr-defined]
        robot.prim_path = f"{{ENV_REGEX_NS}}/{self.prim_name}"  # cfg.name-namespaced (default "Robot")
        robot.spawn.usd_path = c.jaco2_usd  # vendored local copy (resolved in cfg.__post_init__)
        robot.spawn.articulation_props.fix_root_link = c.fixed_base
        robot.init_state.pos = c.base_pos
        robot.init_state.rot = c.base_rot
        robot.init_state.joint_pos = {
            **{f"j2n7s300_joint_{i + 1}": float(q) for i, q in enumerate(c.default_dof_pos)},
            "j2n7s300_joint_finger_[1-3]": c.default_gripper_pos,
            "j2n7s300_joint_finger_tip_[1-3]": c.default_gripper_pos,
        }
        torque_mode = self.control_mode in ("impedance", "osc")
        robot.actuators["arm"].stiffness = 0.0 if torque_mode else c.arm_stiffness
        robot.actuators["arm"].damping = 0.0 if torque_mode else c.arm_damping
        if c.arm_effort_limit is not None:
            robot.actuators["arm"].effort_limit_sim = c.arm_effort_limit
        robot.actuators["gripper"].stiffness = c.gripper_stiffness
        robot.actuators["gripper"].damping = c.gripper_damping
        if c.gripper_effort_limit is not None:
            robot.actuators["gripper"].effort_limit_sim = c.gripper_effort_limit
        if c.gravity_compensation:
            import isaaclab.sim as sim_utils

            robot.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, max_depenetration_velocity=5.0
            )
        return {self.name: robot}

    # ----- lifecycle (hooks; the base orchestrates bind -> on_bind -> build_controller) ---------
    def on_bind(self, env: BaseEnv) -> None:
        self.articulation: Articulation = env.iscene[self.name]

    def build_controller(self) -> CompositeController:
        """`composite([<arm controller>, joint(fingers)])` for the active mode — same shape as
        FrankaRobot, with a 6-DOF finger leaf (3 proximal + 3 tip position targets)."""
        torque_mode = self.control_mode in ("impedance", "osc")
        ctrl_dt = self.TORQUE_CONTROL_DT if torque_mode else self.JOINT_CONTROL_DT
        gripper = JointController(JointControllerCfg(self.GRIPPER_JOINTS, dt=ctrl_dt), command_type="position")
        if torque_mode:
            ts_cfg = TaskSpaceControllerCfg(
                dt=ctrl_dt,  # target rate; the torque law recomputes every physics step
                ee_body=self.EE_BODY,
                arm_joint_names=self.ARM_JOINTS,
                nullspace_dof_pos=self.cfg.nullspace_dof_pos or self.cfg.default_dof_pos,  # () -> home pose
                ema_factor=0.2,  # smooth the action stream
            )
            cls = TaskSpaceImpedanceController if self.control_mode == "impedance" else OperationalSpaceController
            arm: Any = cls(ts_cfg)
        elif self.control_mode == "joint":
            arm = JointController(JointControllerCfg(self.ARM_JOINTS, dt=ctrl_dt), command_type="position")
        else:
            raise ValueError(f"unknown Jaco2 control_mode {self.control_mode!r}; known: {self.control_modes}")
        return CompositeController([arm, gripper])

    def reset(self, env_ids: torch.Tensor) -> None:
        """Home pose: default joint state; hold all joint position targets at default (the finger PD
        holds, the arm targets are inert in torque mode). Root written to the fixed spawn pose + origin."""
        art = self.articulation
        jp = art.data.default_joint_pos[env_ids].clone()
        jv = art.data.default_joint_vel[env_ids].clone()
        art.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
        art.set_joint_position_target(jp, env_ids=env_ids)
        art.set_joint_effort_target(torch.zeros_like(jp), env_ids=env_ids)
        root = art.data.default_root_state[env_ids].clone()
        root[:, 0:3] += self.env.iscene.env_origins[env_ids]
        art.write_root_state_to_sim(root, env_ids)
        if self.controller is not None:
            self.controller.reset(env_ids)

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        mode = self.control_mode
        if mode == "impedance":
            arm = "7 arm joints by task-space impedance (joint torque); the action is 6 end-effector pose deltas"
        elif mode == "osc":
            arm = "7 arm joints by operational-space control (joint torque); the action is 6 end-effector pose deltas"
        else:
            arm = "7 arm joints by direct position targets"
        return (
            f"A Kinova Jaco2 7-DOF arm with its integrated 3-finger hand, fixed to the table. Control "
            f"mode '{mode}': {arm}, plus 6 finger joints by direct position target (order: proximal "
            f"1-3 then tip 1-3; 0.2 rad = open, ~1.2 rad = closed). Action dim {self.action_dim}."
        )
