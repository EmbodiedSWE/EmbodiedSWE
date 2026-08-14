"""XArm7Robot — a UFACTORY xArm7 arm + the UFACTORY parallel gripper, for table-top manipulation.

A fixed-base 7-DOF arm (~0.70 m reach, 3.5 kg payload) with the vendor's linkage gripper. The
vendored USD set (`assets/xarm7/` + `assets/xarm_gripper/`, from the Isaac Sim 5.1 asset library's
`Robots/Ufactory/{xarm7,xarm_gripper}` — the robot's default `Variant_Set` composes the gripper in)
keeps NVIDIA's authored effort limits (50/50/30/30/30/20/20 N*m — UFACTORY's real joint ratings).
One local patch, applied at vendor time: the gripper attach joint
(`/UF_ROBOT/gripper/root_joint`, link7 -> gripper base) shipped with its own
`PhysicsArticulationRootAPI`, which splits the robot into TWO articulations under PhysX — the API
is removed in the vendored copy so IsaacLab sees one 13-joint articulation.

The gripper is a fingertip linkage with ONE actuated joint (`drive_joint`, 0 = fully open ->
~0.85 rad closed); the five follower joints ride PhysX mimic constraints authored in the asset, so
they get a zero-gain actuator group and are never commanded.

Control modes mirror `FrankaRobot` (the controllers are embodiment-generic; only names/gains are
per-robot):

  - "osc"       -> arm by operational-space control (torque); action = 6 EE pose deltas + 1 gripper = 7.
  - "impedance" -> arm by Jacobian-transpose task-space impedance (torque); same 7-D action.
  - "joint"     -> arm by direct joint position targets; action = 7 arm + 1 gripper.

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
class XArm7RobotCfg(BaseRobotCfg):
    """Config for `XArm7Robot`. `control_mode` is inherited from `BaseRobotCfg` ("" -> first declared,
    i.e. "osc"). Arm PD gains are used in "joint" mode only (the torque modes zero them); the
    gripper's single drive joint is always position-controlled."""

    fixed_base: bool = True  # weld the base to the world (a table-mounted arm)
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)  # base at the table level
    base_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz; faces +x
    # Arm position-PD gains — used in "joint" mode only (the torque modes zero them).
    arm_stiffness: float = 400.0
    arm_damping: float = 80.0
    # Arm actuator effort cap [N*m]; None -> keep the vendored asset's authored per-joint ratings
    # (50/50/30/30/30/20/20 — UFACTORY's real xArm7 limits).
    arm_effort_limit: float | None = None
    # Gripper drive-joint PD gains (always position-controlled; the linkage followers are passive
    # mimic joints). Torque units — the knuckle links weigh ~30-50 g.
    gripper_stiffness: float = 100.0
    gripper_damping: float = 10.0
    gripper_effort_limit: float | None = None  # None -> the asset's authored 1000 (plenty)
    # Home posture of the 7 arm joints: a forward-facing ready pose with the elbow well bent and
    # the tool already pitched toward the table (zero = the arm pointing straight up; joint4's
    # lower limit is only -11 deg, so a near-straight elbow leaves the wrist no room to flip the
    # tool down). Retune per task.
    default_dof_pos: tuple[float, ...] = (0.0, -0.35, 0.0, 1.15, 0.0, 1.5, 0.0)
    # Gripper drive-joint home (rad): 0 = fully open (~85 mm aperture), ~0.85 = closed.
    default_gripper_pos: float = 0.0
    # Posture the task-space nullspace pulls toward; () -> use default_dof_pos.
    nullspace_dof_pos: tuple[float, ...] = ()
    xarm7_usd: str = ""  # "" -> the vendored robots/assets/xarm7/xarm7.usd

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parent / "assets" / "xarm7"
        self.xarm7_usd = self.xarm7_usd or str(assets / "xarm7.usd")


@ROBOTS.register("xarm7")
class XArm7Robot(BaseRobot):
    """UFACTORY xArm7 arm + vendor parallel gripper. `apply_action` delegates to the controller for
    `control_mode`; the gripper's `drive_joint` is always a direct position target, the 7 arm joints
    by the mode's arm controller (torque-mode task-space control, or position JointController)."""

    control_modes: tuple[str, ...] = ("osc", "impedance", "joint")  # osc default

    cfg: XArm7RobotCfg

    ARM_JOINTS: tuple[str, ...] = ("joint[1-7]",)
    GRIPPER_JOINTS: tuple[str, ...] = ("drive_joint",)  # the linkage's one actuated joint
    # The five linkage followers: PhysX mimic constraints drive them; their actuator group only
    # pins the (absent) USD drives to zero so nothing fights the mimics.
    PASSIVE_JOINTS: tuple[str, ...] = (
        "left_finger_joint",
        "left_inner_knuckle_joint",
        "right_inner_knuckle_joint",
        "right_outer_knuckle_joint",
        "right_finger_joint",
    )
    EE_BODY: str = "xarm_gripper_base_link"  # the OSC control frame (the gripper's base body)

    # Action/target rate (s) — same scheme as FrankaRobot.
    TORQUE_CONTROL_DT: float = 1.0 / 15.0
    JOINT_CONTROL_DT: float = 0.02

    def __init__(self, cfg: XArm7RobotCfg | None = None) -> None:
        super().__init__(cfg or XArm7RobotCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """The xArm7 articulation, authored from scratch over the vendored USD (no isaaclab_assets
        preset exists for this arm). Arm gains follow the mode; the mimic followers stay passive."""
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg

        c = self.cfg
        torque_mode = self.control_mode in ("impedance", "osc")
        return {
            self.name: ArticulationCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{self.prim_name}",  # cfg.name-namespaced (default "Robot")
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.xarm7_usd,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        max_depenetration_velocity=5.0,
                    ),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False,  # the linkage fingers interleave; asset has no filters
                        solver_position_iteration_count=8,
                        solver_velocity_iteration_count=0,
                        fix_root_link=c.fixed_base,
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=c.base_pos,
                    rot=c.base_rot,
                    joint_pos={
                        **{f"joint{i + 1}": float(q) for i, q in enumerate(c.default_dof_pos)},
                        "drive_joint": c.default_gripper_pos,
                    },
                ),
                actuators={
                    "xarm7_arm": ImplicitActuatorCfg(
                        joint_names_expr=list(self.ARM_JOINTS),
                        stiffness=0.0 if torque_mode else c.arm_stiffness,
                        damping=0.0 if torque_mode else c.arm_damping,
                        effort_limit_sim=c.arm_effort_limit,  # None -> the authored per-joint ratings
                        friction=0.0,  # the asset authors physxJoint:jointFriction=1.0 on every arm
                        # joint — PhysX scales it by the transmitted constraint force, a large
                        # Coulomb brake under gravity load. The benchmark's other arms author no
                        # joint friction: zero it for parity.
                    ),
                    "xarm7_gripper": ImplicitActuatorCfg(
                        joint_names_expr=list(self.GRIPPER_JOINTS),
                        stiffness=c.gripper_stiffness,
                        damping=c.gripper_damping,
                        effort_limit_sim=c.gripper_effort_limit,
                    ),
                    "xarm7_gripper_passive": ImplicitActuatorCfg(
                        joint_names_expr=list(self.PASSIVE_JOINTS),
                        stiffness=0.0,  # mimic constraints drive these; keep the drives silent
                        damping=0.0,
                    ),
                },
                soft_joint_pos_limit_factor=1.0,
            )
        }

    # ----- lifecycle (hooks; the base orchestrates bind -> on_bind -> build_controller) ---------
    def on_bind(self, env: BaseEnv) -> None:
        self.articulation: Articulation = env.iscene[self.name]
        # The asset authors physxJoint:jointFriction=1.0 on every ARM joint; PhysX scales that
        # by the transmitted constraint force — a large Coulomb brake under gravity load. The
        # actuator cfg's `friction=0.0` only updates IsaacLab's data buffer for implicit
        # actuators — it is NOT pushed to PhysX — so write it through explicitly.
        art = self.articulation
        zeros = torch.zeros_like(art.data.joint_friction_coeff)
        write = getattr(art, "write_joint_friction_coefficient_to_sim", None) or getattr(
            art, "write_joint_friction_to_sim"
        )
        write(zeros)

    def build_controller(self) -> CompositeController:
        """`composite([<arm controller>, joint(gripper)])` for the active mode — same shape as
        FrankaRobot, with a 1-DOF gripper leaf (the linkage's drive joint)."""
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
            raise ValueError(f"unknown xArm7 control_mode {self.control_mode!r}; known: {self.control_modes}")
        return CompositeController([arm, gripper])

    def reset(self, env_ids: torch.Tensor) -> None:
        """Home pose: default joint state; hold all joint position targets at default (the gripper PD
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
            f"A UFACTORY xArm7 arm (7-DOF cobot, ~0.70 m reach) with the vendor's parallel linkage "
            f"gripper, fixed to the table. Control mode '{mode}': {arm}, plus 1 gripper drive joint by "
            f"direct position target (0 = open ~85 mm, 0.85 rad = closed; the finger linkage follows "
            f"by mimic constraint). Action dim {self.action_dim}."
        )
