"""CobottaPro1300Robot — a Denso Cobotta Pro 1300 arm with an OnRobot RG6 gripper, for table-top
manipulation.

A fixed-base 6-DOF industrial cobot (~1.3 m reach, 12 kg payload) with the RG6 mounted, as shipped
by the Isaac Sim 5.1 asset library (`Robots/Denso/CobottaPro1300`, vendored unmodified under
`assets/cobotta_pro_1300/`). The RG6 is a linkage gripper with ONE actuated joint (`finger_joint`;
negative = open toward the ~160 mm aperture, positive = closed) whose five follower joints ride
the asset's PhysX mimic constraints — their authored drives are kept as-is so the linkage behaves
exactly as shipped, and they are never commanded.

Control modes mirror `FrankaRobot` (the controllers are embodiment-generic; only names/gains are
per-robot). On a 6-DOF arm the task-space nullspace is degenerate away from singularities, so the
posture term mostly guards joint limits:

  - "osc"       -> arm by operational-space control (torque); action = 6 EE pose deltas + 1 gripper = 7.
  - "impedance" -> arm by Jacobian-transpose task-space impedance (torque); same 7-D action.
  - "joint"     -> arm by direct joint position targets; action = 6 arm + 1 gripper.

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
class CobottaPro1300RobotCfg(BaseRobotCfg):
    """Config for `CobottaPro1300Robot`. `control_mode` is inherited from `BaseRobotCfg` ("" ->
    first declared, i.e. "osc"). Arm PD gains are used in "joint" mode only (the torque modes zero
    them); the RG6's drive joint is always position-controlled."""

    fixed_base: bool = True  # weld the base to the world (a table-mounted arm)
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)  # base at the table level
    base_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz; faces +x
    # Arm position-PD gains — used in "joint" mode only (the asset's authored drives are per-joint
    # servo-stiff; these give the benchmark's uniform behavior instead).
    arm_stiffness: float = 400.0
    arm_damping: float = 80.0
    # Arm actuator effort cap [N*m]; None -> the asset's authored 60 per joint.
    arm_effort_limit: float | None = None
    # RG6 drive-joint PD gains (always position-controlled; the linkage followers are mimic-driven).
    gripper_stiffness: float = 200.0
    gripper_damping: float = 20.0
    gripper_effort_limit: float | None = 50.0  # the asset's authored 60000 is a placeholder; cap sane
    # Home posture of the 6 arm joints: an elbow-up ready pose leaning forward (+x); zero = the
    # 1.3 m arm pointing straight up. Retune per task.
    default_dof_pos: tuple[float, ...] = (0.0, 0.6, 1.8, 0.0, 0.75, 0.0)
    # RG6 drive-joint home (rad). MUST stay 0.0 at spawn: the five linkage followers are authored
    # at 0 and ride mimic constraints — an init finger_joint away from 0 violates the constraint
    # on the first step and blows the articulation up. Open the gripper by COMMAND after reset
    # (negative = toward the ~160 mm aperture).
    default_gripper_pos: float = 0.0
    # Posture the task-space nullspace pulls toward; () -> use default_dof_pos.
    nullspace_dof_pos: tuple[float, ...] = ()
    cobotta_usd: str = ""  # "" -> the vendored robots/assets/cobotta_pro_1300/cobotta_pro_1300.usd

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parent / "assets" / "cobotta_pro_1300"
        self.cobotta_usd = self.cobotta_usd or str(assets / "cobotta_pro_1300.usd")


@ROBOTS.register("cobotta_pro_1300")
class CobottaPro1300Robot(BaseRobot):
    """Denso Cobotta Pro 1300 arm + OnRobot RG6. `apply_action` delegates to the controller for
    `control_mode`; the RG6's `finger_joint` is always a direct position target, the 6 arm joints by
    the mode's arm controller (torque-mode task-space control, or position JointController)."""

    control_modes: tuple[str, ...] = ("osc", "impedance", "joint")  # osc default

    cfg: CobottaPro1300RobotCfg

    ARM_JOINTS: tuple[str, ...] = ("joint_[1-6]",)
    GRIPPER_JOINTS: tuple[str, ...] = ("finger_joint",)  # the RG6's one actuated joint
    # The five linkage followers: the asset authors both mimic constraints AND assist drives on
    # them — keep the authored drives (stiffness/damping None = read from USD) so the linkage
    # behaves exactly as shipped; they are never commanded.
    PASSIVE_JOINTS: tuple[str, ...] = (
        "left_inner_knuckle_joint",
        "right_inner_knuckle_joint",
        "right_outer_knuckle_joint",
        "left_inner_finger_joint",
        "right_inner_finger_joint",
    )
    EE_BODY: str = "onrobot_rg6_base_link"  # the OSC control frame (the gripper's base body)

    # Action/target rate (s) — same scheme as FrankaRobot.
    TORQUE_CONTROL_DT: float = 1.0 / 15.0
    JOINT_CONTROL_DT: float = 0.02

    def __init__(self, cfg: CobottaPro1300RobotCfg | None = None) -> None:
        super().__init__(cfg or CobottaPro1300RobotCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """The Cobotta articulation, authored from scratch over the vendored USD (no isaaclab_assets
        preset exists for this arm). Arm gains follow the mode; the RG6 followers keep their
        authored assist drives and ride the mimic constraints."""
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg

        c = self.cfg
        torque_mode = self.control_mode in ("impedance", "osc")
        return {
            self.name: ArticulationCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{self.prim_name}",  # cfg.name-namespaced (default "Robot")
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.cobotta_usd,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        max_depenetration_velocity=5.0,
                    ),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False,  # the RG6 linkage interleaves; asset has no filters
                        solver_position_iteration_count=8,
                        solver_velocity_iteration_count=0,
                        fix_root_link=c.fixed_base,
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=c.base_pos,
                    rot=c.base_rot,
                    joint_pos={
                        **{f"joint_{i + 1}": float(q) for i, q in enumerate(c.default_dof_pos)},
                        "finger_joint": c.default_gripper_pos,
                    },
                ),
                actuators={
                    "cobotta_arm": ImplicitActuatorCfg(
                        joint_names_expr=list(self.ARM_JOINTS),
                        stiffness=0.0 if torque_mode else c.arm_stiffness,
                        damping=0.0 if torque_mode else c.arm_damping,
                        effort_limit_sim=c.arm_effort_limit,  # None -> the authored 60 N*m per joint
                    ),
                    "rg6_drive": ImplicitActuatorCfg(
                        joint_names_expr=list(self.GRIPPER_JOINTS),
                        stiffness=c.gripper_stiffness,
                        damping=c.gripper_damping,
                        effort_limit_sim=c.gripper_effort_limit,
                    ),
                    "rg6_passive": ImplicitActuatorCfg(
                        joint_names_expr=list(self.PASSIVE_JOINTS),
                        stiffness=None,  # None = keep the asset's authored assist drives
                        damping=None,
                    ),
                },
                soft_joint_pos_limit_factor=1.0,
            )
        }

    # ----- lifecycle (hooks; the base orchestrates bind -> on_bind -> build_controller) ---------
    def on_bind(self, env: BaseEnv) -> None:
        self.articulation: Articulation = env.iscene[self.name]

    def build_controller(self) -> CompositeController:
        """`composite([<arm controller>, joint(gripper)])` for the active mode — same shape as
        FrankaRobot, with a 1-DOF gripper leaf (the RG6's drive joint)."""
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
            raise ValueError(
                f"unknown Cobotta control_mode {self.control_mode!r}; known: {self.control_modes}"
            )
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
            arm = "6 arm joints by task-space impedance (joint torque); the action is 6 end-effector pose deltas"
        elif mode == "osc":
            arm = "6 arm joints by operational-space control (joint torque); the action is 6 end-effector pose deltas"
        else:
            arm = "6 arm joints by direct position targets"
        return (
            f"A Denso Cobotta Pro 1300 arm (6-DOF cobot, ~1.3 m reach) with an OnRobot RG6 parallel "
            f"linkage gripper, fixed to the table. Control mode '{mode}': {arm}, plus 1 gripper drive "
            f"joint by direct position target (negative = open toward ~160 mm, positive = closed; the "
            f"finger linkage follows by mimic constraint). Action dim {self.action_dim}."
        )
