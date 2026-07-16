"""WxaiRobot — a Trossen WidowX AI (WXAI) follower arm, for table-top manipulation.

The follower arm of the Trossen AI kits (the ALOHA lineage: Trossen replaced the ViperX-based
ALOHA kits with this line — see `Aloha` in `multi.py` for the bimanual pair). A fixed-base 6-DOF
arm with a parallel gripper, much smaller than a Franka (~0.5 m reach), so it suits benches where
the Franka crowds the work. The vendored USD (`assets/wxai/wxai_follower.usd`, from Trossen's
official `trossen_ai_isaac` repo @ e5fccea5b3d4, BSD-3; one local patch — the jointless
`ee_gripper_link` marker's rigid body is disabled so it doesn't free-fall) is self-contained and
carries Trossen-tuned joint drives, which we keep by default (gain cfg fields default to None ->
USD values). A D405-style wrist camera is part of the asset (mesh-only geometry; no sensor prim).

Control modes mirror `FrankaRobot` (the controllers are embodiment-generic; only names/gains are
per-robot), but "joint" is the DEFAULT here — the torque modes are wired identically yet untuned
on this arm (low effort limits: 27 Nm shoulder / 7 Nm wrist), so treat them as experimental:

  - "joint"     -> arm by direct joint position targets (`JointController`); action = 6 arm + 1 gripper.
  - "osc"       -> arm by operational-space control (torque); action = 6 EE pose deltas + 1 gripper.
  - "impedance" -> arm by Jacobian-transpose task-space impedance (torque); same 7-D action.

The gripper is ONE dof: `left_carriage_joint` position; the right carriage is a PhysX mimic joint
inside the USD and follows on its own (it is deliberately left without an actuator — Isaac Lab
logs a coverage warning, which is expected). Heavy imports are deferred so registration stays
app-free.
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
from robobench.core import ROBOTS, BaseRobot, BaseRobotCfg, info, tunable

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from robobench.core import BaseEnv


@dataclass
class WxaiRobotCfg(BaseRobotCfg):
    """Config for `WxaiRobot`. `control_mode` is inherited from `BaseRobotCfg` ("" -> first declared,
    i.e. "joint"). Gain fields default to None = keep the Trossen-tuned drives baked in the USD
    (the torque modes zero the arm gains regardless); set a float to override."""

    fixed_base: bool = info(True)  # weld the base to the world (a table-mounted arm)
    base_pos: tuple[float, float, float] = tunable((0.0, 0.0, 0.0))  # base at the table level
    base_rot: tuple[float, float, float, float] = tunable((1.0, 0.0, 0.0, 0.0))  # wxyz; faces +x
    # Arm position-PD gains — None -> the USD's Trossen-tuned values (664/735/738 shoulder,
    # 34-62 wrist; damping ~1%). Used in "joint" mode only (torque modes zero them).
    arm_stiffness: float | None = tunable(None)
    arm_damping: float | None = tunable(None)
    # Gripper PD gains — None -> the USD's values (very stiff carriage drive). Always position-controlled.
    gripper_stiffness: float | None = tunable(None)
    gripper_damping: float | None = tunable(None)
    # Home posture of the 6 arm joints (Trossen's zero pose: folded upright). Retune per task.
    default_dof_pos: tuple[float, ...] = tunable((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    # Gripper carriage home (m): 0 closed .. 0.044 open. Start open, ready to grasp.
    default_gripper_pos: float = tunable(0.044)
    # Posture the task-space nullspace pulls toward; () -> use default_dof_pos. (On a 6-DOF arm the
    # nullspace is degenerate away from singularities — this mostly matters near joint limits.)
    nullspace_dof_pos: tuple[float, ...] = tunable(())
    wxai_usd: str = info("")  # "" -> the vendored assets/wxai/wxai_follower.usd

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parent / "assets" / "wxai"
        self.wxai_usd = self.wxai_usd or str(assets / "wxai_follower.usd")


@ROBOTS.register("wxai")
class WxaiRobot(BaseRobot):
    """Trossen WXAI follower arm + parallel gripper. `apply_action` delegates to the controller for
    `control_mode`; the gripper carriage is always a direct position target, the 6 arm joints by the
    mode's arm controller (position JointController, or torque-mode task-space control)."""

    control_modes: tuple[str, ...] = ("joint", "osc", "impedance")  # joint default: USD-tuned PD

    cfg: WxaiRobotCfg

    ARM_JOINTS: tuple[str, ...] = ("joint_[0-5]",)
    GRIPPER_JOINTS: tuple[str, ...] = ("left_carriage_joint",)  # right carriage: PhysX mimic in-USD
    # The gripper base link — the wrist-most REAL articulation link. The USD's `ee_gripper_link`
    # marker is a jointless orphan body (PhysX drops it from the articulation, so it has no
    # Jacobian; its rigid body is disabled in the vendored USD so it doesn't free-fall at spawn).
    EE_BODY: str = "link_6"

    # Action/target rate (s) — same scheme as FrankaRobot.
    TORQUE_CONTROL_DT: float = 1.0 / 15.0
    JOINT_CONTROL_DT: float = 0.02

    def __init__(self, cfg: WxaiRobotCfg | None = None) -> None:
        super().__init__(cfg or WxaiRobotCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """The WXAI articulation, authored from scratch over the vendored USD (no isaaclab_assets
        preset exists for this arm). Solver/actuator layout follows Trossen's own `WXAI_BASE_CFG`:
        implicit actuators with stiffness/damping None = keep the USD's tuned drives; the mimic'd
        right carriage joint stays unactuated on purpose."""
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg

        c = self.cfg
        torque_mode = self.control_mode in ("impedance", "osc")
        return {
            self.name: ArticulationCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{self.prim_name}",  # cfg.name-namespaced (default "Robot")
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.wxai_usd,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        max_depenetration_velocity=5.0,
                    ),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=True,
                        solver_position_iteration_count=8,
                        solver_velocity_iteration_count=0,
                        fix_root_link=c.fixed_base,
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=c.base_pos,
                    rot=c.base_rot,
                    joint_pos={
                        **{f"joint_{i}": float(q) for i, q in enumerate(c.default_dof_pos)},
                        "left_carriage_joint": c.default_gripper_pos,
                    },
                ),
                actuators={
                    "wxai_arm": ImplicitActuatorCfg(
                        joint_names_expr=list(self.ARM_JOINTS),
                        stiffness=0.0 if torque_mode else c.arm_stiffness,
                        damping=0.0 if torque_mode else c.arm_damping,
                    ),
                    "wxai_gripper": ImplicitActuatorCfg(
                        joint_names_expr=list(self.GRIPPER_JOINTS),
                        stiffness=c.gripper_stiffness,
                        damping=c.gripper_damping,
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
        FrankaRobot, with a 1-dof gripper leaf."""
        torque_mode = self.control_mode in ("impedance", "osc")
        ctrl_dt = self.TORQUE_CONTROL_DT if torque_mode else self.JOINT_CONTROL_DT
        gripper = JointController(JointControllerCfg(self.GRIPPER_JOINTS, dt=ctrl_dt), command_type="position")
        if torque_mode:
            ts_cfg = TaskSpaceControllerCfg(
                dt=ctrl_dt,
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
            raise ValueError(f"unknown WXAI control_mode {self.control_mode!r}; known: {self.control_modes}")
        return CompositeController([arm, gripper])

    def reset(self, env_ids: torch.Tensor) -> None:
        """Home pose: default joint state; hold all joint position targets at default. Root written
        to the fixed spawn pose + origin."""
        art = self.articulation
        jp = art.data.default_joint_pos[env_ids].clone()
        jv = art.data.default_joint_vel[env_ids].clone()
        art.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
        art.set_joint_position_target(jp, env_ids=env_ids)
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
            f"A Trossen WidowX AI arm (small 6-DOF cobot, ~0.5 m reach) with a parallel gripper, fixed "
            f"to the table. Control mode '{mode}': {arm}, plus 1 gripper carriage by direct position "
            f"target (0 closed .. 0.044 open; the second finger mirrors it). Action dim {self.action_dim}."
        )
