"""PiperRobot — an AgileX PiPER arm, for table-top manipulation.

A fixed-base 6-DOF arm with a two-finger parallel gripper — a compact cobot (~0.6 m reach, 1.5 kg
payload) that fits benches where the Franka crowds the work. The vendored USD set
(`assets/piper/`, from AgileX's official `piper_isaac_sim` repo @ 8e1f88fdb7af — the
`v100_realsense_camera_v2` conversion their own `piper_v2.usd` wrapper points at, unmodified) is
Isaac's URDF-converter output; a
D435-style wrist camera rides on link6 as mesh-only geometry (no sensor prim). The converter's
authored drive gains are placeholder-weak, so unlike `WxaiRobot` this preset ALWAYS overrides
them (defaults 400/80, AgileX's own Isaac tutorial numbers).

Control modes mirror `FrankaRobot` (the controllers are embodiment-generic; only names/gains are
per-robot), but "joint" is the DEFAULT here — the torque modes are wired identically yet untuned
on this arm, so treat them as experimental:

  - "joint"     -> arm by direct joint position targets (`JointController`); action = 6 arm + 2 gripper.
  - "osc"       -> arm by operational-space control (torque); action = 6 EE pose deltas + 2 gripper.
  - "impedance" -> arm by Jacobian-transpose task-space impedance (torque); same 8-D action.

The gripper is TWO mirrored prismatic fingers (joint7: 0..0.05 m, joint8: -0.05..0 m — no in-USD
mimic, both are actuated), always direct position targets. Heavy imports are deferred so
registration stays app-free.
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
class PiperRobotCfg(BaseRobotCfg):
    """Config for `PiperRobot`. `control_mode` is inherited from `BaseRobotCfg` ("" -> first declared,
    i.e. "joint"). Arm gains are ALWAYS applied over the USD (the converter's authored drives are
    placeholder-weak); the torque modes zero them regardless."""

    fixed_base: bool = True  # weld the base to the world (a table-mounted arm)
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)  # base at the table level
    base_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz; faces +x
    # Arm position-PD gains — AgileX's Isaac tutorial values. Used in "joint" mode only.
    arm_stiffness: float = 400.0
    arm_damping: float = 80.0
    # Gripper PD gains (always position-controlled; the USD's linear drives are ~0.2 stiffness).
    gripper_stiffness: float = 2000.0
    gripper_damping: float = 100.0
    # Home posture of the 6 arm joints: a mild forward-lean ready pose (zero = folded straight up;
    # joint2 in [0, pi], joint3 in [-2.70, 0]). Retune per task.
    default_dof_pos: tuple[float, ...] = (0.0, 1.0, -0.8, 0.0, 0.6, 0.0)
    # Gripper finger home (m): joint7 in [0, 0.05], joint8 mirrors in [-0.05, 0]. Start open.
    default_gripper_pos: tuple[float, float] = (0.04, -0.04)
    # Posture the task-space nullspace pulls toward; () -> use default_dof_pos. (On a 6-DOF arm the
    # nullspace is degenerate away from singularities — this mostly matters near joint limits.)
    nullspace_dof_pos: tuple[float, ...] = ()
    piper_usd: str = ""  # "" -> the vendored assets/piper/ wrapper USD

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parent / "assets" / "piper"
        self.piper_usd = self.piper_usd or str(assets / "piper_description_v100_realsense_camera_v2.usd")


@ROBOTS.register("piper")
class PiperRobot(BaseRobot):
    """AgileX PiPER arm + two-finger gripper. `apply_action` delegates to the controller for
    `control_mode`; the 2 gripper fingers are always direct position targets, the 6 arm joints by
    the mode's arm controller (position JointController, or torque-mode task-space control)."""

    control_modes: tuple[str, ...] = ("joint", "osc", "impedance")  # joint default: simple PD

    cfg: PiperRobotCfg

    ARM_JOINTS: tuple[str, ...] = ("joint[1-6]",)
    GRIPPER_JOINTS: tuple[str, ...] = ("joint[7-8]",)  # mirrored prismatic fingers, both actuated
    EE_BODY: str = "link6"  # the gripper base (fingers link7/link8 hang off it)

    # Action/target rate (s) — same scheme as FrankaRobot.
    TORQUE_CONTROL_DT: float = 1.0 / 15.0
    JOINT_CONTROL_DT: float = 0.02

    def __init__(self, cfg: PiperRobotCfg | None = None) -> None:
        super().__init__(cfg or PiperRobotCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """The PiPER articulation, authored from scratch over the vendored USD (no isaaclab_assets
        preset exists for this arm). Implicit actuators override the converter's placeholder drives
        with the cfg gains (or zero them for the torque modes)."""
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg

        c = self.cfg
        torque_mode = self.control_mode in ("impedance", "osc")
        return {
            self.name: ArticulationCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{self.prim_name}",  # cfg.name-namespaced (default "Robot")
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.piper_usd,
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
                        **{f"joint{i + 1}": float(q) for i, q in enumerate(c.default_dof_pos)},
                        "joint7": c.default_gripper_pos[0],
                        "joint8": c.default_gripper_pos[1],
                    },
                ),
                actuators={
                    "piper_arm": ImplicitActuatorCfg(
                        joint_names_expr=list(self.ARM_JOINTS),
                        stiffness=0.0 if torque_mode else c.arm_stiffness,
                        damping=0.0 if torque_mode else c.arm_damping,
                    ),
                    "piper_gripper": ImplicitActuatorCfg(
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
        FrankaRobot (2-dof gripper leaf)."""
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
            raise ValueError(f"unknown PiPER control_mode {self.control_mode!r}; known: {self.control_modes}")
        return CompositeController([arm, gripper])

    def reset(self, env_ids: torch.Tensor) -> None:
        """Home pose: default joint state; hold all joint position targets at default. Root written
        to the fixed spawn pose + origin."""
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
            f"An AgileX PiPER arm (small 6-DOF cobot, ~0.6 m reach) with a two-finger parallel gripper, "
            f"fixed to the table. Control mode '{mode}': {arm}, plus 2 mirrored gripper fingers by direct "
            f"position target (joint7 0..0.05 m, joint8 -0.05..0 m). Action dim {self.action_dim}."
        )
