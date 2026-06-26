"""FrankaRobot — a Franka Emika Panda arm + parallel gripper, for table-top manipulation.

A fixed-base 7-DOF arm with a 2-finger gripper. Three control modes (the gripper is always direct
position targets; switching the mode swaps only the arm controller):

  - "impedance" -> arm by Jacobian-transpose task-space impedance (`TaskSpaceImpedanceController`,
                   joint torque); action = 6 EE pose deltas + 2 gripper = 8. (Default; the form Isaac's
                   Factory tasks use.)
  - "osc"       -> arm by operational-space control (`OperationalSpaceController`, inertia-shaped
                   torque); same 8-D action.
  - "joint"     -> arm by direct joint position targets (`JointController`); action = 7 arm + 2 gripper.

The two torque modes load the arm actuators in TORQUE mode (zero stiffness/damping) so the
controller's torques drive them; "joint" keeps the arm position PD. So `action_dim` and the arm
actuator setup follow `control_mode`. Heavy imports are deferred so registration stays app-free.
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
class FrankaRobotCfg(BaseRobotCfg):
    """Config for `FrankaRobot`. `control_mode` is inherited from `BaseRobotCfg` ("" -> first declared).
    `base_pos`/`base_rot` are placement dials; `fixed_base` + the asset are structural. The arm PD gains
    are used only in "joint" mode (OSC zeroes them for torque control); the gripper PD is always on."""

    fixed_base: bool = info(True)  # weld the base to the world (a table-mounted arm)
    base_pos: tuple[float, float, float] = tunable((0.0, 0.0, 0.0))  # base at the table level
    base_rot: tuple[float, float, float, float] = tunable((1.0, 0.0, 0.0, 0.0))  # wxyz; faces +x
    # Arm position-PD gains — used in "joint" mode only (OSC sets the arm actuators to torque mode).
    arm_stiffness: float = tunable(400.0)
    arm_damping: float = tunable(80.0)
    # Gripper PD gains (always position-controlled; holds / grasps the part).
    gripper_stiffness: float = tunable(2000.0)
    gripper_damping: float = tunable(100.0)
    franka_usd: str = info("")  # "" -> the vendored robots/assets/franka/panda_instanceable.usd

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parent / "assets" / "franka"
        self.franka_usd = self.franka_usd or str(assets / "panda_instanceable.usd")


@ROBOTS.register("franka")
class FrankaRobot(BaseRobot):
    """Franka Panda arm + gripper. `apply_action` delegates to the controller for `control_mode`; the
    2 gripper fingers are always direct position targets, the 7 arm joints by the mode's arm controller
    (torque-mode OSC, or position-mode JointController)."""

    control_modes: tuple[str, ...] = ("impedance", "osc", "joint")
    cfg: FrankaRobotCfg

    ARM_JOINTS: tuple[str, ...] = ("panda_joint[1-7]",)
    GRIPPER_JOINTS: tuple[str, ...] = ("panda_finger_joint.*",)
    EE_BODY: str = "panda_hand"  # the OSC control frame (a real body with a Jacobian)

    def __init__(self, cfg: FrankaRobotCfg | None = None) -> None:
        super().__init__(cfg or FrankaRobotCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """The Franka articulation (`FRANKA_PANDA_HIGH_PD_CFG`), base fixed per `cfg.fixed_base`, spawned
        at the configured pose. In the torque modes ("impedance"/"osc") the arm actuators are set to
        torque mode (zero stiffness/damping); in "joint" mode they keep position PD. The gripper is always PD."""
        from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

        c = self.cfg
        robot = FRANKA_PANDA_HIGH_PD_CFG.copy()  # type: ignore[attr-defined]
        robot.prim_path = "{ENV_REGEX_NS}/Robot"
        robot.spawn.usd_path = c.franka_usd  # vendored local panda (resolved in cfg.__post_init__)
        robot.spawn.articulation_props.fix_root_link = c.fixed_base
        robot.init_state.pos = c.base_pos
        robot.init_state.rot = c.base_rot
        torque_mode = self.control_mode in ("impedance", "osc")
        for arm_act in ("panda_shoulder", "panda_forearm"):
            robot.actuators[arm_act].stiffness = 0.0 if torque_mode else c.arm_stiffness
            robot.actuators[arm_act].damping = 0.0 if torque_mode else c.arm_damping
        robot.actuators["panda_hand"].stiffness = c.gripper_stiffness
        robot.actuators["panda_hand"].damping = c.gripper_damping
        return {"robot": robot}

    # ----- lifecycle (hooks; the base orchestrates bind -> on_bind -> build_controller) ---------
    def on_bind(self, env: BaseEnv) -> None:
        self.articulation: Articulation = env.iscene["robot"]

    def build_controller(self) -> CompositeController:
        """`composite([<arm controller>, joint(gripper)])` for the active mode. The gripper is always a
        position JointController (2 fingers); the arm controller is OSC (torque) or JointController
        (position)."""
        gripper = JointController(JointControllerCfg(self.GRIPPER_JOINTS), command_type="position")
        if self.control_mode in ("impedance", "osc"):
            ts_cfg = TaskSpaceControllerCfg(ee_body=self.EE_BODY, arm_joint_names=self.ARM_JOINTS)
            cls = TaskSpaceImpedanceController if self.control_mode == "impedance" else OperationalSpaceController
            arm: Any = cls(ts_cfg)
        elif self.control_mode == "joint":
            arm = JointController(JointControllerCfg(self.ARM_JOINTS), command_type="position")
        else:
            raise ValueError(f"unknown Franka control_mode {self.control_mode!r}; known: {self.control_modes}")
        return CompositeController([arm, gripper])

    def reset(self, env_ids: torch.Tensor) -> None:
        """Home pose: default joint state; hold all joint position targets at default (the gripper PD
        holds, the arm targets are inert in torque mode). Root written to the fixed spawn pose + origin."""
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

    # NOTE: apply_action / action_dim are inherited from BaseRobot — they delegate to self.controller
    # (composite of arm + gripper). The OSC arm writes effort, the gripper writes position, each through
    # the sink it captured at bind.

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        art = self.articulation
        return {
            "root": art.data.root_state_w[env_ids].clone(),
            "joint_pos": art.data.joint_pos[env_ids].clone(),
            "joint_vel": art.data.joint_vel[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        art = self.articulation
        art.write_root_state_to_sim(state["root"], env_ids)
        art.write_joint_state_to_sim(state["joint_pos"], state["joint_vel"], env_ids=env_ids)
        art.set_joint_position_target(state["joint_pos"], env_ids=env_ids)

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
            f"A Franka Emika Panda arm with a parallel-jaw gripper, fixed to the table. Control mode "
            f"'{mode}': {arm}, plus 2 gripper fingers by direct position target. Action dim {self.action_dim}."
        )
