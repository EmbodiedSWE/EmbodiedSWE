"""FrankaRobot — a Franka Emika Panda arm + parallel gripper, for table-top manipulation.

A fixed-base 7-DOF arm with a 2-finger gripper. Three control modes (the gripper is always direct
position targets; switching the mode swaps only the arm controller):

  - "osc"       -> arm by operational-space control (`OperationalSpaceController`, inertia-shaped
                   torque); action = 6 EE pose deltas + 2 gripper = 8. Default — the inertia decoupling
                   keeps the low-inertia wrist smooth.
  - "impedance" -> arm by Jacobian-transpose task-space impedance (`TaskSpaceImpedanceController`);
                   same 8-D action. The form Isaac's Factory tasks use; on this arm it can shake the
                   wrist (no inertia decoupling), so it's not the default.
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
    # Arm actuator effort cap [N*m]; None -> keep the preset's real-Panda limits (87/12). Raise for
    # scripted joint-position tracking that must not crawl at the real limits (kinematic-demo ports).
    arm_effort_limit: float | None = tunable(None)
    # Body-level gravity compensation fraction (Newton/MuJoCo backend only; 1.0 = weightless arm).
    # The preset's PhysX `disable_gravity` flag is IGNORED by the Newton pipeline — MuJoCo needs
    # `gravcomp`, else a kp=400 servo sags ~tau_g/kp (~0.1 rad on the shoulder when extended).
    # None -> leave the preset untouched (required under the PhysX/2.x venv).
    gravity_compensation: float | None = tunable(None)
    # Gripper PD gains (always position-controlled; holds / grasps the part).
    gripper_stiffness: float = tunable(2000.0)
    gripper_damping: float = tunable(100.0)
    # Gripper actuator effort cap [N]; None -> keep the preset's default. Raise for pinch grips
    # that must not saturate (e.g. 500.0 for cloth, the isaaclab soft-lift tasks' value).
    gripper_effort_limit: float | None = tunable(None)
    # Home posture of the 7 arm joints: the arm resets here. A forward-facing ready pose; retune per
    # task (e.g. to start the gripper near the work).
    default_dof_pos: tuple[float, ...] = tunable((0.0015, -0.197, -0.0014, -1.976, -0.00028, 1.78, 0.786))
    # Posture the task-space nullspace pulls toward; () -> use default_dof_pos. A non-singular elbow
    # config that actively resolves the arm's redundancy (keeps the wrist from drifting); kept separate
    # from the home pose on purpose.
    nullspace_dof_pos: tuple[float, ...] = tunable((-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754))
    franka_usd: str = info("")  # "" -> the vendored robots/assets/franka/panda_instanceable.usd

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parent / "assets" / "franka"
        self.franka_usd = self.franka_usd or str(assets / "panda_instanceable.usd")


@ROBOTS.register("franka")
class FrankaRobot(BaseRobot):
    """Franka Panda arm + gripper. `apply_action` delegates to the controller for `control_mode`; the
    2 gripper fingers are always direct position targets, the 7 arm joints by the mode's arm controller
    (torque-mode OSC, or position-mode JointController)."""

    control_modes: tuple[str, ...] = ("osc", "impedance", "joint")  # osc default: smooth on this arm
    cfg: FrankaRobotCfg

    ARM_JOINTS: tuple[str, ...] = ("panda_joint[1-7]",)
    GRIPPER_JOINTS: tuple[str, ...] = ("panda_finger_joint.*",)
    EE_BODY: str = "panda_hand"  # the OSC control frame (a real body with a Jacobian)

    # Action/target rate (s). Torque modes (osc/impedance): ~15 Hz target (Isaac `Factory-NutThread`,
    # decim 8 @ 120 Hz), latched while the torque law recomputes every physics step (see
    # `_TaskSpaceController.apply`). Joint mode: ~50 Hz, held by the actuator PD.
    TORQUE_CONTROL_DT: float = 1.0 / 15.0
    JOINT_CONTROL_DT: float = 0.02

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
        robot.prim_path = f"{{ENV_REGEX_NS}}/{self.prim_name}"  # cfg.name-namespaced (default "Robot")
        robot.spawn.usd_path = c.franka_usd  # vendored local panda (resolved in cfg.__post_init__)
        robot.spawn.articulation_props.fix_root_link = c.fixed_base
        robot.init_state.pos = c.base_pos
        robot.init_state.rot = c.base_rot
        # Home the 7 arm joints to cfg.default_dof_pos (reset() reads this via default_joint_pos);
        # keep the gripper-finger defaults.
        robot.init_state.joint_pos = {
            **robot.init_state.joint_pos,
            **{f"panda_joint{i + 1}": float(q) for i, q in enumerate(c.default_dof_pos)},
        }
        torque_mode = self.control_mode in ("impedance", "osc")
        for arm_act in ("panda_shoulder", "panda_forearm"):
            robot.actuators[arm_act].stiffness = 0.0 if torque_mode else c.arm_stiffness
            robot.actuators[arm_act].damping = 0.0 if torque_mode else c.arm_damping
            if c.arm_effort_limit is not None:
                robot.actuators[arm_act].effort_limit_sim = c.arm_effort_limit
        robot.actuators["panda_hand"].stiffness = c.gripper_stiffness
        robot.actuators["panda_hand"].damping = c.gripper_damping
        if c.gripper_effort_limit is not None:
            robot.actuators["panda_hand"].effort_limit_sim = c.gripper_effort_limit
        if c.gravity_compensation is not None:
            # Newton-backend gravity compensation: swap in the MuJoCo rigid-body schema carrying
            # `gravcomp` (imported lazily so the PhysX/2.x venv never touches isaaclab_newton).
            from isaaclab_newton.sim.schemas.schemas_cfg import MujocoRigidBodyPropertiesCfg

            robot.spawn.rigid_props = MujocoRigidBodyPropertiesCfg(gravcomp=c.gravity_compensation)
        return {self.name: robot}

    # ----- lifecycle (hooks; the base orchestrates bind -> on_bind -> build_controller) ---------
    def on_bind(self, env: BaseEnv) -> None:
        self.articulation: Articulation = env.iscene[self.name]

    def build_controller(self) -> CompositeController:
        """`composite([<arm controller>, joint(gripper)])` for the active mode. The gripper is always a
        position JointController (2 fingers); the arm controller is OSC (torque) or JointController
        (position)."""
        torque_mode = self.control_mode in ("impedance", "osc")
        ctrl_dt = self.TORQUE_CONTROL_DT if torque_mode else self.JOINT_CONTROL_DT
        gripper = JointController(JointControllerCfg(self.GRIPPER_JOINTS, dt=ctrl_dt), command_type="position")
        if torque_mode:
            ts_cfg = TaskSpaceControllerCfg(
                dt=ctrl_dt,  # target rate; the torque law itself recomputes every physics step (see apply)
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
        art.set_joint_effort_target(torch.zeros_like(jp), env_ids=env_ids)
        root = art.data.default_root_state[env_ids].clone()
        root[:, 0:3] += self.env.iscene.env_origins[env_ids]
        art.write_root_state_to_sim(root, env_ids)
        if self.controller is not None:
            self.controller.reset(env_ids)

    # NOTE: apply_action / action_dim are inherited from BaseRobot — they delegate to self.controller
    # (composite of arm + gripper). The OSC arm writes effort, the gripper writes position, each through
    # the sink it captured at bind.

    # get_state / set_state are inherited from BaseRobot (single-articulation sim state + controller).

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
