"""Attached-gripper composite arms — seven bare-flange manipulators from the Isaac Sim 5.1 asset
library, each with an end-effector welded on at vendor time (`assets/composites/<name>/`):

  - `z1_lite6g`      Unitree Z1 (6-DOF, ~0.74 m)      + UFACTORY Lite6 gripper (2 prismatic jaws)
  - `rizon4_2f85`    Flexiv Rizon 4 (7-DOF, ~0.8 m)    + Robotiq 2F-85
  - `gen3n7_2f85`    Kinova Gen3 N7 (7-DOF, ~0.9 m)    + Robotiq 2F-85
  - `sawyer_egu50`   Rethink Sawyer (7-DOF + head pan) + Schunk EGU-50 (2 prismatic jaws)
  - `crx10ial_2f85`  Fanuc CRX-10iA/L (6-DOF, 1.4 m)   + Robotiq 2F-85
  - `tm12_2f85`      Techman TM12 (6-DOF, 1.3 m)       + Robotiq 2F-85
  - `festo_2f85`     Festo Cobot (6-DOF, pneumatic)    + Robotiq 2F-85

Each composite USD references the vendored arm as its root and the vendored gripper under
`/<name>/gripper`, posed at the arm's flange rest transform with a FixedJoint authored from the
composed rest poses (so nothing snaps at spawn) and the gripper's own articulation root stripped —
the same pattern the xArm7 asset uses natively. The Robotiq 2F-85 and the Lite6 gripper are
mimic-linkage grippers (ONE driven joint; followers ride PhysX mimic constraints and must spawn at
their authored rest); the EGU-50 is a driven prismatic jaw with a mimic'd twin.

The OSC control frame (and the task-side weld body) is the LAST ARM LINK, not the gripper base:
several arms name their own base `base_link`, which collides with the Robotiq base's prim name
inside one articulation, and some flange/tool bodies are massless (unreliable jacobian rows).
Hand geometry (approach/pinch axes, pad offsets) is measured live from the finger bodies.

Control modes mirror `FrankaRobot`:
  - "osc"       -> arm by operational-space control (torque); action = 6 EE pose deltas + gripper.
  - "impedance" -> arm by Jacobian-transpose task-space impedance (torque); same layout.
  - "joint"     -> arm by direct joint position targets.

Heavy imports are deferred so registration stays app-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

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
class AttachedArmRobotCfg(BaseRobotCfg):
    """Shared config for the attached-gripper composites. `control_mode` is inherited from
    `BaseRobotCfg` ("" -> first declared, i.e. "osc"). Arm PD gains are used in "joint" mode only
    (the torque modes zero them); the gripper's driven joint(s) are always position-controlled."""

    fixed_base: bool = True  # weld the base to the world (a table-mounted arm)
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)  # base at the table level
    base_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz; faces +x
    # Arm position-PD gains — used in "joint" mode only (the torque modes zero them).
    arm_stiffness: float = 400.0
    arm_damping: float = 80.0
    # Arm actuator effort cap [N*m]; None -> the asset's authored per-joint ratings.
    arm_effort_limit: float | None = None
    # Gripper driven-joint PD gains (always position-controlled; linkage followers are passive).
    gripper_stiffness: float = 100.0
    gripper_damping: float = 10.0
    gripper_effort_limit: float | None = 50.0
    # Home posture of the arm joints (length = the robot's arm DOF); () -> the class default.
    default_dof_pos: tuple[float, ...] = ()
    # Driven gripper joint home. MUST stay at the followers' authored rest (0 for all three
    # grippers here) — mimic linkages detonate if spawned constraint-violating. Open by command.
    default_gripper_pos: float = 0.0
    # Posture the task-space nullspace pulls toward; () -> use default_dof_pos.
    nullspace_dof_pos: tuple[float, ...] = ()
    # Zero authored joint friction through the physx view at bind (the Z1 authors 1.0-2.0 per
    # joint — a force-scaled Coulomb brake; see the xArm7 robot for the same pattern). NOTE: any
    # later `sim.reset()` re-parses authored values — re-apply after one.
    zero_joint_friction: bool = False
    # Arm gravity compensation (PhysX: spawn the robot's bodies with gravity disabled — the
    # analog of the franka cfg's Newton-only `gravcomp`). For weak/light arms whose effort
    # ratings cannot support their own weight under gravity-blind torque control.
    gravity_compensation: bool = False
    usd: str = ""  # "" -> the vendored composite under assets/composites/<registry name>/


class _AttachedArmRobot(BaseRobot):
    """Shared machinery for the composites; concrete robots pin the structural class attrs."""

    control_modes: tuple[str, ...] = ("osc", "impedance", "joint")  # osc default

    cfg: AttachedArmRobotCfg

    #: structural (per concrete robot)
    NAME: ClassVar[str]
    ARM_JOINTS: ClassVar[tuple[str, ...]]
    GRIPPER_JOINTS: ClassVar[tuple[str, ...]]  # the DRIVEN joint(s) only
    PASSIVE_JOINTS: ClassVar[tuple[str, ...]] = ()  # mimic followers
    PASSIVE_KEEP_AUTHORED: ClassVar[bool] = False  # True -> keep the followers' authored drives
    EXTRA_JOINTS: ClassVar[tuple[str, ...]] = ()  # non-arm, non-gripper (e.g. Sawyer's head_pan)
    EE_BODY: ClassVar[str]  # the LAST ARM LINK (see module docstring)
    ARM_HOME: ClassVar[tuple[float, ...]]  # class-default ready pose (cfg.default_dof_pos overrides)

    # Action/target rate (s) — same scheme as FrankaRobot.
    TORQUE_CONTROL_DT: float = 1.0 / 15.0
    JOINT_CONTROL_DT: float = 0.02

    def __init__(self, cfg: AttachedArmRobotCfg | None = None) -> None:
        cfg = cfg or AttachedArmRobotCfg()
        if not cfg.usd:
            cfg.usd = str(
                Path(__file__).resolve().parent / "assets" / "composites" / self.NAME / f"{self.NAME}.usd"
            )
        super().__init__(cfg)

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """The composite articulation, authored from scratch over the vendored combined USD."""
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg

        c = self.cfg
        torque_mode = self.control_mode in ("impedance", "osc")
        home = c.default_dof_pos or self.ARM_HOME
        joint_pos: dict[str, float] = {}
        for expr, q in zip(self._arm_joint_exprs_ordered(), home):
            joint_pos[expr] = float(q)
        for gj in self.GRIPPER_JOINTS:
            joint_pos[gj] = c.default_gripper_pos
        actuators: dict[str, Any] = {
            f"{self.NAME}_arm": ImplicitActuatorCfg(
                joint_names_expr=list(self.ARM_JOINTS),
                stiffness=0.0 if torque_mode else c.arm_stiffness,
                damping=0.0 if torque_mode else c.arm_damping,
                effort_limit_sim=c.arm_effort_limit,
            ),
            f"{self.NAME}_gripper": ImplicitActuatorCfg(
                joint_names_expr=list(self.GRIPPER_JOINTS),
                stiffness=c.gripper_stiffness,
                damping=c.gripper_damping,
                effort_limit_sim=c.gripper_effort_limit,
            ),
        }
        if self.PASSIVE_JOINTS:
            actuators[f"{self.NAME}_passive"] = ImplicitActuatorCfg(
                joint_names_expr=list(self.PASSIVE_JOINTS),
                stiffness=None if self.PASSIVE_KEEP_AUTHORED else 0.0,
                damping=None if self.PASSIVE_KEEP_AUTHORED else 0.0,
            )
        if self.EXTRA_JOINTS:
            actuators[f"{self.NAME}_extra"] = ImplicitActuatorCfg(  # hold non-task joints in place
                joint_names_expr=list(self.EXTRA_JOINTS), stiffness=50.0, damping=5.0
            )
        rigid = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=bool(c.gravity_compensation), max_depenetration_velocity=5.0
        )
        return {
            self.name: ArticulationCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{self.prim_name}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.usd,
                    rigid_props=rigid,
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False,  # linkage fingers interleave; assets ship no filters
                        solver_position_iteration_count=8,
                        solver_velocity_iteration_count=0,
                        fix_root_link=c.fixed_base,
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(pos=c.base_pos, rot=c.base_rot, joint_pos=joint_pos),
                actuators=actuators,
                soft_joint_pos_limit_factor=1.0,
            )
        }

    def _arm_joint_exprs_ordered(self) -> list[str]:
        """Per-joint init expressions, in the arm's kinematic order. Defaults to expanding a
        single `<stem>[a-b]` regex; robots with irregular names override ARM_JOINT_LIST."""
        lst = getattr(self, "ARM_JOINT_LIST", None)
        if lst:
            return list(lst)
        # expand "stem[i-j]" (the common case)
        import re

        m = re.fullmatch(r"(.*)\[(\d)-(\d)\](.*)", self.ARM_JOINTS[0])
        assert m, f"{self.NAME}: provide ARM_JOINT_LIST for irregular joint names"
        stem, lo, hi, tail = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        return [f"{stem}{i}{tail}" for i in range(lo, hi + 1)]

    # ----- lifecycle ------------------------------------------------------------------------------
    def on_bind(self, env: BaseEnv) -> None:
        self.articulation: Articulation = env.iscene[self.name]
        if self.cfg.zero_joint_friction:
            art = self.articulation
            fr = art.root_physx_view.get_dof_friction_coefficients()
            fr[:] = 0.0
            art.root_physx_view.set_dof_friction_coefficients(fr, torch.arange(env.num_envs, device="cpu"))

    def build_controller(self) -> CompositeController:
        torque_mode = self.control_mode in ("impedance", "osc")
        ctrl_dt = self.TORQUE_CONTROL_DT if torque_mode else self.JOINT_CONTROL_DT
        gripper = JointController(JointControllerCfg(self.GRIPPER_JOINTS, dt=ctrl_dt), command_type="position")
        if torque_mode:
            ts_cfg = TaskSpaceControllerCfg(
                dt=ctrl_dt,
                ee_body=self.EE_BODY,
                arm_joint_names=self.ARM_JOINTS,
                nullspace_dof_pos=self.cfg.nullspace_dof_pos or self.cfg.default_dof_pos or self.ARM_HOME,
                ema_factor=0.2,
            )
            cls = TaskSpaceImpedanceController if self.control_mode == "impedance" else OperationalSpaceController
            arm: Any = cls(ts_cfg)
        elif self.control_mode == "joint":
            arm = JointController(JointControllerCfg(self.ARM_JOINTS, dt=ctrl_dt), command_type="position")
        else:
            raise ValueError(f"unknown {self.NAME} control_mode {self.control_mode!r}; known: {self.control_modes}")
        return CompositeController([arm, gripper])

    def reset(self, env_ids: torch.Tensor) -> None:
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

    # ----- description ----------------------------------------------------------------------------
    ARM_DESC: ClassVar[str] = ""
    GRIP_DESC: ClassVar[str] = ""

    def describe(self) -> str:
        mode = self.control_mode
        n_arm = len(self._arm_joint_exprs_ordered())
        if mode in ("osc", "impedance"):
            law = "operational-space control" if mode == "osc" else "task-space impedance"
            arm = f"{n_arm} arm joints by {law} (joint torque); the action is 6 end-effector pose deltas"
        else:
            arm = f"{n_arm} arm joints by direct position targets"
        return (
            f"{self.ARM_DESC}, fixed to the table, with {self.GRIP_DESC}. Control mode '{mode}': "
            f"{arm}, plus {len(self.GRIPPER_JOINTS)} gripper joint(s) by direct position target. "
            f"Action dim {self.action_dim}."
        )


# ----- the seven concrete composites --------------------------------------------------------------

_ROBOTIQ_PASSIVE = (
    "right_outer_knuckle_joint",
    "right_inner_finger_joint",
    "right_inner_finger_knuckle_joint",
    "left_inner_finger_knuckle_joint",
    "left_inner_finger_joint",
)


@ROBOTS.register("z1_lite6g")
class Z1Lite6GRobot(_AttachedArmRobot):
    NAME = "z1_lite6g"
    ARM_JOINTS = ("joint[1-6]",)
    GRIPPER_JOINTS = ("finger_joint1", "finger_joint2")  # both jaws driven, mirrored targets
    # (the asset's mimic on finger_joint2 is stripped in the composite: a zero-gain mimic
    # follower jitters; the piper drives its mirrored fingers the same way)
    EE_BODY = "link06"
    ARM_HOME = (0.0, 1.2, -1.0, 0.0, 0.0, 0.0)
    ARM_DESC = "A Unitree Z1 arm (small 6-DOF, ~0.74 m reach)"
    GRIP_DESC = "a UFACTORY Lite6 two-jaw gripper (one driven prismatic jaw, its twin mimicked)"


@ROBOTS.register("rizon4_2f85")
class Rizon42F85Robot(_AttachedArmRobot):
    NAME = "rizon4_2f85"
    ARM_JOINTS = ("joint[1-7]",)
    GRIPPER_JOINTS = ("finger_joint",)
    PASSIVE_JOINTS = _ROBOTIQ_PASSIVE
    EE_BODY = "link7"
    ARM_HOME = (0.0, -0.7, 0.0, 1.5, 0.0, 0.7, 0.0)
    ARM_DESC = "A Flexiv Rizon 4 arm (7-DOF force-controlled cobot, ~0.8 m reach)"
    GRIP_DESC = "a Robotiq 2F-85 gripper (one driven knuckle joint, five mimic followers)"


@ROBOTS.register("gen3n7_2f85")
class Gen3N72F85Robot(_AttachedArmRobot):
    NAME = "gen3n7_2f85"
    ARM_JOINTS = ("joint_[1-7]",)
    GRIPPER_JOINTS = ("finger_joint",)
    PASSIVE_JOINTS = _ROBOTIQ_PASSIVE
    EE_BODY = "bracelet_link"
    ARM_HOME = (0.0, 0.65, 0.0, 1.89, 0.0, 0.6, -1.57)  # the isaaclab_assets preset's ready pose
    ARM_DESC = "A Kinova Gen3 arm (7-DOF cobot, ~0.9 m reach)"
    GRIP_DESC = "a Robotiq 2F-85 gripper (one driven knuckle joint, five mimic followers)"


@ROBOTS.register("sawyer_egu50")
class SawyerEGU50Robot(_AttachedArmRobot):
    NAME = "sawyer_egu50"
    ARM_JOINTS = ("right_j[0-6]",)
    ARM_JOINT_LIST = tuple(f"right_j{i}" for i in range(7))
    GRIPPER_JOINTS = ("Jaw_Drive",)
    PASSIVE_JOINTS = ("HUB_GBA_2_TRANS_3",)
    PASSIVE_KEEP_AUTHORED = True  # the EGU's mimic'd jaw ships an authored assist drive
    EXTRA_JOINTS = ("head_pan",)  # not part of the task chain; held in place
    EE_BODY = "right_l6"
    ARM_HOME = (0.0, -1.18, 0.0, 2.18, 0.0, 0.57, 3.14)  # Sawyer's tucked ready pose
    ARM_DESC = "A Rethink Sawyer arm (7-DOF cobot, ~1.26 m reach, screen head)"
    GRIP_DESC = "a Schunk EGU-50 two-jaw parallel gripper (one driven prismatic jaw, its twin mimicked)"


@ROBOTS.register("crx10ial_2f85")
class CRX10iAL2F85Robot(_AttachedArmRobot):
    NAME = "crx10ial_2f85"
    ARM_JOINTS = ("joint_[1-6]",)
    GRIPPER_JOINTS = ("finger_joint",)
    PASSIVE_JOINTS = _ROBOTIQ_PASSIVE
    EE_BODY = "link_6"
    ARM_HOME = (0.0, 0.4, -0.4, 0.0, -1.1, 0.0)
    ARM_DESC = "A Fanuc CRX-10iA/L arm (6-DOF cobot, ~1.4 m reach)"
    GRIP_DESC = "a Robotiq 2F-85 gripper (one driven knuckle joint, five mimic followers)"


@ROBOTS.register("tm12_2f85")
class TM122F85Robot(_AttachedArmRobot):
    NAME = "tm12_2f85"
    ARM_JOINTS = (
        "shoulder_1_joint",
        "shoulder_2_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    ARM_JOINT_LIST = ARM_JOINTS
    GRIPPER_JOINTS = ("finger_joint",)
    PASSIVE_JOINTS = _ROBOTIQ_PASSIVE
    EE_BODY = "link_6"
    ARM_HOME = (0.0, -0.5, 1.6, -1.1, 1.57, 0.0)
    ARM_DESC = "A Techman TM12 arm (6-DOF cobot, ~1.3 m reach)"
    GRIP_DESC = "a Robotiq 2F-85 gripper (one driven knuckle joint, five mimic followers)"


@ROBOTS.register("festo_2f85")
class Festo2F85Robot(_AttachedArmRobot):
    NAME = "festo_2f85"
    ARM_JOINTS = ("a[1-6]",)
    GRIPPER_JOINTS = ("finger_joint",)
    PASSIVE_JOINTS = _ROBOTIQ_PASSIVE
    EE_BODY = "link_6"
    ARM_HOME = (0.0, -0.6, -1.4, 0.0, -0.8, 0.0)  # a3 in [-205, 65] deg: the elbow bends negative
    ARM_DESC = "A Festo pneumatic cobot arm (6-DOF)"
    GRIP_DESC = "a Robotiq 2F-85 gripper (one driven knuckle joint, five mimic followers)"
