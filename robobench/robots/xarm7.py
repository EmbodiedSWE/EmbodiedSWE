"""XArm7Robot — a UFACTORY xArm7 arm + a CONFIGURABLE end-effector, for table-top manipulation.

A fixed-base 7-DOF arm (~0.70 m reach, 3.5 kg payload). The end-effector is a cfg dial
(`XArm7RobotCfg.gripper`): "xarm" (default) is the vendor's linkage gripper baked into the stock
USD; "panda_hand" spawns a baked composite from `assets/composites/xarm7_panda_hand/`
(written offline by `assets/gripper/make_composites.py`: the arm USD's own `Variant_Set=None`
goes bare-flange and the gripper USD is welded at link7's rest frame). Per-gripper runtime
structure (joint names, mimic followers, control frame, actuator defaults) lives in
`XArm7Robot.GRIPPERS`; the action layout follows the driven-joint count (one-joint linkages: 7;
two-finger hands: 8).

The vendored USD set (`assets/xarm7/` + `assets/xarm_gripper/`, from the Isaac Sim 5.1 asset
library's `Robots/Ufactory/{xarm7,xarm_gripper}` — the robot's default `Variant_Set` composes the
gripper in) keeps NVIDIA's authored effort limits (50/50/30/30/30/20/20 N*m — UFACTORY's real
joint ratings). One local patch, applied at vendor time: the gripper attach joint
(`/UF_ROBOT/gripper/root_joint`, link7 -> gripper base) shipped with its own
`PhysicsArticulationRootAPI`, which splits the robot into TWO articulations under PhysX — the API
is removed in the vendored copy so IsaacLab sees one 13-joint articulation.

Each gripper is a fingertip linkage with ONE actuated joint (0 = fully open -> ~0.85 rad closed);
the five follower joints ride PhysX mimic constraints authored in the asset, so they get a
zero-gain actuator group and are never commanded.

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
from robobench.core.assets import asset_path
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
    # Gripper drive-joint PD gains (always position-controlled; linkage followers stay passive).
    # None -> the gripper choice's own defaults from `XArm7Robot.GRIPPERS` — units differ per
    # gripper (N*m/rad for the revolute linkages, N/m for the panda hand's prismatic fingers).
    gripper_stiffness: float | None = None
    gripper_damping: float | None = None
    gripper_effort_limit: float | None = None
    # Home posture of the 7 arm joints: a forward-facing ready pose with the elbow well bent and
    # the tool already pitched toward the table (zero = the arm pointing straight up; joint4's
    # lower limit is only -11 deg, so a near-straight elbow leaves the wrist no room to flip the
    # tool down). Retune per task.
    default_dof_pos: tuple[float, ...] = (0.0, -0.35, 0.0, 1.15, 0.0, 1.5, 0.0)
    # Gripper drive-joint home; None -> the gripper choice's default (the linkages spawn at 0 =
    # open, the panda hand at 0.04 m = open — its sense is inverted and its units are metres).
    default_gripper_pos: float | None = None
    # Which end-effector rides on the arm — the GRIPPER-LEVEL dial. "xarm" (default) is the
    # vendor's linkage gripper baked into the stock USD; every other choice spawns the baked
    # composite assets/composites/xarm7_<gripper>/. Known choices live in `XArm7Robot.GRIPPERS`;
    # adding one = add the gripper asset + a make_composites.py table entry + a GRIPPERS entry.
    gripper: str = "xarm"
    # Posture the task-space nullspace pulls toward; () -> use default_dof_pos.
    nullspace_dof_pos: tuple[float, ...] = ()
    # Arm gravity compensation (PhysX: spawn the robot's bodies with gravity disabled — the analog
    # of the franka cfg's Newton-only `gravcomp`, and the same dial `Jaco2N7RobotCfg` /
    # `AttachedArmRobotCfg` carry). The task-space laws are gravity-blind (tau = J^T F_task +
    # tau_null, no gravity term), so under torque control this arm cannot hold itself against
    # gravity across a long reach. The real xArm7's controller gravity-compensates.
    gravity_compensation: bool = False
    xarm7_usd: str = ""  # "" -> the vendored robots/assets/xarm7/xarm7.usd



@ROBOTS.register("xarm7")
class XArm7Robot(BaseRobot):
    """UFACTORY xArm7 arm + a configurable end-effector (`cfg.gripper`; "xarm" = the vendor's
    linkage gripper, the default). `apply_action` delegates to the controller for `control_mode`;
    the gripper's driven joint is always a direct position target, the 7 arm joints by the mode's
    arm controller (torque-mode task-space control, or position JointController)."""

    control_modes: tuple[str, ...] = ("osc", "impedance", "joint")  # osc default

    cfg: XArm7RobotCfg

    ARM_JOINTS: tuple[str, ...] = ("joint[1-7]",)

    #: One entry per gripper choice: which joints are driven, which are passive mimic followers,
    #: the control frame, actuator defaults (used where the cfg leaves the gripper fields None),
    #: the spawn pose, and the describe() text. Non-default grippers also name their USD and
    #: the spawn pose, and the describe() text. The action size follows the driven-joint count
    #: (one-joint linkages: 7; the two-finger panda hand: 8).
    GRIPPERS: dict[str, dict[str, Any]] = {
        "xarm": dict(
            gripper_joints=("drive_joint",),
            passive_joints=("left_finger_joint", "left_inner_knuckle_joint",
                            "right_inner_knuckle_joint", "right_outer_knuckle_joint",
                            "right_finger_joint"),
            ee_body="xarm_gripper_base_link",  # coincident with link7, 0.54 kg
            stiffness=100.0, damping=10.0, effort=None,  # N*m/rad; None -> the authored 1000
            default_pos=0.0,  # 0 = open ~85 mm, 0.85 rad = closed
            desc="the vendor's parallel linkage gripper (0 = open ~85 mm, 0.85 rad = closed; the "
                 "finger linkage follows by mimic constraint)",
        ),
        "panda_hand": dict(
            gripper_joints=("panda_finger_joint1", "panda_finger_joint2"),
            passive_joints=(),
            ee_body="link7",
            stiffness=2000.0, damping=100.0, effort=200.0,  # N/m, N*s/m, N — prismatic
            default_pos=0.04,  # metres; 0 = closed, 0.04 = open (the franka convention)
            desc="the Franka hand (two driven prismatic fingers, METRES: 0 = closed, "
                 "0.04 = open ~80 mm)",
        ),
    }

    # Action/target rate (s) — same scheme as FrankaRobot.
    TORQUE_CONTROL_DT: float = 1.0 / 15.0
    JOINT_CONTROL_DT: float = 0.02

    def __init__(self, cfg: XArm7RobotCfg | None = None) -> None:
        cfg = cfg or XArm7RobotCfg()
        spec = self.GRIPPERS.get(cfg.gripper)
        if spec is None:
            raise ValueError(f"unknown xArm7 gripper {cfg.gripper!r}; known: {sorted(self.GRIPPERS)}")
        # instance-level structural attrs, resolved from the gripper choice
        self.GRIPPER_JOINTS: tuple[str, ...] = spec["gripper_joints"]
        self.PASSIVE_JOINTS: tuple[str, ...] = spec["passive_joints"]
        self.EE_BODY: str = spec["ee_body"]
        self._grip_desc: str = spec["desc"]
        self._gspec = spec  # actuator/spawn defaults for cfg fields left None
        if cfg.gripper == "xarm":
            # Scene grasp-weld contract keys (robobench.core.grasp_weld): the vendor linkage's
            # mirrored fingertip pads ride the finger bodies; one driven joint, followers by
            # mimic. (The panda_hand choice needs no entry — the contract's panda defaults fit
            # the composite; instance-level because the hand is a cfg choice.)
            self.GRASP_IFACE = dict(
                hand_body="xarm_gripper_base_link",
                finger_joints="drive_joint",
                approach=(0.0, 0.0, 1.0),
                pinch_offset=0.1287,  # hand -> pad centre along approach (live-measured)
                closure=("aperture",),
                pad_bodies=("left_finger", "right_finger"),
                pinch_axis=(0.0, 1.0, 0.0),
                sep_off=0.052,  # finger-origin -> pad inner face, 26 mm per side
                stall_vel=0.05,  # rad/s on the single driven linkage joint
                band_dist=0.018,  # 67 mm pads bite ~10 mm deep near a part's top edge: the
                # pinch centre rides off the grip band by construction (measured 11.5 mm)
            )
        super().__init__(cfg)

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """The xArm7 articulation, authored from scratch over the vendored USD (no isaaclab_assets
        preset exists for this arm). Arm gains follow the mode; the mimic followers stay passive."""
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg

        c = self.cfg
        torque_mode = self.control_mode in ("impedance", "osc")
        # The USD to spawn: an explicit cfg override, the stock arm+vendor-gripper USD, or the
        # gripper choice's baked composite (written by assets/gripper/make_composites.py).
        # Resolved here at spawn time, never stored on the cfg — an eagerly-stored path would
        # survive `dataclasses.replace(cfg, gripper=...)` and silently spawn the old gripper.
        assets_dir = asset_path(Path(__file__).resolve().parent / "assets")
        usd = c.xarm7_usd or (
            str(assets_dir / "xarm7" / "xarm7.usd") if c.gripper == "xarm"
            else str(assets_dir / "composites" / f"xarm7_{c.gripper}" / f"xarm7_{c.gripper}.usda"))
        gs = self._gspec  # cfg gripper fields left None fall back to the gripper's own defaults
        grip_pos = c.default_gripper_pos if c.default_gripper_pos is not None else gs["default_pos"]
        actuators: dict[str, Any] = {
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
                stiffness=c.gripper_stiffness if c.gripper_stiffness is not None else gs["stiffness"],
                damping=c.gripper_damping if c.gripper_damping is not None else gs["damping"],
                effort_limit_sim=c.gripper_effort_limit if c.gripper_effort_limit is not None else gs["effort"],
            ),
        }
        if self.PASSIVE_JOINTS:
            actuators["xarm7_gripper_passive"] = ImplicitActuatorCfg(
                joint_names_expr=list(self.PASSIVE_JOINTS),
                stiffness=0.0,  # mimic constraints drive these; keep the drives silent
                damping=0.0,
            )
        return {
            self.name: ArticulationCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{self.prim_name}",  # cfg.name-namespaced (default "Robot")
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=bool(c.gravity_compensation),
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
                        **{gj: grip_pos for gj in self.GRIPPER_JOINTS},
                    },
                ),
                actuators=actuators,
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
        # Zero it through the PHYSX VIEW, not `write_joint_friction_coefficient_to_sim`: for
        # implicit actuators that call updates only IsaacLab's data buffer, leaving PhysX's own
        # coefficients at the authored 1.0 (`art.data.joint_friction_coeff` then reads 0.0 while
        # the brake is still live). Same escape hatch `AttachedArmRobot.on_bind` uses.
        art = self.articulation
        fr = art.root_physx_view.get_dof_friction_coefficients()
        fr[:] = 0.0
        art.root_physx_view.set_dof_friction_coefficients(fr, torch.arange(env.num_envs, device="cpu"))

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
        n_grip = len(self.GRIPPER_JOINTS)
        return (
            f"A UFACTORY xArm7 arm (7-DOF cobot, ~0.70 m reach) with {self._grip_desc}, fixed to "
            f"the table. Control mode '{mode}': {arm}, plus {n_grip} gripper joint(s) by direct "
            f"position target. Action dim {self.action_dim}."
        )
