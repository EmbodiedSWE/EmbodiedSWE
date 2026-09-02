"""G1Robot — Unitree G1 humanoid, fixed base, upper-body manipulation.

The first concrete robobench embodiment. Wraps Isaac's `G1_29DOF_CFG` (29-DOF G1 with the
three-finger hands) but pins the pelvis (`fix_root_link=True`) so only the upper body acts — the
fixed-base manipulator of `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0`. The USD is **vendored
locally** (`robots/assets/g1/`) so the package stays relocatable (no Nucleus path).

Control modes (the actuation of the *same* hardware — see CLAUDE.md "embodiment vs control mode").
Both share one shape: a controller over the 17 arm+waist DOFs **+** direct joint targets for the 14
three-finger-hand DOFs, composed with `composite`. Switching the mode swaps only the arm controller:
  - `joint`   — arms+waist by `JointController` (action = 17 + 14 = 31 joint position targets).
  - `pink_ik` — arms+waist by Pink IK (action = two wrist poses = 14, + 14 hands). [built next]

This is the first test of the layer design: the same scene + this robot, switch `control_mode`,
and `action_dim` / the controller follow. Heavy imports (isaaclab / isaaclab_assets) are deferred so
importing this module — and registering the robot — stays app-free.

================================================================================================
TODO (NOT built yet) — HAND VARIANT as a build-time option, like `fixed_base`. Notes for future:

The G1 hand is **build-time hardware**, not a control mode (new hardware -> new robot variant). Isaac
offers three hands and varies them two ways:
  - USD variant sets: `g1.usd` has `left_hand`/`right_hand` variant sets = {ThreeFinger, Inspire,
    None}, default ThreeFinger.
  - Separate cfgs: `G1_29DOF_CFG` -> `g1.usd` (ThreeFinger); `G1_INSPIRE_FTP_CFG` = a copy with
    `spawn.usd_path = ".../g1_29dof_inspire_hand.usd"`, `activate_contact_sensors=True`, and a
    **redefined** `actuators["hands"]` for the inspire joints.

To add it here: a `hand: str = "three_finger"` field on `G1RobotCfg` driving, in `assets()`,
(1) which USD / USD-variant to spawn, and (2) the hand joint set — so `HAND_JOINTS` (the joint-name
patterns) and the hand controller's `action_dim` become **variant-dependent** (3-finger=14 DOF !=
5-finger inspire != none=0). The whole `composite([arms, joint(hands)])` shape still holds; only the
hands leaf (its joints / DOF count) changes.

CAVEAT: the vendored `robots/assets/g1/g1.usd` is **ThreeFinger-only** — the Inspire variant was
pruned (~75 MB) by `robots/assets/fetch_g1.py` (PRUNE list). So `three_finger` works and `none` likely
does (adds nothing); `inspire` needs re-vendoring `g1_29dof_inspire_hand.usd` (or un-pruning + re-fetch).
================================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from robobench.controllers import (
    CompositeController,
    FrameTaskCfg,
    JointController,
    JointControllerCfg,
    PinkIKController,
    PinkIKControllerCfg,
)
from robobench.core import ROBOTS, BaseRobot, BaseRobotCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from robobench.core import BaseEnv

@dataclass
class G1RobotCfg(BaseRobotCfg):
    """Config for `G1Robot`. `control_mode` is inherited from `BaseRobotCfg` ("" -> first declared).
    `base_pos`/`base_rot` are placement dials; `fixed_base` + the asset path are structural
    (build-time)."""

    # Base fixity — a build-time variant (Isaac's `spawn.articulation_props.fix_root_link`): True welds
    # the pelvis to the world (stationary upper-body manipulator); False = mobile base,
    # which then needs a leg controller to balance/walk (loco-manip — not built yet). Not a control
    # mode: it changes the articulation's structure (changing it = a rebuild).
    fixed_base: bool = True
    # Spawn pose of the pelvis — the G1's natural standing pose, but a placement/reachability dial an
    # agent may adjust (move closer to the table, rotate to face it).
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.75)
    base_rot: tuple[float, float, float, float] = (0.7071, 0.0, 0.0, 0.7071)  # wxyz; faces +y

    # Upper-body actuator PD gains (stiffness Kp / damping Kd) — THE "articulation PD" that tracks the
    # position targets a `JointController`/IK writes. Defaults are Isaac's `G1_29DOF_CFG` values;
    # exposed here so contact compliance is easy to dial (softer arms -> more compliant
    # insertion — research dir. 5). Applied to the copied cfg's actuators in `assets()`. (Legs/feet
    # are DCMotor loco actuators, irrelevant to the fixed-base manipulator, so left at Isaac defaults.)
    arm_stiffness: float = 3000.0
    arm_damping: float = 10.0
    hand_stiffness: float = 20.0
    hand_damping: float = 2.0
    waist_stiffness: float = 5000.0
    waist_damping: float = 5.0

    g1_usd: str = ""  # "" -> the vendored robots/assets/g1/g1.usd
    g1_urdf: str = ""  # "" -> the vendored kinematics URDF (used by the pink_ik control mode)
    # Optional joint-name -> angle overrides applied to the articulation's INITIAL pose (and
    # therefore to what reset() restores). Isaac's G1 default rests both wrists at
    # (+/-0.15, +0.20, +0.09) relative to the pelvis -- for a tabletop cell that is INSIDE
    # the work area, resting the hands among (or into) the props; solves then have to
    # special-case their first move away from home ("re-homing rakes the hand through the
    # produce", the clear_organics lesson). A binding can instead park the arms bent and
    # retracted here.
    init_joint_overrides: dict | None = None

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parent / "assets" / "g1"
        self.g1_usd = self.g1_usd or str(assets / "g1.usd")
        self.g1_urdf = self.g1_urdf or str(assets / "g1_29dof_with_hand_only_kinematics.urdf")


@ROBOTS.register("g1")
class G1Robot(BaseRobot):
    """G1 humanoid manipulator. Base fixity is a build-time variant via `cfg.fixed_base` (default True
    = pelvis welded to the world, a stationary upper-body manipulator; False = mobile,
    which needs a leg controller, not built yet). `apply_action` delegates to the controller selected
    by `control_mode`; `action_dim` follows it. The 14 three-finger-hand DOFs are always direct joint
    targets; the 17 arm+waist DOFs are driven by the mode's arm controller."""

    control_modes: tuple[str, ...] = ("joint", "pink_ik")
    cfg: G1RobotCfg

    # End-effector bodies (left, right): the wrist links the pink_ik frames track and
    # generic tooling (robot_binding_smoke) reads.
    EE_BODIES: tuple[str, str] = ("left_wrist_yaw_link", "right_wrist_yaw_link")

    # Upper-body control period (s): ~50 Hz (Isaac `Isaac-PickPlace-FixedBaseUpperBodyIK-G1`, decim 4 @
    # 200 Hz). `bind` rounds to the nearest sim-step multiple -> period 2 (60 Hz) at the 120 Hz table.
    CONTROL_DT: float = 0.02

    # Upper-body joint groups (regexes, by G1_29DOF_CFG's actuator names; find_joints resolves the
    # indices). arms+waist = 17 (per arm: shoulder pitch/roll/yaw, elbow, wrist pitch/roll/yaw; +
    # waist yaw/roll/pitch); hands = 14 (three-finger index/middle/thumb).
    ARM_WAIST_JOINTS: tuple[str, ...] = (
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_.*_joint",
        "waist_.*_joint",
    )
    HAND_JOINTS: tuple[str, ...] = (".*_index_.*", ".*_middle_.*", ".*_thumb_.*")

    def __init__(self, cfg: G1RobotCfg | None = None) -> None:
        super().__init__(cfg or G1RobotCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """The G1 articulation: `G1_29DOF_CFG` re-pointed at the vendored USD, base fixed per
        `cfg.fixed_base`, spawned at the configured base pose, with the upper-body PD gains from cfg.
        `.copy()` is a deep copy, so mutating these actuators doesn't touch the shared module cfg."""
        from isaaclab_assets.robots.unitree import G1_29DOF_CFG

        c = self.cfg
        robot = G1_29DOF_CFG.copy()  # type: ignore[attr-defined]
        robot.prim_path = f"{{ENV_REGEX_NS}}/{self.prim_name}"  # cfg.name-namespaced (default "Robot")
        robot.spawn.usd_path = c.g1_usd
        robot.spawn.articulation_props.fix_root_link = c.fixed_base  # weld pelvis to world if fixed
        robot.init_state.pos = c.base_pos
        robot.init_state.rot = c.base_rot
        if c.init_joint_overrides:
            robot.init_state.joint_pos = {**robot.init_state.joint_pos,
                                          **c.init_joint_overrides}
        # The articulation PD (Isaac runs the loop in PhysX for these implicit actuators) — from cfg.
        robot.actuators["arms"].stiffness = c.arm_stiffness
        robot.actuators["arms"].damping = c.arm_damping
        robot.actuators["hands"].stiffness = c.hand_stiffness
        robot.actuators["hands"].damping = c.hand_damping
        robot.actuators["waist"].stiffness = c.waist_stiffness
        robot.actuators["waist"].damping = c.waist_damping
        return {self.name: robot}

    # ----- lifecycle (hooks; the base orchestrates bind -> on_bind -> build_controller) ---------
    def on_bind(self, env: BaseEnv) -> None:
        """Grab the articulation handle; the base then builds + binds the controller against it."""
        self.articulation: Articulation = env.iscene[self.name]

    def build_controller(self) -> CompositeController:
        """`composite([<arm controller>, joint(hands)])` for the active `control_mode`. Switching the
        mode swaps only the arm controller — the hands are always direct joint targets:
          - 'joint'   -> arm+waist by JointController  (action = 17 joint targets + 14 hands = 31)
          - 'pink_ik' -> arm+waist by whole-body Pink IK (action = 2 wrist poses = 14 + 14 hands = 28)
        """
        # G1's arm + hand actuators are position-PD (G1_29DOF_CFG), so these write position targets
        # (raw pass-through: identity scale/offset; the PD gains live in the actuators, not here).
        hands = JointController(JointControllerCfg(self.HAND_JOINTS, dt=self.CONTROL_DT), command_type="position")
        if self.control_mode == "joint":
            arms: Any = JointController(
                JointControllerCfg(self.ARM_WAIST_JOINTS, dt=self.CONTROL_DT), command_type="position"
            )
        elif self.control_mode == "pink_ik":
            # Pink IK targets the two wrists; base = pelvis. URDF link names = USD names prefixed with
            # the robot name; null-space holds the (redundant) shoulders + waist near home.
            pfx = "g1_29dof_with_hand_rev_1_0_"
            arms = PinkIKController(
                PinkIKControllerCfg(
                    dt=self.CONTROL_DT,
                    urdf_path=self.cfg.g1_urdf,
                    base_link="pelvis",
                    base_link_frame=pfx + "pelvis",
                    frames=(FrameTaskCfg(pfx + "left_wrist_yaw_link"), FrameTaskCfg(pfx + "right_wrist_yaw_link")),
                    joint_names=self.ARM_WAIST_JOINTS,  # the 17-DOF chain the IK solves over
                    nullspace_joints=(
                        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                        "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
                    ),
                )
            )
        else:
            raise ValueError(f"unknown G1 control_mode {self.control_mode!r}; known: {self.control_modes}")
        return CompositeController([arms, hands])

    def reset(self, env_ids: torch.Tensor) -> None:
        """Home pose: default joint state, and all joint position targets held at default (so the
        un-driven legs stay put). Root written to the (fixed) spawn pose + env origin."""
        art = self.articulation
        jp = art.data.default_joint_pos[env_ids].clone()
        jv = art.data.default_joint_vel[env_ids].clone()
        art.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
        art.set_joint_position_target(jp, env_ids=env_ids)  # hold every joint; step() overrides the driven subset
        art.set_joint_effort_target(torch.zeros_like(jp), env_ids=env_ids)
        root = art.data.default_root_state[env_ids].clone()
        root[:, 0:3] += self.env.iscene.env_origins[env_ids]
        art.write_root_state_to_sim(root, env_ids)
        if self.controller is not None:
            self.controller.reset(env_ids)

    # NOTE: apply_action / action_dim are inherited from BaseRobot — they delegate to `self.controller`
    # (composite of arms+hands). Each sub-controller writes its own joints through the actuator sink it
    # captured at bind (the default sink routes through `self.articulation`), so no action code here.

    # get_state / set_state are inherited from BaseRobot (single-articulation sim state + controller).

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        subs = getattr(self.controller, "controllers", None)  # composite([arms, hands]) once bound
        n_arm = len(subs[0].joint_ids) if subs else 17
        n_hand = len(subs[1].joint_ids) if subs else 14
        base = (
            "fixed at the pelvis (legs locked), a stationary upper-body manipulator"
            if self.cfg.fixed_base
            else "on a mobile base"
        )
        return (
            f"A Unitree G1 humanoid with three-finger hands, {base}. Control mode '{self.control_mode}': "
            f"{n_arm} arm+waist joints driven by the "
            f"{'joint controller (direct position targets)' if self.control_mode == 'joint' else 'Pink IK solver (wrist poses)'}, "
            f"plus {n_hand} hand joints by direct position target. Action dim {self.action_dim}."
        )
