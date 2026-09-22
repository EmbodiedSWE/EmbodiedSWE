"""G1Robot — Unitree G1 humanoid, fixed base, upper-body manipulation.

The first concrete robobench embodiment. Wraps Isaac's `G1_29DOF_CFG` (29-DOF G1 with the
three-finger hands) but pins the pelvis (`fix_root_link=True`) so only the upper body acts — the
fixed-base manipulator of `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0`. The USD is **vendored
locally** (`robots/assets/g1/`) so the package stays relocatable (no Nucleus path).

Control modes (the actuation of the *same* hardware — see CLAUDE.md "embodiment vs control mode").
All share one shape: a controller over the 17 arm+waist DOFs **+** direct joint targets for the 14
three-finger-hand DOFs, composed with `composite`. The mode swaps the arm controller, and the two
`loco_*` modes append a third leaf that drives the 12 leg DOFs:
  - `joint`         — arms+waist by `JointController` (action = 17 + 14 = 31 joint position targets).
  - `pink_ik`       — arms+waist by Pink IK (action = two wrist poses = 14, + 14 hands = 28).
  - `loco_pink_ik`  — `pink_ik` on a FREE pelvis + legs by a frozen locomotion policy taking a base
                      command. Action = 28 + 4 = **32**, laid out `[L wrist 7, R wrist 7, hands 14,
                      vx vy wz hip_height]` — deliberately Isaac Lab's own ordering for its G1
                      loco-manipulation task, so their retargeters and datasets line up with ours.
  - `loco_joint`    — the same legs under direct arm joint targets. Action = 31 + 4 = 35.

The `loco_*` modes REQUIRE `fixed_base=False`; `build_controller` refuses the combination rather
than let a welded pelvis silently run a walking policy. See `controllers/loco_policy.py` for what
the base command means and what the checkpoint expects back.

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
from robobench.core.assets import asset_path
from typing import TYPE_CHECKING, Any

import torch

from robobench.controllers import (
    CompositeController,
    FrameTaskCfg,
    JointController,
    JointControllerCfg,
    LocoPolicyController,
    LocoPolicyControllerCfg,
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
    # which then needs a leg controller to balance/walk — that is what the `loco_*` control modes
    # supply. Base fixity is still NOT a control mode: it changes the articulation's structure
    # (changing it = a rebuild), which is why the two are cross-checked in `build_controller`.
    fixed_base: bool = True
    # Spawn pose of the pelvis — the G1's natural standing pose, but a placement/reachability dial an
    # agent may adjust (move closer to the table, rotate to face it).
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.75)
    base_rot: tuple[float, float, float, float] = (0.7071, 0.0, 0.0, 0.7071)  # wxyz; faces +y

    # Upper-body actuator PD gains (stiffness Kp / damping Kd) — THE "articulation PD" that tracks the
    # position targets a `JointController`/IK writes. Defaults are Isaac's `G1_29DOF_CFG` values;
    # exposed here so contact compliance is easy to dial (softer arms -> more compliant
    # insertion — research dir. 5). Applied to the copied cfg's actuators in `assets()`.
    # LEGS AND FEET ARE DELIBERATELY ABSENT from this list and must stay so: they are `G1_29DOF_CFG`'s
    # DCMotor actuators (Kp 100-200 legs, 20 ankles), and the frozen locomotion policy the `loco_*`
    # modes run was trained against exactly those gains. Irrelevant to the fixed-base manipulator;
    # load-bearing the moment the robot walks.
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
    # The frozen locomotion checkpoint the `loco_*` modes run on the legs; "" -> the vendored AGILE
    # policy (robots/assets/fetch_g1_locomotion.py). Swapping this file — for HOMIE, or something
    # trained here — is how you change gaits without touching code, as long as the new checkpoint
    # keeps the observation contract in `controllers/loco_policy.py`.
    loco_policy: str = ""

    def __post_init__(self) -> None:
        assets = asset_path(Path(__file__).resolve().parent / "assets") / "g1"
        self.g1_usd = self.g1_usd or str(assets / "g1.usd")
        self.g1_urdf = self.g1_urdf or str(assets / "g1_29dof_with_hand_only_kinematics.urdf")
        self.loco_policy = self.loco_policy or str(assets / "policies" / "agile_locomotion.pt")


@ROBOTS.register("g1")
class G1Robot(BaseRobot):
    """G1 humanoid manipulator. Base fixity is a build-time variant via `cfg.fixed_base` (default True
    = pelvis welded to the world, a stationary upper-body manipulator; False = mobile, walked by the
    `loco_*` control modes' frozen policy). `apply_action` delegates to the controller selected
    by `control_mode`; `action_dim` follows it. The 14 three-finger-hand DOFs are always direct joint
    targets; the 17 arm+waist DOFs are driven by the mode's arm controller; in the `loco_*` modes the
    12 leg DOFs are driven by the locomotion policy from a 4-dim base command."""

    control_modes: tuple[str, ...] = ("joint", "pink_ik", "loco_pink_ik", "loco_joint")
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
    # Lower-body groups, used only by the `loco_*` modes. `LEG_JOINTS` = the 12 DOFs the locomotion
    # policy drives; `BODY_JOINTS` = the 29 DOFs that appear in its observation, i.e. every joint but
    # the 14 hands. Both are resolved in ARTICULATION order (robobench's `find_joints` and Isaac's
    # `SceneEntityCfg` agree on `preserve_order=False`), which is what the checkpoint was trained on.
    LEG_JOINTS: tuple[str, ...] = (".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint")
    BODY_JOINTS: tuple[str, ...] = ARM_WAIST_JOINTS + LEG_JOINTS

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
        """`composite([<arm controller>, joint(hands)[, loco_policy(legs)]])` for the active
        `control_mode`. The mode swaps the arm controller — the hands are always direct joint
        targets — and the `loco_*` modes append the legs leaf:
          - 'joint'        -> arm+waist by JointController  (action = 17 joint targets + 14 hands = 31)
          - 'pink_ik'      -> arm+waist by whole-body Pink IK (action = 2 wrist poses = 14 + 14 hands = 28)
          - 'loco_pink_ik' -> the above + 4 base-command dims for the legs = 32
          - 'loco_joint'   -> 'joint' + 4 = 35
        """
        mode, loco = self.control_mode, self.control_mode.startswith("loco_")
        if loco and self.cfg.fixed_base:
            raise ValueError(
                f"control_mode {mode!r} drives the legs, but this G1 was built with fixed_base=True "
                "(pelvis welded to the world) — a walking policy against a pinned root produces no "
                "motion and no error. Build it with G1RobotCfg(fixed_base=False)."
            )
        arm_mode = mode[len("loco_"):] if loco else mode

        # G1's arm + hand actuators are position-PD (G1_29DOF_CFG), so these write position targets
        # (raw pass-through: identity scale/offset; the PD gains live in the actuators, not here).
        hands = JointController(JointControllerCfg(self.HAND_JOINTS, dt=self.CONTROL_DT), command_type="position")
        if arm_mode == "joint":
            arms: Any = JointController(
                JointControllerCfg(self.ARM_WAIST_JOINTS, dt=self.CONTROL_DT), command_type="position"
            )
        elif arm_mode == "pink_ik":
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
        if not loco:
            return CompositeController([arms, hands])
        # The legs leaf runs at the same 50 Hz period as the arms, so the composite's period is 4 at
        # this scene's 1/200 s sim dt and both leaves fire together on substep 0 — matching Isaac
        # Lab's decimation=4 @ 200 Hz for the same checkpoint.
        legs = LocoPolicyController(
            LocoPolicyControllerCfg(
                dt=self.CONTROL_DT,
                policy_path=self.cfg.loco_policy,
                joint_names=self.LEG_JOINTS,
                obs_joint_names=self.BODY_JOINTS,
            )
        )
        return CompositeController([arms, hands, legs])

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
        subs = getattr(self.controller, "controllers", None)  # composite([arms, hands[, legs]]) once bound
        n_arm = len(subs[0].joint_ids) if subs else 17
        n_hand = len(subs[1].joint_ids) if subs else 14
        loco = self.control_mode.startswith("loco_")
        n_leg = len(subs[2].joint_ids) if (subs and len(subs) > 2) else 12
        arm_ctrl = (
            "joint controller (direct position targets)"
            if self.control_mode.endswith("joint")
            else "Pink IK solver (absolute wrist poses)"
        )
        base = (
            "standing on its own two legs and free to walk"
            if loco
            else "fixed at the pelvis (legs locked), a stationary upper-body manipulator"
        )
        out = (
            f"A Unitree G1 humanoid with three-finger hands, {base}. Control mode "
            f"'{self.control_mode}': {n_arm} arm+waist joints driven by the {arm_ctrl}, "
            f"plus {n_hand} hand joints by direct position target."
        )
        if loco:
            out += (
                f" The last 4 action dims are a BASE COMMAND for the {n_leg} leg joints — "
                "[vx, vy, wz, hip_height]: forward and sideways speed in m/s and turn rate in rad/s, "
                "all in the robot's own frame, plus the standing height of the pelvis in m. A frozen "
                "locomotion policy turns that command into leg motion and keeps the robot balanced, "
                "so you steer the robot rather than stepping it; all-zero velocity with a sensible "
                "hip height (about 0.72) is a stationary balanced stand, which is what you want while "
                "the arms work."
            )
            if not self.control_mode.endswith("joint"):
                # Only the IK mode commands poses; joint mode's targets already travel with the body.
                out += (
                    " Wrist poses are absolute and in the env frame, so while the robot walks you "
                    "must re-issue them every step to keep the hands where you want them."
                )
        return out + f" Action dim {self.action_dim}."
