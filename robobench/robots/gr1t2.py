"""GR1T2Robot — Fourier GR1-T2 humanoid, fixed base, upper-body manipulation.

The second concrete embodiment, built on the same pattern as `G1Robot` (read that first). Wraps
Isaac's `GR1T2_HIGH_PD_CFG` (the manipulation-tuned GR1T2 with 6-DOF Fourier hands) and pins the
pelvis (`fix_root_link`) for a stationary upper-body manipulator. USD is **vendored locally**
(`robots/assets/gr1t2/`, ~6 MB; the UDIM appearance textures are skipped — geometry/physics intact).

Control modes (same shape as G1: `composite([<arm controller>, joint(hands)])`):
  - `joint`   — arm+waist by `JointController` (direct position targets).
  - `pink_ik` — arm+waist by whole-body Pink IK (action = two wrist poses = 14, + hands).

Unlike G1 (which has a ready-made kinematics URDF on Nucleus), GR1T2 has none — its URDF is generated
once from the vendored USD via `convert_usd_to_urdf` and vendored alongside it
(`GR1T2_fourier_hand_6dof_kinematics.urdf`); the link names below are read from that URDF.

Heavy imports (isaaclab / isaaclab_assets) are deferred so importing this module stays app-free.
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
from robobench.core import ROBOTS, BaseRobot, BaseRobotCfg, info, tunable

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from robobench.core import BaseEnv


@dataclass
class GR1T2RobotCfg(BaseRobotCfg):
    """Config for `GR1T2Robot`. `base_pos`/`base_rot` are `tunable` placement dials; `fixed_base` +
    asset path are `info`; the upper-body PD gains are `tunable` (default = Isaac's HIGH_PD values)."""

    fixed_base: bool = info(True)  # weld the pelvis to the world (build-time variant; see G1Robot)
    base_pos: tuple[float, float, float] = tunable((0.0, 0.0, 0.95))  # GR1T2's natural standing height
    base_rot: tuple[float, float, float, float] = tunable((1.0, 0.0, 0.0, 0.0))  # wxyz; identity (faces +x)
    # Arm + waist (trunk) PD gains — the articulation PD that tracks position targets. Defaults are
    # Isaac's `GR1T2_HIGH_PD_CFG` manipulation values; tunable for compliance studies. Hands keep the
    # USD/cfg defaults.
    arm_stiffness: float = tunable(4400.0)
    arm_damping: float = tunable(40.0)
    waist_stiffness: float = tunable(4400.0)
    waist_damping: float = tunable(40.0)
    gr1t2_usd: str = info("")  # "" -> the vendored robots/assets/gr1t2/GR1T2_fourier_hand_6dof.usd
    gr1t2_urdf: str = info("")  # "" -> the vendored kinematics URDF (used by the pink_ik control mode)

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parent / "assets" / "gr1t2"
        self.gr1t2_usd = self.gr1t2_usd or str(assets / "GR1T2_fourier_hand_6dof.usd")
        self.gr1t2_urdf = self.gr1t2_urdf or str(assets / "GR1T2_fourier_hand_6dof_kinematics.urdf")


@ROBOTS.register("gr1t2")
class GR1T2Robot(BaseRobot):
    """Fixed-base GR1-T2 upper-body manipulator. `apply_action`/`action_dim` are inherited from
    `BaseRobot` and delegate to the controller for the active `control_mode`."""

    control_modes: tuple[str, ...] = ("joint", "pink_ik")
    cfg: GR1T2RobotCfg

    # Upper-body control period (s): ~50 Hz (Isaac `Isaac-PickPlace-FixedBaseUpperBodyIK-G1`, decim 4 @
    # 200 Hz). `bind` rounds to the nearest sim-step multiple -> period 2 (60 Hz) at the 120 Hz table.
    CONTROL_DT: float = 0.02

    # Upper-body joint groups (regexes; find_joints resolves indices). arms = shoulder pitch/roll/yaw,
    # elbow pitch, wrist yaw/roll/pitch (per side); + waist; hands = the Fourier 6-DOF hand joints.
    ARM_WAIST_JOINTS: tuple[str, ...] = (".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*", "waist_.*")
    HAND_JOINTS: tuple[str, ...] = ("R_.*", "L_.*")

    def __init__(self, cfg: GR1T2RobotCfg | None = None) -> None:
        super().__init__(cfg or GR1T2RobotCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """`GR1T2_HIGH_PD_CFG` re-pointed at the vendored USD, base fixed per `cfg.fixed_base`, spawned
        at the configured base pose, with the upper-body PD gains from cfg. `.copy()` deep-copies, so
        mutating the actuators doesn't touch the shared module cfg."""
        from isaaclab_assets.robots.fourier import GR1T2_HIGH_PD_CFG

        c = self.cfg
        robot = GR1T2_HIGH_PD_CFG.copy()  # type: ignore[attr-defined]
        robot.prim_path = "{ENV_REGEX_NS}/Robot"
        robot.spawn.usd_path = c.gr1t2_usd
        robot.spawn.articulation_props.fix_root_link = c.fixed_base
        robot.init_state.pos = c.base_pos
        robot.init_state.rot = c.base_rot
        for grp in ("right-arm", "left-arm"):
            robot.actuators[grp].stiffness = c.arm_stiffness
            robot.actuators[grp].damping = c.arm_damping
        robot.actuators["trunk"].stiffness = c.waist_stiffness
        robot.actuators["trunk"].damping = c.waist_damping
        return {"robot": robot}

    # ----- lifecycle (hooks; the base orchestrates bind -> on_bind -> build_controller) ---------
    def on_bind(self, env: BaseEnv) -> None:
        self.articulation: Articulation = env.iscene["robot"]

    def build_controller(self) -> CompositeController:
        """`composite([<arm controller>, joint(hands)])` for the active mode. Hands are always direct
        position targets; switching the mode swaps only the arm controller."""
        hands = JointController(JointControllerCfg(self.HAND_JOINTS, dt=self.CONTROL_DT), command_type="position")
        if self.control_mode == "joint":
            arms: Any = JointController(
                JointControllerCfg(self.ARM_WAIST_JOINTS, dt=self.CONTROL_DT), command_type="position"
            )
        elif self.control_mode == "pink_ik":
            # Pink IK targets the two hands; base = base_link. URDF link names = USD names prefixed with
            # the robot name; null-space holds the (redundant) shoulders + waist near home.
            pfx = "GR1T2_fourier_hand_6dof_"
            arms = PinkIKController(
                PinkIKControllerCfg(
                    dt=self.CONTROL_DT,
                    urdf_path=self.cfg.gr1t2_urdf,
                    base_link="base_link",
                    base_link_frame=pfx + "base_link",
                    frames=(FrameTaskCfg(pfx + "left_hand_pitch_link"), FrameTaskCfg(pfx + "right_hand_pitch_link")),
                    joint_names=self.ARM_WAIST_JOINTS,  # the 17-DOF chain the IK solves over
                    nullspace_joints=(
                        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                        "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
                    ),
                )
            )
        else:
            raise ValueError(f"unknown GR1T2 control_mode {self.control_mode!r}; known: {self.control_modes}")
        return CompositeController([arms, hands])

    def reset(self, env_ids: torch.Tensor) -> None:
        """Home pose: default joint state, all targets held at default (un-driven joints stay put),
        root written to the (fixed) spawn pose + env origin."""
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

    # NOTE: apply_action / action_dim are inherited from BaseRobot (delegate to self.controller).

    # get_state / set_state are inherited from BaseRobot (single-articulation sim state + controller).

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        subs = getattr(self.controller, "controllers", None)
        n_arm = len(subs[0].joint_ids) if subs else 17
        n_hand = len(subs[1].joint_ids) if subs else 22
        base = "fixed at the pelvis (legs locked), a stationary upper-body manipulator" if self.cfg.fixed_base else "on a mobile base"
        return (
            f"A Fourier GR1-T2 humanoid with 6-DOF dexterous hands, {base}. Control mode "
            f"'{self.control_mode}': {n_arm} arm+waist joints by direct position targets, plus {n_hand} "
            f"hand joints by direct position target. Action dim {self.action_dim}."
        )
