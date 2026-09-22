"""Wx250sRobot — a Trossen/Interbotix WidowX 250 6DOF arm, for table-top manipulation.

The classic Interbotix arm of the ALOHA lineage (stationary ALOHA rigs pair these X-series
arms), predating the WXAI kits (`wxai.py`). A fixed-base 6-DOF arm (waist / shoulder / elbow /
forearm_roll / wrist_angle / wrist_rotate) with a two-finger parallel gripper. The vendored USD
(`assets/wx250s/wx250s/wx250s.usda`) is converted from the OFFICIAL mujoco_menagerie
`trossen_wx250s` MJCF (itself derived from Interbotix's public URDF, BSD-3) with isaaclab's
`MjcfConverter` (fix_base=True); regenerate by re-running the converter over the menagerie file
(then re-apply the one hand edit in `payloads/materials.usda`: the converter drops the MJCF
'black' texture and leaves the arm WHITE — diffuseColor is set to the texture's mean, near-black).

Built for the NEWTON backend (the deformable knot scene's aloha binding): the default USD is the
`assets/wx250s/wx250s_newton.usda` overlay, which authors nonzero drive gains (the converter
writes stiffness=0, and Newton's MuJoCo conversion only creates position actuators for drives
with nonzero gains at parse time), extends the finger travel to aperture 0 (the stock gripper
bottoms out at a 10.8 mm fingertip aperture — real ALOHA rigs bolt on custom fingertips; the
travel extension is the sim stand-in), and adds flat pad-face box colliders (the menagerie
model's only finger colliders are four r=0.6 mm spheres).

BOTH fingers are actuated (the menagerie right-mimics-left equality does not survive
conversion): the gripper action is 2 position targets, and the right finger's coordinate is the
MIRROR of the left's — command ``(g, -g)`` for a symmetric mouth (left in [0.0096, 0.037] m,
right in [-0.037, -0.0096]). Aperture between the pad faces = 2*(g - 0.0096).

Control: "joint" only — 6 direct arm position targets + the 2 finger targets (action dim 8).
Explicit PD gains are REQUIRED (the converted USD carries only the overlay's placeholders);
the defaults below track well under the Newton/MJWarp backend. Heavy imports are deferred so
registration stays app-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from robobench.core.assets import asset_path
from typing import TYPE_CHECKING, Any

import torch

from robobench.controllers import CompositeController, JointController, JointControllerCfg
from robobench.core import ROBOTS, BaseRobot, BaseRobotCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from robobench.core import BaseEnv


@dataclass
class Wx250sRobotCfg(BaseRobotCfg):
    """Config for `Wx250sRobot`. `control_mode` is inherited from `BaseRobotCfg` ("" -> "joint")."""

    fixed_base: bool = True  # weld the base to the world (a table-mounted arm)
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)  # base at the table level
    base_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz on 2.x; xyzw on develop
    # Arm position-PD gains [per rad]. Explicit floats required — the converted USD has no
    # usable drive gains (see the module docstring).
    arm_stiffness: float = 600.0
    arm_damping: float = 30.0
    # Finger PD gains [per m]. Always position-controlled.
    gripper_stiffness: float = 2000.0
    gripper_damping: float = 100.0
    # Home posture of the 6 arm joints (all-zero = the upright candle pose). Retune per task.
    default_dof_pos: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # Left-finger home [m] (0.0096 closed .. 0.037 open); the right finger starts at the mirror.
    default_gripper_pos: float = 0.037
    wx250s_usd: str = ""  # "" -> the vendored assets/wx250s/wx250s_newton.usda overlay

    def __post_init__(self) -> None:
        assets = asset_path(Path(__file__).resolve().parent / "assets") / "wx250s"
        self.wx250s_usd = self.wx250s_usd or str(assets / "wx250s_newton.usda")


@ROBOTS.register("wx250s")
class Wx250sRobot(BaseRobot):
    """Interbotix WidowX 250 6DOF + parallel gripper, joint position control. The gripper is 2
    direct finger targets — command the right as the NEGATED left for a symmetric mouth."""

    control_modes: tuple[str, ...] = ("joint",)

    cfg: Wx250sRobotCfg

    ARM_JOINTS: tuple[str, ...] = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
    GRIPPER_JOINTS: tuple[str, ...] = ("left_finger", "right_finger")
    # The palm link the fingers hang from (+x = finger axis). Prim names are mangled by the
    # MJCF converter ("tn__wx250sgripper_link_VS"), so match by substring regex.
    EE_BODY: str = ".*gripper_link.*"

    JOINT_CONTROL_DT: float = 0.02

    def __init__(self, cfg: Wx250sRobotCfg | None = None) -> None:
        super().__init__(cfg or Wx250sRobotCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """The WX250s articulation over the vendored converted USD."""
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg

        c = self.cfg
        return {
            self.name: ArticulationCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{self.prim_name}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.wx250s_usd,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        max_depenetration_velocity=5.0,
                    ),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False,
                        solver_position_iteration_count=8,
                        solver_velocity_iteration_count=0,
                        # None = keep as authored: the converter (fix_base=True) already welds
                        # the base to the world, and its ArticulationRootAPI sits on a scope
                        # prim isaaclab's re-welding path cannot handle.
                        fix_root_link=None,
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=c.base_pos,
                    rot=c.base_rot,
                    joint_pos={
                        **{j: float(q) for j, q in zip(self.ARM_JOINTS, c.default_dof_pos)},
                        "left_finger": c.default_gripper_pos,
                        "right_finger": -c.default_gripper_pos,
                    },
                ),
                actuators={
                    "wx250s_arm": ImplicitActuatorCfg(
                        joint_names_expr=list(self.ARM_JOINTS),
                        stiffness=c.arm_stiffness,
                        damping=c.arm_damping,
                    ),
                    "wx250s_gripper": ImplicitActuatorCfg(
                        joint_names_expr=list(self.GRIPPER_JOINTS),
                        stiffness=c.gripper_stiffness,
                        damping=c.gripper_damping,
                    ),
                },
                soft_joint_pos_limit_factor=1.0,
            )
        }

    # ----- lifecycle ------------------------------------------------------------------------------
    def on_bind(self, env: BaseEnv) -> None:
        self.articulation: Articulation = env.iscene[self.name]

    def build_controller(self) -> CompositeController:
        """`composite([joint(arm), joint(gripper)])` — 6 arm + 2 finger position targets."""
        if self.control_mode != "joint":
            raise ValueError(f"unknown WX250s control_mode {self.control_mode!r}; known: {self.control_modes}")
        arm = JointController(JointControllerCfg(self.ARM_JOINTS, dt=self.JOINT_CONTROL_DT), command_type="position")
        gripper = JointController(
            JointControllerCfg(self.GRIPPER_JOINTS, dt=self.JOINT_CONTROL_DT), command_type="position"
        )
        return CompositeController([arm, gripper])

    def reset(self, env_ids: torch.Tensor) -> None:
        """Home pose: default joint state; hold all joint position targets at default."""
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
        return (
            f"An Interbotix WidowX 250 6DOF arm (~0.65 m reach) with a two-finger parallel gripper, "
            f"fixed to the table. Control mode 'joint': 6 arm joints by direct position targets, plus "
            f"2 finger position targets (left 0.0096 closed .. 0.037 open; command the right as the "
            f"negated left for a symmetric mouth). Action dim {self.action_dim}."
        )
