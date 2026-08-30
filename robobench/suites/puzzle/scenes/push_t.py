"""PushTScene — push a red T block onto its matching gray T pad.

This is a scene-level port of RoboDojo's ``push_T`` task using the original authored
assets. The task is intentionally a simple humanoid manipulation primitive: the block and
target begin with the same yaw, so solving requires one planar push rather than a pick or a
compound reorientation. A small shared reset translation/yaw keeps the rollout live-state
driven without changing that difficulty.

The source success thresholds are preserved: block/target XY error at most 7 mm, orientation
error at most 7 degrees, and no lift above the table. CoSiGen additionally requires the block
to be settled before success. Heavy Isaac Lab imports remain deferred for app-free discovery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg, info, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


@dataclass
class PushTSceneCfg(BaseCfg):
    """Rubric, layout, and measured source-asset constants for :class:`PushTScene`."""

    # Source rubric, verbatim.
    xy_tolerance: float = tunable(0.007)
    orientation_tolerance_deg: float = tunable(7.0)
    lift_tolerance: float = tunable(0.010)

    # Settling and graded progress.
    settle_linear_speed: float = tunable(0.03)
    settle_angular_speed: float = tunable(0.20)
    near_xy_tolerance: float = tunable(0.025)
    close_xy_tolerance: float = tunable(0.015)

    # Simple G1-native 3 cm push. Both poses are rotated/translated together on reset.
    # The short stroke keeps the full interaction inside G1's stable table-contact
    # workspace; the pair sits toward the front edge so the G1 can remain clear of
    # the table chassis. Task difficulty comes from strict alignment, not arm reach.
    block_pos: tuple[float, float] = tunable((0.0, -0.20))
    target_pos: tuple[float, float] = tunable((0.0, -0.17))
    reset_pos_jitter: float = tunable(0.006)
    base_yaw_deg: float = tunable(90.0)  # broad T bar faces the robot for a stable +y push
    reset_yaw_jitter_deg: float = tunable(5.0)

    # Measured from the official metadata / USD extents.
    block_half_height: float = info(0.0075)
    block_mass: float = info(0.35)
    block_reset_clearance: float = info(0.0015)
    target_lift: float = info(0.0008)
    surface_z: float = info(0.7)
    table_depth_scale: float = info(1.5)
    workbench_pos: tuple[float, float] = info((0.0, 0.0))
    asset_dir: str = info("")
    block_usd: str = info("")
    target_usd: str = info("")
    workbench_usd: str = info("")

    def __post_init__(self) -> None:
        suite_assets = Path(__file__).resolve().parents[1] / "assets"
        task_assets = suite_assets / "push_t"
        self.asset_dir = self.asset_dir or str(task_assets)
        self.block_usd = self.block_usd or str(task_assets / "block" / "main.usda")
        self.target_usd = self.target_usd or str(task_assets / "target_pad" / "source.usdc")
        self.workbench_usd = self.workbench_usd or str(
            suite_assets.parents[1]
            / "assembly"
            / "assets"
            / "props"
            / "packing_table"
            / "SM_HeavyDutyPackingTable_C02_01_physics.usd"
        )


@SCENES.register("push_t")
class PushTScene(BaseScene):
    cfg: PushTSceneCfg

    def __init__(self, cfg: PushTSceneCfg | None = None) -> None:
        super().__init__(cfg or PushTSceneCfg())

    def assets(self) -> dict[str, Any]:
        """Packing table, kinematic target pad, and one dynamic T block."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        for usd in (c.block_usd, c.target_usd, c.workbench_usd):
            if not Path(usd).is_file():
                raise FileNotFoundError(
                    f"{usd} not found — run scripts/vendor_push_t_assets.py for task assets"
                )

        table_scale = 0.01
        table_top_offset = 0.994
        table_height = 0.994
        table_z = c.surface_z - table_top_offset
        ground_z = c.surface_z - table_height
        wx, wy = c.workbench_pos

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    usd_path=str(
                        Path(__file__).resolve().parents[2]
                        / "assembly"
                        / "assets"
                        / "props"
                        / "ground"
                        / "default_ground.usd"
                    )
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, ground_z)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "workbench": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.workbench_usd,
                    scale=(table_scale, table_scale * c.table_depth_scale, table_scale),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(wx, wy, table_z)),
            ),
            "target": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PushTTarget",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.target_usd,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.target_pos[0], c.target_pos[1], c.surface_z + c.target_lift)
                ),
            ),
            "block": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PushTBlock",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.block_usd,
                    activate_contact_sensors=True,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.block_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True, contact_offset=0.002, rest_offset=0.0
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=2,
                        max_depenetration_velocity=0.25,
                        linear_damping=0.15,
                        # A low-profile wood block on a workbench should not spin freely from
                        # a fingertip impulse.  Bound and damp yaw so this introductory task
                        # rewards straight pushing instead of requiring a recovery maneuver.
                        angular_damping=1.5,
                        max_angular_velocity=0.035,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(
                        c.block_pos[0],
                        c.block_pos[1],
                        c.surface_z + c.block_half_height + c.block_reset_clearance,
                    )
                ),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**21,
                "gpu_collision_stack_size": 2**27,
            },
        )

    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.block: RigidObject = env.iscene["block"]
        self.target: RigidObject = env.iscene["target"]
        self.env_origins = env.iscene.env_origins

    def reset(self, env_ids: torch.Tensor) -> None:
        """Place a same-yaw block/target pair with small shared layout randomization."""
        c = self.cfg
        device = self.env.device
        count = len(env_ids)
        origins = self.env_origins[env_ids]
        yaw = math.radians(c.base_yaw_deg) + (
            torch.rand(count, device=device) * 2 - 1
        ) * math.radians(c.reset_yaw_jitter_deg)
        layout_yaw = yaw - math.radians(c.base_yaw_deg)
        translation = (torch.rand(count, 2, device=device) * 2 - 1) * c.reset_pos_jitter

        def planar_pose(xy: tuple[float, float], z: float) -> torch.Tensor:
            local = torch.tensor(xy, device=device).expand(count, 2)
            # The nominal layout advances along world +y while the asset's local +x
            # (its long stem) is authored at +90 degrees. Rotate both only by the shared
            # jitter so the wide rear bar stays perpendicular to the push direction.
            cos_yaw, sin_yaw = torch.cos(layout_yaw), torch.sin(layout_yaw)
            rotated = torch.stack(
                [
                    cos_yaw * local[:, 0] - sin_yaw * local[:, 1],
                    sin_yaw * local[:, 0] + cos_yaw * local[:, 1],
                ],
                dim=1,
            )
            state = torch.zeros(count, 13, device=device)
            state[:, 0:2] = rotated + translation
            state[:, 2] = z
            state[:, 3] = torch.cos(yaw / 2)
            state[:, 6] = torch.sin(yaw / 2)
            state[:, 0:3] += origins
            return state

        self.target.write_root_state_to_sim(
            planar_pose(c.target_pos, c.surface_z + c.target_lift), env_ids
        )
        self.block.write_root_state_to_sim(
            planar_pose(
                c.block_pos,
                c.surface_z + c.block_half_height + c.block_reset_clearance,
            ),
            env_ids,
        )

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "block": self.block.data.root_state_w[env_ids].clone(),
            "target": self.target.data.root_state_w[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.block.write_root_state_to_sim(state["block"], env_ids)
        self.target.write_root_state_to_sim(state["target"], env_ids)

    def describe(self) -> str:
        return (
            "A small red T-shaped block rests flat near the front of a work table. Just ahead "
            "is a slightly larger, very thin dark-gray T-shaped target pad with the same "
            "orientation.\nGoal: push the red T-shaped block along the table until its shape "
            "aligns precisely with the gray T-shaped pad. Keep the block flat on the table; "
            "lifting it does not solve the task."
        )

    def position_error(self) -> torch.Tensor:
        """Planar centre distance between the block and target, in metres."""
        return (self.block.data.root_pos_w[:, :2] - self.target.data.root_pos_w[:, :2]).norm(
            dim=1
        )

    def orientation_error(self) -> torch.Tensor:
        """Shortest full 3-D quaternion error between block and target, in radians."""
        dot = (self.block.data.root_quat_w * self.target.data.root_quat_w).sum(dim=1)
        return 2.0 * torch.acos(dot.abs().clamp(max=1.0))

    def not_lifted(self) -> torch.Tensor:
        """The block bottom remains within the source task's 10 mm lift allowance."""
        local_z = (self.block.data.root_pos_w - self.env_origins)[:, 2]
        bottom_z = local_z - self.cfg.block_half_height
        return (bottom_z <= self.cfg.surface_z + self.cfg.lift_tolerance) & (
            bottom_z >= self.cfg.surface_z - 0.005
        )

    def settled(self) -> torch.Tensor:
        return (
            self.block.data.root_lin_vel_w.norm(dim=1) < self.cfg.settle_linear_speed
        ) & (self.block.data.root_ang_vel_w.norm(dim=1) < self.cfg.settle_angular_speed)

    def score(self) -> torch.Tensor:
        """Graded 0–100 progress while retaining the source's strict success gate."""
        c = self.cfg
        xy = self.position_error()
        angle = self.orientation_error()
        result = torch.zeros_like(xy, dtype=torch.long)
        result = torch.where(xy <= c.near_xy_tolerance, 40, result)
        result = torch.where(xy <= c.close_xy_tolerance, 70, result)
        result = torch.where(xy <= c.xy_tolerance, 80, result)
        pose_ok = (xy <= c.xy_tolerance) & (
            angle <= math.radians(c.orientation_tolerance_deg)
        )
        result = torch.where(pose_ok & self.not_lifted(), 90, result)
        return torch.where(self.success(), 100, result)

    def success(self) -> torch.Tensor:
        c = self.cfg
        return (
            (self.position_error() <= c.xy_tolerance)
            & (self.orientation_error() <= math.radians(c.orientation_tolerance_deg))
            & self.not_lifted()
            & self.settled()
        )
