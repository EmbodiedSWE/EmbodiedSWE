"""ClassifyObjectsScene — sort scattered coloured blocks into their matching coloured zones.

The object world for the "classify / sort the objects" task: `per_category` blocks of each of
`categories` colours lie scattered, flat, on a work surface, with one marked coloured zone per
category. **Goal (carried here, no task layer): pick each block up and place it in the zone
whose colour matches the block — red blocks in the red zone, blue blocks in the blue zone.**

This is a LONG-HORIZON task: every block is a pick-and-place sub-goal, and the agent must
IDENTIFY each block's category and route it to the correct zone (a block dropped in the wrong
zone does not count), then recover from any misplacement. Difficulty is in the
identification + sequencing + recovery, not any single grasp.

Judged geometrically: a block is "sorted" when its centre is within `zone_half` (xy) of ITS
category's zone centre, resting on the surface (height within `z_tol`), and settled
(|v| < `settle_speed`). `score()` = fraction of blocks correctly sorted (0-100); `success()`
requires every block in its matching zone.

Per-episode randomization: zone positions (small xy jitter) and every block's scatter pose
(arc slot + xy jitter + free yaw), and WHICH blocks are which colour is fixed per episode but
oracle-visible — a memorized fixed routing fails under the pose randomization.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg, info, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ClassifyObjectsSceneCfg(BaseCfg):
    """Config for `ClassifyObjectsScene`. Tolerances are SOFT by design (a floor task)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    zone_half: float = tunable(0.055)  # block centre within this of its zone centre, xy (m)
    z_tol: float = tunable(0.02)  # block height within this of resting-on-surface
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    reset_pos_jitter: float = tunable(0.025)  # uniform +/- xy jitter (blocks) at reset
    reset_yaw_deg: float = tunable(120.0)  # uniform +/- yaw per block at reset
    zone_jitter: float = tunable(0.015)  # uniform +/- xy jitter of the zones at reset

    # --- tunable: placement (table-relative xy) ----------------------------------------------
    surface_z: float | None = tunable(None)
    # one zone centre per category (same order as `categories`); defaults front-left / front-right
    zone_pos: tuple = tunable(((-0.12, -0.13), (0.0, -0.13), (0.12, -0.13)))
    scatter_center: tuple = tunable((0.0, -0.28))  # blocks start scattered behind the zones
    scatter_radii: tuple = tunable((0.10,))
    scatter_arc: tuple = tunable((205.0, 335.0))

    # --- info: work surface (shared packing-suite table pattern) -----------------------------
    table: str = info("packing")
    table_depth_scale: float = info(1.5)
    workbench_pos: tuple[float, float] | None = info(None)
    workbench_usd: str = info("")
    TABLES: ClassVar[dict[str, dict[str, Any]]] = {
        "lab_table": {"usd": ("lab_table", "table_instanceable.usd"), "scale": 1.0,
                      "orient": (0.70711, 0.0, 0.0, 0.70711), "surface_z": 0.0,
                      "pos": (0.40, -0.03), "top_offset": 0.0, "height": 1.05,
                      "kinematic": False},
        "packing": {"usd": ("packing_table", "SM_HeavyDutyPackingTable_C02_01_physics.usd"),
                    "scale": 0.01, "orient": (1.0, 0.0, 0.0, 0.0), "surface_z": 0.994,
                    "pos": (0.0, 0.0), "top_offset": 0.994, "height": 0.994,
                    "kinematic": True},
    }

    # --- info: structure (categories + blocks) -----------------------------------------------
    # (category name, RGB colour); one marked zone per category, blocks coloured to match.
    categories: tuple = info((
        ("red", (0.80, 0.20, 0.20)),
        ("blue", (0.20, 0.55, 0.85)),
        ("green", (0.30, 0.70, 0.30)),
    ))
    per_category: int = info(1)  # blocks of each category (total = len(categories)*per_category)
    block_size: float = info(0.045)  # cube edge (m) — graspable by the G1 three-finger hand
    block_mass: float = info(0.025)
    contact_offset: float = info(0.004)
    zone_thickness: float = info(0.004)  # thin visual mat marking a zone (no collision)
    zone_mat_half: float = info(0.038)  # visual mat half-size (< zone_half so the 3 mats sit clearly apart)

    asset_dir: str = info("")

    # Derived (filled in __post_init__): ((block_name, category_index), ...)
    manifest: tuple = field(default=None, init=False)

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.asset_dir = self.asset_dir or str(assets / "classify_objects")
        preset = self.TABLES[self.table]
        if self.surface_z is None:
            self.surface_z = preset["surface_z"]
        if self.workbench_pos is None:
            self.workbench_pos = preset["pos"]
        self.workbench_usd = self.workbench_usd or str(
            assets.parents[1] / "assembly" / "assets" / "props" / preset["usd"][0] / preset["usd"][1])
        flat = []
        for ci, (cname, _rgb) in enumerate(self.categories):
            for j in range(self.per_category):
                flat.append((f"{cname}_{j}", ci))
        self.manifest = tuple(flat)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("classify_objects")
class ClassifyObjectsScene(BaseScene):
    cfg: ClassifyObjectsSceneCfg

    def __init__(self, cfg: ClassifyObjectsSceneCfg | None = None) -> None:
        super().__init__(cfg or ClassifyObjectsSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z
        s = c.block_size
        preset = c.TABLES[c.table]
        wx, wy = c.workbench_pos
        table_z = z0 - preset["top_offset"]
        ground_z = z0 - preset["height"]
        ts = preset["scale"]
        table_spawn = sim_utils.UsdFileCfg(usd_path=c.workbench_usd,
                                           scale=(ts, ts * c.table_depth_scale, ts))
        if preset["kinematic"]:
            table_spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(usd_path=str(
                    Path(c.asset_dir).parents[2] / "assembly" / "assets" / "props"
                    / "ground" / "default_ground.usd")),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, ground_z)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "workbench": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(wx, wy, table_z), rot=preset["orient"]),
                spawn=table_spawn,
            ),
        }
        if c.table == "lab_table":
            out["table_backing"] = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/TableBacking",
                spawn=sim_utils.CuboidCfg(size=(1.66, 1.42, 0.05),
                                          collision_props=sim_utils.CollisionPropertiesCfg(),
                                          visible=False),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(wx - 0.37, wy + 0.395, z0 - 0.0035 - 0.025)))

        # one visual zone mat per category, tinted its colour (brightened so it reads as a target)
        for ci, (cname, rgb) in enumerate(c.categories):
            zx, zy = c.zone_pos[ci]
            tint = tuple(v for v in rgb)  # raw category colour (deeper, easier to read)
            out[f"zone_{cname}"] = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Zone_" + cname,
                spawn=sim_utils.CuboidCfg(
                    size=(2 * c.zone_mat_half, 2 * c.zone_mat_half, c.zone_thickness),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tint, roughness=0.9)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(zx, zy, z0 + c.zone_thickness / 2)),
            )

        # the coloured blocks, scattered flat at nominal arc slots (reset re-places)
        a0, a1 = (math.radians(v) for v in c.scatter_arc)
        n = len(c.manifest)
        cx, cy = c.scatter_center
        for i, (name, ci) in enumerate(c.manifest):
            ang = a0 + (a1 - a0) * (i + 0.5) / n
            r = c.scatter_radii[i % len(c.scatter_radii)]
            rgb = c.categories[ci][1]
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Block_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(s, s, s),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=32, solver_velocity_iteration_count=1,
                        max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.05),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.block_mass),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.9, dynamic_friction=0.9, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=0.7)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(cx + r * math.cos(ang), cy + r * math.sin(ang), z0 + s / 2 + 0.002)),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(dt=1.0 / 120.0, physx={
            "solver_type": 1, "bounce_threshold_velocity": 0.2,
            "friction_offset_threshold": 0.01, "friction_correlation_distance": 0.00625,
            "gpu_max_rigid_contact_count": 2**23, "gpu_max_rigid_patch_count": 2**23,
            "gpu_collision_stack_size": 2**28, "gpu_max_num_partitions": 1})

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.blocks: dict[str, RigidObject] = {name: env.iscene[name] for name, _ci in c.manifest}
        self.env_origins = env.iscene.env_origins
        # per-block category index + current (jittered) zone centres, filled at reset
        self._cat = torch.tensor([ci for _n, ci in c.manifest], device=env.device)  # (B,)
        base = torch.tensor(c.zone_pos, device=env.device)  # (C, 2)
        self._zones = base[None].repeat(env.num_envs, 1, 1)  # (N, C, 2)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        s = c.block_size
        yaw_amp = math.radians(c.reset_yaw_deg)

        # jitter the zone centres for this episode
        base = torch.tensor(c.zone_pos, device=dev)  # (C,2)
        self._zones[env_ids] = base[None] + (torch.rand(m, len(c.categories), 2, device=dev) * 2 - 1) \
            * c.zone_jitter

        a0, a1 = (math.radians(v) for v in c.scatter_arc)
        n = len(c.manifest)
        cx, cy = c.scatter_center
        min_gap = s * 1.5
        placed: list[torch.Tensor] = []
        for i, (name, _ci) in enumerate(c.manifest):
            ang = a0 + (a1 - a0) * (i + 0.5) / n
            r = c.scatter_radii[i % len(c.scatter_radii)]
            slot = torch.tensor([cx + r * math.cos(ang), cy + r * math.sin(ang)], device=dev)
            ctr = slot + (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            for _try in range(12):
                bad = torch.zeros(m, dtype=torch.bool, device=dev)
                for qc in placed:
                    bad |= (ctr - qc).norm(dim=-1) < min_gap
                if not bad.any():
                    break
                ctr[bad] = slot + (torch.rand(int(bad.sum()), 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            placed.append(ctr)
            yaw = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = ctr
            st[:, 2] = c.surface_z + s / 2 + 0.002
            half = yaw / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            self.blocks[name].write_root_state_to_sim(st, env_ids)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {"blocks": {n: b.data.root_state_w[env_ids].clone() for n, b in self.blocks.items()},
                "zones": self._zones[env_ids].clone()}

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for n, b in self.blocks.items():
            b.write_root_state_to_sim(state["blocks"][n], env_ids)
        self._zones[env_ids] = state["zones"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        cats = ", ".join(f"{c.per_category} {name}" for name, _rgb in c.categories)
        names = " and ".join(name for name, _rgb in c.categories)
        return (
            f"{len(c.manifest)} coloured wooden blocks ({c.block_size * 1000:.0f} mm cubes) lie "
            f"scattered on a work table: {cats}. On the table are {len(c.categories)} marked "
            f"coloured zones, one per colour ({names}). "
            f"Goal: sort every block into the zone whose colour matches it — put each block "
            f"fully onto its matching-colour zone. A block left off the zones, or placed on the "
            f"wrong colour's zone, does not count."
        )

    # ----- progress / rubric ----------------------------------------------------------------------
    def _block_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos_w (N,B,3), |lin_vel| (N,B)) for all blocks, manifest order."""
        pos = torch.stack([b.data.root_pos_w for b in self.blocks.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.blocks.values()], dim=1)
        return pos, vel

    def sorted_mask(self) -> torch.Tensor:
        """(N, B) bool: block resting on the surface, settled, and within `zone_half` of ITS
        own category's (jittered) zone centre."""
        c = self.cfg
        pos, vel = self._block_tensors()
        rel = pos - self.env_origins[:, None, :]  # env-local
        # each block's own zone centre: gather by category index -> (N, B, 2)
        own_zone = self._zones[:, self._cat, :]
        near = (rel[:, :, :2] - own_zone).norm(dim=-1) < c.zone_half
        on_surface = (rel[:, :, 2] - (c.surface_z + c.block_size / 2)).abs() < c.z_tol
        settled = vel < c.settle_speed
        return near & on_surface & settled

    def n_sorted(self) -> torch.Tensor:
        """(N,) int: number of correctly-sorted blocks."""
        return self.sorted_mask().sum(dim=1)

    def score(self) -> torch.Tensor:
        """(N,) int in [0,100]: fraction of blocks correctly sorted."""
        return (100 * self.n_sorted() / len(self.cfg.manifest)).round().long()

    def success(self) -> torch.Tensor:
        """(N,) bool: every block sorted into its matching-colour zone."""
        return self.sorted_mask().all(dim=1)
