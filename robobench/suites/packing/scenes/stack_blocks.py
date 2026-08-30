"""StackBlocksScene — stack scattered blocks into a single aligned tower on a marked pad.

The object world for the "stack the blocks" task: `n_blocks` identical cubes lie scattered,
flat, on a work surface, next to a marked square pad. **Goal (carried here, no task layer):
pick the blocks up one by one and stack them into a single vertical tower, centred on the
pad, largest face down, each block squarely on top of the one below.**

This is a deliberately LONG-HORIZON task: with `n_blocks` cubes there are `n_blocks`
sequential pick-and-place sub-goals, every placement must land within a lateral tolerance of
the growing tower (a sloppy placement topples it and the run must recover), and the tower is
only complete when every block is up and everything has settled. The value is in the
sequencing and the recovery, not in any single grasp.

Judged geometrically against a fixed vertical column of slots above the pad centre, so the
test is identical wherever the pad sits and does not care which physical block ends up at
which height (the cubes are identical): a block "fills slot L" when its centre is within
`align_tol` of the pad axis (xy), its height matches the slot (surface + block/2 + L*block
within `z_tol`), it is UPRIGHT (a face level with the ground, within `upright_max_deg`), and
both it and the tower have settled (|v| < `settle_speed`). `score()` climbs with the height
of the contiguous tower built from the pad up; `success()` requires all `n_blocks` slots
filled by distinct blocks — one clean tower, no leftovers.

Per-episode randomization (task-family knobs): pad pose (xy jitter), and every block's
scatter pose (arc slot + xy jitter + free yaw about the vertical, lying flat on a face), so a
memorized fixed placement sequence fails.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
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
class StackBlocksSceneCfg(BaseCfg):
    """Config for `StackBlocksScene`. Rubric tolerances are SOFT by design (a floor task that
    ranks weak agents); harden `align_tol` / `z_tol` for curriculum variants."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    align_tol: float = tunable(0.030)  # block centre within this of the pad axis, xy (m).
    # For the default 50 mm cube this permits ~40% overhang before a slot stops counting — a
    # tower leaning further than that is genuinely unstable; tighten for a stricter variant.
    z_tol: float = tunable(0.020)  # block height within this of its slot's nominal z (m)
    upright_max_deg: float = tunable(25.0)  # a block face level with the ground within this
    settle_speed: float = tunable(0.05)  # max |lin vel| (block AND tower) when judging (m/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    reset_pos_jitter: float = tunable(0.025)  # uniform +/- xy jitter (pad AND blocks) at reset
    reset_yaw_deg: float = tunable(120.0)  # uniform +/- yaw per block at reset (about vertical)

    # --- tunable: placement (table-relative xy; the table itself sits at TABLES pos) ---------
    surface_z: float | None = tunable(None)  # work-surface height (m); None -> the preset's
    pad_pos: tuple = tunable((0.20, 0.0))  # tower pad centre on the surface
    scatter_center: tuple = tunable((-0.10, 0.0))  # scatter-arc centre (blocks start here)
    scatter_radii: tuple = tunable((0.16,))  # scatter arc radii
    scatter_arc: tuple = tunable((90.0, 270.0))  # scatter arc (deg) around scatter_center

    # --- info: work surface (the pen_holder / ikea / motherboard vendored-table pattern) -----
    table: str = info("packing")  # which work surface: "lab_table" | "packing"
    table_depth_scale: float = info(1.5)  # y-stretch (matches the sibling packing scenes)
    workbench_pos: tuple[float, float] | None = info(None)  # xy the table sits at; None -> preset
    workbench_usd: str = info("")  # empty -> the preset's vendored USD
    # Same presets as the sibling scenes (see pen_holder.py for the measured-footprint notes).
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

    # --- info: structure (the blocks + the pad) ----------------------------------------------
    n_blocks: int = info(4)  # tower height / number of cubes
    block_size: float = info(0.045)  # cube edge length (m) — a standard wooden toy block
    block_mass: float = info(0.025)  # per cube (m) — a light hardwood cube
    contact_offset: float = info(0.004)  # speculative contact margin (m); catches fast stacks
    # Distinct colours so the tower reads instantly on camera and a future ordered variant can
    # name them; identity is oracle-visible, there is no hidden state.
    block_colors: tuple = info((
        (0.80, 0.20, 0.20),  # red
        (0.20, 0.55, 0.85),  # blue
        (0.30, 0.70, 0.30),  # green
        (0.90, 0.75, 0.20),  # yellow
    ))
    pad_color: tuple = info((0.15, 0.15, 0.17))  # the dark target pad
    pad_thickness: float = info(0.004)  # a thin visual mat on the surface (no collision)

    # Asset dir (reserved for vendored block USDs; empty -> authored cubes at spawn).
    asset_dir: str = info("")

    # Derived (filled in __post_init__).
    names: tuple = field(default=None, init=False)  # ("block_0", ...)

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.asset_dir = self.asset_dir or str(assets / "stack_blocks")
        preset = self.TABLES[self.table]
        if self.surface_z is None:
            self.surface_z = preset["surface_z"]
        if self.workbench_pos is None:
            self.workbench_pos = preset["pos"]
        self.workbench_usd = self.workbench_usd or str(
            assets.parents[1] / "assembly" / "assets" / "props" / preset["usd"][0] / preset["usd"][1])
        self.names = tuple(f"block_{i}" for i in range(self.n_blocks))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("stack_blocks")
class StackBlocksScene(BaseScene):
    cfg: StackBlocksSceneCfg

    def __init__(self, cfg: StackBlocksSceneCfg | None = None) -> None:
        super().__init__(cfg or StackBlocksSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Floor, light, the vendored work table, a thin visual pad marking the tower base,
        and `n_blocks` coloured cubes scattered flat on the surface (reset() re-places them).
        Blocks carry a depenetration cap + light damping so a cube nudged in the stack crosses
        the settle gate promptly instead of ringing."""
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
            # Back the lab table's partial collision top (see pen_holder.py TABLES note).
            out["table_backing"] = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/TableBacking",
                spawn=sim_utils.CuboidCfg(
                    size=(1.66, 1.42, 0.05),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visible=False,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(wx - 0.37, wy + 0.395, z0 - 0.0035 - 0.025)),
            )

        # Visual-only target pad (no collision — a flat mat flush with the surface).
        out["pad"] = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Pad",
            spawn=sim_utils.CuboidCfg(
                size=(3 * s, 3 * s, c.pad_thickness),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=c.pad_color, roughness=0.9),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(c.pad_pos[0], c.pad_pos[1], z0 + c.pad_thickness / 2)),
        )

        # The coloured cubes, scattered flat at nominal arc slots (reset re-places).
        a0, a1 = (math.radians(v) for v in c.scatter_arc)
        n = c.n_blocks
        cx, cy = c.scatter_center
        for i, name in enumerate(c.names):
            ang = a0 + (a1 - a0) * (i + 0.5) / n
            r = c.scatter_radii[i % len(c.scatter_radii)]
            col = c.block_colors[i % len(c.block_colors)]
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Block_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(s, s, s),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05,
                        angular_damping=0.05,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.block_mass),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.9, dynamic_friction=0.9, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=col, roughness=0.7),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(cx + r * math.cos(ang), cy + r * math.sin(ang), z0 + s / 2 + 0.002)),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab block handles + env origins. Pure passive physics — no joints, no post_step."""
        super().bind(env)
        c = self.cfg
        self.blocks: dict[str, RigidObject] = {name: env.iscene[name] for name in c.names}
        self.env_origins = env.iscene.env_origins

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the pad (fixed, only the blocks move relative to it — the pad is
        a static visual so its 'reset' is just the jittered nominal), then scatter every block
        flat on the surface at its arc slot + xy jitter + free yaw, rejecting overlaps so the
        depenetration pop cannot hurl a cube off the table."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        s = c.block_size
        yaw_amp = math.radians(c.reset_yaw_deg)

        a0, a1 = (math.radians(v) for v in c.scatter_arc)
        n = c.n_blocks
        cx, cy = c.scatter_center
        min_gap = s * 1.5  # centre-to-centre keep-out for two flat cubes (+ margin)

        placed: list[torch.Tensor] = []  # (m,2) centres already placed this reset
        for i, name in enumerate(c.names):
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
            half = yaw / 2  # yaw about world +z: q = (cos, 0, 0, sin)
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            self.blocks[name].write_root_state_to_sim(st, env_ids)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {"blocks": {n: b.data.root_state_w[env_ids].clone()
                           for n, b in self.blocks.items()}}

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for n, b in self.blocks.items():
            b.write_root_state_to_sim(state["blocks"][n], env_ids)

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"{c.n_blocks} coloured wooden blocks ({c.block_size * 1000:.0f} mm cubes) lie "
            f"scattered flat on a work table, beside a marked square pad. "
            f"Goal: stack all {c.n_blocks} blocks into a single upright tower centred on the "
            f"pad — one block squarely on top of the next, faces level, so the finished tower "
            f"stands {c.n_blocks} blocks high and stays standing. A block placed off-centre "
            f"enough to overhang, set down crooked, or left off the pad does not count toward "
            f"the tower."
        )

    # ----- progress / rubric ---------------------------------------------------------------------
    def _block_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos_w (N,B,3), quat_wxyz (N,B,4), |lin_vel| (N,B)) for all blocks, cfg order."""
        pos = torch.stack([b.data.root_pos_w for b in self.blocks.values()], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.blocks.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.blocks.values()], dim=1)
        return pos, quat, vel

    def _upright(self, quat: torch.Tensor) -> torch.Tensor:
        """(N,B) bool: a cube face is level with the ground — i.e. one body axis is within
        `upright_max_deg` of world-up. Rejects a cube resting on an edge or corner."""
        from isaaclab.utils.math import matrix_from_quat

        n, bch = quat.shape[0], quat.shape[1]
        rot = matrix_from_quat(quat.reshape(n * bch, 4)).reshape(n, bch, 3, 3)
        # world-up component of each body axis = the last ROW of the rotation matrix.
        up_comp = rot[:, :, 2, :].abs()  # (N,B,3): |axis_k . world_up|
        best = up_comp.max(dim=-1).values  # the most-vertical body axis
        return best >= math.cos(math.radians(self.cfg.upright_max_deg))

    def _slots(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per block: (filled (N,B) bool, level (N,B) long). A block fills a tower slot when it
        is over the pad axis (xy within `align_tol`), UPRIGHT, SETTLED, the tower is settled,
        and its height matches an integer slot L in [0, n_blocks) within `z_tol`."""
        c = self.cfg
        pos, quat, vel = self._block_tensors()
        # pad + heights are env-local; subtract the env origin.
        rel = pos - self.env_origins[:, None, :]
        pad = torch.tensor([c.pad_pos[0], c.pad_pos[1]], device=pos.device)
        near_axis = (rel[:, :, :2] - pad).norm(dim=-1) < c.align_tol

        z0 = c.surface_z + c.block_size / 2  # slot-0 nominal centre height
        lvl_f = (rel[:, :, 2] - z0) / c.block_size
        level = lvl_f.round().long()
        on_slot = (lvl_f - level).abs() * c.block_size < c.z_tol
        in_range = (level >= 0) & (level < c.n_blocks)

        upright = self._upright(quat)
        settled = vel < c.settle_speed
        tower_still = (vel < c.settle_speed).all(dim=1, keepdim=True)  # whole scene quiet
        filled = near_axis & on_slot & in_range & upright & settled & tower_still
        return filled, level.clamp(0, c.n_blocks - 1)

    def _level_counts(self) -> torch.Tensor:
        """(N, n_blocks) int: how many blocks currently fill each tower slot L."""
        c = self.cfg
        filled, level = self._slots()
        onehot = torch.nn.functional.one_hot(level, num_classes=c.n_blocks)  # (N,B,L)
        return (onehot * filled.unsqueeze(-1)).sum(dim=1)  # (N,L)

    def tower_height(self) -> torch.Tensor:
        """(N,) int: height of the contiguous tower built from the pad up (slot 0, 1, 2, …
        each occupied by at least one block until the first gap)."""
        counts = self._level_counts()  # (N,L)
        occupied = counts >= 1
        # contiguous prefix length: cumulative product of occupancy along the level axis.
        prefix = torch.cumprod(occupied.long(), dim=1)
        return prefix.sum(dim=1)

    def score(self) -> torch.Tensor:
        """(N,) int in [0,100]: linear in the contiguous tower height (0 blocks -> 0, all
        `n_blocks` -> 100)."""
        h = self.tower_height().float()
        return (100.0 * h / self.cfg.n_blocks).round().long()

    def success(self) -> torch.Tensor:
        """(N,) bool: one clean tower — every slot 0..n_blocks-1 filled by exactly one block
        (so all blocks are up, aligned, upright and settled, with none left over)."""
        counts = self._level_counts()  # (N,L)
        return (counts == 1).all(dim=1)
