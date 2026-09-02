"""PushShapesScene — push three differently-shaped blocks onto their matching pads.

A multi-stage extension of RoboDojo's single-block ``push_T``. Three plates (T, X, L) start
in a row in front of the robot, each yawed away from its pad; three colour-keyed pads sit
in a row beyond them. Solving requires, per piece, a pivot push to correct yaw and a
translation to seat it — and doing so without knocking an already-seated neighbour out of
tolerance, which is what makes the pad row deliberately tight.

Why this shape of task: the single-push original needed no reorientation (block and pad
shared a yaw) and its rotation was clamped ~500x below the Isaac Lab default, so the
orientation tolerance only absorbed push-induced error instead of being a goal. Here yaw is
a goal, rotation is physical, and the per-piece correspondence gives real stages.

Progress is latched per stage in ``post_step`` (the suite's coffee/syringe idiom) so partial
credit survives a later mistake, while ``success()`` stays a LIVE conjunction — disturbing a
seated piece therefore still fails the run.

Source tolerances are preserved per piece: centre XY error at most 7 mm, orientation error
at most 7 degrees, and no lift above the bench. Heavy Isaac Lab imports stay deferred for
app-free discovery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv

PIECES = ("t", "x", "l")


@dataclass
class PushShapesSceneCfg(BaseCfg):
    """Rubric, layout, and measured asset constants for :class:`PushShapesScene`."""

    # Per-piece success gate, the RoboDojo push_T rubric verbatim.
    xy_tolerance: float = 0.007
    orientation_tolerance_deg: float = 7.0
    lift_tolerance: float = 0.010

    # Settling and the graded bands used for per-piece partial credit.
    settle_linear_speed: float = 0.03
    settle_angular_speed: float = 0.20
    near_xy_tolerance: float = 0.030
    yaw_stage_tolerance_deg: float = 10.0

    # Layout. Blocks sit nearer the robot than their stations so every piece is pushed away
    # from the body (+y), the stroke the G1's bench-contact workspace actually affords.
    #
    # Display scale for the whole piece set (blocks AND pads, so silhouettes keep matching).
    # 1.15 reads clearly larger on camera while the recompacted X/L footprints still clear
    # their neighbours at reachable spacings.
    piece_scale: float = 1.15
    # Station centres (pad poses), one per piece in PIECES order. MUST mirror STATIONS in
    # scripts/author_push_shapes_assets.py: the recessed board is authored with its cutouts
    # at exactly these poses, and that script also VALIDATES the layout (cutout separation,
    # spawn clearance, push corridors) before it will write assets. The x placement is
    # reach-derived -- the right hand completes at x in [0, +0.12], the leftmost station is
    # finished by the two-hand relay -- and depth is bound by RUNWAY: a piece must spawn
    # clear of its own cutout, so each lane needs a push of at least a piece depth plus
    # margins inside a workspace only ~150 mm deep.
    pad_stations: tuple[tuple[float, float], ...] = (
        (-0.066, -0.205), (0.018, -0.162), (0.122, -0.188),
    )
    # Blocks spawn this far behind their station, straight down-lane (+y push). Per piece:
    # the T's and L's runways are trimmed to keep spawns inside the arms' proven contact
    # range (their stations sit in the workspace's deep corners).
    push_lengths: tuple[float, float, float] = (0.112, 0.118, 0.110)
    # The raised plate the pieces slide on. An aligned piece DROPS into its cutout; that
    # drop is what seated() grades. Must match the authored board asset's thickness.
    board_thickness: float = 0.006
    # Yaw the pads share; each BLOCK additionally starts off by yaw_offset_deg so correcting
    # orientation is part of the task rather than a disturbance budget.
    base_yaw_deg: float = 90.0
    # The reorientation the task demands. 18 degrees proved to sit past what a single-contact
    # push can correct while also translating ~8 cm: the steering has authority to hold a
    # small error but the push induces yaw faster than it can shed a large one. 10 degrees
    # still exceeds the 7 degree success gate, so correcting it remains mandatory rather than
    # incidental, while leaving the task solvable. Raise it once a stronger controller exists.
    # 15 +/- 3 degrees: the worst-case start (12) must sit OUTSIDE yaw_stage_tolerance_deg
    # (10) or a yaw stage can latch at reset (caught by the smoke as a nonzero reset score
    # when 10 +/- 4 could jitter down to 6.7), and the best case (18) stays inside the range
    # the reference policy demonstrably corrects (it recovered 20-degree starts).
    yaw_offset_deg: float = 15.0
    # 3 mm, matching SPAWN_JITTER in the authoring script's layout validation: spawn
    # footprints are proven clear of every cutout only up to this jitter.
    reset_pos_jitter: float = 0.003
    reset_yaw_jitter_deg: float = 3.0

    # Measured from the authored/vendored USD extents (scripts/author_push_shapes_assets.py,
    # and the RoboDojo T at 80x60x15 mm).
    block_half_height: float = 0.0075
    block_mass: float = 0.35
    block_reset_clearance: float = 0.0015
    pad_lift: float = 0.0008
    surface_z: float = 0.7
    table_depth_scale: float = 1.5
    workbench_pos: tuple[float, float] = (0.0, 0.0)
    asset_dir: str = ""
    block_usd: dict[str, str] = field(default_factory=dict)
    pad_usd: dict[str, str] = field(default_factory=dict)
    board_usd: str = ""
    workbench_usd: str = ""

    def __post_init__(self) -> None:
        suite_assets = Path(__file__).resolve().parents[1] / "assets"
        shapes = suite_assets / "push_shapes"
        self.asset_dir = self.asset_dir or str(shapes)
        # The T BLOCK stays the vendored RoboDojo mesh; X and L are authored beside it.
        self.block_usd = self.block_usd or {
            "t": str(suite_assets / "push_shapes" / "block_t" / "main.usda"),
            "x": str(shapes / "block_x" / "main.usda"),
            "l": str(shapes / "block_l" / "main.usda"),
        }
        self.pad_usd = self.pad_usd or {
            key: str(shapes / f"pad_{key}" / "main.usda") for key in PIECES
        }
        self.board_usd = self.board_usd or str(shapes / "board" / "main.usda")
        self.workbench_usd = self.workbench_usd or str(
            suite_assets.parents[1]
            / "assembly"
            / "assets"
            / "props"
            / "packing_table"
            / "SM_HeavyDutyPackingTable_C02_01_physics.usd"
        )

    def station(self, index: int) -> tuple[float, float]:
        """Pad (station) centre for piece `index`."""
        return self.pad_stations[index]

    def block_spawn(self, index: int) -> tuple[float, float]:
        """Block spawn centre: straight down-lane behind the station."""
        px, py = self.pad_stations[index]
        return px, py - self.push_lengths[index]

    @property
    def slide_z(self) -> float:
        """Resting height of a block centre while it slides ON the board."""
        return self.surface_z + self.board_thickness + self.block_half_height * self.piece_scale


@SCENES.register("push_shapes")
class PushShapesScene(BaseScene):
    cfg: PushShapesSceneCfg

    # Latched stage flags, in the order they can physically be earned.
    STAGES = ("t_yaw", "t_seated", "x_yaw", "x_seated", "l_yaw", "l_seated", "all_seated")

    def __init__(self, cfg: PushShapesSceneCfg | None = None) -> None:
        super().__init__(cfg or PushShapesSceneCfg())

    # ----- assets ---------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Bench, three kinematic non-colliding pads, and three dynamic blocks."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        for usd in (*c.block_usd.values(), *c.pad_usd.values(), c.board_usd,
                    c.workbench_usd):
            if not Path(usd).is_file():
                raise FileNotFoundError(
                    f"{usd} not found — run scripts/author_push_shapes_assets.py (authored "
                    f"pieces) and scripts/vendor_push_shapes_t_asset.py (the RoboDojo T)"
                )

        table_scale = 0.01
        table_top_offset = 0.994
        table_height = 0.994
        table_z = c.surface_z - table_top_offset
        ground_z = c.surface_z - table_height
        wx, wy = c.workbench_pos

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    usd_path=str(
                        Path(__file__).resolve().parents[2]
                        / "assembly" / "assets" / "props" / "ground" / "default_ground.usd"
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
            # The recessed work board: pieces slide on its top; the success condition is a
            # block dropping into its station's cutout. Authored with cutouts at EXACTLY
            # cfg.pad_stations (see the authoring script's validation). The board is not a
            # graded object, so it spawns unscaled with exact colliders and full collision.
            "board": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Board",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.board_usd,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(0.0, 0.0, c.surface_z + c.board_thickness / 2.0)
                ),
            ),
        }

        for index, key in enumerate(PIECES):
            station_x, station_y = c.station(index)
            spawn_x, spawn_y = c.block_spawn(index)
            out[f"pad_{key}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Pad{key.upper()}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.pad_usd[key],
                    scale=(c.piece_scale, c.piece_scale, c.piece_scale),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(station_x, station_y, c.surface_z + c.pad_lift)
                ),
            )
            out[f"block_{key}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Block{key.upper()}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.block_usd[key],
                    scale=(c.piece_scale, c.piece_scale, c.piece_scale),
                    activate_contact_sensors=True,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.block_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True, contact_offset=0.002, rest_offset=0.0
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=2,
                        max_depenetration_velocity=0.25,
                        # Physical damping for a light plate on a laminate bench. The
                        # the first port clamped max_angular_velocity to 0.035 (~500x under the
                        # Isaac Lab default), which made deliberate reorientation impossible;
                        # rotation is a GOAL here, so only mild damping remains.
                        linear_damping=0.15,
                        angular_damping=0.35,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(spawn_x, spawn_y, c.slide_z + c.block_reset_clearance)
                ),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**21,
                "gpu_collision_stack_size": 2**27,
            },
        )

    # ----- binding --------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.blocks: dict[str, RigidObject] = {k: env.iscene[f"block_{k}"] for k in PIECES}
        self.pads: dict[str, RigidObject] = {k: env.iscene[f"pad_{k}"] for k in PIECES}
        self.env_origins = env.iscene.env_origins
        self._flags = torch.zeros(
            env.num_envs, len(self.STAGES), dtype=torch.bool, device=env.device
        )

    # ----- reset / state --------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Place each pad at its lane and each block one lane-length nearer, yawed off."""
        c = self.cfg
        device = self.env.device
        count = len(env_ids)
        origins = self.env_origins[env_ids]
        base = math.radians(c.base_yaw_deg)

        def pose(x: float, y: float, z: float, yaw: torch.Tensor,
                 pos_jitter: float) -> torch.Tensor:
            jitter = (torch.rand(count, 2, device=device) * 2 - 1) * pos_jitter
            state = torch.zeros(count, 13, device=device)
            state[:, 0] = x + jitter[:, 0]
            state[:, 1] = y + jitter[:, 1]
            state[:, 2] = z
            state[:, 3] = torch.cos(yaw / 2)
            state[:, 6] = torch.sin(yaw / 2)
            state[:, 0:3] += origins
            return state

        for index, key in enumerate(PIECES):
            station_x, station_y = c.station(index)
            spawn_x, spawn_y = c.block_spawn(index)
            pad_yaw = torch.full((count,), base, device=device)
            # Alternate the offset sign per lane so a solution cannot assume one pivot
            # direction, and jitter it so the magnitude is not memorizable either.
            sign = 1.0 if index % 2 == 0 else -1.0
            offset = math.radians(c.yaw_offset_deg) * sign
            jitter = (torch.rand(count, device=device) * 2 - 1) * math.radians(
                c.reset_yaw_jitter_deg
            )
            block_yaw = pad_yaw + offset + jitter
            # Pads take NO positional jitter: they mark the board's cutouts, which are
            # authored geometry and cannot move with them.
            self.pads[key].write_root_state_to_sim(
                pose(station_x, station_y, c.surface_z + c.pad_lift, pad_yaw,
                     pos_jitter=0.0), env_ids
            )
            self.blocks[key].write_root_state_to_sim(
                pose(spawn_x, spawn_y, c.slide_z + c.block_reset_clearance, block_yaw,
                     pos_jitter=c.reset_pos_jitter),
                env_ids,
            )
        self._flags[env_ids] = False

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        state: dict[str, Any] = {"flags": self._flags[env_ids].clone()}
        for key in PIECES:
            state[f"block_{key}"] = self.blocks[key].data.root_state_w[env_ids].clone()
            state[f"pad_{key}"] = self.pads[key].data.root_state_w[env_ids].clone()
        return state

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for key in PIECES:
            self.blocks[key].write_root_state_to_sim(state[f"block_{key}"], env_ids)
            self.pads[key].write_root_state_to_sim(state[f"pad_{key}"], env_ids)
        if "flags" in state:
            self._flags[env_ids] = state["flags"].to(self._flags.device)

    def describe(self) -> str:
        return (
            "A raised work board lies on the table in front of you, with three shape "
            "cutouts sunk into it: a T, a cross, and an L, their recess floors coloured "
            "dark red, dark blue, and dark amber. Three flat coloured blocks rest on the "
            "board, each some distance behind the cutout of its own shape: a red T, a blue "
            "cross, and an amber L.\n"
            "Each block starts rotated away from its cutout's orientation.\n"
            "Goal: slide every block across the board into the cutout of its own shape so "
            "that it drops in and lies flat inside its recess, matching the recess outline "
            "in position and rotation. The blocks can only be pushed; lifting them does "
            "not solve the task. A block that arrives misaligned will catch on the recess "
            "rim instead of dropping in, and a wedged block is hard to recover — align "
            "each block before it reaches its cutout."
        )

    # ----- per-piece measurements -----------------------------------------------------
    def position_error(self, key: str) -> torch.Tensor:
        """Planar centre distance between one block and its pad, in metres."""
        block = self.blocks[key].data.root_pos_w[:, :2]
        pad = self.pads[key].data.root_pos_w[:, :2]
        return (block - pad).norm(dim=1)

    def orientation_error(self, key: str) -> torch.Tensor:
        """Shortest full 3-D quaternion error between one block and its pad, in radians."""
        dot = (
            self.blocks[key].data.root_quat_w * self.pads[key].data.root_quat_w
        ).sum(dim=1)
        return 2.0 * torch.acos(dot.abs().clamp(max=1.0))

    def block_bottom_z(self, key: str) -> torch.Tensor:
        """Height of the block's underside above the env origin."""
        local_z = (self.blocks[key].data.root_pos_w - self.env_origins)[:, 2]
        return local_z - self.cfg.block_half_height * self.cfg.piece_scale

    def not_lifted(self, key: str) -> torch.Tensor:
        """The block bottom stays inside the 10 mm lift allowance above the BOARD top
        while sliding (and may of course sit lower, inside a recess)."""
        bottom_z = self.block_bottom_z(key)
        top = self.cfg.surface_z + self.cfg.board_thickness
        return (bottom_z <= top + self.cfg.lift_tolerance) & (
            bottom_z >= self.cfg.surface_z - 0.005
        )

    def dropped(self, key: str) -> torch.Tensor:
        """The block has physically fallen INTO a recess: its underside is below the board
        top by most of the board thickness, i.e. it rests on the bench, not on the board."""
        return self.block_bottom_z(key) <= self.cfg.surface_z + 0.002

    def settled(self, key: str) -> torch.Tensor:
        block = self.blocks[key].data
        return (block.root_lin_vel_w.norm(dim=1) < self.cfg.settle_linear_speed) & (
            block.root_ang_vel_w.norm(dim=1) < self.cfg.settle_angular_speed
        )

    def yaw_matched(self, key: str) -> torch.Tensor:
        """Coarse orientation stage: within `yaw_stage_tolerance_deg` of the pad."""
        return self.orientation_error(key) <= math.radians(self.cfg.yaw_stage_tolerance_deg)

    def seated(self, key: str) -> torch.Tensor:
        """The per-piece success gate: aligned with its own station AND physically dropped
        into the recess. The drop is the task's defining event -- a block resting on the
        board top over its cutout, however well aligned, has not been placed."""
        c = self.cfg
        return (
            (self.position_error(key) <= c.xy_tolerance)
            & (self.orientation_error(key) <= math.radians(c.orientation_tolerance_deg))
            & self.not_lifted(key)
            & self.dropped(key)
        )

    # ----- staging --------------------------------------------------------------------
    def post_step(self) -> None:
        """Latch stage progress. Latching is what makes partial credit survive a later
        mistake; `success()` stays live so the mistake still costs the run."""
        live = self.stage_live()
        self._flags |= live

    def stage_live(self) -> torch.Tensor:
        """(N, n_stages) bool: the stage conditions as measured RIGHT NOW."""
        columns = []
        for key in PIECES:
            columns.append(self.yaw_matched(key))
            columns.append(self.seated(key) & self.settled(key))
        columns.append(
            torch.stack([self.seated(k) & self.settled(k) for k in PIECES], dim=1).all(dim=1)
        )
        return torch.stack(columns, dim=1)

    def stage_flags(self) -> torch.Tensor:
        return self._flags.clone()

    def score(self) -> torch.Tensor:
        """(N,) int 0-100: even partial credit over the latched stage flags."""
        earned = self._flags.long().sum(dim=1)
        total = len(self.STAGES)
        return (earned * 100 + total // 2) // total

    def success(self) -> torch.Tensor:
        """(N,) bool: all three pieces seated on their own pad and settled, measured LIVE."""
        return torch.stack(
            [self.seated(k) & self.settled(k) for k in PIECES], dim=1
        ).all(dim=1)
