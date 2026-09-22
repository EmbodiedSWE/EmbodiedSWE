"""BoxToBinScene — a cardboard box picked off a shelf board and carried to a sorting bin on a low table.

A steel shelving unit stands in front of the robot's start pose with a 20 cm cardboard box on its
middle board; a low table with an open blue sorting bin stands about 1.4 m away around a ~100 deg
turn. Goal (carried here — no task layer): pick the box off the shelf, carry it to the table, and
leave it resting inside the bin. Like `wheel_carry`, the distance is the point — the bin is past any
fixed-base reach, so the task cannot be finished without the robot moving its own base — but the
route adds what `wheel_carry`'s straight corridor does not: a genuine heading change.

**Nothing about the robot is in here** — the scene is embodiment-agnostic, as every `BaseScene` is.

LINEAGE. This is a measured port of IsaacLab-Arena's `galileo_g1_locomanip_pick_and_place` task
(github.com/isaac-sim/IsaacLab-Arena). Every default below is a number MEASURED off that task's
assets rather than re-invented: the same brown box (0.20 m, 0.1 kg, friction 5.0), the same
`SM_OpenIndustrialSteelShelving_A03` shelf with the box on the 0.763 m board, the same
`sm_tabletop_a01` table z-scaled 0.70 so its top lands at 0.531 m, and the same blue sorting bin
scaled (4, 2, 1) to a 0.80 x 0.50 m tray with 5 cm walls. Arena's frame (robot facing +x) is rotated
+90 deg so the layout meets this repo's convention (a G1 spawns at the origin facing +y, no
placement override needed): the shelf face is 0.46 m ahead of the start pose, exactly Arena's
stand-off, and the bin sits to the robot's right where Arena's navigation waypoints send it. See
`assets/fetch_galileo_assets.py` for asset provenance (Arena's own USDs, vendored + extracted).

GEOMETRY. Everything stands on the floor at z = 0 — no sunk furniture here (unlike `wheel_carry`):
the shelf and table assets put their feet at their origin, so world heights read straight off the
assets. The shelf's three boards land at 0.375-0.413 / 0.725-0.763 / 1.449-1.488 m; the box rests on
the MIDDLE board (top 0.763) with 0.69 m of clear air to the board above, so the pick is a
waist-height front approach. The bin's interior floor is ~3 mm above the table top with the rim at
+0.051 — a flat tray the box cannot roll out of, but shallow enough that `placed()` must tell
"resting inside" apart from "balanced on the rim" (the height band does exactly that).

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from robobench.core.assets import asset_path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv

_SQ2 = math.sqrt(0.5)


@dataclass
class BoxToBinSceneCfg(BaseCfg):
    """Config for `BoxToBinScene`. Nothing is locked — a variant is just a copy with a few fields
    changed. The layout dials (`bin_pos`, `shelf_pos`) set the transit length and the turn angle;
    the defaults reproduce the Arena task's measured geometry."""

    # --- the task objects' placements (env-local, floor at z = 0) -------------------------------
    #: Shelf footprint center. The unit is 0.46 m deep x 1.61 m wide x 1.49 m tall; spawned at
    #: yaw +90 deg its width runs along x and its FRONT face is at y = shelf_pos[1] - 0.23, i.e.
    #: 0.463 — the stand-off Arena gives the robot at its start pose.
    shelf_pos: tuple[float, float] = (0.172, 0.693)
    #: Top of the middle shelf board — the pick surface (measured off the asset; the boards are at
    #: 0.375-0.413 / 0.725-0.763 / 1.449-1.488).
    pick_board_z: float = 0.763
    #: Where the box rests on that board (its center, so the box spans y 0.478-0.678 — the front
    #: face nearly flush with the shelf's front edge, as Arena spawns it).
    box_init: tuple[float, float] = (0.0, 0.5785)
    box_drop: float = 0.003  # spawn drop height above the resting pose (m)
    #: Arena spawns the box rolled/yawed pi (a cosmetic flip of a textured cube); identity here.
    box_init_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz

    #: Table footprint center and the z scale that makes the raw 0.758 m table the galileo room's
    #: low 0.531 m one. Spawned at yaw +90 deg: its 1.80 m length runs along y.
    table_pos: tuple[float, float] = (1.953, -0.2025)
    table_z_scale: float = 0.70
    #: Bin origin (base center). It stands on the table top, long side along y, its west wall at
    #: the table edge nearest the robot's approach.
    bin_pos: tuple[float, float] = (1.8072, -0.245)
    bin_scale: tuple[float, float, float] = (4.0, 2.0, 1.0)  # Arena's own stretch of the raw tray

    # --- success / progress bands ----------------------------------------------------------------
    # The box is "placed" when its origin sits inside the bin's interior inset by the box's half
    # width (so the whole box is genuinely inside, not overhanging a wall), no higher than
    # `place_max_h` above the table top, and settled. The inset bands are measured off the scaled
    # bin: interior half-extents 0.24 (x) / 0.36 (y) minus the 0.10 half-box.
    place_half_x: float = 0.14
    place_half_y: float = 0.26
    #: Max box-ORIGIN height above the table top to count as placed: bin floor (~3 mm) + half box
    #: (0.10) + settle tolerance. A box lying flat across the rim sits at +0.151 and must NOT count.
    place_max_h: float = 0.136
    settle_vel: float = 0.20  # max |v| per axis (m/s, and rad/s) to count as come-to-rest
    #: The `lifted` latch (see `post_step`): EITHER the box origin rises this far above its resting
    #: height on the board (a straight lift), OR it is seen held up inside the APRON just in front
    #: of the shelf face (a slide-off pick — equally legitimate on open shelving). The apron is a
    #: bounded window (`apron_depth` out from the face, `apron_half_x` around the box's start x),
    #: so a box already across the room — on the table, in the bin — can never trip this arm; any
    #: legitimate extraction passes through it.
    lift_h: float = 0.05
    carry_min_z: float = 0.35  # the box is "held up" above this (a box on the floor rests at 0.10)
    apron_depth: float = 0.40  # how far out from the shelf face the apron window reaches (m)
    apron_half_x: float = 0.50  # half width of the apron window around the box's start x (m)
    drop_z: float = 0.30  # below this the box has fallen to the floor -> `dropped()`
    reset_pos_jitter: float = 0.0  # uniform +/- xy jitter for the box at reset (m). Arena randomizes
    # +/- 0.025; the default 0 keeps the layout deterministic and bit-reproducible.
    #: Box friction (static = dynamic), written across the box's collider at bind. The asset authors
    #: a deliberately grippy 5.0 cardboard material; this dial flattens it to one value.
    box_friction: float = 5.0

    # --- structure, asset facts (fixed) -----------------------------------------------------------
    #: Raw asset numbers the derived properties below build on (all measured; see the fetch script).
    BOX_HALF: ClassVar[float] = 0.10
    TABLE_RAW_H: ClassVar[float] = 0.758
    BIN_WALL_Z: ClassVar[float] = 0.0505  # rim height above the bin origin (z scale is 1)
    BIN_FLOOR_Z: ClassVar[float] = 0.003  # interior floor plate top above the bin origin
    #: Scaled interior half-extents in the BIN's spawned orientation (long side along y):
    #: raw inner half 0.090 x 4 -> 0.36 along y, raw 0.120 x 2 -> 0.24 along x.
    BIN_INNER_HALF_X: ClassVar[float] = 0.24
    BIN_INNER_HALF_Y: ClassVar[float] = 0.36
    SHELF_SIZE: ClassVar[tuple[float, float, float]] = (0.460, 1.607, 1.488)  # depth, width, height

    light_intensity: float = 3000.0  # the scene's dome light
    light_color: tuple[float, float, float] = (0.75, 0.75, 0.75)
    # Asset USDs; empty -> the vendored trees under this suite's `assets/`. Re-vendor with
    # `uv run --with usd-core --with pillow python robobench/suites/locomanip/assets/fetch_galileo_assets.py`.
    asset_dir: str = ""
    box_usd: str = ""
    bin_usd: str = ""
    table_usd: str = ""
    shelf_usd: str = ""

    def __post_init__(self) -> None:
        assets = asset_path(Path(__file__).resolve().parents[1] / "assets")
        self.asset_dir = self.asset_dir or str(assets)
        d = Path(self.asset_dir)
        self.box_usd = self.box_usd or str(d / "brown_box" / "brown_box.usd")
        props = d / "galileo_props" / "exhaust_pipe_task" / "exhaust_pipe_assets"
        self.bin_usd = self.bin_usd or str(props / "blue_sorting_bin.usd")
        self.table_usd = self.table_usd or str(props / "table.usd")
        self.shelf_usd = self.shelf_usd or str(d / "galileo_shelf" / "galileo_shelf.usd")

    @property
    def table_top_z(self) -> float:
        """World height of the table top (the raw asset z-scaled onto the floor)."""
        return self.TABLE_RAW_H * self.table_z_scale

    @property
    def shelf_front_y(self) -> float:
        """World y of the shelf's front face — the plane a pick has to reach past."""
        return self.shelf_pos[1] - self.SHELF_SIZE[0] / 2

    def box_init_world(self) -> tuple[float, float, float]:
        """The box's ENV-LOCAL start pose: resting on the middle board plus the spawn drop."""
        return (self.box_init[0], self.box_init[1], self.pick_board_z + self.BOX_HALF + self.box_drop)

    def bin_interior(self) -> tuple[tuple[float, float], tuple[float, float], float, float]:
        """The bin interior in ENV-LOCAL coords: (x band, y band, floor top z, rim z) — what both
        `placed()` and anyone aiming a put-down read."""
        bx, by = self.bin_pos
        return ((bx - self.BIN_INNER_HALF_X, bx + self.BIN_INNER_HALF_X),
                (by - self.BIN_INNER_HALF_Y, by + self.BIN_INNER_HALF_Y),
                self.table_top_z + self.BIN_FLOOR_Z,
                self.table_top_z + self.BIN_WALL_Z)

    def place_box_world(self) -> tuple[tuple[float, float], tuple[float, float], float]:
        """The placed() target zone in ENV-LOCAL coords: (x band, y band, max box-origin height) —
        the interior inset so the whole 0.2 m box is inside, below the rim-excluding height cap."""
        bx, by = self.bin_pos
        return ((bx - self.place_half_x, bx + self.place_half_x),
                (by - self.place_half_y, by + self.place_half_y),
                self.table_top_z + self.place_max_h)

    def stand_xy(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Working poses (x, y, heading rad) at each station, ENV-LOCAL: (pick, place). Not
        constraints — the pick pose is where the robot already starts (facing +y, the Arena
        stand-off), and the place pose is Arena's own navigation goal, rotated into this frame."""
        return ((0.0, 0.0, math.pi / 2), (1.287, -0.0955, -0.21))


@SCENES.register("box_to_bin")
class BoxToBinScene(BaseScene):
    cfg: BoxToBinSceneCfg

    #: Per-env world variation at generation (engine sampler). Only the box is a live rigid body
    #: with a PhysX view — shelf, table, and bin are static spawns — so friction is the knob.
    PHYSICAL_PARAMS: ClassVar[dict[str, dict | None]] = {
        "box_friction": {"dist": "uniform", "lo": 4.0, "hi": 5.0,
                         "reason": "downward from the asset's authored 5.0 cardboard (its pristine, "
                                   "deliberately grippy max) — corrugated board's friction varies "
                                   "down from that with surface wear/dust, not up; the low end is "
                                   "where the two-palm grip actually has to work"},
    }

    #: Stage-wide look knobs for visual replay (data_engine render.py --visual_draw).
    VISUAL_PARAMS: ClassVar[dict[str, dict | None]] = {
        "light_intensity": {"dist": "uniform", "lo": 2500.0, "hi": 3500.0,
                            "reason": "around the 3000 nominal"},
    }

    #: L5 external views (see BaseScene.CAMERAS). `wide` stands off to the south-west and frames
    #: the shelf, the table, and the corner between them (the turn is the story); `pick` and
    #: `place` are the per-station close-ups, both angled so the robot does not block the object.
    CAMERAS: ClassVar[dict[str, dict]] = {
        "wide": {"eye": (-1.60, -2.60, 2.10), "target": (1.00, 0.20, 0.70), "focal": 18.0,
                 "bands": {
                     "eye_x": {"dist": "uniform", "lo": -1.90, "hi": -1.30},
                     "eye_y": {"dist": "uniform", "lo": -2.90, "hi": -2.30},
                     "eye_z": {"dist": "uniform", "lo": 1.90, "hi": 2.30},
                 }},
        "pick": {"eye": (-1.50, -0.35, 1.45), "target": (0.00, 0.58, 0.85), "focal": 20.0},
        "place": {"eye": (1.05, -1.95, 1.45), "target": (1.81, -0.25, 0.60), "focal": 20.0},
    }

    def __init__(self, cfg: BoxToBinSceneCfg | None = None) -> None:
        super().__init__(cfg or BoxToBinSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, dome light, the three pieces of static furniture, and the loose box. The box
        keeps the asset's own cube collider and density-derived 0.1 kg mass; only its friction is
        overridden (at bind). All furniture spawns at yaw +/-90 deg — the vendored assets are
        authored with the Arena room's axes, and this scene is that layout rotated to face +y."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        for usd in (c.box_usd, c.bin_usd, c.table_usd, c.shelf_usd):
            if not Path(usd).is_file():
                raise FileNotFoundError(
                    f"{usd} not found — re-vendor with `uv run --with usd-core --with pillow "
                    f"python robobench/suites/locomanip/assets/fetch_galileo_assets.py`")
        bx, by, bz = c.box_init_world()
        yaw_pos = (_SQ2, 0.0, 0.0, _SQ2)   # +90 deg about z, wxyz
        yaw_neg = (_SQ2, 0.0, 0.0, -_SQ2)  # -90 deg (the bin: Arena's pi flip plus the frame turn)
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(usd_path=str(
                    asset_path(Path(__file__).resolve().parents[2] / "assembly" / "assets") / "props" / "ground"
                    / "default_ground.usd")),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=c.light_intensity, color=c.light_color),
            ),
            # The SHELF authors colliders and physics materials but no rigid body, so it is already
            # static — asking for `kinematic_enabled` would find no rigid-body prim to set it on.
            "shelf": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Shelf",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(c.shelf_pos[0], c.shelf_pos[1], 0.0),
                                                        rot=yaw_pos),
                spawn=sim_utils.UsdFileCfg(usd_path=c.shelf_usd),
            ),
            # The TABLE and the BIN each author a PhysicsRigidBodyAPI, so here `kinematic_enabled`
            # is load-bearing: without it they are free bodies the robot (or the box) would shove.
            "table": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(c.table_pos[0], c.table_pos[1], 0.0),
                                                        rot=yaw_pos),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.table_usd,
                    scale=(1.0, 1.0, c.table_z_scale),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
            ),
            "bin": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Bin",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(c.bin_pos[0], c.bin_pos[1], c.table_top_z),
                                                        rot=yaw_neg),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.bin_usd,
                    scale=c.bin_scale,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
            ),
            "box": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Box",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.box_usd,
                    activate_contact_sensors=True,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(bx, by, bz), rot=c.box_init_quat),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        # Same substrate as wheel_carry: dt = 1/200 divides evenly into the humanoid control
        # periods (a 0.02 s period is exactly 4 physics steps), and the widened GPU buffers carry
        # over — mesh colliders (shelf boards, bin walls) against a walking robot's self-collisions.
        return SimCfg(
            dt=1.0 / 200.0,
            render={"enable_reflections": True},
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
            },
        )

    # ----- lifecycle ----------------------------------------------------------------------------
    def apply_visual_params(self, env: BaseEnv, values: dict[str, Any]) -> None:
        """The live subset of `VISUAL_PARAMS` (see BaseScene): the dome intensity is an attribute
        write (also reaching the light at build via the cfg, so this is redundant-but-harmless on a
        fresh build)."""
        unknown = set(values) - set(self.VISUAL_PARAMS)
        if unknown:
            raise ValueError(f"{type(self).__name__} cannot apply visuals: {sorted(unknown)}")
        if "light_intensity" in values:
            env.stage.GetPrimAtPath("/World/light").GetAttribute("inputs:intensity").Set(
                float(values["light_intensity"]))

    def apply_physical_params(self, env: BaseEnv, values: dict[str, list]) -> None:
        """Write the box's friction PER ENV (static = dynamic), `values[name]` one value per env.
        `bind()` routes the nominal through here with uniform values, so this is THE friction path —
        per-env sampling reuses it, never a copy."""
        unknown = set(values) - set(self.PHYSICAL_PARAMS)
        if unknown:
            raise ValueError(f"{type(self).__name__} cannot apply per-env: {sorted(unknown)}")
        if "box_friction" in values:
            mats = self.box.root_physx_view.get_material_properties()  # (n, n_shapes, 3)
            mats[..., 0:2] = torch.tensor(values["box_friction"], dtype=torch.float32).view(-1, 1, 1)
            self.box.root_physx_view.set_material_properties(mats, torch.arange(env.num_envs, device="cpu"))

    def bind(self, env: BaseEnv) -> None:
        """Grab the box handle, cache env origins, allocate the journey latches, and set friction.
        Once, after the build."""
        super().bind(env)
        self.box: RigidObject = env.iscene["box"]
        self.env_origins = env.iscene.env_origins
        z = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._lifted, self._transited = z.clone(), z.clone()
        self.apply_physical_params(env, {n: [getattr(self.cfg, n)] * env.num_envs for n in self.PHYSICAL_PARAMS})

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Advance the journey latches at sim rate. `lifted` = the box got clear of the pick board —
        either lifted straight up, or seen held up in the apron just in front of the shelf face
        (open shelving affords both pick styles; the apron bound is what keeps a box that is
        already elsewhere — in the bin, say — from tripping the arm). `transited` = it then
        travelled most of the way to the bin. Both are one-way: they prove the path was walked;
        the live `placed()` predicate is what proves the destination."""
        p = self.box.data.root_pos_w - self.env_origins
        c = self.cfg
        rest_z = c.pick_board_z + c.BOX_HALF
        up = p[:, 2] > rest_z + c.lift_h
        drawn_out = ((p[:, 1] < c.shelf_front_y - 0.05) & (p[:, 1] > c.shelf_front_y - c.apron_depth)
                     & ((p[:, 0] - c.box_init[0]).abs() < c.apron_half_x) & (p[:, 2] > c.carry_min_z))
        self._lifted |= up | drawn_out
        self._transited |= self._lifted & (self.carried_fraction() > 0.9)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Re-place the box on the middle board at its start pose (+ optional xy jitter), at rest,
        and clear the journey latches. The furniture is static and never moves."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = self.env_origins[env_ids] + torch.tensor(c.box_init_world(), device=dev)
        if c.reset_pos_jitter:
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
        st[:, 3:7] = torch.tensor(c.box_init_quat, device=dev)
        self.box.write_root_state_to_sim(st, env_ids)
        self._lifted[env_ids] = False
        self._transited[env_ids] = False

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        """Restorable state: the box's world root state (13) plus the journey latches. Everything
        else in the scene is static furniture. The latches travel with the state so a snapshot
        restored mid-carry does not silently lose the milestones already earned."""
        return {
            "box": self.box.data.root_state_w[env_ids].clone(),
            "lifted": self._lifted[env_ids].clone(),
            "transited": self._transited[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        """Restore `get_state`: the box's root state and the journey latches."""
        self.box.write_root_state_to_sim(state["box"], env_ids)
        self._lifted[env_ids] = state["lifted"]
        self._transited[env_ids] = state["transited"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        _, _, fz, rz = c.bin_interior()
        (px, py, ph), (tx, ty, th) = c.stand_xy()
        return (
            f"A steel shelving unit stands with its face {c.shelf_front_y:.2f} m ahead of the "
            f"start pose, boards at 0.41, 0.76 and 1.49 m. On the middle board sits a brown "
            f"cardboard box, 20 cm on a side and about 0.1 kg, its center {c.pick_board_z + c.BOX_HALF:.2f} m "
            f"up, nearly flush with the shelf's front edge and with 0.69 m of clear air above it. "
            f"To the right, about 1.4 m away, a low table (top {c.table_top_z:.2f} m) carries an "
            f"open blue sorting bin: interior about {2 * c.BIN_INNER_HALF_X * 100:.0f} x "
            f"{2 * c.BIN_INNER_HALF_Y * 100:.0f} cm around ({c.bin_pos[0]:.2f}, {c.bin_pos[1]:.2f}), "
            f"floor at {fz:.2f} m, walls just {rz - fz + 0.003:.2f} m tall — a shallow tray the box "
            f"drops into, not a deep basket.\n"
            f"Goal: pick the box up off the shelf, carry it to the table, and leave it resting "
            f"inside the bin. The task is complete once the box sits fully inside the bin's walls, "
            f"released and come to rest — not still held, not straddling the rim.\n"
            f"The bin is further away than any arm reaches, so the box can only get there if the "
            f"robot itself does — and the table stands off the shelf's axis, so the route turns. "
            f"Working poses (x, y, heading): about ({px:.2f}, {py:.2f}, {ph:+.2f}) for the pick — "
            f"the start pose already faces the shelf squarely — and ({tx:.2f}, {ty:.2f}, {th:+.2f}) "
            f"for the place; they are a starting guess, not a requirement."
        )

    # ----- progress (public: reads how far the task has got) ------------------------------------
    def success(self) -> torch.Tensor:
        """Whether the task is done, `(num_envs,)`. The latches prove the journey — the box really
        was taken off the shelf and carried across, not spawned or slid there — and `placed()`
        proves the destination is where it ended up."""
        return self._lifted & self._transited & self.placed()

    def score(self) -> torch.Tensor:
        """Progress in [0, 100], `(num_envs,)`, over the ordered chain lifted -> transited ->
        placed. A stage reached out of order earns nothing until its predecessors are in: the chain
        is the task. `carried_fraction` fills in the middle so a run that walks half way and drops
        the box scores above one that never moved."""
        part = 40.0 * self.carried_fraction().clamp(0.0, 1.0)
        s = torch.zeros_like(part)
        s = torch.where(self._lifted, 20.0 + part, s)
        s = torch.where(self._lifted & self._transited, torch.full_like(s, 70.0), s)
        return torch.where(self.success(), torch.full_like(s, 100.0), s)

    def placed(self) -> torch.Tensor:
        """Whether the box is placed, `(num_envs,)`: its origin inside the bin's inset interior
        bands, no higher than `place_max_h` above the table top (which excludes a box balanced flat
        across the rim at +0.151), and settled to under `settle_vel` on every axis.
        Embodiment-agnostic by construction: it reads the box and the bin only, never the robot."""
        (x0, x1), (y0, y1), max_z = self.cfg.place_box_world()
        p = self.box.data.root_pos_w - self.env_origins  # env-local, like every target here
        inside = (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1) & (p[:, 2] < max_z)
        return inside & self.settled()

    def settled(self) -> torch.Tensor:
        """Whether the box has come to rest, `(num_envs,)`: every component of its linear AND
        angular velocity under `settle_vel`."""
        return self.box.data.root_vel_w.abs().amax(dim=-1) < self.cfg.settle_vel

    def dropped(self) -> torch.Tensor:
        """Whether the box has fallen to the floor, `(num_envs,)` — a readable signal, not a forced
        reset: this scene never terminates an episode on its own. A live hazard for the whole
        transit, since the box spends the walk in mid-air."""
        z = self.box.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return z < self.cfg.drop_z

    def carried_fraction(self) -> torch.Tensor:
        """How far along the pick -> place transit the box has got, `(num_envs,)` in [0, 1] — the
        fraction of the start-to-bin distance closed, measured on the BOX (the thing the task is
        about), not on the robot. Unlike wheel_carry's straight-x corridor this transit turns, so
        the fraction is radial: 1 - (xy distance left to the bin center) / (the full span)."""
        c = self.cfg
        p = self.box.data.root_pos_w - self.env_origins
        target = torch.tensor(c.bin_pos, device=p.device)
        span = float(torch.tensor(
            [c.box_init[0] - c.bin_pos[0], c.box_init[1] - c.bin_pos[1]]).norm())
        left = (p[:, :2] - target).norm(dim=-1)
        return (1.0 - left / max(span, 1e-6)).clamp(0.0, 1.0)

    def box_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The box's ENV-LOCAL position `(num_envs, 3)` and orientation `(num_envs, 4)` wxyz — the
        frame every target in this scene is quoted in."""
        return self.box.data.root_pos_w - self.env_origins, self.box.data.root_quat_w
