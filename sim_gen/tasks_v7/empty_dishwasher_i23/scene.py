"""TileShuntScene — route the RED tile through a captive 2x3 sliding-block puzzle to
the corner cell marked by the green post (sim_gen task `empty_dishwasher_i23`).

Derived from rlbench/empty_dishwasher, but STRATEGICALLY different: the seed is an
open-appliance EXTRACTION — swing the dishwasher door down, pull the sliding rack
out along its prismatic run, then pick the plate up and OUT of the machine. Every
interaction there serves removal: create an opening, expose the payload, lift it
away. Here removal is physically impossible and appears nowhere in the goal — the
plan is a combinatorial IN-PLACE REARRANGEMENT:

  - a shallow frame holds SIX cells (2 rows x 3 columns) under a fixed slotted LID;
    FIVE square tiles live in the cells, each carrying a knob that sticks up through
    the lid's ladder of slots. The 60 mm tile plates are captive under the lid
    (slots are 24 mm) — no tile can ever be lifted out, and the lid gap is too thin
    for a tile to climb another;
  - exactly ONE cell is empty. The only physically possible move is to slide a tile
    ADJACENT to the empty cell into it (a full line of tiles jams against the frame
    wall — smoke proves both the captivity and the blocked-line discipline by
    force probes);
  - the goal is to park the ONE RED tile centered in the corner cell nearest the
    green post outside the frame. Since the red tile can only advance into the
    empty cell, the solver must first CYCLE THE WHITE DISTRACTOR TILES around the
    frame to walk the empty cell onto the red tile's path — a make-way plan (the
    classic 15-puzzle discipline), replanned per episode from the randomized
    arrangement.

A solver therefore needs a different PLAN from the seed (no door, no rack, no
grasp-and-remove; instead: read the arrangement, plan a move sequence over a shared
empty cell, execute constrained one-cell slides) and a different code structure (an
occupancy model and a move planner instead of an open-pull-lift script).

success() iff, settled (all five tiles |v| < settle_lin):
  - the RED tile's plate center is within `goal_tol` of the goal-corner cell
    center (frame frame), inside the captive height band (under the lid — a tile
    perched ON the lid is rejected);
  - all five tiles are inside the frame interior in the captive band.

score() is latched every physics substep (credit never evaporates):
  0.15 * made-way (any WHITE tile settled into a DIFFERENT cell than it started)
+ 0.55 * best red progress (fraction of the red tile's initial cell-manhattan
         distance to the goal that has been closed, gated on the red tile being
         captive inside the frame — a tile teleported outside earns nothing)
+ 0.15 * red-at-goal (the red plate center first enters the goal tolerance),
capped at 0.85; exactly 1.0 iff success(). Doing nothing scores ~0, and the seed's
strategy (take the payload OUT of the appliance) is rejected: a red tile outside
the frame fails in_frame and can never reach success.

Assets are fully procedural (axis-aligned `UsdGeom.Cube` + `UsdGeom.Cylinder`
knobs, explicit 1 mm contact offsets — the ~2 cm default would eat the 5-10 mm
working clearances):
  - frame: KINEMATIC compound, ~230 x 160 x 43 mm: floor slab, four walls up to
    lid-top height, and a 12-patch lid whose openings form a slot ladder along the
    row and column lines (slot width 24 mm);
  - tiles: DYNAMIC compounds, one RED + four WHITE, identical geometry: 60 x 60 x
    22 mm plate (captive: 60 > 24) + 16 mm dia x 58 mm knob (rides the slots,
    always graspable above the lid);
  - marker: KINEMATIC green post placed diagonally outside the goal corner.

Per-episode randomization (verified by READBACK in smoke): frame xy + free yaw
(the whole cell lattice rotates — read it from the scene), the tile arrangement
(which cell is empty, where the red tile starts) and the goal corner, sampled so
the red tile never starts on the goal.
Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _cyl(stage, path: str, radius: float, height: float, center, color,
         contact_offset: float) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    r, h = float(radius), float(height)
    seg.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _rigid_dynamic(root, mass: float) -> None:
    """Dynamic rigid-body armor on a compound root: mass (PhysX derives inertia from
    the child colliders), damping so tiles park promptly, no sleeping while we judge
    velocities, and the depenetration cap."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.10)
    px.CreateAngularDampingAttr(0.20)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _spawn_frame(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC puzzle frame at `prim_path`. Origin = frame centre at
    ground level. Floor slab under the 3x2 cell lattice, four walls rising to the
    lid top, and a lid of 12 rectangular patches leaving a slot LADDER open along
    the three column lines (x = -pitch, 0, +pitch) and the two row lines
    (y = +/- pitch/2)."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(5.0)

    co = cfg.contact_offset
    body_c, lid_c = cfg.body_color, cfg.lid_color
    p = cfg.pitch
    hx, hy = cfg.inner_hx, cfg.inner_hy
    wt, ft = cfg.wall_t, cfg.floor_t
    lz0, lt = cfg.lid_z0, cfg.lid_t
    wh = lz0 + lt  # walls run ground -> lid top (no escape gap at the rim)
    sh = cfg.slot_w / 2

    # floor slab (the sliding surface)
    _box(stage, f"{prim_path}/floor", (2 * hx, 2 * hy, ft), (0.0, 0.0, ft / 2), body_c, co)
    # walls (y-walls span the full outer x extent)
    _box(stage, f"{prim_path}/wall_yn", (2 * hx + 2 * wt, wt, wh),
         (0.0, -(hy + wt / 2), wh / 2), body_c, co)
    _box(stage, f"{prim_path}/wall_yp", (2 * hx + 2 * wt, wt, wh),
         (0.0, hy + wt / 2, wh / 2), body_c, co)
    _box(stage, f"{prim_path}/wall_xn", (wt, 2 * hy, wh), (-(hx + wt / 2), 0.0, wh / 2),
         body_c, co)
    _box(stage, f"{prim_path}/wall_xp", (wt, 2 * hy, wh), (hx + wt / 2, 0.0, wh / 2),
         body_c, co)
    # lid patches: solid x-intervals between the three column slots, solid
    # y-intervals between the two row slots
    xs = ((-hx, -p - sh), (-p + sh, -sh), (sh, p - sh), (p + sh, hx))
    ys = ((-hy, -p / 2 - sh), (-p / 2 + sh, p / 2 - sh), (p / 2 + sh, hy))
    for a, (x0, x1) in enumerate(xs):
        for b, (y0, y1) in enumerate(ys):
            _box(stage, f"{prim_path}/lid_{a}{b}", (x1 - x0, y1 - y0, lt),
                 ((x0 + x1) / 2, (y0 + y1) / 2, lz0 + lt / 2), lid_c, co)
    return root


def _spawn_tile(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a DYNAMIC puzzle tile at `prim_path`. Origin = PLATE CENTRE; local +z
    up: square plate (captive under the lid) + knob cylinder (rides the slots,
    graspable above the lid)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass)

    co, color = cfg.contact_offset, cfg.color
    _box(stage, f"{prim_path}/plate", (cfg.tile_w, cfg.tile_w, cfg.tile_h),
         (0.0, 0.0, 0.0), color, co)
    _cyl(stage, f"{prim_path}/knob", cfg.knob_r, cfg.knob_len,
         (0.0, 0.0, cfg.tile_h / 2 + cfg.knob_len / 2), color, co)
    return root


def _frame_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "frame" not in _SPAWNER_CACHE:

        @configclass
        class ShuntFrameSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_frame)
            pitch: float = 0.070
            inner_hx: float = 0.105
            inner_hy: float = 0.070
            wall_t: float = 0.010
            floor_t: float = 0.008
            lid_z0: float = 0.035
            lid_t: float = 0.008
            slot_w: float = 0.024
            body_color: tuple = (0.45, 0.47, 0.52)
            lid_color: tuple = (0.30, 0.32, 0.37)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["frame"] = ShuntFrameSpawnerCfg

    return _SPAWNER_CACHE["frame"](
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        **kw,
    )


def _tile_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tile" not in _SPAWNER_CACHE:

        @configclass
        class ShuntTileSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tile)
            tile_w: float = 0.060
            tile_h: float = 0.022
            knob_r: float = 0.008
            knob_len: float = 0.058
            mass: float = 0.15
            color: tuple = (0.92, 0.92, 0.92)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["tile"] = ShuntTileSpawnerCfg

    return _SPAWNER_CACHE["tile"](
        mass_props=sim_utils.MassPropertiesCfg(mass=kw["mass"]),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        **kw,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TileShuntSceneCfg(BaseCfg):
    """Config for `TileShuntScene`. Honesty knobs asserted in `__post_init__`: the
    plate is captive under the lid slots (no lift-out), tiles slide their cells with
    real clearance, the knob rides the slots with real clearance and stays graspable
    above the lid, and the goal tolerance cleanly separates adjacent cells."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    goal_tol: float = tunable(0.018)  # red plate centre within this of the goal cell centre (m)
    settle_lin: float = tunable(0.05)  # max |lin vel| of every tile when judging (m/s)
    band_z_lo: float = tunable(0.012)  # captive height band for a tile ORIGIN (plate centre):
    band_z_hi: float = tunable(0.026)  # resting 0.019, lid-pressed 0.024, lid-percher 0.054

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    frame_jitter: float = tunable(0.04)  # uniform +/- xy jitter of the frame at reset (m)
    frame_yaw_deg: float = tunable(180.0)  # uniform +/- frame yaw (free — read the lattice)
    shuffle: bool = tunable(True)  # randomize arrangement + goal corner (demo sets False)

    # --- tunable: placement ------------------------------------------------------------------
    frame_pos: tuple = tunable((0.0, 0.0))  # frame centre, nominal

    # --- info: lattice -----------------------------------------------------------------------
    pitch: float = info(0.070)  # cell pitch; cells k=0..5 at (x,y)=((k%3-1)p, (k//3-.5)p)
    ncols: int = info(3)
    nrows: int = info(2)
    # --- info: frame structure ---------------------------------------------------------------
    inner_hx: float = info(0.105)  # interior half-extent x (= 1.5 * pitch)
    inner_hy: float = info(0.070)  # interior half-extent y (= 1.0 * pitch)
    wall_t: float = info(0.010)
    floor_t: float = info(0.008)
    lid_z0: float = info(0.035)  # lid underside height (floor_t + tile_h + 5 mm headroom)
    lid_t: float = info(0.008)
    slot_w: float = info(0.024)  # lid slot width (24 mm << 60 mm plate: captive)
    # --- info: tile structure ----------------------------------------------------------------
    tile_w: float = info(0.060)
    tile_h: float = info(0.022)
    knob_r: float = info(0.008)
    knob_len: float = info(0.058)  # tile top -> 45 mm proud of the lid top (grasp length)
    tile_mass: float = info(0.15)
    n_tiles: int = info(5)  # 1 red + 4 white in 6 cells (exactly one empty)
    # --- info: marker ------------------------------------------------------------------------
    marker_size: tuple = info((0.025, 0.025, 0.100))
    marker_off: float = info(0.030)  # diagonal gap between wall outer face and the post
    # --- info: colors ------------------------------------------------------------------------
    body_color: tuple = info((0.45, 0.47, 0.52))
    lid_color: tuple = info((0.30, 0.32, 0.37))
    red_color: tuple = info((0.85, 0.10, 0.10))
    white_color: tuple = info((0.92, 0.92, 0.92))
    marker_color: tuple = info((0.10, 0.80, 0.20))
    contact_offset: float = info(0.001)  # explicit: default ~2 cm would jam the 10 mm clearances

    # Derived (filled in __post_init__).
    rest_z: float = field(default=None, init=False)  # tile origin height at rest
    lid_z1: float = field(default=None, init=False)  # lid top

    def __post_init__(self) -> None:
        self.rest_z = self.floor_t + self.tile_h / 2
        self.lid_z1 = self.lid_z0 + self.lid_t

        assert self.tile_w + 0.008 <= self.pitch, (
            "tiles must slide cell-to-cell with real clearance")
        assert self.tile_w >= self.slot_w + 0.020, (
            "plate must be captive under the lid slots (no lift-out loophole)")
        assert 2 * self.knob_r + 0.006 <= self.slot_w, (
            "knob must ride the slots with real clearance")
        assert self.lid_z0 >= self.floor_t + self.tile_h + 0.003, (
            "tile needs headroom under the lid")
        assert self.lid_z0 - (self.floor_t + self.tile_h) <= 0.008, (
            "headroom must stay too thin for a tile to climb or wedge over another")
        # Knob top above the frame base = floor top + tile height + knob length.
        assert self.floor_t + self.tile_h + self.knob_len >= self.lid_z1 + 0.035, (
            "knob must stand at least 35 mm proud of the lid top (graspable)")
        assert self.goal_tol <= self.pitch / 2 - 0.012, (
            "goal tolerance must cleanly separate adjacent cells")
        assert abs(self.inner_hx - 1.5 * self.pitch) < 1e-6 and abs(self.inner_hy - self.pitch) < 1e-6, (
            "interior must be exactly the 3 x 2 cell lattice")
        assert self.band_z_lo < self.rest_z < self.band_z_hi, "rest pose inside the captive band"
        assert self.band_z_hi < self.lid_z1 + self.tile_h / 2, (
            "captive band must reject a tile perched on the lid")
        assert self.ncols == 3 and self.nrows == 2 and self.n_tiles == 5, (
            "rubric latches assume the 2x3, one-empty-cell family")


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("tile_shunt")
class TileShuntScene(BaseScene):
    cfg: TileShuntSceneCfg

    def __init__(self, cfg: TileShuntSceneCfg | None = None) -> None:
        super().__init__(cfg or TileShuntSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        frame_kw = dict(
            pitch=c.pitch, inner_hx=c.inner_hx, inner_hy=c.inner_hy, wall_t=c.wall_t,
            floor_t=c.floor_t, lid_z0=c.lid_z0, lid_t=c.lid_t, slot_w=c.slot_w,
            body_color=c.body_color, lid_color=c.lid_color, contact_offset=c.contact_offset,
        )
        tile_kw = dict(
            tile_w=c.tile_w, tile_h=c.tile_h, knob_r=c.knob_r, knob_len=c.knob_len,
            mass=c.tile_mass, contact_offset=c.contact_offset,
        )
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "frame": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Frame",
                spawn=_frame_spawner_cfg(**frame_kw),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.frame_pos[0], c.frame_pos[1], 0.0)),
            ),
            "marker": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Marker",
                spawn=sim_utils.CuboidCfg(
                    size=c.marker_size,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.marker_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.frame_pos[0] + c.inner_hx + c.wall_t + c.marker_off,
                         c.frame_pos[1] + c.inner_hy + c.wall_t + c.marker_off,
                         c.marker_size[2] / 2 + 0.001)),
            ),
        }
        # tiles at nominal cells 0..4 (reset re-places everything); tile 0 is RED
        for t in range(c.n_tiles):
            col, row = t % c.ncols, t // c.ncols
            color = c.red_color if t == 0 else c.white_color
            out[f"tile_{t}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tile_" + str(t),
                spawn=_tile_spawner_cfg(color=color, **tile_kw),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.frame_pos[0] + (col - 1) * c.pitch,
                         c.frame_pos[1] + (row - 0.5) * c.pitch,
                         c.rest_z + 0.001)),
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
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.frame: RigidObject = env.iscene["frame"]
        self.marker: RigidObject = env.iscene["marker"]
        self.tiles: list[RigidObject] = [env.iscene[f"tile_{t}"] for t in range(c.n_tiles)]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # cell centres, frame frame: cell k at ((k%3-1)p, (k//3-0.5)p)
        k = torch.arange(c.ncols * c.nrows, device=dev)
        self.cell_xy = torch.stack([(k % c.ncols - 1).float() * c.pitch,
                                    (k // c.ncols - 0.5).float() * c.pitch], dim=-1)
        self.corner_cells = torch.tensor([0, 2, 3, 5], device=dev)
        # per-episode state
        self.tile_cell0 = torch.zeros(n, c.n_tiles, dtype=torch.long, device=dev)
        self.goal_cell = torch.zeros(n, dtype=torch.long, device=dev)
        self.d0 = torch.ones(n, device=dev)
        # progress latches
        self.moved_latch = torch.zeros(n, device=dev)
        self.prog_latch = torch.zeros(n, device=dev)
        self.atgoal_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: frame with xy jitter + free yaw, tile arrangement sampled
        (5 tiles into 6 cells — exactly one empty), goal corner sampled so the RED
        tile never starts there, green post written diagonally outside the goal
        corner, latches zeroed and the initial red->goal distance captured."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- frame: xy jitter + free yaw ---
        fxy = torch.tensor(c.frame_pos, device=dev).expand(m, 2).clone()
        fxy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.frame_jitter
        fyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.frame_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = fxy
        st[:, 3] = torch.cos(fyaw / 2)
        st[:, 6] = torch.sin(fyaw / 2)
        st[:, 0:3] += origin
        self.frame.write_root_state_to_sim(st, env_ids)

        # --- arrangement: a permutation of the 6 cells; first 5 -> tiles (0 = RED) ---
        if c.shuffle:
            perm = torch.rand(m, 6, device=dev).argsort(dim=1)
        else:
            perm = torch.arange(6, device=dev).expand(m, 6).clone()
        cells = perm[:, : c.n_tiles]
        red = cells[:, 0]

        # --- goal corner: uniform over the 4 corners, never the red start cell ---
        g = self.corner_cells[torch.randint(0, 4, (m,), device=dev)]
        for _ in range(24):
            bad = g == red
            if not bad.any():
                break
            g[bad] = self.corner_cells[torch.randint(0, 4, (int(bad.sum()),), device=dev)]
        self.tile_cell0[env_ids] = cells
        self.goal_cell[env_ids] = g
        cc = lambda k: torch.stack([k % c.ncols, k // c.ncols], dim=-1).float()  # noqa: E731
        self.d0[env_ids] = (cc(red) - cc(g)).abs().sum(dim=-1).clamp(min=1.0)

        # --- tiles at their cells (frame frame -> world) ---
        ca, sa = torch.cos(fyaw), torch.sin(fyaw)
        for t in range(c.n_tiles):
            loc = self.cell_xy[cells[:, t]]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = fxy[:, 0] + ca * loc[:, 0] - sa * loc[:, 1]
            st[:, 1] = fxy[:, 1] + sa * loc[:, 0] + ca * loc[:, 1]
            st[:, 2] = c.rest_z + 0.001
            st[:, 3] = torch.cos(fyaw / 2)
            st[:, 6] = torch.sin(fyaw / 2)
            st[:, 0:3] += origin
            self.tiles[t].write_root_state_to_sim(st, env_ids)

        # --- marker: diagonally outside the goal corner ---
        sx = torch.where(g % c.ncols == 0, -1.0, 1.0)
        sy = torch.where(g // c.ncols == 0, -1.0, 1.0)
        mx = sx * (c.inner_hx + c.wall_t + c.marker_off)
        my = sy * (c.inner_hy + c.wall_t + c.marker_off)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = fxy[:, 0] + ca * mx - sa * my
        st[:, 1] = fxy[:, 1] + sa * mx + ca * my
        st[:, 2] = c.marker_size[2] / 2 + 0.001
        st[:, 3] = torch.cos(fyaw / 2)
        st[:, 6] = torch.sin(fyaw / 2)
        st[:, 0:3] += origin
        self.marker.write_root_state_to_sim(st, env_ids)

        # --- latches ---
        self.moved_latch[env_ids] = 0.0
        self.prog_latch[env_ids] = 0.0
        self.atgoal_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "frame": self.frame.data.root_state_w[env_ids].clone(),
            "marker": self.marker.data.root_state_w[env_ids].clone(),
            "tiles": [b.data.root_state_w[env_ids].clone() for b in self.tiles],
            "tile_cell0": self.tile_cell0[env_ids].clone(),
            "goal_cell": self.goal_cell[env_ids].clone(),
            "d0": self.d0[env_ids].clone(),
            "moved_latch": self.moved_latch[env_ids].clone(),
            "prog_latch": self.prog_latch[env_ids].clone(),
            "atgoal_latch": self.atgoal_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.frame.write_root_state_to_sim(state["frame"], env_ids)
        self.marker.write_root_state_to_sim(state["marker"], env_ids)
        for b, s in zip(self.tiles, state["tiles"]):
            b.write_root_state_to_sim(s, env_ids)
        self.tile_cell0[env_ids] = state["tile_cell0"]
        self.goal_cell[env_ids] = state["goal_cell"]
        self.d0[env_ids] = state["d0"]
        self.moved_latch[env_ids] = state["moved_latch"]
        self.prog_latch[env_ids] = state["prog_latch"]
        self.atgoal_latch[env_ids] = state["atgoal_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A shallow gray puzzle frame ({2 * (c.inner_hx + c.wall_t) * 1000:.0f} x "
            f"{2 * (c.inner_hy + c.wall_t) * 1000:.0f} x {c.lid_z1 * 1000:.0f} mm) sits on "
            f"the floor. Its darker fixed lid is cut by a LADDER of "
            f"{c.slot_w * 1000:.0f} mm slots along three column lines and two row lines, "
            f"outlining a 2 x 3 lattice of six cells ({c.pitch * 1000:.0f} mm apart). Under "
            f"the lid, five square tiles ({c.tile_w * 1000:.0f} mm plates) fill five of the "
            f"six cells; each tile's {2 * c.knob_r * 1000:.0f} mm round knob sticks up "
            f"through the slots, standing about 45 mm proud of the lid — ONE knob and tile "
            f"are RED, four are WHITE. Exactly one cell has no knob over it: that cell is "
            f"EMPTY. The plates are wider than the slots, so NO tile can be lifted out — "
            f"pulling a knob up just presses its plate against the lid. A tile can only "
            f"SLIDE along the slots, and only INTO the empty cell: a line of tiles pushed "
            f"toward a wall jams. A green post stands diagonally outside ONE corner of the "
            f"frame: the corner CELL nearest the green post is the goal cell (the frame's "
            f"position and heading vary — read the lattice and the post from the scene).\n"
            f"Goal: shunt tiles one cell at a time — each move slides one tile into the "
            f"current empty cell — until the RED tile rests centered in the goal corner "
            f"cell (within {c.goal_tol * 1000:.0f} mm of that cell's centre). You will "
            f"usually have to move WHITE tiles first to walk the empty cell around the "
            f"frame onto the red tile's path; white tiles may end up in any cells. Judged "
            f"only when every tile has settled: the red tile centered in the goal corner "
            f"cell, all five tiles flat in their cells under the lid. A red tile in any "
            f"other cell, resting between cells, or balanced on top of the lid counts for "
            f"nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the knobbed tiles along the lid slots, one tile at a time into the "
            "empty cell, until the red tile sits centered in the corner cell nearest "
            "the green post. Tiles cannot be lifted out of the frame."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _frame_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> frame body frame (origin = frame centre, ground level)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.frame.data.root_quat_w,
                                  p_w - self.frame.data.root_pos_w)

    def tiles_local(self) -> torch.Tensor:
        """(N, T, 3): every tile origin in the frame frame."""
        pos = torch.stack([b.data.root_pos_w for b in self.tiles], dim=1)
        n, t = pos.shape[0], pos.shape[1]
        from isaaclab.utils.math import quat_apply_inverse

        fq = self.frame.data.root_quat_w[:, None, :].expand(n, t, 4).reshape(n * t, 4)
        fp = self.frame.data.root_pos_w[:, None, :]
        return quat_apply_inverse(fq, (pos - fp).reshape(n * t, 3)).reshape(n, t, 3)

    def _yaw(self, body) -> torch.Tensor:
        """(N,) yaw of a (near-)pure-yaw body."""
        q = body.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    # ----- predicates -------------------------------------------------------------------------
    def in_band(self) -> torch.Tensor:
        """(N, T) bool: tile captive inside the frame — origin within the interior
        footprint and inside the under-lid height band (a tile on the lid, on the
        floor outside, or held in the air all fail)."""
        c = self.cfg
        loc = self.tiles_local()
        return ((loc[:, :, 0].abs() < c.inner_hx) & (loc[:, :, 1].abs() < c.inner_hy)
                & (loc[:, :, 2] > c.band_z_lo) & (loc[:, :, 2] < c.band_z_hi))

    def red_goal_dist(self) -> torch.Tensor:
        """(N,) continuous cell-manhattan distance of the RED tile to the goal cell
        centre, in cell units (frame frame)."""
        loc = self.tiles_local()[:, 0, :2]
        g = self.cell_xy[self.goal_cell]
        return (loc - g).abs().sum(dim=-1) / self.cfg.pitch

    def red_at_goal(self) -> torch.Tensor:
        """(N,) bool: red plate centre within `goal_tol` of the goal cell centre,
        captive under the lid."""
        loc = self.tiles_local()[:, 0, :]
        g = self.cell_xy[self.goal_cell]
        near = (loc[:, :2] - g).norm(dim=-1) < self.cfg.goal_tol
        return near & self.in_band()[:, 0]

    def settled(self) -> torch.Tensor:
        """(N,) bool: every tile |lin vel| below `settle_lin`."""
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.tiles], dim=1)
        return (vel < self.cfg.settle_lin).all(dim=1)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch make-way, red progress and red-at-goal each physics substep, so
        transient progress keeps its credit. All credit is gated on the tile being
        captive inside the frame — a tile removed from the frame earns nothing."""
        c = self.cfg
        band = self.in_band()
        loc = self.tiles_local()
        # red progress toward the goal (fraction of the initial distance closed)
        d = self.red_goal_dist()
        prog = ((self.d0 - d) / self.d0).clamp(0.0, 1.0) * band[:, 0].float()
        self.prog_latch = torch.maximum(self.prog_latch, prog)
        self.atgoal_latch = torch.maximum(self.atgoal_latch, self.red_at_goal().float())
        # make-way: any WHITE tile settled into a DIFFERENT cell than it started
        own = self.cell_xy[self.tile_cell0[:, 1:]]
        d_own = (loc[:, 1:, :2] - own).norm(dim=-1)
        d_near = (loc[:, 1:, None, :2]
                  - self.cell_xy[None, None, :, :]).norm(dim=-1).min(dim=-1).values
        moved = (d_own > 0.055) & (d_near < 0.020) & band[:, 1:]
        self.moved_latch = torch.maximum(self.moved_latch, moved.any(dim=1).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: red tile centered in the goal corner cell + all five tiles
        captive in the frame, everything settled."""
        return self.red_at_goal() & self.in_band().all(dim=1) & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 * latched make-way + 0.55 * latched red
        progress + 0.15 * latched red-at-goal, capped at 0.85; exactly 1.0 iff
        success(). Doing nothing scores ~0; a red tile taken OUT of the frame (the
        seed's extraction strategy) earns nothing."""
        base = (0.15 * self.moved_latch + 0.55 * self.prog_latch
                + 0.15 * self.atgoal_latch).clamp(0.0, 0.85)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="tile_shunt", robot="null"))
