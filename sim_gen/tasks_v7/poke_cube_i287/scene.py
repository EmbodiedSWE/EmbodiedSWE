"""ShuntTrayScene — a CAPTIVE 2x2 sliding-tile puzzle: shunt the RED tile onto the
blue-marked cell of a walled tray whose tiles cannot be lifted out. Derived from
maniskill/poke_cube, but no single push can score: the marked cell always starts
COVERED by a gray tile, and the tray's slotted cover makes sliding the only
legal motion — the solver must plan an ordered SEQUENCE of shunts through the one
empty cell.

Seed (maniskill/poke_cube): grasp a peg tool and poke a red cube a few centimeters
across an open plane into a painted goal region — one object, one planar stroke,
judged by the poked object's own xy distance to the goal. Here:

- The work surface becomes a TRAY: a 2x2 grid of cells (70 mm pitch) enclosed by
  walls and spanned by a flat COVER at wall height. The tiles ride UNDER the
  cover; only each tile's knob pokes up through a square-ring SLOT cut in the
  cover. Geometry makes the tiles CAPTIVE (asserted in `__post_init__`): the
  cover's central island and outer rim overhang every reachable tile footprint
  (5 mm headroom, slot far narrower than a tile) — so no lift, drag-out, or
  tip-out exists, and any diagonal shortcut across the middle jams the knob on
  the island. The only motion the tray admits is an in-plane shunt of a tile
  into the currently-empty cell.
- The goal cell (blue-painted floor) ALWAYS starts covered by a gray tile, and two
  56 mm tiles cannot share a 70 mm cell — so the seed's entire plan (one straight
  push of the red object at the goal) jams against the occupying gray and is
  REJECTED (smoke constructs exactly that end state). Execution order is REQUIRED
  and geometrically forced: the occupying gray must vacate the blue cell (through
  the empty cell, possibly after other tiles rotate the vacancy around the ring)
  before the red tile can enter.
- A solver therefore needs a different PLAN (combinatorial: route the vacancy so
  the occupier leaves the goal cell, then bring red in — a 2-6 move breadth-first
  shunt sequence that depends on the sampled permutation) and different CODE
  STRUCTURE (per-move slide control with occupancy readback and replanning,
  instead of one position servo on one object).

Assets are fully procedural:
  - tray: ONE kinematic compound rigid body — floor plate, 4 walls (41 mm), and a
    COVER at z 41-47 mm (4 rim slabs + a central island) pierced by a square-ring
    knob slot — repositioned per episode (xy jitter + free yaw).
  - marker: a thin kinematic blue plate sunk into the tray floor at the sampled
    goal cell, its top 0.3 mm proud (a visible blue ring frames whichever tile
    covers it). VISUAL-ONLY — no collider, so slides never park on its edge.
  - tiles: DYNAMIC 56x56x36 mm bodies with a 16x16x24 mm push knob on top
    (the knob rises through the cover slot: a parallel-jaw fingertip pushes or
    pinches it to shunt the tile; lifting is blocked by the cover). Authored
    mass, diagonal inertia, friction material, damping, solver iterations.

Per-episode randomization (readback-verifiable): the (goal, red, empty) cell
permutation — uniform over all 24 assignments with the goal cell always covered
by a gray tile — plus tray xy jitter and free tray yaw.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.25 * vacated_ever — the occupying gray ever left the goal cell (latched);
                        the geometrically-forced FIRST stage of any solution
  0.15 * red_moved    — the red tile ever left its start cell (latched)
  0.30 * approach     — running max of 1 - d/d0 (red-to-goal-cell distance,
                        d0 = the start-cell distance; exactly 0 at reset)
  1.0 iff success()   — red tile settled ON the blue cell: center within
                        `success_tol` of the cell center, at floor height,
                        upright, all tiles still. Non-success cap 0.70.

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

TILE_NAMES = ("red_tile", "gray_a", "gray_b")

# cell index -> (sx, sy) sign of the cell center (+-pitch/2); orthogonal adjacency
CELL_SIGNS = ((-1, -1), (1, -1), (-1, 1), (1, 1))
CELL_ADJ = {0: (1, 2), 1: (0, 3), 2: (0, 3), 3: (1, 2)}

_SPAWNER_CACHE: dict[str, Any] = {}


# ----- custom spawners -------------------------------------------------------------------------
def _root_xform(prim_path: str, translation, orientation):
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    return stage, xform.GetPrim()


def _material(stage, path: str, static: float, dynamic: float):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, center, size, color, mat=None, contact_offset=0.0015):
    """One child box; collider + material iff `mat` is given (else visual-only)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if mat is not None:
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(box.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    return box


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the tray: ONE kinematic compound rigid body. Local frame origin at the
    center of the floor plate's TOP surface. Children: floor plate, 4 walls, and a
    COVER at wall height — 4 rim slabs + a central island, leaving a square-ring
    slot for the tile knobs — all colliders with an explicit friction material."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(8.0)
    mat = _material(stage, f"{prim_path}/physmat", cfg.friction, cfg.friction * 0.95)

    inner = float(cfg.inner_half)          # interior half-width (wall inner face)
    wt = float(cfg.wall_t)
    wh = float(cfg.wall_h)                 # wall top = cover underside
    lt = float(cfg.lip_t)                  # cover thickness
    so = float(cfg.slot_out)               # slot outer edge = rim inner edge
    si = float(cfg.slot_in)                # slot inner edge = island half-width
    span = 2 * inner + 2 * wt              # wall length (closes the corners)
    co = float(cfg.contact_offset)

    _box(stage, f"{prim_path}/floor", (0, 0, -cfg.plate_t / 2),
         (cfg.plate_s, cfg.plate_s, cfg.plate_t), cfg.floor_color, mat, co)
    for i, (nx, ny) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
        d = inner + wt / 2
        _box(stage, f"{prim_path}/wall_{i}",
             (nx * d, ny * d, wh / 2),
             (wt if nx else span, wt if ny else span, wh),
             cfg.wall_color, mat, co)
        # cover rim slab: from the wall outer face inward to slot_out, z [wh, wh+lt]
        rw = inner + wt - so               # slab width (overhang + wall)
        rd = so + rw / 2
        _box(stage, f"{prim_path}/rim_{i}",
             (nx * rd, ny * rd, wh + lt / 2),
             (rw if nx else span, rw if ny else span, lt),
             cfg.lip_color, mat, co)
    # cover island: central slab between the slot bands. With the rim it overhangs
    # every reachable tile footprint (tiles captive) and it blocks any diagonal
    # knob path across the tray center (ordering forced).
    _box(stage, f"{prim_path}/island", (0, 0, wh + lt / 2),
         (2 * si, 2 * si, lt), cfg.lip_color, mat, co)
    return root


def _spawn_tile(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one tile: DYNAMIC rigid body — 56 mm body box + push knob, authored
    mass + diagonal inertia + CoM, friction material, damping, solver iterations."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    s, h = float(cfg.body_s), float(cfg.body_h)
    ixx = float(cfg.mass) / 12.0 * (s * s + h * h)
    izz = float(cfg.mass) / 12.0 * (2 * s * s)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(ixx, ixx, izz))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.15)
    pxrb.CreateAngularDampingAttr(0.40)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)

    mat = _material(stage, f"{prim_path}/physmat", cfg.friction, cfg.friction * 0.95)
    co = float(cfg.contact_offset)
    # body box centered at the root origin; knob on top (also a collider: it is the
    # push/pinch handle, real contacts happen on it)
    _box(stage, f"{prim_path}/body", (0, 0, 0), (s, s, h), cfg.color, mat, co)
    _box(stage, f"{prim_path}/knob",
         (0, 0, h / 2 + float(cfg.knob_h) / 2),
         (cfg.knob_s, cfg.knob_s, cfg.knob_h), cfg.color, mat, co)
    return root


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tray" not in _SPAWNER_CACHE:

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            inner_half: float = 0.070
            wall_t: float = 0.012
            wall_h: float = 0.041
            lip_t: float = 0.006
            slot_out: float = 0.053
            slot_in: float = 0.018
            plate_s: float = 0.190
            plate_t: float = 0.012
            friction: float = 0.35
            contact_offset: float = 0.0015
            floor_color: tuple = (0.78, 0.74, 0.66)
            wall_color: tuple = (0.35, 0.27, 0.20)
            lip_color: tuple = (0.27, 0.21, 0.15)

        @configclass
        class TileSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tile)
            body_s: float = 0.056
            body_h: float = 0.036
            knob_s: float = 0.016
            knob_h: float = 0.024
            mass: float = 0.20
            friction: float = 0.35
            contact_offset: float = 0.0015
            color: tuple = (0.6, 0.6, 0.6)

        _SPAWNER_CACHE["tray"] = TraySpawnerCfg
        _SPAWNER_CACHE["tile"] = TileSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ShuntTraySceneCfg(BaseCfg):
    """Config for `ShuntTrayScene`. The captivity interlock and the forced ordering
    are GEOMETRIC and asserted below: the cover's rim + island overhang every
    reachable tile footprint with 5 mm headroom (no lift-out or tip-out exists),
    the knob slot is far narrower than a tile, the island blocks diagonal knob
    paths, and two tiles cannot share a cell (the covered goal cell must be
    vacated before red can enter)."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    success_tol: float = tunable(0.025)  # red center within this (xy, tray frame) of the goal cell
    z_tol: float = tunable(0.008)  # red center height within this of the resting height
    upright_max_deg: float = tunable(15.0)  # red body +z within this of world-up
    settle_speed: float = tunable(0.03)  # max |lin vel| of every tile when judging (m/s)
    settle_omega: float = tunable(0.50)  # max |ang vel| of every tile when judging (rad/s)
    latch_travel: float = tunable(0.045)  # vacate / red-moved latch radius (m) — > success_tol,
    # < pitch, so only a real cell change fires it

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    tray_jitter: float = tunable(0.03)  # +/- xy jitter on the tray anchor
    tray_yaw_deg: float = tunable(180.0)  # +/- free tray yaw
    anchor: tuple = tunable((0.40, 0.0))  # tray anchor (xy) — comfortable Franka reach from
    # a base at the origin (all knobs at r 0.26-0.55 m, tops ~85 mm above ground)

    # --- info: tray geometry (tray-local, origin at the floor plate top center) -----------------
    pitch: float = info(0.070)  # cell pitch; cell centers at (+-pitch/2, +-pitch/2)
    inner_half: float = info(0.070)  # interior half-width (2x2 cells exactly)
    wall_t: float = info(0.012)
    wall_h: float = info(0.041)  # wall top = cover underside height
    lip_t: float = info(0.006)  # cover thickness (cover spans z wall_h..wall_h+lip_t)
    slot_out: float = info(0.053)  # knob-slot outer edge (= cover rim inner edge)
    slot_in: float = info(0.018)  # knob-slot inner edge (= cover island half-width)
    plate_s: float = info(0.190)
    plate_t: float = info(0.012)
    marker_s: float = info(0.064)  # blue goal plate (proud by marker_proud)
    marker_t: float = info(0.008)
    marker_proud: float = info(0.0003)

    # --- info: tiles ----------------------------------------------------------------------------
    tile_s: float = info(0.056)  # body footprint (cell slack = pitch - tile_s = 14 mm)
    tile_h: float = info(0.036)  # body height (5 mm below the cover underside)
    knob_s: float = info(0.016)  # push/pinch knob (rises to 60 mm, 13 mm above the cover)
    knob_h: float = info(0.024)
    tile_mass: float = info(0.20)
    friction: float = info(0.35)
    contact_offset: float = info(0.0015)
    red_color: tuple = info((0.85, 0.08, 0.08))
    gray_color: tuple = info((0.55, 0.55, 0.58))
    marker_color: tuple = info((0.10, 0.30, 0.95))

    # --- info: rubric weights -------------------------------------------------------------------
    w_vacate: float = info(0.25)
    w_redmove: float = info(0.15)
    w_app: float = info(0.30)  # 0.25 + 0.15 + 0.30 = 0.70 = the non-success cap

    # Derived (filled in __post_init__).
    rest_z: float = field(default=0.0, init=False)  # tile center resting height (tray frame)

    def __post_init__(self) -> None:
        self.rest_z = self.tile_h / 2
        half_tile = self.tile_s / 2
        slack = self.pitch - self.tile_s          # in-cell slack (14 mm)
        max_center = self.pitch / 2 + slack / 2   # tile-center extreme, pressed to a wall
        min_center = self.pitch / 2 - slack / 2   # tile-center extreme, pressed inward
        # CAPTIVITY: tiles ride UNDER the cover with small headroom; the cover's rim
        # and island overhang every reachable footprint; the slot is far narrower
        # than a tile. No lift-out, tip-out, or squeeze-through exists.
        assert self.tile_h + 0.004 <= self.wall_h, "tile body too tall to ride under the cover"
        assert self.tile_s > (self.slot_out - self.slot_in) + 0.010, \
            "a tile could fit through the knob slot"
        assert min_center - half_tile < self.slot_in - 0.004, \
            "cover island does not overhang the tile footprint"
        assert min_center + half_tile > self.slot_out + 0.002, \
            "cover rim does not overhang the tile footprint"
        # knob travel: the knob stays inside the slot at both in-cell extremes
        assert max_center + self.knob_s / 2 < self.slot_out - 0.0015, \
            "knob can jam on the cover rim"
        assert min_center - self.knob_s / 2 > self.slot_in + 0.0015, \
            "knob can jam on the cover island"
        # knob rises clear of the cover top (graspable / pushable)
        assert self.tile_h + self.knob_h > self.wall_h + self.lip_t + 0.008, \
            "knob does not rise above the cover"
        # the island blocks any knob path across the tray center (ordering forced)
        assert self.slot_in > self.knob_s / 2 + 0.002, \
            "a diagonal knob shortcut fits past the island"
        # two tiles cannot both be within success_tol of one cell center
        assert 2 * self.success_tol < self.tile_s, "success_tol admits two tiles in one cell"
        # a tile fully in its cell fits the interior
        assert self.pitch / 2 + half_tile < self.inner_half + 1e-9, "tiles overlap the walls"

    def cell_center(self, i: int) -> tuple[float, float]:
        sx, sy = CELL_SIGNS[i]
        return (sx * self.pitch / 2, sy * self.pitch / 2)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("shunt_tray")
class ShuntTrayScene(BaseScene):
    cfg: ShuntTraySceneCfg

    def __init__(self, cfg: ShuntTraySceneCfg | None = None) -> None:
        super().__init__(cfg or ShuntTraySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        tray_spawn = sp["tray"](
            mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            inner_half=c.inner_half, wall_t=c.wall_t, wall_h=c.wall_h,
            lip_t=c.lip_t, slot_out=c.slot_out, slot_in=c.slot_in,
            plate_s=c.plate_s, plate_t=c.plate_t, friction=c.friction,
            contact_offset=c.contact_offset,
        )

        def tile_spawn(color):
            return sp["tile"](
                mass_props=sim_utils.MassPropertiesCfg(mass=c.tile_mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                body_s=c.tile_s, body_h=c.tile_h, knob_s=c.knob_s, knob_h=c.knob_h,
                mass=c.tile_mass, friction=c.friction,
                contact_offset=c.contact_offset, color=color,
            )

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=c.friction, dynamic_friction=c.friction * 0.95,
                    restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=tray_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.anchor[0], c.anchor[1], c.plate_t)),
            ),
            "marker": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Marker",
                spawn=sim_utils.CuboidCfg(
                    size=(c.marker_s, c.marker_s, c.marker_t),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                    # VISUAL-ONLY (no collision_props -> no collider is authored):
                    # even a 0.3 mm proud collider edge is a vertical wall to a
                    # quasi-static slide and parks the tile at the cell boundary.
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.marker_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.anchor[0], c.anchor[1] - c.pitch / 2,
                         c.plate_t + c.marker_proud - c.marker_t / 2)),
            ),
        }
        for name in TILE_NAMES:
            color = c.red_color if name == "red_tile" else c.gray_color
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tile_" + name,
                spawn=tile_spawn(color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.anchor[0], c.anchor[1], c.plate_t + c.rest_z + 0.002)),
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
                "enable_external_forces_every_iteration": True,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.tray: RigidObject = env.iscene["tray"]
        self.marker: RigidObject = env.iscene["marker"]
        self.tiles: dict[str, RigidObject] = {n: env.iscene[n] for n in TILE_NAMES}
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self._centers = torch.tensor([c.cell_center(i) for i in range(4)], device=dev)
        # episode assignment (readback-verifiable): goal cell, red start cell, empty cell
        self._t_cell = torch.zeros(n, dtype=torch.long, device=dev)
        self._r_cell = torch.zeros(n, dtype=torch.long, device=dev)
        self._e_cell = torch.zeros(n, dtype=torch.long, device=dev)
        self._d0 = torch.full((n,), c.pitch, device=dev)  # red start distance to goal
        # latches: partial progress survives transient achievements (rubric requirement)
        self._vacated_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._red_moved_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._app_max = torch.zeros(n, device=dev)

    # ----- frames --------------------------------------------------------------------------------
    def tray_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N, 3) world points -> tray-local frame (origin at floor-plate top center)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.tray.data.root_quat_w,
                                  pos_w - self.tray.data.root_pos_w)

    def tile_local_xy(self, name: str) -> torch.Tensor:
        """(N, 2) tile center in the tray frame (xy)."""
        return self.tray_local(self.tiles[name].data.root_pos_w)[:, :2]

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: tray re-posed (anchor + jitter, free yaw); the cell
        permutation sampled uniformly over all 24 (goal, red, empty) assignments —
        the goal cell is ALWAYS covered by gray_a; marker sunk into the goal cell;
        tiles dropped 2 mm above their cells, yaw-aligned with the tray; latches
        cleared. Discrete draws use rand+argsort after burning draws (the first
        post-seed draws are near-degenerate across seeds)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        for _ in range(3):  # burn near-degenerate first draws
            torch.rand(4, device=dev)

        # --- tray pose ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.tray_yaw_deg)
        half = yaw / 2
        cy, sy = torch.cos(half), torch.sin(half)
        cyf, syf = torch.cos(yaw), torch.sin(yaw)
        tray_xy = torch.tensor(c.anchor, device=dev).expand(m, 2).clone()
        tray_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.tray_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = tray_xy
        st[:, 2] = c.plate_t
        st[:, 3], st[:, 6] = cy, sy
        st[:, 0:3] += origin
        self.tray.write_root_state_to_sim(st, env_ids)

        def to_world(local_xy: torch.Tensor, z_local: torch.Tensor | float) -> torch.Tensor:
            """(m, 2) tray-local xy + z -> (m, 3) world (yaw rotation + tray pose)."""
            wx = tray_xy[:, 0] + cyf * local_xy[:, 0] - syf * local_xy[:, 1]
            wy = tray_xy[:, 1] + syf * local_xy[:, 0] + cyf * local_xy[:, 1]
            wz = torch.full((m,), 0.0, device=dev) + c.plate_t + z_local
            return torch.stack([wx, wy, wz], dim=1)

        # --- permutation: perm[:,0]=goal (gray_a), 1=red, 2=empty, 3=gray_b ---
        perm = torch.rand(m, 4, device=dev).argsort(dim=1)
        self._t_cell[env_ids] = perm[:, 0]
        self._r_cell[env_ids] = perm[:, 1]
        self._e_cell[env_ids] = perm[:, 2]
        self._d0[env_ids] = (self._centers[perm[:, 0]] - self._centers[perm[:, 1]]).norm(dim=-1)

        # --- marker: sunk into the goal cell, top `marker_proud` above the floor ---
        mk = torch.zeros(m, 13, device=dev)
        mk[:, 0:3] = to_world(self._centers[perm[:, 0]],
                              c.marker_proud - c.marker_t / 2)
        mk[:, 3], mk[:, 6] = cy, sy
        mk[:, 0:3] += origin
        self.marker.write_root_state_to_sim(mk, env_ids)

        # --- tiles: gray_a covers the goal, red at its cell, gray_b at the last cell ---
        for name, col in (("gray_a", 0), ("red_tile", 1), ("gray_b", 3)):
            ts = torch.zeros(m, 13, device=dev)
            ts[:, 0:3] = to_world(self._centers[perm[:, col]], c.rest_z + 0.002)
            ts[:, 3], ts[:, 6] = cy, sy
            ts[:, 0:3] += origin
            self.tiles[name].write_root_state_to_sim(ts, env_ids)

        # --- clear latches ---
        self._vacated_ever[env_ids] = False
        self._red_moved_ever[env_ids] = False
        self._app_max[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "marker": self.marker.data.root_state_w[env_ids].clone(),
            "tiles": {n: b.data.root_state_w[env_ids].clone() for n, b in self.tiles.items()},
            "t_cell": self._t_cell[env_ids].clone(),
            "r_cell": self._r_cell[env_ids].clone(),
            "e_cell": self._e_cell[env_ids].clone(),
            "d0": self._d0[env_ids].clone(),
            "vacated_ever": self._vacated_ever[env_ids].clone(),
            "red_moved_ever": self._red_moved_ever[env_ids].clone(),
            "app_max": self._app_max[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.marker.write_root_state_to_sim(state["marker"], env_ids)
        for n, b in self.tiles.items():
            b.write_root_state_to_sim(state["tiles"][n], env_ids)
        self._t_cell[env_ids] = state["t_cell"]
        self._r_cell[env_ids] = state["r_cell"]
        self._e_cell[env_ids] = state["e_cell"]
        self._d0[env_ids] = state["d0"]
        self._vacated_ever[env_ids] = state["vacated_ever"]
        self._red_moved_ever[env_ids] = state["red_moved_ever"]
        self._app_max[env_ids] = state["app_max"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A square wooden TRAY sits on the floor: a 2x2 grid of four cells "
            f"({c.pitch * 1000:.0f} mm pitch) enclosed by dark walls. A flat dark "
            f"COVER spans the tray at wall height: the tiles ride BENEATH it and "
            f"only each tile's small square knob pokes up through a square-ring "
            f"SLOT cut in the cover, so the tiles CANNOT be lifted or tipped out "
            f"of the tray — they can only SLIDE from cell to cell, and the cover's "
            f"central island blocks any diagonal shortcut across the middle. Three "
            f"square tiles ({c.tile_s * 1000:.0f} mm wide) occupy three of the "
            f"cells: one RED tile and two GRAY tiles. The fourth cell is EMPTY. "
            f"The floor of exactly one cell is painted BLUE (a blue plate sunk "
            f"into the floor — its rim shows around whatever tile covers it): that "
            f"is the GOAL cell, and it always starts covered by one of the GRAY "
            f"tiles.\n"
            f"Goal: the RED tile must end up resting ON the blue cell — centered "
            f"within {c.success_tol * 1000:.0f} mm of the blue cell's center, flat "
            f"on the tray floor, with every tile at rest.\n"
            f"A tile can only move by sliding into the empty cell (push or pinch "
            f"its knob above the cover and slide it — two tiles never fit one "
            f"cell, and diagonal slides jam the knob on the cover's island). "
            f"Because the blue cell starts covered, you MUST first shunt the "
            f"covering gray tile off the blue cell (into the empty cell, rotating "
            f"the vacancy around the grid as needed) before the red tile can be "
            f"slid in. Plan the shunt sequence; pushing the red tile straight at "
            f"the occupied blue cell only jams it against the gray tile and "
            f"scores nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the tiles in the tray to bring the RED tile onto the blue-marked "
            "cell. The blue cell starts covered by a gray tile — shunt that gray "
            "tile out of the way through the empty cell first. Tiles cannot be "
            "lifted out of the tray; only sliding moves work."
        )

    # ----- readings / rubric ---------------------------------------------------------------------
    def goal_center_local(self) -> torch.Tensor:
        """(N, 2) goal-cell center in the tray frame."""
        return self._centers[self._t_cell]

    def red_dist(self) -> torch.Tensor:
        """(N,) planar distance red center -> goal-cell center (tray frame)."""
        return (self.tile_local_xy("red_tile") - self.goal_center_local()).norm(dim=-1)

    def red_in_goal(self) -> torch.Tensor:
        """(N,) bool: red center within `success_tol` of the goal-cell center."""
        return self.red_dist() <= self.cfg.success_tol

    def red_on_floor(self) -> torch.Tensor:
        """(N,) bool: red center at resting height in the tray frame (rejects a tile
        stacked on another tile, pressed against the cover, or airborne)."""
        z = self.tray_local(self.tiles["red_tile"].data.root_pos_w)[:, 2]
        return (z - self.cfg.rest_z).abs() < self.cfg.z_tol

    def red_upright(self) -> torch.Tensor:
        """(N,) bool: red body +z within `upright_max_deg` of world-up."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        up = quat_apply(self.tiles["red_tile"].data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.upright_max_deg))

    def settled(self) -> torch.Tensor:
        """(N,) bool: EVERY tile below the settle gates."""
        c = self.cfg
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for b in self.tiles.values():
            ok &= (b.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
            ok &= (b.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega)
        return ok

    def occupancy(self) -> torch.Tensor:
        """(N, 4) long: cell -> tile index (-1 empty; 0 red, 1 gray_a, 2 gray_b),
        nearest-center assignment within pitch/2."""
        n = self.env.num_envs
        occ = torch.full((n, 4), -1, dtype=torch.long, device=self.env.device)
        for ti, name in enumerate(TILE_NAMES):
            xy = self.tile_local_xy(name)
            d = (xy.unsqueeze(1) - self._centers.unsqueeze(0)).norm(dim=-1)  # (N,4)
            best = d.argmin(dim=1)
            close = d.gather(1, best.unsqueeze(1)).squeeze(1) < self.cfg.pitch / 2
            for e in range(n):
                if bool(close[e]):
                    occ[e, best[e]] = ti
        return occ

    def _update_latches(self) -> None:
        c = self.cfg
        ga = (self.tile_local_xy("gray_a") - self.goal_center_local()).norm(dim=-1)
        self._vacated_ever |= ga > c.latch_travel
        rd = (self.tile_local_xy("red_tile") - self._centers[self._r_cell]).norm(dim=-1)
        self._red_moved_ever |= rd > c.latch_travel
        app = ((self._d0 - self.red_dist()) / self._d0.clamp(min=1e-6)).clamp(0.0, 1.0)
        app = torch.nan_to_num(app, nan=0.0, posinf=0.0, neginf=0.0)
        self._app_max = torch.maximum(self._app_max, app)

    # ----- step-coupled bookkeeping (every substep) ----------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No mechanism plant (the tray is kinematic, tiles free) — just latch rubric
        progress every step so transient achievements keep credit."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: red tile resting ON the blue cell — center within tolerance,
        at floor height, upright, every tile settled. Live physical outcome only."""
        self._update_latches()
        return (self.red_in_goal() & self.red_on_floor() & self.red_upright()
                & self.settled())

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25*vacated + 0.15*red_moved + 0.30*approach — all
        latched, exactly 0 for the null policy, capped 0.70 — and 1.0 iff success()
        holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_vacate * self._vacated_ever.float()
                + c.w_redmove * self._red_moved_ever.float()
                + c.w_app * self._app_max).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; tiles are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="shunt_tray", robot="null"))
