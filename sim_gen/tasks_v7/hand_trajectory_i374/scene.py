"""TileShuffleScene — a sliding-tile keystone shuffle: route the GOLD tile through a walled
3x2 lattice, under a slotted roof, into the green-marked target cell — by shuffling the OTHER
tiles out of the way (seed: pick_place/hand_trajectory, strategically inverted).

The seed task grasps a box and flies the HAND through free space along five memorized floating
waypoint markers, in order. Here every element of that strategy is inverted:

- Nothing is ever carried, and nothing CAN be: the playfield is a walled 3x2 cell lattice
  covered by a slotted ROOF 5 mm above the tile tops. Each tile carries a knob that pokes up
  through the roof's slot network; the tile body cannot pass the 30 mm slots, so every tile is
  topologically captive. The only actuation surface is the protruding knob (or the tile edge
  through the slot mouth) — planar pushes only.
- There is no memorized route: 5 tiles fill 6 cells, so exactly ONE cell is vacant, and a tile
  can only move into the CURRENT vacancy. The move sequence is not a fixed list — it depends on
  the per-episode random layout (gold cell, target cell, vacancy cell, frame pose), and most of
  the work is moving NON-goal tiles to route the vacancy around the gold tile. The solver must
  read the configuration and plan combinatorially, not track waypoints.
- The roof's slot network is also the MOVE LAW: slots run only along the two row lines and the
  three column lines, so a knobbed tile physically cannot move diagonally or free-roam — the
  Manhattan lattice moves of a sliding puzzle are enforced by geometry, not by the rubric.
- The goal is a settled containment outcome (gold tile resting centered in the marked cell),
  not a hover near a virtual marker.

Geometry honesty is asserted in `__post_init__`: the tile cannot pass the slot (54 mm
interference), the knob rides the slot with clearance, the roof headroom traps the tile, the
knob protrudes far enough above the roof to be pushed, resting tiles pass the progress z gate
while any above-roof pose fails it, and the target is always >= 2 lattice moves from the gold
tile's start (a null policy scores 0).

Rubric: score() = 0.08 once any tile has been displaced a full cell pitch (moved latch)
+ 0.52 * latched best progress of the gold tile toward the target measured in lattice distance
(d0 - d_min)/d0; exactly 1.0 iff success() = gold tile centered in the target cell (z-gated)
with every tile settled. Latched credit never evaporates; max non-success score is 0.60.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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

TILE_NAMES = ("gold", "grey0", "grey1", "grey2", "grey3")


def _mdist(i: int, j: int) -> int:
    """Lattice (Manhattan) distance between cell indices; cell i = (col=i//2, row=i%2)."""
    return abs(i // 2 - j // 2) + abs(i % 2 - j % 2)


# ----- custom compound spawner (tile = slab + knob) ---------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


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


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _phys_material(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _bind_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_tile(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one DYNAMIC tile: origin at the slab's bottom centre; a square slab plus a
    slender knob cylinder on top (the only part that protrudes above the slotted roof)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(cfg.mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, float(cfg.com_z)))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in cfg.inertia]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    px.CreateLinearDampingAttr(0.6)
    px.CreateAngularDampingAttr(2.0)

    body_mat = _phys_material(stage, f"{prim_path}/bodymat", cfg.mu_static, cfg.mu_dynamic)
    knob_mat = _phys_material(stage, f"{prim_path}/knobmat", cfg.knob_mu, cfg.knob_mu)

    slab = UsdGeom.Cube.Define(stage, f"{prim_path}/slab")
    slab.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(slab.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(cfg.tile_h) / 2))
    xf.AddScaleOp().Set(Gf.Vec3f(float(cfg.tile_s), float(cfg.tile_s), float(cfg.tile_h)))
    slab.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    _collide(slab.GetPrim(), cfg.contact_offset)
    _bind_material(slab.GetPrim(), body_mat)

    knob = UsdGeom.Cylinder.Define(stage, f"{prim_path}/knob")
    knob.CreateRadiusAttr(float(cfg.knob_r))
    knob.CreateHeightAttr(float(cfg.knob_h))
    knob.CreateAxisAttr("Z")
    kz = float(cfg.tile_h) + float(cfg.knob_h) / 2
    UsdGeom.Xformable(knob.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, kz))
    knob.CreateExtentAttr([Gf.Vec3f(-float(cfg.knob_r), -float(cfg.knob_r), -float(cfg.knob_h) / 2),
                           Gf.Vec3f(float(cfg.knob_r), float(cfg.knob_r), float(cfg.knob_h) / 2)])
    knob.CreateDisplayColorAttr([Gf.Vec3f(*cfg.knob_color)])
    _collide(knob.GetPrim(), cfg.contact_offset)
    _bind_material(knob.GetPrim(), knob_mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound tile spawner configclass (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tile" not in _SPAWNER_CACHE:

        @configclass
        class TileSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tile)
            tile_s: float = 0.084
            tile_h: float = 0.040
            knob_r: float = 0.010
            knob_h: float = 0.040
            mass: float = 0.15
            com_z: float = 0.021
            inertia: tuple = (1.1e-4, 1.1e-4, 1.8e-4)
            mu_static: float = 0.30
            mu_dynamic: float = 0.25
            knob_mu: float = 0.10
            color: tuple = (0.55, 0.55, 0.58)
            knob_color: tuple = (0.15, 0.15, 0.17)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["tile"] = TileSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TileShuffleSceneCfg(BaseCfg):
    """Config for `TileShuffleScene`. Geometry lives in the FRAME-LOCAL frame: origin at the
    lattice centre, z = 0 at the floor top. Cells are a 3x2 lattice with pitch `pitch`;
    cell index i = col * 2 + row (col 0..2 along +x, row 0..1 along +y)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    cell_tol: float = tunable(0.022)  # in-cell xy tolerance around a cell centre (m)
    z_gate: float = tunable(0.035)  # progress z gate: tile ROOT (slab bottom) below this
    settle_speed: float = tunable(0.08)  # max tile |lin vel| when judging success (m/s)
    moved_thresh: float = tunable(0.055)  # xy displacement from start cell = "moved" latch (m)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    frame_pos: tuple = tunable((0.45, 0.0))  # nominal lattice centre on the ground (m)
    frame_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the frame per episode
    frame_yaw_deg: float = tunable(12.0)  # uniform +/- yaw of the frame per episode
    spawn_jitter: float = tunable(0.0015)  # tile spawn xy jitter inside its cell (m)
    min_dist: int = tunable(2)  # min lattice distance gold-start -> target

    # --- info: structure (frame-local constants; pieces derived in __post_init__) ------------
    pitch: float = info(0.090)  # cell pitch (m); interior = 3p x 2p
    tile_s: float = info(0.084)  # tile slab side
    tile_h: float = info(0.040)  # tile slab height
    knob_r: float = info(0.010)
    knob_h: float = info(0.040)  # knob spans z tile_h .. tile_h + knob_h
    tile_mass: float = info(0.15)
    wall_t: float = info(0.020)
    wall_h: float = info(0.045)  # wall top = roof underside
    roof_t: float = info(0.008)
    slot_hw: float = info(0.015)  # slot half-width (roof openings along cell-centre lines)
    floor_t: float = info(0.020)
    pad_off: float = info(0.030)  # target marker pads: corner offsets from the cell centre
    pad_size: tuple = info((0.016, 0.016, 0.004))
    contact_offset: float = info(0.0015)
    gold_color: tuple = info((0.95, 0.78, 0.12))
    grey_color: tuple = info((0.55, 0.55, 0.58))
    frame_color: tuple = info((0.35, 0.38, 0.44))
    roof_color: tuple = info((0.60, 0.68, 0.78))
    pad_color: tuple = info((0.10, 0.80, 0.20))

    # Derived (filled in __post_init__).
    pieces: tuple = field(default=None, init=False)  # ((name, size, local centre, rgb), ...)
    cells: tuple = field(default=None, init=False)  # 6 cell centres (x, y), index = col*2+row
    z_off: float = field(default=None, init=False)  # local z=0 above the ground plane

    def __post_init__(self) -> None:
        p, wt, wh, rt, sw = self.pitch, self.wall_t, self.wall_h, self.roof_t, self.slot_hw
        ihx, ihy = 1.5 * p, 1.0 * p  # interior half extents (walls' inner faces)
        ohx, ohy = ihx + wt, ihy + wt  # outer extents (roof cover)
        rz = wh + rt / 2  # roof piece centre z
        slate, roofc = self.frame_color, self.roof_color
        # roof mid band (y in [-p/2+sw .. p/2-sw]) is split into 4 pieces by the 3 column slots
        my = (p / 2 - sw)  # mid-band half height = 0.030
        seg_x = [(-ohx, -p - sw), (-p + sw, -sw), (sw, p - sw), (p + sw, ohx)]
        mids = tuple(
            (f"roof_mid{i}", (x1 - x0, 2 * my, rt), ((x0 + x1) / 2, 0.0, rz), roofc)
            for i, (x0, x1) in enumerate(seg_x))
        band_h = ohy - (my + 2 * sw)  # row-slot outer edge (y = p/2 + sw) .. roof edge
        band_cy = (ohy + my + 2 * sw) / 2
        self.pieces = (
            ("floor", (2 * ohx + 0.03, 2 * ohy + 0.03, self.floor_t),
             (0.0, 0.0, -self.floor_t / 2), (0.28, 0.28, 0.31)),
            ("wall_s", (2 * ohx, wt, wh), (0.0, -(ihy + wt / 2), wh / 2), slate),
            ("wall_n", (2 * ohx, wt, wh), (0.0, ihy + wt / 2, wh / 2), slate),
            ("wall_w", (wt, 2 * ihy, wh), (-(ihx + wt / 2), 0.0, wh / 2), slate),
            ("wall_e", (wt, 2 * ihy, wh), (ihx + wt / 2, 0.0, wh / 2), slate),
            ("roof_bs", (2 * ohx, band_h, rt), (0.0, -band_cy, rz), roofc),
            ("roof_bn", (2 * ohx, band_h, rt), (0.0, band_cy, rz), roofc),
        ) + mids
        self.cells = tuple((c * p - p, r * p - p / 2) for c in range(3) for r in range(2))
        self.z_off = self.floor_t  # floor slab bottom rests on the ground plane

        # --- honesty asserts -----------------------------------------------------------------
        # Captivity: the tile slab cannot pass any slot; the roof headroom traps it.
        assert self.tile_s - 2 * sw >= 0.050, "tile must interfere with the slot by >= 50 mm"
        headroom = wh - self.tile_h
        assert 0.003 <= headroom <= 0.008, f"roof headroom must trap the tile ({headroom})"
        # The knob rides the slot with real clearance (contact offsets included) and protrudes
        # far enough above the roof top to be pushed by a fingertip.
        assert sw - self.knob_r - 2 * self.contact_offset >= 0.001, "knob must clear the slot"
        protrude = self.tile_h + self.knob_h - (wh + rt)
        assert protrude >= 0.020, f"knob must protrude >= 20 mm above the roof ({protrude})"
        # Lattice play: tiles fit their cells and clear the walls, but cannot bypass a
        # neighbour (no two tiles fit one cell's span).
        play = p - self.tile_s
        assert 0.004 <= play <= 0.010, f"cell play out of band ({play})"
        assert 2 * self.tile_s > p + 0.05, "two tiles must never share one pitch"
        # In-cell tolerance: generous for a resting tile, decisive against a straddle.
        assert self.cell_tol >= play / 2 + self.spawn_jitter + 0.010, "resting tile must latch"
        assert p / 2 - self.cell_tol >= 0.010, "a straddle between cells must not latch"
        assert self.moved_thresh >= self.cell_tol + 0.020, "moved latch needs a real move"
        assert self.moved_thresh <= p - 0.020, "a full cell move must latch moved"
        # z gates: a resting tile root sits at ~0; anything at/above the roof fails.
        assert self.z_gate >= 0.020, "resting tile must pass the z gate"
        assert (wh + rt) - self.z_gate >= 0.015, "an above-roof tile must fail the z gate"
        # Null policy scores 0: the target is always >= 2 lattice moves from the gold start.
        assert self.min_dist >= 2, "gold must start >= 2 lattice moves from the target"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("tile_shuffle")
class TileShuffleScene(BaseScene):
    cfg: TileShuffleSceneCfg

    def __init__(self, cfg: TileShuffleSceneCfg | None = None) -> None:
        super().__init__(cfg or TileShuffleSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic frame pieces (re-pinned per reset), the 4 collision-free
        target marker pads, and the 5 dynamic knobbed tiles."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        floor_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.25, dynamic_friction=0.25, restitution=0.0)
        slick_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.12, dynamic_friction=0.10, restitution=0.0)

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
        }
        for name, size, ctr, rgb in c.pieces:
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Frame_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=floor_mat if name == "floor" else slick_mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.frame_pos[0] + ctr[0], c.frame_pos[1] + ctr[1], c.z_off + ctr[2])),
            )
        # target marker pads: kinematic, VISUAL-ONLY (no collider — a proud collider edge
        # would wall the sliding knobs); re-pinned onto the target cell's roof corners.
        for i in range(4):
            out[f"pad{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad" + str(i),
                spawn=sim_utils.CuboidCfg(
                    size=c.pad_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=None,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.frame_pos[0], c.frame_pos[1],
                         c.z_off + c.wall_h + c.roof_t + c.pad_size[2] / 2)),
            )
        # tiles: one gold + four grey, all identical physics
        for i, name in enumerate(TILE_NAMES):
            gold = name == "gold"
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tile_" + name,
                spawn=cls["tile"](
                    tile_s=c.tile_s, tile_h=c.tile_h, knob_r=c.knob_r, knob_h=c.knob_h,
                    mass=c.tile_mass, contact_offset=c.contact_offset,
                    color=c.gold_color if gold else c.grey_color,
                    knob_color=c.gold_color if gold else (0.15, 0.15, 0.17)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.frame_pos[0] + c.cells[i][0], c.frame_pos[1] + c.cells[i][1],
                         c.z_off + 0.001)),
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
        """Grab handles + allocate the per-env frame pose, layout, and latches."""
        super().bind(env)
        c = self.cfg
        n = env.num_envs
        dev = env.device
        self.frame: dict[str, RigidObject] = {name: env.iscene[name]
                                              for name, _s, _c, _rgb in c.pieces}
        self.pads: list[RigidObject] = [env.iscene[f"pad{i}"] for i in range(4)]
        self.tiles: dict[str, RigidObject] = {name: env.iscene[name] for name in TILE_NAMES}
        self.env_origins = env.iscene.env_origins
        self.f_pos = torch.zeros(n, 2, device=dev)  # frame centre xy (env-local)
        self.f_yaw = torch.zeros(n, device=dev)
        self.gold_start = torch.zeros(n, dtype=torch.long, device=dev)
        self.target = torch.zeros(n, dtype=torch.long, device=dev)
        self.blank = torch.zeros(n, dtype=torch.long, device=dev)
        self.start_xy = torch.zeros(n, len(TILE_NAMES), 2, device=dev)  # start CELL centres
        self.moved = torch.zeros(n, dtype=torch.bool, device=dev)
        self.d0 = torch.zeros(n, device=dev)
        self.d_min = torch.zeros(n, device=dev)
        self._cells = torch.tensor(c.cells, device=dev)  # (6, 2)
        self._md = torch.tensor([[_mdist(i, j) for j in range(6)] for i in range(6)],
                                device=dev, dtype=torch.float)  # (6, 6)

    # ----- frame transforms ---------------------------------------------------------------------
    def local_to_world(self, p_local: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """(m, 3) frame-local points -> world, applying yaw + frame centre + env origin."""
        co, si = torch.cos(self.f_yaw[env_ids]), torch.sin(self.f_yaw[env_ids])
        out = torch.empty_like(p_local)
        out[:, 0] = self.f_pos[env_ids, 0] + co * p_local[:, 0] - si * p_local[:, 1]
        out[:, 1] = self.f_pos[env_ids, 1] + si * p_local[:, 0] + co * p_local[:, 1]
        out[:, 2] = p_local[:, 2] + self.cfg.z_off
        return out + self.env_origins[env_ids]

    def world_to_local(self, p_world: torch.Tensor) -> torch.Tensor:
        """(N, 3) world points -> frame-local, all envs."""
        p = p_world - self.env_origins
        co, si = torch.cos(self.f_yaw), torch.sin(self.f_yaw)
        dx = p[:, 0] - self.f_pos[:, 0]
        dy = p[:, 1] - self.f_pos[:, 1]
        out = torch.empty_like(p)
        out[:, 0] = co * dx + si * dy
        out[:, 1] = -si * dx + co * dy
        out[:, 2] = p[:, 2] - self.cfg.z_off
        return out

    def _pin_frame(self, env_ids: torch.Tensor) -> None:
        """Write the kinematic frame pieces + the 4 target pads at the current frame pose."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        half = self.f_yaw[env_ids] / 2
        qw, qz = torch.cos(half), torch.sin(half)
        for name, _size, ctr, _rgb in c.pieces:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = self.local_to_world(
                torch.tensor(ctr, device=dev).expand(m, 3).clone(), env_ids)
            st[:, 3] = qw
            st[:, 6] = qz
            self.frame[name].write_root_state_to_sim(st, env_ids)
        tc = self._cells[self.target[env_ids]]  # (m, 2)
        pz = c.wall_h + c.roof_t + c.pad_size[2] / 2
        for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            pl = torch.zeros(m, 3, device=dev)
            pl[:, 0] = tc[:, 0] + sx * c.pad_off
            pl[:, 1] = tc[:, 1] + sy * c.pad_off
            pl[:, 2] = pz
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = self.local_to_world(pl, env_ids)
            st[:, 3] = qw
            st[:, 6] = qz
            self.pads[i].write_root_state_to_sim(st, env_ids)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the frame pose (xy jitter + yaw), sample the layout (gold
        cell, target >= min_dist away, vacancy cell, greys fill the rest), re-pin the frame +
        pads, seat the tiles in their cells, clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        _ = torch.rand(4, device=dev)  # burn post-seed draws (first draws are degenerate)

        self.f_pos[env_ids, 0] = c.frame_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.frame_jitter
        self.f_pos[env_ids, 1] = c.frame_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.frame_jitter
        self.f_yaw[env_ids] = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.frame_yaw_deg)

        ids = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
        assign = torch.zeros(m, len(TILE_NAMES), dtype=torch.long, device=dev)
        for k, e in enumerate(ids):
            g = int(torch.randint(0, 6, (1,), device=dev))
            cand = [t for t in range(6) if _mdist(g, t) >= c.min_dist]
            t_cell = cand[int(torch.randint(0, len(cand), (1,), device=dev))]
            rest = [t for t in range(6) if t != g]
            b = rest[int(torch.randint(0, len(rest), (1,), device=dev))]
            greys = [t for t in rest if t != b]
            self.gold_start[e] = g
            self.target[e] = t_cell
            self.blank[e] = b
            assign[k, 0] = g
            for j, cell in enumerate(greys):
                assign[k, j + 1] = cell

        self._pin_frame(env_ids)

        for j, name in enumerate(TILE_NAMES):
            cc = self._cells[assign[:, j]]  # (m, 2)
            jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.spawn_jitter
            pl = torch.zeros(m, 3, device=dev)
            pl[:, :2] = cc + jit
            pl[:, 2] = 0.001
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = self.local_to_world(pl, env_ids)
            half = self.f_yaw[env_ids] / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            self.tiles[name].write_root_state_to_sim(st, env_ids)
            self.start_xy[env_ids, j] = cc

        self.moved[env_ids] = False
        self.d0[env_ids] = self._md[self.gold_start[env_ids], self.target[env_ids]]
        self.d_min[env_ids] = self.d0[env_ids]

    # ----- geometry queries ---------------------------------------------------------------------
    def tiles_local(self) -> torch.Tensor:
        """(N, 5, 3) tile roots (slab bottom centres) in the frame-local frame; order gold,
        grey0..grey3."""
        w = torch.stack([self.tiles[n].data.root_pos_w for n in TILE_NAMES], dim=1)
        out = torch.empty_like(w)
        for j in range(w.shape[1]):
            out[:, j] = self.world_to_local(w[:, j])
        return out

    def gold_cell(self) -> torch.Tensor:
        """(N,) long: the gold tile's current cell index, or -1 if not seated in any cell
        (off-centre beyond `cell_tol`, or failing the z gate)."""
        c = self.cfg
        p = self.world_to_local(self.tiles["gold"].data.root_pos_w)
        d = (p[:, None, :2] - self._cells[None]).norm(dim=-1)  # (N, 6)
        mind, idx = d.min(dim=1)
        ok = (mind < c.cell_tol) & (p[:, 2] < c.z_gate)
        return torch.where(ok, idx, torch.full_like(idx, -1))

    def settled(self) -> torch.Tensor:
        v = torch.stack([self.tiles[n].data.root_lin_vel_w.norm(dim=-1) for n in TILE_NAMES],
                        dim=1)
        return (v < self.cfg.settle_speed).all(dim=1)

    # ----- mechanics (every substep) ------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch `moved` (any tile displaced a real cell move, z-gated) and `d_min` (best
        lattice distance of the gold tile to the target, only while seated in a cell)."""
        c = self.cfg
        p = self.tiles_local()  # (N, 5, 3)
        zok = p[..., 2] < c.z_gate
        disp = (p[..., :2] - self.start_xy).norm(dim=-1)  # (N, 5)
        self.moved |= (zok & (disp > c.moved_thresh)).any(dim=1)
        gc = self.gold_cell()
        seated = gc >= 0
        d = self._md[gc.clamp(min=0), self.target]
        self.d_min = torch.where(seated, torch.minimum(self.d_min, d), self.d_min)

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "tiles": {n: self.tiles[n].data.root_state_w[env_ids].clone() for n in TILE_NAMES},
            "frame": {"f_pos": self.f_pos[env_ids].clone(), "f_yaw": self.f_yaw[env_ids].clone()},
            "layout": {"gold_start": self.gold_start[env_ids].clone(),
                       "target": self.target[env_ids].clone(),
                       "blank": self.blank[env_ids].clone(),
                       "start_xy": self.start_xy[env_ids].clone()},
            "latches": {"moved": self.moved[env_ids].clone(), "d0": self.d0[env_ids].clone(),
                        "d_min": self.d_min[env_ids].clone()},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.f_pos[env_ids] = state["frame"]["f_pos"]
        self.f_yaw[env_ids] = state["frame"]["f_yaw"]
        self.gold_start[env_ids] = state["layout"]["gold_start"]
        self.target[env_ids] = state["layout"]["target"]
        self.blank[env_ids] = state["layout"]["blank"]
        self.start_xy[env_ids] = state["layout"]["start_xy"]
        self._pin_frame(env_ids)
        for n in TILE_NAMES:
            self.tiles[n].write_root_state_to_sim(state["tiles"][n], env_ids)
        self.moved[env_ids] = state["latches"]["moved"]
        self.d0[env_ids] = state["latches"]["d0"]
        self.d_min[env_ids] = state["latches"]["d_min"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A walled tray on the ground holds a 3 x 2 lattice of square cells "
            f"({c.pitch * 100:.0f} cm pitch). Five sliding tiles fill five of the six cells — "
            f"four grey and one GOLD — so exactly one cell is vacant. A slotted roof covers the "
            f"tray {1000 * (c.wall_h - c.tile_h):.0f} mm above the tile tops: each tile's knob "
            f"pokes up through the roof's slot network (slots run only along the row and column "
            f"centre lines), so tiles cannot be lifted out, cannot move diagonally, and can only "
            f"slide into the current vacancy. The target cell is marked by four green pads on "
            f"the roof at its corners. The tray's position and heading, the tile layout, the "
            f"vacancy, and the target cell all vary per episode.\n"
            f"Goal: slide tiles one at a time — pushing sideways on the protruding knobs — to "
            f"shuffle the vacancy around until the GOLD tile can be walked into the green-marked "
            f"target cell, and leave it seated there. Most of the work is moving the OTHER "
            f"tiles to route the vacancy; the gold tile itself can only ever move into an "
            f"adjacent empty cell. No grasp is required anywhere."
        )

    def instruction(self) -> str:
        return (
            "Shuffle the sliding tiles by pushing their knobs: move grey tiles into the vacant "
            "cell to route the vacancy next to the gold tile, then walk the gold tile step by "
            "step into the cell marked by the four green corner pads, and leave it seated there."
        )

    # ----- progress / rubric --------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the gold tile is seated in the TARGET cell (within `cell_tol`, z-gated)
        and every tile is settled."""
        return (self.gold_cell() == self.target) & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.08 once any tile has made a real move + 0.52 * latched best
        lattice progress of the gold tile toward the target; exactly 1.0 iff success().
        Latched credit never evaporates; the null policy scores 0 (d0 >= 2)."""
        prog = (self.d0 - self.d_min) / self.d0.clamp(min=1.0)
        s = 0.08 * self.moved.float() + 0.52 * prog
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("simgen", lambda: EnvCfg(scene="tile_shuffle", robot="null", env_spacing=3))
