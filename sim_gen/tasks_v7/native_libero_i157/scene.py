"""DominoRelayScene — build a correctly SPACED row of tiles between two floor marks,
then topple the first tile so a self-propagating cascade carries the fall to the far
mark (sim_gen task `native_libero_i157`).

Derived from libero/native_libero, but STRATEGICALLY different: every LIBERO task is a
single-object relocation or fixture articulation — pick object X and place it inside /
on region Y (or open/close a drawer or appliance), judged by static BDDL ``on``/``in``
predicates over goal regions. Here NO object has a goal region of its own and no single
relocation can ever reach the goal. The task content is the RELATIVE GEOMETRY of six
identical tiles: they must be stood upright in a lane between a red START disc and a
green TARGET plate at face-to-face pitches inside the topple window (each gap small
enough that a falling tile strikes the next well above its center of mass, wide enough
that the row does not lean), and then the FIRST tile must be tipped over so the chain
reaction — pure gravity + tile-on-tile contact, with no further intervention — carries
the fall tile by tile down the lane until the LAST tile's head lands on the target
plate. A solver needs a different PLAN from the seed (plan spacings from A to B under
a per-episode distance/heading draw, place N interchangeable objects relative to EACH
OTHER, then trigger one dynamic event and stand back) and a different code structure
(the rubric is a chain-topology audit — per-tile fall direction, pairwise LAP contacts
where each tile's head rests ON the next tile's body, an anchored first tile and a
head-on-plate last tile — not object-in-region checks).

The goal state is judged from settled poses only:
  - lane membership: tile center within `lane_half` of segment A->B;
  - every lane tile has FALLEN toward B (long axis within `fall_axis_z_max` of
    horizontal, horizontal heading within ~60 deg of the A->B direction);
  - the lane tiles, ordered along A->B, form a LAPPED carpet: each tile's head
    (top-face center) lies horizontally within `lap_xy` of the next tile's center AND
    rests at `lap_z_min` or higher — i.e. propped ON the next tile, the physical
    signature of a cascade at correct pitch (tiles laid flat end-to-end on the ground
    rest at ~t/2 and are rejected);
  - the first lane tile's foot is within `anchor_tol` of the START disc center A;
  - the last lane tile's head lies within `pad_r` of the TARGET plate center B (and
    is down, not standing over it);
  - everything settled. A single tile can never both anchor at A and reach B
    (asserted in `__post_init__`), and the 6-tile even-pitch bridge lands inside the
    reliable topple window for every sampled A->B distance (also asserted).

score() is graded and latched (credit never evaporates): 0.05 per tile ever
simultaneously standing in the lane (build credit, cap 0.30) + 0.05 per tile ever
simultaneously fallen-toward-B in the lane (cascade credit, cap 0.30) + 0.10 the
first time any fallen tile's head lies on the target plate = 0.70 cap; 1.0 iff
success(). The null policy scores ~0 (all tiles start lying flat in a depot far
outside the lane).

Per-episode randomization (readback-verified in smoke): the START disc position, the
A->B heading and distance (so the required number of tiles and every pitch change),
and the depot tiles' slot permutation + xy jitter + free yaw. Markers are
VISUAL-ONLY kinematic bodies (no colliders — tiles always stand on flat ground) that
the reset teleports to the sampled A/B, which are also stored per env for judging.
Assets are fully procedural; heavy imports (isaaclab, pxr) are deferred so importing
this module — and registering the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- custom spawners --------------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _material(stage, path: str, static: float = 0.55, dynamic: float = 0.45):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float,
                kinematic: bool = False):
    """One rigid-body root Xform with the standard physics armor (zero sleep /
    stabilization thresholds: a sleeping body silently ignores applied wrenches,
    which the solve/smoke force probes depend on)."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    mapi = UsdPhysics.MassAPI.Apply(root)
    mapi.CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return root, pxrb


def _spawn_tile(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One domino tile: a single box, thickness x width x height, dynamic."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    # generous damping: fallen tiles rest head-on-edge (lapped carpet) — sharp-edge
    # contacts produce phantom limit-cycle velocities on GPU PhysX without it
    pxrb.CreateLinearDampingAttr(0.08)
    pxrb.CreateAngularDampingAttr(0.30)
    mat = _material(stage, f"{prim_path}/phys_mat")
    seg = UsdGeom.Cube.Define(stage, f"{prim_path}/body")
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddScaleOp().Set(Gf.Vec3f(float(cfg.t), float(cfg.w), float(cfg.h)))
    seg.CreateDisplayColorAttr([Gf.Vec3f(0.92, 0.55, 0.12)])  # orange
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(cfg.contact_offset))
    px.CreateRestOffsetAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    return root


def _spawn_marker(prim_path: str, cfg: Any, translation=None, orientation=None):
    """VISUAL-ONLY floor marker: a kinematic rigid body (so reset can teleport it per
    env) carrying one thin colored box with NO collider — tiles always stand on flat
    ground; the marker never touches physics."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 0.05,
                              kinematic=True)
    seg = UsdGeom.Cube.Define(stage, f"{prim_path}/face")
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    sxf.AddScaleOp().Set(Gf.Vec3f(float(cfg.sx), float(cfg.sy), 0.0015))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tile" not in _SPAWNER_CACHE:

        @configclass
        class TileSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tile)
            t: float = 0.012
            w: float = 0.036
            h: float = 0.060
            mass: float = 0.030
            contact_offset: float = 0.002

        @configclass
        class MarkerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_marker)
            sx: float = 0.07
            sy: float = 0.07
            color: tuple = (0.85, 0.10, 0.10)

        _SPAWNER_CACHE.update(tile=TileSpawnerCfg, marker=MarkerSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DominoRelaySceneCfg(BaseCfg):
    """Config for `DominoRelayScene`. The chain honesty (one tile can never bridge
    A->B; six tiles at even pitch always land inside the reliable topple window) is
    asserted in `__post_init__` from these numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    lane_half: float = tunable(0.06)  # lane half-width around segment A->B
    lane_over: float = tunable(0.06)  # lane extension beyond A (behind) and B (ahead)
    fall_axis_z_max: float = tunable(0.55)  # |axis_z| below this counts as fallen
    stand_axis_z_min: float = tunable(0.90)  # axis_z above this counts as standing
    dir_dot_min: float = tunable(0.50)  # fallen heading within ~60 deg of A->B
    anchor_tol: float = tunable(0.045)  # first tile's foot within this of A (xy)
    lap_xy: float = tunable(0.050)  # head of tile i within this of tile i+1 center (xy)
    lap_z_min: float = tunable(0.009)  # head resting ON the next tile, not the ground
    pad_r: float = tunable(0.050)  # last tile's head within this of B (xy)
    head_z_max: float = tunable(0.040)  # ... and down (not standing over the plate)
    # settled = LOOSE instantaneous velocity ceilings AND tiny pose drift across >= 2
    # consecutive 30-substep windows (lapped tiles rest head-on-edge: GPU PhysX edge
    # contacts hold a phantom ~0.7 rad/s angular limit cycle in place — velocity
    # thresholds alone cannot separate that artifact from real motion, pose drift can)
    settle_lin: float = tunable(0.10)  # instantaneous |lin vel| ceiling (m/s)
    settle_ang: float = tunable(2.00)  # instantaneous |ang vel| ceiling (rad/s)
    settle_drift: float = tunable(0.002)  # max center drift per window (m)
    settle_axis_drift: float = tunable(0.03)  # max long-axis drift per window (unit vec)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    start_xy: tuple = tunable((0.06, 0.01))  # nominal START disc center A
    start_jitter: tuple = tunable((0.020, 0.030))  # +- xy jitter on A
    dist_range: tuple = tunable((0.22, 0.30))  # |A->B| draw
    heading_deg: float = tunable(25.0)  # A->B heading: +x rotated by U(-this, +this)
    depot_slots: tuple = tunable(((0.02, -0.29), (0.10, -0.31), (0.18, -0.29),
                                  (0.26, -0.31), (0.02, -0.37), (0.10, -0.39)))
    depot_jitter: float = tunable(0.015)  # +- xy jitter per tile at its permuted slot
    depot_yaw_deg: float = tunable(180.0)  # +- free yaw per lying tile

    # --- info: structure ---------------------------------------------------------------------
    n_tiles: int = info(6)
    tile_t: float = info(0.012)  # thickness (the falling direction)
    tile_w: float = info(0.036)  # width (across the lane)
    tile_h: float = info(0.060)  # height (the long axis)
    tile_mass: float = info(0.030)
    gap_min: float = info(0.018)  # reliable topple window: face-to-face gap floor
    gap_max: float = info(0.042)  # ... and ceiling (strike stays above mid-height)
    contact_offset: float = info(0.002)
    settle_window: int = info(30)  # substeps per pose-drift window
    settle_streak: int = info(2)  # consecutive quiet windows required

    def __post_init__(self) -> None:
        c = self
        reach = math.sqrt(c.tile_h ** 2 - c.tile_t ** 2)  # fallen head reach from foot
        # -- one tile can NEVER both anchor at A and put its head on the plate at B --
        one_tile_span = c.anchor_tol + reach + c.pad_r
        assert one_tile_span < c.dist_range[0] - 0.005, \
            f"one tile must never bridge A->B ({one_tile_span:.3f} vs {c.dist_range[0]:.3f})"
        # -- six tiles at even pitch always fit the reliable topple window --
        for d in c.dist_range:
            pitch = (d - reach) / (c.n_tiles - 1)
            gap = pitch - c.tile_t
            assert c.gap_min <= gap <= c.gap_max, \
                f"6-tile even gap {gap:.3f} outside topple window at distance {d:.3f}"
        # -- a flat tile's head (~t/2) sits below the lap gate; a propped head sits above --
        assert c.tile_t / 2 + 0.002 < c.lap_z_min < c.tile_t, \
            "lap_z_min must separate flat-on-ground heads from propped heads"
        # -- the lane never reaches the depot --
        b_y_min = c.start_xy[1] - c.start_jitter[1] \
            - c.dist_range[1] * math.sin(math.radians(c.heading_deg))
        lane_low = b_y_min - c.lane_half - 0.01
        depot_top = max(y for _x, y in c.depot_slots) + c.depot_jitter + c.tile_h / 2
        assert depot_top < lane_low, \
            f"depot must stay outside every possible lane ({depot_top:.3f} vs {lane_low:.3f})"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("domino_relay")
class DominoRelayScene(BaseScene):
    cfg: DominoRelaySceneCfg

    def __init__(self, cfg: DominoRelaySceneCfg | None = None) -> None:
        super().__init__(cfg or DominoRelaySceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.60, dynamic_friction=0.50, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "start_mark": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/StartMark",
                spawn=sp["marker"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    sx=0.07, sy=0.07, color=(0.85, 0.10, 0.10)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.start_xy[0], c.start_xy[1], 0.0008)),
            ),
            "target_mark": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/TargetMark",
                spawn=sp["marker"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    sx=0.10, sy=0.10, color=(0.10, 0.70, 0.20)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.start_xy[0] + 0.20, c.start_xy[1], 0.0008)),
            ),
        }
        for i in range(c.n_tiles):
            sx, sy = c.depot_slots[i]
            out[f"tile_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tile" + str(i),
                spawn=sp["tile"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.tile_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    t=c.tile_t, w=c.tile_w, h=c.tile_h, mass=c.tile_mass,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx, sy, c.tile_t / 2 + 0.002),
                    rot=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)),
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
        n, dev = env.num_envs, env.device
        c = self.cfg
        self.tiles: list[RigidObject] = [env.iscene[f"tile_{i}"] for i in range(c.n_tiles)]
        self.start_mark: RigidObject = env.iscene["start_mark"]
        self.target_mark: RigidObject = env.iscene["target_mark"]
        self.env_origins = env.iscene.env_origins
        # per-env episode parameters (sampled at reset; judged against)
        self.A = torch.zeros(n, 2, device=dev)  # start disc center
        self.B = torch.zeros(n, 2, device=dev)  # target plate center
        self.U = torch.zeros(n, 2, device=dev)  # unit A->B
        self.D = torch.zeros(n, device=dev)  # |A->B|
        # progress latches (post_step)
        self.stand_latch = torch.zeros(n, device=dev)  # max tiles standing in lane
        self.fall_latch = torch.zeros(n, device=dev)  # max tiles fallen-toward-B in lane
        self.pad_latch = torch.zeros(n, device=dev)  # any fallen head ever on the plate
        # pose-drift stillness tracker (windowed streak; see cfg.settle_*)
        self._ref_pos = torch.zeros(n, c.n_tiles, 3, device=dev)
        self._ref_axis = torch.zeros(n, c.n_tiles, 3, device=dev)
        self._drift_streak = torch.zeros(n, device=dev)
        self._settle_tick = 0

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample A (jitter), heading and distance -> B; teleport the
        visual markers there; scatter the tiles lying flat at PERMUTED depot slots
        with xy jitter + free yaw; zero the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        ax = c.start_xy[0] + (torch.rand(m, device=dev) * 2 - 1) * c.start_jitter[0]
        ay = c.start_xy[1] + (torch.rand(m, device=dev) * 2 - 1) * c.start_jitter[1]
        phi = torch.deg2rad((torch.rand(m, device=dev) * 2 - 1) * c.heading_deg)
        d = c.dist_range[0] + torch.rand(m, device=dev) * (c.dist_range[1] - c.dist_range[0])
        u = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
        a = torch.stack([ax, ay], dim=-1)
        b = a + u * d.unsqueeze(-1)
        self.A[env_ids], self.B[env_ids] = a, b
        self.U[env_ids], self.D[env_ids] = u, d

        for mark, xy in ((self.start_mark, a), (self.target_mark, b)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = 0.0008
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            mark.write_root_state_to_sim(st, env_ids)

        slots = torch.tensor(c.depot_slots, device=dev)  # (K, 2)
        perm_rank = torch.rand(m, c.n_tiles, device=dev).argsort(dim=1)  # (m, K)
        yaw_amp = math.radians(c.depot_yaw_deg)
        c45 = math.cos(math.pi / 4)
        for i, tile in enumerate(self.tiles):
            sl = slots[perm_rank[:, i]]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = sl + (torch.rand(m, 2, device=dev) * 2 - 1) * c.depot_jitter
            st[:, 2] = c.tile_t / 2 + 0.002
            # lying flat on the big face: q = qz(yaw) * qy(90 deg)
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(half) * c45
            st[:, 4] = -torch.sin(half) * c45
            st[:, 5] = torch.cos(half) * c45
            st[:, 6] = torch.sin(half) * c45
            st[:, 0:3] += origin
            tile.write_root_state_to_sim(st, env_ids)

        self.stand_latch[env_ids] = 0.0
        self.fall_latch[env_ids] = 0.0
        self.pad_latch[env_ids] = 0.0
        self._drift_streak[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "tiles": {f"tile_{i}": t.data.root_state_w[env_ids].clone()
                      for i, t in enumerate(self.tiles)},
            "A": self.A[env_ids].clone(), "B": self.B[env_ids].clone(),
            "U": self.U[env_ids].clone(), "D": self.D[env_ids].clone(),
            "stand_latch": self.stand_latch[env_ids].clone(),
            "fall_latch": self.fall_latch[env_ids].clone(),
            "pad_latch": self.pad_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for i, t in enumerate(self.tiles):
            t.write_root_state_to_sim(state["tiles"][f"tile_{i}"], env_ids)
        self.A[env_ids], self.B[env_ids] = state["A"], state["B"]
        self.U[env_ids], self.D[env_ids] = state["U"], state["D"]
        self.stand_latch[env_ids] = state["stand_latch"]
        self.fall_latch[env_ids] = state["fall_latch"]
        self.pad_latch[env_ids] = state["pad_latch"]
        self._drift_streak[env_ids] = 0.0  # conservative: re-earn stillness

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "On the floor lie a small RED START DISC and, some distance away, a larger "
            "GREEN TARGET PLATE (both flat, purely painted marks — the distance and "
            f"direction between them change every episode; here they sit roughly "
            f"{int(c.dist_range[0] * 100)}-{int(c.dist_range[1] * 100)} cm apart). Off to "
            f"the side, {c.n_tiles} identical ORANGE DOMINO TILES "
            f"({int(c.tile_t * 1000)} mm thick, {int(c.tile_w * 1000)} mm wide, "
            f"{int(c.tile_h * 1000)} mm tall) lie flat in a loose depot; their positions "
            "and orientations change every episode — find them by looking.\n"
            "Goal: build a domino run from the red disc to the green plate and set it "
            "off. Stand tiles upright in the lane between the marks, flat faces toward "
            "the red disc, the first tile standing ON the red disc, successive tiles "
            f"spaced so each face-to-face gap is roughly {int(c.gap_min * 1000)}-"
            f"{int(c.gap_max * 1000)} mm (a falling tile must strike the next one high "
            "up; too wide and the chain dies, and the last tile must stand close enough "
            "to the plate that its head lands ON the plate when it falls). Then tip the "
            "FIRST tile over toward the green plate and let the chain reaction do the "
            "rest — the cascade must run tile to tile, each fallen tile left resting "
            "propped on the next (the lapped carpet a real domino run leaves), and the "
            "last tile's head must come to rest on the green plate. You do not need "
            "every tile: use as many as the distance requires and leave the spares in "
            "the depot — any extra tile left inside the lane (standing, sideways, or "
            "flat) spoils the run. Judged when everything is at rest: first tile's "
            "foot at the red disc, every lane tile fallen toward the plate, each "
            "propped on the next in one connected carpet, last head on the green "
            "plate. Tiles laid down flat by hand end-to-end (resting on the ground "
            "instead of on each other), a chain fallen the wrong way, a chain that "
            "stops short, or a first tile away from the red disc all fail."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Stand the orange domino tiles in an evenly spaced row from the red disc "
            "to the green plate — first tile on the disc, gaps small enough to "
            "propagate a fall — then tip the first tile toward the plate so the chain "
            "reaction topples every tile and the last one lands with its head on the "
            "green plate. Leave no extra tile in the lane."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _tile_frames(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(centers (N,K,3) env-frame, axis (N,K,3) world unit long-axis, lin vel (N,K),
        ang vel (N,K)) for all tiles, index order."""
        from isaaclab.utils.math import quat_apply

        n, k = self.env.num_envs, len(self.tiles)
        pos = torch.stack([t.data.root_pos_w for t in self.tiles], dim=1) \
            - self.env_origins[:, None, :]
        quat = torch.stack([t.data.root_quat_w for t in self.tiles], dim=1)
        ez = torch.tensor([0.0, 0.0, 1.0], device=pos.device).expand(n * k, 3)
        axis = quat_apply(quat.reshape(n * k, 4), ez).reshape(n, k, 3)
        lin = torch.stack([t.data.root_lin_vel_w.norm(dim=-1) for t in self.tiles], dim=1)
        ang = torch.stack([t.data.root_ang_vel_w.norm(dim=-1) for t in self.tiles], dim=1)
        return pos, axis, lin, ang

    def _lane_geometry(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(proj (N,K) along A->B, perp (N,K) distance to the lane axis, in_lane (N,K))."""
        c = self.cfg
        rel = pos[:, :, :2] - self.A[:, None, :]
        proj = (rel * self.U[:, None, :]).sum(-1)
        perp = (rel - proj.unsqueeze(-1) * self.U[:, None, :]).norm(dim=-1)
        in_lane = (perp < c.lane_half) & (proj > -c.lane_over) \
            & (proj < self.D[:, None] + c.lane_over)
        return proj, perp, in_lane

    def tile_report(self) -> dict[str, torch.Tensor]:
        """All per-tile predicates in one pass (shapes (N,K) unless noted)."""
        c = self.cfg
        pos, axis, lin, ang = self._tile_frames()
        proj, perp, in_lane = self._lane_geometry(pos)
        head = pos + axis * (c.tile_h / 2)  # top-face center (env frame)
        foot = pos - axis * (c.tile_h / 2)
        axis_xy = axis[:, :, :2]
        axis_xy_n = axis_xy / axis_xy.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        dir_dot = (axis_xy_n * self.U[:, None, :]).sum(-1)
        fallen = (axis[:, :, 2].abs() < c.fall_axis_z_max) & (dir_dot > c.dir_dot_min)
        standing = axis[:, :, 2] > c.stand_axis_z_min
        head_on_pad = ((head[:, :, :2] - self.B[:, None, :]).norm(dim=-1) < c.pad_r) \
            & fallen & (head[:, :, 2] < c.head_z_max)
        still = (lin < c.settle_lin) & (ang < c.settle_ang)
        return {"pos": pos, "axis": axis, "head": head, "foot": foot, "proj": proj,
                "perp": perp, "in_lane": in_lane, "fallen": fallen, "standing": standing,
                "head_on_pad": head_on_pad, "still": still, "dir_dot": dir_dot}

    def settled(self) -> torch.Tensor:
        """(N,) bool: every tile at rest — loose instantaneous velocity ceilings AND
        pose drift below `settle_drift`/`settle_axis_drift` across the last
        `settle_streak` windowed checks (post_step maintains the streak). Velocity
        thresholds alone cannot reject the GPU edge-contact phantom limit cycle that
        lapped tiles carry while visibly motionless; pose drift can."""
        _pos, _axis, lin, ang = self._tile_frames()
        vel_ok = ((lin < self.cfg.settle_lin) & (ang < self.cfg.settle_ang)).all(dim=1)
        return vel_ok & (self._drift_streak >= self.cfg.settle_streak)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch each demonstrated stage every physics substep: tiles simultaneously
        standing in the lane (build), tiles simultaneously fallen-toward-B in the lane
        (cascade), any fallen head on the target plate (delivery). Also advance the
        windowed pose-drift stillness streak."""
        c = self.cfg
        r = self.tile_report()
        self.stand_latch = torch.maximum(
            self.stand_latch, (r["standing"] & r["in_lane"]).sum(dim=1).float())
        self.fall_latch = torch.maximum(
            self.fall_latch, (r["fallen"] & r["in_lane"]).sum(dim=1).float())
        self.pad_latch = torch.maximum(
            self.pad_latch, r["head_on_pad"].any(dim=1).float())
        self._settle_tick += 1
        if self._settle_tick >= c.settle_window:
            self._settle_tick = 0
            dp = (r["pos"] - self._ref_pos).norm(dim=-1).amax(dim=1)
            da = (r["axis"] - self._ref_axis).norm(dim=-1).amax(dim=1)
            quiet = (dp < c.settle_drift) & (da < c.settle_axis_drift)
            self._drift_streak = torch.where(
                quiet, self._drift_streak + 1.0, torch.zeros_like(self._drift_streak))
            self._ref_pos = r["pos"].clone()
            self._ref_axis = r["axis"].clone()

    # ----- rubric -----------------------------------------------------------------------------
    def chain_ok(self) -> torch.Tensor:
        """(N,) bool: the settled chain-topology audit. Ordered along A->B, the lane
        tiles must ALL be fallen toward B, the first tile's foot anchored at A, each
        tile's head propped ON the next (the cascade's lapped-carpet signature), and
        the last tile's head on the target plate."""
        c = self.cfg
        r = self.tile_report()
        n = self.env.num_envs
        out = torch.zeros(n, dtype=torch.bool, device=self.A.device)
        for e in range(n):
            idx = [k for k in range(len(self.tiles)) if bool(r["in_lane"][e, k])]
            if len(idx) < 2:
                continue
            if not all(bool(r["fallen"][e, k]) for k in idx):
                continue
            idx.sort(key=lambda k: float(r["proj"][e, k]))
            first, last = idx[0], idx[-1]
            if float((r["foot"][e, first, :2] - self.A[e]).norm()) > c.anchor_tol:
                continue
            lapped = True
            for i, j in zip(idx[:-1], idx[1:]):
                gap_xy = float((r["head"][e, i, :2] - r["pos"][e, j, :2]).norm())
                if gap_xy > c.lap_xy or float(r["head"][e, i, 2]) < c.lap_z_min:
                    lapped = False
                    break
            if not lapped:
                continue
            if not bool(r["head_on_pad"][e, last]):
                continue
            out[e] = True
        return out

    def success(self) -> torch.Tensor:
        """(N,) bool: the settled lapped carpet runs from the red disc to the green
        plate — chain topology OK and every tile at rest."""
        return self.chain_ok() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: latched 0.05 per tile ever standing in the lane (cap
        0.30) + 0.05 per tile ever fallen-toward-B in the lane (cap 0.30) + 0.10 head
        ever on the plate; cap 0.70; 1.0 iff success(). Credit never evaporates; the
        null policy scores ~0 (tiles start flat in the depot, outside every lane)."""
        base = (0.05 * self.stand_latch.clamp(max=6.0)
                + 0.05 * self.fall_latch.clamp(max=6.0)
                + 0.10 * self.pad_latch).clamp(0.0, 0.70)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="domino_relay", robot="null", env_spacing=3.0))
