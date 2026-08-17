"""StokerStoveScene — turn on the cold pellet stove by FUELLING it: push at
least 3 fuel briquettes through the one-way stoker flap into the closed
firebox, and keep the butter stick out (sim_gen task
`libero_kitchen_scene3_turn_on_the_stove_i193`).

Derived from libero_90/libero_kitchen_scene3_turn_on_the_stove, but
STRATEGICALLY different. The seed "turns on" the stove by rotating a knob past
a JOINT-ANGLE threshold (`flat_stove.joint_pos[:, 0] > 0.5`) — a single fixture
actuation judged directly from a fixture joint. Here there is NO knob, no
fixture joint angle anywhere in the rubric, and no single actuation event:

  (1) the stove is cold because its FIREBOX IS EMPTY. "Turn it on" means
      FUEL it: at least `need`(=3) of the 4 dark fuel briquettes must end up
      INSIDE the closed firebox — a cumulative, multi-object physical-
      containment outcome, not a threshold on any joint.
  (2) the only way in is the STOKER PORT at the end of a loading chute: a
      gravity one-way flap (hinged at the top, swings inward only — its
      revolute joint LIMITS are the stops) guards the port. Feeding is
      IRREVERSIBLE: inside, a briquette sits 26 mm below the sill behind a
      flap that only opens inward, so it cannot be pushed back out.
  (3) the top of the firebox is a GRATE whose gaps are narrower than a
      briquette — fuel cannot be dropped in from above; it must be fed
      through the port with a sustained push along the chute.
  (4) a BUTTER STICK that WOULD fit through the port is on the counter:
      fuelling is spoiled if the butter ends up inside (a real exclusion
      constraint, not decoration). The stove lamp turns orange only when
      >= 3 briquettes are in AND the butter is not.

Assets are fully procedural (native PhysX box colliders; the flap rides a
spawn-authored per-env revolute X joint whose LIMITS [-0.5 deg, +88 deg] are
the hard stops; the joint's collision filter disables ONLY the flap<->firebox
pair, so briquettes contact both):
  - counter (KINEMATIC): 800 x 640 x 140 mm kitchen bench.
  - firebox (KINEMATIC, FIXED pose — it anchors the flap joint and kinematic
    joint anchors are world-fixed): 200 x 180 x 170 mm box, 8 mm walls, with
    a 46 x 46 mm stoker port in the front wall (sill 34 mm up), a 5-bar top
    grate (16 mm gaps), a loading CHUTE running south from the port (channel
    exactly port-wide, 14 mm side walls — low enough to set a gripped
    briquette down into it), and a lamp above the port (visual only).
  - flap (DYNAMIC, 20 g): hangs from the top of the port on the inside,
    covering it; swings inward under a briquette's push, falls shut again.
  - 4 fuel briquettes (DYNAMIC): 32 mm charcoal-dark cubes, 50 g.
  - butter stick (DYNAMIC): 50 x 32 x 30 mm pale-yellow block, 40 g — fits
    through the port, so keeping it out is a live constraint.

Rubric (0..1; anchored in the demonstrated solve; monotone along it because
fed briquettes physically cannot come back out):
  0.25 per briquette inside the firebox (latched inside-and-calm AND
       currently inside), up to 3 -> 0.75
  x0.2 spoilage factor while the butter is inside
  1.0 iff success(): >= `need` briquettes inside, butter NOT inside, every
       counted briquette calm, all bodies finite. Doing nothing scores 0.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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


# ----- procedural compound spawners -------------------------------------------------------------
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


def _box(stage, path: str, size, center, color, contact_offset: float | None) -> None:
    """Author one box child; `contact_offset=None` -> VISUAL ONLY (no collider)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(seg.GetPrim(), contact_offset)


def _rigid_root(root, mass: float, kinematic: bool, lin_damp: float = 0.0,
                ang_damp: float = 0.0) -> None:
    from pxr import PhysxSchema, UsdPhysics

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:  # corpus-validated authoring: never author the attr False
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    if not kinematic:
        px.CreateSleepThresholdAttr(0.0)
        px.CreateStabilizationThresholdAttr(0.0)


def _spawn_counter(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC kitchen bench. Origin = deck centre at GROUND level; the solid
    counter block fills z in [0, deck_h]."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg  # unwrap: the spawner cfg carries the scene cfg in its `cfg` field
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 40.0, kinematic=True)
    _box(stage, f"{prim_path}/deck",
         (2 * cfg.counter_half_x, 2 * cfg.counter_half_y, cfg.deck_h),
         (0.0, 0.0, cfg.deck_h / 2), cfg.deck_color, cfg.contact_offset)
    return root


def _spawn_stove(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC pellet stove. Origin = firebox base centre ON the deck top.
    Children (all one kinematic body): floor slab, back/east/west walls, the
    front wall split around the 46 x 46 mm stoker port (below / above / left /
    right of it), the 5-bar top grate, the loading chute (floor level with the
    sill + two low side walls), and the lamp + trim (VISUAL ONLY). The stove
    anchors the flap joint, so it is NEVER moved after spawn (kinematic joint
    anchors are world-fixed)."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 12.0, kinematic=True)

    co, col = cfg.contact_offset, cfg.stove_color
    hx, hy, h, wt, ft = cfg.fire_hx, cfg.fire_hy, cfg.fire_h, cfg.wall_t, cfg.floor_t
    wall_h = h - ft
    wall_zc = ft + wall_h / 2
    inner_face_y = -(hy - wt)          # front wall inner face (local y)
    front_yc = -(hy - wt / 2)          # front wall slab centre (local y)

    # floor slab (full footprint) + back / side walls
    _box(stage, f"{prim_path}/floor", (2 * hx, 2 * hy, ft), (0.0, 0.0, ft / 2), col, co)
    _box(stage, f"{prim_path}/wall_n", (2 * hx, wt, wall_h),
         (0.0, hy - wt / 2, wall_zc), col, co)
    for tag, sx in (("wall_e", 1.0), ("wall_w", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (wt, 2 * hy - 2 * wt, wall_h),
             (sx * (hx - wt / 2), 0.0, wall_zc), col, co)

    # front (south) wall, split around the stoker port
    pw, sz, tz = cfg.port_w, cfg.sill_z, cfg.port_top_z
    _box(stage, f"{prim_path}/wall_s_below", (pw, wt, sz - ft),
         (0.0, front_yc, (ft + sz) / 2), col, co)
    _box(stage, f"{prim_path}/wall_s_above", (pw, wt, h - tz),
         (0.0, front_yc, (tz + h) / 2), col, co)
    side_w = hx - pw / 2
    for tag, sx in (("wall_s_e", 1.0), ("wall_s_w", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (side_w, wt, wall_h),
             (sx * (pw / 2 + side_w / 2), 0.0 + front_yc, wall_zc), col, co)

    # top grate: bars along y, gaps too narrow for a briquette
    for k in range(cfg.grate_bars):
        xb = (k - (cfg.grate_bars - 1) / 2) * cfg.grate_pitch
        _box(stage, f"{prim_path}/bar_{k}", (cfg.grate_bar_w, 2 * hy, cfg.grate_bar_t),
             (xb, 0.0, cfg.grate_zc), cfg.grate_color, co)

    # loading chute: floor level with the sill, channel exactly port-wide
    cy0, cy1 = cfg.chute_y0, cfg.chute_y1
    _box(stage, f"{prim_path}/chute_floor", (2 * (cfg.chan_hw + cfg.chute_wall_t),
         cy1 - cy0, sz), (0.0, (cy0 + cy1) / 2, sz / 2), cfg.chute_color, co)
    wl = cfg.chute_wall_y1 - cy0
    for tag, sx in (("chute_e", 1.0), ("chute_w", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (cfg.chute_wall_t, wl, cfg.chute_wall_h),
             (sx * (cfg.chan_hw + cfg.chute_wall_t / 2), cy0 + wl / 2,
              sz + cfg.chute_wall_h / 2), cfg.chute_color, co)

    # lamp + port trim (VISUAL ONLY; post_step recolors the lamp when lit)
    _box(stage, f"{prim_path}/lamp", (0.056, 0.004, 0.032),
         (0.0, -(hy + 0.002), 0.128), cfg.lamp_off, None)
    _box(stage, f"{prim_path}/trim", (0.056, 0.003, 0.006),
         (0.0, -(hy + 0.0015), tz + 0.004), cfg.trim_color, None)
    return root


def _spawn_flap(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC stoker flap. Origin = plate CENTRE; the hinge sits flap_len/2
    above it (the bind-time revolute joint anchors there). One thin plate."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.flap_mass, kinematic=False, ang_damp=cfg.flap_damp)
    _box(stage, f"{prim_path}/plate", (cfg.flap_w, cfg.flap_t, cfg.flap_len),
         (0.0, 0.0, 0.0), cfg.flap_color, cfg.contact_offset)
    return root


def _spawn_pellet(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC fuel briquette: one charcoal-dark cube."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.pellet_mass, kinematic=False, lin_damp=0.1, ang_damp=0.2)
    s = cfg.pellet_s
    _box(stage, f"{prim_path}/cube", (s, s, s), (0.0, 0.0, 0.0),
         cfg.pellet_color, cfg.contact_offset)
    return root


def _spawn_butter(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC butter stick: pale-yellow block that FITS through the port
    (its exclusion from the firebox is therefore a live constraint)."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.butter_mass, kinematic=False, lin_damp=0.1, ang_damp=0.2)
    _box(stage, f"{prim_path}/block", (cfg.butter_lx, cfg.butter_ly, cfg.butter_lz),
         (0.0, 0.0, 0.0), cfg.butter_color, cfg.contact_offset)
    return root


def _compound_spawner_cfg(kind: str, spawn_fn, scene_cfg: Any, mass: float,
                          kinematic: bool, lin_damp: float = 0.0,
                          ang_damp: float = 0.0) -> Any:
    """Build (once) and instantiate a RigidObjectSpawnerCfg subclass wrapping
    `spawn_fn`, carrying the scene cfg through a single `cfg` field. The
    spawner-cfg rigid_props are applied by the isaaclab clone wrapper AFTER the
    spawn fn runs, so they are the authoritative word (sleep stays OFF: sleeping
    GPU bodies freeze mid-settle and ignore velocity writes — corpus lesson)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if kind not in _SPAWNER_CACHE:

        @configclass
        class _Cfg(RigidObjectSpawnerCfg):
            func: Callable = clone(spawn_fn)
            cfg: Any = None

        _Cfg.__name__ = f"{kind.title()}SpawnerCfg"
        _SPAWNER_CACHE[kind] = _Cfg

    if kinematic:
        rp = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
    else:
        rp = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False, sleep_threshold=0.0, stabilization_threshold=0.0,
            max_depenetration_velocity=0.5,
            linear_damping=lin_damp, angular_damping=ang_damp,
            solver_position_iteration_count=32, solver_velocity_iteration_count=1)
    return _SPAWNER_CACHE[kind](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=rp,
        cfg=scene_cfg,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class StokerStoveSceneCfg(BaseCfg):
    """Config for `StokerStoveScene`. The honesty geometry is asserted in
    __post_init__: briquettes pass the port/flap with real clearance but NOT
    the grate gaps (no drop-in from above); the sill + one-way flap make the
    firebox a trap (no escape); the butter genuinely fits through the port
    (exclusion is a live constraint); a briquette's push genuinely overpowers
    the flap at every angle."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    need: int = tunable(3)                # briquettes required inside for success
    inside_margin: float = tunable(0.006)  # xy margin inside the cavity walls (m)
    inside_z_lo: float = tunable(0.012)   # inside window, z above the stove base (m)
    inside_z_hi: float = tunable(0.150)   # ... and below (must stay under the grate)
    latch_speed: float = tunable(0.10)    # calm gate for the inside latch (m/s)
    settle_speed: float = tunable(0.05)   # counted briquettes must be this calm at success

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    slot_y: float = tunable(-0.200)       # item spawn row on the front apron (m)
    slot_jitter: float = tunable(0.012)   # +/- xy jitter per item (m)
    slot_xs: tuple = tunable((-0.28, -0.14, 0.0, 0.14, 0.28))  # 5 slots, permuted

    # --- info: counter ---------------------------------------------------------------------------
    deck_h: float = info(0.140)
    counter_half_x: float = info(0.400)
    counter_half_y: float = info(0.320)
    # --- info: firebox (FIXED pose — flap joint anchor; never randomized) ------------------------
    fire_x: float = info(-0.080)
    fire_y: float = info(0.100)
    fire_hx: float = info(0.100)          # outer half-extents
    fire_hy: float = info(0.090)
    fire_h: float = info(0.170)
    wall_t: float = info(0.008)
    floor_t: float = info(0.008)
    # --- info: stoker port + flap ----------------------------------------------------------------
    port_w: float = info(0.046)           # port clear width (x)
    sill_z: float = info(0.034)           # port bottom (local z above stove base)
    port_top_z: float = info(0.080)       # port top
    flap_w: float = info(0.056)
    flap_t: float = info(0.005)
    flap_len: float = info(0.052)
    flap_mass: float = info(0.020)
    flap_damp: float = info(1.0)
    hinge_z: float = info(0.084)          # hinge height (local z)
    flap_y: float = info(-0.0785)         # flap plane (local y, 1 mm inside the wall face)
    flap_lo_deg: float = info(-0.5)       # closed stop (outward swing DENIED)
    flap_hi_deg: float = info(88.0)       # inward stop
    # --- info: chute (floor level with the sill; channel exactly port-wide) ----------------------
    chute_y0: float = info(-0.230)        # south end (local y)
    chute_y1: float = info(-0.084)        # floor tucks under the front wall
    chute_wall_y1: float = info(-0.090)   # side walls stop at the wall face
    chan_hw: float = info(0.023)          # channel half-width (= port_w / 2)
    chute_wall_t: float = info(0.008)
    chute_wall_h: float = info(0.014)     # LOW: a gripped briquette sets down into it
    # --- info: top grate -------------------------------------------------------------------------
    grate_bars: int = info(5)
    grate_bar_w: float = info(0.024)
    grate_bar_t: float = info(0.008)
    grate_pitch: float = info(0.040)      # gap = pitch - bar_w = 16 mm << briquette
    grate_zc: float = info(0.166)         # bar centre (local z; top flush with walls)
    # --- info: items -----------------------------------------------------------------------------
    n_pellets: int = info(4)
    pellet_s: float = info(0.032)
    pellet_mass: float = info(0.050)
    butter_lx: float = info(0.050)
    butter_ly: float = info(0.032)
    butter_lz: float = info(0.030)
    butter_mass: float = info(0.040)
    # --- info: misc ------------------------------------------------------------------------------
    contact_offset: float = info(0.0015)
    deck_color: tuple = info((0.42, 0.44, 0.48))
    stove_color: tuple = info((0.24, 0.23, 0.22))
    grate_color: tuple = info((0.13, 0.13, 0.14))
    chute_color: tuple = info((0.30, 0.30, 0.33))
    flap_color: tuple = info((0.55, 0.30, 0.10))
    pellet_color: tuple = info((0.09, 0.08, 0.08))
    butter_color: tuple = info((0.94, 0.87, 0.55))
    lamp_off: tuple = info((0.16, 0.05, 0.04))
    lamp_on: tuple = info((1.00, 0.45, 0.05))
    trim_color: tuple = info((0.95, 0.60, 0.10))

    # ----- derived geometry ----------------------------------------------------------------------
    @property
    def inner_hx(self) -> float:
        return self.fire_hx - self.wall_t

    @property
    def inner_hy(self) -> float:
        return self.fire_hy - self.wall_t

    @property
    def port_h(self) -> float:
        return self.port_top_z - self.sill_z

    @property
    def flap_rest_z(self) -> float:
        """Flap CENTRE (local z above the stove base) when hanging closed."""
        return self.hinge_z - self.flap_len / 2

    @property
    def grate_z_lo(self) -> float:
        """Underside of the grate bars (local z)."""
        return self.grate_zc - self.grate_bar_t / 2

    @property
    def pellet_spawn_z(self) -> float:
        return self.deck_h + self.pellet_s / 2 + 0.002

    @property
    def butter_spawn_z(self) -> float:
        return self.deck_h + self.butter_lz / 2 + 0.002

    @property
    def chute_top_z(self) -> float:
        """Chute floor top = the sill top (local z): a briquette slides
        straight over the sill through the port."""
        return self.sill_z

    def __post_init__(self) -> None:
        # Honesty-by-construction asserts (the geometric claims the task rests on).
        g = 9.81
        # -- the port admits a briquette (and the butter!) with real clearance --
        assert self.port_w - self.pellet_s >= 0.010, \
            "the port must pass a briquette with >= 10 mm total clearance"
        assert self.port_h - self.pellet_s >= 0.010, \
            "the port height must clear a briquette sliding on the sill"
        assert 2 * self.chan_hw >= self.pellet_s + 0.010, \
            "the chute channel must guide a briquette without wedging"
        assert abs(2 * self.chan_hw - self.port_w) < 1e-9, \
            "the chute channel must be exactly port-wide (it aims the push)"
        assert self.butter_ly < self.port_w - 0.008 and \
            self.butter_lz < self.port_h - 0.008, \
            "the butter MUST fit through the port (exclusion is a live constraint)"
        # -- the flap covers the port and genuinely yields to a briquette --
        assert self.flap_w >= self.port_w + 0.008, \
            "the flap must overhang the port on both sides"
        assert self.hinge_z - self.flap_len <= self.sill_z, \
            "the flap curtain must reach down past the sill"
        assert self.hinge_z >= self.port_top_z + 0.002, \
            "the hinge must tuck above the port opening"
        assert self.flap_y - self.flap_t / 2 > -(self.fire_hy - self.wall_t) + 1e-4, \
            "the closed flap must hang clear of the front wall's inner face"
        need_deg = math.degrees(math.acos(
            (self.hinge_z - (self.sill_z + self.pellet_s)) / self.flap_len))
        assert need_deg <= self.flap_hi_deg - 5.0, \
            "the flap must open far enough for a briquette to pass under it"
        assert self.flap_mass * g / 2 < 0.35 * self.pellet_mass * g, \
            "holding the flap open must cost a small fraction of a briquette's weight"
        # -- the firebox is a trap: no drop-in from above, no way back out --
        gap = self.grate_pitch - self.grate_bar_w
        assert gap + 0.008 < self.pellet_s, \
            "the grate gaps must deny a briquette with margin"
        assert gap + 0.008 < min(self.butter_lx, self.butter_ly, self.butter_lz), \
            "the grate gaps must deny the butter too"
        assert (self.grate_bars - 1) * self.grate_pitch + self.grate_bar_w \
            >= 2 * self.inner_hx - 1e-9, \
            "the grate must span the whole cavity (no edge hole)"
        assert self.sill_z - self.floor_t >= 0.020, \
            "the sill must sit >= 20 mm above the interior floor (escape denial)"
        assert self.flap_lo_deg > -2.0, \
            "the closed stop must block outward swing almost immediately"
        # -- the judged inside window sits strictly inside the cavity --
        assert self.inside_z_lo > self.floor_t, \
            "the inside window must start above the interior floor top"
        assert self.inside_z_hi < self.grate_z_lo - 0.008, \
            "the inside window must end below the grate"
        assert self.inside_margin < self.pellet_s / 2, \
            "a briquette resting against a wall must still be judged inside"
        # -- rubric sanity --
        assert 1 <= self.need <= self.n_pellets - 1, \
            "success must be reachable with a briquette to spare"
        # -- layout: slots clear of the chute, each other, and the counter edge --
        half_diag = max(self.pellet_s * math.sqrt(2) / 2,
                        math.hypot(self.butter_lx, self.butter_ly) / 2)
        chute_s_end = self.fire_y + self.chute_y0
        assert self.slot_y + self.slot_jitter + half_diag < chute_s_end - 0.004, \
            "the spawn row must sit clear (south) of the chute"
        xs = sorted(self.slot_xs)
        for a, b in zip(xs, xs[1:]):
            assert b - a >= 2 * (half_diag + self.slot_jitter) + 0.010, \
                "adjacent slots must never collide at any jitter/yaw"
        assert max(abs(x) for x in xs) + self.slot_jitter + half_diag \
            < self.counter_half_x - 0.020
        assert abs(self.slot_y) + self.slot_jitter + half_diag \
            < self.counter_half_y - 0.020
        assert len(self.slot_xs) == self.n_pellets + 1, \
            "one slot per item (4 briquettes + butter)"
        # -- stove + chute inside the counter --
        assert abs(self.fire_x) + self.fire_hx < self.counter_half_x - 0.020
        assert self.fire_y + self.fire_hy < self.counter_half_y - 0.020
        assert chute_s_end > -self.counter_half_y + 0.020


# ----- small quaternion helper (wxyz, torch, batched) -------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("stoker_stove")
class StokerStoveScene(BaseScene):
    cfg: StokerStoveSceneCfg

    def __init__(self, cfg: StokerStoveSceneCfg | None = None) -> None:
        super().__init__(cfg or StokerStoveSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
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
            "counter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Counter",
                spawn=_compound_spawner_cfg("counter", _spawn_counter, c, 40.0,
                                            kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            # Stove authored at its FIXED pose; the bind-time flap joint anchors
            # here and the stove is NEVER moved (kinematic joint anchors are
            # world-fixed).
            "stove": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stove",
                spawn=_compound_spawner_cfg("stove", _spawn_stove, c, 12.0,
                                            kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fire_x, c.fire_y, c.deck_h)),
            ),
            "flap": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Flap",
                spawn=_compound_spawner_cfg("flap", _spawn_flap, c, c.flap_mass,
                                            kinematic=False, ang_damp=c.flap_damp),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fire_x, c.fire_y + c.flap_y, c.deck_h + c.flap_rest_z)),
            ),
            "butter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Butter",
                spawn=_compound_spawner_cfg("butter", _spawn_butter, c,
                                            c.butter_mass, kinematic=False,
                                            lin_damp=0.1, ang_damp=0.2),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_xs[4], c.slot_y, c.butter_spawn_z)),
            ),
        }
        for k in range(c.n_pellets):
            out[f"pellet{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pellet" + str(k),
                spawn=_compound_spawner_cfg("pellet", _spawn_pellet, c,
                                            c.pellet_mass, kinematic=False,
                                            lin_damp=0.1, ang_damp=0.2),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_xs[k], c.slot_y, c.pellet_spawn_z)),
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.counter: RigidObject = env.iscene["counter"]
        self.stove: RigidObject = env.iscene["stove"]
        self.flap: RigidObject = env.iscene["flap"]
        self.butter: RigidObject = env.iscene["butter"]
        self.pellets: list[RigidObject] = [
            env.iscene[f"pellet{k}"] for k in range(self.cfg.n_pellets)]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # latch: briquette was ever inside-and-calm (counting also requires
        # CURRENTLY inside, so the latch can never over-count)
        self._latched = torch.zeros(n, self.cfg.n_pellets, dtype=torch.bool, device=dev)
        self._lamp_shown: list[bool | None] = [None] * n
        self._lamp_prims = None  # lazily grabbed in _refresh_lamps
        self._author_joints()

    def _author_joints(self) -> None:
        """Per-env flap hinge, authored ONCE at bind time against the AUTHORED
        poses: a revolute X joint anchored on the KINEMATIC stove (its joint
        anchors are world-fixed, so the stove must never move). The joint
        LIMITS are the one-way stops — lower (-0.5 deg) is the closed stop
        that DENIES outward swing, upper (+88 deg) the inward stop. Joint
        collision filtering only disables the stove<->flap pair, so briquettes
        still contact both."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        c = self.cfg
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/flap_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Stove"])
            j.CreateBody1Rel().SetTargets([f"{base}/Flap"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, float(c.flap_y), float(c.hinge_z)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, float(c.flap_len / 2)))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(float(c.flap_lo_deg))
            j.CreateUpperLimitAttr(float(c.flap_hi_deg))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: counter and stove re-asserted at their FIXED poses
        (the stove anchors the flap joint — kinematic joint anchors are
        world-fixed), flap re-closed, and the 5 items (4 briquettes + butter)
        PERMUTED over the 5 apron slots with xy jitter and free yaw (argsort
        of torch.rand — the FIRST torch.randint after manual_seed is
        near-constant), latches and lamp cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, xy, z, quat=None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3:7] = quat if quat is not None else torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=dev).expand(m, 4)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        write(self.counter, torch.zeros(m, 2, device=dev),
              torch.zeros(m, device=dev))
        stove_xy = torch.tensor([c.fire_x, c.fire_y], device=dev).expand(m, 2)
        write(self.stove, stove_xy, c.deck_h)
        flap_xy = torch.tensor([c.fire_x, c.fire_y + c.flap_y], device=dev).expand(m, 2)
        write(self.flap, flap_xy, c.deck_h + c.flap_rest_z)

        # --- items: permutation over the 5 slots + jitter + free yaw ---
        slot_xs = torch.tensor(c.slot_xs, device=dev)
        perm = torch.argsort(torch.rand(m, len(c.slot_xs), device=dev), dim=1)
        bodies = [*self.pellets, self.butter]
        z_of = [c.pellet_spawn_z] * c.n_pellets + [c.butter_spawn_z]
        for i, body in enumerate(bodies):
            sx = slot_xs[perm[:, i]] + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            sy = c.slot_y + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            yaw = torch.rand(m, device=dev) * 2 * math.pi
            write(body, torch.stack([sx, sy], dim=-1), z_of[i], _qz(yaw))

        # --- clear latches + lamp ---
        self._latched[env_ids] = False
        for e in env_ids.tolist():
            self._lamp_shown[e] = None  # force a lamp refresh next post_step

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "counter": self.counter.data.root_state_w[env_ids].clone(),
            "stove": self.stove.data.root_state_w[env_ids].clone(),
            "flap": self.flap.data.root_state_w[env_ids].clone(),
            "butter": self.butter.data.root_state_w[env_ids].clone(),
            "latched": self._latched[env_ids].clone(),
        }
        for k, p in enumerate(self.pellets):
            out[f"pellet{k}"] = p.data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for name in ("counter", "stove", "flap", "butter"):
            getattr(self, name).write_root_state_to_sim(state[name], env_ids)
        for k, p in enumerate(self.pellets):
            p.write_root_state_to_sim(state[f"pellet{k}"], env_ids)
        self._latched[env_ids] = state["latched"]
        for e in env_ids.tolist():
            self._lamp_shown[e] = None

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        gap = c.grate_pitch - c.grate_bar_w
        return (
            f"A steel KITCHEN BENCH ({2 * c.counter_half_x * 100:.0f} x "
            f"{2 * c.counter_half_y * 100:.0f} cm, {c.deck_h * 100:.0f} cm tall). "
            f"On its back-left area stands a cold PELLET STOVE: a dark iron "
            f"firebox ({2 * c.fire_hx * 100:.0f} x {2 * c.fire_hy * 100:.0f} x "
            f"{c.fire_h * 100:.0f} cm) whose top is a slotted GRATE — the "
            f"{gap * 1000:.0f} mm gaps between its bars are far too narrow to drop "
            f"fuel through. The only way in is the STOKER PORT low in the front "
            f"wall: a {c.port_w * 1000:.0f} mm square opening (sill "
            f"{c.sill_z * 1000:.0f} mm up) covered from the inside by a copper "
            f"one-way FLAP hinged at its top edge. The flap swings INWARD freely "
            f"under a gentle push but its hinge blocks any outward swing, and the "
            f"interior floor lies {(c.sill_z - c.floor_t) * 1000:.0f} mm below the "
            f"sill — whatever goes in stays in. A shallow loading CHUTE runs from "
            f"the front of the bench straight into the port: a "
            f"{2 * c.chan_hw * 1000:.0f} mm wide channel between two "
            f"{c.chute_wall_h * 1000:.0f} mm lips, its floor level with the sill. "
            f"An unlit lamp sits above the port. Along the front apron lie FIVE "
            f"items in a shuffled row (order, exact position and yaw are "
            f"randomized per episode): four charcoal-dark FUEL BRIQUETTES "
            f"({c.pellet_s * 1000:.0f} mm cubes) and one pale-yellow BUTTER STICK "
            f"({c.butter_lx * 1000:.0f} x {c.butter_ly * 1000:.0f} x "
            f"{c.butter_lz * 1000:.0f} mm) — the butter also fits through the "
            f"port, and fuelling is SPOILED if it ends up inside.\n"
            f"Goal: turn the stove on by fuelling it. Set a briquette into the "
            f"chute channel and push it along the chute through the flap until it "
            f"drops into the firebox; repeat until at least {c.need} briquettes "
            f"are inside. Keep the butter out of the firebox. The lamp turns "
            f"orange when the stove is properly fuelled ({c.need}+ briquettes in, "
            f"no butter). Success: at least {c.need} briquettes resting inside "
            f"the firebox, the butter NOT inside, everything settled."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Turn on the pellet stove by fuelling it: push three of the dark "
            "fuel briquettes along the chute through the one-way stoker flap "
            "into the firebox. Leave the butter stick out."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _rel(self, body) -> torch.Tensor:
        """(N,3) body origin relative to the stove base centre. The stove is
        kinematic, never rotates and is never randomized, so world-axis
        deltas ARE its body frame."""
        return body.data.root_pos_w - self.stove.data.root_pos_w

    def _inside_pos(self, rel: torch.Tensor) -> torch.Tensor:
        """(N,) bool: a point (stove-local) is inside the judged firebox
        window — strictly within the cavity walls, above the interior floor,
        below the grate."""
        c = self.cfg
        x_ok = rel[:, 0].abs() < c.inner_hx - c.inside_margin
        y_ok = rel[:, 1].abs() < c.inner_hy - c.inside_margin
        z_ok = (rel[:, 2] > c.inside_z_lo) & (rel[:, 2] < c.inside_z_hi)
        return x_ok & y_ok & z_ok

    def pellets_inside(self) -> torch.Tensor:
        """(N, n_pellets) bool: briquette k currently inside the firebox."""
        return torch.stack(
            [self._inside_pos(self._rel(p)) for p in self.pellets], dim=1)

    def pellet_speeds(self) -> torch.Tensor:
        """(N, n_pellets) briquette |lin vel|."""
        return torch.stack(
            [p.data.root_lin_vel_w.norm(dim=-1) for p in self.pellets], dim=1)

    def butter_inside(self) -> torch.Tensor:
        """(N,) bool: the butter is currently inside the firebox."""
        return self._inside_pos(self._rel(self.butter))

    def counted(self) -> torch.Tensor:
        """(N, n_pellets) bool: latched inside-and-calm AND currently inside."""
        return self._latched & self.pellets_inside()

    def count(self) -> torch.Tensor:
        """(N,) long: briquettes counted as fed."""
        return self.counted().sum(dim=1)

    def flap_angle(self) -> torch.Tensor:
        """(N,) flap hinge angle in radians (+ = inward). The flap only ever
        rotates about the hinge X axis, so the quat is (cos a/2, sin a/2, 0, 0)."""
        q = self.flap.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 1], q[:, 0])

    def _update_latches(self) -> None:
        self._latched |= self.pellets_inside() & \
            (self.pellet_speeds() < self.cfg.latch_speed)

    def _refresh_lamps(self) -> None:
        from pxr import Gf

        if self._lamp_prims is None:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            self._lamp_prims = [
                UsdGeom.Cube(stage.GetPrimAtPath(f"/World/envs/env_{i}/Stove/lamp"))
                for i in range(self.env.num_envs)]
        c = self.cfg
        lit = (self.count() >= c.need) & ~self.butter_inside()
        for e in range(self.env.num_envs):
            key = bool(lit[e])
            if key != self._lamp_shown[e]:
                self._lamp_shown[e] = key
                col = c.lamp_on if key else c.lamp_off
                self._lamp_prims[e].GetDisplayColorAttr().Set([Gf.Vec3f(*col)])

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Once per physics substep: advance the inside latches and refresh
        the lamps. The scene owns NO forces — the flap is pure passive
        gravity against its joint limits."""
        self._update_latches()
        self._refresh_lamps()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: at least `need` briquettes counted inside the firebox
        (latched inside-and-calm AND currently inside — a physical
        containment outcome; feeding is irreversible by construction), the
        butter NOT inside, every counted briquette calm, all bodies finite."""
        self._update_latches()
        c = self.cfg
        cnt = self.counted()
        calm = ((~cnt) | (self.pellet_speeds() < c.settle_speed)).all(dim=1)
        finite = torch.isfinite(self.flap.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.butter.data.root_pos_w).all(dim=-1)
        for p in self.pellets:
            finite &= torch.isfinite(p.data.root_pos_w).all(dim=-1)
        return (cnt.sum(dim=1) >= c.need) & ~self.butter_inside() & calm & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25 per briquette fed (up to 3 -> 0.75; ~0
        for doing nothing; monotone along the solve because fed briquettes
        cannot come back out), x0.2 while the butter spoils the firebox —
        and exactly 1.0 iff success() holds live."""
        self._update_latches()
        k = self.count().clamp(max=3).float()
        base = 0.25 * k
        base = torch.where(self.butter_inside(), base * 0.2, base)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="stoker_stove", robot="null"))
