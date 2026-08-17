"""SpindleRollScene — thread the free spindle through the roll's core, then hang the
assembly in the bracket's two open slots (sim_gen task `put_toilet_roll_on_stand_i188`).

Derived from rlbench/put_toilet_roll_on_stand, but STRATEGICALLY different: the seed is
pick-and-insert onto a FIXTURE — grab the roll and slide its core sideways onto a fixed
cantilever peg; one object, one insertion, the stand does all the holding. Here the
stand CANNOT accept the bare roll at all: the bracket is two upright slotted plates,
whose open-top slots (22 mm) are far too narrow for the roll (66 mm) and whose solid
plates block all axial access. The task is a two-stage ASSEMBLY:

  1. THREAD — pass a free steel SPINDLE (a loose rod with two ball ends) axially
     through the roll's hollow core. The roll is braced in a V-CRADLE against a
     notched backstop so the axial push does not just shove it away; the ball ends
     (26 mm) pass the core (40 mm) but are the load-bearing tolerance everywhere else.
  2. HANG — carry the threaded assembly to the bracket and lower the spindle's two
     exposed end segments into the two open-top slots (funnel mouths guide them). The
     ball ends are WIDER than the slots, so they land OUTSIDE the plates and retain
     the spindle axially; the roll ends up SUSPENDED on the spindle between the
     plates, clear of the base — the load path is roll -> spindle -> both seats.

A solver needs a different PLAN (assemble a carrier through the object, then mount the
compound — the roll never touches the stand) and different CODE (a through-the-core
predicate: rod line inside the core with protrusion beyond BOTH faces; a bilateral
two-seat predicate; a hang-height band), not different constants. Execution order is
geometrically forced: with the spindle already seated, the plates block the only axial
approach to it, so threading can only happen off the stand (asserted by construction).

success(): threaded (spindle line within `thread_radial_tol` of the roll axis, aligned,
protruding > `protrude_min` beyond both faces) AND seated (spindle in the rack frame:
centered on the slot line within `seat_yz_tol` in y and z at `SEAT_Z`, aligned with the
slot axis, axially centered) AND hanging (roll center in the hang band under the
spindle, its lowest point clear of the slab) AND settled (sustained stillness of BOTH
free bodies). All judged on settled poses; nothing is welded, pinned, or held.

score() is graded and latched (credit never evaporates): 0.20 once the roll has ever
been seated in the cradle + 0.25 once the spindle has ever been threaded through the
core + 0.15 once the threaded assembly has ever been lifted to `lift_z` (cap 0.60);
1.0 iff success(). The null policy scores ~0 (roll and spindle spawn lying on the
ground away from both fixtures; `lift_z` is above every ground/cradle/seat rest pose).

Per-episode randomization (readback-verified in smoke): cradle pose (xy jitter + yaw),
rack pose (xy jitter + yaw), roll and spindle ground poses (jitter + free yaw), and
which SIDE of the workspace holds cradle vs rack. Assets are fully procedural compound
spawners (a 12-sided box tube for the roll, capsule + ball ends for the spindle, box
fixtures — one rigid body each; decorations authored idempotently). Heavy imports
(isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- shared geometry constants (spawners + cfg assertions + solve/smoke) ---------------------
ROLL_N = 12                      # tube facets
CORE_R = 0.020                   # core inner inradius (the threading aperture)
WALL_T = 0.012                   # tube wall thickness
OUT_FLAT = CORE_R + WALL_T       # outer inradius (grasp width across flats = 64 mm)
OUT_CORNER = OUT_FLAT / math.cos(math.pi / ROLL_N)  # outer corner radius (~33.1 mm)
ROLL_HALF_W = 0.045              # roll half width (width 90 mm)

ROD_R = 0.007                    # spindle shaft radius (14 mm rod)
ROD_SHAFT_HALF = 0.100           # capsule cylinder half-height (shaft spans +-100 mm)
CAP_R = 0.013                    # ball-end radius (26 mm ball)
CAP_Z = 0.098                    # ball-end centers at local z = +-CAP_Z
ROD_TIP = CAP_Z + CAP_R          # spindle end-to-end half-length (111 mm)

CRADLE_REST_Z = 0.047            # roll axis height resting in the 90-deg V (apex ~1 mm)
BACKSTOP_X = 0.045               # backstop inner face (cradle local +x = thread axis)

SLOT_W = 0.022                   # bracket slot width (shaft passes, ball ends do not)
PLATE_X_IN = 0.060               # bracket plate inner faces at +-PLATE_X_IN
PLATE_X_OUT = 0.070              # bracket plate outer faces
SEAT_Z = 0.085                   # seated spindle AXIS height (seat surface at 78 mm)
SLAB_TOP = 0.020                 # bracket base slab top
FUNNEL_DEG = 35.0                # funnel wedge angle from vertical
HANG_DROP = CORE_R - ROD_R       # hanging roll center sits this far below the spindle axis


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


def _material(stage, path: str, static: float = 0.6, dynamic: float = 0.5):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, color, contact_offset: float, material) -> None:
    from pxr import Gf, PhysxSchema, UsdPhysics, UsdShade

    prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float, material=None,
         quat=None) -> None:
    """One box child prim (translate -> orient -> scale, authored exactly once — the
    duplicate-xformOp trap is avoided by never re-authoring an existing prim's ops)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if quat is not None:
        w, x, y, z = (float(v) for v in quat)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    _collide(seg, color, contact_offset, material)


def _capsule(stage, path: str, radius: float, height: float, center, quat, color,
             contact_offset: float, material=None) -> None:
    from pxr import UsdGeom

    seg = UsdGeom.Capsule.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    _apply_xform(seg, center, quat)
    _collide(seg, color, contact_offset, material)


def _sphere(stage, path: str, radius: float, center, color, contact_offset: float,
            material=None) -> None:
    from pxr import UsdGeom

    seg = UsdGeom.Sphere.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    _apply_xform(seg, center, None)
    _collide(seg, color, contact_offset, material)


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float,
                kinematic: bool = False):
    """One rigid-body root Xform with the physics armor (zero sleep/stabilization
    thresholds: a sleeping body silently ignores the applied wrenches the solve and
    smoke probes depend on; velocity iterations 4 kill the capsule-on-box creep)."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return root, pxrb


def _spawn_roll(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The paper roll: a 12-sided box TUBE (real hollow core — the threading aperture
    is enforced by collision, not bookkeeping). Local axis +z; origin = tube center."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.15)
    pxrb.CreateAngularDampingAttr(2.00)  # arrests the roll-on-rod pendulum limit cycle
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    white = (0.93, 0.92, 0.88)
    w = 2.0 * OUT_FLAT * math.tan(math.pi / ROLL_N) + 0.001  # tangential closure
    r_mid = CORE_R + WALL_T / 2
    for i in range(ROLL_N):
        ang = 2.0 * math.pi * i / ROLL_N
        cx, cy = r_mid * math.cos(ang), r_mid * math.sin(ang)
        # box radial axis = local x rotated by ang about z
        q = (math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2))
        _box(stage, f"{prim_path}/wall_{i}", (WALL_T, w, 2 * ROLL_HALF_W),
             (cx, cy, 0.0), white, co, material=mat, quat=q)
    return root


def _spawn_spindle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The spindle: steel shaft capsule (local axis +z) + two dark BALL ends. The
    balls pass the core (26 < 40 mm) but not the slots (26 > 22 mm)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.15)
    pxrb.CreateAngularDampingAttr(2.00)  # arrests the axle-spin phantom on the seats
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    steel, dark = (0.72, 0.74, 0.78), (0.12, 0.12, 0.14)
    _capsule(stage, f"{prim_path}/shaft", ROD_R, 2 * ROD_SHAFT_HALF, (0.0, 0.0, 0.0),
             (1.0, 0.0, 0.0, 0.0), steel, co, material=mat)
    for s, nm in ((+1.0, "ball_p"), (-1.0, "ball_n")):
        _sphere(stage, f"{prim_path}/{nm}", CAP_R, (0.0, 0.0, s * CAP_Z), dark, co,
                material=mat)
    return root


def _spawn_cradle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC loading cradle: base pad, 90-deg V-block, and a notched BACKSTOP
    plate at local +x (the thread axis). The notch (38 x 54 mm, bottom-center) lets
    the spindle's leading ball exit — at its physical riding height (ball on the core
    bottom, top ~0.053) — while the plate braces the roll's face annulus against the
    axial threading push. Origin = footprint center at ground level."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 10.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    blue, dark = (0.28, 0.38, 0.60), (0.20, 0.24, 0.34)
    _box(stage, f"{prim_path}/base", (0.130, 0.150, 0.008), (0.010, 0.0, 0.004),
         dark, co, material=mat)
    # V plates: top surfaces meet at the apex line (y=0, z=0.001), +-45 deg
    c45 = math.cos(math.pi / 4)
    for s, nm in ((+1.0, "v_py"), (-1.0, "v_ny")):
        # top surface: from apex outward along d=(0, s*c45, c45); box center is the
        # surface center minus half-thickness along the surface normal (0,-s*c45,c45)
        cy = s * (0.040 * c45 + 0.006 * c45)
        cz = 0.001 + 0.040 * c45 - 0.006 * c45
        q = (math.cos(s * math.pi / 8), math.sin(s * math.pi / 8), 0.0, 0.0)
        _box(stage, f"{prim_path}/{nm}", (0.060, 0.080, 0.012), (0.0, cy, cz), blue,
             co, material=mat, quat=q)
    # backstop: lintel above the notch + two side blocks beside it
    _box(stage, f"{prim_path}/stop_top", (0.010, 0.140, 0.054), (0.050, 0.0, 0.085),
         blue, co, material=mat)
    for s in (+1.0, -1.0):
        _box(stage, f"{prim_path}/stop_side_{'p' if s > 0 else 'n'}",
             (0.010, 0.051, 0.054), (0.050, s * 0.0445, 0.031), blue, co,
             material=mat)
    return root


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC hanging bracket: base slab + two upright slotted plates at local
    x = +-0.065 (spindle axis = local +x). Each plate: two columns flanking a 22 mm
    open-top slot, a seat block below the slot topped by a 90-deg V-GROOVE bearing
    (the shaft nests on two contact lines at axis height SEAT_Z — a flat seat lets
    the capsule spin/creep in place, the phantom limit cycle), and two funnel wedges
    above the slot mouth. Origin = footprint center at ground level."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 10.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    green, dark = (0.22, 0.42, 0.30), (0.16, 0.20, 0.18)
    _box(stage, f"{prim_path}/slab", (0.240, 0.160, 0.020), (0.0, 0.0, 0.010), dark,
         co, material=mat)
    th = math.radians(FUNNEL_DEG)
    for sx, px in ((+1.0, "p"), (-1.0, "n")):
        x = sx * 0.065
        for sy, py in ((+1.0, "p"), (-1.0, "n")):
            _box(stage, f"{prim_path}/col_{px}{py}", (0.010, 0.044, 0.098),
                 (x, sy * 0.033, 0.069), green, co, material=mat)
            # funnel wedge: guide plane through the slot mouth edge (y=+-0.011,
            # z=0.118), leaning outward by FUNNEL_DEG from vertical
            d = (0.0, sy * math.sin(th), math.cos(th))
            nrm = (0.0, sy * math.cos(th), -math.sin(th))
            cyz = (0.011 * sy + 0.030 * d[1] + 0.006 * nrm[1],
                   0.118 + 0.030 * d[2] + 0.006 * nrm[2])
            q = (math.cos(sy * -th / 2), math.sin(sy * -th / 2), 0.0, 0.0)
            _box(stage, f"{prim_path}/fun_{px}{py}", (0.010, 0.012, 0.060),
                 (x, cyz[0], cyz[1]), green, co, material=mat, quat=q)
        _box(stage, f"{prim_path}/seat_{px}", (0.010, SLOT_W, 0.070), (x, 0.0, 0.035),
             green, co, material=mat)
        # V-groove bearing: two +-45-deg plates whose TOP surfaces meet at the apex
        # line (y=0, z = SEAT_Z - ROD_R*sqrt(2)); the nested shaft axis sits at SEAT_Z
        c45v = math.cos(math.pi / 4)
        apex = SEAT_Z - ROD_R * math.sqrt(2.0)
        for sy, py in ((+1.0, "p"), (-1.0, "n")):
            vy = sy * (0.010 + 0.003) * c45v
            vz = apex + (0.010 - 0.003) * c45v
            q = (math.cos(sy * math.pi / 8), math.sin(sy * math.pi / 8), 0.0, 0.0)
            _box(stage, f"{prim_path}/vseat_{px}{py}", (0.010, 0.020, 0.006),
                 (x, vy, vz), green, co, material=mat, quat=q)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "roll" not in _SPAWNER_CACHE:

        @configclass
        class RollSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_roll)
            mass: float = 0.10
            contact_offset: float = 0.002

        @configclass
        class SpindleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_spindle)
            mass: float = 0.08
            contact_offset: float = 0.002

        @configclass
        class CradleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cradle)
            contact_offset: float = 0.002

        @configclass
        class RackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(roll=RollSpawnerCfg, spindle=SpindleSpawnerCfg,
                              cradle=CradleSpawnerCfg, rack=RackSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SpindleRollSceneCfg(BaseCfg):
    """Config for `SpindleRollScene`. The rubric honesty is asserted in
    `__post_init__`: a hanging roll reads inside every tolerance by construction,
    while each wrong outcome (rod alongside the roll, roll perched on the rod, the
    assembly resting on the slab, partial threading) reads outside with margin."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    thread_radial_tol: float = tunable(0.016)  # rod line offset from the roll axis (m)
    thread_align_max_deg: float = tunable(15.0)  # rod axis vs roll axis
    protrude_min: float = tunable(0.010)  # rod tip beyond EACH roll face (m)
    seat_yz_tol: float = tunable(0.008)  # rod center |y| and |z-SEAT_Z| in the rack frame
    seat_x_max: float = tunable(0.020)  # rod axial centering in the rack frame
    seat_align_max_deg: float = tunable(10.0)  # rod axis vs the rack slot axis
    hang_x_max: float = tunable(0.025)  # roll axial position between the plates
    hang_y_max: float = tunable(0.020)  # roll lateral offset under the spindle
    hang_band: float = tunable(0.009)  # |roll z - (SEAT_Z - HANG_DROP)| (m)
    clear_min: float = tunable(0.008)  # roll lowest point above the slab top (m)
    settle_lin: float = tunable(0.04)  # max |lin vel| (roll AND spindle) when judging
    settle_ang: float = tunable(0.50)  # max |ang vel| when judging (rad/s)
    settle_steps: int = tunable(30)  # substeps of SUSTAINED stillness (0.25 s at 120 Hz)
    # progress-latch geometry
    cradle_tol: float = tunable(0.035)  # roll center to the cradle rest point
    cradle_align_max_deg: float = tunable(25.0)  # roll axis vs the cradle thread axis
    lift_z: float = tunable(0.130)  # spindle height that latches "carried" (threaded)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    fixture_jitter: float = tunable(0.04)  # +-xy jitter of cradle and rack
    cradle_yaw_deg: float = tunable(40.0)  # +-yaw of the cradle
    rack_yaw_deg: float = tunable(20.0)  # +-yaw of the rack
    item_jitter: float = tunable(0.05)  # +-xy jitter of roll and spindle spawns
    side_swap: bool = tunable(True)  # randomly swap which side cradle vs rack sit on

    # --- info: layout (env frame; the Franka base pose argument lives in TASK.md) ------------
    cradle_pos: tuple = info((0.32, -0.30))  # nominal cradle center (side s=+1)
    rack_pos: tuple = info((0.44, 0.30))  # nominal rack center (side s=+1)
    roll_spawn: tuple = info((0.14, -0.12))  # roll ground spawn (side s=+1)
    rod_spawn: tuple = info((0.14, 0.12))  # spindle ground spawn (side s=+1)
    roll_mass: float = info(0.10)
    rod_mass: float = info(0.08)
    contact_offset: float = info(0.002)
    seat_z: float = info(SEAT_Z)
    hang_z: float = info(SEAT_Z - HANG_DROP)

    def __post_init__(self) -> None:
        c = self
        # -- threading aperture: the ball ends pass the core with real clearance --
        assert CAP_R + 0.004 < CORE_R, "ball ends must pass the core"
        # -- threaded radial tolerance is honest by construction --
        assert HANG_DROP + 0.002 <= c.thread_radial_tol, \
            "a hanging roll (rod on the core top) must read inside thread_radial_tol"
        assert c.thread_radial_tol < CORE_R - ROD_R + WALL_T / 2, \
            "thread_radial_tol must only accept a rod INSIDE the core"
        assert OUT_FLAT + ROD_R > c.thread_radial_tol + 0.020, \
            "a rod lying ALONGSIDE the roll must be rejected with margin"
        # -- protrusion clause honest: a centered full thread protrudes far past it --
        assert ROD_TIP - ROLL_HALF_W > c.protrude_min + 0.020, \
            "a fully threaded spindle must protrude well past protrude_min"
        # -- the slots take the shaft, never the balls; balls land outside the plates --
        assert 2 * CAP_R > SLOT_W + 0.003, "ball ends must NOT pass the slots"
        assert SLOT_W > 2 * ROD_R + 0.006, "the shaft must pass the slots with room"
        assert (CAP_Z - CAP_R) - PLATE_X_OUT > 0.012, \
            "axial drop-in margin: balls must clear the plate outer faces"
        assert PLATE_X_IN - ROLL_HALF_W > 0.012, "the roll must fit between the plates"
        # -- the bare roll cannot enter the stand (the anti-seed clause) --
        assert 2 * OUT_FLAT > SLOT_W + 0.030, "the roll can never enter a slot"
        # -- a seated assembly hangs the roll clear of the slab --
        assert SEAT_Z - HANG_DROP - OUT_CORNER - SLAB_TOP > c.clear_min + 0.008, \
            "a hanging roll must clear the slab with margin"
        # -- the hang band excludes a roll resting ON the slab --
        rest_on_slab = SLAB_TOP + OUT_FLAT  # (to OUT_CORNER; use the closer flat rest)
        assert abs(rest_on_slab - c.hang_z) > c.hang_band + 0.008, \
            "a roll resting on the slab must read outside the hang band"
        # -- an assembly resting on the slab reads far outside the seat tolerance --
        rod_on_slab = SLAB_TOP + OUT_FLAT - HANG_DROP
        assert SEAT_Z - rod_on_slab > c.seat_yz_tol + 0.030, \
            "an assembly resting on the slab must be rejected by seating"
        # -- backstop notch passes the leading ball with margin (notch: |y|<0.019,
        # z<0.058; the ball transits riding the CORE BOTTOM, center at axis-7 mm) --
        ball_ride_z = CRADLE_REST_Z - (CORE_R - CAP_R)  # ball resting on the core bottom
        assert ball_ride_z + CAP_R < 0.058 - 0.004 and ball_ride_z - CAP_R > 0.004, \
            "the leading ball must pass the backstop notch vertically"
        assert 0.019 > CAP_R + 0.003, "the notch must pass the ball laterally"
        # -- progress latches cannot fire at rest --
        assert c.lift_z > SEAT_Z + 0.020, \
            "a directly-constructed seated pose must not latch 'carried'"
        assert c.lift_z > CRADLE_REST_Z + 0.050 and c.lift_z > OUT_CORNER + 0.050, \
            "no ground/cradle rest pose may latch 'carried'"
        # -- stillness must be sustained past a swing turning point --
        assert c.settle_steps >= 12, "settled() must out-last a turning point"
        # -- layout: spawns clear of fixtures even at worst-case jitter --
        for item, fix, r_item, r_fix in (
                (c.roll_spawn, c.cradle_pos, 0.05, 0.10),
                (c.roll_spawn, c.rack_pos, 0.05, 0.14),
                (c.rod_spawn, c.cradle_pos, 0.12, 0.10),
                (c.rod_spawn, c.rack_pos, 0.12, 0.14)):
            d = math.hypot(item[0] - fix[0], item[1] - fix[1])
            assert d > c.item_jitter + c.fixture_jitter + r_item + r_fix - 0.02, \
                f"spawn strip {item} may overlap fixture {fix}"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("spindle_roll")
class SpindleRollScene(BaseScene):
    cfg: SpindleRollSceneCfg

    def __init__(self, cfg: SpindleRollSceneCfg | None = None) -> None:
        super().__init__(cfg or SpindleRollSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "cradle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cradle",
                spawn=sp["cradle"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cradle_pos[0], c.cradle_pos[1], 0.0)),
            ),
            "rack": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=sp["rack"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_pos[0], c.rack_pos[1], 0.0)),
            ),
            "roll": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Roll",
                spawn=sp["roll"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.roll_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.roll_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.roll_spawn[0], c.roll_spawn[1], OUT_FLAT + 0.002),
                    rot=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)),
            ),
            "spindle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Spindle",
                spawn=sp["spindle"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.rod_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.rod_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rod_spawn[0], c.rod_spawn[1], CAP_R + 0.002),
                    rot=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)),
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
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.cradle: RigidObject = env.iscene["cradle"]
        self.rack: RigidObject = env.iscene["rack"]
        self.roll: RigidObject = env.iscene["roll"]
        self.spindle: RigidObject = env.iscene["spindle"]
        self.env_origins = env.iscene.env_origins
        # progress latches (post_step)
        self.cradled_latch = torch.zeros(n, device=dev)
        self.thread_latch = torch.zeros(n, device=dev)
        self.carry_latch = torch.zeros(n, device=dev)
        # sustained-stillness counter (consecutive still substeps, reset by motion)
        self.still_count = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: cradle and rack re-placed (side sample + jitter + yaw), roll
        and spindle lying on the ground (jitter + free yaw), latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def qz(yaw: torch.Tensor) -> torch.Tensor:
            half = yaw / 2
            z = torch.zeros_like(half)
            return torch.stack([torch.cos(half), z, z, torch.sin(half)], dim=-1)

        def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            aw, ax, ay, az = a.unbind(-1)
            bw, bx, by, bz = b.unbind(-1)
            return torch.stack([
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw], dim=-1)

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # side: cradle on -y or +y, rack on the other (torch.rand, not randint — the
        # first randint after manual_seed is near-constant across seeds)
        if c.side_swap:
            side = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        else:
            side = torch.ones(m, device=dev)

        def fixture(body, nominal, yaw_deg):
            p = torch.zeros(m, 3, device=dev)
            p[:, 0] = nominal[0]
            p[:, 1] = side * nominal[1]
            p[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.fixture_jitter
            yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(yaw_deg)
            write(body, p, qz(yaw))

        fixture(self.cradle, c.cradle_pos, c.cradle_yaw_deg)
        fixture(self.rack, c.rack_pos, c.rack_yaw_deg)

        # roll and spindle: lying on the ground, jitter + free yaw
        c45 = math.cos(math.pi / 4)
        q_lie = torch.tensor([c45, 0.0, c45, 0.0], device=dev).expand(m, 4)
        for body, nominal, z0 in ((self.roll, c.roll_spawn, OUT_FLAT + 0.002),
                                  (self.spindle, c.rod_spawn, CAP_R + 0.002)):
            p = torch.zeros(m, 3, device=dev)
            p[:, 0] = nominal[0]
            p[:, 1] = side * nominal[1]
            p[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.item_jitter
            p[:, 2] = z0
            yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            write(body, p, qmul(qz(yaw), q_lie))

        self.cradled_latch[env_ids] = 0.0
        self.thread_latch[env_ids] = 0.0
        self.carry_latch[env_ids] = 0.0
        self.still_count[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"cradle": self.cradle, "rack": self.rack, "roll": self.roll,
                  "spindle": self.spindle}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "cradled_latch": self.cradled_latch[env_ids].clone(),
            "thread_latch": self.thread_latch[env_ids].clone(),
            "carry_latch": self.carry_latch[env_ids].clone(),
            "still_count": self.still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"cradle": self.cradle, "rack": self.rack, "roll": self.roll,
                  "spindle": self.spindle}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.cradled_latch[env_ids] = state["cradled_latch"]
        self.thread_latch[env_ids] = state["thread_latch"]
        self.carry_latch[env_ids] = state["carry_latch"]
        self.still_count[env_ids] = state["still_count"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "A workshop floor holds four things (their positions, headings, and which "
            "side each fixture sits on change every episode — read them by looking):\n"
            "  - a WHITE PAPER ROLL: a short, wide 12-sided tube, 66 mm across and "
            "90 mm long, with an open 40 mm hollow core through its middle, lying on "
            "its side on the ground;\n"
            "  - a steel SPINDLE: a loose 14 mm rod, 222 mm long, with a dark 26 mm "
            "BALL at each end, lying on the ground;\n"
            "  - a BLUE LOADING CRADLE: a V-block that holds the roll on its side, "
            "with a notched blue backstop plate at ONE end of the V (the notch lets "
            "the spindle's ball pass while the plate braces the roll's face);\n"
            "  - a GREEN HANGING BRACKET: a base slab carrying two upright slotted "
            "plates; each plate has an open-top 22 mm slot with a flared funnel "
            "mouth, and the two slots face each other 130 mm apart at 85 mm height.\n"
            "Goal: make the roll HANG on the bracket, carried by the spindle. The "
            "slots are far too narrow for the roll and the plates block sideways "
            "access, so the bare roll can never be put on the bracket: you must "
            "FIRST pass the spindle all the way through the roll's core so the rod "
            "sticks out of BOTH faces (lay the roll in the cradle against the "
            "backstop and push the spindle in along the V's axis — the notch lets "
            "the leading ball out), and THEN carry the threaded assembly to the "
            "bracket and lower the spindle's two exposed end segments into the two "
            "slot funnels. Done right, the shaft rests in both seats, the two balls "
            "end up OUTSIDE the plates (they are wider than the slots and lock the "
            "spindle in), and the roll hangs freely on the spindle between the "
            "plates, clear of the base. A roll resting on the bracket base, "
            "balanced on top of the spindle without being threaded, or a spindle "
            "seated without the roll on it, does not count. Everything must be at "
            "rest when judged."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Thread the ball-ended steel spindle all the way through the white "
            "roll's hollow core (brace the roll in the blue cradle), then hang the "
            "assembly on the green bracket by lowering the spindle's exposed ends "
            "into the two open slots, so the roll hangs on the spindle between the "
            "plates, clear of the base. The bare roll cannot be mounted; thread the "
            "spindle first."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _axis_w(self, body) -> torch.Tensor:
        """(N,3): body local +z axis in world."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        return quat_apply(body.data.root_quat_w, ez)

    def thread_metrics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(radial, align_cos, prot_lo, prot_hi): the rod LINE vs the roll axis —
        radial offset at the roll's mid-plane, |cos| axis alignment, and the rod tip
        protrusions beyond the -z / +z roll faces (positive = sticking out)."""
        from isaaclab.utils.math import quat_apply_inverse

        q_roll = self.roll.data.root_quat_w
        p = quat_apply_inverse(q_roll, self.spindle.data.root_pos_w - self.roll.data.root_pos_w)
        a = quat_apply_inverse(q_roll, self._axis_w(self.spindle))
        az = a[:, 2].clamp(min=-1.0, max=1.0)
        # rod line point at the roll mid-plane (z=0): t = -p_z / a_z (guard small a_z)
        t0 = torch.where(az.abs() > 0.2, -p[:, 2] / torch.where(az.abs() < 1e-6,
                         torch.full_like(az, 1e-6), az), torch.zeros_like(az))
        t0 = t0.clamp(-ROD_TIP, ROD_TIP)
        mid = p + a * t0.unsqueeze(-1)
        radial = mid[:, 0:2].norm(dim=-1)
        # tip z-coordinates in the roll frame
        z_a = p[:, 2] + a[:, 2] * ROD_TIP
        z_b = p[:, 2] - a[:, 2] * ROD_TIP
        z_hi = torch.maximum(z_a, z_b)
        z_lo = torch.minimum(z_a, z_b)
        prot_hi = z_hi - ROLL_HALF_W
        prot_lo = -ROLL_HALF_W - z_lo
        return radial, az.abs(), prot_lo, prot_hi

    def threaded(self) -> torch.Tensor:
        """(N,) bool: the spindle passes THROUGH the core — rod line inside the core
        at the roll mid-plane, axes aligned, tips protruding beyond both faces."""
        c = self.cfg
        radial, align, prot_lo, prot_hi = self.thread_metrics()
        return (radial < c.thread_radial_tol) \
            & (align > math.cos(math.radians(c.thread_align_max_deg))) \
            & (prot_lo > c.protrude_min) & (prot_hi > c.protrude_min)

    def rod_in_rack(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos (N,3), axis (N,3)): spindle center and axis in the RACK frame."""
        from isaaclab.utils.math import quat_apply_inverse

        q = self.rack.data.root_quat_w
        p = quat_apply_inverse(q, self.spindle.data.root_pos_w - self.rack.data.root_pos_w)
        a = quat_apply_inverse(q, self._axis_w(self.spindle))
        return p, a

    def seated(self) -> torch.Tensor:
        """(N,) bool: the spindle rests in BOTH seats — on the slot line (|y|, |z -
        SEAT_Z| within `seat_yz_tol`), aligned with the slot axis (local x), axially
        centered so both exposed segments span the plates."""
        c = self.cfg
        p, a = self.rod_in_rack()
        on_line = (p[:, 1].abs() < c.seat_yz_tol) \
            & ((p[:, 2] - SEAT_Z).abs() < c.seat_yz_tol)
        aligned = a[:, 0].abs() > math.cos(math.radians(c.seat_align_max_deg))
        centered = p[:, 0].abs() < c.seat_x_max
        return on_line & aligned & centered

    def roll_in_rack(self) -> torch.Tensor:
        """(N,3): roll center in the RACK frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.rack.data.root_quat_w,
                                  self.roll.data.root_pos_w - self.rack.data.root_pos_w)

    def hanging(self) -> torch.Tensor:
        """(N,) bool: the roll hangs UNDER the seated spindle between the plates —
        center in the hang band below SEAT_Z, laterally under the slot line, axially
        between the plates, lowest point clear of the slab."""
        c = self.cfg
        p = self.roll_in_rack()
        in_band = (p[:, 2] - c.hang_z).abs() < c.hang_band
        under = p[:, 1].abs() < c.hang_y_max
        between = p[:, 0].abs() < c.hang_x_max
        clear = p[:, 2] - OUT_CORNER > SLAB_TOP + c.clear_min
        return in_band & under & between & clear

    def roll_in_cradle(self) -> torch.Tensor:
        """(N,) bool: roll seated in the cradle V, aligned with the thread axis."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        q = self.cradle.data.root_quat_w
        p = quat_apply_inverse(q, self.roll.data.root_pos_w - self.cradle.data.root_pos_w)
        rest = torch.tensor([0.0, 0.0, CRADLE_REST_Z], device=p.device)
        near = (p - rest).norm(dim=-1) < c.cradle_tol
        ax = quat_apply_inverse(q, self._axis_w(self.roll))
        aligned = ax[:, 0].abs() > math.cos(math.radians(c.cradle_align_max_deg))
        return near & aligned

    def _still_now(self) -> torch.Tensor:
        """(N,) bool: BOTH free bodies instantaneously below the stillness thresholds
        (a swing crosses this briefly at turning points — never judge on it directly)."""
        c = self.cfg
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for b in (self.roll, self.spindle):
            ok = ok & (b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
                    & (b.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        return ok

    def settled(self) -> torch.Tensor:
        """(N,) bool: stillness SUSTAINED for `settle_steps` consecutive substeps."""
        return self.still_count >= float(self.cfg.settle_steps)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch each demonstrated stage every substep: roll ever seated in the
        cradle; spindle ever threaded through the core; threaded assembly ever
        lifted to `lift_z`. Plus the sustained-stillness counter."""
        self.cradled_latch = torch.maximum(self.cradled_latch,
                                           self.roll_in_cradle().float())
        thr = self.threaded()
        self.thread_latch = torch.maximum(self.thread_latch, thr.float())
        rod_z = self.spindle.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        self.carry_latch = torch.maximum(self.carry_latch,
                                         (thr & (rod_z > self.cfg.lift_z)).float())
        self.still_count = torch.where(self._still_now(), self.still_count + 1.0,
                                       torch.zeros_like(self.still_count))

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the roll hangs on the spindle, the spindle rests in both
        bracket seats, everything settled — the assembled, mounted, suspended state."""
        return self.threaded() & self.seated() & self.hanging() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.20 roll ever cradled + 0.25 spindle ever threaded +
        0.15 threaded assembly ever lifted to the bracket (cap 0.60); 1.0 iff
        success(). Latched — credit never evaporates; the null policy scores ~0."""
        base = (0.20 * self.cradled_latch + 0.25 * self.thread_latch
                + 0.15 * self.carry_latch).clamp(0.0, 0.60)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="spindle_roll", robot="null", env_spacing=3.0))
