"""CradleClampScene — lay the blue bar into the V-saddle, then seat the clamp bracket
over it (sim_gen task `pick_single_ycb_i191`).

Derived from maniskill/pick_single_ycb, but STRATEGICALLY different: the seed is a
single free-space pick — grasp one loose YCB object and raise it 7.5 cm; the checker is
a pure z-position shift, the goal state is "object held in the air", and the plan is
one grasp + one lift. Here lifting is only the trivial transport move and holds almost
no credit: the goal state is a TWO-PART ORDERED ASSEMBLY on a fixture. The BLUE
square bar must be re-oriented and laid HORIZONTALLY into the fixture's V-saddle in
the DIAMOND orientation — axis along the saddle line AND rolled 45 deg so its two
lower faces lie flush on the 45 deg V faces
(gravity + the V faces seat it), and then the YELLOW U-clamp bracket must be
seated over it — its two legs pressed down into the fixture's two mortise pockets so
its cross-bar closes the saddle from above and the blue bar is captive underneath. A
solver needs a different PLAN (identify the target bar by color among a decoy, re-orient
it to saddle-parallel, seat it in the V, then fetch the bracket, align its leg line
with the pocket line, and press it home) and a different code structure (a fixture-frame
saddle predicate + a fixture-frame bracket-seat predicate + an execution order enforced
by geometry — not a z-shift check on one body). The seed's own end state — the bar
lifted well off the ground — is expressible here and rejected: a raised bar earns only
a small latched transport credit and can never satisfy success() (smoke constructs the
lifted state and asserts rejection).

EXECUTION ORDER IS REQUIRED, and enforced by geometry, not rubric fiat: once the
bracket is seated in the empty fixture, its cross-bar underside (96 mm) blocks the
saddle from above, and side entry is impossible because rolling in over a V crest
lifts the bar's crown to ~121 mm at the cross-bar's edge — above the 96 mm underside
(asserted in __post_init__; smoke drives the bar at a seated empty bracket with a real
force and shows it stalls). Bar first, bracket second is the only order that works.

success(): in the FIXTURE'S body frame — (a) the blue bar rests in the saddle: axis
horizontal along the saddle line (within `bar_align_max_deg`), center inside the
saddle window (|x|,|y| tolerances, z in the V-rest band bracketing the true flush
diamond-rest height apex_z + apothem*sqrt(2) — a bar dropped in FACE-DOWN rests on
its corner edges 12 mm higher and is rejected by the band); (b) the bracket is seated: origin inside the seat window,
z in the seat band (legs bottomed INSIDE the pockets — legs standing on the pocket
walls or the cross-bar resting on anything else reads >= 20 mm high and is rejected),
upright, leg line parallel to the pocket line; (c) both settled. The RED decoy bar in
the saddle counts for nothing (wrong object, the seed's "which object" identity made
load-bearing).

score() is graded and latched (credit never evaporates): 0.06 * bar ever lifted +
0.14 * best airborne approach of the bar to the saddle + 0.20 * ever bar-in-saddle
(loose window) + 0.06 * bracket ever lifted + 0.14 * best airborne approach of the
bracket to its seat, capped at 0.60; 0.9 the moment both geometric predicates hold
simultaneously; 1.0 iff success(). The null policy scores ~0.

Assets are fully procedural, one rigid body each (compound spawners; child colliders
of one body never self-collide):
  - fixture (KINEMATIC, gray): base slab + two 45-deg V-blocks (two angled slabs each)
    forming a saddle whose axis is local +y, + two mortise pockets (4 thin walls each)
    on the saddle line at y = +/-64 mm, outboard of the bar's ends.
  - blue bar (dynamic, 150 g): square prism, 40 mm across flats, length 80 mm,
    axis = local +z. On the ground it rests face-down; seating it in the V demands a
    45 deg roll to the diamond orientation, where its faces sit flush on the V faces.
  - red decoy bar (dynamic): identical shape, RED — the wrong object.
  - bracket (dynamic, 120 g, yellow): cross-bar 60 x 150 x 12 mm + two legs
    36 x 10 x 66 mm + a grasp knob 14 x 32 x 30 mm on top. Stands stably on its legs.
Contact offsets are explicit and small (1 mm): the default ~2 cm offset would eat the
4 mm leg-to-pocket clearance and the 7 mm captivity gap that make the assembly real.

Per-episode randomization (readback-verified in smoke): fixture xy jitter + FREE yaw
(the saddle line and pocket line change direction every episode); blue bar, decoy and
bracket each at jittered spawns with free yaw. Heavy imports (isaaclab, pxr) are
deferred so importing this module — and registering the scene — stays app-free.
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


def _box(stage, path: str, size, center, color, contact_offset: float, orient=None) -> None:
    """Author one collidable box child prim (translate -> orient -> scale, authored once
    — idempotent per prim, the duplicate-xformOp trap)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _dynamic_armor(root, mass: float) -> None:
    """Rigid-body armor for the dynamic parts: depenetration cap, damping (the bar in
    the V and the bracket in its pockets live on resting contacts whose impulse noise
    must decay, not ring), solver iterations for crisp contacts, and ZERO
    sleep/stabilization thresholds (a sleeping body silently ignores applied external
    wrenches)."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.15)
    pxrb.CreateAngularDampingAttr(1.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)


def _spawn_fixture(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC cradle fixture at `prim_path`. Origin = base center at
    GROUND level; saddle axis = local +y; mortise pockets on the saddle line at
    y = +/-pocket_y."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(8.0)

    co = cfg.contact_offset
    # base slab
    _box(stage, f"{prim_path}/base", (cfg.base_x, cfg.base_y, cfg.base_t),
         (0.0, 0.0, cfg.base_t / 2), cfg.base_color, co)
    # V-blocks: two per station, 45-deg slabs meeting at the apex line (0, y_b, apex_z)
    s = math.sqrt(0.5)
    half = math.pi / 8  # 45 deg / 2 about +y
    for sign_y in (-1.0, 1.0):
        y_b = sign_y * cfg.vblock_y
        for sign_x, tag in ((1.0, "r"), (-1.0, "l")):
            cx = sign_x * (cfg.slope_len / 2 * s + cfg.slab_t / 2 * s)
            cz = cfg.apex_z + cfg.slope_len / 2 * s - cfg.slab_t / 2 * s
            _box(stage, f"{prim_path}/v{tag}_{'n' if sign_y < 0 else 'p'}",
                 (cfg.slab_t, cfg.vblock_t, cfg.slope_len),
                 (cx, y_b, cz), cfg.block_color, co,
                 orient=(math.cos(half), 0.0, sign_x * math.sin(half), 0.0))
    # mortise pockets: 4 thin walls each, interior pocket_ix * pocket_iy, top at wall_top
    wz = cfg.base_t + cfg.wall_h / 2
    for sign_y in (-1.0, 1.0):
        y_p = sign_y * cfg.pocket_y
        nm = "n" if sign_y < 0 else "p"
        gy = cfg.pocket_iy / 2 + cfg.wall_t / 2
        _box(stage, f"{prim_path}/wi_{nm}", (cfg.pocket_ix + 2 * cfg.wall_t, cfg.wall_t, cfg.wall_h),
             (0.0, y_p - gy, wz), cfg.block_color, co)
        _box(stage, f"{prim_path}/wo_{nm}", (cfg.pocket_ix + 2 * cfg.wall_t, cfg.wall_t, cfg.wall_h),
             (0.0, y_p + gy, wz), cfg.block_color, co)
        gx = cfg.pocket_ix / 2 + cfg.wall_t / 2
        _box(stage, f"{prim_path}/we_{nm}", (cfg.wall_t, cfg.pocket_iy, cfg.wall_h),
             (gx, y_p, wz), cfg.block_color, co)
        _box(stage, f"{prim_path}/ww_{nm}", (cfg.wall_t, cfg.pocket_iy, cfg.wall_h),
             (-gx, y_p, wz), cfg.block_color, co)
    return root


def _spawn_bar(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one dynamic SQUARE bar at `prim_path`: a single box, axis = local +z.
    Origin = bar center. (A union of two 45-deg boxes is NOT an octagon — it is an
    8-pointed star whose corners protrude past the other box's faces; a plain square
    seated corner-down instead rests its two lower FACES flush on the 45-deg V.)"""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _dynamic_armor(root, cfg.mass_props.mass)
    w, ln = cfg.across_flats, cfg.length
    _box(stage, f"{prim_path}/body", (w, w, ln), (0.0, 0.0, 0.0), cfg.color,
         cfg.contact_offset)
    return root


def _spawn_bracket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the dynamic U-clamp bracket at `prim_path`: cross-bar + two legs
    (leg line = local y) + grasp knob on top. Origin = cross-bar center."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _dynamic_armor(root, cfg.mass_props.mass)
    co = cfg.contact_offset
    _box(stage, f"{prim_path}/bar", (cfg.bar_x, cfg.bar_y, cfg.bar_t),
         (0.0, 0.0, 0.0), cfg.color, co)
    for sign_y, nm in ((-1.0, "n"), (1.0, "p")):
        _box(stage, f"{prim_path}/leg_{nm}", (cfg.leg_x, cfg.leg_y, cfg.leg_len),
             (0.0, sign_y * cfg.leg_pitch, -(cfg.bar_t / 2 + cfg.leg_len / 2)),
             cfg.color, co)
    _box(stage, f"{prim_path}/knob", (cfg.knob_x, cfg.knob_y, cfg.knob_h),
         (0.0, 0.0, cfg.bar_t / 2 + cfg.knob_h / 2), cfg.knob_color, co)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (explicit @configclass subclasses
    of RigidObjectSpawnerCfg, defined lazily so the module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "fixture" not in _SPAWNER_CACHE:

        @configclass
        class CradleFixtureSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_fixture)
            base_x: float = 0.30
            base_y: float = 0.26
            base_t: float = 0.030
            apex_z: float = 0.031
            slope_len: float = 0.080
            slab_t: float = 0.012
            vblock_y: float = 0.026
            vblock_t: float = 0.022
            pocket_y: float = 0.064
            pocket_ix: float = 0.044
            pocket_iy: float = 0.018
            wall_t: float = 0.006
            wall_h: float = 0.022
            base_color: tuple = (0.35, 0.35, 0.38)
            block_color: tuple = (0.5, 0.5, 0.55)
            contact_offset: float = 0.001

        @configclass
        class SquareBarSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bar)
            across_flats: float = 0.040
            length: float = 0.080
            color: tuple = (0.1, 0.2, 0.8)
            contact_offset: float = 0.001

        @configclass
        class ClampBracketSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bracket)
            bar_x: float = 0.060
            bar_y: float = 0.150
            bar_t: float = 0.012
            leg_x: float = 0.036
            leg_y: float = 0.010
            leg_len: float = 0.066
            leg_pitch: float = 0.064
            knob_x: float = 0.014
            knob_y: float = 0.032
            knob_h: float = 0.030
            color: tuple = (0.85, 0.70, 0.10)
            knob_color: tuple = (0.30, 0.30, 0.32)
            contact_offset: float = 0.001

        _SPAWNER_CACHE.update(fixture=CradleFixtureSpawnerCfg, bar=SquareBarSpawnerCfg,
                              bracket=ClampBracketSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CradleClampSceneCfg(BaseCfg):
    """Config for `CradleClampScene`. The strategic honesty knobs are geometric and
    asserted in `__post_init__`: the bar seats flush in the 45-deg V at a computable
    height, the bracket's legs fit the pockets with real clearance, a seated bracket's
    cross-bar makes the saddle unreachable (captivity + order), and legs standing on
    the pocket walls read far outside the seat band."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    bar_xy_tol: float = tunable(0.012)  # |x| of the bar center off the valley line (fixture)
    bar_y_tol: float = tunable(0.012)  # |y| of the bar center off the saddle center
    bar_z_tol: float = tunable(0.007)  # bar center z within this of the flush V-rest height
    bar_align_max_deg: float = tunable(12.0)  # bar axis vs the saddle line
    seat_xy_tol: float = tunable(0.010)  # bracket origin |x|,|y| in the fixture frame
    seat_z_lo: float = tunable(-0.006)  # bracket z minus seat_z, lower bound
    seat_z_hi: float = tunable(0.005)  # ... upper bound (legs-on-walls reads +0.022)
    seat_upright_max_deg: float = tunable(10.0)  # bracket +z vs fixture +z
    seat_yaw_max_deg: float = tunable(10.0)  # bracket leg line vs pocket line (|dot|)
    settle_lin: float = tunable(0.05)  # max |lin vel| when judging success (m/s)
    settle_ang: float = tunable(0.8)  # max |ang vel| when judging success (rad/s)
    lift_z_bar: float = tunable(0.11)  # bar center height that latches "lifted"
    lift_z_bracket: float = tunable(0.13)  # bracket origin height that latches "lifted"
    air_z: float = tunable(0.05)  # min height gating the approach latches (airborne)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    fixture_pos: tuple = tunable((0.30, 0.0))  # fixture base center (m)
    fixture_jitter: float = tunable(0.04)  # uniform +/- xy jitter at reset
    fixture_yaw_deg: float = tunable(180.0)  # uniform +/- yaw at reset (free yaw)
    bar_pos: tuple = tunable((-0.02, -0.20))  # blue bar spawn center
    bracket_pos: tuple = tunable((-0.02, 0.20))  # bracket spawn center
    decoy_pos: tuple = tunable((-0.20, 0.0))  # red decoy spawn center
    part_jitter: float = tunable(0.04)  # uniform +/- xy jitter per part at reset
    part_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per part at reset

    # --- info: fixture structure -------------------------------------------------------------
    base_x: float = info(0.30)
    base_y: float = info(0.26)
    base_t: float = info(0.030)  # base slab thickness (pocket floors = base top)
    apex_z: float = info(0.031)  # V apex line height
    slope_len: float = info(0.080)  # V face slope length
    slab_t: float = info(0.012)
    vblock_y: float = info(0.026)  # V-block stations at y = +/- this
    vblock_t: float = info(0.022)  # V-block thickness along y
    pocket_y: float = info(0.064)  # pocket centers at y = +/- this
    pocket_ix: float = info(0.044)  # pocket interior along x
    pocket_iy: float = info(0.018)  # pocket interior along y
    wall_t: float = info(0.006)
    wall_h: float = info(0.022)  # pocket wall height above the base top
    # --- info: bar structure -----------------------------------------------------------------
    across_flats: float = info(0.040)  # square bar across-flats (apothem * 2); jaw grasp span
    bar_len: float = info(0.080)
    bar_mass: float = info(0.15)
    # --- info: bracket structure -------------------------------------------------------------
    cbar_x: float = info(0.060)  # cross-bar width (spans the saddle in x)
    cbar_y: float = info(0.150)
    cbar_t: float = info(0.012)
    leg_x: float = info(0.036)
    leg_y: float = info(0.010)
    leg_len: float = info(0.066)
    knob_x: float = info(0.014)  # grasp knob across the jaw
    knob_y: float = info(0.032)
    knob_h: float = info(0.030)
    bracket_mass: float = info(0.12)
    contact_offset: float = info(0.001)

    # Derived (filled in __post_init__).
    apothem: float = field(default=None, init=False)
    bar_rest_z: float = field(default=None, init=False)  # flush V-rest center height
    seat_z: float = field(default=None, init=False)  # seated bracket origin height
    crown_z: float = field(default=None, init=False)  # seated bar top-corner height
    cbar_under_z: float = field(default=None, init=False)  # seated cross-bar underside
    ground_rest_z: float = field(default=None, init=False)  # bar on flat ground
    stand_z: float = field(default=None, init=False)  # bracket standing on its legs

    def __post_init__(self) -> None:
        self.apothem = self.across_flats / 2
        # Diamond rest: the square bar corner-down, both lower faces flush on the V
        # faces; contact distance = apothem, center = apex + apothem*sqrt(2).
        self.bar_rest_z = self.apex_z + self.apothem * math.sqrt(2.0)
        self.seat_z = self.base_t + self.leg_len + self.cbar_t / 2
        across_corners_half = self.apothem * math.sqrt(2.0)  # square corner radius
        self.crown_z = self.bar_rest_z + across_corners_half  # top corner, seated
        self.cbar_under_z = self.seat_z - self.cbar_t / 2
        self.ground_rest_z = self.apothem
        self.stand_z = self.leg_len + self.cbar_t / 2
        # Face-down in the V the bar rests on its corner EDGES 12 mm higher — outside
        # the z band, so the diamond roll is genuinely required.
        face_down_rest = self.apex_z + across_corners_half * math.sqrt(2.0)
        assert face_down_rest - self.bar_rest_z > self.bar_z_tol + 0.004, \
            "face-down V rest must sit clearly above the diamond-rest z band"
        # -- the mechanism must be real (geometry asserts) --
        # captivity: the seated bar's crown sits under the seated cross-bar
        assert self.crown_z + 0.004 < self.cbar_under_z, \
            "seated cross-bar must cap the seated bar (captivity gap)"
        # order: side entry with the bracket seated is blocked — a bar riding the V face
        # at the cross-bar's x-edge has its crown above the cross-bar underside
        edge_x = self.cbar_x / 2
        crown_at_edge = edge_x + self.apex_z + self.apothem * math.sqrt(2.0) + across_corners_half
        assert crown_at_edge > self.cbar_under_z + 0.015, \
            "a bar entering over the V crest must clash with the seated cross-bar"
        # legs fit the pockets with real clearance, and legs-on-walls is discriminated
        assert self.pocket_ix - self.leg_x >= 0.006, "pocket x clearance"
        assert self.pocket_iy - self.leg_y >= 0.006, "pocket y clearance"
        assert self.wall_h > (self.seat_z_hi - self.seat_z_lo), \
            "legs standing on the pocket walls must read outside the seat band"
        # the bar's ends stay clear of the pocket walls and the legs clear the bar
        wall_in = self.pocket_y - self.pocket_iy / 2 - self.wall_t
        assert self.bar_len / 2 + 0.006 < wall_in, "bar ends clear of the pocket walls"
        leg_in = self.pocket_y - self.leg_y / 2
        assert self.bar_len / 2 + 0.008 < leg_in, "bracket legs clear of the seated bar"
        # V geometry: contact points inside the slope span; bar floats above the apex gap
        assert self.apothem * math.sqrt(2.0) < self.slope_len, "V slope long enough"
        assert self.bar_rest_z - across_corners_half > self.apex_z - 0.002, \
            "bar rests on the V faces, not the apex gap"
        # embodiment: jaw spans
        assert self.across_flats <= 0.06, "bar must fit the Franka jaw across flats"
        assert self.knob_x <= 0.05, "knob must fit the Franka jaw"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("cradle_clamp")
class CradleClampScene(BaseScene):
    cfg: CradleClampSceneCfg

    def __init__(self, cfg: CradleClampSceneCfg | None = None) -> None:
        super().__init__(cfg or CradleClampSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        fix_cls, bar_cls, br_cls = spawners["fixture"], spawners["bar"], spawners["bracket"]

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
            "fixture": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Fixture",
                spawn=fix_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    base_x=c.base_x, base_y=c.base_y, base_t=c.base_t, apex_z=c.apex_z,
                    slope_len=c.slope_len, slab_t=c.slab_t, vblock_y=c.vblock_y,
                    vblock_t=c.vblock_t, pocket_y=c.pocket_y, pocket_ix=c.pocket_ix,
                    pocket_iy=c.pocket_iy, wall_t=c.wall_t, wall_h=c.wall_h,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fixture_pos[0], c.fixture_pos[1], 0.0)),
            ),
            "bar_blue": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BarBlue",
                spawn=bar_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bar_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    across_flats=c.across_flats, length=c.bar_len,
                    color=(0.10, 0.20, 0.80), contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bar_pos[0], c.bar_pos[1], 0.05)),
            ),
            "bar_red": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BarRed",
                spawn=bar_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bar_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    across_flats=c.across_flats, length=c.bar_len,
                    color=(0.80, 0.12, 0.10), contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.decoy_pos[0], c.decoy_pos[1], 0.05)),
            ),
            "bracket": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bracket",
                spawn=br_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bracket_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    bar_x=c.cbar_x, bar_y=c.cbar_y, bar_t=c.cbar_t, leg_x=c.leg_x,
                    leg_y=c.leg_y, leg_len=c.leg_len, leg_pitch=c.pocket_y,
                    knob_x=c.knob_x, knob_y=c.knob_y, knob_h=c.knob_h,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bracket_pos[0], c.bracket_pos[1], 0.10)),
            ),
        }
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
        self.fixture: RigidObject = env.iscene["fixture"]
        self.bar: RigidObject = env.iscene["bar_blue"]
        self.decoy: RigidObject = env.iscene["bar_red"]
        self.bracket: RigidObject = env.iscene["bracket"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.d_ref_bar = torch.full((n,), 0.5, device=dev)  # reset bar->saddle distance
        self.d_ref_br = torch.full((n,), 0.5, device=dev)  # reset bracket->seat distance
        self.bar_lift_latch = torch.zeros(n, device=dev)
        self.bar_appr_latch = torch.zeros(n, device=dev)
        self.saddle_latch = torch.zeros(n, device=dev)
        self.br_lift_latch = torch.zeros(n, device=dev)
        self.br_appr_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: fixture at its slot with xy jitter + free yaw; blue bar,
        decoy and bracket at their jittered spawns with free yaw (bar/decoy lying on a
        flat, bracket standing on its legs); latches zeroed; approach baselines
        captured from the sampled poses."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, xy: torch.Tensor, z: float, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3:7] = quat
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        def qz(yaw: torch.Tensor) -> torch.Tensor:
            zeros = torch.zeros_like(yaw)
            return torch.stack([torch.cos(yaw / 2), zeros, zeros, torch.sin(yaw / 2)], dim=-1)

        # --- fixture: slot + jitter + free yaw ---
        fxy = torch.tensor(c.fixture_pos, device=dev).expand(m, 2).clone()
        fxy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.fixture_jitter
        fyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.fixture_yaw_deg)
        write(self.fixture, fxy, 0.0, qz(fyaw))

        # --- blue bar / decoy: lying on a flat, free yaw of the axis ---
        # lying with axis horizontal: q = qz(yaw) * qy(90 deg); qy90 = (c45, 0, c45, 0)
        c45 = math.cos(math.pi / 4)
        for body, pos in ((self.bar, c.bar_pos), (self.decoy, c.decoy_pos)):
            xy = torch.tensor(pos, device=dev).expand(m, 2).clone()
            xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.part_jitter
            yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.part_yaw_deg)
            half = yaw / 2
            quat = torch.stack([torch.cos(half) * c45, -torch.sin(half) * c45,
                                torch.cos(half) * c45, torch.sin(half) * c45], dim=-1)
            write(body, xy, c.ground_rest_z + 0.003, quat)

        # --- bracket: standing on its legs, free yaw ---
        bxy = torch.tensor(c.bracket_pos, device=dev).expand(m, 2).clone()
        bxy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.part_jitter
        byaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.part_yaw_deg)
        write(self.bracket, bxy, c.stand_z + 0.003, qz(byaw))

        # --- latches + approach baselines ---
        self.bar_lift_latch[env_ids] = 0.0
        self.bar_appr_latch[env_ids] = 0.0
        self.saddle_latch[env_ids] = 0.0
        self.br_lift_latch[env_ids] = 0.0
        self.br_appr_latch[env_ids] = 0.0
        saddle_w = torch.stack([fxy[:, 0], fxy[:, 1],
                                torch.full((m,), c.bar_rest_z, device=dev)], dim=-1)
        bar_xy = self.bar.data.root_pos_w[env_ids, :2] - origin[:, :2]
        br_xy = self.bracket.data.root_pos_w[env_ids, :2] - origin[:, :2]
        d_bar = (torch.cat([bar_xy, torch.full((m, 1), c.ground_rest_z, device=dev)], dim=-1)
                 - saddle_w).norm(dim=-1)
        seat_w = torch.stack([fxy[:, 0], fxy[:, 1],
                              torch.full((m,), c.seat_z, device=dev)], dim=-1)
        d_br = (torch.cat([br_xy, torch.full((m, 1), c.stand_z, device=dev)], dim=-1)
                - seat_w).norm(dim=-1)
        self.d_ref_bar[env_ids] = d_bar.clamp(min=0.10)
        self.d_ref_br[env_ids] = d_br.clamp(min=0.10)

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "fixture": self.fixture.data.root_state_w[env_ids].clone(),
            "bar": self.bar.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "bracket": self.bracket.data.root_state_w[env_ids].clone(),
            "d_ref_bar": self.d_ref_bar[env_ids].clone(),
            "d_ref_br": self.d_ref_br[env_ids].clone(),
            "latches": torch.stack([self.bar_lift_latch[env_ids],
                                    self.bar_appr_latch[env_ids],
                                    self.saddle_latch[env_ids],
                                    self.br_lift_latch[env_ids],
                                    self.br_appr_latch[env_ids]], dim=-1),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.fixture.write_root_state_to_sim(state["fixture"], env_ids)
        self.bar.write_root_state_to_sim(state["bar"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.bracket.write_root_state_to_sim(state["bracket"], env_ids)
        self.d_ref_bar[env_ids] = state["d_ref_bar"]
        self.d_ref_br[env_ids] = state["d_ref_br"]
        lat = state["latches"]
        self.bar_lift_latch[env_ids] = lat[:, 0]
        self.bar_appr_latch[env_ids] = lat[:, 1]
        self.saddle_latch[env_ids] = lat[:, 2]
        self.br_lift_latch[env_ids] = lat[:, 3]
        self.br_appr_latch[env_ids] = lat[:, 4]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A gray service fixture stands on the ground: a flat base plate carrying a "
            f"V-shaped cradle saddle across its middle (two pairs of 45-degree V-blocks "
            f"forming one straight saddle line) and, on the same line just beyond the "
            f"saddle on both sides, two small rectangular pocket wells "
            f"({c.pocket_ix * 1000:.0f} x {c.pocket_iy * 1000:.0f} mm openings, "
            f"{c.wall_h * 1000:.0f} mm deep). The fixture's position and the direction "
            f"of its saddle line change every episode — read them by looking. Nearby on "
            f"the ground lie: a BLUE square bar ({c.across_flats * 1000:.0f} mm "
            f"across flats, {c.bar_len * 1000:.0f} mm long), a RED square bar of the "
            f"same size (a decoy — leave it alone), and a YELLOW U-clamp bracket "
            f"standing on its two legs (a {c.cbar_x * 1000:.0f} x "
            f"{c.cbar_y * 1000:.0f} mm cross-bar with a small dark grasp knob on top "
            f"and two downward legs that match the fixture's pocket wells).\n"
            f"Goal, in this order: FIRST lay the BLUE bar horizontally into the "
            f"V-saddle, its axis along the saddle line and ROLLED 45 degrees to the "
            f"diamond orientation so both its lower faces lie flush on the V faces "
            f"(dropped in face-down it perches on its corner edges, visibly proud of "
            f"the flush seat, and does not count). THEN pick up "
            f"the clamp bracket by its knob, align its two legs with the two pocket "
            f"wells, and press it down over the blue bar until both legs bottom out in "
            f"the wells — the cross-bar then caps the saddle and the blue bar is "
            f"locked underneath. The order is forced by the geometry: once the bracket "
            f"is seated, the saddle is closed and the bar cannot enter. Success needs "
            f"the blue bar seated in the V AND the bracket fully seated in its "
            f"pockets, both at rest. The red bar in the saddle, the bracket perched on "
            f"its pocket walls or resting anywhere else, or a bar left leaning on the "
            f"fixture count for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lay the blue square bar into the fixture's V-saddle along the saddle "
            "line, rolled 45 degrees so it sits flush in the V, then press the yellow "
            "clamp bracket down over it so both legs seat fully in the two pocket "
            "wells and the cross-bar locks the bar in. Do the bar first — a seated "
            "bracket closes the saddle. Do not use the red bar."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _fixture_local(self, body) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos (N,3), quat (N,4)) of `body` in the fixture's body frame."""
        from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul

        rel = body.data.root_pos_w - self.fixture.data.root_pos_w
        fq = self.fixture.data.root_quat_w
        loc = quat_apply_inverse(fq, rel)
        q = quat_mul(quat_inv(fq), body.data.root_quat_w)
        return loc, q

    def _bar_in_saddle(self, body, *, xy_tol: float, y_tol: float, z_tol: float,
                       align_max_deg: float) -> torch.Tensor:
        """(N,) bool: `body` (a square bar, axis local +z) resting in the saddle —
        fixture-frame center in the window, axis along the saddle line (local +y)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        loc, q = self._fixture_local(body)
        n = loc.shape[0]
        ez = torch.tensor([0.0, 0.0, 1.0], device=loc.device).expand(n, 3)
        axis = quat_apply(q, ez)  # bar axis in the FIXTURE frame
        aligned = axis[:, 1].abs() >= math.cos(math.radians(align_max_deg))
        return (loc[:, 0].abs() <= xy_tol) & (loc[:, 1].abs() <= y_tol) \
            & ((loc[:, 2] - c.bar_rest_z).abs() <= z_tol) & aligned

    def bar_in_saddle(self) -> torch.Tensor:
        """(N,) bool, geometric: the BLUE bar seated in the V-saddle."""
        c = self.cfg
        return self._bar_in_saddle(self.bar, xy_tol=c.bar_xy_tol, y_tol=c.bar_y_tol,
                                   z_tol=c.bar_z_tol, align_max_deg=c.bar_align_max_deg)

    def bracket_seated(self) -> torch.Tensor:
        """(N,) bool, geometric: the bracket's legs bottomed inside the two pockets —
        fixture-frame origin in the seat window and band, upright, leg line parallel
        to the pocket line. Legs standing ON the pocket walls read +wall_h (22 mm)
        high; the cross-bar resting on the bar's crown with legs out of the pockets
        cannot keep |x|,|y| and yaw inside the window (walls push the legs out)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        loc, q = self._fixture_local(self.bracket)
        n = loc.shape[0]
        ez = torch.tensor([0.0, 0.0, 1.0], device=loc.device).expand(n, 3)
        ey = torch.tensor([0.0, 1.0, 0.0], device=loc.device).expand(n, 3)
        up = quat_apply(q, ez)  # bracket +z in the fixture frame
        leg_line = quat_apply(q, ey)
        upright = up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(c.seat_upright_max_deg))
        yaw_ok = leg_line[:, 1].abs() >= math.cos(math.radians(c.seat_yaw_max_deg))
        dz = loc[:, 2] - c.seat_z
        return (loc[:, 0].abs() <= c.seat_xy_tol) & (loc[:, 1].abs() <= c.seat_xy_tol) \
            & (dz >= c.seat_z_lo) & (dz <= c.seat_z_hi) & upright & yaw_ok

    def settled(self, body) -> torch.Tensor:
        return (body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    def assembled(self) -> torch.Tensor:
        """(N,) bool: both geometric predicates hold simultaneously."""
        return self.bar_in_saddle() & self.bracket_seated()

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch: bar/bracket ever-lifted, best airborne approach to saddle/seat, and
        ever bar-in-saddle (loose window), each physics substep."""
        c = self.cfg
        dev = self.env.device
        fq = self.fixture.data.root_quat_w
        fp = self.fixture.data.root_pos_w
        from isaaclab.utils.math import quat_apply

        # bar
        bz = (self.bar.data.root_pos_w - self.env_origins)[:, 2]
        self.bar_lift_latch = torch.maximum(self.bar_lift_latch,
                                            (bz > c.lift_z_bar).float())
        saddle_l = torch.tensor([0.0, 0.0, c.bar_rest_z], device=dev).expand(fp.shape[0], 3)
        saddle_w = fp + quat_apply(fq, saddle_l)
        d = (self.bar.data.root_pos_w - saddle_w).norm(dim=-1)
        appr = (1.0 - d / self.d_ref_bar).clamp(0.0, 1.0) * (bz > c.air_z).float()
        self.bar_appr_latch = torch.maximum(self.bar_appr_latch, appr)
        sad = self._bar_in_saddle(self.bar, xy_tol=c.bar_xy_tol + 0.004,
                                  y_tol=c.bar_y_tol + 0.004, z_tol=c.bar_z_tol + 0.003,
                                  align_max_deg=c.bar_align_max_deg + 4.0)
        self.saddle_latch = torch.maximum(self.saddle_latch, sad.float())
        # bracket
        kz = (self.bracket.data.root_pos_w - self.env_origins)[:, 2]
        self.br_lift_latch = torch.maximum(self.br_lift_latch,
                                           (kz > c.lift_z_bracket).float())
        seat_l = torch.tensor([0.0, 0.0, c.seat_z], device=dev).expand(fp.shape[0], 3)
        seat_w = fp + quat_apply(fq, seat_l)
        d2 = (self.bracket.data.root_pos_w - seat_w).norm(dim=-1)
        appr2 = (1.0 - d2 / self.d_ref_br).clamp(0.0, 1.0) * (kz > c.air_z).float()
        self.br_appr_latch = torch.maximum(self.br_appr_latch, appr2)

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: blue bar seated in the V-saddle AND bracket seated in its
        pockets, both settled."""
        return self.assembled() & self.settled(self.bar) & self.settled(self.bracket)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.06*bar lift + 0.14*bar approach + 0.20*ever-in-saddle
        + 0.06*bracket lift + 0.14*bracket approach (latched, cap 0.60); 0.9 once both
        geometric predicates hold; 1.0 iff success. Doing nothing scores ~0; the seed's
        plan (lift the object into the air) scores <= ~0.15 and can never succeed."""
        base = (0.06 * self.bar_lift_latch + 0.14 * self.bar_appr_latch
                + 0.20 * self.saddle_latch + 0.06 * self.br_lift_latch
                + 0.14 * self.br_appr_latch).clamp(0.0, 0.60)
        s = torch.where(self.assembled(), torch.maximum(base, base.new_tensor(0.9)), base)
        return torch.where(self.success(), s.new_tensor(1.0), s)


register_env("simgen", lambda: EnvCfg(scene="cradle_clamp", robot="null"))
