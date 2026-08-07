"""ServiceCarouselScene — fetch the plate around the carousel, load the MIDDLE bowl, rotate
it back to the serve mark under the hutch.

Derived from libero_90 kitchen_scene2 "put the black bowl in the middle on the plate". The
seed selects one of three identical loose bowls by table position ("the middle one") and does
a single free grasp-carry-place onto an openly reachable flat plate. Here the placement
itself stays (a bowl is still seated on the plate through contact) but everything around it
is replaced by MECHANISM-MEDIATED TRANSPORT:

  * the plate is NOT freely reachable: it sits keyed in a shallow pocket on a rotating
    CAROUSEL disc, and it starts parked under a low service HUTCH (roof 83 mm above the
    disc, wide in both directions) where no hand — and no top-down placement — fits;
  * the plate cannot be extracted either: it sits flush inside its pocket (top level with
    the pocket wall, 3 mm radial slack), so there is no rim to pinch — the CAROUSEL is the
    only way to move it;
  * the goal position is DIFFERENT from the start position: the plate starts 25-45 deg off
    the serve mark (randomly signed), and must end CENTERED on the green serve stripe under
    the hutch. So the seed's end state "bowl on the plate where the plate is" is a rejected
    outcome here (delivery azimuth gate) — constructed and rejected in the smoke;
  * the required plan is rotate -> load -> rotate: bring the pocket around (~180 deg) to
    the open loading side by driving the carousel (four pegs on the disc rim are the
    handles), seat the MIDDLE bowl of the three-bowl row on the plate, then rotate the
    LOADED carousel back until the plate parks on the serve mark. The load must ride the
    rotation — a bowl that slides off or tips during the return fails;
  * both rotations and the seating are physical: the disc is a dynamic body on an authored
    revolute joint (angular damping, no motors), the plate and bowl are carried by friction.

Success (state predicate, all simultaneous, settled):
  * the episode's MIDDLE bowl (middle slot of the three-bowl row at reset) sits upright ON
    the plate, near its centre;
  * the plate sits in its pocket at the SERVE azimuth (within `serve_tol_deg` of the green
    stripe), upright, at pocket height;
  * everything settled (bowls + plate slow, disc angular speed below `disc_settle_w`).

Rubric (graded 0..1, latched in post_step, monotone along the demonstrated solution;
1.0 iff success()):
  0.00  nothing happened (the plate starts under the hutch, off the mark -> score 0)
  +0.25 FETCHED: the plate has been carried by the carousel into the open LOADING window
        (azimuth within `load_window_deg` of the robot side) at least once, disc slow;
  +0.35 LOADED: the middle bowl has been seated on the plate WHILE the plate was in the
        loading window (a bowl teleported/placed onto the still-hidden plate latches
        nothing), bowl slow;
  1.00  success() (overrides the 0.60 partial sum).

Honesty of the gates:
  * bowl_xy_tol (35 mm): a bowl fully on the 150 mm plate can rest at most ~27 mm
    off-centre (plate r 75 - base r 48), so an accepted bowl is genuinely on the plate;
    a 55 mm off-centre bowl rests flat bridging plate and pocket wall (wall top is flush
    with the plate top) — a constructible, settled, rejected near-miss;
  * delivery azimuth: `serve_tol_deg` (15 deg) vs a start offset of 25-45 deg — the
    reset state (and the seed-strategy state built on it) is outside the window by
    construction; a 20 deg parked delivery is the constructible near-miss;
  * plate_in_pocket: radial band (+-30 mm about the 115 mm pocket radius) and height band
    tie the plate to the pocket floor — a plate lifted onto the disc top or the counter
    does not count;
  * upright gates on bowl and plate reject a tipped/inverted bowl that rode the return;
  * both latches carry velocity gates (disc / bowl) so fly-through states latch nothing.

Assets are fully procedural (no external files):
  * carousel disc: dynamic compound (r 200 mm cylinder, four 18 mm peg handles at r 170,
    an 8-box pocket ring of inner r 78 mm at r 115) on a spawn-authored revolute joint to
    the kinematic hub (joint authored in bind(), body0 = hub — the fixture is never
    re-posed, so the world-fixed anchor quirk is moot);
  * hutch: kinematic compound (roof slab with 83 mm clearance over the disc, full back
    wall OUTSIDE the disc rim, two front posts outside the rim, green serve stripe on the
    back wall) — geometry that blocks top-down access to the serve sector while the pegs
    (70 mm) and a loaded plate (62 mm) sweep beneath it with clearance BY CONSTRUCTION
    (live collisions: hutch and disc are separate bodies, not a filtered joint pair);
  * plate: plain white cylinder (150 x 12 mm), flush inside the pocket;
  * bowls: three identical BLACK octagonal bowls (outer ~100 mm, 50 mm tall, 7 mm rim
    wall — the rim is the pinch feature), one rigid body each, explicit MassAPI.

Per-episode randomization (verified by readback in the smoke): which physical bowl body
occupies the middle slot (target identity permutation), the plate's start azimuth under the
hutch (signed 25-45 deg), per-bowl xy jitter and free yaw.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
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


# ----- custom compound spawners -------------------------------------------------------------------
# One rigid body per object: root Xform with RigidBodyAPI (+ explicit MassAPI on dynamics —
# overlapping child shapes would double-count density), child collider shapes. Authored through
# `clone()` so per-env replication is idempotent (no duplicate xformOps).

_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_root(stage, prim_path: str, translation, orientation, *, kinematic: bool,
                mass: float | None = None, ang_damp: float = 0.0, lin_damp: float = 0.0):
    """Root Xform + RigidBodyAPI (+ MassAPI / damping / depenetration cap on dynamics)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    else:
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
        px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        px.CreateMaxDepenetrationVelocityAttr(0.5)
        px.CreateLinearDampingAttr(lin_damp)
        px.CreateAngularDampingAttr(ang_damp)
        px.CreateSolverPositionIterationCountAttr(16)
        # 4 velocity iterations: 1 leaves GPU TGS contacts with a constant phantom creep
        # (observed ~0.04 rad/s disc drift under zero torque + damping)
        px.CreateSolverVelocityIterationCountAttr(4)
    return root


def _box(stage, prim_path: str, name: str, tx, ty, tz, sx, sy, sz, color,
         contact_offset: float | None, yaw: float = 0.0):
    """Child box; collision only when contact_offset is given (None -> visual only).

    xformOp order is translate -> orient -> scale, so the unit cube is scaled in its OWN
    frame first, then yawed — required for the non-axis-aligned pocket/rim wall segments
    (orient authored after scale would shear the box along the parent axes)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    seg = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(float(tx), float(ty), float(tz)))
    if yaw:
        sxf.AddOrientOp().Set(Gf.Quatf(math.cos(yaw / 2), Gf.Vec3f(0.0, 0.0, math.sin(yaw / 2))))
    sxf.AddScaleOp().Set(Gf.Vec3f(float(sx), float(sy), float(sz)))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
    return seg


def _cyl(stage, prim_path: str, name: str, tx, ty, tz, radius, height, color,
         contact_offset: float, yaw: float = 0.0):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{name}")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    cxf = UsdGeom.Xformable(cyl.GetPrim())
    cxf.AddTranslateOp().Set(Gf.Vec3d(float(tx), float(ty), float(tz)))
    if yaw:
        cxf.AddOrientOp().Set(Gf.Quatf(math.cos(yaw / 2), Gf.Vec3f(0.0, 0.0, math.sin(yaw / 2))))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(cyl.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    return cyl


def _spawn_disc(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The carousel disc: flat cylinder + four peg handles at the rim + an 8-box pocket
    ring (inner r `pocket_inner_r`) around the pocket mount point on local +x. Root origin
    at the disc centre (the joint anchor)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _apply_root(stage, prim_path, translation, orientation, kinematic=False,
                       mass=cfg.mass_props.mass, ang_damp=cfg.ang_damp)
    co = float(cfg.contact_offset)
    _cyl(stage, prim_path, "disc", 0.0, 0.0, 0.0, cfg.disc_r, cfg.disc_h, cfg.color, co)
    for k in range(4):
        a = math.radians(45.0 + 90.0 * k)
        _cyl(stage, prim_path, f"peg_{k}", cfg.peg_mount_r * math.cos(a),
             cfg.peg_mount_r * math.sin(a), cfg.disc_h / 2 + cfg.peg_h / 2,
             cfg.peg_r, cfg.peg_h, cfg.peg_color, co)
    n_w = 8
    rw = cfg.pocket_inner_r + cfg.wall_t / 2
    seg_l = 2 * rw * math.tan(math.pi / n_w) + 0.004
    for k in range(n_w):
        a = 2 * math.pi * k / n_w
        wx = cfg.pocket_mount_r + rw * math.cos(a)
        wy = rw * math.sin(a)
        _box(stage, prim_path, f"pwall_{k}", wx, wy, cfg.disc_h / 2 + cfg.wall_h / 2,
             cfg.wall_t, seg_l, cfg.wall_h, cfg.wall_color, co, yaw=a)
    return root


def _spawn_hub(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The kinematic pivot pedestal under the disc. Root origin at the counter top on the
    carousel axis (the revolute joint's body0)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _apply_root(stage, prim_path, translation, orientation, kinematic=True)
    _cyl(stage, prim_path, "pedestal", 0.0, 0.0, cfg.ped_h / 2, cfg.ped_r, cfg.ped_h,
         cfg.color, float(cfg.contact_offset))
    return root


def _spawn_hood(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The kinematic service hutch over the serve sector: roof slab (lower face
    `clear_h` above the counter), full back wall OUTSIDE the disc rim, two front posts
    outside the rim, and a green serve stripe (visual) on the back wall's inner face.
    Root origin on the carousel axis at counter height; the serve direction is local +x."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _apply_root(stage, prim_path, translation, orientation, kinematic=True)
    co = float(cfg.contact_offset)
    x0, x1 = cfg.roof_x0, cfg.roof_x1
    _box(stage, prim_path, "roof", (x0 + x1) / 2, 0.0, cfg.clear_h + cfg.roof_t / 2,
         x1 - x0, cfg.roof_w, cfg.roof_t, cfg.color, co)
    wall_h = cfg.clear_h + cfg.roof_t
    _box(stage, prim_path, "back", x1 + cfg.roof_t / 2, 0.0, wall_h / 2,
         cfg.roof_t, cfg.roof_w, wall_h, cfg.color, co)
    for s, nm in ((1.0, "post_l"), (-1.0, "post_r")):
        _cyl(stage, prim_path, nm, cfg.post_x, s * cfg.post_y, cfg.clear_h / 2,
             cfg.post_r, cfg.clear_h, cfg.color, co)
    # serve stripe: visual only (green), centred on local +x, on the back wall inner face
    _box(stage, prim_path, "stripe", x1 - 0.003, 0.0, cfg.clear_h / 2 + 0.02,
         0.005, 0.030, 0.10, cfg.stripe_color, None)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A black octagonal bowl: floor disc + 8 vertical rim-wall boxes. Root origin at the
    floor slab centre (CoM low: MassAPI puts the mass at the body origin)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _apply_root(stage, prim_path, translation, orientation, kinematic=False,
                       mass=cfg.mass_props.mass, ang_damp=0.10, lin_damp=0.05)
    co = float(cfg.contact_offset)
    _cyl(stage, prim_path, "floor", 0.0, 0.0, 0.0, cfg.floor_r, cfg.floor_h, cfg.color, co)
    n_w = 8
    rw = cfg.floor_r - cfg.wall_t / 2 + 0.002  # wall mid-line; outer face ~ floor_r + t/2
    seg_l = 2 * rw * math.tan(math.pi / n_w) + 0.003
    for k in range(n_w):
        a = 2 * math.pi * k / n_w
        _box(stage, prim_path, f"wall_{k}", rw * math.cos(a), rw * math.sin(a),
             cfg.floor_h / 2 + cfg.wall_h / 2, cfg.wall_t, seg_l, cfg.wall_h,
             cfg.color, co, yaw=a)
    return root


def _mk_spawner(key: str, func: Callable, defaults: dict) -> Callable:
    """Build (once) and cache a RigidObjectSpawnerCfg subclass around `func`."""
    import isaaclab.sim as sim_utils  # noqa: F401  (kept for parity with sibling scenes)
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if key not in _SPAWNER_CACHE:
        ns = {"func": clone(func), **defaults}
        ns["__annotations__"] = {"func": Callable,
                                 **{k: type(v).__name__ for k, v in defaults.items()}}
        _SPAWNER_CACHE[key] = configclass(type(f"{key.title()}SpawnerCfg",
                                               (RigidObjectSpawnerCfg,), ns))
    return _SPAWNER_CACHE[key]


# ----- scene cfg ------------------------------------------------------------------------------------
@dataclass
class ServiceCarouselSceneCfg(BaseCfg):
    """Config for `ServiceCarouselScene`. Gate honesty margins are derived in the module
    docstring."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    serve_tol_deg: float = tunable(15.0)  # plate azimuth within this of the serve stripe (+x).
    # Start offset is 25-45 deg, so the reset/seed-strategy state is outside by construction.
    load_window_deg: float = tunable(40.0)  # loading window: plate azimuth within this of the
    # robot side (180 deg) — where the FETCHED and LOADED latches may fire.
    bowl_xy_tol: float = tunable(0.035)  # bowl centre to plate axis, plate frame (m). Fully-on
    # max ~27 mm; a 55 mm off-centre bowl rests flat bridging the pocket wall -> rejected.
    bowl_z_tol: float = tunable(0.010)  # |bowl bottom - plate top| below this (m)
    bowl_tilt_max_deg: float = tunable(15.0)  # bowl local +z within this of world-up
    plate_tilt_max_deg: float = tunable(10.0)  # plate axis within this of world-up
    pocket_r_tol: float = tunable(0.030)  # |plate radius from hub - pocket_mount_r| below this
    pocket_z_tol: float = tunable(0.008)  # |plate centre z - seated pocket z| below this
    settle_speed: float = tunable(0.05)  # max |lin v| of plate and bowls when judging (m/s)
    disc_settle_w: float = tunable(0.15)  # max disc |ang v| when judging (rad/s)
    latch_disc_w: float = tunable(0.50)  # FETCHED latches only while the disc is this slow
    latch_speed: float = tunable(0.10)  # LOADED latches only while the bowl is this slow

    # --- tunable: randomization (the task-family knobs) -------------------------------------------
    start_az_min_deg: float = tunable(25.0)  # plate start azimuth off the mark, lower bound
    start_az_max_deg: float = tunable(45.0)  # ... upper bound (sign random)
    bowl_jitter: float = tunable(0.015)  # uniform +- xy jitter per bowl at reset
    shuffle_bowls: bool = tunable(True)  # per-episode slot permutation (target identity)

    # --- tunable: placement (counter frame; intended arm base at (-0.45, 0, surface_z)) ----------
    surface_z: float = tunable(0.20)  # counter height; the arm base is mounted on the counter
    hub_xy: tuple = tunable((0.15, 0.0))  # carousel axis (fixture: NEVER re-posed — the
    # spawn-authored joint anchor is world-fixed)
    row_x: float = tunable(-0.22)  # bowl row x
    row_dy: float = tunable(0.15)  # bowl slot y spacing (slots at -dy, 0, +dy)

    # --- info: structure --------------------------------------------------------------------------
    bench_size: tuple = info((1.5, 1.3))  # kinematic counter slab top (x, y)
    disc_r: float = info(0.20)
    disc_h: float = info(0.020)
    disc_z_off: float = info(0.042)  # disc centre above the counter (joint anchor height)
    disc_mass: float = info(1.2)
    disc_ang_damp: float = info(0.30)  # passive brake: the carousel coasts to a stop
    peg_r: float = info(0.009)
    peg_h: float = info(0.070)
    peg_mount_r: float = info(0.170)
    pocket_mount_r: float = info(0.115)  # pocket centre radius from the disc axis (local +x)
    pocket_inner_r: float = info(0.078)  # pocket ring inner face (plate r 75 + 3 mm slack)
    pocket_wall_t: float = info(0.006)
    pocket_wall_h: float = info(0.012)  # wall top FLUSH with the seated plate top (no rim to pinch)
    ped_r: float = info(0.050)
    ped_h: float = info(0.030)
    clear_h: float = info(0.135)  # hutch roof lower face above the counter: pegs top 0.122,
    # loaded plate+bowl top 0.114 sweep beneath; no hand fits the 83 mm slot over the disc
    roof_x0: float = info(0.020)  # roof span, hub-local x (serve pocket at 0.115 is covered)
    roof_x1: float = info(0.235)
    roof_w: float = info(0.480)  # roof y width: side reach-in must travel > 0.13 m under it
    roof_t: float = info(0.012)
    post_x: float = info(0.040)
    post_y: float = info(0.225)  # posts at r 0.228 from the axis — 16 mm clear of the rim
    post_r: float = info(0.012)
    plate_r: float = info(0.075)
    plate_h: float = info(0.012)
    plate_mass: float = info(0.25)
    n_bowls: int = info(3)
    bowl_floor_r: float = info(0.048)
    bowl_floor_h: float = info(0.008)
    bowl_wall_t: float = info(0.007)
    bowl_wall_h: float = info(0.042)  # bowl total height 50 mm; rim wall is the pinch feature
    bowl_mass: float = info(0.15)
    contact_offset: float = info(0.002)
    disc_color: tuple = info((0.38, 0.26, 0.15))
    peg_color: tuple = info((0.62, 0.63, 0.66))
    pocket_wall_color: tuple = info((0.58, 0.42, 0.22))
    hub_color: tuple = info((0.45, 0.45, 0.48))
    hood_color: tuple = info((0.50, 0.36, 0.20))
    stripe_color: tuple = info((0.10, 0.75, 0.20))
    plate_color: tuple = info((0.93, 0.92, 0.88))
    bowl_color: tuple = info((0.06, 0.06, 0.07))
    bench_color: tuple = info((0.35, 0.35, 0.38))


# ----- scene -----------------------------------------------------------------------------------------
@SCENES.register("service_carousel")
class ServiceCarouselScene(BaseScene):
    cfg: ServiceCarouselSceneCfg

    def __init__(self, cfg: ServiceCarouselSceneCfg | None = None) -> None:
        super().__init__(cfg or ServiceCarouselSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, counter slab, the kinematic hub + hutch (fixtures, never re-posed),
        the carousel disc, the plate (nominally at the serve pocket) and the three bowls
        (reset() re-places the movables and samples the randomization)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z
        hx, hy = c.hub_xy

        disc_cfg = _mk_spawner("disc", _spawn_disc, {
            "disc_r": 0.20, "disc_h": 0.02, "peg_r": 0.009, "peg_h": 0.07,
            "peg_mount_r": 0.17, "pocket_mount_r": 0.115, "pocket_inner_r": 0.078,
            "wall_t": 0.006, "wall_h": 0.012, "ang_damp": 0.30, "contact_offset": 0.002,
            "color": (0.38, 0.26, 0.15), "peg_color": (0.62, 0.63, 0.66),
            "wall_color": (0.58, 0.42, 0.22)})
        hub_cfg = _mk_spawner("hub", _spawn_hub, {
            "ped_r": 0.05, "ped_h": 0.03, "contact_offset": 0.002,
            "color": (0.45, 0.45, 0.48)})
        hood_cfg = _mk_spawner("hood", _spawn_hood, {
            "clear_h": 0.135, "roof_x0": 0.020, "roof_x1": 0.235, "roof_w": 0.48,
            "roof_t": 0.012, "post_x": 0.040, "post_y": 0.225, "post_r": 0.012,
            "contact_offset": 0.002, "color": (0.50, 0.36, 0.20),
            "stripe_color": (0.10, 0.75, 0.20)})
        bowl_cfg = _mk_spawner("bowl", _spawn_bowl, {
            "floor_r": 0.048, "floor_h": 0.008, "wall_t": 0.007, "wall_h": 0.042,
            "contact_offset": 0.002, "color": (0.06, 0.06, 0.07)})

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
            "bench": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bench_size[0], c.bench_size[1], z0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.bench_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.05, 0.0, z0 / 2)),
            ),
            "hub": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Hub",
                spawn=hub_cfg(
                    mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, z0)),
            ),
            "hood": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Hood",
                spawn=hood_cfg(
                    mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, z0)),
            ),
            "disc": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Disc",
                spawn=disc_cfg(
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.disc_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, z0 + c.disc_z_off)),
            ),
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate",
                spawn=sim_utils.CylinderCfg(
                    radius=c.plate_r, height=c.plate_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.5),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.plate_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.plate_color),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.7, dynamic_friction=0.6, restitution=0.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hx + c.pocket_mount_r, hy,
                         z0 + c.disc_z_off + c.disc_h / 2 + c.plate_h / 2 + 0.002)),
            ),
        }
        for i in range(c.n_bowls):
            out[f"bowl_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl_" + str(i),
                spawn=bowl_cfg(
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.row_x, (i - 1) * c.row_dy, z0 + c.bowl_floor_h / 2 + 0.002)),
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

    # ----- lifecycle --------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles, author the revolute joint (per env; body0 = the kinematic hub, which
        is never re-posed, so the world-fixed anchor is exactly the carousel axis), allocate
        the episode identity and the progress latches."""
        super().bind(env)
        c = self.cfg
        n = env.num_envs
        dev = env.device
        self.hub: RigidObject = env.iscene["hub"]
        self.hood: RigidObject = env.iscene["hood"]
        self.disc: RigidObject = env.iscene["disc"]
        self.plate: RigidObject = env.iscene["plate"]
        self.bowls: list[RigidObject] = [env.iscene[f"bowl_{i}"] for i in range(c.n_bowls)]
        self.env_origins = env.iscene.env_origins
        self._author_joint()
        # episode identity / randomization readback
        self.mid_idx = torch.ones(n, dtype=torch.long, device=dev)  # bowl body in the MIDDLE slot
        self.slot_of = torch.zeros(n, c.n_bowls, dtype=torch.long, device=dev)
        self.yaw0 = torch.zeros(n, device=dev)  # plate start azimuth (rad, signed)
        # progress latches (post_step; cleared per reset)
        self.ever_fetched = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_loaded = torch.zeros(n, dtype=torch.bool, device=dev)

    def _author_joint(self) -> None:
        """Per env: one revolute joint (axis Z) between the kinematic hub and the disc, on the
        carousel axis at `disc_z_off` above the counter. Continuous (no limits). The joint
        pair (hub pedestal <-> disc) is collision-filtered by PhysX; the HUTCH is a separate
        body, so disc/pegs/payload vs hutch collisions stay LIVE."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/carousel_pivot")
            j.CreateBody0Rel().SetTargets([f"{base}/Hub"])
            j.CreateBody1Rel().SetTargets([f"{base}/Disc"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.disc_z_off))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the plate's start azimuth (signed 25-45 deg off the serve
        mark, under the hutch), rotate the disc there, seat the plate in the pocket, deal
        the three bowls over the row slots by a random permutation with jitter + free yaw,
        clear the latches. The hub/hutch fixtures are never re-posed (joint anchor)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        z0 = c.surface_z
        hx, hy = c.hub_xy

        # --- disc: start azimuth (the pocket is disc-local +x, so pocket azimuth = disc yaw)
        mag = math.radians(c.start_az_min_deg) + torch.rand(m, device=dev) * (
            math.radians(c.start_az_max_deg) - math.radians(c.start_az_min_deg))
        sign = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        yaw0 = mag * sign
        self.yaw0[env_ids] = yaw0
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = hx
        st[:, 1] = hy
        st[:, 2] = z0 + c.disc_z_off
        st[:, 3] = torch.cos(yaw0 / 2)
        st[:, 6] = torch.sin(yaw0 / 2)
        st[:, 0:3] += origin
        self.disc.write_root_state_to_sim(st, env_ids)

        # --- plate: seated in the pocket at the start azimuth
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = hx + c.pocket_mount_r * torch.cos(yaw0)
        st[:, 1] = hy + c.pocket_mount_r * torch.sin(yaw0)
        st[:, 2] = z0 + c.disc_z_off + c.disc_h / 2 + c.plate_h / 2 + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.plate.write_root_state_to_sim(st, env_ids)

        # --- bowls: random slot permutation + jitter + free yaw
        if c.shuffle_bowls:
            slot_of = torch.rand(m, c.n_bowls, device=dev).argsort(dim=1)  # bowl -> slot
        else:
            slot_of = torch.arange(c.n_bowls, device=dev).expand(m, c.n_bowls).clone()
        self.slot_of[env_ids] = slot_of
        self.mid_idx[env_ids] = (slot_of == 1).float().argmax(dim=1)
        for b, bowl in enumerate(self.bowls):
            slot = slot_of[:, b].float() - 1.0  # -1, 0, +1
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.row_x
            st[:, 1] = slot * c.row_dy
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.bowl_jitter
            st[:, 2] = z0 + c.bowl_floor_h / 2 + 0.002
            half = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            bowl.write_root_state_to_sim(st, env_ids)

        for latch in (self.ever_fetched, self.ever_loaded):
            latch[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch progress milestones at sim rate. Velocity-gated (disc / bowl) so states flown
        through latch nothing; latched credit survives later mishaps, so along a correct
        trajectory the printed score never decreases."""
        c = self.cfg
        disc_slow = self.disc.data.root_ang_vel_w[:, 2].abs() < c.latch_disc_w
        fetched = self.plate_in_load_window() & disc_slow
        self.ever_fetched |= fetched
        bowl_speed = torch.stack(
            [b.data.root_lin_vel_w.norm(dim=-1) for b in self.bowls], dim=1)
        slow_mid = bowl_speed.gather(1, self.mid_idx.unsqueeze(1)).squeeze(1) < c.latch_speed
        self.ever_loaded |= self.target_on_plate() & self.plate_in_load_window() & slow_mid

    # ----- state (full, restorable) ------------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "disc": self.disc.data.root_state_w[env_ids].clone(),
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "bowls": [b.data.root_state_w[env_ids].clone() for b in self.bowls],
            "mid_idx": self.mid_idx[env_ids].clone(),
            "slot_of": self.slot_of[env_ids].clone(),
            "yaw0": self.yaw0[env_ids].clone(),
            "latches": torch.stack([self.ever_fetched[env_ids],
                                    self.ever_loaded[env_ids]], dim=1),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.disc.write_root_state_to_sim(state["disc"], env_ids)
        self.plate.write_root_state_to_sim(state["plate"], env_ids)
        for b, st in zip(self.bowls, state["bowls"]):
            b.write_root_state_to_sim(st, env_ids)
        self.mid_idx[env_ids] = state["mid_idx"]
        self.slot_of[env_ids] = state["slot_of"]
        self.yaw0[env_ids] = state["yaw0"]
        lat = state["latches"]
        self.ever_fetched[env_ids] = lat[:, 0]
        self.ever_loaded[env_ids] = lat[:, 1]

    # ----- description -------------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A kitchen counter. Nearest you, THREE identical BLACK bowls "
            f"(~{2 * (c.bowl_floor_r + c.bowl_wall_t) * 1000:.0f} mm across, "
            f"{(c.bowl_floor_h + c.bowl_wall_h) * 1000 + 4:.0f} mm tall, "
            f"{c.bowl_wall_t * 1000:.0f} mm rim wall) stand in a row, side by side. Beyond them "
            f"spins a wooden SERVICE CAROUSEL: a dark round disc "
            f"({2 * c.disc_r * 1000:.0f} mm across) on a center pivot, carrying four upright "
            f"gray PEG handles ({c.peg_h * 1000:.0f} mm tall) near its rim and one shallow "
            f"round POCKET holding a round WHITE plate ({2 * c.plate_r * 1000:.0f} mm across, "
            f"flush inside the pocket — it cannot be picked out). The FAR sector of the "
            f"carousel is covered by a wooden service HUTCH (a low roof "
            f"{(c.clear_h - c.disc_z_off - c.disc_h / 2) * 1000:.0f} mm above the disc, with a "
            f"back wall); a GREEN vertical stripe on the hutch's back wall marks the SERVE "
            f"position. The plate starts parked under the hutch but OFF the green mark (25-45 "
            f"degrees to one side, varying per episode); the bowl row and which bowl stands "
            f"where also vary per episode.\n"
            f"Goal: serve the MIDDLE bowl of the row — the one standing between the other two "
            f"— on the plate, parked at the serve mark. The hutch is too low to reach under, "
            f"so: rotate the carousel (push the peg handles; the disc spins freely on its "
            f"pivot and coasts to a stop) until the plate comes around to the open side near "
            f"the bowls; set the MIDDLE bowl upright onto the plate, near its centre; then "
            f"rotate the carousel back — with the bowl riding the plate — until the plate is "
            f"parked under the hutch, centred on the green stripe (within "
            f"{c.serve_tol_deg:.0f} degrees). The bowl must end upright on the plate and the "
            f"plate seated in its pocket; a bowl that tips over or slides off during the "
            f"ride, a wrong (outer) bowl, a bowl set on the bare disc, or a plate parked off "
            f"the stripe does not count. Everything must come to rest."
        )

    def instruction(self) -> str:
        return (
            "Rotate the carousel by its pegs to bring the white plate out from under the "
            "hutch, set the MIDDLE black bowl of the three upright onto the plate, then "
            "rotate the carousel back until the plate parks under the hutch centred on the "
            "green stripe."
        )

    # ----- geometric predicates ------------------------------------------------------------------------
    def _up_z(self, quat: torch.Tensor) -> torch.Tensor:
        """z-component of a body's local +z in world (...,) for tilt gates."""
        from isaaclab.utils.math import quat_apply

        shape = quat.shape[:-1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(*shape, 3)
        return quat_apply(quat.reshape(-1, 4), ez.reshape(-1, 3)).reshape(*shape, 3)[..., 2]

    def disc_yaw(self) -> torch.Tensor:
        """(N,) disc yaw about the pivot (rad, wrapped). The joint admits only z-rotation, so
        the root quat is qz(psi); the pocket is disc-local +x, so pocket azimuth = yaw."""
        q = self.disc.data.root_quat_w
        psi = 2.0 * torch.atan2(q[:, 3], q[:, 0])
        return torch.remainder(psi + math.pi, 2 * math.pi) - math.pi

    def plate_azimuth(self) -> torch.Tensor:
        """(N,) the PLATE's azimuth around the carousel axis (rad; 0 = serve stripe = +x,
        pi = the open loading side). Judged from the plate body itself, not the disc."""
        rel = self.plate.data.root_pos_w - self.hub.data.root_pos_w
        return torch.atan2(rel[:, 1], rel[:, 0])

    def plate_in_pocket(self) -> torch.Tensor:
        """(N,) bool: plate upright, at pocket radius from the axis, at seated pocket height."""
        c = self.cfg
        rel = self.plate.data.root_pos_w - self.hub.data.root_pos_w
        r = rel[:, :2].norm(dim=-1)
        seat_z = c.disc_z_off + c.disc_h / 2 + c.plate_h / 2
        near_r = (r - c.pocket_mount_r).abs() < c.pocket_r_tol
        near_z = (rel[:, 2] - seat_z).abs() < c.pocket_z_tol
        upright = self._up_z(self.plate.data.root_quat_w).clamp(-1, 1) >= \
            math.cos(math.radians(c.plate_tilt_max_deg))
        return near_r & near_z & upright

    def _az_within(self, center: float, half_deg: float) -> torch.Tensor:
        d = self.plate_azimuth() - center
        d = torch.remainder(d + math.pi, 2 * math.pi) - math.pi
        return d.abs() < math.radians(half_deg)

    def plate_in_load_window(self) -> torch.Tensor:
        """(N,) bool: plate in the pocket, swung around to the open loading side."""
        return self.plate_in_pocket() & self._az_within(math.pi, self.cfg.load_window_deg)

    def plate_delivered(self) -> torch.Tensor:
        """(N,) bool: plate in the pocket, parked on the serve stripe (azimuth 0)."""
        return self.plate_in_pocket() & self._az_within(0.0, self.cfg.serve_tol_deg)

    def bowls_on_plate(self) -> torch.Tensor:
        """(N, B) bool: bowl upright, its bottom at the plate top, centred within
        `bowl_xy_tol` of the plate axis (plate frame), plate upright."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        n, nb = self.env.num_envs, c.n_bowls
        pos = torch.stack([b.data.root_pos_w for b in self.bowls], dim=1)  # (N,B,3)
        quat = torch.stack([b.data.root_quat_w for b in self.bowls], dim=1)
        pq = self.plate.data.root_quat_w[:, None, :].expand(n, nb, 4).reshape(-1, 4)
        pp = self.plate.data.root_pos_w[:, None, :]
        loc = quat_apply_inverse(pq, (pos - pp).reshape(-1, 3)).reshape(n, nb, 3)
        near = loc[:, :, :2].norm(dim=-1) < c.bowl_xy_tol
        upright = self._up_z(quat).clamp(-1, 1) >= math.cos(math.radians(c.bowl_tilt_max_deg))
        bowl_bottom = pos[:, :, 2] - c.bowl_floor_h / 2
        plate_top = (self.plate.data.root_pos_w[:, 2] + c.plate_h / 2).unsqueeze(1)
        on_top = (bowl_bottom - plate_top).abs() < c.bowl_z_tol
        plate_up = self._up_z(self.plate.data.root_quat_w).clamp(-1, 1) >= \
            math.cos(math.radians(c.plate_tilt_max_deg))
        return near & upright & on_top & plate_up.unsqueeze(1)

    def target_on_plate(self) -> torch.Tensor:
        """(N,) bool: the episode's MIDDLE bowl sits on the plate."""
        return self.bowls_on_plate().gather(1, self.mid_idx.unsqueeze(1)).squeeze(1)

    def settled(self) -> torch.Tensor:
        """(N,) bool: plate and every bowl slow, disc angular speed below `disc_settle_w`."""
        c = self.cfg
        still = self.plate.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        for b in self.bowls:
            still &= b.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return still & (self.disc.data.root_ang_vel_w[:, 2].abs() < c.disc_settle_w)

    def success(self) -> torch.Tensor:
        """(N,) bool: middle bowl on the plate + plate parked on the serve stripe in its
        pocket + everything settled. Pure state predicate."""
        return self.target_on_plate() & self.plate_delivered() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float 0..1 — additive latched milestones (monotone along the demonstrated
        solution): +0.25 the plate fetched into the loading window, +0.35 the middle bowl
        seated on the plate in the loading window; 1.0 iff success()."""
        s = torch.zeros(self.env.num_envs, device=self.env.device)
        s = s + 0.25 * self.ever_fetched.float()
        s = s + 0.35 * self.ever_loaded.float()
        return torch.where(self.success(), torch.ones_like(s), s)


# Scene-level task (robot="null"): solve.py is the teleport certificate; the intended
# embodiment (single Franka + parallel jaw) is argued in TASK.md.
register_env("simgen", lambda: EnvCfg(scene="service_carousel", robot="null"))
