"""CargoTramScene — load a cargo cube into an open-top cart, push the loaded cart up a
ramp and park it in a roofed summit pocket (sim_gen task `base_i257`).

Derived from pick_place/base, but STRATEGICALLY different: the seed is direct
grasp-and-carry — close the jaw on a 4 cm cube and translate it through a sequence of
free-space waypoints to a goal pose (one grasp, one guided transport, judged by
waypoint tracking on the held object). Here the graspable cube is only the CARGO of a
larger vehicle: a yellow open-top CART that must be loaded on the flat loading zone,
then pushed (never lifted) up a ramp into a summit POCKET whose roof canopy leaves
only a 31 mm slit above the parked cart's walls — a 40 mm cube physically CANNOT be
loaded after parking, so the load-then-deliver order is forced by geometry, not by the
rubric. A blue DECOY cube of identical size must stay out of the cart. A solver needs
a different PLAN from the seed (indirect delivery: the judged object rides a vehicle
that is itself pushed, and the last leg is a push along a walled channel, not a carry)
and a different CODE STRUCTURE (body-frame containment in a moving carrier + carrier
pose windows + an exclusion predicate — not gripper-distance waypoint tracking).

The order-forcing and drivability are real physics, asserted in `__post_init__` from
the cfg's own numbers:
  - canopy underside minus parked-cart wall top = 31 mm < the 40 mm cube edge (a cube
    cannot pass the slit in ANY orientation — a cube's minimum width is its edge), and
    the diagonal aperture from the canopy front edge to the parked bucket rim is
    also < the cube edge, so the bucket mouth is unreachable once the cart is parked;
  - the same canopy clears the cart during the crest transit even in the worst
    cantilevered pose (crest_z + cart_len*sin(theta) + cart_h*cos(theta) + margin);
  - tan(ramp angle) exceeds the cart/station pair friction by a wide margin, so the
    cart cannot park itself mid-ramp — it must be actively pushed the whole climb;
  - the crest drops 8 mm onto the pocket floor: the cart tips in and seats against
    the back wall; the pocket is only 6 mm longer than the cart.

success(): the RED cube rests inside the cart's bucket (cart BODY-frame containment)
AND the blue decoy does not, AND the cart is parked in the pocket (env-frame x/z
windows + level), AND cart and cargo are settled. Identity is enforced by name: the
decoy in the bucket, an empty parked cart, or the cargo dumped on the pocket floor
all count for nothing.

score() is graded and latched (credit never evaporates): 0.20 cargo ever loaded +
0.20 loaded cart ever past mid-ramp + 0.20 loaded cart ever at the parking windows
(cap 0.60); 1.0 iff success(). The null policy scores ~0 (cart spawns on the flat,
cubes on the open floor beside the channel).

Per-episode randomization (readback-verified in smoke): the cargo/decoy pair occupies
an ordered pair of 2 out of 4 floor slots (12 assignments) with xy jitter + free yaw,
and the cart's starting x on the flat is drawn from a band. Assets are fully
procedural compound-box spawners (one rigid body each). Heavy imports (isaaclab, pxr)
are deferred so importing this module — and registering the scene — stays app-free.
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


def _box(stage, path: str, size, center, color, contact_offset: float, material=None,
         orientation=None) -> None:
    """Author one box child prim (translate -> orient -> scale, authored once — the
    duplicate xformOp trap is avoided by never re-authoring an existing prim's ops)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float,
                kinematic: bool = False, com_at_origin: bool = False):
    """Author one rigid-body root Xform with the standard physics armor (zero
    sleep/stabilization thresholds: a sleeping body silently ignores applied wrenches,
    which the solve/smoke force probes depend on). `com_at_origin` pins the center of
    mass to the body origin explicitly (the cart's origin is its bottom center, so
    the push wrench at the CoM adds no pitch torque and the cart is bottom-heavy)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    mapi = UsdPhysics.MassAPI.Apply(root)
    mapi.CreateMassAttr(float(mass))
    if com_at_origin:
        mapi.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return root, pxrb


def _spawn_station(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC tramway station, one compound rigid body, origin at the env
    origin on the ground: flat loading slab, inclined ramp slab (oriented box), summit
    pocket slab (8 mm below the crest), two full-length side walls, back wall, and
    the roof CANOPY over the pocket. All surfaces bind one SLICK material (the
    cart/station friction pair must stay far below tan(ramp angle) — asserted)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 30.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    c = cfg
    co = c.contact_offset
    slab = (0.55, 0.55, 0.58)
    wallc = (0.35, 0.35, 0.40)
    roofc = (0.20, 0.20, 0.24)

    fx0, fx1 = c.flat_x
    ft = c.flat_top
    # flat loading slab
    _box(stage, f"{prim_path}/flat", (fx1 - fx0, 2 * c.wall_inner_y, ft),
         ((fx0 + fx1) / 2, 0.0, ft / 2), slab, co, material=mat)
    # ramp slab: top surface from (fx1, ft) to the crest, extended down-slope under
    # the flat so the seam is a smooth kink
    th = math.atan2(c.ramp_rise, c.ramp_run)
    sn, cs = math.sin(th), math.cos(th)
    ext = 0.03
    length = c.ramp_run / cs + ext
    ax, az = fx1 - ext * cs, ft - ext * sn          # extended lower end (top surface)
    bx, bz = c.crest_x, c.crest_z                    # crest (top surface)
    tcx, tcz = (ax + bx) / 2, (az + bz) / 2          # top-surface center
    cx, cz = tcx + 0.010 * sn, tcz - 0.010 * cs      # box center (thickness 0.02)
    half = th / 2
    _box(stage, f"{prim_path}/ramp", (length, 2 * c.wall_inner_y, 0.020),
         (cx, 0.0, cz), slab, co, material=mat,
         orientation=(math.cos(half), 0.0, -math.sin(half), 0.0))
    # pocket slab: solid up to pocket_top; starts where the ramp surface is already
    # above it (crest_x - 0.02), so nothing protrudes through the ramp surface
    px0, px1 = c.crest_x - 0.020, c.back_x[1]
    _box(stage, f"{prim_path}/pocket", (px1 - px0, 2 * c.wall_inner_y, c.pocket_top),
         ((px0 + px1) / 2, 0.0, c.pocket_top / 2), slab, co, material=mat)
    # side walls (full channel length, up to the canopy underside)
    wx0, wx1 = fx0, c.back_x[1]
    for sgn, nm in ((1.0, "wall_py"), (-1.0, "wall_ny")):
        _box(stage, f"{prim_path}/{nm}",
             (wx1 - wx0, c.wall_t, c.wall_top),
             ((wx0 + wx1) / 2, sgn * (c.wall_inner_y + c.wall_t / 2), c.wall_top / 2),
             wallc, co, material=mat)
    # back wall (between the side walls)
    _box(stage, f"{prim_path}/back", (c.back_x[1] - c.back_x[0], 2 * c.wall_inner_y,
                                      c.wall_top),
         ((c.back_x[0] + c.back_x[1]) / 2, 0.0, c.wall_top / 2), wallc, co,
         material=mat)
    # roof canopy over the pocket (rests on the wall tops, spans the full width)
    cw = 2 * (c.wall_inner_y + c.wall_t)
    _box(stage, f"{prim_path}/canopy",
         (c.back_x[1] - c.canopy_x0, cw, c.canopy_t),
         ((c.canopy_x0 + c.back_x[1]) / 2, 0.0, c.wall_top + c.canopy_t / 2),
         roofc, co, material=mat)
    return root


def _spawn_cart(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The YELLOW open-top cart, one dynamic rigid body, origin at the BOTTOM CENTER
    (and the authored CoM pinned there: the push wrench adds no pitch torque and the
    cart is maximally stable): floor slab + four bucket walls."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass,
                             com_at_origin=True)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.10)
    mat = _material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    c = cfg
    co = c.contact_offset
    yellow, rim = (0.90, 0.75, 0.10), (0.75, 0.60, 0.08)
    L, h, wt, ftk = c.length, c.height, c.wall_t, c.floor_t
    wall_h = h - ftk
    wz = ftk + wall_h / 2
    boxes = [
        ("floor", (L, L, ftk), (0.0, 0.0, ftk / 2), yellow),
        ("wall_px", (wt, L, wall_h), (+(L - wt) / 2, 0.0, wz), rim),
        ("wall_nx", (wt, L, wall_h), (-(L - wt) / 2, 0.0, wz), rim),
        ("wall_py", (L - 2 * wt, wt, wall_h), (0.0, +(L - wt) / 2, wz), rim),
        ("wall_ny", (L - 2 * wt, wt, wall_h), (0.0, -(L - wt) / 2, wz), rim),
    ]
    for name, size, center, col in boxes:
        _box(stage, f"{prim_path}/{name}", size, center, col, co, material=mat)
    return root


def _spawn_cube(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One 40 mm cube (graspable by any parallel jaw); color distinguishes the RED
    cargo from the BLUE decoy."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.10)
    mat = _material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    _box(stage, f"{prim_path}/body", (cfg.edge, cfg.edge, cfg.edge), (0.0, 0.0, 0.0),
         tuple(cfg.color), cfg.contact_offset, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "station" not in _SPAWNER_CACHE:

        @configclass
        class StationSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_station)
            flat_x: tuple = (0.14, 0.445)
            flat_top: float = 0.020
            ramp_run: float = 0.32
            ramp_rise: float = 0.08
            crest_x: float = 0.765
            crest_z: float = 0.100
            pocket_top: float = 0.092
            back_x: tuple = (0.863, 0.885)
            wall_inner_y: float = 0.054
            wall_t: float = 0.020
            wall_top: float = 0.213
            canopy_x0: float = 0.788
            canopy_t: float = 0.020
            mu_static: float = 0.12
            mu_dynamic: float = 0.10
            contact_offset: float = 0.002

        @configclass
        class CartSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cart)
            mass: float = 0.25
            length: float = 0.092
            height: float = 0.090
            wall_t: float = 0.008
            floor_t: float = 0.010
            mu_static: float = 0.12
            mu_dynamic: float = 0.10
            contact_offset: float = 0.002

        @configclass
        class CubeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cube)
            mass: float = 0.06
            edge: float = 0.04
            color: tuple = (0.85, 0.12, 0.12)
            mu_static: float = 0.60
            mu_dynamic: float = 0.50
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(station=StationSpawnerCfg, cart=CartSpawnerCfg,
                              cube=CubeSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CargoTramSceneCfg(BaseCfg):
    """Config for `CargoTramScene`. The order-forcing (a parked cart cannot be
    loaded), the transit clearance, and the no-self-parking friction margin are
    asserted in `__post_init__` from these numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    park_x: tuple = tunable((0.8125, 0.8235))  # cart-center window in the pocket, env x
    park_z: tuple = tunable((0.084, 0.108))    # ... env z (on-floor rest = 0.092)
    park_y_abs: float = tunable(0.02)          # |cart y| bound (channel keeps it ~<8 mm)
    upright_deg: float = tunable(10.0)         # cart z-axis within this cone of vertical
    in_xy: float = tunable(0.034)              # cube-in-bucket |x|,|y| bound (cart frame)
    in_z: tuple = tunable((0.014, 0.078))      # cube-in-bucket z window (cart frame)
    climb_x: float = tunable(0.60)             # loaded cart ever past this x -> climb latch
    settle_lin: float = tunable(0.08)          # max cart/cargo |lin vel| at judging (m/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    cart_start_x: tuple = tunable((0.21, 0.33))  # cart center start band on the flat
    cube_slots: tuple = tunable(((0.20, -0.20), (0.33, -0.20),
                                 (0.20, -0.33), (0.33, -0.33)))
    cube_jitter: float = tunable(0.025)   # +-xy jitter per cube at its drawn slot
    cube_yaw_deg: float = tunable(180.0)  # +- free yaw per cube

    # --- info: structure (the geometry the spawners author) ----------------------------------
    flat_x: tuple = info((0.14, 0.445))   # flat loading slab x span
    flat_top: float = info(0.020)         # flat slab top height
    ramp_run: float = info(0.32)          # ramp horizontal run (flat end -> crest)
    ramp_rise: float = info(0.08)         # ramp rise (tan = 0.25, angle ~14 deg)
    crest_x: float = info(0.765)          # crest = flat_x[1] + ramp_run (asserted)
    crest_z: float = info(0.100)          # crest top = flat_top + ramp_rise (asserted)
    pocket_top: float = info(0.092)       # pocket floor top (8 mm below the crest)
    back_x: tuple = info((0.863, 0.885))  # back wall x span (inner face = pocket end)
    wall_inner_y: float = info(0.054)     # side-wall inner faces at +-this (108 mm channel)
    wall_t: float = info(0.020)           # side-wall thickness
    wall_top: float = info(0.213)         # side/back wall top = canopy underside
    canopy_x0: float = info(0.788)        # canopy front edge (past the crest)
    canopy_t: float = info(0.020)         # canopy slab thickness
    cart_len: float = info(0.092)         # cart outer footprint (square)
    cart_h: float = info(0.090)           # cart outer height (wall top)
    cart_wall_t: float = info(0.008)      # bucket wall thickness (interior 76 mm)
    cart_floor_t: float = info(0.010)     # bucket floor thickness
    cart_mass: float = info(0.25)
    cube_edge: float = info(0.04)
    cube_mass: float = info(0.06)
    mu_slick: float = info(0.12)          # cart & station static friction (pair 0.12)
    contact_offset: float = info(0.002)

    def __post_init__(self) -> None:
        c = self
        th = math.atan2(c.ramp_rise, c.ramp_run)
        sn, cs = math.sin(th), math.cos(th)
        # -- geometry consistency --
        assert abs(c.flat_x[1] + c.ramp_run - c.crest_x) < 1e-9
        assert abs(c.flat_top + c.ramp_rise - c.crest_z) < 1e-9
        # -- the pocket seats the cart: 6 mm slack, 8 mm drop-in step --
        pocket_len = c.back_x[0] - c.crest_x
        assert 0.004 <= pocket_len - c.cart_len <= 0.020, "pocket must barely fit the cart"
        assert 0.005 <= c.crest_z - c.pocket_top <= 0.015, "crest step must drop the cart in"
        # -- the bucket takes one cube in any yaw; two cubes must STACK, not spread --
        interior = c.cart_len - 2 * c.cart_wall_t
        assert interior > c.cube_edge * math.sqrt(2.0) + 0.004, "yawed cube must fit the bucket"
        assert 2 * c.cube_edge > interior, "two cubes must stack (side-by-side must not fit)"
        # -- ORDER-FORCING: a parked cart cannot be loaded --
        gap = c.wall_top - (c.pocket_top + c.cart_h)  # canopy slit above parked walls
        assert 0.010 <= gap <= c.cube_edge - 0.0075, \
            f"canopy slit {gap:.3f} must be a real gap yet block a {c.cube_edge:.3f} cube"
        mouth_lo = c.park_x[0] - interior / 2  # bucket mouth near edge, worst (least) parked x
        aperture = math.hypot(c.canopy_x0 - mouth_lo, gap)
        assert aperture <= c.cube_edge - 0.005, \
            f"canopy-to-rim aperture {aperture:.4f} must block the cube"
        assert c.canopy_x0 - mouth_lo <= c.cube_edge - 0.004, \
            "exposed bucket-mouth strip must be too narrow for a vertical drop"
        # -- but the DRIVING cart clears the canopy, even cantilevered at the crest --
        worst = c.crest_z + c.cart_len * sn + c.cart_h * cs
        assert c.wall_top >= worst + 0.002, \
            f"canopy {c.wall_top:.3f} must clear the crest transit {worst:.4f}"
        assert c.canopy_x0 >= c.crest_x + 0.02, "canopy must start past the crest"
        # -- no self-parking on the ramp: gravity beats pair friction --
        assert math.tan(th) >= c.mu_slick + 0.08, "ramp must be too steep to park on"
        # -- channel guides but never pinches the cart --
        chan = 2 * c.wall_inner_y - c.cart_len
        assert 0.012 <= chan <= 0.030, "channel slack must guide, not pinch"
        assert c.park_y_abs >= chan / 2, "any in-channel y must be inside the y bound"
        # -- park windows are honest by construction --
        rest_x = c.back_x[0] - c.cart_len / 2  # seated against the back wall
        assert c.park_x[0] <= rest_x <= c.park_x[1], "seated rest must be inside the x window"
        assert c.park_x[0] - c.cart_len / 2 >= c.crest_x + 0.0005, \
            "a crest-bridging cart must fall below the x window"
        assert c.park_z[0] <= c.pocket_top <= c.park_z[1], "floor rest inside the z window"
        assert c.park_z[1] < c.pocket_top + c.cube_edge, \
            "a cart perched on a stray cube must be above the z window"
        assert c.park_z[1] < c.crest_z + 0.010, "an on-crest perch must fail x, not hide in z"
        # -- bucket containment windows are honest by construction --
        floor_rest = c.cart_floor_t + c.cube_edge / 2
        assert c.in_z[0] < floor_rest < c.in_z[1], "on-floor cube inside the z window"
        assert floor_rest + c.cube_edge < c.in_z[1] + c.cube_edge / 2, \
            "a second stacked cube must still be judgeable"  # stacked center 0.070 < 0.078
        assert c.in_z[1] < c.cart_h - 0.008, "a cube perched on the rim must fail z"
        assert c.in_xy >= interior / 2 - c.cube_edge / 2, \
            "a wall-hugging in-bucket rest must pass xy"
        assert c.in_xy < interior / 2, "the xy bound must stay inside the bucket"
        assert c.in_xy < c.cart_len / 2 + c.cube_edge / 2 - 0.010, \
            "a cube hugging the OUTSIDE of a wall must fail xy"
        # -- randomization bands stay legal --
        assert c.cart_start_x[0] - c.cart_len / 2 >= c.flat_x[0] + 0.005
        assert c.cart_start_x[1] + c.cart_len / 2 <= c.flat_x[1] - 0.020
        diag_half = c.cube_edge * math.sqrt(2.0) / 2
        for sx, sy in c.cube_slots:
            assert abs(sy) - c.cube_jitter - diag_half > c.wall_inner_y + c.wall_t + 0.004, \
                "cube slots must stay clear of the channel walls"
            assert 0.12 < sx < 0.42, "cube slots must stay in the loading apron"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("cargo_tram")
class CargoTramScene(BaseScene):
    cfg: CargoTramSceneCfg

    def __init__(self, cfg: CargoTramSceneCfg | None = None) -> None:
        super().__init__(cfg or CargoTramSceneCfg())

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
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "station": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Station",
                spawn=sp["station"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    flat_x=c.flat_x, flat_top=c.flat_top, ramp_run=c.ramp_run,
                    ramp_rise=c.ramp_rise, crest_x=c.crest_x, crest_z=c.crest_z,
                    pocket_top=c.pocket_top, back_x=c.back_x,
                    wall_inner_y=c.wall_inner_y, wall_t=c.wall_t, wall_top=c.wall_top,
                    canopy_x0=c.canopy_x0, canopy_t=c.canopy_t,
                    mu_static=c.mu_slick, mu_dynamic=c.mu_slick - 0.02,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "cart": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cart",
                spawn=sp["cart"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cart_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.cart_mass, length=c.cart_len, height=c.cart_h,
                    wall_t=c.cart_wall_t, floor_t=c.cart_floor_t,
                    mu_static=c.mu_slick, mu_dynamic=c.mu_slick - 0.02,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.27, 0.0, c.flat_top + 0.001)),
            ),
            "cargo": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cargo",
                spawn=sp["cube"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.cube_mass, edge=c.cube_edge, color=(0.85, 0.12, 0.12),
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cube_slots[0][0], c.cube_slots[0][1], c.cube_edge / 2 + 0.002)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=sp["cube"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.cube_mass, edge=c.cube_edge, color=(0.15, 0.35, 0.85),
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cube_slots[1][0], c.cube_slots[1][1], c.cube_edge / 2 + 0.002)),
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
        n, dev = env.num_envs, env.device
        self.station: RigidObject = env.iscene["station"]
        self.cart: RigidObject = env.iscene["cart"]
        self.cargo: RigidObject = env.iscene["cargo"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        # progress latches (post_step): cargo ever loaded, loaded cart ever past
        # mid-ramp, loaded cart ever at the parking windows
        self.load_latch = torch.zeros(n, device=dev)
        self.climb_latch = torch.zeros(n, device=dev)
        self.dock_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: cart level on the flat at a drawn x, cargo/decoy at an
        ordered pair of 2-of-4 floor slots with xy jitter + free yaw; latches
        zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.cart_start_x[0] + torch.rand(m, device=dev) * (
            c.cart_start_x[1] - c.cart_start_x[0])
        st[:, 2] = c.flat_top + 0.001
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.cart.write_root_state_to_sim(st, env_ids)

        # ordered distinct slot pair: rank 4 uniforms, cargo takes the argsort's
        # first column, decoy the second (12 equally-likely assignments)
        slots = torch.tensor(c.cube_slots, device=dev)  # (4, 2)
        rank = torch.rand(m, len(c.cube_slots), device=dev).argsort(dim=1)
        yaw_amp = math.radians(c.cube_yaw_deg)
        for k, cube in ((0, self.cargo), (1, self.decoy)):
            sl = slots[rank[:, k]]  # (m, 2)
            cst = torch.zeros(m, 13, device=dev)
            cst[:, 0:2] = sl + (torch.rand(m, 2, device=dev) * 2 - 1) * c.cube_jitter
            cst[:, 2] = c.cube_edge / 2 + 0.002
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            cst[:, 3] = torch.cos(half)
            cst[:, 6] = torch.sin(half)
            cst[:, 0:3] += origin
            cube.write_root_state_to_sim(cst, env_ids)

        self.load_latch[env_ids] = 0.0
        self.climb_latch[env_ids] = 0.0
        self.dock_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"cart": self.cart, "cargo": self.cargo, "decoy": self.decoy}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "load_latch": self.load_latch[env_ids].clone(),
            "climb_latch": self.climb_latch[env_ids].clone(),
            "dock_latch": self.dock_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"cart": self.cart, "cargo": self.cargo, "decoy": self.decoy}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.load_latch[env_ids] = state["load_latch"]
        self.climb_latch[env_ids] = state["climb_latch"]
        self.dock_latch[env_ids] = state["dock_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A walled tramway CHANNEL (interior about 11 cm wide, walls on both "
            "sides) runs along +x across the floor: first a FLAT loading zone, then "
            "a straight RAMP climbing about 14 degrees, then — just past the crest — "
            "a shallow summit POCKET whose floor sits 8 mm BELOW the crest, ending "
            "at a back wall. A dark roof CANOPY covers the pocket from above; its "
            "underside is 21.3 cm up and its front edge starts just past the crest, "
            "so the flat and the ramp are open to the sky but the pocket is roofed.\n"
            "On the flat sits a YELLOW open-top CART (9.2 cm square, 9 cm tall, "
            "bucket interior 7.6 cm square, floor 1 cm thick); its starting spot "
            "along the flat varies per episode. On the open floor beside the "
            "channel lie two 4 cm cubes, one RED and one BLUE; they occupy two of "
            "four scatter slots with extra position jitter and free yaw every "
            "episode — find them by looking. Both cubes are easily graspable; the "
            "cart is a vehicle: push it along the channel (the walls guide it), "
            "never lift it.\n"
            "Goal: put the RED cube into the cart's bucket while the cart is still "
            "in the open (drop it in from above), then push the loaded cart along "
            "the channel, up the ramp — the ramp is too steep and slick for the "
            "cart to rest on, so keep pushing — over the crest, letting it drop "
            "into the summit pocket, and seat it against the back wall under the "
            "canopy. Leave everything at rest: the red cube inside the bucket, the "
            "cart parked level in the pocket. The BLUE cube is a decoy and must NOT "
            "be in the bucket. The order is forced by the roof: a parked cart's rim "
            "sits only about 3 cm below the canopy, and a 4 cm cube cannot pass "
            "that slit in any orientation — the cart can only be loaded before the "
            "climb. An empty parked cart, the red cube dumped on the pocket floor "
            "or on the canopy, the blue cube in the bucket, or a still-moving cart "
            "all count for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Drop the red cube into the yellow cart's bucket, then push the cart "
            "along the walled channel, up the ramp and over the crest, and seat it "
            "against the back wall of the roofed summit pocket. Keep the blue cube "
            "out of the cart. Load first — the roof gap over a parked cart is too "
            "small to admit the cube."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _rel(self, body) -> torch.Tensor:
        return body.data.root_pos_w - self.env_origins

    def _cube_in_cart(self, cube) -> torch.Tensor:
        """(N,) bool, geometric: cube center inside the cart's bucket (CART BODY
        frame — the carrier moves, so containment must ride with it)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        rel = quat_apply_inverse(self.cart.data.root_quat_w,
                                 cube.data.root_pos_w - self.cart.data.root_pos_w)
        return (rel[:, 0].abs() < c.in_xy) & (rel[:, 1].abs() < c.in_xy) \
            & (rel[:, 2] > c.in_z[0]) & (rel[:, 2] < c.in_z[1])

    def cargo_in_cart(self) -> torch.Tensor:
        return self._cube_in_cart(self.cargo)

    def decoy_in_cart(self) -> torch.Tensor:
        return self._cube_in_cart(self.decoy)

    def cart_upright(self) -> torch.Tensor:
        """(N,) bool: cart z-axis within `upright_deg` of vertical (rejects the
        crest-bridging perch, which tilts ~5 deg but fails x, and any toppled cart)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        up = quat_apply(self.cart.data.root_quat_w, ez)
        return up[:, 2] >= math.cos(math.radians(self.cfg.upright_deg))

    def cart_parked(self) -> torch.Tensor:
        """(N,) bool, geometric: cart origin inside the pocket windows (env frame;
        the station is static), level. The x window accepts every flat-on-floor
        pocket rest (the pocket is only 6 mm longer than the cart) and rejects
        crest-bridging perches; the z window rejects a cart perched on a stray cube
        or still on the ramp — asserted in __post_init__."""
        c = self.cfg
        rel = self._rel(self.cart)
        return (rel[:, 0] >= c.park_x[0]) & (rel[:, 0] <= c.park_x[1]) \
            & (rel[:, 1].abs() <= c.park_y_abs) \
            & (rel[:, 2] >= c.park_z[0]) & (rel[:, 2] <= c.park_z[1]) \
            & self.cart_upright()

    def settled(self) -> torch.Tensor:
        """(N,) bool: cart AND cargo at rest."""
        c = self.cfg
        return (self.cart.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.cargo.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch each demonstrated stage every physics substep: cargo loaded, loaded
        cart past mid-ramp, loaded cart at the parking windows."""
        c = self.cfg
        loaded = self.cargo_in_cart()
        relx = self._rel(self.cart)[:, 0]
        relz = self._rel(self.cart)[:, 2]
        self.load_latch = torch.maximum(self.load_latch, loaded.float())
        self.climb_latch = torch.maximum(
            self.climb_latch, (loaded & (relx > c.climb_x)).float())
        docked = loaded & (relx >= c.park_x[0]) & (relx <= c.park_x[1]) \
            & (relz >= c.park_z[0]) & (relz <= c.park_z[1])
        self.dock_latch = torch.maximum(self.dock_latch, docked.float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the RED cube rides inside the bucket of the cart parked in the
        summit pocket, the BLUE decoy does not, everything settled. Identity is by
        name — the decoy in the bucket blocks success and an empty parked cart
        counts for nothing; the canopy makes load-before-park physically necessary."""
        return self.cargo_in_cart() & ~self.decoy_in_cart() \
            & self.cart_parked() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.20 cargo ever loaded + 0.20 loaded cart ever past
        mid-ramp + 0.20 loaded cart ever at the parking windows (cap 0.60); 1.0 iff
        success(). Latched — credit never evaporates; the null policy scores ~0
        (cart on the flat, cubes on the open floor)."""
        base = (0.20 * self.load_latch + 0.20 * self.climb_latch
                + 0.20 * self.dock_latch).clamp(0.0, 0.60)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="cargo_tram", robot="null", env_spacing=3.0))
