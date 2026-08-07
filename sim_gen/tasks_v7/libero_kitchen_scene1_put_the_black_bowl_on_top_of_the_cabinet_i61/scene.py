"""CartFerryScene — dock the rolling cart under the covered ledge, slide the black bowl
onto its counter top, then ferry it home
(libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i61).

Derived from libero_90 kitchen_scene1 "put the black bowl on top of the cabinet", where
the whole task is one pick-and-place of the bowl onto a static elevated surface. Here
the transport relation is INVERTED: the bowl can never be carried by the arm at all —
the elevated FIXTURE that moves is the vehicle, and the bowl rides it.

The black bowl starts on a high ledge inside a roofed alcove at the far end of a
curb-fenced lane. Three facts make the seed's plan (grasp the bowl, carry it, set it
down) physically impossible:

  1. the bowl's outer diameter (10.4 cm) exceeds the parallel jaw's 8 cm span, so the
     only grasp is a top-down rim pinch (one finger inside the cup, one outside);
  2. the roof leaves only ~1.4 cm above the bowl's rim, so no fingertip — let alone a
     hand — fits above it while it is in the alcove: it cannot be pinched or lifted,
     only slid along the ledge and out the open front;
  3. directly below the ledge's open front, a full-width SLOT is cut through the deck
     down to the floor. A bowl slid off the ledge with nothing below falls in and is
     gone (narrow, deep, under the roof overhang — unreachable; a permanent fail latch).

The intended strategy:

  PHASE 1  DOCK — push the rolling cart down the lane until its base seats against the
           dock stop. Its cantilevered counter top then bridges the slot: the top's far
           edge stops ~2.5 cm short of the ledge, 2 mm below it.
  PHASE 2  TRANSFER — slide the bowl along the ledge, out the alcove's open front,
           across the small step down onto the cart's counter top (the bowl's 10.4 cm
           base easily bridges the gap — but only when the cart is docked).
  PHASE 3  FERRY — pull the cart by its handle back to the HOME end of the lane with
           the bowl riding upright on the counter top. Gentle accelerations only: the
           bowl is held by friction, nothing restrains it.

Success is judged on the PHYSICAL terminal state: cart in the home band of the lane,
bowl upright and at rest ON the cart's counter top (judged in the CART body frame),
everything settled, the transfer latched while the cart was forward (near dock — the
only place a slide across can physically work), and the bowl never lost to the slot.

Mechanism notes:
  - The whole static structure (deck, slot, curbs, stops, plinth, ledge, rails, back
    wall, roof) is ONE kinematic compound body, re-posed per reset (xy jitter + yaw);
    every predicate is evaluated in the fixture's body frame, so randomization is real.
    No joints anywhere — nothing anchors to a moved kinematic body.
  - The cart is a free dynamic compound body guided by the lane curbs; explicit physics
    materials are bound everywhere (custom-spawner colliders otherwise default to
    mu~0.5), so push/pull forces and the ride friction budget are calibrated.
  - The bowl is an octagonal open cup (pen-holder compound pattern), BLACK like the
    seed's akita bowl.

Everything is procedural. Heavy imports (isaaclab, pxr) are deferred so importing this
module stays app-free.
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
_SPAWNER_CACHE: dict[str, Any] = {}


def _bind_phys_material(stage, prim_path: str, root, mu_s: float, mu_d: float) -> None:
    """Author a physics material under the body and bind it to the whole subtree.
    Custom-spawner colliders otherwise get an uncontrolled default (~0.5)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, f"{prim_path}/physmat")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu_s))
    api.CreateDynamicFrictionAttr(float(mu_d))
    api.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(root).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_fixture(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One KINEMATIC rigid body: the whole static structure. Body frame: origin on the
    floor at the slot/lane axis; the lane runs along -x (home end), the plinth along +x.
    Deck top z=0.12, ledge top z=0.382, roof underside z=0.458."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(50.0)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)

    def _box(path, size, center, color):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        bxf = UsdGeom.Xformable(cube.GetPrim())
        bxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        bxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    deck, curb, plinth, ledge, roof, trench = (cfg.deck_color, cfg.curb_color,
                                               cfg.plinth_color, cfg.ledge_color,
                                               cfg.roof_color, cfg.trench_color)
    # deck with the slot cut through it (slot x in (-0.085, 0.07), full lane width)
    _box(f"{prim_path}/deck_lane", (0.635, 0.44, 0.12), (-0.4025, 0.0, 0.06), deck)
    _box(f"{prim_path}/deck_far", (0.31, 0.44, 0.12), (0.225, 0.0, 0.06), deck)
    for sgn in (-1.0, 1.0):
        _box(f"{prim_path}/slot_end_{'p' if sgn > 0 else 'n'}",
             (0.155, 0.02, 0.12), (-0.0075, sgn * 0.21, 0.06), trench)
        # lane curbs (guide the cart), ledge side rails (bowl exits only frontward)
        _box(f"{prim_path}/curb_{'p' if sgn > 0 else 'n'}",
             (0.635, 0.03, 0.05), (-0.4025, sgn * 0.151, 0.145), curb)
        _box(f"{prim_path}/rail_{'p' if sgn > 0 else 'n'}",
             (0.24, 0.02, 0.03), (0.175, sgn * 0.10, 0.397), curb)
    # travel stops for the cart base
    _box(f"{prim_path}/dock_stop", (0.03, 0.44, 0.06), (-0.10, 0.0, 0.15), curb)
    _box(f"{prim_path}/rear_stop", (0.03, 0.44, 0.06), (-0.65, 0.0, 0.15), curb)
    # plinth + ledge + side rails (above) + back wall + roof
    _box(f"{prim_path}/plinth", (0.28, 0.30, 0.25), (0.21, 0.0, 0.245), plinth)
    _box(f"{prim_path}/ledge", (0.24, 0.18, 0.012), (0.175, 0.0, 0.376), ledge)
    _box(f"{prim_path}/back_wall", (0.055, 0.18, 0.076), (0.3225, 0.0, 0.420), plinth)
    _box(f"{prim_path}/roof", (0.435, 0.26, 0.012), (0.1325, 0.0, 0.464), roof)
    _bind_phys_material(stage, prim_path, root, cfg.mu, cfg.mu - 0.03)
    return root


def _spawn_cart(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One dynamic rigid body: the rolling cart. Body frame: origin at the BASE CENTER
    (mid-height of the 0.248 m base box); the counter top plate (top at local z=0.136)
    cantilevers toward +x; the pull handle stands at the -x end."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    def _box(path, size, center, color):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        bxf = UsdGeom.Xformable(cube.GetPrim())
        bxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        bxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        collide(cube.GetPrim())

    _box(f"{prim_path}/base", (0.24, 0.26, 0.248), (0.0, 0.0, 0.0), cfg.body_color)
    _box(f"{prim_path}/plate", (0.385, 0.26, 0.012), (0.0725, 0.0, 0.130),
         cfg.plate_color)
    for sgn in (-1.0, 1.0):
        _box(f"{prim_path}/post_{'p' if sgn > 0 else 'n'}",
             (0.02, 0.02, 0.164), (-0.10, sgn * 0.09, 0.218), cfg.handle_color)
    bar = UsdGeom.Cylinder.Define(stage, f"{prim_path}/handlebar")
    bar.CreateAxisAttr("Y")
    bar.CreateRadiusAttr(0.012)
    bar.CreateHeightAttr(0.22)
    bar.CreateExtentAttr([Gf.Vec3f(-0.012, -0.11, -0.012), Gf.Vec3f(0.012, 0.11, 0.012)])
    UsdGeom.Xformable(bar.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(-0.10, 0.0, 0.31))
    bar.CreateDisplayColorAttr([Gf.Vec3f(*cfg.handle_color)])
    collide(bar.GetPrim())
    _bind_phys_material(stage, prim_path, root, cfg.mu, cfg.mu - 0.03)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: the BLACK bowl — an open octagonal cup (bottom disc + 8 wall
    segments). Body frame: axis = +z (up when upright), origin at mid-height."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.1)

    color = Gf.Vec3f(*cfg.color)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    outer_r = cfg.inner_r + cfg.wall_t
    bot = UsdGeom.Cylinder.Define(stage, f"{prim_path}/bottom")
    bot.CreateRadiusAttr(outer_r)
    bot.CreateHeightAttr(cfg.bot_t)
    bot.CreateExtentAttr([Gf.Vec3f(-outer_r, -outer_r, -cfg.bot_t / 2),
                          Gf.Vec3f(outer_r, outer_r, cfg.bot_t / 2)])
    UsdGeom.Xformable(bot.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, -cfg.height / 2 + cfg.bot_t / 2))
    bot.CreateDisplayColorAttr([color])
    collide(bot.GetPrim())

    n = 8
    r_mid = cfg.inner_r + cfg.wall_t / 2
    seg_len = 2 * (cfg.inner_r + cfg.wall_t) * math.tan(math.pi / n) + 0.002
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang), 0.0))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(cfg.wall_t, seg_len, cfg.height))
        seg.CreateDisplayColorAttr([color])
        collide(seg.GetPrim())
    _bind_phys_material(stage, prim_path, root, cfg.mu, cfg.mu - 0.05)
    return root


def _fixture_spawner_cfg(c: CartFerrySceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "fixture" not in _SPAWNER_CACHE:

        @configclass
        class FixtureSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_fixture)
            mu: float = 0.25
            deck_color: tuple = (0.45, 0.45, 0.48)
            curb_color: tuple = (0.75, 0.60, 0.15)
            plinth_color: tuple = (0.50, 0.35, 0.22)
            ledge_color: tuple = (0.62, 0.46, 0.30)
            roof_color: tuple = (0.30, 0.25, 0.20)
            trench_color: tuple = (0.18, 0.18, 0.20)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["fixture"] = FixtureSpawnerCfg

    return _SPAWNER_CACHE["fixture"](
        mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mu=c.fixture_mu, deck_color=c.deck_color, curb_color=c.curb_color,
        plinth_color=c.plinth_color, ledge_color=c.ledge_color,
        roof_color=c.roof_color, trench_color=c.trench_color,
        contact_offset=c.contact_offset,
    )


def _cart_spawner_cfg(c: CartFerrySceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cart" not in _SPAWNER_CACHE:

        @configclass
        class CartSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cart)
            mu: float = 0.30
            body_color: tuple = (0.20, 0.35, 0.55)
            plate_color: tuple = (0.72, 0.72, 0.74)
            handle_color: tuple = (0.15, 0.15, 0.17)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["cart"] = CartSpawnerCfg

    return _SPAWNER_CACHE["cart"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.cart_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mu=c.cart_mu, body_color=c.cart_body_color, plate_color=c.cart_plate_color,
        handle_color=c.cart_handle_color, contact_offset=c.contact_offset,
    )


def _bowl_spawner_cfg(c: CartFerrySceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bowl" not in _SPAWNER_CACHE:

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            inner_r: float = 0.042
            wall_t: float = 0.010
            height: float = 0.062
            bot_t: float = 0.010
            mu: float = 0.45
            color: tuple = (0.08, 0.08, 0.09)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["bowl"] = BowlSpawnerCfg

    return _SPAWNER_CACHE["bowl"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=c.bowl_inner_r, wall_t=c.bowl_wall_t, height=c.bowl_h,
        bot_t=c.bowl_bot_t, mu=c.bowl_mu, color=c.bowl_color, contact_offset=0.002,
    )


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class CartFerrySceneCfg(BaseCfg):
    """Config for `CartFerryScene`. All geometry below is FIXTURE-FRAME (origin on the
    floor at the lane axis; lane toward -x = HOME end, plinth toward +x). The fixture
    root itself is re-posed per reset (xy jitter + yaw), so nothing is world-anchored.

    Key fixture-frame landmarks: deck top z=0.12; slot through the deck x in
    (-0.085, 0.07), full lane width, down to the floor; dock stop face x=-0.115 (cart
    base center at dock = -0.235); ledge top z=0.382, front edge x=0.055; roof
    underside z=0.458; cart counter-top at z=0.380 (cart root z=0.244)."""

    # --- tunable: rubric thresholds ---------------------------------------------------------------
    settle_speed: float = tunable(0.05)   # max |lin vel| (cart, bowl) when judging (m/s)
    settle_w: float = tunable(0.15)       # max |ang vel| of the cart when judging (rad/s)
    bowl_up_max_deg: float = tunable(15.0)  # bowl axis within this of world-up
    # cart base center, fixture frame:
    dock_x_lo: float = tunable(-0.260)    # dock band (seated against the dock stop)
    dock_x_hi: float = tunable(-0.205)
    fwd_x: float = tunable(-0.280)        # "cart forward" bound for the transfer latch
    ride_x: float = tunable(-0.330)       # bowl-aboard past here latches "ride"
    home_x: float = tunable(-0.400)       # home band: base center below this
    lane_y_abs: float = tunable(0.040)    # cart on the lane axis
    # bowl root in the CART frame to count "on the counter top" (rest z = 0.167).
    # x band keeps the bowl base FULLY on the plate (edges at -0.12/+0.265, r=0.052):
    aboard_x_lo: float = tunable(-0.060)
    aboard_x_hi: float = tunable(0.210)
    aboard_y_abs: float = tunable(0.105)
    aboard_z_lo: float = tunable(0.150)
    aboard_z_hi: float = tunable(0.200)
    # bowl root in the FIXTURE frame to count "lost to the slot" (permanent):
    void_x_lo: float = tunable(-0.115)
    void_x_hi: float = tunable(0.070)
    void_z: float = tunable(0.115)        # below deck top while over the slot

    # --- tunable: randomization -------------------------------------------------------------------
    fix_jitter: float = tunable(0.04)     # fixture root xy jitter (+/- m)
    fix_yaw_deg: float = tunable(8.0)     # fixture root yaw (+/- deg)
    cart_x0: tuple = tunable((-0.49, -0.43))  # cart spawn band (fixture x)
    cart_y0: float = tunable(0.004)       # cart spawn y jitter (curb clearance is 6 mm)
    bowl_x0: tuple = tunable((0.17, 0.23))    # bowl spawn band on the ledge (fixture x)
    bowl_y0: float = tunable(0.015)       # bowl spawn y jitter

    # --- info: fixture ----------------------------------------------------------------------------
    fixture_mu: float = info(0.25)
    deck_top: float = info(0.12)
    ledge_top: float = info(0.382)
    ledge_front_x: float = info(0.055)
    roof_under: float = info(0.458)
    dock_stop_face_x: float = info(-0.115)
    deck_color: tuple = info((0.45, 0.45, 0.48))
    curb_color: tuple = info((0.75, 0.60, 0.15))
    plinth_color: tuple = info((0.50, 0.35, 0.22))
    ledge_color: tuple = info((0.62, 0.46, 0.30))
    roof_color: tuple = info((0.30, 0.25, 0.20))
    trench_color: tuple = info((0.18, 0.18, 0.20))
    contact_offset: float = info(0.003)

    # --- info: cart -------------------------------------------------------------------------------
    cart_mass: float = info(4.0)
    cart_mu: float = info(0.30)
    cart_root_z: float = info(0.244)      # base center height when on the deck
    cart_top_local_z: float = info(0.136)  # counter-top surface, cart frame
    cart_plate_front_local_x: float = info(0.265)
    cart_body_color: tuple = info((0.20, 0.35, 0.55))
    cart_plate_color: tuple = info((0.72, 0.72, 0.74))
    cart_handle_color: tuple = info((0.15, 0.15, 0.17))

    # --- info: bowl -------------------------------------------------------------------------------
    bowl_inner_r: float = info(0.042)
    bowl_wall_t: float = info(0.010)
    bowl_h: float = info(0.062)
    bowl_bot_t: float = info(0.010)
    bowl_mass: float = info(0.15)
    bowl_mu: float = info(0.45)
    bowl_color: tuple = info((0.08, 0.08, 0.09))


# ----- scene ----------------------------------------------------------------------------------------
@SCENES.register("cart_ferry")
class CartFerryScene(BaseScene):
    cfg: CartFerrySceneCfg

    def __init__(self, cfg: CartFerrySceneCfg | None = None) -> None:
        super().__init__(cfg or CartFerrySceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
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
            "fixture": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Fixture",
                spawn=_fixture_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "cart": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cart",
                spawn=_cart_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.46, 0.0, c.cart_root_z)),
            ),
            "bowl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl",
                spawn=_bowl_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.20, 0.0, c.ledge_top + c.bowl_h / 2 + 0.002)),
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
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.fixture: RigidObject = env.iscene["fixture"]
        self.cart: RigidObject = env.iscene["cart"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.env_origins = env.iscene.env_origins
        # latched progress / failure (post_step)
        self._docked_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._transferred_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._ride_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._home_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._voided_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        # reset grace re-pin buffers
        self._grace = torch.zeros(n, dtype=torch.long, device=dev)
        self._pin_states = {k: torch.zeros(n, 13, device=dev)
                            for k in ("fixture", "cart", "bowl")}

    # ----- frames ---------------------------------------------------------------------------------
    def _to_fix(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> fixture body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.fixture.data.root_quat_w,
                                  pos_w - self.fixture.data.root_pos_w)

    def _to_cart(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> cart body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.cart.data.root_quat_w,
                                  pos_w - self.cart.data.root_pos_w)

    def _axis_up(self, body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)

    # ----- predicates -----------------------------------------------------------------------------
    def cart_fix_xy(self) -> torch.Tensor:
        """(N,2) cart base center in the fixture frame (x = lane coordinate)."""
        return self._to_fix(self.cart.data.root_pos_w)[:, :2]

    def cart_upright(self) -> torch.Tensor:
        return self._axis_up(self.cart) > 0.95

    def cart_docked(self) -> torch.Tensor:
        """(N,) bool: cart base seated in the dock band, on the lane axis, upright."""
        c = self.cfg
        xy = self.cart_fix_xy()
        return ((xy[:, 0] > c.dock_x_lo) & (xy[:, 0] < c.dock_x_hi)
                & (xy[:, 1].abs() < c.lane_y_abs) & self.cart_upright())

    def cart_forward(self) -> torch.Tensor:
        """(N,) bool: cart far enough forward that a ledge->cart slide can bridge."""
        c = self.cfg
        xy = self.cart_fix_xy()
        return (xy[:, 0] > c.fwd_x) & (xy[:, 1].abs() < c.lane_y_abs) & self.cart_upright()

    def cart_home(self) -> torch.Tensor:
        """(N,) bool: cart back in the HOME band of the lane, upright."""
        c = self.cfg
        xy = self.cart_fix_xy()
        return ((xy[:, 0] < c.home_x) & (xy[:, 1].abs() < c.lane_y_abs)
                & self.cart_upright())

    def bowl_upright(self) -> torch.Tensor:
        return self._axis_up(self.bowl) >= math.cos(math.radians(self.cfg.bowl_up_max_deg))

    def bowl_on_cart(self) -> torch.Tensor:
        """(N,) bool: bowl upright, resting ON the cart's counter top (CART frame —
        rides with the cart wherever it is)."""
        c = self.cfg
        loc = self._to_cart(self.bowl.data.root_pos_w)
        return ((loc[:, 0] > c.aboard_x_lo) & (loc[:, 0] < c.aboard_x_hi)
                & (loc[:, 1].abs() < c.aboard_y_abs)
                & (loc[:, 2] > c.aboard_z_lo) & (loc[:, 2] < c.aboard_z_hi)
                & self.bowl_upright() & self.cart_upright())

    def bowl_in_slot(self) -> torch.Tensor:
        """(N,) bool: bowl fell through the slot (below deck top over the slot span)."""
        c = self.cfg
        loc = self._to_fix(self.bowl.data.root_pos_w)
        return ((loc[:, 0] > c.void_x_lo) & (loc[:, 0] < c.void_x_hi)
                & (loc[:, 1].abs() < 0.23) & (loc[:, 2] < c.void_z))

    def settled(self) -> torch.Tensor:
        c = self.cfg
        sb = self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        sk = self.cart.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        wk = self.cart.data.root_ang_vel_w.norm(dim=-1) < c.settle_w
        return sb & sk & wk

    # ----- mechanism (every substep) --------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        # reset grace: re-pin freshly reset bodies while write timing settles
        gids = (self._grace > 0).nonzero(as_tuple=False).squeeze(-1)
        if len(gids):
            for name, body in (("fixture", self.fixture), ("cart", self.cart),
                               ("bowl", self.bowl)):
                body.write_root_state_to_sim(self._pin_states[name][gids], gids)
            self._grace[gids] -= 1
            return

        aboard = self.bowl_on_cart()
        self._docked_ever |= self.cart_docked()
        self._transferred_ever |= aboard & self.cart_forward()
        self._ride_ever |= (aboard & self._transferred_ever
                            & (self.cart_fix_xy()[:, 0] < self.cfg.ride_x))
        self._home_ever |= aboard & self._transferred_ever & self.cart_home()
        self._voided_ever |= self.bowl_in_slot()

    # ----- reset ----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Sample the fixture root pose (xy jitter + yaw), then place cart and bowl at
        their fixture-frame spawn slots. Everything written with zero velocity; a
        2-substep grace re-pin absorbs write-timing races."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # fixture pose
        yaw = torch.deg2rad((torch.rand(m, device=dev) * 2 - 1) * c.fix_yaw_deg)
        half = yaw / 2
        fq = torch.zeros(m, 4, device=dev)
        fq[:, 0] = torch.cos(half)
        fq[:, 3] = torch.sin(half)
        fp = torch.zeros(m, 3, device=dev)
        fp[:, :2] = (torch.rand(m, 2, device=dev) * 2 - 1) * c.fix_jitter
        fp += origin
        fst = torch.zeros(m, 13, device=dev)
        fst[:, 0:3] = fp
        fst[:, 3:7] = fq
        self.fixture.write_root_state_to_sim(fst, env_ids)
        self._pin_states["fixture"][env_ids] = fst

        cy, sy = torch.cos(yaw), torch.sin(yaw)

        def to_world(loc: torch.Tensor) -> torch.Tensor:
            w = torch.zeros(m, 3, device=dev)
            w[:, 0] = cy * loc[:, 0] - sy * loc[:, 1]
            w[:, 1] = sy * loc[:, 0] + cy * loc[:, 1]
            w[:, 2] = loc[:, 2]
            return w + fp

        # cart: home end of the lane, aligned with the fixture
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.cart_x0[0] + (c.cart_x0[1] - c.cart_x0[0]) * torch.rand(m, device=dev)
        loc[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.cart_y0
        loc[:, 2] = c.cart_root_z + 0.002
        kst = torch.zeros(m, 13, device=dev)
        kst[:, 0:3] = to_world(loc)
        kst[:, 3:7] = fq
        self.cart.write_root_state_to_sim(kst, env_ids)
        self._pin_states["cart"][env_ids] = kst

        # bowl: on the ledge, free yaw
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.bowl_x0[0] + (c.bowl_x0[1] - c.bowl_x0[0]) * torch.rand(m, device=dev)
        loc[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.bowl_y0
        loc[:, 2] = c.ledge_top + c.bowl_h / 2 + 0.002
        bst = torch.zeros(m, 13, device=dev)
        bst[:, 0:3] = to_world(loc)
        bhalf = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        bst[:, 3] = torch.cos(bhalf)
        bst[:, 6] = torch.sin(bhalf)
        self.bowl.write_root_state_to_sim(bst, env_ids)
        self._pin_states["bowl"][env_ids] = bst

        for lat in (self._docked_ever, self._transferred_ever, self._ride_ever,
                    self._home_ever, self._voided_ever):
            lat[env_ids] = False
        self._grace[env_ids] = 2

    # ----- state (full, restorable) ---------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"fixture": self.fixture, "cart": self.cart, "bowl": self.bowl}
        return {
            "bodies": {k: b.data.root_state_w[env_ids].clone() for k, b in bodies.items()},
            "latches": torch.stack(
                [self._docked_ever[env_ids], self._transferred_ever[env_ids],
                 self._ride_ever[env_ids], self._home_ever[env_ids],
                 self._voided_ever[env_ids]], dim=1).clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"fixture": self.fixture, "cart": self.cart, "bowl": self.bowl}
        for k, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][k], env_ids)
        lat = state["latches"]
        self._docked_ever[env_ids] = lat[:, 0]
        self._transferred_ever[env_ids] = lat[:, 1]
        self._ride_ever[env_ids] = lat[:, 2]
        self._home_ever[env_ids] = lat[:, 3]
        self._voided_ever[env_ids] = lat[:, 4]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "A raised gray deck (12 cm high) carries a straight LANE fenced by two "
            "yellow curbs. A blue rolling CART sits at the near (HOME) end of the lane: "
            "24 x 26 cm base, a light-gray COUNTER TOP at 38 cm height that cantilevers "
            "39 cm toward the far end, and a black pull HANDLE bar (2.4 cm thick, 55 cm "
            "high) at its near end. At the far end of the lane stands a wooden tower "
            "whose LEDGE (38 cm high) holds a BLACK BOWL — an open cup, 10.4 cm across, "
            "6 cm tall. The ledge sits inside an alcove: side rails, a back wall, and a "
            "dark ROOF only ~1.4 cm above the bowl's rim. The bowl is TOO WIDE for the "
            "8 cm gripper to squeeze from outside, and the roof leaves no room for a "
            "finger above the rim, so the bowl cannot be grasped or lifted there — it "
            "can only be SLID along the ledge and out the alcove's open front. Directly "
            "below that open front a full-width SLOT is cut through the deck down to "
            "the floor: a bowl slid off with nothing beneath falls in and is "
            "irrecoverable (the slot is deep, narrow, and under the roof). The cart is "
            "the only thing that can bridge it: pushed all the way down the lane it "
            "seats against a dock stop, and its counter top then reaches to ~2.5 cm "
            "from the ledge edge, 2 mm below it — close enough for the bowl's wide "
            "base to slide straight across. Goal: the bowl standing upright, at rest, "
            "ON the cart's counter top, with the cart back in the HOME third of the "
            "lane (base center past 40 cm from the slot on the near side) and "
            "everything settled. The transfer only counts done near the dock (the only "
            "place a slide across is physically possible), and the bowl must never "
            "fall into the slot. When ferrying the cart home, accelerate gently — "
            "only friction holds the bowl on the counter top."
        )

    def instruction(self) -> str:
        return (
            "Push the blue cart down the fenced lane until it seats against the dock "
            "stop under the wooden tower, slide the black bowl out of its covered "
            "alcove onto the cart's counter top, then pull the cart by its handle back "
            "to the home end of the lane with the bowl riding upright on top. Never "
            "let the bowl fall into the slot in front of the ledge, and pull gently — "
            "nothing but friction holds the bowl on the cart."
        )

    # ----- rubric ---------------------------------------------------------------------------------
    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched stages of the demonstrated solution:
        0.20 cart ever docked + 0.30 bowl ever transferred onto the cart (while the
        cart was forward at the dock) + 0.15 bowl ridden past mid-lane + 0.35 bowl
        aboard with the cart home; exactly 1.0 iff success(). Losing the bowl to the
        slot permanently caps the score at 0.20 (the dock credit is all that can
        remain). Null policy ~0; the seed's plan (grasp the bowl and carry it) cannot
        even start."""
        s = (0.20 * self._docked_ever.float()
             + 0.30 * self._transferred_ever.float()
             + 0.15 * self._ride_ever.float()
             + 0.35 * self._home_ever.float())
        s = torch.where(self._voided_ever, s.clamp(max=0.20), s)
        return torch.where(self.success(),
                           torch.ones(self.env.num_envs, device=self.env.device),
                           s.clamp(0.0, 0.95))

    def success(self) -> torch.Tensor:
        """(N,) bool: cart in the home band with the bowl upright at rest on its
        counter top, everything settled, the transfer having happened at the dock (the
        only physically possible place), and the bowl never lost to the slot."""
        return (self.cart_home() & self.bowl_on_cart() & self.settled()
                & self._transferred_ever & ~self._voided_ever)


register_env("simgen", lambda: EnvCfg(scene="cart_ferry", robot="null"))
