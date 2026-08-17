"""CellarRollStowScene — unbar the low wine locker, ROLL the bottle in lying down, re-bar the slot.

Derived from libero_90/kitchen_scene4 "put the wine bottle in the bottom drawer of the
cabinet" (pick the bottle off the table, lower it upright into a passively open sliding
drawer, judged by a static bbox test), but the receptacle and the required manipulation
are both replaced:

  - The receptacle is a WINE LOCKER: a twin-bay, floor-standing block whose bay
    interiors are only ~85-93 mm tall under a fixed roof. A standing bottle is 175 mm
    tall, so the seed's move — carry the bottle upright and lower it into the
    receptacle — is geometrically impossible: the bottle DOES NOT FIT upright, and the
    roof forbids any top insertion. The only way in is the front LETTERBOX SLOT
    (~84 mm tall), reached by a declined ramp: the bottle must be REORIENTED TO
    HORIZONTAL, laid on the ramp, and GRAVITY-ROLLED through the slot; the bay floor
    is pitched inward so the rolling bottle parks itself against the back wall.
  - The TARGET bay is identified by MECHANISM STATE, not by a label: it is the bay
    whose slot is BARRED by a loose red bar resting in U-notches on two posts across
    the ramp. The bar must be lifted OUT first (seated, it blocks the roll path: the
    gap under it is 20 mm and over it 40 mm, both smaller than the 60 mm bottle), and
    after the bottle is stowed the bar must be SEATED BACK in the same notches. The
    twin bay is an open DECOY — stowing the bottle there fails.
  - There is no articulation anywhere: the "mechanism" is free rigid bodies + gravity.
    The load-bearing physics is the rolling transit through the letterbox aperture and
    the drop-seating of the bar between its notch prongs.

A solver therefore needs a different plan from the seed (identify the barred bay ->
remove the bar -> reorient the bottle to horizontal and deliver it by ROLLING, not by
lowering -> restore the bar) and a different code structure (procedural kinematic
compound fixture + free bodies, a mechanism-state rubric with latched stages; no
asset loading, no articulated drawer, no static-bbox-only test).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - cellar: KINEMATIC compound — per bay: pitched interior floor, roof, cheeks, back
    wall, declined approach ramp with side rails, and two notch posts (each a post +
    two prongs forming an upward-open U) flanking the ramp near the sill.
  - wine: DYNAMIC compound — green body cylinder (60 mm dia x 130 mm) + neck cylinder
    (24 mm dia x 45 mm) along local +z, plus a 0.2 mm-proud exact-capsule rolling band
    (GPU cylinder colliders are faceted hulls that refuse to roll on shallow slopes).
    Origin at the body cylinder's centre.
  - bar: DYNAMIC single red box (20 x 320 x 20 mm), long axis local +y. Seated it
    rests on the post tops between the prongs, blocking the slot.

Per-episode randomization (readback-verifiable): which bay is barred (Bernoulli side
draw — the target side), bar y jitter within its notches, and the bottle's spawn
x/y/yaw on the open floor.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.15 * unbarred   — bar ever displaced > 6 cm from its seated pose (latched bool)
  0.20 * approach   — bottle approach to the TARGET slot mouth, gated on unbarred
                      (latched running max; ~0 for doing nothing)
  0.25 * stowed     — bottle ever inside the target bay, lying down (latched bool)
  0.25 * rebar      — bar proximity to its seat, counted only while the bottle is
                      currently stowed (re-barring an empty locker earns nothing;
                      the bar starting seated earns nothing)
  1.0 iff success() — bottle lying at rest inside the TARGET bay AND the bar seated
                      back in the TARGET notches (pose + alignment window), everything
                      settled. Non-success cap 0.85.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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
# One rigid body per object, several child colliders, authored with raw pxr APIs; only
# `isaaclab.sim.utils.clone` is borrowed (regex-resolve + per-env replication).

_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, quat, color, collide: Callable) -> None:
    """Author one box collider-child (translate -> orient -> scale xformOps, authored once)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    w, x, y, z = (float(v) for v in quat)
    xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


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


def _spawn_cellar(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the twin-bay wine locker: KINEMATIC compound of axis-aligned and pitched
    boxes, precomputed in the scene cfg as (name, center, size, quat, color) specs."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    for name, center, size, quat, color in cfg.boxes:
        _add_box(stage, f"{prim_path}/{name}", center=center, size=size, quat=quat,
                 color=color, collide=collide)
    return root


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the wine bottle: DYNAMIC body cylinder + neck cylinder along local +z,
    PLUS a 0.2 mm-proud EXACT CAPSULE rolling band buried in the body. Cylinder
    colliders become faceted convex hulls on the GPU pipeline and a faceted prism
    parks on a facet instead of rolling down the 7.4 deg ramp; the capsule primitive
    is exact, so the lying bottle rolls on the capsule while the cylinder's flat ends
    (which the shorter capsule never reaches) keep the standing spawn stable. Origin
    at the body centre (MassAPI mass -> CoM at the origin = straight rolling)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    # A lying bottle is a roller: enough angular damping to kill the back-wall rebound
    # oscillation in a few seconds, small enough that the 7.4 deg ramp still drives it;
    # velocity iters 4 kill the GPU capsule-on-box phantom-creep artifact.
    pxrb.CreateLinearDampingAttr(0.06)
    pxrb.CreateAngularDampingAttr(0.30)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    # The bottle must START rolling from rest on a shallow slope: PhysX sleep freezes
    # a momentarily-still body mid-ramp (gravity does not wake sleepers) and the
    # stabilization pass glues slow bodies. Both off for this body.
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    color = Gf.Vec3f(*cfg.color)
    for nm, r, h, z0 in (("body", cfg.body_r, cfg.body_h, 0.0),
                         ("neck", cfg.neck_r, cfg.neck_h, cfg.body_h / 2 + cfg.neck_h / 2)):
        cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{nm}")
        cyl.CreateRadiusAttr(r)
        cyl.CreateHeightAttr(h)
        cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
        UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z0))
        cyl.CreateDisplayColorAttr([color])
        collide(cyl.GetPrim())
    # Exact-capsule rolling band: radius 0.2 mm proud of the cylinder hull (so lying
    # contact runs through the smooth capsule), tips >= 4 mm short of the flat ends
    # (so standing contact stays on the cylinder end rim, statically stable).
    rr = cfg.body_r + 0.0002
    ch = cfg.body_h - 2.0 * rr - 0.010
    cap = UsdGeom.Capsule.Define(stage, f"{prim_path}/roller")
    cap.CreateRadiusAttr(rr)
    cap.CreateHeightAttr(ch)
    cap.CreateExtentAttr([Gf.Vec3f(-rr, -rr, -(ch / 2 + rr)), Gf.Vec3f(rr, rr, ch / 2 + rr)])
    cap.CreateDisplayColorAttr([color])
    collide(cap.GetPrim())
    # SLICK physics material on the band, combine-mode MIN (beats the ~0.5 default of
    # everything else): the GPU contact solver caps a driven roll-from-rest at a few
    # mm/s (constant-velocity artifact), so the lying bottle GLIDES down the ramp on
    # a near-frictionless band instead of fighting the rolling artifact. Restitution
    # 0 (min-combined too) keeps the back-wall arrival dead.
    from pxr import UsdShade

    mat = UsdShade.Material.Define(stage, f"{prim_path}/slick_mat")
    mapi = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mapi.CreateStaticFrictionAttr(0.03)
    mapi.CreateDynamicFrictionAttr(0.03)
    mapi.CreateRestitutionAttr(0.0)
    pmat = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    pmat.CreateFrictionCombineModeAttr("min")
    pmat.CreateRestitutionCombineModeAttr("min")
    UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    return root


def _spawn_bar(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the red slot bar: DYNAMIC single box, long axis local +y."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.10)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    collide = _make_collide(cfg)
    _add_box(stage, f"{prim_path}/bar", center=(0.0, 0.0, 0.0), size=cfg.size,
             quat=(1.0, 0.0, 0.0, 0.0), color=cfg.color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the three compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cellar" not in _SPAWNER_CACHE:

        @configclass
        class CellarSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cellar)
            boxes: tuple = ()
            contact_offset: float = 0.002

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            body_r: float = 0.030
            body_h: float = 0.130
            neck_r: float = 0.012
            neck_h: float = 0.045
            color: tuple = (0.10, 0.40, 0.15)
            contact_offset: float = 0.002

        @configclass
        class BarSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bar)
            size: tuple = (0.020, 0.320, 0.020)
            color: tuple = (0.75, 0.10, 0.08)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(cellar=CellarSpawnerCfg, bottle=BottleSpawnerCfg,
                              bar=BarSpawnerCfg)
    return _SPAWNER_CACHE


def _pitched(x0: float, z0: float, x1: float, z1: float, y: float, width: float,
             thick: float) -> tuple[tuple, tuple, tuple]:
    """(center, size, quat) for a slab whose TOP surface runs (x0,z0)->(x1,z1) at
    lateral centre y. Positive pitch = downhill toward +x (rotation about +y)."""
    th = math.atan2(z0 - z1, x1 - x0)
    length = math.hypot(x1 - x0, z1 - z0)
    mx, mz = (x0 + x1) / 2, (z0 + z1) / 2
    cx = mx - math.sin(th) * thick / 2
    cz = mz - math.cos(th) * thick / 2
    return ((cx, y, cz), (length, width, thick),
            (math.cos(th / 2), 0.0, math.sin(th / 2), 0.0))


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CellarRollStowSceneCfg(BaseCfg):
    """Config for `CellarRollStowScene`. The interlocks are metric: bay interior height
    (93 mm max) < standing bottle (175 mm) so the bottle only fits LYING; the roof
    forbids top insertion; the seated bar leaves 20 mm under and 40 mm over itself,
    both < the 60 mm bottle body, so the roll REQUIRES the bar out; and the bay floor
    is pitched inward so a bottle that makes it through parks at the back wall."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    settle_lin: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.60)  # max |ang vel| when judging (rad/s)
    lying_max_axis_z: float = tunable(0.35)  # |bottle axis . z| below this = lying down
    seat_dx: float = tunable(0.012)  # bar seated: |x - notch x| window (m)
    seat_dy: float = tunable(0.050)  # bar seated: |y - bay centre| window (m)
    seat_z_lo: float = tunable(0.046)  # bar seated: centre z window (nominal 0.055)
    seat_z_hi: float = tunable(0.064)
    seat_align: float = tunable(0.95)  # bar seated: |long axis . world y| minimum
    unbar_dist: float = tunable(0.06)  # "unbarred" latch: bar displaced beyond this
    approach_d0: float = tunable(0.50)  # approach ramp: p = 1 - d/approach_d0
    rebar_d0: float = tunable(0.30)  # rebar proximity ramp while stowed

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    side_sample: bool = tunable(True)  # Bernoulli barred-side draw (demo sets False -> +y)
    spawn_x: tuple = tunable((0.18, 0.26))  # bottle spawn x band (open floor)
    spawn_y: tuple = tunable((-0.20, 0.20))  # bottle spawn y band
    bar_y_jitter: float = tunable(0.008)  # bar seated y jitter within its notches

    # --- info: layout (single Franka base at the origin; interactions at r 0.24-0.55 m) ---------
    bay_cy: float = info(0.17)  # bay centres at y = +/- this
    bay_w_in: float = info(0.24)  # bay interior width (between cheek inner faces)
    sill_x: float = info(0.50)  # slot sill plane
    back_in_x: float = info(0.66)  # back wall interior face
    roof_lo_z: float = info(0.105)  # roof underside = slot top edge
    ramp_x0: float = info(0.298)  # ramp top surface: (x0, z0) -> (x1, z1)
    ramp_z0: float = info(0.048)
    ramp_x1: float = info(0.502)
    ramp_z1: float = info(0.0215)
    floor_x0: float = info(0.498)  # bay floor top surface, pitched inward (downhill +x)
    floor_z0: float = info(0.0205)
    floor_x1: float = info(0.664)
    floor_z1: float = info(0.0118)
    post_x: float = info(0.472)  # notch posts: bar seat x
    post_dy: float = info(0.135)  # post centres at bay_cy +/- this
    post_h: float = info(0.045)  # post top = bar underside when seated
    prong_dx: float = info(0.023)  # prong centres at post_x +/- this (gap 34 mm)
    prong_t: float = info(0.012)
    prong_h: float = info(0.035)
    bar_size: tuple = info((0.020, 0.320, 0.020))
    bar_seat_z: float = info(0.055)  # seated bar centre z (post_h + bar/2)
    bar_mass: float = info(0.15)
    mouth_z: float = info(0.055)  # slot mouth reference point height
    # bottle
    body_r: float = info(0.030)
    body_h: float = info(0.130)
    neck_r: float = info(0.012)
    neck_h: float = info(0.045)
    wine_mass: float = info(0.30)
    # colors
    col_body: tuple = info((0.46, 0.44, 0.48))  # locker gray
    col_ramp: tuple = info((0.58, 0.52, 0.44))  # ramp tan
    col_post: tuple = info((0.32, 0.28, 0.26))  # dark posts
    col_wine: tuple = info((0.10, 0.40, 0.15))  # green bottle
    col_bar: tuple = info((0.78, 0.10, 0.08))  # red bar
    contact_offset: float = info(0.002)
    # containment box (world, x/z fixed; y centred on the bay): centre must be well
    # past the sill (no doorway rests count) and under the roof
    bay_box_x: tuple = info((0.545, 0.656))
    bay_box_dy: float = info(0.105)
    bay_box_z: tuple = info((0.0, 0.088))
    # rubric weights (0.15 + 0.20 + 0.25 + 0.25 = 0.85 = the non-success cap)
    w_unbar: float = info(0.15)
    w_app: float = info(0.20)
    w_stow: float = info(0.25)
    w_rebar: float = info(0.25)

    # Derived (filled in __post_init__).
    boxes: tuple = field(default=None, init=False)  # cellar (name, center, size, quat, color)

    def __post_init__(self) -> None:
        qi = (1.0, 0.0, 0.0, 0.0)
        th_r = math.atan2(self.ramp_z0 - self.ramp_z1, self.ramp_x1 - self.ramp_x0)

        def zr(x: float) -> float:
            return self.ramp_z0 - (x - self.ramp_x0) * math.tan(th_r)

        boxes: list[tuple] = []
        for cy, tag in ((self.bay_cy, "l"), (-self.bay_cy, "r")):
            half_w = self.bay_w_in / 2
            # pitched interior floor
            c, s, q = _pitched(self.floor_x0, self.floor_z0, self.floor_x1, self.floor_z1,
                               cy, self.bay_w_in, 0.020)
            boxes.append((f"floor_{tag}", c, s, q, self.col_ramp))
            # roof (x 0.49..0.68), cheeks, back wall
            boxes.append((f"roof_{tag}", (0.585, cy, 0.115), (0.190, 0.264, 0.020), qi,
                          self.col_body))
            for sgn, sub in ((1.0, "a"), (-1.0, "b")):
                boxes.append((f"cheek_{tag}{sub}",
                              (0.585, cy + sgn * (half_w + 0.006), 0.0525),
                              (0.190, 0.012, 0.105), qi, self.col_body))
            boxes.append((f"back_{tag}", (0.67, cy, 0.0625), (0.020, 0.264, 0.125), qi,
                          self.col_body))
            # approach ramp + upper side rails (rails end at x=0.44, clear of the bar)
            c, s, q = _pitched(self.ramp_x0, self.ramp_z0, self.ramp_x1, self.ramp_z1,
                               cy, self.bay_w_in, 0.020)
            boxes.append((f"ramp_{tag}", c, s, q, self.col_ramp))
            for sgn, sub in ((1.0, "a"), (-1.0, "b")):
                c, s, q = _pitched(0.300, zr(0.300) + 0.035, 0.440, zr(0.440) + 0.035,
                                   cy + sgn * 0.112, 0.008, 0.035)
                boxes.append((f"rail_{tag}{sub}", c, s, q, self.col_ramp))
            # notch posts (post + two prongs = upward-open U) flanking the ramp
            for sgn, sub in ((1.0, "a"), (-1.0, "b")):
                py = cy + sgn * self.post_dy
                boxes.append((f"post_{tag}{sub}", (self.post_x, py, self.post_h / 2),
                              (0.030, 0.030, self.post_h), qi, self.col_post))
                for sx, sub2 in ((1.0, "f"), (-1.0, "n")):
                    boxes.append((f"prong_{tag}{sub}{sub2}",
                                  (self.post_x + sx * self.prong_dx, py,
                                   self.post_h + self.prong_h / 2),
                                  (self.prong_t, 0.030, self.prong_h), qi, self.col_post))
        self.boxes = tuple(boxes)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("cellar_roll_stow")
class CellarRollStowScene(BaseScene):
    cfg: CellarRollStowSceneCfg

    def __init__(self, cfg: CellarRollStowSceneCfg | None = None) -> None:
        super().__init__(cfg or CellarRollStowSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "cellar": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cellar",
                spawn=spawners["cellar"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    boxes=c.boxes, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "wine": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Wine",
                spawn=spawners["bottle"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.wine_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    body_r=c.body_r, body_h=c.body_h, neck_r=c.neck_r, neck_h=c.neck_h,
                    color=c.col_wine, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.22, 0.0, c.body_h / 2 + 0.002)),
            ),
            "bar": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bar",
                spawn=spawners["bar"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bar_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    size=c.bar_size, color=c.col_bar, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.post_x, c.bay_cy, c.bar_seat_z + 0.001)),
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
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.cellar: RigidObject = env.iscene["cellar"]
        self.wine: RigidObject = env.iscene["wine"]
        self.bar: RigidObject = env.iscene["bar"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.target_sign = torch.ones(n, device=dev)  # +1 -> +y bay barred, -1 -> -y
        # latches: partial progress survives transient achievements (rubric requirement)
        self._unbarred = torch.zeros(n, dtype=torch.bool, device=dev)
        self._app_max = torch.zeros(n, device=dev)
        self._stowed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._rebar_max = torch.zeros(n, device=dev)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: draw the barred (= target) side, seat the bar in that side's
        notches (start state of the mechanism; small y jitter), stand the bottle at a
        random x/y/yaw on the open floor, re-assert the kinematic cellar, clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- cellar (kinematic, fixed) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.cellar.write_root_state_to_sim(st, env_ids)

        # --- target side draw (torch.rand comparison, not randint) ---
        if c.side_sample:
            sign = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        else:
            sign = torch.ones(m, device=dev)
        self.target_sign[env_ids] = sign

        # --- bar: seated in the target side's notches (+0.5 mm settle drop) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.post_x
        st[:, 1] = sign * c.bay_cy + (torch.rand(m, device=dev) * 2 - 1) * c.bar_y_jitter
        st[:, 2] = c.bar_seat_z + 0.0005
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.bar.write_root_state_to_sim(st, env_ids)

        # --- bottle: standing upright on the open floor, x/y/yaw random ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.spawn_x[0] + torch.rand(m, device=dev) * (c.spawn_x[1] - c.spawn_x[0])
        st[:, 1] = c.spawn_y[0] + torch.rand(m, device=dev) * (c.spawn_y[1] - c.spawn_y[0])
        st[:, 2] = c.body_h / 2 + 0.002
        half = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.wine.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._unbarred[env_ids] = False
        self._app_max[env_ids] = 0.0
        self._stowed[env_ids] = False
        self._rebar_max[env_ids] = 0.0

    # ----- state (full, restorable) ---------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cellar": self.cellar.data.root_state_w[env_ids].clone(),
            "wine": self.wine.data.root_state_w[env_ids].clone(),
            "bar": self.bar.data.root_state_w[env_ids].clone(),
            "target_sign": self.target_sign[env_ids].clone(),
            "unbarred": self._unbarred[env_ids].clone(),
            "app_max": self._app_max[env_ids].clone(),
            "stowed": self._stowed[env_ids].clone(),
            "rebar_max": self._rebar_max[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cellar.write_root_state_to_sim(state["cellar"], env_ids)
        self.wine.write_root_state_to_sim(state["wine"], env_ids)
        self.bar.write_root_state_to_sim(state["bar"], env_ids)
        self.target_sign[env_ids] = state["target_sign"]
        self._unbarred[env_ids] = state["unbarred"]
        self._app_max[env_ids] = state["app_max"]
        self._stowed[env_ids] = state["stowed"]
        self._rebar_max[env_ids] = state["rebar_max"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        slot_h = (c.roof_lo_z - c.ramp_z1) * 1000
        return (
            f"A gray floor-standing WINE LOCKER with two identical low bays sits ahead, "
            f"side by side (bay centres {c.bay_cy * 100:.0f} cm left and right). Each bay "
            f"is a roofed pocket only ~9 cm tall inside, open to the front through a "
            f"LETTERBOX SLOT (~{slot_h:.0f} mm tall, {c.bay_w_in * 100:.0f} cm wide) at "
            f"the top of a shallow tan RAMP that descends into it; inside, the floor "
            f"tilts gently inward toward the back wall. A GREEN wine bottle "
            f"({2 * c.body_r * 100:.0f} cm body, {c.body_h * 100:.0f} cm long plus a "
            f"{2 * c.neck_r * 10:.1f} cm-thick neck, ~{(c.body_h + c.neck_h) * 100:.1f} cm "
            f"overall) stands upright on the open floor in front. ONE bay — it varies "
            f"between episodes — has a loose RED BAR lying across its ramp, seated in "
            f"U-notches on two dark posts: that BARRED bay is the TARGET. The other, "
            f"open bay is a decoy.\n"
            f"Goal: the green bottle must end up AT REST INSIDE the target bay, and the "
            f"red bar must be SEATED BACK in the same notches it started in. The bottle "
            f"is much taller than the bay, so it can never go in (or fit) upright and the "
            f"roof blocks insertion from above: lay it on its SIDE on the target ramp, "
            f"axis across the ramp, and let it ROLL down through the slot — the tilted "
            f"bay floor will carry it to the back wall. The seated bar blocks the roll "
            f"path (the gaps under and over it are smaller than the bottle), so you must "
            f"first LIFT THE BAR OUT of its notches and set it aside, roll the bottle "
            f"in, then drop the bar back so it rests level in both notches. Stowing the "
            f"bottle in the unbarred decoy bay, leaving it on a ramp, on the roof, or "
            f"anywhere outside the target bay is failure, as is leaving the bar "
            f"unseated, tilted, or in the decoy bay's notches."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lift the red bar out of the notches on the barred bay of the wine locker "
            "and set it aside. Lay the green bottle on its side on that bay's ramp so "
            "it rolls in through the low slot and rests inside, then seat the red bar "
            "back in the same notches. Do not put the bottle in the other, unbarred bay."
        )

    # ----- readings / rubric ----------------------------------------------------------------------
    def _rel(self, body: RigidObject) -> torch.Tensor:
        return body.data.root_pos_w - self.env_origins

    def seat_point(self, sign: torch.Tensor) -> torch.Tensor:
        """(N,3) nominal seated bar centre for bay side `sign` (+1/-1), env-local."""
        c = self.cfg
        n = sign.shape[0]
        p = torch.zeros(n, 3, device=sign.device)
        p[:, 0] = c.post_x
        p[:, 1] = sign * c.bay_cy
        p[:, 2] = c.bar_seat_z
        return p

    def bar_seated(self, sign: torch.Tensor | None = None) -> torch.Tensor:
        """(N,) bool: bar centre inside the seat window of bay side `sign` (default the
        TARGET side) AND its long axis level along world y."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        if sign is None:
            sign = self.target_sign
        p = self._rel(self.bar)
        seat = self.seat_point(sign)
        in_win = ((p[:, 0] - seat[:, 0]).abs() <= c.seat_dx) \
            & ((p[:, 1] - seat[:, 1]).abs() <= c.seat_dy) \
            & (p[:, 2] >= c.seat_z_lo) & (p[:, 2] <= c.seat_z_hi)
        ey = torch.zeros_like(p)
        ey[:, 1] = 1.0
        axis = quat_apply(self.bar.data.root_quat_w, ey)
        aligned = axis[:, 1].abs() >= c.seat_align
        return in_win & aligned

    def bottle_axis(self) -> torch.Tensor:
        """(N,3) world direction of the bottle's local +z (body->neck)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.zeros(self.env.num_envs, 3, device=self.env.device)
        ez[:, 2] = 1.0
        return quat_apply(self.wine.data.root_quat_w, ez)

    def bottle_lying(self) -> torch.Tensor:
        """(N,) bool: bottle axis within `lying_max_axis_z` of horizontal."""
        return self.bottle_axis()[:, 2].abs() <= self.cfg.lying_max_axis_z

    def bottle_in_bay(self, sign: torch.Tensor | None = None) -> torch.Tensor:
        """(N,) bool, geometric: bottle centre inside bay side `sign`'s interior box
        (centre well past the sill — doorway rests do not count — and under the roof)."""
        c = self.cfg
        if sign is None:
            sign = self.target_sign
        p = self._rel(self.wine)
        return (p[:, 0] >= c.bay_box_x[0]) & (p[:, 0] <= c.bay_box_x[1]) \
            & ((p[:, 1] - sign * c.bay_cy).abs() <= c.bay_box_dy) \
            & (p[:, 2] >= c.bay_box_z[0]) & (p[:, 2] <= c.bay_box_z[1])

    def _mouth_point(self) -> torch.Tensor:
        """(N,3) the TARGET slot mouth reference point, env-local."""
        c = self.cfg
        n = self.env.num_envs
        p = torch.zeros(n, 3, device=self.env.device)
        p[:, 0] = c.sill_x - 0.01
        p[:, 1] = self.target_sign * c.bay_cy
        p[:, 2] = c.mouth_z
        return p

    def _update_latches(self) -> None:
        c = self.cfg
        bar_p = self._rel(self.bar)
        d_seat = (bar_p - self.seat_point(self.target_sign)).norm(dim=-1)
        self._unbarred |= d_seat > c.unbar_dist
        d = (self._rel(self.wine) - self._mouth_point()).norm(dim=-1)
        app = (1.0 - d / c.approach_d0).clamp(0.0, 1.0) * self._unbarred.float()
        self._app_max = torch.maximum(self._app_max,
                                      torch.nan_to_num(app, nan=0.0, posinf=0.0, neginf=0.0))
        inside = self.bottle_in_bay() & self.bottle_lying()
        self._stowed |= inside
        rebar = (1.0 - d_seat / c.rebar_d0).clamp(0.0, 1.0) * inside.float()
        self._rebar_max = torch.maximum(
            self._rebar_max, torch.nan_to_num(rebar, nan=0.0, posinf=0.0, neginf=0.0))

    # ----- step-coupled bookkeeping (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No driven mechanics — the whole task is free rigid bodies + gravity. Latch
        rubric progress at sim rate."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool, physical outcomes only: bottle LYING at rest inside the TARGET
        bay AND the bar SEATED back in the target notches, everything settled."""
        c = self.cfg
        self._update_latches()
        wine_still = (self.wine.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.wine.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        bar_still = (self.bar.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.bar.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        return (self.bottle_in_bay() & self.bottle_lying() & self.bar_seated()
                & wine_still & bar_still)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15*unbarred + 0.20*mouth-approach (gated on unbarred)
        + 0.25*stowed + 0.25*rebar-while-stowed — all latched, ~0 for doing nothing,
        capped 0.85 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_unbar * self._unbarred.float() + c.w_app * self._app_max
                + c.w_stow * self._stowed.float()
                + c.w_rebar * self._rebar_max).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are placed/released through scene handles.
register_env("simgen", lambda: EnvCfg(scene="cellar_roll_stow", robot="null"))
