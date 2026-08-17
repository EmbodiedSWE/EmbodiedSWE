"""CarouselDishRackScene — align, insert through the one gate, stow (sim_gen task
`put_plate_in_colored_dish_rack_i86`).

Derived from rlbench/put_plate_in_colored_dish_rack, but STRATEGICALLY different: the
seed is one prehensile transport — lift THE plate off its stand and lower it into the
open slot of the color-named dish rack; the rack is static, the slot is open to the sky,
and the color binding is given for free. Here the rack is a MECHANISM the solver must
drive through a forced three-stage sequence, and the plate is never lifted at all. A
four-bay dish carousel (wedge bays split by radial fins on a turntable, spawn-authored
revolute joint) sits inside a fixed housing that is CLOSED from above by a solid roof
and walled all around except for ONE side gate; a feed shelf stands outside the gate
with the plate lying flat on it. Only the bay currently facing the gate is reachable —
and the roof gap over the fins is thinner than the plate, so a plate can never be
dropped in from above or levered from bay to bay. The goal bay is BLUE, identified by
the blue pointer arm + knob of the compass rose on top of the carousel shaft (each arm
points at its own bay; a matching color tab lies on each bay floor, visible through the
gate). Success requires the plate resting on the BLUE bay floor AND the carousel turned
so the blue bay is STOWED at least `stow_min_deg` away from the gate.

A solver therefore needs a different PLAN and different code structure than the seed:
(1) READ the compass rose and ROTATE the carousel (push a knob) until the blue bay
faces the gate — the goal location is not static, it must be actively positioned;
(2) SLIDE the plate flat across the shelf through the gate aperture until it rests in
the bay (a planar shuffleboard push under the roof — no grasp, no lift);
(3) ROTATE the loaded carousel again so the blue bay carries the plate away behind the
wall (the turntable carries the plate by floor friction + fin push — real transported
contact). The order is enforced by physics, not by the rubric: the gate is the only way
in, only the aligned bay is reachable, and stowing is only meaningful after insertion.

success(): plate center inside the BLUE wedge in the CAROUSEL's body frame (radial band
+ angular window + on-the-floor z window that rejects roof-top, fin-top and straddling
poses), carousel stowed (|blue-bay-to-gate offset| >= stow_min_deg), and everything
PERSISTENTLY still (counter-latch — teleport writes and pushes reset it).

score(), latched in post_step (credit never evaporates): 0.10 once the plate has ever
rested fully inside the carousel (any bay), + 0.15 once the blue bay has ever been
ALIGNED with the gate (|offset| <= aligned_max_deg — reset guarantees the episode
starts >= `min_offset_deg` away, so a null policy can never earn it), + 0.30 once the
plate has ever rested in the BLUE bay for `seat_steps` consecutive slow steps (no
fly-through credit); capped at 0.55; exactly 1.0 iff success(). Null policy scores ~0.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - housing (KINEMATIC, never teleported — the joint's body0 anchor is world-fixed):
    ground-standing base disc, 14 tangential wall segments leaving one 45 deg gate
    opening at housing +x, a solid 4-box roof with a small square center hole for the
    shaft, and the feed shelf whose top sits 1 mm ABOVE the platform top (the plate
    steps DOWN into the carousel, never climbs).
  - carousel (dynamic, one rigid body): platform disc, hub, 4 radial fins at the bay
    boundaries, center shaft rising through the roof hole, and the compass rose on top
    (4 colored pointer arms with graspable end knobs, one per bay) + 4 colored floor
    tabs (visual-only). Authored mass/CoM-on-axis/inertia; angular damping stops the
    spin promptly. Revolute joint (axis Z, free) to the housing, authored in the
    spawner so it exists before the physics parse; the joint pair's collisions are
    filtered by design (5 mm annular clearance everywhere).
  - plate (dynamic): a flat ceramic-white cylinder, diameter deliberately larger than
    every roof/wall clearance except the gate.
Friction materials are bound explicitly (the default-material trap): the platform
carries the plate like a lazy susan when the carousel turns.

Per-episode randomization (readback-verified in smoke): the carousel's initial angle —
the blue bay starts a signed uniform `min_offset_deg`..180 deg away from the gate, so
the required rotation direction and magnitude change every episode — and the plate's
spawn pose on the shelf (xy jitter + free yaw).

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

BAY_COLORS = (
    ("red", (0.85, 0.15, 0.15)),
    ("green", (0.15, 0.70, 0.20)),
    ("blue", (0.15, 0.35, 0.90)),
    ("yellow", (0.90, 0.80, 0.10)),
)
BLUE_BAY = 2  # bay k has its wedge center at carousel-frame angle k * 90 deg


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


def _friction_material(stage, path: str, static: float, dynamic: float):
    """One USD physics material (explicit binding — the default-material ~0.5 trap)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, contact_offset: float, material) -> None:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float, material=None,
         yaw: float = 0.0, collide: bool = True) -> None:
    """Author one box child prim (translate -> orient -> scale, authored once each —
    idempotent per prim, the duplicate-xformOp trap)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw:
        half = 0.5 * yaw
        sxf.AddOrientOp().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, 0.0, math.sin(half))))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        _collide(seg.GetPrim(), contact_offset, material)


def _cyl(stage, path: str, radius: float, height: float, center, color,
         contact_offset: float, material=None, collide: bool = True) -> None:
    """Author one z-axis cylinder child prim (same shape family the corpus's native
    CylinderCfg bodies use)."""
    from pxr import Gf, UsdGeom

    cy = UsdGeom.Cylinder.Define(stage, path)
    cy.CreateRadiusAttr(float(radius))
    cy.CreateHeightAttr(float(height))
    cy.CreateAxisAttr("Z")
    cy.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                         Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.Xformable(cy.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    cy.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        _collide(cy.GetPrim(), contact_offset, material)


def _spawn_housing(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC housing; body origin ON the carousel axis AT platform-top height
    (joint anchors read (0,0,0)). Base disc down to the ground, 14 tangential wall
    segments (one 45 deg gate opening at +x), solid roof with a center shaft hole,
    feed shelf through the gate."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(20.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    grey = (0.42, 0.42, 0.46)
    dark = (0.30, 0.30, 0.34)
    wood = (0.62, 0.48, 0.30)

    # base disc: ground (z=-h0) up to just under the platform bottom
    _cyl(stage, f"{prim_path}/base", cfg.wall_r_in + cfg.wall_t + 0.001,
         cfg.h0 - 0.025, (0.0, 0.0, -(cfg.h0 + 0.025) / 2), dark, co, mat)
    # wall: 16-gon segments, skip the two flanking the gate (opening = +/-22.5 deg)
    n_seg = 16
    r_mid = cfg.wall_r_in + cfg.wall_t / 2
    seg_len = 2.0 * r_mid * math.tan(math.pi / n_seg)
    wall_h = cfg.wall_z1 - cfg.wall_z0
    for k in range(n_seg):
        th = (2.0 * math.pi / n_seg) * k + math.pi / n_seg
        if abs(((th + math.pi) % (2.0 * math.pi)) - math.pi) < math.radians(cfg.gate_half_deg):
            continue  # the gate opening
        _box(stage, f"{prim_path}/wall_{k}", (seg_len, cfg.wall_t, wall_h),
             (r_mid * math.cos(th), r_mid * math.sin(th), (cfg.wall_z0 + cfg.wall_z1) / 2),
             grey, co, mat, yaw=th + math.pi / 2)
    # roof: solid 4-box annulus with a center square hole for the shaft
    zr = (cfg.roof_z0 + cfg.roof_z1) / 2
    tr = cfg.roof_z1 - cfg.roof_z0
    ext = cfg.wall_r_in + cfg.wall_t + 0.003  # half-width of the roof square
    hole = cfg.roof_hole_half
    _box(stage, f"{prim_path}/roof_a", (2 * ext, ext - hole, tr),
         (0.0, (ext + hole) / 2, zr), grey, co, mat)
    _box(stage, f"{prim_path}/roof_b", (2 * ext, ext - hole, tr),
         (0.0, -(ext + hole) / 2, zr), grey, co, mat)
    _box(stage, f"{prim_path}/roof_c", (ext - hole, 2 * hole, tr),
         ((ext + hole) / 2, 0.0, zr), grey, co, mat)
    _box(stage, f"{prim_path}/roof_d", (ext - hole, 2 * hole, tr),
         (-(ext + hole) / 2, 0.0, zr), grey, co, mat)
    # feed shelf: top 1 mm ABOVE the platform top, ground-standing, through the gate
    sx, sy = cfg.shelf_size
    _box(stage, f"{prim_path}/shelf", (sx, sy, cfg.h0 + 0.001),
         (cfg.shelf_x0 + sx / 2, 0.0, (0.001 - cfg.h0) / 2), wood, co, mat)
    return root


def _spawn_carousel(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The turntable, ONE dynamic rigid body; body origin on the axis at platform-top
    height. Platform disc, hub, 4 radial fins (bay boundaries at 45+90k deg), shaft
    through the roof hole, compass rose (4 colored pointer arms + end knobs, arm k
    points at bay k's wedge center) and 4 visual-only colored floor tabs. Mass/CoM(on
    axis)/inertia authored; angular damping stops the spin promptly."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.01))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.012, 0.012, 0.018))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(float(cfg.ang_damping))
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    silver = (0.68, 0.68, 0.72)
    steel = (0.52, 0.52, 0.58)
    # platform disc (top at z=0) + hub + fins + shaft
    _cyl(stage, f"{prim_path}/platform", cfg.plat_r, cfg.plat_t,
         (0.0, 0.0, -cfg.plat_t / 2), silver, co, mat)
    _cyl(stage, f"{prim_path}/hub", cfg.hub_r, cfg.fin_h,
         (0.0, 0.0, cfg.fin_h / 2), steel, co, mat)
    fin_len = cfg.plat_r - cfg.hub_r
    for k in range(4):
        th = math.radians(45.0 + 90.0 * k)
        rmid = (cfg.plat_r + cfg.hub_r) / 2
        _box(stage, f"{prim_path}/fin_{k}", (fin_len, cfg.fin_t, cfg.fin_h),
             (rmid * math.cos(th), rmid * math.sin(th), cfg.fin_h / 2),
             steel, co, mat, yaw=th)
    _cyl(stage, f"{prim_path}/shaft", cfg.shaft_r, cfg.shaft_top,
         (0.0, 0.0, cfg.shaft_top / 2), steel, co, mat)
    # compass rose: arm k + end knob point at bay k's wedge center; floor tab in bay k
    for k, (_nm, col) in enumerate(BAY_COLORS):
        th = math.radians(90.0 * k)
        cth, sth = math.cos(th), math.sin(th)
        _box(stage, f"{prim_path}/arm_{k}", (cfg.arm_len, 0.016, 0.014),
             ((0.015 + cfg.arm_len / 2) * cth, (0.015 + cfg.arm_len / 2) * sth,
              cfg.shaft_top + 0.007), col, co, mat, yaw=th)
        _cyl(stage, f"{prim_path}/knob_{k}", cfg.knob_r, cfg.knob_h,
             (cfg.knob_orbit * cth, cfg.knob_orbit * sth,
              cfg.shaft_top + 0.014 + cfg.knob_h / 2), col, co, mat)
        _box(stage, f"{prim_path}/tab_{k}", (0.035, 0.025, 0.003),
             (0.055 * cth, 0.055 * sth, 0.002), col, co, collide=False)

    # The revolute axis to the sibling Housing — authored IN THE SPAWNER so it exists
    # BEFORE the physics parse. Axis Z, free (no limits); both body origins sit on the
    # axis at the same height, so every anchor is (0,0,0); pair collision stays
    # FILTERED (5 mm annular clearance everywhere by design).
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/axle")
    j.CreateBody0Rel().SetTargets([f"{base}/Housing"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(False)
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazy: module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "housing" not in _SPAWNER_CACHE:

        @configclass
        class HousingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_housing)
            h0: float = 0.06
            wall_r_in: float = 0.165
            wall_t: float = 0.012
            wall_z0: float = -0.02
            wall_z1: float = 0.095
            roof_z0: float = 0.10
            roof_z1: float = 0.112
            roof_hole_half: float = 0.018
            gate_half_deg: float = 22.5
            shelf_x0: float = 0.162
            shelf_size: tuple = (0.20, 0.16)
            mu_static: float = 0.6
            mu_dynamic: float = 0.5
            contact_offset: float = 0.0015

        @configclass
        class CarouselSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_carousel)
            plat_r: float = 0.16
            plat_t: float = 0.015
            hub_r: float = 0.04
            fin_t: float = 0.008
            fin_h: float = 0.09
            shaft_r: float = 0.012
            shaft_top: float = 0.17
            arm_len: float = 0.09
            knob_r: float = 0.011
            knob_h: float = 0.04
            knob_orbit: float = 0.095
            mass: float = 1.2
            ang_damping: float = 1.2
            mu_static: float = 0.6
            mu_dynamic: float = 0.5
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(housing=HousingSpawnerCfg, carousel=CarouselSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CarouselDishRackSceneCfg(BaseCfg):
    """Config for `CarouselDishRackScene`. `__post_init__` asserts the honesty
    invariants: the gate passes the plate with margin, the roof gap over the fins is
    thinner than the plate (no top drop-in, no bay hopping), the z window accepts only
    a plate resting on the bay floor, the radial band rejects a plate straddling the
    gate threshold, and the sampled initial offset can never start aligned."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    aligned_max_deg: float = tunable(15.0)  # |blue-bay-to-gate offset| counting as ALIGNED
    stow_min_deg: float = tunable(45.0)  # |offset| required for STOWED (success clause)
    bay_half_deg: float = tunable(30.0)  # angular half-window of bay membership
    r_lo: float = tunable(0.055)  # radial band of bay membership (carousel frame) ...
    r_hi: float = tunable(0.120)  # ... hub contact at 0.085, wedge center at ~0.10
    entered_r: float = tunable(0.125)  # plate fully inside the carousel (any bay)
    z_lo: float = tunable(0.000)  # plate-center z window over the bay floor ...
    z_hi: float = tunable(0.030)  # ... rejects fin-top perches, roof-top and shelf-side
    settle_lin: float = tunable(0.05)  # max plate |lin vel| counted still (m/s)
    settle_ang: float = tunable(1.0)  # max plate |ang vel| counted still (rad/s)
    settle_car: float = tunable(0.10)  # max carousel |ang vel| counted still (rad/s)
    settle_steps_min: int = tunable(24)  # stillness must persist this many steps
    seat_steps: int = tunable(24)  # in-blue-bay must persist this many slow steps to latch

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    min_offset_deg: float = tunable(55.0)  # |initial blue-bay offset| lower bound
    max_offset_deg: float = tunable(180.0)  # ... upper bound (sign random too)
    plate_spawn_x: float = tunable(0.27)  # plate spawn center on the shelf (housing x)
    plate_jitter: tuple = tunable((0.02, 0.03))  # uniform +/- xy jitter at reset
    plate_yaw_deg: float = tunable(180.0)  # uniform +/- spawn yaw (a plate is round —
    # yaw is visual-only, but it exercises the wrench frame-drag path honestly)

    # --- info: structure ---------------------------------------------------------------------
    h0: float = info(0.06)  # platform-top height above the ground = body origins' z
    plat_r: float = info(0.16)
    plat_t: float = info(0.015)
    hub_r: float = info(0.04)
    fin_t: float = info(0.008)
    fin_h: float = info(0.09)
    wall_r_in: float = info(0.165)
    wall_t: float = info(0.012)
    roof_z0: float = info(0.10)  # roof underside above the platform top
    roof_z1: float = info(0.112)
    gate_half_deg: float = info(22.5)
    shelf_x0: float = info(0.162)  # shelf inner edge (housing x)
    shelf_size: tuple = info((0.20, 0.16))
    shaft_top: float = info(0.17)
    knob_orbit: float = info(0.095)
    knob_r: float = info(0.011)
    plate_r: float = info(0.045)
    plate_h: float = info(0.012)
    plate_m: float = info(0.12)
    car_mass: float = info(1.2)
    mu_static: float = info(0.6)
    mu_dynamic: float = info(0.5)
    contact_offset: float = info(0.0015)

    # Derived (filled in __post_init__).
    plate_rest_z: float = field(default=None, init=False)  # plate center over the floor

    def __post_init__(self) -> None:
        self.plate_rest_z = self.plate_h / 2 + self.contact_offset
        gate_chord = 2.0 * self.wall_r_in * math.sin(math.radians(self.gate_half_deg))
        assert gate_chord >= 2 * self.plate_r + 0.025, \
            f"gate must pass the plate with margin (chord {gate_chord:.3f})"
        assert self.roof_z0 - self.fin_h < self.plate_h, \
            "roof gap over the fins must be thinner than the plate (no bay hopping)"
        assert self.roof_z0 > self.plate_h + 0.02, "gate aperture must pass the plate"
        assert self.z_lo <= self.plate_rest_z <= self.z_hi, "z window must accept rest"
        assert self.z_hi < self.fin_h + self.plate_h / 2, \
            "z window must reject a plate perched on a fin or the hub"
        assert self.roof_z1 + self.plate_h / 2 > self.z_hi, "z window must reject roof-top"
        assert self.r_lo < self.hub_r + self.plate_r < self.r_hi, \
            "radial band must accept the hub-contact rest"
        assert self.r_hi <= self.plat_r - self.plate_r + 0.005, \
            "radial band must reject a plate straddling the platform edge"
        assert self.entered_r <= self.plat_r - self.plate_r + 0.01
        assert self.min_offset_deg > self.aligned_max_deg + 20.0, \
            "an episode can never start aligned (null policy earns nothing)"
        assert self.stow_min_deg > self.aligned_max_deg + 15.0, "align/stow hysteresis"
        assert 2 * self.knob_r <= 0.075, "knob must fit a parallel jaw"
        assert self.bay_half_deg < 45.0, "bay window must stay inside the wedge"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("carousel_dish_rack")
class CarouselDishRackScene(BaseScene):
    cfg: CarouselDishRackSceneCfg

    def __init__(self, cfg: CarouselDishRackSceneCfg | None = None) -> None:
        super().__init__(cfg or CarouselDishRackSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_static, dynamic_friction=c.mu_dynamic,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "housing": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Housing",
                spawn=spawners["housing"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    h0=c.h0, wall_r_in=c.wall_r_in, wall_t=c.wall_t,
                    roof_z0=c.roof_z0, roof_z1=c.roof_z1,
                    gate_half_deg=c.gate_half_deg, shelf_x0=c.shelf_x0,
                    shelf_size=c.shelf_size, mu_static=c.mu_static,
                    mu_dynamic=c.mu_dynamic, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.h0)),
            ),
            "carousel": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Carousel",
                spawn=spawners["carousel"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.car_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    plat_r=c.plat_r, plat_t=c.plat_t, hub_r=c.hub_r,
                    fin_t=c.fin_t, fin_h=c.fin_h, shaft_top=c.shaft_top,
                    knob_orbit=c.knob_orbit, knob_r=c.knob_r, mass=c.car_mass,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.h0)),
            ),
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate",
                spawn=sim_utils.CylinderCfg(
                    radius=c.plate_r, height=c.plate_h, axis="Z",
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.10, angular_damping=0.20,
                        disable_gravity=False),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.plate_m),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_static, dynamic_friction=c.mu_dynamic,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.93, 0.93, 0.88)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.plate_spawn_x, 0.0, c.h0 + 0.008)),
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
        self.housing: RigidObject = env.iscene["housing"]
        self.carousel: RigidObject = env.iscene["carousel"]
        self.plate: RigidObject = env.iscene["plate"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.offset0 = torch.zeros(n, device=dev)  # sampled initial blue-bay offset (rad)
        self.entered_latch = torch.zeros(n, device=dev)
        self.aligned_latch = torch.zeros(n, device=dev)
        self.inbay_latch = torch.zeros(n, device=dev)
        self.seat_count = torch.zeros(n, device=dev)
        self.still_count = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: the housing NEVER moves (kinematic joint anchor). Sample the
        carousel's initial angle so the blue bay starts a signed
        `min_offset_deg`..`max_offset_deg` away from the gate (writing the hinge DOF —
        consistent with the joint), scatter the plate on the shelf, zero the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- carousel: blue bay a random signed offset away from the gate ---
        lo, hi = math.radians(c.min_offset_deg), math.radians(c.max_offset_deg)
        mag = lo + torch.rand(m, device=dev) * (hi - lo)
        sign = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        off = sign * mag
        self.offset0[env_ids] = off
        yaw = off - math.pi  # blue bay center (carousel-frame 180 deg) at housing angle=off
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = origin
        st[:, 2] += c.h0
        st[:, 3] = torch.cos(yaw / 2)
        st[:, 6] = torch.sin(yaw / 2)
        self.carousel.write_root_state_to_sim(st, env_ids)

        # --- plate: flat on the shelf, xy jitter + free yaw ---
        jx, jy = c.plate_jitter
        yawp = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.plate_yaw_deg) / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.plate_spawn_x + (torch.rand(m, device=dev) * 2 - 1) * jx
        st[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * jy
        st[:, 2] = c.h0 + 0.001 + c.plate_h / 2 + 0.002
        st[:, 0:3] += origin
        st[:, 3] = torch.cos(yawp)
        st[:, 6] = torch.sin(yawp)
        self.plate.write_root_state_to_sim(st, env_ids)

        for buf in (self.entered_latch, self.aligned_latch, self.inbay_latch,
                    self.seat_count, self.still_count):
            buf[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "carousel": self.carousel.data.root_state_w[env_ids].clone(),
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "offset0": self.offset0[env_ids].clone(),
            "entered_latch": self.entered_latch[env_ids].clone(),
            "aligned_latch": self.aligned_latch[env_ids].clone(),
            "inbay_latch": self.inbay_latch[env_ids].clone(),
            "seat_count": self.seat_count[env_ids].clone(),
            "still_count": self.still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.carousel.write_root_state_to_sim(state["carousel"], env_ids)
        self.plate.write_root_state_to_sim(state["plate"], env_ids)
        self.offset0[env_ids] = state["offset0"]
        self.entered_latch[env_ids] = state["entered_latch"]
        self.aligned_latch[env_ids] = state["aligned_latch"]
        self.inbay_latch[env_ids] = state["inbay_latch"]
        self.seat_count[env_ids] = state["seat_count"]
        self.still_count[env_ids] = state["still_count"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A covered dish carousel stands on the floor: a fixed round housing "
            f"(~{2 * (c.wall_r_in + c.wall_t) * 100:.0f} cm across, walls "
            f"~{(c.h0 + c.roof_z1) * 100:.0f} cm tall) with a SOLID ROOF and a single "
            f"open GATE in its side wall (a {2 * c.gate_half_deg:.0f}-degree opening, "
            f"~{2 * c.wall_r_in * math.sin(math.radians(c.gate_half_deg)) * 100:.0f} cm "
            f"wide). A wooden feed shelf stands outside the gate, its top flush with "
            f"the turntable inside, and a round white plate "
            f"({2 * c.plate_r * 100:.0f} cm across, {c.plate_h * 1000:.0f} mm thick) "
            f"lies flat on the shelf. Inside, a turntable divided by four radial fins "
            f"into four wedge-shaped bays spins freely on a vertical axle. Each bay is "
            f"color-coded RED, GREEN, BLUE or YELLOW: a compass rose on top of the "
            f"axle (above the roof) has four colored pointer arms, each ending in a "
            f"small knob, and each arm points exactly at its own bay; a matching color "
            f"tab lies on each bay floor, visible through the gate. Only the bay "
            f"currently facing the gate can be reached — the roof blocks all entry "
            f"from above, and the gap over the fins is thinner than the plate, so the "
            f"plate can never be dropped in or moved between bays.\n"
            f"Goal, in the order physics forces: (1) rotate the turntable (push or "
            f"pull a pointer knob) until the BLUE arm points at the gate, i.e. the "
            f"blue bay faces the opening (the blue bay starts at least "
            f"{c.min_offset_deg:.0f} degrees away, a random side each episode); "
            f"(2) slide the plate FLAT along the shelf, through the gate, into the "
            f"blue bay until it rests on the bay floor (against the hub is perfect — "
            f"never lift it over the wall); (3) rotate the turntable at least "
            f"{c.stow_min_deg:.0f} degrees more, either way, so the loaded blue bay is "
            f"STOWED behind the wall, away from the gate. Success is judged with "
            f"everything at rest: the plate on the BLUE bay floor and the blue bay "
            f"stowed. A plate in any other bay, left on the shelf, jammed half through "
            f"the gate, perched on the roof, or a blue bay still facing the gate does "
            f"not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Rotate the carousel by its pointer knobs until the blue bay faces the "
            "side gate, slide the plate flat through the gate onto the blue bay "
            "floor, then rotate the carousel at least a quarter turn so the loaded "
            "blue bay is stowed behind the wall. A plate in any other bay, left "
            "outside, or a blue bay still at the gate fails."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    @staticmethod
    def _yaw(q: torch.Tensor) -> torch.Tensor:
        """(N,) yaw about z from a (N,4) wxyz quaternion."""
        return torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                           1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))

    @staticmethod
    def _wrap(a: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(a), torch.cos(a))

    def bay_offset(self) -> torch.Tensor:
        """(N,) signed angle (rad) from the gate direction (housing +x) to the BLUE
        bay's wedge center. 0 = aligned with the gate."""
        return self._wrap(self._yaw(self.carousel.data.root_quat_w) + math.pi)

    def plate_local(self) -> torch.Tensor:
        """(N, 3) plate center in the CAROUSEL's body frame (origin on the axis at
        platform-top height — the bays live in this frame)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = self.plate.data.root_pos_w - self.carousel.data.root_pos_w
        return quat_apply_inverse(self.carousel.data.root_quat_w, rel)

    def in_blue_bay(self) -> torch.Tensor:
        """(N,) bool, geometric: plate center in the blue wedge's radial band +
        angular window, resting on the bay floor (z window rejects roof-top, fin-top
        and straddling poses)."""
        c = self.cfg
        loc = self.plate_local()
        r = loc[:, :2].norm(dim=-1)
        th = torch.atan2(loc[:, 1], loc[:, 0])
        d_ang = self._wrap(th - math.pi).abs()
        return ((r >= c.r_lo) & (r <= c.r_hi)
                & (d_ang <= math.radians(c.bay_half_deg))
                & (loc[:, 2] >= c.z_lo) & (loc[:, 2] <= c.z_hi))

    def entered(self) -> torch.Tensor:
        """(N,) bool: plate resting fully inside the carousel (any bay)."""
        c = self.cfg
        loc = self.plate_local()
        r = loc[:, :2].norm(dim=-1)
        return (r <= c.entered_r) & (loc[:, 2] >= c.z_lo) & (loc[:, 2] <= c.z_hi)

    def aligned(self) -> torch.Tensor:
        """(N,) bool: blue bay within `aligned_max_deg` of the gate."""
        return self.bay_offset().abs() <= math.radians(self.cfg.aligned_max_deg)

    def stowed(self) -> torch.Tensor:
        """(N,) bool: blue bay at least `stow_min_deg` away from the gate."""
        return self.bay_offset().abs() >= math.radians(self.cfg.stow_min_deg)

    def _still_now(self) -> torch.Tensor:
        c = self.cfg
        return ((self.plate.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.plate.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
                & (self.carousel.data.root_ang_vel_w.norm(dim=-1) < c.settle_car))

    def settled(self) -> torch.Tensor:
        """(N,) bool: stillness has PERSISTED `settle_steps_min` consecutive steps
        (counter-latch in post_step — teleport writes and pushes reset it)."""
        return self.still_count >= self.cfg.settle_steps_min

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Stillness counter + progress latches: plate ever fully inside; blue bay ever
        aligned with the gate; plate ever seated in the blue bay for `seat_steps`
        consecutive SLOW steps (a fly-through never latches)."""
        c = self.cfg
        self.still_count = (self.still_count + 1.0) * self._still_now().float()
        self.entered_latch = torch.maximum(self.entered_latch, self.entered().float())
        self.aligned_latch = torch.maximum(self.aligned_latch, self.aligned().float())
        slow = ((self.plate.data.root_lin_vel_w.norm(dim=-1) < 2.0 * c.settle_lin)
                & (self.plate.data.root_ang_vel_w.norm(dim=-1) < 2.0 * c.settle_ang))
        self.seat_count = (self.seat_count + 1.0) * (self.in_blue_bay() & slow).float()
        self.inbay_latch = torch.maximum(
            self.inbay_latch, (self.seat_count >= c.seat_steps).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: plate resting on the BLUE bay floor, blue bay STOWED away from
        the gate, everything persistently still."""
        return self.in_blue_bay() & self.stowed() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 plate ever fully inside (any bay) + 0.15 blue bay
        ever aligned with the gate + 0.30 plate ever seated in the blue bay (all
        latched — credit never evaporates); 1.0 iff success(). Null policy ~0 (the
        episode starts misaligned with the plate outside)."""
        base = (0.10 * self.entered_latch + 0.15 * self.aligned_latch
                + 0.30 * self.inbay_latch).clamp(0.0, 0.55)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="carousel_dish_rack", robot="null",
                                      env_spacing=3.0))
