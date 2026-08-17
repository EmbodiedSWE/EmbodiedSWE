"""JuiceCarouselScene — index a shrouded carousel to its window, extract the ORANGE
juice carton through the aperture, deliver it to the floor basket
(sim_gen task `libero_pick_orange_juice_i256`).

Derived from libero/libero_pick_orange_juice — "pick the orange juice and place it in
the basket": target among distractors, one free grasp-carry-release into an always-open
static basket. STRATEGICALLY different: here the target starts PHYSICALLY INACCESSIBLE.
Three cartons (orange JUICE = target, white MILK and green SODA = distractors) stand on
a powered-lazy-susan style CAROUSEL: a rotating disc under a fixed shroud (ring wall +
roof) whose only opening is one front WINDOW sector. At reset the juice carton sits
45-175 degrees away from the window — behind the wall, unreachable and unmovable-out.
The solver must first REGULATE A CONTINUOUS ROTARY DOF: spin the carousel (crank knob
on top, or push the exposed carton tangentially) until the juice slot aligns with the
window; then EXTRACT the carton through the aperture (a low radial drag onto the flush
porch — the roof forbids lifting inside the drum); then deliver it to the tan basket on
the floor. The two distractors must STAY on the carousel. So the plan skeleton changes
from "grasp target, place into open receptacle" to "servo a rotary mechanism into
alignment, pull the payload through the one aperture, then deliver" — closed-loop
positioning of a mechanism, not a free pick.

The order rotate -> extract -> deliver is PHYSICALLY forced, not rubric-forced: the
wall blocks any misaligned extraction (smoke proves the same drag that exits an aligned
carton wedges a misaligned one against the wall), and a carton cannot be delivered
before it is out.

Goal state (success()): juice carton settled INSIDE the basket (basket-frame
containment below the rim), both distractors still riding the carousel disc, carousel
at rest, everything settled and finite. score(): latched, monotone along the
demonstrated solution — 0.20 x best alignment progress (normalized wrapped-azimuth
error reduction, running max) + 0.15 alignment latch (juice slot ever settled inside
the align band at the window) + 0.30 extraction latch (juice ever calm outside the
shroud) ; 1.0 iff success(). Null policy ~0 (reset guarantees >= 45 deg misalignment
and the carousel holds its angle: nothing latches).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - base (dynamic 25 kg fixture — NEVER kinematic, so the revolute anchor follows the
    reset teleport): ground slab, pedestal, 12-segment ring wall with one 64 deg
    window sector, flush porch chords outside the window, roof plates with a central
    axle hole.
  - turntable (the ONLY mechanism DOF): disc + axle + crank arm + red knob, on a
    frictionless vertical RevoluteJoint into the base (no limits — continuous), with
    angular damping so pushed rotation decays instead of coasting (constant-torque
    coast overshoot trap). The joint pair base<->turntable is collision-filtered (USD
    default): the disc is guided by the joint, the cartons are blocked by REAL wall
    contact.
  - cartons (60 x 60 x 110 mm, 0.35 kg): JUICE orange, MILK white, SODA green, at
    three 120 deg slots on the disc. Low disc/carton friction (0.30 pair-averaged) so
    a CoM-height drag SLIDES the carton (mu * h_com < half-width: it can never tip
    while dragged), while rotation accelerations stay far below the slip bound.
  - basket (tan walled tray, on the floor beside the machine): the delivery target.

Per-episode randomization (readback-verified in smoke): base xy jitter + FREE yaw (the
window faces a random direction), the juice's slot assignment (which of the three
120 deg slots), the initial misalignment offset (sign x 45..175 deg), and the basket's
floor position (arc around the window side) + yaw. Heavy imports (isaaclab, pxr) are
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
    from robobench.core import BaseEnv


# ----- geometry constants (single source of truth: spawners + cfg asserts + rubric) ------------
SLAB_S = 0.52  # base ground slab square
SLAB_T = 0.024  # slab thickness
PED_S = 0.090  # pedestal square
PED_H = 0.056  # pedestal height (slab top -> 0.080)
DISC_R = 0.160  # turntable disc radius
DISC_T = 0.020  # disc thickness
DISC_Z = 0.093  # disc CENTER height (base frame) — the revolute anchor; top at 0.103
DISC_TOP = DISC_Z + DISC_T / 2  # 0.103, the riding surface
SLOT_R = 0.105  # carton slot radius on the disc
ITEM_S = 0.060  # carton square cross-section
ITEM_H = 0.110  # carton height
WALL_RI = 0.175  # ring wall inner radius
WALL_T = 0.012  # ring wall thickness
WALL_Z0 = 0.085  # ring wall bottom (below disc top: nothing passes underneath)
WALL_Z1 = 0.260  # ring wall top = roof underside
WIN_HALF_DEG = 32.0  # window sector half-angle (window centered on base +x)
ROOF_T = 0.014  # roof plate thickness (roof spans 0.260 -> 0.274)
ROOF_HALF = 0.23  # roof square half-size
ROOF_HOLE = 0.040  # roof central hole half-size (axle passes through)
AXLE_S = 0.040  # axle square
AXLE_TOP = 0.300  # axle top height (base frame)
CRANK_R = 0.125  # knob orbit radius
KNOB_S = 0.030  # knob square (parallel-jaw pinch / push target)
KNOB_H = 0.070  # knob height
PORCH_RO = 0.300  # porch outer radius (flush landing outside the window)
PORCH_RI = 0.153  # porch inner radius — UNDERLAPS the disc rim (base<->turntable is
# joint-collision-filtered, so the overlap is free): the disc->porch handover is a
# continuous flush surface with NO annular gap (a 5 mm gap edge-hooks a sliding
# carton's trailing corner and wedges it, verified on the forge)
BASKET_S = 0.200  # basket plate square
BASKET_T = 0.014  # basket plate thickness
RIM_H = 0.075  # basket rim height
RIM_T = 0.010  # basket rim thickness


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (N,3) by quaternions q (N,4), wxyz."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- custom compound spawners ----------------------------------------------------------------
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
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None, yaw_deg: float = 0.0) -> None:
    """Author one colliding box child prim (translate -> orient -> scale, authored
    once — idempotent per prim, the duplicate-xformOp trap)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw_deg:
        half = math.radians(yaw_deg) / 2
        sxf.AddOrientOp().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, 0.0, math.sin(half))))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _cylinder(stage, path: str, radius: float, height: float, center, color,
              contact_offset: float, material=None) -> None:
    """One colliding z-axis cylinder child prim."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr("Z")
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(cyl.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float,
                *, iters: int = 16, vel_iters: int = 4, lin_damp: float = 0.05,
                ang_damp: float = 0.05):
    """Author a dynamic compound-body root with zeroed sleep (a sleeping fixture
    would freeze its joint anchors)."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(1.0)
    pxrb.CreateSolverPositionIterationCountAttr(iters)
    pxrb.CreateSolverVelocityIterationCountAttr(vel_iters)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    return root


def _spawn_base(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The shroud fixture: slab + pedestal + 12-segment ring wall with one window
    sector + porch chords + roof plates with a central axle hole. One heavy DYNAMIC
    compound body (the revolute anchor follows reset teleports). Origin = slab
    center on the ground; the window is centered on the base +x axis."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _rigid_root(stage, prim_path, translation, orientation, cfg.mass_props.mass,
                       lin_damp=0.5, ang_damp=0.5)
    slide = _friction_material(stage, f"{prim_path}/slide_mat", cfg.mu_slide_s, cfg.mu_slide_d)
    wallm = _friction_material(stage, f"{prim_path}/wall_mat", 0.20, 0.18)
    body = _friction_material(stage, f"{prim_path}/body_mat", 0.60, 0.55)
    dark = (0.24, 0.26, 0.29)
    gray = (0.42, 0.45, 0.48)
    wallc = (0.55, 0.58, 0.62)
    roofc = (0.34, 0.36, 0.40)
    # slab + pedestal
    _box(stage, f"{prim_path}/slab", (SLAB_S, SLAB_S, SLAB_T), (0.0, 0.0, SLAB_T / 2),
         dark, 0.003, material=body)
    _box(stage, f"{prim_path}/pedestal", (PED_S, PED_S, PED_H),
         (0.0, 0.0, SLAB_T + PED_H / 2), gray, 0.003, material=body)
    # ring wall: 12 segments covering [WIN_HALF, 360 - WIN_HALF]
    n_seg = 12
    span = 360.0 - 2 * WIN_HALF_DEG
    seg_arc = span / n_seg
    r_mid = WALL_RI + WALL_T / 2
    seg_len = 2 * (WALL_RI + WALL_T) * math.tan(math.radians(seg_arc / 2)) + 0.006
    wall_h = WALL_Z1 - WALL_Z0
    for k in range(n_seg):
        ang = WIN_HALF_DEG + seg_arc * (k + 0.5)
        a = math.radians(ang)
        _box(stage, f"{prim_path}/wall_{k}", (WALL_T, seg_len, wall_h),
             (r_mid * math.cos(a), r_mid * math.sin(a), (WALL_Z0 + WALL_Z1) / 2),
             wallc, 0.003, material=wallm, yaw_deg=ang)
    # porch: 3 flush chord plates spanning the window sector, tops at DISC_TOP,
    # reaching inward UNDER the disc rim (gap-free handover; overlap is filtered)
    p_len = PORCH_RO - PORCH_RI
    p_mid = (PORCH_RO + PORCH_RI) / 2
    for k, ang in enumerate((-2 * WIN_HALF_DEG / 3, 0.0, 2 * WIN_HALF_DEG / 3)):
        a = math.radians(ang)
        _box(stage, f"{prim_path}/porch_{k}", (p_len, 0.095, 0.012),
             (p_mid * math.cos(a), p_mid * math.sin(a), DISC_TOP - 0.006),
             gray, 0.003, material=slide, yaw_deg=ang)
    # roof plates (central ROOF_HOLE square hole for the axle)
    zc = WALL_Z1 + ROOF_T / 2
    _box(stage, f"{prim_path}/roof_n", (2 * ROOF_HALF, ROOF_HALF - ROOF_HOLE, ROOF_T),
         (0.0, (ROOF_HALF + ROOF_HOLE) / 2, zc), roofc, 0.003, material=body)
    _box(stage, f"{prim_path}/roof_s", (2 * ROOF_HALF, ROOF_HALF - ROOF_HOLE, ROOF_T),
         (0.0, -(ROOF_HALF + ROOF_HOLE) / 2, zc), roofc, 0.003, material=body)
    _box(stage, f"{prim_path}/roof_e", (ROOF_HALF - ROOF_HOLE, 2 * ROOF_HOLE, ROOF_T),
         ((ROOF_HALF + ROOF_HOLE) / 2, 0.0, zc), roofc, 0.003, material=body)
    _box(stage, f"{prim_path}/roof_w", (ROOF_HALF - ROOF_HOLE, 2 * ROOF_HOLE, ROOF_T),
         (-(ROOF_HALF + ROOF_HOLE) / 2, 0.0, zc), roofc, 0.003, material=body)
    return root


def _spawn_turntable(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The carousel: disc + axle + crank arm + red knob, one dynamic compound body,
    on a vertical RevoluteJoint into the sibling base (no limits — continuous;
    angular damping is the only resistance: no joint friction — physxJoint
    jointFriction is silently ignored outside articulations anyway). Origin = disc
    CENTER (the joint anchor)."""
    import omni.usd
    from pxr import Gf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _rigid_root(stage, prim_path, translation, orientation, cfg.mass_props.mass,
                       ang_damp=cfg.wheel_damp, lin_damp=0.2)
    slide = _friction_material(stage, f"{prim_path}/disc_mat", cfg.mu_slide_s, cfg.mu_slide_d)
    discc = (0.25, 0.28, 0.32)
    darkm = (0.30, 0.32, 0.35)
    red = (0.80, 0.14, 0.10)
    _cylinder(stage, f"{prim_path}/disc", DISC_R, DISC_T, (0.0, 0.0, 0.0),
              discc, 0.003, material=slide)
    axle_h = AXLE_TOP - (DISC_Z + DISC_T / 2)
    _box(stage, f"{prim_path}/axle", (AXLE_S, AXLE_S, axle_h),
         (0.0, 0.0, DISC_T / 2 + axle_h / 2), darkm, 0.003, material=slide)
    arm_z = AXLE_TOP - DISC_Z + 0.009  # local z of crank-arm center
    _box(stage, f"{prim_path}/crank", (CRANK_R + 0.02, 0.028, 0.018),
         ((CRANK_R + 0.02) / 2 - 0.01, 0.0, arm_z), darkm, 0.003, material=slide)
    _box(stage, f"{prim_path}/knob", (KNOB_S, KNOB_S, KNOB_H),
         (CRANK_R, 0.0, arm_z + 0.009 + KNOB_H / 2), red, 0.003, material=slide)

    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/spin")
    j.CreateBody0Rel().SetTargets([f"{base}/Base"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(DISC_Z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    # no limits: continuous rotation (a limit near +/-180 would sit on the wrap
    # boundary — the revolute slingshot trap)
    return root


def _spawn_carton(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One carton: a single colored box, low-friction faces (slides under a
    CoM-height drag long before it can tip: mu * h_com < half-width)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _rigid_root(stage, prim_path, translation, orientation, cfg.mass_props.mass,
                       lin_damp=0.05, ang_damp=0.05)
    mat = _friction_material(stage, f"{prim_path}/mat", cfg.mu_slide_s, cfg.mu_slide_d)
    _box(stage, f"{prim_path}/body", (ITEM_S, ITEM_S, ITEM_H), (0.0, 0.0, 0.0),
         cfg.color, 0.003, material=mat)
    return root


def _spawn_basket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The delivery basket: tan walled tray, dynamic, on the floor. Origin = plate
    center on the ground."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _rigid_root(stage, prim_path, translation, orientation, cfg.mass_props.mass,
                       lin_damp=0.3, ang_damp=0.3)
    mat = _friction_material(stage, f"{prim_path}/mat", 0.55, 0.50)
    tan = (0.76, 0.56, 0.30)
    _box(stage, f"{prim_path}/plate", (BASKET_S, BASKET_S, BASKET_T),
         (0.0, 0.0, BASKET_T / 2), tan, 0.004, material=mat)
    off = BASKET_S / 2 - RIM_T / 2
    for tag, sx, sy in (("e", 1.0, 0.0), ("w", -1.0, 0.0),
                        ("n", 0.0, 1.0), ("s", 0.0, -1.0)):
        size = (RIM_T, BASKET_S, RIM_H) if sx else (BASKET_S, RIM_T, RIM_H)
        _box(stage, f"{prim_path}/rim_{tag}", size,
             (sx * off, sy * off, BASKET_T + RIM_H / 2), tan, 0.003, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazy: module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "base" not in _SPAWNER_CACHE:

        @configclass
        class BaseSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_base)
            mu_slide_s: float = 0.30
            mu_slide_d: float = 0.28

        @configclass
        class TurntableSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_turntable)
            mu_slide_s: float = 0.30
            mu_slide_d: float = 0.28
            wheel_damp: float = 2.0

        @configclass
        class CartonSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_carton)
            color: tuple = (0.95, 0.55, 0.10)
            mu_slide_s: float = 0.30
            mu_slide_d: float = 0.28

        @configclass
        class BasketSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_basket)

        _SPAWNER_CACHE.update(base=BaseSpawnerCfg, turntable=TurntableSpawnerCfg,
                              carton=CartonSpawnerCfg, basket=BasketSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class JuiceCarouselSceneCfg(BaseCfg):
    """Config for `JuiceCarouselScene`. The honesty knobs are asserted in
    `__post_init__`: the window admits an aligned carton with margin while the align
    band stays inside the geometric pass band, the reset offset guarantees the null
    policy never aligns, a CoM-height drag slides the carton (it cannot tip), and the
    basket containment bound accepts every physically-inside rest while rejecting
    rim-perches and outside rests."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    align_deg: float = tunable(10.0)  # align latch: |window azimuth error| below this
    wheel_still: float = tunable(0.30)  # ... and carousel |ang vel z| below this (rad/s)
    extract_r: float = tunable(0.22)  # extraction latch: juice base-frame radius beyond
    extract_v: float = tunable(0.40)  # ... moving slower than this (m/s)
    basket_xy: float = tunable(0.065)  # in-basket: |x|,|y| in basket frame below this
    basket_z: tuple = tunable((0.028, 0.086))  # ... carton center z band (basket frame)
    disc_r_max: float = tunable(0.150)  # on-disc: base-frame xy radius below this
    disc_z_band: tuple = tunable((0.118, 0.200))  # ... center z band (base frame)
    settle_lin: float = tunable(0.08)  # settle gate (m/s) — above the GPU phantom-velocity
    # readback artifact band (frozen positions can read 0.05-0.07 m/s)

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    base_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the base at reset (m)
    base_yaw_max: float = tunable(180.0)  # uniform +/- base yaw (deg; FREE window heading)
    off_lo_deg: float = tunable(45.0)  # initial misalignment magnitude, lower bound
    off_hi_deg: float = tunable(175.0)  # ... upper bound (sign also random)
    basket_arc_deg: float = tunable(75.0)  # basket azimuth within +/- this of the window
    basket_r: float = tunable(0.48)  # basket distance from the base center (m)
    basket_r_jitter: float = tunable(0.06)  # uniform + jitter on that distance

    # --- info: structure -----------------------------------------------------------------------
    base_mass: float = info(25.0)  # heavy dynamic fixture (anchor follows teleports)
    wheel_mass: float = info(1.8)
    carton_mass: float = info(0.35)
    basket_mass: float = info(1.2)
    wheel_damp: float = info(2.0)  # turntable angular damping (coast time const 0.5 s)
    mu_slide_s: float = info(0.30)  # disc/porch/carton faces (the slide-not-tip pair)
    mu_slide_d: float = info(0.28)
    win_half_deg: float = info(WIN_HALF_DEG)

    # Derived (filled in __post_init__).
    slot_angles: tuple = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.slot_angles = (0.0, 2 * math.pi / 3, 4 * math.pi / 3)
        half_diag = ITEM_S * math.sqrt(2) / 2
        # -- the window admits an aligned carton with margin --
        chord = 2 * WALL_RI * math.sin(math.radians(WIN_HALF_DEG))
        assert chord > 2 * half_diag + 0.02, "window chord must admit the carton diagonal"
        # -- the align band sits inside the geometric pass band --
        pass_deg = WIN_HALF_DEG - math.degrees(math.asin(half_diag / WALL_RI))
        assert pass_deg > self.align_deg + 4.0, \
            f"align band {self.align_deg} deg must sit inside the {pass_deg:.1f} deg pass band"
        # -- null policy never aligns: reset offset far outside the align band --
        assert self.off_lo_deg > self.align_deg + 20.0, \
            "reset misalignment must start far outside the align band"
        assert self.off_hi_deg < 179.0, "keep the offset off the 180 deg wrap boundary"
        # -- a CoM-height drag SLIDES the carton; it cannot tip --
        assert self.mu_slide_s * (ITEM_H / 2) < ITEM_S / 2 * 0.6, \
            "carton must slide long before it can tip under a CoM drag"
        # -- embodiment: jaw span, knob pinch, reach heights --
        assert ITEM_S < 0.078, "carton must fit an ~80 mm parallel jaw"
        assert KNOB_S < 0.05, "crank knob must be a comfortable pinch/push target"
        assert AXLE_TOP + KNOB_H + 0.02 < 0.60, "knob must stay at comfortable reach height"
        # -- basket containment bound: accepts every inside rest, rejects rim/outside --
        interior_half = BASKET_S / 2 - RIM_T
        assert self.basket_xy > interior_half - ITEM_S / math.sqrt(2), \
            "in-basket bound must accept a corner-wedged (45 deg yawed) carton"
        assert self.basket_xy < BASKET_S / 2 - RIM_T / 2 - 0.02, \
            "in-basket bound must reject a carton perched on the rim"
        assert self.basket_z[1] < BASKET_T + RIM_H, \
            "in-basket z band must stay below the rim top (rejects rim perches)"
        assert interior_half > ITEM_S * math.sqrt(2) / 2 + 0.03, \
            "basket interior must admit the carton with drop margin"
        # -- extraction latch strictly outside the shroud, on the porch --
        assert self.extract_r > WALL_RI + WALL_T + 0.02, \
            "extraction radius must lie outside the wall"
        assert self.extract_r < PORCH_RO - 0.04, "extraction radius must lie on the porch"
        # -- gap-free disc->porch handover; riding cartons still clear the porch --
        assert PORCH_RI < DISC_R - 0.005, "porch must underlap the disc rim (no gap)"
        assert PORCH_RI > SLOT_R + ITEM_S * math.sqrt(2) / 2 + 0.005, \
            "porch underlap must stay clear of cartons riding their slots"
        # -- the basket never blocks the porch --
        assert self.basket_r - BASKET_S * math.sqrt(2) / 2 > PORCH_RO + 0.02, \
            "basket arc must stay clear of the porch"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("juice_carousel")
class JuiceCarouselScene(BaseScene):
    cfg: JuiceCarouselSceneCfg

    def __init__(self, cfg: JuiceCarouselSceneCfg | None = None) -> None:
        super().__init__(cfg or JuiceCarouselSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        mass = sim_utils.MassPropertiesCfg
        rigid = sim_utils.RigidBodyPropertiesCfg

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
            # NOTE: the base MUST spawn before the turntable (its joint targets it).
            "base": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Base",
                spawn=sp["base"](mass_props=mass(mass=c.base_mass), rigid_props=rigid(),
                                 mu_slide_s=c.mu_slide_s, mu_slide_d=c.mu_slide_d),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "turntable": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Turntable",
                spawn=sp["turntable"](mass_props=mass(mass=c.wheel_mass),
                                      rigid_props=rigid(), wheel_damp=c.wheel_damp,
                                      mu_slide_s=c.mu_slide_s, mu_slide_d=c.mu_slide_d),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, DISC_Z)),
            ),
        }
        item_z = DISC_TOP + ITEM_H / 2 + 0.002
        for name, color, slot in (("juice", (0.95, 0.55, 0.10), 0),
                                  ("milk", (0.93, 0.93, 0.90), 1),
                                  ("soda", (0.15, 0.62, 0.25), 2)):
            a = self.cfg.slot_angles[slot]
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Carton_" + name,
                spawn=sp["carton"](mass_props=mass(mass=c.carton_mass),
                                   rigid_props=rigid(), color=color,
                                   mu_slide_s=c.mu_slide_s, mu_slide_d=c.mu_slide_d),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(SLOT_R * math.cos(a), SLOT_R * math.sin(a), item_z)),
            )
        out["basket"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Basket",
            spawn=sp["basket"](mass_props=mass(mass=c.basket_mass), rigid_props=rigid()),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(c.basket_r, 0.0, 0.0)),
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
        for nm in ("base", "turntable", "juice", "milk", "soda", "basket"):
            setattr(self, nm, env.iscene[nm])
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # post-reset warmup countdown (steps): while > 0, err0 re-latches from
        # readback (absorbing the settle transient) and the score latches stay frozen.
        self.warmup = torch.zeros(n, dtype=torch.long, device=dev)
        self.err0 = torch.full((n,), math.pi / 2, device=dev)  # initial |az error|
        self.err_min = torch.full((n,), math.pi / 2, device=dev)  # running min (latched)
        self.align_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self.extract_latch = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter + free-yaw the base (heavy DYNAMIC teleport — the
        revolute anchor follows), write the turntable at the sampled misalignment
        angle and every carton on its sampled slot IN THE NEW FRAMES (the whole
        linkage in one write — teleporting one body of a joint pair gets
        depenetration-fought), drop the basket on its floor arc, clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # --- base: xy jitter + free yaw (the window heading) ---
        bxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.base_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.base_yaw_max)
        q_b = _qz(yaw)
        zeros = torch.zeros(m, 1, device=dev)
        base_pos = torch.cat([bxy, zeros], dim=-1)
        write(self.base, base_pos, q_b)

        # --- juice slot + misalignment offset (torch.rand draws: the first-randint
        # degeneracy trap) ---
        slot = torch.rand(m, 3, device=dev).argmax(dim=1)  # juice's slot index
        sign = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        off = sign * (math.radians(c.off_lo_deg)
                      + torch.rand(m, device=dev)
                      * math.radians(c.off_hi_deg - c.off_lo_deg))
        slot_ang = torch.tensor(c.slot_angles, device=dev)
        theta0 = off - slot_ang[slot]  # juice azimuth in the base frame = off
        q_t = _qmul(q_b, _qz(theta0))

        # --- turntable exactly on its anchor pose ---
        tt_pos = base_pos + _qapply(q_b, torch.tensor([0.0, 0.0, DISC_Z],
                                                      device=dev).expand(m, 3))
        write(self.turntable, tt_pos, q_t)

        # --- cartons on their slots (juice at `slot`, distractors at the other two) ---
        item_lz = DISC_T / 2 + ITEM_H / 2 + 0.002
        for body, ds in ((self.juice, 0), (self.milk, 1), (self.soda, 2)):
            a = slot_ang[(slot + ds) % 3]
            local = torch.stack([SLOT_R * torch.cos(a), SLOT_R * torch.sin(a),
                                 torch.full((m,), item_lz, device=dev)], dim=-1)
            write(body, tt_pos + _qapply(q_t, local), _qmul(q_t, _qz(a)))

        # --- basket on its floor arc around the window side ---
        b_ang = yaw + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.basket_arc_deg)
        b_rad = c.basket_r + torch.rand(m, device=dev) * c.basket_r_jitter
        b_xy = bxy + torch.stack([b_rad * torch.cos(b_ang),
                                  b_rad * torch.sin(b_ang)], dim=-1)
        write(self.basket, torch.cat([b_xy, zeros], dim=-1),
              _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi))

        # --- clear the score latches; arm the warmup re-latch window ---
        self.err0[env_ids] = off.abs()
        self.err_min[env_ids] = off.abs()
        self.align_latch[env_ids] = False
        self.extract_latch[env_ids] = False
        self.warmup[env_ids] = 60

    # ----- state (full, restorable) -----------------------------------------------------------
    _BODIES = ("base", "turntable", "juice", "milk", "soda", "basket")

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {nm: getattr(self, nm).data.root_state_w[env_ids].clone()
               for nm in self._BODIES}
        for nm in ("warmup", "err0", "err_min", "align_latch", "extract_latch"):
            out[nm] = getattr(self, nm)[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm in self._BODIES:
            getattr(self, nm).write_root_state_to_sim(state[nm], env_ids)
        for nm in ("warmup", "err0", "err_min", "align_latch", "extract_latch"):
            getattr(self, nm)[env_ids] = state[nm]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A carousel cabinet stands on the floor: a gray base slab carrying a "
            f"rotating disc ({DISC_R * 100:.0f} cm radius) under a fixed shroud — a "
            f"ring wall with a roof — whose ONLY opening is one front WINDOW sector "
            f"about {2 * WIN_HALF_DEG:.0f} degrees wide, with a flush landing porch "
            f"outside it. Three cartons ({ITEM_S * 1000:.0f} mm square, "
            f"{ITEM_H * 1000:.0f} mm tall) stand on the disc at three equally spaced "
            f"slots: an ORANGE juice carton (the target), a WHITE milk carton and a "
            f"GREEN soda carton (distractors). The disc spins freely on its center "
            f"bearing, in either direction, without stops; a red KNOB on a crank arm "
            f"above the roof turns with it and is the intended handle — push or "
            f"carry the knob along its circle to rotate the disc (pushing the carton "
            f"visible in the window sideways also works). The roof is too low to "
            f"lift a carton inside the shroud, and the wall blocks any carton that "
            f"is not lined up with the window: a carton can leave ONLY by sliding "
            f"out through the window, low, onto the porch. A TAN walled basket sits "
            f"on the floor nearby. The cabinet's position and heading (so the "
            f"window's direction), which slot holds the orange carton, its initial "
            f"angle away from the window ({c.off_lo_deg:.0f}-{c.off_hi_deg:.0f} "
            f"degrees, either way), and the basket's position all change every "
            f"episode — read the scene by looking.\n"
            f"Goal: rotate the carousel until the ORANGE juice carton is lined up "
            f"with the window (within about {c.align_deg:.0f} degrees), slide it out "
            f"through the window onto the porch, then put it INSIDE the basket "
            f"(fully below the rim) and leave everything at rest. The white and "
            f"green cartons must STAY on the carousel disc — do not pull them out or "
            f"knock them off. Rotating first is not optional: the wall physically "
            f"blocks extraction at any other angle."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Rotate the carousel by its red knob until the ORANGE juice carton lines "
            "up with the front window, slide it out through the window onto the "
            "porch, and place it inside the tan basket on the floor. Leave the white "
            "and green cartons on the carousel and everything at rest."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def base_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> base frame (window along +x)."""
        return _qapply(_qinv(self.base.data.root_quat_w),
                       pos_w - self.base.data.root_pos_w)

    def basket_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        return _qapply(_qinv(self.basket.data.root_quat_w),
                       pos_w - self.basket.data.root_pos_w)

    def juice_polar(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(r, az, z) of the juice carton in the base frame. az in (-pi, pi],
        0 = window center — atan2 wraps it natively."""
        loc = self.base_local(self.juice.data.root_pos_w)
        return loc[:, :2].norm(dim=-1), torch.atan2(loc[:, 1], loc[:, 0]), loc[:, 2]

    def wheel_speed(self) -> torch.Tensor:
        """(N,) carousel |angular velocity| about the vertical."""
        return self.turntable.data.root_ang_vel_w[:, 2].abs()

    def on_disc(self, body) -> torch.Tensor:
        """(N,) bool: body riding the disc (base-frame radius + height band)."""
        c = self.cfg
        loc = self.base_local(body.data.root_pos_w)
        r = loc[:, :2].norm(dim=-1)
        return (r < c.disc_r_max) & (loc[:, 2] > c.disc_z_band[0]) \
            & (loc[:, 2] < c.disc_z_band[1])

    def juice_in_basket(self) -> torch.Tensor:
        """(N,) bool: juice carton contained in the basket, below the rim."""
        c = self.cfg
        loc = self.basket_local(self.juice.data.root_pos_w)
        return (loc[:, 0].abs() < c.basket_xy) & (loc[:, 1].abs() < c.basket_xy) \
            & (loc[:, 2] > c.basket_z[0]) & (loc[:, 2] < c.basket_z[1])

    def settled(self, body) -> torch.Tensor:
        return (body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < 1.0)

    def all_settled(self) -> torch.Tensor:
        return self.settled(self.juice) & self.settled(self.milk) \
            & self.settled(self.soda) & self.settled(self.basket) \
            & (self.wheel_speed() < self.cfg.wheel_still)

    # ----- step-coupled latches ---------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch score stages at sim rate. During the post-reset WARMUP window,
        err0/err_min re-latch from readback (absorbing the settle transient) and the
        latches stay frozen. Alignment tracking is gated on the juice still riding
        the disc (its azimuth is meaningless once it is out)."""
        c = self.cfg
        r, az, _z = self.juice_polar()
        err = az.abs()
        on = self.on_disc(self.juice)
        v = self.juice.data.root_lin_vel_w.norm(dim=-1)

        warm = self.warmup > 0
        if bool(warm.any()):
            relatch = warm & on
            self.err0 = torch.where(relatch, err.clamp(min=0.30), self.err0)
            self.err_min = torch.where(relatch, err, self.err_min)
            self.warmup = self.warmup - warm.long()
        live = ~warm
        self.err_min = torch.where(live & on, torch.minimum(self.err_min, err),
                                   self.err_min)
        self.align_latch |= live & on & (err < math.radians(c.align_deg)) \
            & (self.wheel_speed() < c.wheel_still)
        self.extract_latch |= live & (r > c.extract_r) & (v < c.extract_v) \
            & (self.base_local(self.juice.data.root_pos_w)[:, 2] < 0.30)

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: juice settled inside the basket, both distractors still on the
        disc and settled, carousel at rest."""
        return self.juice_in_basket() & self.on_disc(self.milk) \
            & self.on_disc(self.soda) & self.all_settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.20 x best alignment progress (latched running max
        of normalized azimuth-error reduction) + 0.15 alignment latch + 0.30
        extraction latch; 1.0 iff success(). Latched credit never evaporates under
        correct behavior; null policy ~0 (reset misalignment >= off_lo and the
        carousel holds its angle)."""
        best = ((self.err0 - self.err_min) / self.err0).clamp(0.0, 1.0)
        base = 0.20 * best + 0.15 * self.align_latch.float() \
            + 0.30 * self.extract_latch.float()
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="juice_carousel", robot="null", env_spacing=3.0))
