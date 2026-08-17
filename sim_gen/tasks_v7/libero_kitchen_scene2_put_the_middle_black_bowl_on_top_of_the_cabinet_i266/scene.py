"""ScoopLiftCourtScene — load the MIDDLE black bowl into the swing-lift's scoop and
crank the lift so it pours the bowl through the side window into the roofed court on
top of the cabinet (libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i266).

Derived from libero_90 kitchen_scene2 "put the middle black bowl on top of the
cabinet", where the whole task is one vertical pick-and-place: grasp the middle of
three identical bowls and release it above the cabinet's always-free flat top. Here
the cabinet top is a walled COURT sealed under a full ROOF — there is NO vertical
entry at all. The only way in is a side WINDOW in the court's wall facing a passive
SWING-LIFT that stands beside the cabinet: a heavy pylon base carrying a hinged arm
with an open-top SCOOP pocket on the cabinet side and a LEVER PADDLE on the other.
The agent never lifts the bowl to the top itself: it sets the middle bowl into the
scoop at waist height, then cranks the lever so the arm swings up ~150 deg; past
vertical the pocket mouth tips toward the window and gravity pours the bowl out,
through the window, onto the court seat. The arm is gravity-BISTABLE (its CoM is on
the scoop side), so it rests pressed on the load stop or, once cranked past
vertical, on the dump stop — no one needs to hold it.

The seed's plan — carry the bowl above the cabinet and release — executed here parks
the bowl ON THE ROOF, outside the court: a settled, scored failure (smoke negative).
Court credit is latch-gated on the full mechanism history (loaded in the scoop at
the load stop -> still in the scoop past 60 deg -> still aboard past 95 deg ->
through the window band -> into the court), so a bowl walked in through the window
by hand, or teleported onto the seat, earns nothing.

Mechanism notes:
  - The cabinet + court + staging pad is ONE kinematic compound fixture, re-posed
    per reset (xy jitter + yaw); all predicates are evaluated in the fixture body
    frame, so randomization is real.
  - The lift is a two-body linkage: a DYNAMIC heavy base (a kinematic joint anchor
    would stay world-fixed through reset teleports) and the arm, joined by a
    spawn-authored USD RevoluteJoint (axis y, joint limits are the two stops). The
    whole linkage is teleported coherently at reset; the arm origin sits ON the
    hinge axis so any reset angle is a pure pose write.
  - Arm masses are per-child DENSITY (root mass on a custom spawner is silently
    ignored; densities give the true CoM + inertia the bistability needs). The
    scoop cavity is slick (pair mu with the bowl ~0.24) so the pour is reliable;
    the court seat is grippy so the arriving bowl stops inside.
  - The three bowls are IDENTICAL black octagonal cups; which body lands in which
    row slot on the staging pad is a per-episode random permutation, so "the middle
    bowl" is a fresh identity every episode (readback-verified in smoke).
  - `enable_external_forces_every_iteration` is set: without it, external-wrench
    drives on a jointed body are under-applied across TGS iterations.

Everything is procedural. Heavy imports (isaaclab, pxr) are deferred so importing
this module stays app-free.
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


# ----- geometry helpers (shared by spawners and cfg audit) ----------------------------------------
def _arm_parts(c: Any) -> list[tuple]:
    """The arm's box list: (name, size, center, color_key, density). Local frame:
    origin ON the hinge axis, +x toward the scoop; at hinge angle theta the local
    +x axis points (cos theta, 0, sin theta) in the base frame (theta up-positive).
    Scoop cavity: x in [cav_x0, cav_x0 + cav_w], |y| < cav_w/2, floor top at
    z = cav_floor_z, mouth plane at z = cav_mouth_z. The mouth is OPEN (no lip):
    for theta < 90 deg gravity presses the load onto the floor / hinge-side wall,
    so nothing exits during the swing; past ~104 deg it starts to slide out along
    the hinge-side wall — the pour."""
    wt = c.scoop_wall_t
    cw = c.cav_w
    x0 = c.cav_x0
    xm = x0 + cw / 2  # cavity center x
    zf = c.cav_floor_z
    zm = c.cav_mouth_z
    wall_h = (zm - zf) + wt
    wall_cz = (zm + zf - wt) / 2
    return [
        ("hub", (0.05, 0.10, 0.05), (0.0, 0.0, 0.0), "arm", c.arm_density),
        ("beam", (x0 - wt - 0.03, 0.05, 0.03), ((x0 - wt + 0.03) / 2, 0.0, 0.0),
         "arm", c.arm_density),
        ("scoop_floor", (cw + 2 * wt, cw + 2 * wt, wt), (xm, 0.0, zf - wt / 2),
         "scoop", c.arm_density),
        ("scoop_in", (wt, cw + 2 * wt, wall_h), (x0 - wt / 2, 0.0, wall_cz),
         "scoop", c.arm_density),
        ("scoop_out", (wt, cw + 2 * wt, wall_h), (x0 + cw + wt / 2, 0.0, wall_cz),
         "scoop", c.arm_density),
        ("scoop_yp", (cw, wt, wall_h), (xm, (cw + wt) / 2, wall_cz),
         "scoop", c.arm_density),
        ("scoop_yn", (cw, wt, wall_h), (xm, -(cw + wt) / 2, wall_cz),
         "scoop", c.arm_density),
        ("lever", (c.lever_r - c.paddle_t - 0.04, 0.05, 0.02),
         (-(c.lever_r - c.paddle_t + 0.04) / 2, 0.0, 0.0), "arm", c.arm_density),
        ("paddle", (c.paddle_t, 0.11, 0.05), (-(c.lever_r - c.paddle_t / 2), 0.0, 0.0),
         "paddle", c.arm_density),
    ]


def _arm_dyn(c: Any) -> tuple[float, float, float]:
    """(mass, CoM x, I about the hinge/y axis) from the authored part densities
    (box inertia; all parts are axis-aligned in the arm frame)."""
    m_tot, mx, iy = 0.0, 0.0, 0.0
    for _nm, (sx, _sy, sz), (cx, _cy, cz), _col, rho in _arm_parts(c):
        m = sx * _sy * sz * rho
        m_tot += m
        mx += m * cx
        iy += m * ((sx * sx + sz * sz) / 12.0 + cx * cx + cz * cz)
    return m_tot, mx / m_tot, iy


def _scoop_corners(c: Any) -> list[tuple[float, float]]:
    """(x, z) extreme corners of the scoop exterior in the arm frame (the swept
    envelope that must clear the court wall/roof during the swing)."""
    xin = c.cav_x0 - c.scoop_wall_t
    xout = c.cav_x0 + c.cav_w + c.scoop_wall_t
    zlo = c.cav_floor_z - c.scoop_wall_t
    zhi = c.cav_mouth_z
    return [(xin, zlo), (xin, zhi), (xout, zlo), (xout, zhi)]


# ----- custom compound spawners -------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _author_material(stage, path: str, mu_s: float, mu_d: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu_s))
    api.CreateDynamicFrictionAttr(float(mu_d))
    api.CreateRestitutionAttr(0.0)
    return mat


def _bind_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float | None,
                kinematic: bool = False):
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
    if mass is not None:
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    return root, pxrb


def _box(stage, path, size, center, color, contact_offset, material=None,
         density: float | None = None):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    bxf = UsdGeom.Xformable(cube.GetPrim())
    bxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    bxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if density is not None:
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateDensityAttr(float(density))
    if material is not None:
        _bind_material(cube.GetPrim(), material)


def _spawn_fixture(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One KINEMATIC rigid body: cabinet plinth whose top face IS the court seat
    (z=0.46), court walls z 0.46..0.62 with the WINDOW gap (|y|<0.10, full height)
    in the +x wall, full roof z 0.62..0.64 (no vertical entry), and the staging pad
    (top z=0.10) on the -x side. Body frame: origin on the floor under the plinth,
    window/lift toward +x."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _ = _rigid_root(stage, prim_path, translation, orientation, 90.0,
                          kinematic=True)
    c = cfg
    mat_base = _author_material(stage, f"{prim_path}/mat_base", c.mu, c.mu - 0.05)
    mat_seat = _author_material(stage, f"{prim_path}/mat_seat", c.seat_mu, c.seat_mu - 0.05)
    co = c.contact_offset
    # plinth: its top face (z=0.46) IS the court seat (grippy: arrivals stop inside)
    _box(stage, f"{prim_path}/plinth", (0.32, 0.28, 0.46), (0.0, 0.0, 0.23),
         c.body_color, co, material=mat_seat)
    # court walls (interior |x|<0.14, |y|<0.12, z 0.46..0.62)
    _box(stage, f"{prim_path}/wall_xn", (0.02, 0.28, 0.16), (-0.15, 0.0, 0.54),
         c.wall_color, co, material=mat_base)
    for sgn in (1.0, -1.0):
        _box(stage, f"{prim_path}/wall_y{'p' if sgn > 0 else 'n'}",
             (0.28, 0.02, 0.16), (0.0, sgn * 0.13, 0.54),
             c.wall_color, co, material=mat_base)
        # window pillars: the +x wall is open (the WINDOW) for |y|<0.10, full height
        _box(stage, f"{prim_path}/pillar_{'p' if sgn > 0 else 'n'}",
             (0.02, 0.04, 0.16), (0.15, sgn * 0.12, 0.54),
             c.wall_color, co, material=mat_base)
    # full roof (NO opening)
    _box(stage, f"{prim_path}/roof", (0.32, 0.28, 0.02), (0.0, 0.0, 0.63),
         c.roof_color, co, material=mat_base)
    # staging pad for the bowl row
    _box(stage, f"{prim_path}/staging", (0.26, 0.44, 0.10), (-0.42, 0.0, 0.05),
         c.staging_color, co, material=mat_base)
    return root


def _spawn_lift_base(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The heavy DYNAMIC lift base (joint anchor; a kinematic body0's joint frame
    would stay world-fixed through reset teleports). Local frame: origin at the
    footprint center on the ground, hinge point at (0, 0, hinge_h). Two pylons
    straddle the swing plane (inner faces |y|=0.075; every arm part stays inside
    |y|<0.056, so the arm never touches its own support at any angle)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, 45.0)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    c = cfg
    mat = _author_material(stage, f"{prim_path}/mat", 0.8, 0.75)
    co = c.contact_offset
    _box(stage, f"{prim_path}/foot", (0.30, 0.30, 0.03), (0.0, 0.0, 0.015),
         c.base_color, co, material=mat)
    for sgn in (1.0, -1.0):
        tag = "p" if sgn > 0 else "n"
        _box(stage, f"{prim_path}/pylon_{tag}", (0.10, 0.05, c.hinge_h),
             (0.0, sgn * 0.10, c.hinge_h / 2), c.base_color, co, material=mat)
        _box(stage, f"{prim_path}/stub_{tag}", (0.04, 0.02, 0.04),
             (0.0, sgn * 0.065, c.hinge_h), c.base_color, co, material=mat)
    return root


def _spawn_arm(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The DYNAMIC lift arm: origin ON the hinge axis, per-child DENSITY masses
    (root mass_props on a custom spawner is silently ignored; densities yield the
    true CoM + inertia the gravity-bistability needs), plus the spawn-authored
    REVOLUTE joint to the sibling base (post-play joints are dead). Joint angle
    psi about +y; the up-swing angle used everywhere is theta = -psi. Limits:
    theta in [-load_stop_deg, dump_stop_deg] — the two gravity-held stops."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, None)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(float(cfg.arm_ang_damping))
    c = cfg
    mat_arm = _author_material(stage, f"{prim_path}/mat_arm", 0.6, 0.55)
    mat_scoop = _author_material(stage, f"{prim_path}/mat_scoop",
                                 c.scoop_mu, max(c.scoop_mu - 0.01, 0.01))
    colors = {"arm": c.arm_color, "scoop": c.scoop_color, "paddle": c.paddle_color}
    for name, size, center, ckey, density in _arm_parts(c):
        _box(stage, f"{prim_path}/{name}", size, center, colors[ckey],
             c.contact_offset,
             material=(mat_scoop if ckey == "scoop" else mat_arm), density=density)

    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/LiftBase"])
    j.CreateBody1Rel().SetTargets([prim_path])
    # the arm never needs to touch the base: leave the joint pair's default
    # collision filtering ON (the stops are the joint limits)
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.hinge_h)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(c.dump_stop_deg) - 0.5)
    j.CreateUpperLimitAttr(float(c.load_stop_deg) + 0.5)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: a BLACK bowl — an open octagonal cup (bottom disc + 8 wall
    segments). Body frame: axis = +z (up when upright), origin at mid-height."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.15)
    color = Gf.Vec3f(*cfg.color)
    mat = _author_material(stage, f"{prim_path}/physmat", cfg.mu, cfg.mu - 0.05)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)
        _bind_material(prim, mat)

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
    return root


def _spawn_plate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: the distractor plate (a squat cylinder, near-white)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, 0.10)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.1)
    cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/disc")
    cyl.CreateRadiusAttr(cfg.radius)
    cyl.CreateHeightAttr(cfg.height)
    cyl.CreateExtentAttr([Gf.Vec3f(-cfg.radius, -cfg.radius, -cfg.height / 2),
                          Gf.Vec3f(cfg.radius, cfg.radius, cfg.height / 2)])
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(cyl.GetPrim())
    px.CreateContactOffsetAttr(float(cfg.contact_offset))
    px.CreateRestOffsetAttr(0.0)
    mat = _author_material(stage, f"{prim_path}/physmat", 0.4, 0.35)
    _bind_material(cyl.GetPrim(), mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "fixture" not in _SPAWNER_CACHE:

        @configclass
        class FixtureSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_fixture)
            mu: float = 0.60
            seat_mu: float = 0.75
            body_color: tuple = (0.50, 0.35, 0.22)
            wall_color: tuple = (0.58, 0.42, 0.27)
            roof_color: tuple = (0.25, 0.20, 0.16)
            staging_color: tuple = (0.62, 0.55, 0.40)
            contact_offset: float = 0.003

        @configclass
        class LiftBaseSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lift_base)
            hinge_h: float = 0.36
            base_color: tuple = (0.30, 0.32, 0.38)
            contact_offset: float = 0.002

        @configclass
        class ArmSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_arm)
            hinge_h: float = 0.36
            cav_x0: float = 0.2325
            cav_w: float = 0.085
            cav_floor_z: float = -0.030
            cav_mouth_z: float = 0.045
            scoop_wall_t: float = 0.008
            lever_r: float = 0.20
            paddle_t: float = 0.03
            arm_density: float = 800.0
            arm_ang_damping: float = 0.6
            scoop_mu: float = 0.05
            load_stop_deg: float = 20.0
            dump_stop_deg: float = 130.0
            arm_color: tuple = (0.75, 0.55, 0.15)
            scoop_color: tuple = (0.80, 0.62, 0.20)
            paddle_color: tuple = (0.85, 0.75, 0.10)
            contact_offset: float = 0.002

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            inner_r: float = 0.024
            wall_t: float = 0.007
            height: float = 0.036
            bot_t: float = 0.008
            mass: float = 0.18
            mu: float = 0.45
            color: tuple = (0.07, 0.07, 0.08)
            contact_offset: float = 0.002

        @configclass
        class PlateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plate)
            radius: float = 0.075
            height: float = 0.014
            color: tuple = (0.88, 0.88, 0.86)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(fixture=FixtureSpawnerCfg, lift_base=LiftBaseSpawnerCfg,
                              arm=ArmSpawnerCfg, bowl=BowlSpawnerCfg,
                              plate=PlateSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class ScoopLiftCourtSceneCfg(BaseCfg):
    """Config for `ScoopLiftCourtScene`. All fixture geometry is FIXTURE-FRAME
    (origin on the floor under the plinth; window/lift toward +x, staging pad
    toward -x); the fixture root is re-posed per reset (xy jitter + yaw) and the
    lift linkage rides along (small extra xy jitter of its own), so nothing is
    world-anchored. Scoop-pocket geometry is ARM-FRAME (origin on the hinge axis).

    Landmarks: court seat z=0.46, interior |x|<0.14 |y|<0.12, walls to z=0.62,
    roof z 0.62..0.64 (NO opening); window = +x wall gap |y|<0.10, z 0.46..0.62;
    staging pad top z=0.10 centered x=-0.42; lift hinge at (0.38, 0, 0.36); scoop
    cavity 8.5 cm square, 7.5 cm deep, cavity center radius 0.275 from the hinge;
    load stop theta=-20 deg (mouth up, cavity center ~(0.64, 0, 0.27)), dump stop
    theta=+130 deg (mouth 40 deg below horizontal, just outside the window)."""

    # --- tunable: rubric thresholds ---------------------------------------------------------------
    settle_speed: float = tunable(0.05)   # max |lin vel| (bowls, plate) when judging (m/s)
    arm_settle_rate: float = tunable(15.0)  # max |hinge rate| when judging (deg/s)
    # court seat band (target bowl center, fixture frame):
    seat_x_abs: float = tunable(0.13)
    seat_y_abs: float = tunable(0.095)
    seat_z_lo: float = tunable(0.465)
    seat_z_hi: float = tunable(0.53)
    # court occupancy volume (any bowl / plate center; decoy inside blocks success):
    court_x_abs: float = tunable(0.135)
    court_y_abs: float = tunable(0.115)
    court_z_lo: float = tunable(0.455)
    court_z_hi: float = tunable(0.615)
    # window passage band (fixture frame; the only way into the court):
    win_x_lo: float = tunable(0.13)
    win_x_hi: float = tunable(0.23)
    win_y_abs: float = tunable(0.11)
    win_z_lo: float = tunable(0.45)
    win_z_hi: float = tunable(0.63)
    # scoop-pocket band (ARM frame; bowl center "aboard" the scoop):
    scp_x_lo: float = tunable(0.225)
    scp_x_hi: float = tunable(0.325)
    scp_y_abs: float = tunable(0.050)
    scp_z_lo: float = tunable(-0.040)
    scp_z_hi: float = tunable(0.060)
    # mechanism-history angle gates (deg, theta up-positive):
    load_max_deg: float = tunable(10.0)   # "loaded" latch: aboard with the arm at/near the load stop
    lift_min_deg: float = tunable(60.0)   # "lifted" latch: still aboard past this
    deliver_min_deg: float = tunable(95.0)  # "delivered" latch: still aboard past this

    # --- tunable: randomization -------------------------------------------------------------------
    fix_jitter: float = tunable(0.03)     # fixture root xy jitter (+/- m)
    fix_yaw_deg: float = tunable(8.0)     # fixture root yaw (+/- deg)
    lift_dx: float = tunable(0.006)       # lift-base extra jitter along fixture x (+/- m)
    lift_dy: float = tunable(0.012)       # lift-base extra jitter along fixture y (+/- m)
    row_x: float = tunable(-0.42)         # bowl-row line on the staging pad (fixture x)
    row_x_jitter: float = tunable(0.015)
    row_cy_jitter: float = tunable(0.02)  # bowl-row center (fixture y) jitter
    row_spacing: tuple = tunable((0.11, 0.14))  # slot spacing band
    bowl_jitter: float = tunable(0.008)   # per-bowl xy jitter inside its slot
    plate_pos: tuple = tunable((-0.12, 0.36))   # distractor plate center (fixture xy, floor)
    plate_jitter: float = tunable(0.04)

    # --- info: fixture ----------------------------------------------------------------------------
    fixture_mu: float = info(0.60)
    seat_mu: float = info(0.75)
    seat_z: float = info(0.46)
    roof_top_z: float = info(0.64)
    roof_lo_z: float = info(0.62)
    wall_x_out: float = info(0.16)        # court +x wall outer face (fixture x)
    staging_top_z: float = info(0.10)
    body_color: tuple = info((0.50, 0.35, 0.22))
    wall_color: tuple = info((0.58, 0.42, 0.27))
    roof_color: tuple = info((0.25, 0.20, 0.16))
    staging_color: tuple = info((0.62, 0.55, 0.40))
    contact_offset: float = info(0.003)

    # --- info: lift linkage -----------------------------------------------------------------------
    lift_x: float = info(0.38)            # lift base / hinge (fixture x)
    hinge_h: float = info(0.36)           # hinge height above the ground
    load_stop_deg: float = info(20.0)     # load stop at theta = -20 deg
    dump_stop_deg: float = info(130.0)    # dump stop at theta = +130 deg
    cav_x0: float = info(0.2325)          # scoop cavity inner face (arm x)
    cav_w: float = info(0.085)            # scoop cavity width (square)
    cav_floor_z: float = info(-0.030)     # cavity floor top (arm z)
    cav_mouth_z: float = info(0.045)      # cavity mouth plane (arm z, open — no lip)
    scoop_wall_t: float = info(0.008)
    lever_r: float = info(0.20)           # lever paddle outer radius
    paddle_t: float = info(0.03)
    arm_density: float = info(800.0)
    arm_ang_damping: float = info(0.6)
    scoop_mu: float = info(0.05)
    base_color: tuple = info((0.30, 0.32, 0.38))
    arm_color: tuple = info((0.75, 0.55, 0.15))
    scoop_color: tuple = info((0.80, 0.62, 0.20))
    paddle_color: tuple = info((0.85, 0.75, 0.10))

    # --- info: bowls / plate ----------------------------------------------------------------------
    bowl_inner_r: float = info(0.024)
    bowl_wall_t: float = info(0.007)
    bowl_h: float = info(0.036)
    bowl_bot_t: float = info(0.008)
    bowl_mass: float = info(0.18)
    bowl_mu: float = info(0.45)
    bowl_color: tuple = info((0.07, 0.07, 0.08))
    plate_r: float = info(0.075)
    plate_h: float = info(0.014)
    plate_color: tuple = info((0.88, 0.88, 0.86))

    def __post_init__(self) -> None:
        """Geometry audit: the swing must clear everything it does not intend to
        touch, the pour must fit the window, the bowl must fit the Franka jaw and
        the scoop mouth, and the arm must be gravity-bistable."""
        c = self
        bowl_d = 2 * (c.bowl_inner_r + c.bowl_wall_t)
        assert bowl_d <= 0.070, "bowl must fit a Franka parallel jaw (<= 70 mm)"
        assert c.cav_w >= bowl_d + 0.015, "scoop cavity too tight for the bowl"
        # arm parts stay between the pylons (inner faces |y| = 0.075)
        assert max(abs(cn[1]) + s[1] / 2
                   for _n, s, cn, _c, _d in _arm_parts(c)) <= 0.056 + 1e-9, \
            "arm geometry must stay inside |y| < 0.056 (pylon clearance)"
        # lever side never reaches the court wall plane
        lever_reach = math.hypot(c.lever_r, 0.055) + 0.002
        assert c.lift_x - c.wall_x_out > lever_reach + 0.008, \
            "lever sweep must clear the court wall plane"
        # scoop corners: swept envelope vs wall plane and roof corner, over the
        # full travel (sampled)
        roof_c = (c.wall_x_out, c.roof_top_z)
        for lx, lz in _scoop_corners(c):
            r = math.hypot(lx, lz)
            assert r + 0.010 < math.hypot(c.lift_x - roof_c[0], roof_c[1] - c.hinge_h), \
                "scoop sweep must clear the roof corner"
            for k in range(0, 155, 2):
                th = math.radians(-c.load_stop_deg - 0.5 + k)
                if th > math.radians(c.dump_stop_deg + 0.5):
                    break
                x = c.lift_x + lx * math.cos(th) - lz * math.sin(th)
                z = c.hinge_h + lx * math.sin(th) + lz * math.cos(th)
                assert x > c.wall_x_out + 0.008 or (c.seat_z < z < c.roof_lo_z), \
                    f"scoop corner ({lx:.3f},{lz:.3f}) fouls the court at theta={math.degrees(th):.0f}"
        # scoop fits through the window in y
        assert c.cav_w / 2 + c.scoop_wall_t + 0.03 < 0.10, \
            "scoop must fit the window width with margin"
        # gravity bistability: arm CoM well on the scoop side of the hinge
        m, com_x, iy = _arm_dyn(c)
        assert com_x > 0.04, f"arm CoM must sit on the scoop side (com_x={com_x:.3f})"
        assert 0.5 < m < 3.0, f"arm mass out of range ({m:.2f} kg)"
        # at the dump stop the pour ramp (cavity inner wall) is steeper than the
        # bowl/scoop pair friction angle, so the bowl reliably slides out
        ramp = c.dump_stop_deg - 90.0
        pair_mu = 0.5 * (c.scoop_mu + c.bowl_mu)
        assert math.tan(math.radians(ramp)) > pair_mu + 0.15, \
            "pour ramp at the dump stop must beat pair friction with margin"


# ----- scene ----------------------------------------------------------------------------------------
@SCENES.register("scoop_lift_court")
class ScoopLiftCourtScene(BaseScene):
    cfg: ScoopLiftCourtSceneCfg

    def __init__(self, cfg: ScoopLiftCourtSceneCfg | None = None) -> None:
        super().__init__(cfg or ScoopLiftCourtSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        psi0 = math.radians(c.load_stop_deg) / 2  # arm spawns at the load stop
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
                spawn=sp["fixture"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=90.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mu=c.fixture_mu, seat_mu=c.seat_mu, body_color=c.body_color,
                    wall_color=c.wall_color, roof_color=c.roof_color,
                    staging_color=c.staging_color, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "lift_base": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/LiftBase",
                spawn=sp["lift_base"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=45.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    hinge_h=c.hinge_h, base_color=c.base_color, contact_offset=0.002),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.lift_x, 0.0, 0.0)),
            ),
            "arm": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/LiftArm",
                spawn=sp["arm"](
                    mass_props=sim_utils.MassPropertiesCfg(density=c.arm_density),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    hinge_h=c.hinge_h, cav_x0=c.cav_x0, cav_w=c.cav_w,
                    cav_floor_z=c.cav_floor_z, cav_mouth_z=c.cav_mouth_z,
                    scoop_wall_t=c.scoop_wall_t, lever_r=c.lever_r,
                    paddle_t=c.paddle_t, arm_density=c.arm_density,
                    arm_ang_damping=c.arm_ang_damping, scoop_mu=c.scoop_mu,
                    load_stop_deg=c.load_stop_deg, dump_stop_deg=c.dump_stop_deg,
                    arm_color=c.arm_color, scoop_color=c.scoop_color,
                    paddle_color=c.paddle_color, contact_offset=0.002),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.lift_x, 0.0, c.hinge_h),
                    rot=(math.cos(psi0), 0.0, math.sin(psi0), 0.0)),
            ),
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate",
                spawn=sp["plate"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    radius=c.plate_r, height=c.plate_h, color=c.plate_color,
                    contact_offset=0.002),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.plate_pos[0], c.plate_pos[1], c.plate_h / 2 + 0.002)),
            ),
        }
        for i in range(3):
            out[f"bowl_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl_" + str(i),
                spawn=sp["bowl"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    inner_r=c.bowl_inner_r, wall_t=c.bowl_wall_t, height=c.bowl_h,
                    bot_t=c.bowl_bot_t, mass=c.bowl_mass, mu=c.bowl_mu,
                    color=c.bowl_color, contact_offset=0.002),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.row_x, (i - 1) * 0.125,
                         c.staging_top_z + c.bowl_h / 2 + 0.003)),
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
                # external-wrench plant recipe: without this a torque drive on the
                # jointed arm is under-applied across TGS iterations
                "enable_external_forces_every_iteration": True,
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.fixture: RigidObject = env.iscene["fixture"]
        self.base: RigidObject = env.iscene["lift_base"]
        self.arm: RigidObject = env.iscene["arm"]
        self.plate: RigidObject = env.iscene["plate"]
        self.bowls: list[RigidObject] = [env.iscene[f"bowl_{i}"] for i in range(3)]
        self.env_origins = env.iscene.env_origins
        # which body index is "the middle bowl" this episode (sampled at reset)
        self.target_idx = torch.zeros(n, dtype=torch.long, device=dev)
        # latched mechanism history (post_step), per bowl
        self._loaded_ever = torch.zeros(n, 3, dtype=torch.bool, device=dev)
        self._lifted_ever = torch.zeros(n, 3, dtype=torch.bool, device=dev)
        self._delivered_ever = torch.zeros(n, 3, dtype=torch.bool, device=dev)
        self._windowed_ever = torch.zeros(n, 3, dtype=torch.bool, device=dev)
        self._courted_ever = torch.zeros(n, 3, dtype=torch.bool, device=dev)
        # finite-difference hinge rate (root_ang_vel is phantom under wrenches)
        self._theta_prev = torch.zeros(n, device=dev)
        self.arm_rate = torch.zeros(n, device=dev)  # deg/s
        # reset grace re-pin buffers
        self._grace = torch.zeros(n, dtype=torch.long, device=dev)
        self._body_names = ("fixture", "lift_base", "arm", "plate",
                            "bowl_0", "bowl_1", "bowl_2")
        self._pin_states = {k: torch.zeros(n, 13, device=dev) for k in self._body_names}

    def _bodies(self) -> dict[str, RigidObject]:
        return {"fixture": self.fixture, "lift_base": self.base, "arm": self.arm,
                "plate": self.plate, "bowl_0": self.bowls[0],
                "bowl_1": self.bowls[1], "bowl_2": self.bowls[2]}

    # ----- frames ---------------------------------------------------------------------------------
    def _to_frame(self, body: RigidObject, pos_w: torch.Tensor) -> torch.Tensor:
        """(...,3) world points -> `body`'s frame (broadcast over a bodies dim)."""
        from isaaclab.utils.math import quat_apply_inverse

        q = body.data.root_quat_w
        p = body.data.root_pos_w
        if pos_w.dim() == 3:  # (N, B, 3)
            nb = pos_w.shape[1]
            q = q[:, None, :].expand(-1, nb, -1).reshape(-1, 4)
            return quat_apply_inverse(
                q, (pos_w - p[:, None, :]).reshape(-1, 3)).reshape(pos_w.shape)
        return quat_apply_inverse(q, pos_w - p)

    def _bowl_pos_w(self) -> torch.Tensor:
        return torch.stack([b.data.root_pos_w for b in self.bowls], dim=1)

    def bowl_pos_fix(self) -> torch.Tensor:
        """(N, 3bowls, 3) all bowl centers in the fixture frame."""
        return self._to_frame(self.fixture, self._bowl_pos_w())

    def bowl_pos_arm(self) -> torch.Tensor:
        """(N, 3bowls, 3) all bowl centers in the ARM frame (scoop pocket space)."""
        return self._to_frame(self.arm, self._bowl_pos_w())

    def fix_to_world(self, loc: torch.Tensor) -> torch.Tensor:
        """(N,3) fixture-frame points -> world."""
        from isaaclab.utils.math import quat_apply

        return self.fixture.data.root_pos_w + quat_apply(
            self.fixture.data.root_quat_w, loc)

    def arm_to_world(self, loc: torch.Tensor) -> torch.Tensor:
        """(N,3) arm-frame points -> world."""
        from isaaclab.utils.math import quat_apply

        return self.arm.data.root_pos_w + quat_apply(self.arm.data.root_quat_w, loc)

    def arm_deg(self) -> torch.Tensor:
        """(N,) hinge up-swing angle theta in degrees from the base->arm relative
        quaternion about the hinge (y) axis (theta = -psi; there is no joint-state
        API on a plain spawn-authored USD joint). Load stop -20, dump stop +120."""
        qb = self.base.data.root_quat_w
        qa = self.arm.data.root_quat_w
        qb_inv = qb * torch.tensor([1.0, -1.0, -1.0, -1.0], device=qb.device)
        w = (qb_inv[:, 0] * qa[:, 0] - (qb_inv[:, 1:] * qa[:, 1:]).sum(dim=1))
        # vector part of qb_inv * qa
        v = (qb_inv[:, 0:1] * qa[:, 1:] + qa[:, 0:1] * qb_inv[:, 1:]
             + torch.cross(qb_inv[:, 1:], qa[:, 1:], dim=1))
        psi = torch.rad2deg(2.0 * torch.atan2(v[:, 1], w))
        psi = torch.where(psi > 180.0, psi - 360.0, psi)
        psi = torch.where(psi < -180.0, psi + 360.0, psi)
        return -psi

    def _gather_target(self, per_bowl: torch.Tensor) -> torch.Tensor:
        """(N, 3bowls, ...) -> (N, ...) rows for each env's target bowl."""
        idx = self.target_idx.view(-1, *([1] * (per_bowl.dim() - 1)))
        idx = idx.expand(-1, 1, *per_bowl.shape[2:])
        return per_bowl.gather(1, idx).squeeze(1)

    @staticmethod
    def _in_box(loc, x_lo, x_hi, y_abs, z_lo, z_hi) -> torch.Tensor:
        return ((loc[..., 0] > x_lo) & (loc[..., 0] < x_hi)
                & (loc[..., 1].abs() < y_abs)
                & (loc[..., 2] > z_lo) & (loc[..., 2] < z_hi))

    # ----- predicates -----------------------------------------------------------------------------
    def bowls_in_scoop(self) -> torch.Tensor:
        """(N, 3bowls) bool: bowl center inside the scoop pocket (arm frame)."""
        c = self.cfg
        return self._in_box(self.bowl_pos_arm(), c.scp_x_lo, c.scp_x_hi,
                            c.scp_y_abs, c.scp_z_lo, c.scp_z_hi)

    def bowls_in_court(self) -> torch.Tensor:
        """(N, 3bowls) bool: bowl center inside the court occupancy volume."""
        c = self.cfg
        return self._in_box(self.bowl_pos_fix(), -c.court_x_abs, c.court_x_abs,
                            c.court_y_abs, c.court_z_lo, c.court_z_hi)

    def bowls_in_window(self) -> torch.Tensor:
        """(N, 3bowls) bool: bowl center inside the window passage band."""
        c = self.cfg
        return self._in_box(self.bowl_pos_fix(), c.win_x_lo, c.win_x_hi,
                            c.win_y_abs, c.win_z_lo, c.win_z_hi)

    def target_fix(self) -> torch.Tensor:
        """(N, 3) target bowl center in the fixture frame."""
        return self._gather_target(self.bowl_pos_fix())

    def target_seated(self) -> torch.Tensor:
        """(N,) bool: target bowl inside the seat band (live predicate)."""
        c = self.cfg
        loc = self.target_fix()
        return ((loc[:, 0].abs() < c.seat_x_abs) & (loc[:, 1].abs() < c.seat_y_abs)
                & (loc[:, 2] > c.seat_z_lo) & (loc[:, 2] < c.seat_z_hi))

    def decoy_in_court(self) -> torch.Tensor:
        """(N,) bool: any NON-target bowl inside the court volume."""
        n = self.env.num_envs
        inside = self.bowls_in_court()
        mask = torch.ones(n, 3, dtype=torch.bool, device=self.env.device)
        mask.scatter_(1, self.target_idx.view(-1, 1), False)
        return (inside & mask).any(dim=1)

    def plate_in_court(self) -> torch.Tensor:
        """(N,) bool: plate center inside the court volume."""
        c = self.cfg
        loc = self._to_frame(self.fixture, self.plate.data.root_pos_w)
        return self._in_box(loc, -c.court_x_abs, c.court_x_abs, c.court_y_abs,
                            c.court_z_lo, c.court_z_hi)

    def settled(self) -> torch.Tensor:
        c = self.cfg
        ok = self.plate.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        for b in self.bowls:
            ok &= b.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        ok &= self.arm_rate.abs() < c.arm_settle_rate
        return ok

    # ----- mechanism (every substep) --------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        # reset grace: re-pin freshly reset bodies while write timing settles
        gids = (self._grace > 0).nonzero(as_tuple=False).squeeze(-1)
        if len(gids):
            for name, body in self._bodies().items():
                body.write_root_state_to_sim(self._pin_states[name][gids], gids)
            self._grace[gids] -= 1
            return

        theta = self.arm_deg()
        self.arm_rate = (theta - self._theta_prev) / self.env.dt
        self._theta_prev = theta

        c = self.cfg
        aboard = self.bowls_in_scoop()
        th = theta.unsqueeze(1)
        self._loaded_ever |= aboard & (th < c.load_max_deg)
        self._lifted_ever |= self._loaded_ever & aboard & (th > c.lift_min_deg)
        self._delivered_ever |= self._lifted_ever & aboard & (th > c.deliver_min_deg)
        self._windowed_ever |= self._delivered_ever & self.bowls_in_window()
        self._courted_ever |= self._windowed_ever & self.bowls_in_court()

    # ----- reset ----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Sample the fixture root pose (xy jitter + yaw), pose the WHOLE lift
        linkage coherently relative to it (base + arm at the load stop — the arm
        origin is ON the hinge axis, so the stop angle is a pure pose write), a
        random permutation of the three bowl bodies into the three row slots on
        the staging pad (target = the body in the MIDDLE slot), per-bowl jitter +
        free yaw, and the plate. Everything written with zero velocity; a
        2-substep grace re-pin absorbs write races."""
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

        # lift linkage: base on the ground at (lift_x + dx, dy), arm ON the hinge
        # axis at the load stop; both share the fixture yaw
        dx = (torch.rand(m, device=dev) * 2 - 1) * c.lift_dx
        dy = (torch.rand(m, device=dev) * 2 - 1) * c.lift_dy
        bloc = torch.zeros(m, 3, device=dev)
        bloc[:, 0] = c.lift_x + dx
        bloc[:, 1] = dy
        bst = torch.zeros(m, 13, device=dev)
        bst[:, 0:3] = to_world(bloc)
        bst[:, 3:7] = fq
        self.base.write_root_state_to_sim(bst, env_ids)
        self._pin_states["lift_base"][env_ids] = bst

        ast = bst.clone()
        ast[:, 2] += c.hinge_h  # yaw never moves the on-axis point (0, 0, h)
        # q_arm = fq (yaw about z) composed with R_y(psi_load), psi = +load_stop
        ph = torch.full((m,), math.radians(c.load_stop_deg) / 2, device=dev)
        cp, sp = torch.cos(ph), torch.sin(ph)
        ch, sh = torch.cos(half), torch.sin(half)
        ast[:, 3] = ch * cp
        ast[:, 4] = -sh * sp
        ast[:, 5] = ch * sp
        ast[:, 6] = sh * cp
        self.arm.write_root_state_to_sim(ast, env_ids)
        self._pin_states["arm"][env_ids] = ast

        # body -> slot permutation (torch.rand argsort: healthy across seeds)
        perm = torch.rand(m, 3, device=dev).argsort(dim=1)  # perm[e, slot] = body idx
        self.target_idx[env_ids] = perm[:, 1]

        # row geometry on the staging pad
        row_x = c.row_x + (torch.rand(m, device=dev) * 2 - 1) * c.row_x_jitter
        row_cy = (torch.rand(m, device=dev) * 2 - 1) * c.row_cy_jitter
        spacing = (c.row_spacing[0]
                   + (c.row_spacing[1] - c.row_spacing[0]) * torch.rand(m, device=dev))
        slot_of_body = perm.argsort(dim=1)  # slot_of_body[e, body] = slot idx
        for i, bowl in enumerate(self.bowls):
            slot = slot_of_body[:, i].float() - 1.0  # -1, 0, +1 along the row
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = row_x
            loc[:, 1] = row_cy + slot * spacing
            loc[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.bowl_jitter
            loc[:, 2] = c.staging_top_z + c.bowl_h / 2 + 0.003
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = to_world(loc)
            bhalf = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            st[:, 3] = torch.cos(bhalf)
            st[:, 6] = torch.sin(bhalf)
            bowl.write_root_state_to_sim(st, env_ids)
            self._pin_states[f"bowl_{i}"][env_ids] = st

        # plate distractor on the floor
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.plate_pos[0]
        loc[:, 1] = c.plate_pos[1]
        loc[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.plate_jitter
        loc[:, 2] = c.plate_h / 2 + 0.003
        pst = torch.zeros(m, 13, device=dev)
        pst[:, 0:3] = to_world(loc)
        pst[:, 3] = 1.0
        self.plate.write_root_state_to_sim(pst, env_ids)
        self._pin_states["plate"][env_ids] = pst

        for lat in (self._loaded_ever, self._lifted_ever, self._delivered_ever,
                    self._windowed_ever, self._courted_ever):
            lat[env_ids] = False
        self._theta_prev[env_ids] = -c.load_stop_deg
        self.arm_rate[env_ids] = 0.0
        self._grace[env_ids] = 2

    # ----- state (full, restorable) ---------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {k: b.data.root_state_w[env_ids].clone()
                       for k, b in self._bodies().items()},
            "target_idx": self.target_idx[env_ids].clone(),
            "theta_prev": self._theta_prev[env_ids].clone(),
            "latches": torch.cat(
                [self._loaded_ever[env_ids], self._lifted_ever[env_ids],
                 self._delivered_ever[env_ids], self._windowed_ever[env_ids],
                 self._courted_ever[env_ids]], dim=1).clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for k, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][k], env_ids)
        self.target_idx[env_ids] = state["target_idx"]
        self._theta_prev[env_ids] = state["theta_prev"]
        lat = state["latches"]
        self._loaded_ever[env_ids] = lat[:, 0:3]
        self._lifted_ever[env_ids] = lat[:, 3:6]
        self._delivered_ever[env_ids] = lat[:, 6:9]
        self._windowed_ever[env_ids] = lat[:, 9:12]
        self._courted_ever[env_ids] = lat[:, 12:15]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "A wooden cabinet (32 x 28 cm footprint, 46 cm tall) stands on the "
            "floor. Its top is a walled COURT (interior 28 x 24 cm, walls 16 cm "
            "tall) sealed under a full dark ROOF — there is NO opening from above; "
            "nothing can be lowered in. The only entry is the WINDOW: the court "
            "wall facing the lift (the +x side) is open across its middle 20 cm, "
            "full height, from seat level to the roof. Beside that side of the "
            "cabinet stands a SWING-LIFT: a heavy blue-grey pylon base carrying a "
            "yellow ARM on a horizontal hinge (36 cm up). One end of the arm "
            "carries an open-top SCOOP pocket (8.5 cm square, 7.5 cm deep, slick "
            "inside); the other "
            "end is a LEVER with a bright yellow PADDLE. The arm rests tipped "
            "20 deg below horizontal on its LOAD STOP, scoop on the far side from "
            "the cabinet, mouth up, about 27 cm above the floor. Pushing the "
            "paddle down (or pulling the scoop side up) swings the arm up and "
            "over toward the cabinet; at 130 deg it reaches its DUMP STOP, where "
            "the scoop mouth points 40 deg below horizontal, just outside the "
            "window — anything in the pocket slides out, flies through the window "
            "and lands on the court seat (crank the last quarter of the swing "
            "briskly, in about a quarter second, so the load does not dribble out "
            "early against the wall face). The arm is gravity-bistable: released "
            "before vertical it falls back to the load stop; released past "
            "vertical it settles onto the dump stop by itself. On the tan staging "
            "pad on the cabinet's other side stand three IDENTICAL BLACK BOWLS "
            "(open cups, 6.2 cm across, 3.6 cm tall) in a line; a white plate "
            "lies on the floor as a distractor. The task concerns ONLY the bowl "
            "in the MIDDLE of the line (identify it by its position between the "
            "other two at the start). Goal, in effect: put the middle bowl on "
            "top of the cabinet — that is, at rest on the court seat. Because of "
            "the roof, the ONLY way is the lift: set the middle bowl into the "
            "scoop while the arm sits on its load stop, then crank the lever so "
            "the arm swings up past vertical to the dump stop and pours the bowl "
            "through the window into the court. Court credit is gated on that "
            "full mechanism history — a bowl pushed in through the window by "
            "hand, or placed on the roof, earns nothing. The other two bowls and "
            "the plate must stay out of the court. Everything must be at rest at "
            "the end."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Put the MIDDLE black bowl of the three on top of the cabinet using "
            "the swing-lift: place it into the lift's scoop while the arm rests "
            "on its load stop, then push the lever paddle so the arm swings up "
            "past vertical to its dump stop and pours the bowl through the side "
            "window onto the court seat under the roof. The court is sealed from "
            "above — only the lift can deliver the bowl. Keep the other bowls "
            "and the plate out of the court, and leave everything at rest."
        )

    # ----- rubric ---------------------------------------------------------------------------------
    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched stages of the demonstrated solution (all
        evaluated for the TARGET bowl only): 0.15 loaded (in the scoop at the load
        stop) + 0.15 lifted (still aboard past 60 deg) + 0.20 delivered (still
        aboard past 95 deg) + 0.20 through the window band + 0.25 into the court;
        exactly 1.0 iff success(). Null policy ~0; the seed's plan (release the
        bowl over the cabinet) parks it on the roof and scores 0."""
        s = (0.15 * self._gather_target(self._loaded_ever.float())
             + 0.15 * self._gather_target(self._lifted_ever.float())
             + 0.20 * self._gather_target(self._delivered_ever.float())
             + 0.20 * self._gather_target(self._windowed_ever.float())
             + 0.25 * self._gather_target(self._courted_ever.float()))
        return torch.where(self.success(),
                           torch.ones(self.env.num_envs, device=self.env.device),
                           s.clamp(0.0, 0.95))

    def success(self) -> torch.Tensor:
        """(N,) bool: the MIDDLE bowl at rest in the court seat band, having been
        delivered by the lift (full latched mechanism history: loaded -> lifted ->
        delivered -> window -> court); no decoy bowl and not the plate in the
        court; everything settled."""
        return (self.target_seated()
                & self._gather_target(self._courted_ever)
                & ~self.decoy_in_court() & ~self.plate_in_court()
                & self.settled())


register_env("simgen", lambda: EnvCfg(scene="scoop_lift_court", robot="null"))
