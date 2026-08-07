"""CarouselAirlockScene — feed the black ball through the covered carousel to the green
basin (sim_gen task `libero_kitchen_scene10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_i9`).

Derived from libero_90/libero_kitchen_scene10_put_the_black_bowl_in_the_top_drawer_of_the_
cabinet, but STRATEGICALLY different: the seed is a single pick-and-place — open the top
drawer, put the black bowl inside; its checker is pure position containment of the bowl
inside the drawer compartment's bbox, so the instant the object rests in the fixture's
compartment the task is done. Here, PUTTING THE OBJECT INTO THE FIXTURE'S COMPARTMENT IS
WORTHLESS ON ITS OWN: dropping the BLACK ball through the roof's open LOADING WINDOW into
the covered annular carousel well is only the loading step (0.30 of score at most), and
the rubric explicitly rejects it as an end state (smoke check). The judged outcome lives
on the FAR SIDE of a machine: the solver must then DRIVE the mechanism — turn the YELLOW
crossed spokes counter-clockwise so the rotor's angled ORANGE plow vane sweeps the ball
~180 degrees around the covered annulus to a side DISCHARGE GAP, where the ball is expelled
outward, rolls down a hooded chute, and settles in a GREEN basin. A RED decoy ball of the
same size must stay out of the basin. A solver therefore needs a different PLAN (load
through an aperture, then actuate a rotary mechanism, then let gravity deliver) and a
different code structure (an in-annulus load latch, an azimuthal sweep-progress latch, a
covered-transit latch gated on the sweep, and a basin outcome gated on the physical
pathway) — not a bowl-in-bbox check. Teleporting the ball straight into the basin is
rejected: success() requires the load AND transit latches, which can only be earned by
the ball physically being inside the covered well and then physically crossing the
covered discharge corridor after having travelled around the annulus.

Judged in the HOUSING's body frame (kinematic, xy + free yaw randomized, so the window,
gap, chute, and basin directions must be read from the scene). success() iff:
  - the BLACK ball rests IN THE BASIN (housing frame interior box), settled;
  - load latch: the ball was inside the covered annulus (under the roof, on the disc);
  - transit latch: the ball crossed the covered discharge corridor (gap-side azimuth,
    just outside the ring, under the hood) AND had already swept to near the gap azimuth
    INSIDE the annulus (sweep latch >= 0.75) — pushing the ball backwards up the chute
    from outside earns no transit credit (smoke check);
  - the RED decoy is NOT in the basin.
score() is latched every physics substep: 0.10 * best approach of the ball toward the
loading window (normalized by its own spawn distance) + 0.20 * loaded + 0.30 * best
azimuthal sweep progress inside the annulus (0 at the window, 1 at the gap) + 0.25 *
covered transit, capped at 0.85; exactly 1.0 iff success(). Doing nothing scores ~0.

Assets are fully procedural (no external files):
  - housing (KINEMATIC compound): a round covered well — floor disc (r 155 mm), a guard
    ring of wall segments (inner r 145 mm, up to z 112 mm) with a 60 degree DISCHARGE GAP
    on one side, an annular roof (inner r 50 mm, at z ~106 mm) with a 60 degree open
    LOADING WINDOW on the OPPOSITE side, a hood plate continuing the roof over the gap,
    a 6-degree downhill chute with side walls leading from the gap, and a sunken GREEN
    basin (interior ~170 x 160 mm, 80 mm walls) at the chute's foot.
  - rotor (DYNAMIC compound, 0.9 kg): a disc (r 138 mm) resting on the well floor, an
    ORANGE plow vane angled 38 degrees off-radial (so counter-clockwise rotation wedges
    the ball OUTWARD along the ring), a central axle with a bearing collar riding in the
    roof's central hole (5 mm slop), and two YELLOW crossed spokes (300 mm across) above
    the roof — the handle a robot turns.
  - ball: BLACK sphere r 27.5 mm (55 mm — inside a Franka's 80 mm jaw span).
  - decoy: RED sphere, same size (identity control).
Low-friction physics material on housing and rotor (the rotor must spin under modest
torque; the ball must roll freely); explicit small contact offsets.

Per-episode randomization (verified by readback in smoke): housing xy + free yaw, rotor
free yaw (the vane azimuth a solver must look at before dropping the ball), ball xy,
decoy xy, with batched keep-out resampling so nothing spawns intersecting. Heavy imports
(isaaclab, pxr) are deferred so importing this module — and registering the scene —
stays app-free.
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


def _friction_material(stage, path: str, static: float, dynamic: float):
    """One USD physics material (friction is load-bearing: the rotor must spin on the
    floor under modest torque, and the ball must roll along the vane, not drag)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, contact_offset: float, material=None) -> None:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None, orient=None) -> None:
    """One collidable box child prim (translate -> orient -> scale, authored once —
    idempotent per prim, the duplicate-xformOp trap)."""
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
    _collide(seg.GetPrim(), contact_offset, material)


def _cyl(stage, path: str, radius: float, height: float, center_z: float, color,
         contact_offset: float, material=None) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    seg.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(center_z)))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _rigid_dynamic(root, mass: float) -> None:
    """Dynamic rigid-body armor on a compound root: mass (PhysX derives inertia from the
    child colliders), damping so parts settle promptly, no sleeping (a sleeping body
    silently ignores applied external wrenches — the capstan torque depends on this),
    and the depenetration cap."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.10)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _spawn_housing(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC housing at `prim_path`. Origin = well centre at ground
    level; local +x = the loading window, local -x = the discharge gap -> chute ->
    basin. Ring and roof are overlapping box segments (sealed 24-gons); the chute is a
    pitched box with side walls; the basin is a green open-top tray."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(25.0)

    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    n = cfg.n_seg

    # --- well floor ---
    _cyl(stage, f"{prim_path}/floor", cfg.floor_r, cfg.floor_t, cfg.floor_t / 2,
         cfg.body_color, co, material=mat)

    # --- guard ring: 24 overlapping segments, discharge gap at azimuth 180 deg ---
    r_mid = cfg.ring_inner + cfg.ring_t / 2
    ring_tan = 2 * r_mid * math.sin(math.pi / n) * 1.15
    ring_h = cfg.ring_z1 - cfg.floor_t
    ring_zc = (cfg.ring_z1 + cfg.floor_t) / 2
    for k in range(n):
        az = 2 * math.pi * k / n
        if abs(_wrap_deg(math.degrees(az) - 180.0)) <= cfg.gap_half_deg + 1e-6:
            continue
        q = (math.cos(az / 2), 0.0, 0.0, math.sin(az / 2))
        _box(stage, f"{prim_path}/ring_{k}", (cfg.ring_t, ring_tan, ring_h),
             (r_mid * math.cos(az), r_mid * math.sin(az), ring_zc),
             cfg.body_color, co, material=mat, orient=q)

    # --- roof: 24 overlapping annular segments, loading window at azimuth 0 deg ---
    ro_mid = (cfg.roof_inner + cfg.roof_outer) / 2
    ro_rad = cfg.roof_outer - cfg.roof_inner
    roof_tan = 2 * ro_mid * math.sin(math.pi / n) * 1.25
    for k in range(n):
        az = 2 * math.pi * k / n
        if abs(_wrap_deg(math.degrees(az))) <= cfg.window_half_deg + 1e-6:
            continue
        q = (math.cos(az / 2), 0.0, 0.0, math.sin(az / 2))
        _box(stage, f"{prim_path}/roof_{k}", (ro_rad, roof_tan, cfg.roof_t),
             (ro_mid * math.cos(az), ro_mid * math.sin(az), cfg.roof_zc),
             cfg.roof_color, co, material=mat, orient=q)

    # --- hood: continues the cover over the discharge gap and the chute head ---
    _box(stage, f"{prim_path}/hood", cfg.hood_size, cfg.hood_center,
         cfg.roof_color, co, material=mat)

    # --- chute: pitched 6 deg downhill (local -x end low), with side walls ---
    p = math.radians(cfg.chute_pitch_deg)
    qch = (math.cos(p / 2), 0.0, -math.sin(p / 2), 0.0)  # rotation about +y by -pitch
    xhat = (math.cos(p), 0.0, math.sin(p))
    zhat = (-math.sin(p), 0.0, math.cos(p))
    ccx = cfg.chute_start_x - (cfg.chute_len / 2) * xhat[0] - (cfg.chute_t / 2) * zhat[0]
    ccz = cfg.chute_start_top_z - (cfg.chute_len / 2) * xhat[2] - (cfg.chute_t / 2) * zhat[2]
    _box(stage, f"{prim_path}/chute", (cfg.chute_len, cfg.chute_w, cfg.chute_t),
         (ccx, 0.0, ccz), cfg.chute_color, co, material=mat, orient=qch)
    zoff = cfg.chute_t / 2 + cfg.chute_wall_h / 2 - 0.004
    for tag, sgn in (("wall_yp", 1.0), ("wall_yn", -1.0)):
        _box(stage, f"{prim_path}/chute_{tag}",
             (cfg.chute_len, cfg.chute_wall_t, cfg.chute_wall_h),
             (ccx + zoff * zhat[0], sgn * (cfg.chute_w / 2 + cfg.chute_wall_t / 2),
              ccz + zoff * zhat[2]),
             cfg.chute_color, co, material=mat, orient=qch)

    # --- basin: green open-top tray with an entry opening facing the chute ---
    bx0, bx1 = cfg.basin_x_inner
    wt, wh, yh = cfg.basin_wall_t, cfg.basin_wall_h, cfg.basin_y_half_inner
    _box(stage, f"{prim_path}/basin_floor", cfg.basin_floor_size,
         (cfg.basin_cx, 0.0, 0.0), cfg.basin_color, co, material=mat)
    _box(stage, f"{prim_path}/basin_far", (wt, cfg.basin_floor_size[1], wh),
         (bx0 - wt / 2, 0.0, wh / 2), cfg.basin_color, co, material=mat)
    for tag, sgn in (("yp", 1.0), ("yn", -1.0)):
        _box(stage, f"{prim_path}/basin_{tag}", (bx1 - bx0, wt, wh),
             ((bx0 + bx1) / 2, sgn * (yh + wt / 2), wh / 2),
             cfg.basin_color, co, material=mat)
        stub_w = yh + wt - cfg.basin_open_half
        _box(stage, f"{prim_path}/basin_stub_{tag}", (wt, stub_w, wh),
             (bx1 + wt / 2, sgn * (cfg.basin_open_half + stub_w / 2), wh / 2),
             cfg.basin_color, co, material=mat)
    return root


def _spawn_rotor(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC rotor at `prim_path`. Origin = disc-bottom centre; the disc
    rests on the well floor, the axle rises through the roof's central hole with a
    bearing collar riding inside it, the ORANGE plow vane stands on the disc angled
    38 deg off-radial, and the YELLOW crossed spokes sit above the roof."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass)

    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    _cyl(stage, f"{prim_path}/disc", cfg.disc_r, cfg.disc_t, cfg.disc_t / 2,
         cfg.disc_color, co, material=mat)
    vy = math.radians(cfg.vane_yaw_deg)
    _box(stage, f"{prim_path}/vane", (cfg.vane_len, cfg.vane_t, cfg.vane_h),
         (cfg.vane_cx, 0.0, cfg.vane_zc), cfg.vane_color, co, material=mat,
         orient=(math.cos(vy / 2), 0.0, 0.0, math.sin(vy / 2)))
    _cyl(stage, f"{prim_path}/axle", cfg.axle_r, cfg.axle_h,
         cfg.disc_t + cfg.axle_h / 2, cfg.axle_color, co, material=mat)
    _cyl(stage, f"{prim_path}/collar", cfg.collar_r, cfg.collar_h, cfg.collar_zc,
         cfg.axle_color, co, material=mat)
    _box(stage, f"{prim_path}/spoke_a", (cfg.spoke_len, cfg.spoke_w, cfg.spoke_w),
         (0.0, 0.0, cfg.spoke_z), cfg.spoke_color, co, material=mat)
    _box(stage, f"{prim_path}/spoke_b", (cfg.spoke_len, cfg.spoke_w, cfg.spoke_w),
         (0.0, 0.0, cfg.spoke_z), cfg.spoke_color, co, material=mat,
         orient=(math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)))
    return root


def _housing_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "housing" not in _SPAWNER_CACHE:

        @configclass
        class CarouselHousingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_housing)
            floor_r: float = 0.155
            floor_t: float = 0.020
            ring_inner: float = 0.145
            ring_t: float = 0.012
            ring_z1: float = 0.112
            n_seg: int = 24
            gap_half_deg: float = 30.0
            window_half_deg: float = 30.0
            roof_inner: float = 0.050
            roof_outer: float = 0.157
            roof_t: float = 0.012
            roof_zc: float = 0.106
            hood_size: tuple = (0.105, 0.150, 0.012)
            hood_center: tuple = (-0.197, 0.0, 0.106)
            chute_start_x: float = -0.1455
            chute_start_top_z: float = 0.0295
            chute_len: float = 0.160
            chute_w: float = 0.130
            chute_t: float = 0.012
            chute_pitch_deg: float = 6.0
            chute_wall_t: float = 0.010
            chute_wall_h: float = 0.040
            basin_floor_size: tuple = (0.190, 0.204, 0.020)
            basin_cx: float = -0.405
            basin_x_inner: tuple = (-0.485, -0.315)
            basin_y_half_inner: float = 0.080
            basin_wall_t: float = 0.012
            basin_wall_h: float = 0.080
            basin_open_half: float = 0.066
            mu_static: float = 0.10
            mu_dynamic: float = 0.08
            body_color: tuple = (0.55, 0.55, 0.58)
            roof_color: tuple = (0.42, 0.44, 0.50)
            chute_color: tuple = (0.35, 0.38, 0.42)
            basin_color: tuple = (0.10, 0.60, 0.25)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["housing"] = CarouselHousingSpawnerCfg

    return _SPAWNER_CACHE["housing"](
        mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        floor_r=c.floor_r, floor_t=c.floor_t, ring_inner=c.ring_inner, ring_t=c.ring_t,
        ring_z1=c.ring_z1, n_seg=c.n_seg, gap_half_deg=c.gap_half_deg,
        window_half_deg=c.window_half_deg, roof_inner=c.roof_inner, roof_outer=c.roof_outer,
        roof_t=c.roof_t, roof_zc=c.roof_zc, hood_size=c.hood_size, hood_center=c.hood_center,
        chute_start_x=c.chute_start_x, chute_start_top_z=c.chute_start_top_z,
        chute_len=c.chute_len, chute_w=c.chute_w, chute_t=c.chute_t,
        chute_pitch_deg=c.chute_pitch_deg, chute_wall_t=c.chute_wall_t,
        chute_wall_h=c.chute_wall_h, basin_floor_size=c.basin_floor_size,
        basin_cx=c.basin_cx, basin_x_inner=c.basin_x_inner,
        basin_y_half_inner=c.basin_y_half_inner, basin_wall_t=c.basin_wall_t,
        basin_wall_h=c.basin_wall_h, basin_open_half=c.basin_open_half,
        mu_static=c.mu_static, mu_dynamic=c.mu_dynamic, body_color=c.body_color,
        roof_color=c.roof_color, chute_color=c.chute_color, basin_color=c.basin_color,
        contact_offset=c.contact_offset,
    )


def _rotor_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rotor" not in _SPAWNER_CACHE:

        @configclass
        class CarouselRotorSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rotor)
            disc_r: float = 0.138
            disc_t: float = 0.014
            vane_len: float = 0.100
            vane_t: float = 0.012
            vane_h: float = 0.056
            vane_cx: float = 0.085
            vane_zc: float = 0.042
            vane_yaw_deg: float = -38.0
            axle_r: float = 0.030
            axle_h: float = 0.156
            collar_r: float = 0.045
            collar_h: float = 0.030
            collar_zc: float = 0.0855
            spoke_len: float = 0.300
            spoke_w: float = 0.022
            spoke_z: float = 0.160
            mass: float = 0.9
            mu_static: float = 0.10
            mu_dynamic: float = 0.08
            disc_color: tuple = (0.60, 0.60, 0.65)
            vane_color: tuple = (0.95, 0.55, 0.10)
            axle_color: tuple = (0.25, 0.25, 0.28)
            spoke_color: tuple = (0.95, 0.85, 0.10)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["rotor"] = CarouselRotorSpawnerCfg

    return _SPAWNER_CACHE["rotor"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.rotor_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        disc_r=c.disc_r, disc_t=c.disc_t, vane_len=c.vane_len, vane_t=c.vane_t,
        vane_h=c.vane_h, vane_cx=c.vane_cx, vane_zc=c.vane_zc, vane_yaw_deg=c.vane_yaw_deg,
        axle_r=c.axle_r, axle_h=c.axle_h, collar_r=c.collar_r, collar_h=c.collar_h,
        collar_zc=c.collar_zc, spoke_len=c.spoke_len, spoke_w=c.spoke_w, spoke_z=c.spoke_z,
        mass=c.rotor_mass, mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
        disc_color=c.disc_color, vane_color=c.vane_color, axle_color=c.axle_color,
        spoke_color=c.spoke_color, contact_offset=c.contact_offset,
    )


def _ball_spawner_cfg(c: Any, color: tuple) -> Any:
    import isaaclab.sim as sim_utils

    return sim_utils.SphereCfg(
        radius=c.ball_r,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            linear_damping=0.05, angular_damping=0.30, max_depenetration_velocity=0.5,
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            sleep_threshold=0.0, stabilization_threshold=0.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=c.ball_contact_offset, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=c.ball_mu_static, dynamic_friction=c.ball_mu_dynamic,
            restitution=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CarouselAirlockSceneCfg(BaseCfg):
    """Config for `CarouselAirlockScene`. Honesty knobs asserted in `__post_init__`:
    the ball fits through the window and the gap with real clearance, the vane spans the
    annulus cross-section (the ball cannot slip past it, inboard or outboard), the vane
    clears the ring even at full bearing slop, the covered well leaves the ball
    unreachable except through the window, and the chute genuinely feeds the basin."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    settle_lin: float = tunable(0.05)  # max ball |lin vel| when judging (m/s)
    annulus_z_lo: float = tunable(0.040)  # in-annulus band: ball centre above this (m)
    annulus_z_hi: float = tunable(0.098)  # ... and below the roof underside
    transit_half_deg: float = tunable(30.0)  # transit corridor azimuth half-width at the gap
    sweep_gate: float = tunable(0.75)  # transit latches only after this much sweep progress

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    housing_jitter: float = tunable(0.04)  # uniform +/- xy jitter of the housing at reset (m)
    housing_yaw_deg: float = tunable(180.0)  # uniform +/- housing yaw (free — read the layout)
    rotor_yaw_deg: float = tunable(180.0)  # uniform +/- rotor yaw (free — vane azimuth varies)
    ball_jitter: float = tunable(0.09)  # uniform +/- xy jitter of the ball spawn (m)
    decoy_jitter: float = tunable(0.09)  # uniform +/- xy jitter of the decoy spawn (m)

    # --- tunable: placement ------------------------------------------------------------------
    housing_pos: tuple = tunable((0.0, 0.10))  # well centre, nominal
    ball_pos: tuple = tunable((0.40, -0.34))  # black ball, nominal
    decoy_pos: tuple = tunable((-0.40, -0.36))  # red decoy, nominal

    # --- info: housing structure -------------------------------------------------------------
    floor_r: float = info(0.155)  # well floor disc radius (m)
    floor_t: float = info(0.020)  # floor thickness (floor top = well deck)
    ring_inner: float = info(0.145)  # guard-ring inner radius (24-gon apothem)
    ring_t: float = info(0.012)
    ring_z1: float = info(0.112)  # ring top (= roof top)
    n_seg: int = info(24)
    gap_half_deg: float = info(30.0)  # discharge gap half-angle, at azimuth 180 deg
    window_half_deg: float = info(30.0)  # loading window half-angle, at azimuth 0 deg
    roof_inner: float = info(0.050)  # roof central hole (24-gon apothem) — the bearing
    roof_outer: float = info(0.157)
    roof_t: float = info(0.012)
    roof_zc: float = info(0.106)  # roof plate centre height (underside 0.100)
    hood_size: tuple = info((0.105, 0.150, 0.012))
    hood_center: tuple = info((-0.197, 0.0, 0.106))
    chute_start_x: float = info(-0.1455)  # chute head (top edge) local x
    chute_start_top_z: float = info(0.0295)  # ... and its top-surface height
    chute_len: float = info(0.160)
    chute_w: float = info(0.130)  # chute deck width = interior between walls
    chute_t: float = info(0.012)
    chute_pitch_deg: float = info(6.0)  # downhill toward local -x
    chute_wall_t: float = info(0.010)
    chute_wall_h: float = info(0.040)
    basin_floor_size: tuple = info((0.190, 0.204, 0.020))
    basin_cx: float = info(-0.405)
    basin_x_inner: tuple = info((-0.485, -0.315))  # interior span (far-wall face, stub face)
    basin_y_half_inner: float = info(0.080)
    basin_wall_t: float = info(0.012)
    basin_wall_h: float = info(0.080)
    basin_open_half: float = info(0.066)  # entry opening half-width between the stubs
    # --- info: rotor structure ---------------------------------------------------------------
    disc_r: float = info(0.1375)
    disc_t: float = info(0.014)
    vane_len: float = info(0.100)
    vane_t: float = info(0.012)
    vane_h: float = info(0.056)
    vane_cx: float = info(0.085)  # vane centre along rotor local +x
    vane_zc: float = info(0.042)
    vane_yaw_deg: float = info(-38.0)  # plow angle: CCW spin wedges the ball OUTWARD
    axle_r: float = info(0.030)
    axle_h: float = info(0.156)
    collar_r: float = info(0.045)  # bearing collar riding in the roof's central hole
    collar_h: float = info(0.030)
    collar_zc: float = info(0.0855)
    spoke_len: float = info(0.300)
    spoke_w: float = info(0.022)
    spoke_z: float = info(0.160)  # spokes well above the roof top — the turn handle
    rotor_mass: float = info(0.9)
    # --- info: balls -------------------------------------------------------------------------
    ball_r: float = info(0.0275)  # 55 mm ball — inside a Franka's 80 mm jaw span
    ball_mass: float = info(0.12)
    load_r: float = info(0.095)  # mid-annulus radius at the window (drop/reference point)
    # --- info: friction + contact ------------------------------------------------------------
    mu_static: float = info(0.10)  # housing + rotor: slippery (rotor must spin freely)
    mu_dynamic: float = info(0.08)
    ball_mu_static: float = info(0.40)
    ball_mu_dynamic: float = info(0.35)
    contact_offset: float = info(0.001)  # small explicit offsets: slop budgets are mm-scale
    ball_contact_offset: float = info(0.002)
    # --- info: colors ------------------------------------------------------------------------
    body_color: tuple = info((0.55, 0.55, 0.58))
    roof_color: tuple = info((0.42, 0.44, 0.50))
    chute_color: tuple = info((0.35, 0.38, 0.42))
    basin_color: tuple = info((0.10, 0.60, 0.25))
    disc_color: tuple = info((0.60, 0.60, 0.65))
    vane_color: tuple = info((0.95, 0.55, 0.10))
    axle_color: tuple = info((0.25, 0.25, 0.28))
    spoke_color: tuple = info((0.95, 0.85, 0.10))
    ball_color: tuple = info((0.06, 0.06, 0.07))
    decoy_color: tuple = info((0.85, 0.12, 0.10))
    # --- info: spawn keep-out radii (batched rejection resampling at reset) ------------------
    keepout_housing: float = info(0.26)  # around the well centre (ring + spokes reach)
    keepout_chute: float = info(0.15)  # around the chute midpoint
    keepout_basin: float = info(0.16)  # around the basin centre
    keepout_ball_decoy: float = info(0.12)

    # Derived (filled in __post_init__).
    roof_under: float = field(default=None, init=False)  # roof underside height
    disc_top: float = field(default=None, init=False)  # disc top when rotor rests on floor
    chute_end_top: float = field(default=None, init=False)  # chute top surface at its foot
    basin_floor_top: float = field(default=None, init=False)
    vane_reach: float = field(default=None, init=False)  # farthest vane corner radius
    vane_inner_reach: float = field(default=None, init=False)  # widest inner-corner radius
    bearing_slop: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.roof_under = self.roof_zc - self.roof_t / 2
        self.disc_top = self.floor_t + self.disc_t
        p = math.radians(self.chute_pitch_deg)
        self.chute_end_top = self.chute_start_top_z - self.chute_len * math.sin(p)
        self.basin_floor_top = self.basin_floor_size[2] / 2
        self.bearing_slop = self.roof_inner - self.collar_r
        yaw = math.radians(self.vane_yaw_deg)
        cy, sy = math.cos(yaw), math.sin(yaw)
        outer, inner = [], []
        for st in (1.0, -1.0):
            lx, ly = self.vane_len / 2, st * self.vane_t / 2
            outer.append(math.hypot(self.vane_cx + lx * cy - ly * sy, lx * sy + ly * cy))
            inner.append(math.hypot(self.vane_cx - lx * cy - ly * sy, -lx * sy + ly * cy))
        self.vane_reach = max(outer)
        self.vane_inner_reach = max(inner)

        d = 2 * self.ball_r
        assert 2 * self.ring_inner * math.sin(math.radians(self.gap_half_deg)) >= d + 0.02, (
            "discharge gap must pass the ball with real clearance")
        assert 2 * self.load_r * math.sin(math.radians(self.window_half_deg)) >= d + 0.02, (
            "loading window must pass the ball with real clearance")
        assert self.roof_under - self.disc_top >= d + 0.008, (
            "annulus headroom: the ball must ride the disc under the roof")
        assert self.floor_t + self.vane_zc + self.vane_h / 2 <= self.roof_under - 0.004, (
            "vane top must clear the roof underside")
        assert (self.vane_zc + self.vane_h / 2) - self.disc_t >= self.ball_r + 0.015, (
            "vane must stand tall enough above the disc to push the ball at centre height")
        assert 0.003 <= self.bearing_slop <= 0.008, (
            "bearing slop must be small: the rotor may translate only a few mm")
        assert self.vane_reach + self.bearing_slop + 0.002 <= self.ring_inner, (
            "vane must clear the guard ring even at full bearing slop")
        assert self.ring_inner - self.vane_reach <= d - 0.010, (
            "ball must NOT fit between the vane tip and the ring (no slip-past)")
        assert self.vane_inner_reach - self.axle_r <= d - 0.010, (
            "ball must NOT fit between the vane's inner end and the axle (no slip-past)")
        assert self.disc_r + self.bearing_slop + 0.002 <= self.ring_inner, (
            "disc rim must clear the ring at full bearing slop")
        assert d <= 0.075, "ball must fit a Franka's 80 mm jaw span"
        assert self.chute_end_top >= self.basin_floor_top, (
            "chute foot must feed DOWN into the basin, not dive below its floor")
        assert 2 * self.basin_open_half >= d + 0.02, "basin entry opening passes the ball"
        assert 2 * self.basin_open_half >= self.chute_w - 0.006, (
            "basin entry opening must match the chute interior width")
        assert self.floor_t + self.collar_zc - self.collar_h / 2 <= self.roof_under, (
            "bearing collar must span the roof plate band (below its underside)")
        assert self.floor_t + self.collar_zc + self.collar_h / 2 >= self.roof_zc + self.roof_t / 2, (
            "bearing collar must span the roof plate band (above its top)")
        assert self.floor_t + self.spoke_z - self.spoke_w / 2 >= self.ring_z1 + 0.02, (
            "spokes must ride clear above the roof — the graspable handle")
        assert self.basin_x_inner[1] - self.basin_x_inner[0] >= d + 0.03, (
            "basin interior must hold the ball with margin (x)")
        assert 2 * self.basin_y_half_inner >= d + 0.03, (
            "basin interior must hold the ball with margin (y)")
        assert self.annulus_z_lo < self.disc_top + self.ball_r < self.annulus_z_hi, (
            "in-annulus band must bracket a ball resting on the disc")


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("carousel_airlock")
class CarouselAirlockScene(BaseScene):
    cfg: CarouselAirlockSceneCfg

    def __init__(self, cfg: CarouselAirlockSceneCfg | None = None) -> None:
        super().__init__(cfg or CarouselAirlockSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
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
            "housing": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Housing",
                spawn=_housing_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.housing_pos[0], c.housing_pos[1], 0.0)),
            ),
            "rotor": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rotor",
                spawn=_rotor_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.housing_pos[0], c.housing_pos[1], c.floor_t + 0.002)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=_ball_spawner_cfg(c, c.ball_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.ball_pos[0], c.ball_pos[1], c.ball_r + 0.002)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=_ball_spawner_cfg(c, c.decoy_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.decoy_pos[0], c.decoy_pos[1], c.ball_r + 0.002)),
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
        self.rotor: RigidObject = env.iscene["rotor"]
        self.ball: RigidObject = env.iscene["ball"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.d0 = torch.full((n,), 0.50, device=dev)  # ball-spawn -> window distance
        self.approach_latch = torch.zeros(n, device=dev)
        self.load_latch = torch.zeros(n, device=dev)
        self.sweep_latch = torch.zeros(n, device=dev)
        self.transit_latch = torch.zeros(n, device=dev)

    def _sample_clear(self, m: int, nominal: tuple, jitter: float,
                      keepouts: list[tuple[torch.Tensor, float]]) -> torch.Tensor:
        """(m,2) jittered xy around `nominal`, resampled (12 tries, batched) until
        outside every (centre, radius) keep-out — nothing spawns intersecting."""
        dev = self.env.device
        base = torch.tensor(nominal, device=dev).expand(m, 2)
        xy = base + (torch.rand(m, 2, device=dev) * 2 - 1) * jitter
        for _ in range(12):
            bad = torch.zeros(m, dtype=torch.bool, device=dev)
            for ctr, rad in keepouts:
                bad |= (xy - ctr).norm(dim=-1) < rad
            if not bad.any():
                break
            k = int(bad.sum())
            xy[bad] = base[bad] + (torch.rand(k, 2, device=dev) * 2 - 1) * jitter
        return xy

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: housing with xy jitter + free yaw (window/gap/chute/basin all
        rotate), rotor re-centred in the well with free yaw (the vane azimuth a solver
        must read), ball and decoy on the open floor — keep-out resampled; latches
        zeroed and the approach baseline `d0` captured."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def yaw_state(xy: torch.Tensor, z: float, yaw: torch.Tensor) -> torch.Tensor:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            return st

        # --- housing: xy jitter + free yaw ---
        h_xy = torch.tensor(c.housing_pos, device=dev).expand(m, 2).clone()
        h_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.housing_jitter
        h_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.housing_yaw_deg)
        self.housing.write_root_state_to_sim(yaw_state(h_xy, 0.0, h_yaw), env_ids)

        # --- rotor: re-centred in the well, free yaw ---
        r_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rotor_yaw_deg)
        self.rotor.write_root_state_to_sim(
            yaw_state(h_xy.clone(), c.floor_t + 0.002, r_yaw), env_ids)

        # --- structure keep-out centres (housing frame -> world) ---
        ch, sh = torch.cos(h_yaw), torch.sin(h_yaw)

        def local_xy(lx: float, ly: float) -> torch.Tensor:
            return torch.stack([h_xy[:, 0] + lx * ch - ly * sh,
                                h_xy[:, 1] + lx * sh + ly * ch], dim=-1)

        chute_c = local_xy(c.chute_start_x - c.chute_len / 2, 0.0)
        basin_c = local_xy(c.basin_cx, 0.0)

        # --- ball + decoy on the open floor ---
        ball_xy = self._sample_clear(m, c.ball_pos, c.ball_jitter,
                                     [(h_xy, c.keepout_housing), (chute_c, c.keepout_chute),
                                      (basin_c, c.keepout_basin)])
        zero_yaw = torch.zeros(m, device=dev)
        self.ball.write_root_state_to_sim(
            yaw_state(ball_xy, c.ball_r + 0.002, zero_yaw), env_ids)
        dec_xy = self._sample_clear(m, c.decoy_pos, c.decoy_jitter,
                                    [(h_xy, c.keepout_housing), (chute_c, c.keepout_chute),
                                     (basin_c, c.keepout_basin),
                                     (ball_xy, c.keepout_ball_decoy)])
        self.decoy.write_root_state_to_sim(
            yaw_state(dec_xy, c.ball_r + 0.002, zero_yaw), env_ids)

        # --- baselines + latches ---
        window_c = local_xy(c.load_r, 0.0)
        self.d0[env_ids] = (ball_xy - window_c).norm(dim=-1).clamp(min=0.05)
        self.approach_latch[env_ids] = 0.0
        self.load_latch[env_ids] = 0.0
        self.sweep_latch[env_ids] = 0.0
        self.transit_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "housing": self.housing.data.root_state_w[env_ids].clone(),
            "rotor": self.rotor.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "d0": self.d0[env_ids].clone(),
            "approach_latch": self.approach_latch[env_ids].clone(),
            "load_latch": self.load_latch[env_ids].clone(),
            "sweep_latch": self.sweep_latch[env_ids].clone(),
            "transit_latch": self.transit_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.housing.write_root_state_to_sim(state["housing"], env_ids)
        self.rotor.write_root_state_to_sim(state["rotor"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.d0[env_ids] = state["d0"]
        self.approach_latch[env_ids] = state["approach_latch"]
        self.load_latch[env_ids] = state["load_latch"]
        self.sweep_latch[env_ids] = state["sweep_latch"]
        self.transit_latch[env_ids] = state["transit_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A round covered machine stands on the floor: a gray well "
            f"({2 * c.floor_r * 100:.0f} cm across) whose wall rises to "
            f"{c.ring_z1 * 100:.0f} cm and is capped by an annular roof. The roof has ONE "
            f"open sector — a {2 * c.window_half_deg:.0f} degree LOADING WINDOW — and on "
            f"the exactly OPPOSITE side the wall has a {2 * c.gap_half_deg:.0f} degree "
            f"side gap that leads under a hood onto a gently downhill chute ending in a "
            f"GREEN basin (walls {c.basin_wall_h * 100:.0f} cm). Inside the well a rotor "
            f"rests on the floor: a disc carrying an ORANGE vane angled like a snowplow, "
            f"on a central axle whose two YELLOW crossed spokes "
            f"({c.spoke_len * 100:.0f} cm across) stick out ABOVE the roof — the spokes "
            f"are the only handle. A BLACK ball ({2 * c.ball_r * 1000:.0f} mm) and a RED "
            f"ball of the same size lie on the open floor; the red one is a decoy.\n"
            f"Goal: deliver the BLACK ball into the GREEN basin THROUGH the machine. "
            f"First look through the loading window: if the orange vane is parked under "
            f"the window, turn the spokes to move it aside. Drop the black ball through "
            f"the window onto the disc (about {c.load_r * 100:.0f} cm out from the hub). "
            f"Then turn the YELLOW spokes COUNTER-CLOCKWISE (seen from above): the angled "
            f"vane sweeps the ball around the covered ring to the side gap, expels it "
            f"outward under the hood, and it rolls down the chute into the basin. About "
            f"half a turn to one-and-a-half turns is enough; turning clockwise instead "
            f"wedges the ball against the hub and jams.\n"
            f"Only the full path counts: the ball must have been INSIDE the covered well "
            f"and must have swept around to EXIT through the covered side gap before "
            f"resting in the basin. A ball placed straight into the basin, a ball left "
            f"sitting inside the well (that is only the loading step), a ball parked on "
            f"the roof or beside the basin, or the RED decoy in the basin count for "
            f"nothing. Judged when the black ball rests still inside the basin walls "
            f"and the red decoy is elsewhere."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Drop the black ball through the open window in the machine's roof onto the "
            "disc inside, then turn the yellow crossed spokes counter-clockwise so the "
            "orange vane sweeps the ball around to the side gap, down the chute, and "
            "into the green basin. Leave the red ball out of the basin."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _housing_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> housing body frame (origin = well centre at ground)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.housing.data.root_quat_w,
                                  p_w - self.housing.data.root_pos_w)

    def _housing_world_xy(self, lx: float, ly: float) -> torch.Tensor:
        """(N,2) a housing-frame xy point in world coordinates."""
        q = self.housing.data.root_quat_w
        yaw = 2.0 * torch.atan2(q[:, 3], q[:, 0])
        ch, sh = torch.cos(yaw), torch.sin(yaw)
        return torch.stack([self.housing.data.root_pos_w[:, 0] + lx * ch - ly * sh,
                            self.housing.data.root_pos_w[:, 1] + lx * sh + ly * ch], dim=-1)

    def _yaw_of(self, body) -> torch.Tensor:
        q = body.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    # ----- predicates -------------------------------------------------------------------------
    def in_annulus(self, body) -> torch.Tensor:
        """(N,) bool: `body` centre inside the covered annulus — between hub and ring,
        in the on-the-disc height band under the roof."""
        c = self.cfg
        loc = self._housing_local(body.data.root_pos_w)
        r = loc[:, :2].norm(dim=-1)
        return ((r > c.axle_r + 0.002) & (r < c.disc_r + 0.004)
                & (loc[:, 2] > c.annulus_z_lo) & (loc[:, 2] < c.annulus_z_hi))

    def in_transit(self, body) -> torch.Tensor:
        """(N,) bool: `body` centre inside the covered discharge corridor — gap-side
        azimuth, just outside the ring, laterally within the chute, under the hood."""
        c = self.cfg
        loc = self._housing_local(body.data.root_pos_w)
        r = loc[:, :2].norm(dim=-1)
        az = torch.atan2(loc[:, 1], loc[:, 0])
        gap_err = math.pi - az.abs()  # |wrap(az - pi)|
        return ((gap_err < math.radians(c.transit_half_deg)) & (r > 0.135) & (r < 0.22)
                & (loc[:, 1].abs() < c.basin_open_half)
                & (loc[:, 2] > 0.018) & (loc[:, 2] < 0.105))

    def in_basin(self, body) -> torch.Tensor:
        """(N,) bool: `body` centre inside the basin interior box (housing frame),
        resting at basin-floor height."""
        c = self.cfg
        loc = self._housing_local(body.data.root_pos_w)
        return ((loc[:, 0] > c.basin_x_inner[0] + 0.010)
                & (loc[:, 0] < c.basin_x_inner[1] - 0.010)
                & (loc[:, 1].abs() < c.basin_y_half_inner - 0.010)
                & (loc[:, 2] > c.basin_floor_top + 0.005)
                & (loc[:, 2] < c.basin_floor_top + 0.045))

    def settled(self) -> torch.Tensor:
        """(N,) bool: ball |lin vel| below `settle_lin`."""
        return self.ball.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin

    # ----- graded progress --------------------------------------------------------------------
    def approach_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: ball progress toward the loading window, normalized by the
        episode's own spawn distance."""
        win = self._housing_world_xy(self.cfg.load_r, 0.0)
        d = (self.ball.data.root_pos_w[:, :2] - win).norm(dim=-1)
        return (1.0 - d / self.d0).clamp(0.0, 1.0)

    def sweep_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: azimuthal progress of the ball around the annulus, 0 at the
        loading window, 1 at the discharge gap — gated on actually being INSIDE the
        covered annulus (waving the ball above the roof earns nothing)."""
        loc = self._housing_local(self.ball.data.root_pos_w)
        az = torch.atan2(loc[:, 1], loc[:, 0])
        return (az.abs() / math.pi) * self.in_annulus(self.ball).float()

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch best approach, the load (in-annulus) event, best sweep progress, and
        the covered-transit event each physics substep, so transient progress keeps its
        credit. Transit latches ONLY after the sweep latch shows the ball genuinely
        travelled around the inside to the gap (kills the push-it-up-the-chute and
        fish-it-back-out-the-window shortcuts)."""
        self.approach_latch = torch.maximum(self.approach_latch, self.approach_frac())
        self.load_latch = torch.maximum(self.load_latch, self.in_annulus(self.ball).float())
        self.sweep_latch = torch.maximum(self.sweep_latch, self.sweep_frac())
        transit_now = self.in_transit(self.ball) & (self.sweep_latch >= self.cfg.sweep_gate)
        self.transit_latch = torch.maximum(self.transit_latch, transit_now.float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: black ball settled in the basin, having physically been loaded
        into the covered well AND swept out through the covered discharge; red decoy
        NOT in the basin."""
        return (self.in_basin(self.ball) & self.settled()
                & (self.load_latch > 0.5) & (self.transit_latch > 0.5)
                & ~self.in_basin(self.decoy))

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 * latched approach + 0.20 * loaded + 0.30 * latched
        sweep + 0.25 * covered transit, capped at 0.85; exactly 1.0 iff success().
        Doing nothing scores ~0; the seed's strategy (object into the fixture's
        compartment, done) earns at most the load credit."""
        base = (0.10 * self.approach_latch + 0.20 * self.load_latch
                + 0.30 * self.sweep_latch + 0.25 * self.transit_latch).clamp(0.0, 0.85)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="carousel_airlock", robot="null"))
