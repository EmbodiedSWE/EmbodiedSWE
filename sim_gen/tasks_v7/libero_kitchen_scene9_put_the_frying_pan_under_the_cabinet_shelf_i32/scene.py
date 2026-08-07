"""GearTrainDialScene — complete the broken gear train and crank the caged dial onto its
mark (sim_gen task `libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf_i32`).

Derived from libero_90/libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf,
but STRATEGICALLY different: the seed is a single pick-and-place — grasp the frying pan
and set it inside the bbox of the shelf's bottom region; the instant the object's position
enters the region box the task is done. Here NO object is delivered anywhere at all: the
judged outcome is a MACHINE CONFIGURATION. A three-wheel spur-gear train on a base plate
is missing its middle wheel: the ORANGE driver gear (carrying a tall GREEN crank pin) and
a caged, hand-inaccessible OUTPUT wheel (carrying a RED marker bar) are mounted on end
axles, and the centre axle stands empty. Two loose blue parts lie on the floor: a TOOTHED
idler gear and a SMOOTH disc of the same bore (the decoy — it seats but cannot transmit).
The solver must (1) pick the toothed part, (2) seat it over the centre axle so its teeth
interleave with both neighbours, and (3) turn the green crank so that torque transmitted
THROUGH TOOTH CONTACT rotates the caged output until its red marker points at the MAGENTA
stripe on the cage (within `tol_deg`). A solver therefore needs a different PLAN
(diagnose the gap in a drivetrain, select the transmitting part, assemble, then operate
the mechanism to a dial target) and a different code structure (a seat predicate, a
gated angular-error ACCOUNTING that only credits output rotation produced while the
train is physically complete, and a dial-alignment goal) — not an object-in-region check.

Anti-bypass accounting (the load-bearing interaction is tooth-mesh torque transmission,
and it cannot be teleported past): every physics substep the scene integrates the change
of the marker's angular error. The change is credited ONLY while (a) the idler is seated
on the centre axle, (b) the driver is seated on its axle, and (c) the per-substep output
rotation is physically plausible (`dpsi_max` — a teleported yaw write appears as a
radians-scale jump and is discarded; error INCREASES are always charged). success()
requires the accumulated error account — not just the instantaneous pose — to have been
driven below tolerance, so a marker teleported onto the stripe, or rotated while the
train is incomplete, is rejected (smoke checks), and seating the SMOOTH decoy instead of
the gear leaves the crank spinning uselessly (smoke check: no transmission, no credit).

score(), latched every substep: 0.10 * best approach of the idler toward the centre axle
(normalized by its own spawn distance) + 0.20 * idler seated + 0.45 * best gated
alignment progress ((err0 - err_acc)/(err0 - tol)), capped at 0.85; exactly 1.0 iff
success(). Doing nothing scores ~0 (the initial marker error is sampled >= `err0_lo_deg`,
several times the tolerance).

Assets are fully procedural (no external files):
  - frame (KINEMATIC compound): base plate 0.58 x 0.34 m, three vertical stub axles
    (r 11 mm, 58 mm tall) at local x = -0.15 / 0 / +0.15, and around the output axle a
    cage: a 20-segment ring wall (inner r 98 mm, to z 175 mm) whose sector facing the
    centre axle is a full-height MESHING OPENING (the idler drops in from above and its
    teeth reach through; once seated, the idler itself fills it — and the gated account,
    not the wall, rejects trainless spinning), with a MAGENTA stripe segment at
    output-local azimuth +90 deg, capped by an annular roof (inner r 60 mm) leaving a
    central viewing hole above the marker.
  - driver gear (DYNAMIC): 8-tooth paddle gear (outer r 85 mm) with an octagonal bore
    (apothem 16.5 mm) riding the axle, a hollow riser ring and a radial arm carrying the
    GREEN crank pin (r 8 mm, top at z ~170 mm) — the handle a robot pushes in circles.
  - idler gear (DYNAMIC, blue): the same 8-tooth wheel, bare.
  - output wheel (DYNAMIC): the same 8-tooth wheel carrying a RED marker bar on top.
  - decoy (DYNAMIC, blue): same bore, same-height SMOOTH disc (r 58 mm) — it seats on
    the centre axle but clears both neighbours' teeth by 7 mm: no transmission.
Axle spacing 150 mm vs tooth reach 85 mm -> 20 mm of tooth overlap with ~31 deg of
angular backlash; bore slop 5.5 mm; low-friction material so the train spins under a
hand-scale moment (~0.15 N m).

Per-episode randomization (verified by readback in smoke): frame xy + free yaw (all
directions must be read from the scene), driver yaw, OUTPUT INITIAL yaw (the marker's
starting error: magnitude U[err0_lo, err0_hi] deg, random sign — so the required crank
direction and amount change every episode), idler/decoy floor spots (random side swap +
jitter + free yaw). Heavy imports (isaaclab, pxr) are deferred so importing this module
— and registering the scene — stays app-free.
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
    """One USD physics material (low friction is load-bearing: the train must spin under
    a hand-scale crank moment, and tooth flanks must slide, not bind)."""
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


def _cyl(stage, path: str, radius: float, height: float, center, color,
         contact_offset: float, material=None) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    seg.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _rigid_dynamic(root, mass: float) -> None:
    """Dynamic rigid-body armor on a compound root: mass (PhysX derives inertia from the
    child colliders), damping so wheels stop promptly when the crank torque is cut, no
    sleeping (a sleeping body silently ignores applied external wrenches — the crank
    torque depends on this), and the depenetration cap."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.15)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _hub_ring(stage, prim_path: str, cfg: Any, mat, color, z0: float = 0.0,
              h: float | None = None, tag: str = "hub") -> None:
    """One octagonal bore ring: 8 overlapping wall boxes around a central hole (inner
    apothem `bore_a`) — the bearing every wheel rides an axle on. The hole is genuinely
    open (never capped by a solid) so the axle can stand proud through it."""
    if h is None:
        h = cfg.hub_h
    mid_r = cfg.bore_a + cfg.hub_t / 2
    tan_len = 2.0 * mid_r * math.tan(math.pi / 8) * 1.25
    for k in range(8):
        az = 2 * math.pi * k / 8
        q = (math.cos(az / 2), 0.0, 0.0, math.sin(az / 2))
        _box(stage, f"{prim_path}/{tag}_{k}", (cfg.hub_t, tan_len, h),
             (mid_r * math.cos(az), mid_r * math.sin(az), z0 + h / 2),
             color, cfg.contact_offset, material=mat, orient=q)


def _spawn_gear(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one DYNAMIC 8-tooth paddle gear at `prim_path` (origin = hub bottom
    centre): octagonal bore ring, 8 radial tooth boxes, plus optionally the driver's
    boss + arm + GREEN crank pin, or the output's RED marker bar (local +x)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass)

    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    _hub_ring(stage, prim_path, cfg, mat, cfg.body_color)
    tooth_mid = cfg.hub_r + cfg.tooth_len / 2 - 0.001
    for k in range(8):
        az = 2 * math.pi * k / 8
        q = (math.cos(az / 2), 0.0, 0.0, math.sin(az / 2))
        _box(stage, f"{prim_path}/tooth_{k}", (cfg.tooth_len, cfg.tooth_w, cfg.tooth_h),
             (tooth_mid * math.cos(az), tooth_mid * math.sin(az), cfg.tooth_h / 2 + 0.001),
             cfg.body_color, co, material=mat, orient=q)
    if cfg.with_crank:
        # HOLLOW riser ring (not a solid boss): the axle ends below the crank arm and
        # passes freely through the open bore — no interpenetration, driver seats flush.
        _hub_ring(stage, prim_path, cfg, mat, cfg.body_color,
                  z0=cfg.hub_h, h=cfg.boss_h, tag="riser")
        arm_z = cfg.hub_h + cfg.boss_h + cfg.arm_t / 2
        _box(stage, f"{prim_path}/arm", (cfg.pin_r_pos + 0.012, 0.016, cfg.arm_t),
             ((cfg.pin_r_pos + 0.012) / 2, 0.0, arm_z), cfg.body_color, co, material=mat)
        _cyl(stage, f"{prim_path}/pin", cfg.pin_r, cfg.pin_h,
             (cfg.pin_r_pos, 0.0, arm_z + cfg.arm_t / 2 + cfg.pin_h / 2),
             cfg.pin_color, co, material=mat)
    if cfg.with_marker:
        _box(stage, f"{prim_path}/marker", (cfg.marker_len, 0.016, 0.012),
             (cfg.marker_cx, 0.0, cfg.hub_h + 0.008), cfg.marker_color, co, material=mat)
    return root


def _spawn_decoy(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC smooth decoy disc at `prim_path` (origin = hub bottom centre):
    the SAME octagonal bore as the gears (it seats on an axle) surrounded by 12
    overlapping sector boxes forming a toothless rim (outer r `disc_r`) that clears both
    neighbours' teeth — it can be installed, but it transmits nothing."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass)

    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    _hub_ring(stage, prim_path, cfg, mat, cfg.body_color)
    sec_in = cfg.bore_a + cfg.hub_t - 0.002
    sec_mid = (sec_in + cfg.disc_r) / 2
    sec_len = cfg.disc_r - sec_in
    tan_len = 2.0 * sec_mid * math.tan(math.pi / 12) * 1.3
    for k in range(12):
        az = 2 * math.pi * k / 12
        q = (math.cos(az / 2), 0.0, 0.0, math.sin(az / 2))
        _box(stage, f"{prim_path}/sector_{k}", (sec_len, tan_len, cfg.tooth_h),
             (sec_mid * math.cos(az), sec_mid * math.sin(az), cfg.tooth_h / 2 + 0.001),
             cfg.body_color, cfg.contact_offset, material=mat, orient=q)
    return root


def _spawn_frame(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC frame at `prim_path`. Origin = plate centre at ground level;
    driver axle at local (-spacing, 0), centre (idler) axle at (0, 0), output axle at
    (+spacing, 0) surrounded by the slotted cage with the MAGENTA stripe at output-local
    azimuth +90 deg and the annular viewing roof."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(30.0)

    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset

    _box(stage, f"{prim_path}/plate", cfg.plate_size,
         (0.0, 0.0, cfg.plate_size[2] / 2), cfg.plate_color, co, material=mat)
    top = cfg.plate_size[2]
    for tag, ax in (("axle_drv", -cfg.spacing), ("axle_idl", 0.0), ("axle_out", cfg.spacing)):
        _cyl(stage, f"{prim_path}/{tag}", cfg.axle_r, cfg.axle_h,
             (ax, 0.0, top + cfg.axle_h / 2), cfg.axle_color, co, material=mat)

    # --- cage: 20-segment ring wall around the output axle; the sector facing the
    # centre axle (local azimuth 180 deg) is the full-height MESHING OPENING — the
    # idler drops in from above and its teeth reach through it (once seated, the idler
    # itself occupies the opening; the gated error account, not this wall, is what
    # rejects trainless spinning). The segment at azimuth +90 deg is the MAGENTA
    # stripe — the dial's target mark ---
    n = cfg.cage_n
    mid_r = cfg.cage_inner + cfg.cage_t / 2
    tan_len = 2.0 * mid_r * math.sin(math.pi / n) * 1.2
    stripe_k = round(cfg.stripe_az_deg / (360.0 / n)) % n
    for k in range(n):
        az = 2 * math.pi * k / n
        azd = math.degrees(az)
        if abs(_wrap_deg(azd - 180.0)) <= cfg.gap_half_deg + 1e-6:
            continue  # the meshing opening
        q = (math.cos(az / 2), 0.0, 0.0, math.sin(az / 2))
        color = cfg.stripe_color if k == stripe_k else cfg.cage_color
        _box(stage, f"{prim_path}/cage_{k}", (cfg.cage_t, tan_len, cfg.cage_z_top - top),
             (cfg.spacing + mid_r * math.cos(az), mid_r * math.sin(az),
              (top + cfg.cage_z_top) / 2), color, co, material=mat, orient=q)
    ro_mid = (cfg.roof_inner + cfg.roof_outer) / 2
    ro_rad = cfg.roof_outer - cfg.roof_inner
    ro_tan = 2.0 * ro_mid * math.sin(math.pi / n) * 1.25
    for k in range(n):
        az = 2 * math.pi * k / n
        q = (math.cos(az / 2), 0.0, 0.0, math.sin(az / 2))
        color = cfg.stripe_color if k == stripe_k else cfg.roof_color
        _box(stage, f"{prim_path}/roof_{k}", (ro_rad, ro_tan, cfg.roof_t),
             (cfg.spacing + ro_mid * math.cos(az), ro_mid * math.sin(az),
              cfg.cage_z_top + cfg.roof_t / 2), color, co, material=mat, orient=q)
    return root


def _gear_spawner_cfg(c: Any, *, mass: float, body_color: tuple,
                      with_crank: bool = False, with_marker: bool = False) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "gear" not in _SPAWNER_CACHE:

        @configclass
        class GearSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gear)
            bore_a: float = 0.0165
            hub_t: float = 0.013
            hub_r: float = 0.0295
            hub_h: float = 0.032
            tooth_len: float = 0.056
            tooth_w: float = 0.018
            tooth_h: float = 0.030
            boss_h: float = 0.036
            arm_t: float = 0.010
            pin_r: float = 0.008
            pin_h: float = 0.072
            pin_r_pos: float = 0.055
            marker_len: float = 0.070
            marker_cx: float = 0.053
            with_crank: bool = False
            with_marker: bool = False
            mass: float = 0.25
            mu_static: float = 0.10
            mu_dynamic: float = 0.08
            body_color: tuple = (0.5, 0.5, 0.5)
            pin_color: tuple = (0.10, 0.75, 0.20)
            marker_color: tuple = (0.90, 0.10, 0.08)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["gear"] = GearSpawnerCfg

    return _SPAWNER_CACHE["gear"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        bore_a=c.bore_a, hub_t=c.hub_t, hub_r=c.hub_r, hub_h=c.hub_h,
        tooth_len=c.tooth_len, tooth_w=c.tooth_w, tooth_h=c.tooth_h,
        boss_h=c.boss_h, arm_t=c.arm_t, pin_r=c.pin_r, pin_h=c.pin_h,
        pin_r_pos=c.pin_r_pos, marker_len=c.marker_len, marker_cx=c.marker_cx,
        with_crank=with_crank, with_marker=with_marker, mass=mass,
        mu_static=c.mu_static, mu_dynamic=c.mu_dynamic, body_color=body_color,
        pin_color=c.pin_color, marker_color=c.marker_color,
        contact_offset=c.contact_offset,
    )


def _decoy_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "decoy" not in _SPAWNER_CACHE:

        @configclass
        class DecoySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_decoy)
            bore_a: float = 0.0165
            hub_t: float = 0.013
            hub_h: float = 0.032
            tooth_h: float = 0.030
            disc_r: float = 0.058
            mass: float = 0.20
            mu_static: float = 0.10
            mu_dynamic: float = 0.08
            body_color: tuple = (0.15, 0.35, 0.85)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["decoy"] = DecoySpawnerCfg

    return _SPAWNER_CACHE["decoy"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.decoy_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        bore_a=c.bore_a, hub_t=c.hub_t, hub_h=c.hub_h, tooth_h=c.tooth_h, disc_r=c.disc_r,
        mass=c.decoy_mass, mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
        body_color=c.idler_color, contact_offset=c.contact_offset,
    )


def _frame_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "frame" not in _SPAWNER_CACHE:

        @configclass
        class FrameSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_frame)
            plate_size: tuple = (0.58, 0.34, 0.020)
            spacing: float = 0.15
            axle_r: float = 0.011
            axle_h: float = 0.058
            cage_n: int = 20
            cage_inner: float = 0.098
            cage_t: float = 0.010
            cage_z_top: float = 0.175
            gap_half_deg: float = 40.0
            stripe_az_deg: float = 90.0
            roof_inner: float = 0.060
            roof_outer: float = 0.106
            roof_t: float = 0.010
            mu_static: float = 0.10
            mu_dynamic: float = 0.08
            plate_color: tuple = (0.30, 0.30, 0.33)
            axle_color: tuple = (0.15, 0.15, 0.17)
            cage_color: tuple = (0.55, 0.55, 0.58)
            roof_color: tuple = (0.42, 0.44, 0.50)
            stripe_color: tuple = (0.85, 0.10, 0.75)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["frame"] = FrameSpawnerCfg

    return _SPAWNER_CACHE["frame"](
        mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        plate_size=c.plate_size, spacing=c.spacing, axle_r=c.axle_r, axle_h=c.axle_h,
        cage_n=c.cage_n, cage_inner=c.cage_inner, cage_t=c.cage_t,
        cage_z_top=c.cage_z_top, gap_half_deg=c.gap_half_deg,
        stripe_az_deg=c.stripe_az_deg, roof_inner=c.roof_inner, roof_outer=c.roof_outer,
        roof_t=c.roof_t, mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
        plate_color=c.plate_color, axle_color=c.axle_color, cage_color=c.cage_color,
        roof_color=c.roof_color, stripe_color=c.stripe_color,
        contact_offset=c.contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class GearTrainDialSceneCfg(BaseCfg):
    """Config for `GearTrainDialScene`. Honesty knobs asserted in `__post_init__`: the
    teeth genuinely overlap (mesh exists), hubs clear each other (only teeth touch), the
    decoy genuinely clears both neighbours' teeth (no transmission), the bore slop is a
    real but bounded bearing, the cage slot passes the idler's teeth and nothing else,
    the marker swings clear inside the cage, and the initial dial error dwarfs the
    tolerance (null policy ~0)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    tol_deg: float = tunable(12.0)  # marker-on-stripe alignment tolerance (deg)
    settle_lin: float = tunable(0.05)  # max |lin vel| of every loose body when judging (m/s)
    settle_ang: float = tunable(0.30)  # max |ang vel| of every wheel when judging (rad/s)
    seat_xy_tol: float = tunable(0.010)  # seated: bore centre within this of the axle (m)
    seat_z_lo: float = tunable(-0.005)  # seated: hub bottom band around the plate top (m)
    seat_z_hi: float = tunable(0.014)
    upright_min: float = tunable(0.95)  # seated: body z-axis . world z at least this
    dpsi_max: float = tunable(0.12)  # plausible per-substep output rotation (rad) — a
    # teleported yaw write appears as a radians-scale jump and earns nothing

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    base_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the frame at reset (m)
    base_yaw_deg: float = tunable(180.0)  # uniform +/- frame yaw (free — read the layout)
    drv_yaw_deg: float = tunable(180.0)  # uniform +/- driver initial yaw
    err0_lo_deg: float = tunable(100.0)  # initial marker error magnitude, lower bound
    err0_hi_deg: float = tunable(160.0)  # ... upper bound (sign random per episode)
    part_jitter: float = tunable(0.05)  # uniform +/- xy jitter of idler/decoy spawns
    part_yaw_deg: float = tunable(180.0)  # uniform +/- idler/decoy initial yaw
    spawn_y: float = tunable(0.32)  # |frame-local y| of the loose-part floor spots
    side_swap: bool = tunable(True)  # idler/decoy sides swap randomly per episode

    # --- tunable: placement ------------------------------------------------------------------
    base_pos: tuple = tunable((0.0, 0.0))  # frame centre, nominal

    # --- info: frame structure ---------------------------------------------------------------
    plate_size: tuple = info((0.58, 0.34, 0.020))
    spacing: float = info(0.15)  # axle pitch: driver -x, idler centre, output +x
    axle_r: float = info(0.011)
    axle_h: float = info(0.058)  # above the plate top; ends BELOW the driver's crank arm
    cage_n: int = info(20)
    cage_inner: float = info(0.098)
    cage_t: float = info(0.010)
    cage_z_top: float = info(0.175)
    gap_half_deg: float = info(40.0)  # meshing-OPENING half-angle (full height, faces
    # the idler; occupied by the idler once seated — the account guards trainless spin)
    stripe_az_deg: float = info(90.0)  # MAGENTA stripe, output-local azimuth (deg)
    roof_inner: float = info(0.060)  # viewing hole above the marker
    roof_outer: float = info(0.106)
    roof_t: float = info(0.010)
    # --- info: wheel structure ---------------------------------------------------------------
    bore_a: float = info(0.0165)  # octagonal bore inner apothem (bearing slop = bore_a - axle_r)
    hub_t: float = info(0.013)
    hub_r: float = info(0.0295)  # hub outer radius
    hub_h: float = info(0.032)
    tooth_len: float = info(0.056)  # radial; outer reach = hub_r + tooth_len
    tooth_w: float = info(0.018)
    tooth_h: float = info(0.030)
    boss_h: float = info(0.036)  # driver: hollow riser ring above the hub, under the arm
    arm_t: float = info(0.010)
    pin_r: float = info(0.008)  # GREEN crank pin (16 mm dia — inside a Franka's jaw)
    pin_h: float = info(0.072)
    pin_r_pos: float = info(0.046)  # crank radius (arm tip clears a dropping idler)
    marker_len: float = info(0.070)  # RED marker bar on the output, along local +x
    marker_cx: float = info(0.053)
    disc_r: float = info(0.058)  # decoy smooth-disc outer radius
    driver_mass: float = info(0.30)
    idler_mass: float = info(0.25)
    output_mass: float = info(0.28)
    decoy_mass: float = info(0.20)
    # --- info: friction + contact ------------------------------------------------------------
    mu_static: float = info(0.10)
    mu_dynamic: float = info(0.08)
    contact_offset: float = info(0.001)
    # --- info: colors ------------------------------------------------------------------------
    plate_color: tuple = info((0.30, 0.30, 0.33))
    axle_color: tuple = info((0.15, 0.15, 0.17))
    cage_color: tuple = info((0.55, 0.55, 0.58))
    roof_color: tuple = info((0.42, 0.44, 0.50))
    stripe_color: tuple = info((0.85, 0.10, 0.75))
    driver_color: tuple = info((0.90, 0.50, 0.12))
    idler_color: tuple = info((0.15, 0.35, 0.85))
    output_color: tuple = info((0.80, 0.80, 0.82))
    pin_color: tuple = info((0.10, 0.75, 0.20))
    marker_color: tuple = info((0.90, 0.10, 0.08))

    # Derived (filled in __post_init__).
    plate_top: float = field(default=None, init=False)
    gear_r: float = field(default=None, init=False)  # tooth outer reach
    mesh_overlap: float = field(default=None, init=False)
    bore_slop: float = field(default=None, init=False)
    tol: float = field(default=None, init=False)  # tol_deg in rad
    stripe_az: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.plate_top = self.plate_size[2]
        self.gear_r = self.hub_r + self.tooth_len
        self.mesh_overlap = 2 * self.gear_r - self.spacing
        self.bore_slop = self.bore_a - self.axle_r
        self.tol = math.radians(self.tol_deg)
        self.stripe_az = math.radians(self.stripe_az_deg)

        assert 0.015 <= self.mesh_overlap <= self.tooth_len - 0.010, (
            "teeth must genuinely overlap (mesh exists) without hubs colliding")
        assert self.spacing >= self.gear_r + self.hub_r + 0.015, (
            "a neighbour's teeth must clear the other wheel's hub")
        assert self.disc_r + self.gear_r <= self.spacing - 0.005, (
            "the smooth decoy must CLEAR both neighbours' teeth — no transmission")
        assert 0.003 <= self.bore_slop <= 0.008, (
            "bore slop must be a real but bounded bearing clearance")
        assert self.axle_h >= self.hub_h + 0.020, (
            "the axle must stand proud of a seated hub (wheel retained)")
        assert self.hub_h + self.boss_h >= self.axle_h + 0.008, (
            "the axle must end below the driver's crank arm (the riser is hollow — the "
            "axle passes through it, never through the arm)")
        # cage: the idler's teeth pass through the slot sector, and only through it
        d, rr, rho = self.spacing, self.cage_inner + self.cage_t / 2, self.gear_r + 0.003
        cos_th = (d * d + rr * rr - rho * rho) / (2 * d * rr)
        assert math.degrees(math.acos(max(-1.0, min(1.0, cos_th)))) + 3.0 \
            <= self.gap_half_deg, (
            "the meshing opening must pass the idler's full tooth sweep at every height "
            "(the idler is installed by dropping it in from above)")
        assert self.spacing - (self.pin_r_pos + 0.012) >= self.gear_r + 0.004, (
            "the driver's crank arm tip must clear a dropping idler's tooth sweep")
        assert self.marker_cx + self.marker_len / 2 <= self.cage_inner - 0.008, (
            "marker bar must swing clear inside the cage")
        assert self.marker_cx - self.marker_len / 2 >= self.axle_r + 0.005, (
            "marker bar must clear the axle")
        assert self.roof_inner >= 0.05, "viewing hole must keep the marker readable"
        assert self.hub_h + self.boss_h - 0.002 >= self.plate_top + self.tooth_h + 0.008 \
            - self.plate_top, "crank arm must ride above a seated idler's teeth"
        assert math.radians(self.err0_lo_deg) >= 5 * self.tol, (
            "initial dial error must dwarf the tolerance (null policy ~0)")
        assert self.spawn_y - self.part_jitter - self.gear_r >= self.plate_size[1] / 2, (
            "loose parts must spawn on the floor, off the plate")
        assert 2 * self.pin_r <= 0.075, "crank pin must fit a Franka's 80 mm jaw span"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("gear_train_dial")
class GearTrainDialScene(BaseScene):
    cfg: GearTrainDialSceneCfg

    def __init__(self, cfg: GearTrainDialSceneCfg | None = None) -> None:
        super().__init__(cfg or GearTrainDialSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z_seat = c.plate_top + 0.001
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
            "frame": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Frame",
                spawn=_frame_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.base_pos[0], c.base_pos[1], 0.0)),
            ),
            "driver": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Driver",
                spawn=_gear_spawner_cfg(c, mass=c.driver_mass, body_color=c.driver_color,
                                        with_crank=True),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.base_pos[0] - c.spacing, c.base_pos[1], z_seat)),
            ),
            "output": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Output",
                spawn=_gear_spawner_cfg(c, mass=c.output_mass, body_color=c.output_color,
                                        with_marker=True),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.base_pos[0] + c.spacing, c.base_pos[1], z_seat)),
            ),
            "idler": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Idler",
                spawn=_gear_spawner_cfg(c, mass=c.idler_mass, body_color=c.idler_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.base_pos[0], c.base_pos[1] + c.spawn_y, 0.001)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=_decoy_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.base_pos[0], c.base_pos[1] - c.spawn_y, 0.001)),
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
        self.frame: RigidObject = env.iscene["frame"]
        self.driver: RigidObject = env.iscene["driver"]
        self.output: RigidObject = env.iscene["output"]
        self.idler: RigidObject = env.iscene["idler"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.d0 = torch.full((n,), 0.40, device=dev)  # idler spawn -> centre axle distance
        self.err0 = torch.full((n,), 2.0, device=dev)  # initial marker error (rad)
        self.err_acc = torch.full((n,), 2.0, device=dev)  # gated error account (rad)
        self.approach_latch = torch.zeros(n, device=dev)
        self.seat_latch = torch.zeros(n, device=dev)
        self.align_latch = torch.zeros(n, device=dev)
        self.prev_yaw_out = torch.zeros(n, device=dev)
        self.prev_err = torch.full((n,), 2.0, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: frame with xy jitter + free yaw; driver seated with free yaw;
        output seated with its marker error sampled U[err0_lo, err0_hi] deg, random
        sign (the crank direction/amount a solver must derive from the scene); idler and
        decoy on random floor sides with jitter + free yaw; account + latches rebased."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, xy: torch.Tensor, z: float, yaw: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- frame: xy jitter + free yaw ---
        b_xy = torch.tensor(c.base_pos, device=dev).expand(m, 2).clone()
        b_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.base_jitter
        b_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.base_yaw_deg)
        write(self.frame, b_xy, 0.0, b_yaw)
        cb, sb = torch.cos(b_yaw), torch.sin(b_yaw)

        def local_xy(lx: torch.Tensor, ly: torch.Tensor) -> torch.Tensor:
            return torch.stack([b_xy[:, 0] + lx * cb - ly * sb,
                                b_xy[:, 1] + lx * sb + ly * cb], dim=-1)

        zero = torch.zeros(m, device=dev)
        z_seat = c.plate_top + 0.001

        # --- driver: seated on its axle, free yaw ---
        d_yaw = b_yaw + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.drv_yaw_deg)
        write(self.driver, local_xy(zero - c.spacing, zero), z_seat, d_yaw)

        # --- output: seated, marker error sampled (magnitude + sign) ---
        lo, hi = math.radians(c.err0_lo_deg), math.radians(c.err0_hi_deg)
        mag = lo + torch.rand(m, device=dev) * (hi - lo)
        sign = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        rel0 = c.stripe_az + sign * mag
        o_yaw = b_yaw + rel0
        write(self.output, local_xy(zero + c.spacing, zero), z_seat, o_yaw)

        # --- idler + decoy: random floor sides, jitter + free yaw ---
        if c.side_swap:
            side = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        else:
            side = torch.ones(m, device=dev)
        yaw_amp = math.radians(c.part_yaw_deg)
        i_off = (torch.rand(m, 2, device=dev) * 2 - 1) * c.part_jitter
        i_xy = local_xy(i_off[:, 0], side * c.spawn_y + i_off[:, 1])
        write(self.idler, i_xy, 0.001,
              (torch.rand(m, device=dev) * 2 - 1) * yaw_amp)
        d_off = (torch.rand(m, 2, device=dev) * 2 - 1) * c.part_jitter
        write(self.decoy, local_xy(d_off[:, 0], -side * c.spawn_y + d_off[:, 1]), 0.001,
              (torch.rand(m, device=dev) * 2 - 1) * yaw_amp)

        # --- baselines + account + latches ---
        axle_xy = local_xy(zero, zero)
        self.d0[env_ids] = (i_xy - axle_xy).norm(dim=-1).clamp(min=0.05)
        self.err0[env_ids] = mag
        self.err_acc[env_ids] = mag
        self.approach_latch[env_ids] = 0.0
        self.seat_latch[env_ids] = 0.0
        self.align_latch[env_ids] = 0.0
        self.prev_yaw_out[env_ids] = o_yaw
        self.prev_err[env_ids] = mag

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "frame": self.frame.data.root_state_w[env_ids].clone(),
            "driver": self.driver.data.root_state_w[env_ids].clone(),
            "output": self.output.data.root_state_w[env_ids].clone(),
            "idler": self.idler.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "d0": self.d0[env_ids].clone(),
            "err0": self.err0[env_ids].clone(),
            "err_acc": self.err_acc[env_ids].clone(),
            "approach_latch": self.approach_latch[env_ids].clone(),
            "seat_latch": self.seat_latch[env_ids].clone(),
            "align_latch": self.align_latch[env_ids].clone(),
            "prev_yaw_out": self.prev_yaw_out[env_ids].clone(),
            "prev_err": self.prev_err[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.frame.write_root_state_to_sim(state["frame"], env_ids)
        self.driver.write_root_state_to_sim(state["driver"], env_ids)
        self.output.write_root_state_to_sim(state["output"], env_ids)
        self.idler.write_root_state_to_sim(state["idler"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        for k in ("d0", "err0", "err_acc", "approach_latch", "seat_latch",
                  "align_latch", "prev_yaw_out", "prev_err"):
            getattr(self, k)[env_ids] = state[k]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A dark base plate ({c.plate_size[0] * 100:.0f} x "
            f"{c.plate_size[1] * 100:.0f} cm) lies on the floor carrying a broken "
            f"three-wheel gear train: three vertical stub axles in a row, "
            f"{c.spacing * 100:.0f} cm apart. On one end axle sits an ORANGE 8-tooth "
            f"gear topped by a tall GREEN crank pin (top ~{17:.0f} cm up) — the drive "
            f"handle. On the other end axle, enclosed in a round slotted CAGE (open on "
            f"top, {c.cage_z_top * 100:.0f} cm tall), sits a pale output wheel carrying "
            f"a RED marker bar; one full-height cage segment and the roof piece above it "
            f"are MAGENTA — the target mark. The MIDDLE axle is EMPTY: the two end "
            f"wheels are {c.spacing * 2 * 100:.0f} cm apart and cannot touch each other. "
            f"On the floor beside the plate lie two loose BLUE parts with identical "
            f"centre holes: a TOOTHED gear (8 radial teeth, like the mounted wheels) and "
            f"a SMOOTH toothless disc — the disc is a decoy that cannot transmit "
            f"rotation.\n"
            f"Goal: repair and operate the machine. Set the blue TOOTHED gear down over "
            f"the empty middle axle (drop it on; if it rests on the neighbouring teeth, "
            f"wiggle it slightly until it drops flush onto the plate and its teeth "
            f"interleave with BOTH end wheels). Then turn the GREEN crank pin in circles "
            f"— either direction works; pick the shorter way — so the gear train rotates "
            f"the caged wheel until its RED marker bar points at the MAGENTA stripe, "
            f"within {c.tol_deg:.0f} degrees, and leave it there at rest with the train "
            f"still assembled.\n"
            f"Only rotation delivered THROUGH the assembled train counts: the red marker "
            f"only earns credit while the toothed gear is seated on the middle axle, so "
            f"spinning the crank before the gear is installed, installing the smooth "
            f"disc instead, or nudging the marker any other way achieves nothing. The "
            f"output wheel sits enclosed in its cage (the only wall opening faces the "
            f"middle axle and is filled by the toothed gear once installed; the roof "
            f"leaves just a viewing hole over the marker)."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Seat the blue toothed gear (not the smooth disc) on the empty middle axle "
            "so it meshes with both mounted wheels, then turn the green crank pin to "
            "drive the train until the red marker bar on the caged wheel points at the "
            "magenta stripe and rests there. The marker only counts if the assembled "
            "gear train moved it."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _yaw_of(self, body) -> torch.Tensor:
        q = body.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    def _upright(self, body) -> torch.Tensor:
        """(N,) body z-axis . world z (1 = flat)."""
        q = body.data.root_quat_w
        return 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)

    def _frame_world_xy(self, lx: float, ly: float) -> torch.Tensor:
        yaw = self._yaw_of(self.frame)
        cb, sb = torch.cos(yaw), torch.sin(yaw)
        return torch.stack([self.frame.data.root_pos_w[:, 0] + lx * cb - ly * sb,
                            self.frame.data.root_pos_w[:, 1] + lx * sb + ly * cb], dim=-1)

    @staticmethod
    def _wrap(a: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(a), torch.cos(a))

    # ----- predicates -------------------------------------------------------------------------
    def seated(self, body, axle_lx: float) -> torch.Tensor:
        """(N,) bool: `body`'s bore rides the axle at frame-local (axle_lx, 0) — centre
        within `seat_xy_tol` of the axle, hub bottom in the plate-top band, flat."""
        c = self.cfg
        axle = self._frame_world_xy(axle_lx, 0.0)
        near = (body.data.root_pos_w[:, :2] - axle).norm(dim=-1) < c.seat_xy_tol
        dz = (body.data.root_pos_w - self.env_origins)[:, 2] - c.plate_top
        z_ok = (dz > c.seat_z_lo) & (dz < c.seat_z_hi)
        return near & z_ok & (self._upright(body) > c.upright_min)

    def idler_seated(self) -> torch.Tensor:
        return self.seated(self.idler, 0.0)

    def driver_seated(self) -> torch.Tensor:
        return self.seated(self.driver, -self.cfg.spacing)

    def decoy_seated(self) -> torch.Tensor:
        return self.seated(self.decoy, 0.0)

    def marker_err(self) -> torch.Tensor:
        """(N,) |wrapped angle| between the RED marker direction (output local +x) and
        the MAGENTA stripe azimuth, in the frame's body frame (rad)."""
        rel = self._wrap(self._yaw_of(self.output) - self._yaw_of(self.frame))
        return self._wrap(rel - self.cfg.stripe_az).abs()

    def settled(self) -> torch.Tensor:
        """(N,) bool: every loose body slow, every wheel's spin below `settle_ang`."""
        c = self.cfg
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for b in (self.driver, self.idler, self.output, self.decoy):
            ok &= b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        for b in (self.driver, self.idler, self.output):
            ok &= b.data.root_ang_vel_w[:, 2].abs() < c.settle_ang
        return ok

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch idler approach + seat credit, and run the gated error account: per
        substep, the marker-error change is credited only while the TRAIN IS COMPLETE
        (idler AND driver seated) and the output's rotation is physically plausible
        (|dpsi| < dpsi_max — a teleported yaw write is a radians-scale jump and is
        discarded); error increases are always charged. Alignment therefore cannot be
        teleported, and rotation without the train earns nothing."""
        c = self.cfg
        axle = self._frame_world_xy(0.0, 0.0)
        d = (self.idler.data.root_pos_w[:, :2] - axle).norm(dim=-1)
        self.approach_latch = torch.maximum(
            self.approach_latch, (1.0 - d / self.d0).clamp(0.0, 1.0))
        seated_i = self.idler_seated()
        self.seat_latch = torch.maximum(self.seat_latch, seated_i.float())

        yaw_out = self._yaw_of(self.output)
        err = self.marker_err()
        d_psi = self._wrap(yaw_out - self.prev_yaw_out).abs()
        d_err = err - self.prev_err
        gate = (seated_i & self.driver_seated()
                & (d_psi < c.dpsi_max) & (d_err.abs() < c.dpsi_max))
        self.err_acc = (self.err_acc
                        + torch.where(gate, d_err, d_err.clamp(min=0.0))).clamp(0.0, math.pi)
        denom = (self.err0 - c.tol).clamp(min=1e-6)
        self.align_latch = torch.maximum(
            self.align_latch, ((self.err0 - self.err_acc) / denom).clamp(0.0, 1.0))
        self.prev_yaw_out = yaw_out
        self.prev_err = err

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: train complete (idler AND driver seated), RED marker within `tol`
        of the MAGENTA stripe, the gated error ACCOUNT also below `tol` (the alignment
        was produced through the physical train — not written, not trainless), and
        everything at rest."""
        c = self.cfg
        return (self.idler_seated() & self.driver_seated()
                & (self.marker_err() < c.tol) & (self.err_acc < c.tol) & self.settled())

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 * latched idler approach + 0.20 * seated +
        0.45 * latched gated alignment progress, capped at 0.85; exactly 1.0 iff
        success(). Doing nothing scores ~0; the seed's strategy (carry the loose part
        somewhere and set it down) earns at most a sliver of approach credit."""
        base = (0.10 * self.approach_latch + 0.20 * self.seat_latch
                + 0.45 * self.align_latch).clamp(0.0, 0.85)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="gear_train_dial", robot="null"))
