"""FlywheelInterlockScene — stop the running flywheel, post the parcel through its open
sector into the sealed vault, then restart the machine (sim_gen task `approach_grasp_i405`).

Derived from pick_place/approach_grasp, but STRATEGICALLY different: the seed is the
canonical STATIC acquisition primitive — a Franka approaches a red cube resting in a
quiet scene, closes its jaw around it and lifts. Every phase of the seed acts on a world
at rest, and the goal is a held object. Here the world is RUNNING at t=0: a bladed
flywheel is already coasting at 1.8-3.0 rad/s (random sign) on a frictionless bearing in
front of the only opening of a sealed vault, and it will keep spinning essentially
forever if left alone. The red parcel cube starts far from the machine on a loading
deck. The plan is machine-state bracketed, not acquisition-shaped:
  1. ARREST — brake the live wheel to a standstill (the robot drags the rim / the
     yellow handle peg; the sanctioned plant is a bounded torque about the axle);
  2. ALIGN — jog the stopped wheel so its single 90-degree OPEN SECTOR faces the vault
     mouth at six o'clock (everywhere else the blade ring walls the mouth off);
  3. POST — slide the parcel across the deck, through the open sector and the mouth,
     over the sill; gravity drops it into the roofed vault (the mouth is the vault's
     only aperture). A declared SAFETY INTERLOCK is part of the task: if the parcel
     ever overlaps the blade sweep plane while the wheel is still turning, the episode
     is permanently fouled (physically, the blades bat the parcel away);
  4. RESTART — spin the wheel back up and let it coast: success requires the machine
     RUNNING again (>= w_hold rad/s, either direction) with the parcel sealed inside.
The initial state is kinetic, the goal state is kinetic, and the object transaction is
order-forced between two deliberate changes of machine state. Nothing is grasped-and-
lifted as a goal, there is no proximity target, and unlike every rotary corpus task
(carousels, airlock feeders, cranks) the rotor starts ALREADY MOVING and must end
moving: the rotor is never a vehicle for the payload — it is a gatekeeper hazard.

Mechanics (plain rigid bodies + one authored USD revolute joint; the "hand" on the
wheel is a bounded external torque applied in `post_step`, consuming the `wheel_drive`
buffer that solve/smoke probes write): a fixed kinematic RIG (loading deck, back wall
with the mouth aperture, sealed roofed vault behind it) and a dynamic WHEEL — hub + 8
chord plates filling 270 degrees of an annulus + a yellow handle peg — on a continuous
(no-limit) revolute-Y bearing 6 mm in front of the wall, CoM authored ON the axis so
the wheel is gravity-neutral and parks at any azimuth. The parcel is a free 36 mm cube.

Rubric (graded 0..1, latched credit anchored in the demonstrated solve.py trajectory;
parcel latches require PATH CONTINUITY — per-substep travel under `cont_max` — AND a
persistent transit credential `_path_ok`: any discontinuity disarms it, and it only
re-arms while the parcel is back OUTSIDE the wall on the deck side, so a parcel
teleported into (or near) the vault earns nothing, ever; wheel-state latches require
a sustained streak of FD readings):
  - `calmed` (latch): the wheel's finite-difference rate stayed under w_calm for
    calm_n consecutive substeps — the machine was verifiably stopped.
  - `deposited` (latch): after `calmed`, the parcel continuously travelled into the
    vault interior band (only reachable through the mouth: the vault is roofed and
    walled everywhere else).
  - `respun` (latch): after `deposited`, the wheel sustained |rate| >= w_run for
    run_n consecutive substeps — the machine was restarted.
  - `fouled` (permanent latch): the parcel overlapped the blade sweep annulus while
    the wheel was turning faster than w_gate — the declared interlock violation
    (physically a blade strike). Score caps at foul_cap forever.
  - success(): `deposited` AND `respun` AND not `fouled` AND the parcel currently
    rests quietly inside the vault AND the wheel is currently running (|rate| >=
    w_hold) — a live, coasting end state that persists hands-off.
  - score() = 0.15*calmed + 0.35*deposited + 0.20*respun + 0.30*success -> exactly
    1.0 iff success(); ~0 for the null policy (the wheel never self-calms within an
    episode: bearing decay is ~250 s); a fouled episode caps at 0.15.

Per-episode randomization (readback-verified in smoke.py): the wheel's initial angular
VELOCITY (sign and magnitude — braking effort and direction vary), its initial yaw
(the open sector starts anywhere), and the parcel's spawn on the deck. A memorized
fixed torque schedule cannot park the open sector at the mouth from a random yaw.

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


# ----- USD authoring helpers --------------------------------------------------------------------
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


def _decorate(prim, color, contact_offset: float, material=None) -> None:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None, visual_only: bool = False, orient=None) -> None:
    """Author one box child prim (translate -> [orient] -> scale; authored once)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    if visual_only:
        UsdGeom.Gprim(seg.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    else:
        _decorate(seg.GetPrim(), color, contact_offset, material)


def _cyl(stage, path: str, radius: float, height: float, axis: str, center, color,
         contact_offset: float, material=None, visual_only: bool = False) -> None:
    """Author one cylinder child prim."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr(axis)
    seg.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -radius),
                          Gf.Vec3f(radius, radius, radius)])
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    if visual_only:
        UsdGeom.Gprim(seg.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    else:
        _decorate(seg.GetPrim(), color, contact_offset, material)


def _spawn_rig(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC rig. Local frame: wheel sweep plane at y=0, robot side y<0, up +z.
    Loading DECK (top 2 mm above the mouth sill, front edge 10 mm short of the sweep
    plane) on a support block; back WALL (front face y=wall_face) pierced by the one
    square MOUTH at six o'clock under the axle; sealed roofed VAULT behind the mouth
    (floor sunk below the sill, side/back walls, roof — the mouth is the only way in);
    a visual axle stub marks the bearing."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(40.0)
    mat = _friction_material(stage, f"{prim_path}/mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    wf, wt = cfg.wall_face, cfg.wall_t          # wall spans y in [wf, wf+wt]
    hx, ztop = cfg.wall_hx, cfg.wall_top
    mhw, sill, mtop = cfg.mouth_hw, cfg.sill_z, cfg.mouth_top

    # loading deck + support block ---------------------------------------------------------
    y0, y1, dt_ = cfg.deck_y0, cfg.deck_y1, cfg.deck_t
    _box(stage, f"{prim_path}/deck", (2 * cfg.deck_hx, y1 - y0, dt_),
         (0.0, (y0 + y1) / 2, cfg.deck_top - dt_ / 2), cfg.deck_color, co, material=mat)
    _box(stage, f"{prim_path}/deck_leg", (2 * cfg.deck_hx - 0.02, (y1 - y0) * 0.6,
         cfg.deck_top - dt_), (0.0, (y0 + y1) / 2 - 0.02, (cfg.deck_top - dt_) / 2),
         cfg.base_color, co, material=mat)

    # back wall with the mouth aperture ----------------------------------------------------
    yc = wf + wt / 2
    _box(stage, f"{prim_path}/wall_lo", (2 * hx, wt, sill),
         (0.0, yc, sill / 2), cfg.wall_color, co, material=mat)
    _box(stage, f"{prim_path}/wall_hi", (2 * hx, wt, ztop - mtop),
         (0.0, yc, (ztop + mtop) / 2), cfg.wall_color, co, material=mat)
    for sx in (+1, -1):
        _box(stage, f"{prim_path}/wall_{'e' if sx > 0 else 'w'}",
             (hx - mhw, wt, mtop - sill),
             (sx * (mhw + (hx - mhw) / 2), yc, (mtop + sill) / 2),
             cfg.wall_color, co, material=mat)

    # sealed vault behind the mouth (floor, side walls, back wall, roof) -------------------
    vy0 = wf + wt                                # vault interior front plane
    vy1 = vy0 + cfg.vault_len                    # vault interior back plane
    vt = cfg.vault_wall_t
    vhx = cfg.vault_hx
    _box(stage, f"{prim_path}/vault_floor", (2 * vhx + 2 * vt, cfg.vault_len + vt, 0.012),
         (0.0, (vy0 + vy1 + vt) / 2, cfg.vault_floor - 0.006),
         cfg.vault_color, co, material=mat)
    _box(stage, f"{prim_path}/vault_back", (2 * vhx + 2 * vt, vt, cfg.roof_z0 - cfg.vault_floor),
         (0.0, vy1 + vt / 2, (cfg.roof_z0 + cfg.vault_floor) / 2),
         cfg.vault_color, co, material=mat)
    for sx in (+1, -1):
        _box(stage, f"{prim_path}/vault_{'e' if sx > 0 else 'w'}",
             (vt, cfg.vault_len, cfg.roof_z0 - cfg.vault_floor),
             (sx * (vhx + vt / 2), (vy0 + vy1) / 2, (cfg.roof_z0 + cfg.vault_floor) / 2),
             cfg.vault_color, co, material=mat)
    _box(stage, f"{prim_path}/vault_roof", (2 * vhx + 2 * vt, cfg.vault_len + vt, cfg.roof_t),
         (0.0, (vy0 + vy1 + vt) / 2, cfg.roof_z0 + cfg.roof_t / 2),
         cfg.vault_color, co, material=mat)

    # visual axle stub (the revolute joint itself is invisible) ----------------------------
    _cyl(stage, f"{prim_path}/axle_vis", 0.010, wf + 0.010, "Y",
         (0.0, (wf - 0.010) / 2 + 0.005, cfg.z_axis), cfg.steel_color, co, visual_only=True)
    return root


def _spawn_wheel(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC flywheel. Local frame: origin at the AXIS (revolute Y), sweep plane
    y=0. Hub disc + 8 chord plates every 30 deg from +75 to +285 deg (azimuth measured
    from body -z, increasing toward +x) filling 270 deg of the annulus and leaving ONE
    open sector centered on body -z; a yellow handle peg protrudes toward the robot at
    the gap-opposite azimuth. CoM authored AT THE AXIS -> gravity-neutral; explicit
    diagonal inertia; small angular damping (the bearing coasts for minutes)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in cfg.inertia]))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.02)
    px.CreateAngularDampingAttr(float(cfg.ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    mat = _friction_material(stage, f"{prim_path}/mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    r_mid = (cfg.r_in + cfg.r_out) / 2
    r_len = cfg.r_out - cfg.r_in
    _cyl(stage, f"{prim_path}/hub", cfg.hub_r, cfg.blade_t, "Y", (0.0, 0.0, 0.0),
         cfg.hub_color, co, material=mat)
    for k in range(cfg.n_plates):
        phi = math.radians(cfg.plate_phi0_deg + k * cfg.plate_step_deg)
        ctr = (r_mid * math.sin(phi), 0.0, -r_mid * math.cos(phi))
        a = -phi  # box local z -> inward radial (see derivation in cfg asserts)
        q = (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)
        _box(stage, f"{prim_path}/plate_{k}", (cfg.plate_w, cfg.blade_t, r_len),
             ctr, cfg.plate_color, co, material=mat, orient=q)
    # yellow handle peg on the robot-facing face, opposite the open sector (body +z)
    _cyl(stage, f"{prim_path}/handle",
         cfg.handle_rad, cfg.handle_len, "Y",
         (0.0, -(cfg.blade_t / 2 + cfg.handle_len / 2 + 0.002), cfg.handle_pos_r),
         cfg.handle_color, co, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rig" not in _SPAWNER_CACHE:

        @configclass
        class RigSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rig)
            wall_face: float = 0.012
            wall_t: float = 0.012
            wall_hx: float = 0.160
            wall_top: float = 0.300
            mouth_hw: float = 0.025
            sill_z: float = 0.105
            mouth_top: float = 0.155
            z_axis: float = 0.235
            deck_top: float = 0.107
            deck_t: float = 0.020
            deck_hx: float = 0.110
            deck_y0: float = -0.210
            deck_y1: float = -0.016
            vault_hx: float = 0.045
            vault_len: float = 0.100
            vault_floor: float = 0.070
            vault_wall_t: float = 0.010
            roof_z0: float = 0.159
            roof_t: float = 0.012
            mu_static: float = 0.30
            mu_dynamic: float = 0.25
            deck_color: tuple = (0.70, 0.68, 0.62)
            base_color: tuple = (0.18, 0.18, 0.20)
            wall_color: tuple = (0.38, 0.38, 0.42)
            vault_color: tuple = (0.13, 0.32, 0.36)
            steel_color: tuple = (0.65, 0.66, 0.70)
            contact_offset: float = 0.0015

        @configclass
        class WheelSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_wheel)
            r_in: float = 0.055
            r_out: float = 0.150
            hub_r: float = 0.045
            blade_t: float = 0.012
            plate_w: float = 0.084
            n_plates: int = 8
            plate_phi0_deg: float = 75.0
            plate_step_deg: float = 30.0
            handle_rad: float = 0.008
            handle_len: float = 0.045
            handle_pos_r: float = 0.135
            mass: float = 0.55
            inertia: tuple = (0.0025, 0.0050, 0.0025)
            ang_damp: float = 0.006
            mu_static: float = 0.30
            mu_dynamic: float = 0.25
            hub_color: tuple = (0.20, 0.20, 0.24)
            plate_color: tuple = (0.32, 0.44, 0.66)
            handle_color: tuple = (0.92, 0.80, 0.10)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["rig"] = RigSpawnerCfg
        _SPAWNER_CACHE["wheel"] = WheelSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class FlywheelInterlockSceneCfg(BaseCfg):
    """Config for `FlywheelInterlockScene`. Honesty is asserted in __post_init__: the
    blade ring fully walls the mouth off except through the open sector, and the open
    sector clears the mouth with margin beyond align_tol; the vault is reachable only
    through the mouth (roofed, walled, sunk floor); the parcel bridges the deck-to-sill
    gap; a parcel resting anywhere legal never overlaps the sweep annulus; the drive
    torque spins the wheel up/down in a fraction of a second yet the bearing coasts for
    minutes (the null policy never self-calms; the restarted wheel outlasts the
    persistence window); one substep of the sill drop stays far under the continuity
    gate; everything interactive sits inside the documented arm envelope; the rubric
    weights sum to 1."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    w_calm: float = tunable(0.25)       # `calmed`: FD wheel rate below this ... (rad/s)
    calm_n: int = tunable(60)           # ... for this many consecutive substeps (0.5 s)
    w_gate: float = tunable(0.25)       # interlock: parcel in the sweep while faster -> foul
    w_run: float = tunable(1.2)         # `respun`: FD rate at least this ... (rad/s)
    run_n: int = tunable(90)            # ... for this many consecutive substeps (0.75 s)
    w_hold: float = tunable(1.0)        # success: LIVE wheel rate at least this (rad/s)
    settle_v: float = tunable(0.10)     # success: max parcel |lin vel| (m/s)
    cont_max: float = tunable(0.020)    # path continuity: max parcel travel per substep (m)
    foul_cap: float = tunable(0.15)     # score cap once the interlock is violated

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    w0_lo: float = tunable(1.8)         # initial wheel speed magnitude range (rad/s) ...
    w0_hi: float = tunable(3.0)         # ... sign is a coin flip
    px_jit: float = tunable(0.045)      # parcel spawn x, +/- (m)
    py_lo: float = tunable(-0.18)       # parcel spawn y range on the deck (m)
    py_hi: float = tunable(-0.11)

    # --- tunable: plant (difficulty dials) ------------------------------------------------------
    drive_max: float = tunable(0.30)    # |wheel_drive| clamp (N*m about the axle)
    ang_damp: float = tunable(0.006)    # bearing damping (1/s; coast time-constant ~167 s)
    cube_mass: float = tunable(0.05)
    cube_mu: tuple = tunable((0.35, 0.30))

    # --- info: structure (env-local frame: wheel sweep plane y=0, robot at -y) ------------------
    blade_t: float = info(0.012)        # blade/hub thickness (sweep slab |y| < this/2)
    wall_face: float = info(0.012)      # wall front face y (6 mm behind the blades)
    wall_t: float = info(0.012)
    wall_hx: float = info(0.160)
    wall_top: float = info(0.300)
    mouth_hw: float = info(0.025)       # mouth half-width (x)
    mouth_h: float = info(0.050)        # mouth height (z)
    r_mouth: float = info(0.105)        # axle -> mouth center distance (mouth at 6 o'clock)
    z_axis: float = info(0.235)         # bearing height
    r_in: float = info(0.055)           # blade annulus inner radius
    r_out: float = info(0.150)          # blade annulus outer radius
    hub_r: float = info(0.045)
    plate_w: float = info(0.084)        # chord plate tangential width
    n_plates: int = info(8)
    plate_phi0_deg: float = info(75.0)  # first plate azimuth (from body -z, toward +x)
    plate_step_deg: float = info(30.0)
    handle_pos_r: float = info(0.135)   # yellow handle peg radius on the front face
    handle_len: float = info(0.045)
    wheel_mass: float = info(0.55)
    wheel_inertia: tuple = info((0.0025, 0.0050, 0.0025))
    deck_top: float = info(0.107)       # deck top (2 mm ABOVE the sill: transit steps DOWN)
    deck_t: float = info(0.020)
    deck_hx: float = info(0.110)
    deck_y0: float = info(-0.210)
    deck_y1: float = info(-0.016)       # deck front edge (10 mm short of the sweep plane)
    vault_hx: float = info(0.045)       # vault interior half-width
    vault_len: float = info(0.100)      # vault interior depth (y)
    vault_floor: float = info(0.070)    # vault floor top (35 mm below the sill)
    vault_wall_t: float = info(0.010)
    roof_z0: float = info(0.159)        # vault roof underside (4 mm above the mouth top)
    roof_t: float = info(0.012)
    cube: float = info(0.036)           # parcel cube edge
    align_tol: float = info(0.35)       # gap-to-mouth yaw window for a guaranteed clear (rad)
    slab_pad: float = info(0.004)       # interlock slab margin beyond blade+cube half-extents
    base_pos: tuple = info((0.0, -0.50))  # the documented Franka base xy (TASK.md)
    reach: float = info(0.75)           # documented comfortable arm envelope from base_pos
    contact_offset: float = info(0.0015)
    # rubric weights (sum with success weight = 1.0)
    w_calmed: float = info(0.15)
    w_dep: float = info(0.35)
    w_respun: float = info(0.20)
    w_succ: float = info(0.30)

    # Derived (filled in __post_init__).
    sill_z: float = field(default=None, init=False)      # mouth sill top
    mouth_top: float = field(default=None, init=False)
    z_mouth: float = field(default=None, init=False)     # mouth center height
    slab_y: float = field(default=None, init=False)      # interlock slab |y| bound
    deck_rest_z: float = field(default=None, init=False)  # parcel center on the deck
    vault_rest_z: float = field(default=None, init=False)  # parcel center on the vault floor
    dep_y0: float = field(default=None, init=False)      # deposited band (vault interior)
    dep_y1: float = field(default=None, init=False)
    dep_hx: float = field(default=None, init=False)
    dep_z0: float = field(default=None, init=False)
    dep_z1: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.z_mouth = self.z_axis - self.r_mouth
        self.sill_z = self.z_mouth - self.mouth_h / 2
        self.mouth_top = self.z_mouth + self.mouth_h / 2
        self.slab_y = self.blade_t / 2 + self.cube / 2 + self.slab_pad
        self.deck_rest_z = self.deck_top + self.cube / 2
        self.vault_rest_z = self.vault_floor + self.cube / 2
        vy0 = self.wall_face + self.wall_t
        self.dep_y0 = vy0 + self.cube / 2 + 0.002
        self.dep_y1 = vy0 + self.vault_len - self.cube / 2
        self.dep_hx = self.vault_hx - 0.004
        self.dep_z0 = self.vault_floor + 0.006
        self.dep_z1 = self.vault_floor + 0.045

        # the parcel fits the mouth with real but small clearance
        assert 0.010 <= 2 * self.mouth_hw - self.cube <= 0.020, "mouth width clearance"
        assert 0.010 <= self.mouth_h - self.cube <= 0.020, "mouth height clearance"
        # transit steps DOWN from deck to sill (a flush sill walls a slid cube)
        assert 0.001 <= self.deck_top - self.sill_z <= 0.004, "deck-to-sill step"
        # the parcel bridges the deck-edge -> wall-face gap with >= 6 mm engagement
        gap = self.wall_face - self.deck_y1
        assert gap <= self.cube - 0.006, f"parcel cannot bridge the {gap:.3f} m transit gap"
        assert gap >= self.blade_t + 0.008, "blades cannot sweep between deck and wall"
        # blades clear both the deck edge and the wall face
        assert self.wall_face - self.blade_t / 2 >= 0.004, "blades scrape the wall"
        assert -self.deck_y1 - self.blade_t / 2 >= 0.004, "blades scrape the deck edge"
        # blade ring geometry: plates cover the annulus contiguously outside the open
        # sector, and the open sector clears the whole mouth beyond align_tol.
        # (plate box local z maps to the inward radial: R_y(-phi) e_z = (-sin phi, 0,
        # cos phi) = -radial(phi) for radial(phi) = (sin phi, 0, -cos phi).)
        r_hi = self.r_out - 0.004
        half_cov_out = math.asin((self.plate_w / 2) / r_hi)
        assert math.degrees(half_cov_out) * 2 >= self.plate_step_deg + 1.0, \
            "adjacent plates leave a radial crack at the rim"
        # mouth extreme radii from the axle (nearest edge / farthest corner)
        r_near = self.r_mouth - self.mouth_h / 2
        r_far = math.hypot(self.r_mouth + self.mouth_h / 2, self.mouth_hw)
        assert self.r_in + 0.004 <= r_near and r_far <= self.r_out - 0.004, \
            "mouth radial band escapes the blade annulus (a closed wheel would leak)"
        # open sector: first plate's near edge at the mouth's nearest radius vs the
        # mouth's widest azimuth, with align_tol of parking slack plus 4 deg margin
        half_cov_near = math.asin((self.plate_w / 2) / r_near)
        mouth_az = math.atan2(self.mouth_hw, r_near)
        margin = math.radians(self.plate_phi0_deg) - half_cov_near - mouth_az
        assert margin >= self.align_tol + math.radians(4.0), \
            f"open sector margin {math.degrees(margin):.1f} deg too small for align_tol"
        # gravity-neutral wheel: CoM at the axis is authored; hub clears the mouth
        assert self.z_axis - self.hub_r >= self.mouth_top + 0.020, "hub reaches the mouth"
        # the vault is sealed except the mouth: roof above the mouth top, floor sunk
        assert self.roof_z0 >= self.mouth_top + 0.002, "roof lip blocks the mouth"
        assert self.sill_z - self.vault_floor >= 0.020, "vault floor not sunk"
        assert self.roof_z0 - self.vault_floor >= self.cube + 0.040, "vault too shallow"
        assert 2 * self.vault_hx >= self.cube + 0.030, "vault too narrow"
        assert self.dep_y1 - self.dep_y0 >= 0.030, "deposited y band degenerate"
        assert self.dep_z0 < self.vault_rest_z < self.dep_z1 - 0.010, "rest outside band"
        # parcel spawns rest on the deck, clear of edges and far outside the sweep slab
        assert self.py_hi + self.cube / 2 <= -(self.slab_y + 0.050), "spawn near the sweep"
        assert self.py_lo - self.cube / 2 >= self.deck_y0 + 0.004, "spawn off the deck back"
        assert self.px_jit + self.cube / 2 <= self.deck_hx - 0.004, "spawn off the deck side"
        # interlock separation: a fresh episode can never foul or calm spontaneously
        assert self.w0_lo >= 5.0 * self.w_gate, "initial spin too slow vs the interlock gate"
        assert self.w_calm <= self.w_gate, "calm threshold above the interlock gate"
        assert self.w_hold < self.w_run <= self.w0_lo, "run/hold/initial speed ordering"
        # bearing: the null policy never self-calms (60 s decay stays >> w_calm), yet
        # the restarted wheel outlasts the persistence window with margin
        assert self.w0_lo * math.exp(-self.ang_damp * 60.0) > 2.0 * self.w_calm, \
            "the wheel self-calms within an episode - null would earn calm credit"
        assert 1.3 * self.w_run * math.exp(-self.ang_damp * 5.0) > 1.2 * self.w_hold, \
            "a restarted wheel sags below w_hold within the persistence window"
        # drive authority: spin-up/brake in well under a second (I alpha >> requirement)
        alpha = self.drive_max / self.wheel_inertia[1]
        assert alpha >= 5.0 * self.w0_hi, "drive cannot brake the wheel briskly"
        assert alpha >= 5.0 * self.w_run, "drive cannot restart the wheel briskly"
        # continuity gate: one substep of the worst sill drop stays far under the gate
        v_fall = math.sqrt(2 * 9.81 * (self.sill_z + self.cube / 2 - self.vault_rest_z))
        assert 2.0 * v_fall / 120.0 < self.cont_max, "continuity gate rejects the drop"
        # everything interactive inside the documented arm envelope
        bx, by = self.base_pos
        pts = [
            (0.0, -(self.blade_t / 2 + self.handle_len + 0.002),
             self.z_axis + self.handle_pos_r),                     # handle orbit top
            (self.r_out, 0.0, self.z_axis + self.r_out * 0.9),     # rim high side
            (self.deck_hx, self.deck_y0, self.deck_top),           # far deck corner
            (0.0, self.wall_face, self.z_mouth),                   # the mouth itself
        ]
        far = max(math.sqrt((px - bx) ** 2 + (py - by) ** 2 + pz ** 2) for px, py, pz in pts)
        assert far < self.reach, f"workspace point {far:.3f} m outside reach {self.reach} m"
        # weights sum to exactly 1; a fouled episode can never out-earn honest calm work
        s = self.w_calmed + self.w_dep + self.w_respun + self.w_succ
        assert abs(s - 1.0) < 1e-9, "rubric weights must sum to 1"
        assert self.foul_cap <= self.w_calmed + 1e-9, "foul cap above the calm credit"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("flywheel_interlock")
class FlywheelInterlockScene(BaseScene):
    cfg: FlywheelInterlockSceneCfg

    def __init__(self, cfg: FlywheelInterlockSceneCfg | None = None) -> None:
        super().__init__(cfg or FlywheelInterlockSceneCfg())

    # ----- assets --------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic rig, the free-spinning wheel (revolute joint
        authored in bind()), and the red parcel cube."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "rig": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rig",
                spawn=sp["rig"](
                    wall_face=c.wall_face, wall_t=c.wall_t, wall_hx=c.wall_hx,
                    wall_top=c.wall_top, mouth_hw=c.mouth_hw, sill_z=c.sill_z,
                    mouth_top=c.mouth_top, z_axis=c.z_axis, deck_top=c.deck_top,
                    deck_t=c.deck_t, deck_hx=c.deck_hx, deck_y0=c.deck_y0,
                    deck_y1=c.deck_y1, vault_hx=c.vault_hx, vault_len=c.vault_len,
                    vault_floor=c.vault_floor, vault_wall_t=c.vault_wall_t,
                    roof_z0=c.roof_z0, roof_t=c.roof_t, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "wheel": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Wheel",
                spawn=sp["wheel"](
                    r_in=c.r_in, r_out=c.r_out, hub_r=c.hub_r, blade_t=c.blade_t,
                    plate_w=c.plate_w, n_plates=c.n_plates,
                    plate_phi0_deg=c.plate_phi0_deg, plate_step_deg=c.plate_step_deg,
                    handle_pos_r=c.handle_pos_r, handle_len=c.handle_len,
                    mass=c.wheel_mass, inertia=c.wheel_inertia, ang_damp=c.ang_damp,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.z_axis)),
            ),
            "parcel": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Parcel",
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube, c.cube, c.cube),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.05,
                        sleep_threshold=0.0, stabilization_threshold=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.cube_mu[0], dynamic_friction=c.cube_mu[1]),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.85, 0.10, 0.10)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, (c.py_lo + c.py_hi) / 2, c.deck_rest_z + 0.002)),
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
                "enable_external_forces_every_iteration": True,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        self._quat_apply = quat_apply
        self._quat_apply_inv = quat_apply_inverse
        n = env.num_envs
        dev = env.device
        self.rig: RigidObject = env.iscene["rig"]
        self.wheel: RigidObject = env.iscene["wheel"]
        self.parcel: RigidObject = env.iscene["parcel"]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        self._dt = 1.0 / 120.0
        # External drive input (solve/smoke write; post_step consumes + owns the wrench
        # slot — never call set_external_force_and_torque on the wheel directly).
        self.wheel_drive = torch.zeros(n, device=dev)  # torque about the axle (+y)
        # Episode draws (smoke readback).
        self.w0 = torch.zeros(n, device=dev)
        self.th0 = torch.zeros(n, device=dev)
        # Rubric state.
        self.calmed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.deposited = torch.zeros(n, dtype=torch.bool, device=dev)
        self.respun = torch.zeros(n, dtype=torch.bool, device=dev)
        self.fouled = torch.zeros(n, dtype=torch.bool, device=dev)
        self.calm_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self.run_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._pay_prev = torch.zeros(n, 3, device=dev)
        self._prev_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self._path_ok = torch.zeros(n, dtype=torch.bool, device=dev)
        self._th_prev = torch.zeros(n, device=dev)
        self.wheel_w = torch.zeros(n, device=dev)  # FD rate (root_ang_vel is phantom)

    def _author_joints(self) -> None:
        """Per env: rig --revolute Y--> wheel, NO limit attrs (a continuous bearing —
        the wheel must turn whole revolutions in either direction). The joint pair
        never collides; the wheel meets the world only through blade/handle contact."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/wheel_bearing")
            j.CreateBody0Rel().SetTargets([f"{base}/Rig"])
            j.CreateBody1Rel().SetTargets([f"{base}/Wheel"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.z_axis)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            # NO limit attrs: continuous rotation, no 180-deg wrap stop.

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the wheel's initial yaw (uniform full circle) and its
        LIVE angular velocity (coin-flip sign, magnitude in [w0_lo, w0_hi]), and the
        parcel's deck spawn; write the spinning wheel state; zero every latch and the
        drive buffer. Draws are burned first (degenerate-first-draw guard)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        for _ in range(4):
            torch.rand(2 * m + 4, device=dev)  # burn the correlated RNG prefix

        sgn = torch.where(torch.rand(m, device=dev) < 0.5,
                          -torch.ones(m, device=dev), torch.ones(m, device=dev))
        w0 = sgn * (c.w0_lo + torch.rand(m, device=dev) * (c.w0_hi - c.w0_lo))
        th0 = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        px = (torch.rand(m, device=dev) * 2 - 1) * c.px_jit
        py = c.py_lo + torch.rand(m, device=dev) * (c.py_hi - c.py_lo)
        self.w0[env_ids] = w0
        self.th0[env_ids] = th0

        st = torch.zeros(m, 13, device=dev)  # wheel: yaw th0 about +y, LIVE spin w0
        st[:, 2] = c.z_axis
        st[:, 0:3] += origin
        st[:, 3] = torch.cos(th0 / 2)
        st[:, 5] = torch.sin(th0 / 2)
        st[:, 11] = w0                       # angular velocity about world +y
        self.wheel.write_root_state_to_sim(st, env_ids)

        st = torch.zeros(m, 13, device=dev)  # parcel on the deck
        st[:, 0] = px
        st[:, 1] = py
        st[:, 2] = c.deck_rest_z + 0.002
        st[:, 0:3] += origin
        st[:, 3] = 1.0
        self.parcel.write_root_state_to_sim(st, env_ids)

        self.calmed[env_ids] = False
        self.deposited[env_ids] = False
        self.respun[env_ids] = False
        self.fouled[env_ids] = False
        self.calm_streak[env_ids] = 0
        self.run_streak[env_ids] = 0
        self._pay_prev[env_ids] = 0.0
        self._prev_valid[env_ids] = False
        self._path_ok[env_ids] = True  # spawn is on the deck, outside the wall
        self._th_prev[env_ids] = th0
        self.wheel_w[env_ids] = w0
        self.wheel_drive[env_ids] = 0.0

    # ----- readings ------------------------------------------------------------------------------
    def _local(self, body) -> torch.Tensor:
        """(N, 3) body center in the env-local rig frame (rig fixed at the origin)."""
        return body.data.root_pos_w - self.env_origins

    def wheel_yaw(self) -> torch.Tensor:
        """(N,) wheel angle about +y in (-pi, pi]: 0 = open sector at six o'clock
        (aligned with the mouth); R_y(th) e_x = (cos th, 0, -sin th)."""
        n = self.env.num_envs
        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(n, 3)
        v = self._quat_apply(self.wheel.data.root_quat_w, ex)
        return torch.atan2(-v[:, 2], v[:, 0])

    def gap_aligned(self) -> torch.Tensor:
        """(N,) bool: the open sector is parked over the mouth within align_tol."""
        return self.wheel_yaw().abs() < self.cfg.align_tol

    def in_vault(self) -> torch.Tensor:
        """(N,) bool: parcel currently inside the vault interior band (live)."""
        c = self.cfg
        p = self._local(self.parcel)
        return ((p[:, 0].abs() < c.dep_hx) & (p[:, 1] > c.dep_y0) & (p[:, 1] < c.dep_y1)
                & (p[:, 2] > c.dep_z0) & (p[:, 2] < c.dep_z1))

    def in_sweep(self, p: torch.Tensor) -> torch.Tensor:
        """(N,) bool: a parcel at env-local p overlaps the blade sweep annulus slab."""
        c = self.cfg
        rad = torch.sqrt(p[:, 0] ** 2 + (p[:, 2] - c.z_axis) ** 2)
        return ((p[:, 1].abs() < c.slab_y)
                & (rad > c.r_in - 0.5 * c.cube) & (rad < c.r_out + 0.5 * c.cube))

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the parcel was posted along the latched path and now rests
        quietly inside the sealed vault, the interlock was never violated, and the
        machine is RUNNING again (a live coasting wheel)."""
        c = self.cfg
        v = self.parcel.data.root_lin_vel_w.norm(dim=-1)
        return (self.deposited & self.respun & ~self.fouled & self.in_vault()
                & (v < c.settle_v) & (self.wheel_w.abs() >= c.w_hold))

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: latched machine-state credit + live success. Exactly
        1.0 iff success(); ~0 for the null policy (the wheel never self-calms); an
        interlock-fouled episode caps at foul_cap forever."""
        c = self.cfg
        s = (c.w_calmed * self.calmed.float() + c.w_dep * self.deposited.float()
             + c.w_respun * self.respun.float() + c.w_succ * self.success().float())
        return torch.where(self.fouled, s.clamp(max=c.foul_cap), s)

    # ----- step-coupled mechanics (every substep) -------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Wheel plant (clamped drive buffer applied as a body-frame torque about the
        axle), the FD wheel rate, then the streak/continuity-gated latches."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device

        # --- plant: bounded hand torque about the axle (+y) ---
        q = self.wheel.data.root_quat_w
        tau = torch.zeros(n, 3, device=dev)
        tau[:, 1] = self.wheel_drive.clamp(-c.drive_max, c.drive_max)
        tau_body = self._quat_apply_inv(q, tau)  # wrench slot takes BODY-frame torque
        self.wheel.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), tau_body.reshape(n, 1, 3))

        # --- FD wheel rate (root_ang_vel is phantom under external wrenches) ---
        th = self.wheel_yaw()
        dth = th - self._th_prev
        dth = torch.where(dth > math.pi, dth - 2 * math.pi, dth)
        dth = torch.where(dth < -math.pi, dth + 2 * math.pi, dth)
        self.wheel_w = torch.where(self._prev_valid, dth / self._dt, self.wheel_w)
        self._th_prev = th

        # --- latches ---
        w_abs = self.wheel_w.abs()
        self.calm_streak = torch.where(self._prev_valid & (w_abs < c.w_calm),
                                       self.calm_streak + 1,
                                       torch.zeros_like(self.calm_streak))
        self.calmed = self.calmed | (self.calm_streak >= c.calm_n)

        p = self._local(self.parcel)
        p = torch.nan_to_num(p, nan=1e3, posinf=1e3, neginf=-1e3)
        pw = self.parcel.data.root_pos_w
        jump = (pw - self._pay_prev).norm(dim=-1)
        cont = self._prev_valid & (jump < c.cont_max)

        # transit credential: any discontinuity disarms; re-arms only while the parcel
        # is back OUTSIDE the wall on the deck side (transport teleports on the deck
        # are legal; a teleport into or near the vault stays disarmed forever).
        disarm = self._prev_valid & (jump >= c.cont_max)
        outside = p[:, 1] < -(c.slab_y + 0.010)
        self._path_ok = (self._path_ok & ~disarm) | (outside & ~disarm)

        # declared interlock: parcel in the sweep annulus while the wheel turns = foul
        self.fouled = self.fouled | (self.in_sweep(p) & (w_abs > c.w_gate))

        self.deposited = self.deposited | (cont & self._path_ok & self.calmed
                                           & ~self.fouled & self.in_vault())

        self.run_streak = torch.where(
            self._prev_valid & self.deposited & (w_abs >= c.w_run),
            self.run_streak + 1, torch.zeros_like(self.run_streak))
        self.respun = self.respun | (self.run_streak >= c.run_n)

        self._pay_prev = pw.clone()
        self._prev_valid = torch.ones_like(self._prev_valid)

    # ----- state (full, restorable) --------------------------------------------------------------
    def _bodies(self) -> dict[str, Any]:
        return {"rig": self.rig, "wheel": self.wheel, "parcel": self.parcel}

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self._bodies().items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("w0", "th0", "calmed", "deposited", "respun", "fouled",
                               "calm_streak", "run_streak", "_pay_prev", "_prev_valid",
                               "_path_ok", "_th_prev", "wheel_w", "wheel_drive")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A machine is RUNNING in front of the robot: a bladed flywheel "
            f"({c.r_out * 100:.0f} cm radius, blue chord plates on a dark hub, one "
            f"quarter of the ring OPEN, a yellow handle peg on its face opposite the "
            f"opening) is already coasting at 1.8-3.0 rad/s — direction varies — on a "
            f"nearly frictionless bearing {c.z_axis * 100:.0f} cm up, and it will not "
            f"stop on its own. Six millimetres behind the wheel stands a wall whose "
            f"only opening is a {2 * c.mouth_hw * 1000:.0f} mm square MOUTH at six "
            f"o'clock, directly under the axle; behind the mouth is a sealed teal "
            f"VAULT (roofed, walled, floor sunk {1000 * (c.sill_z - c.vault_floor):.0f} "
            f"mm below the sill) — the mouth is its only entrance, and the wheel's "
            f"blade ring walls the mouth off except when the open sector is parked "
            f"over it. A RED parcel cube ({c.cube * 1000:.0f} mm) waits on the loading "
            f"deck in front, whose top sits 2 mm above the mouth sill.\n"
            f"Goal, in order: (1) BRAKE the spinning wheel to a standstill (drag the "
            f"rim or the yellow peg); (2) JOG the stopped wheel until the open sector "
            f"faces the mouth; (3) POST the parcel — slide it across the deck, through "
            f"the opening and the mouth, so it drops into the vault; (4) RESTART the "
            f"machine — spin the wheel back up (either direction) and let it coast at "
            f"{c.w_hold:.1f} rad/s or more with the parcel sealed inside. SAFETY "
            f"INTERLOCK: if the parcel ever enters the blade sweep while the wheel is "
            f"turning faster than {c.w_gate:.2f} rad/s, the blades strike it and the "
            f"episode is permanently spoiled. The wheel must be verifiably stopped "
            f"({c.w_calm:.2f} rad/s for {c.calm_n / 120:.1f} s) before the parcel may "
            f"pass, and must run again at the end — a stopped machine with the parcel "
            f"inside is NOT success."
        )

    def instruction(self) -> str:
        """SHORT imperative form for VLA training."""
        return (
            "Brake the coasting flywheel to a stop, rotate it so its open quarter "
            "faces the square mouth under the axle, slide the red parcel across the "
            "deck through the opening so it falls into the sealed vault, then spin "
            "the wheel back up and leave it running. Never let the parcel near the "
            "blades while the wheel is moving — a strike spoils the episode for good."
        )


# ----- runnable env: scene physics only (NullRobot solve/smoke) -> "simgen.flywheel_interlock" --
register_env("simgen", lambda: EnvCfg(scene="flywheel_interlock", robot="null"))
