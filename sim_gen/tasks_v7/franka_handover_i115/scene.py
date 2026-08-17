"""TransferCarouselScene — hand a parcel across a sealed counter through a rotating
pass-through carousel (sim_gen task `franka_handover_i115`).

Derived from bimanual/franka_handover, but STRATEGICALLY different: the seed's whole
skill is a direct arm-to-arm transfer — two Frankas reach the same cube and one hands
it to the other through free space. Here there is ONE arm and NO direct path: the
station's delivery alcove is SEALED (arc wall + canopy roof + side pillars), so the
parcel physically cannot be carried, dropped, or slid to the delivery spot. The only
way across the boundary is the station's own PASS-THROUGH CAROUSEL: a rotating drum
whose open-top bay is loadable only while it faces the open side, then must be
rotated (by pushing the drum's yellow pegs) until the loaded bay sits inside the
sealed alcove. The "handover" is re-interpreted as a mechanism-mediated transfer with
a PHYSICALLY ENFORCED ordering: load first, rotate second — the canopy's 14 mm slot
over the drum makes loading in the alcove impossible, and the sealed perimeter makes
every non-carousel route impossible (smoke proves both with settled constructs).

The scene also forces PERCEPTION: three parcels (red / green / blue cubes) rest on
the apron; a colour tile on the corner pedestal names the one parcel that must be
transferred. Distractors must be left alone (a distractor in the bay at judging time
fails the episode).

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide): station (heavy DYNAMIC compound — a kinematic root would orphan the
drum joint anchor when reset teleports it: slab, 12-segment arc wall, canopy roof,
diametral-plane pillars, corner pedestal), drum (DYNAMIC compound on an authored
revolute Z joint: platter, hub, walled bay, three push pegs; explicit MassAPI mass +
CoM + diagonal inertia so the plant's discrete damping is auditable), three parcel
cubes, three kinematic colour tiles (the target's tile on the pedestal, the others
parked off-field). The drum carries a post_step torque plant (viscous damping +
external `drum_drive` clamp) with a finite-difference hinge rate — `root_ang_vel_w`
is phantom under external wrenches on this stack.

Per-episode randomization (readback-verified by smoke): station yaw FREE (+/-180 deg)
+ xy jitter, initial drum angle theta0 (+/-40 deg), WHICH colour is the target, the
colour->apron-slot permutation, parcel jitter + yaw. All discrete draws derive from
torch.rand (the first torch.randint after manual_seed is degenerate on this stack).

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.35  loaded    — the TARGET parcel ever seated in the bay (latched)
  0.40  progress  — latched max rotation progress toward the alcove, gated ABOARD
                    (only counts while the target rides the bay)
capped at 0.75; exactly 1.0 iff success(): target parcel seated in the bay, bay in
the delivery sector inside the alcove, drum at rest, no distractor in the bay,
everything settled and finite. Null policy ~0 (nothing is loaded, nothing turns).
The seed's strategy — carry the cube to the destination — earns ~0: the sealed
canopy stops the carry (the parcel ends up resting ON the roof) and progress is
aboard-gated, so even spinning the empty drum scores 0.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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


# ----- custom compound spawners -----------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _root_xform(prim_path: str, translation, orientation):
    """Define an Xform root and author its (idempotent, single) translate/orient ops."""
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


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _span(stage, path: str, *, x, y, z, color, collide: Callable):
    """Box child from axis spans (x0, x1), (y0, y1), (z0, z1)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2))
    xf.AddScaleOp().Set(Gf.Vec3f(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _rbox(stage, path: str, *, center, size, yaw: float, color, collide: Callable):
    """Yaw-rotated box child (for arc-wall segments)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    h = yaw / 2
    xf.AddOrientOp().Set(Gf.Quatf(math.cos(h), Gf.Vec3f(0.0, 0.0, math.sin(h))))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _cyl(stage, path: str, *, cx, cy, r, z0, z1, color, collide: Callable):
    """Z-axis cylinder child from a centre, radius and z span."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(float(r))
    h = float(z1 - z0)
    cyl.CreateHeightAttr(h)
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(float(cx), float(cy), float((z0 + z1) / 2)))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    return cyl.GetPrim()


def _author_mass(root, mass: float, com, inertia) -> None:
    """Explicit MassAPI mass + CoM + diagonal inertia. On this stack, MassAPI mass on a
    compound root leaves the CoM at the body ORIGIN and the shape-derived inertia is
    unknown — author all three so plant stability (b*dt/I) is auditable."""
    from pxr import Gf, UsdPhysics

    api = UsdPhysics.MassAPI.Apply(root)
    api.CreateMassAttr(float(mass))
    api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))


def _mk_material(prim_path: str, name: str, mu_s: float, mu_d: float, combine: str) -> str:
    import isaaclab.sim as sim_utils

    mat_path = f"{prim_path}/{name}"
    sim_utils.spawn_rigid_body_material(mat_path, sim_utils.RigidBodyMaterialCfg(
        static_friction=float(mu_s), dynamic_friction=float(mu_d), restitution=0.0,
        friction_combine_mode=combine))
    return mat_path


def _spawn_station(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the station at `prim_path`: heavy DYNAMIC compound. Local frame: origin
    at the carousel pivot on the slab TOP (z=0); +x is the OPEN (loading) side; the
    sealed delivery alcove (arc wall + canopy + pillars) occupies the -x half; the
    corner pedestal (colour-tile stand) sits on the +x,+y corner of the slab."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    S = c.slab_half

    # --- slab (the counter top the whole station stands on) --------------------------
    _span(stage, f"{prim_path}/slab", x=(-S, S), y=(-S, S), z=(-c.slab_t, 0.0),
          color=c.slab_color, collide=collide)

    # --- arc wall: 12 yaw-rotated segments spanning azimuth 90..270 deg ---------------
    rm = c.wall_r_in + c.wall_t / 2
    step = math.pi / c.wall_seg  # 180 deg / segments
    seg_len = 2.0 * (c.wall_r_in + c.wall_t) * math.sin(step / 2) + 0.004
    for k in range(c.wall_seg):
        az = math.pi / 2 + step / 2 + k * step
        _rbox(stage, f"{prim_path}/wall_{k}",
              center=(rm * math.cos(az), rm * math.sin(az), c.wall_z1 / 2),
              size=(c.wall_t, seg_len, c.wall_z1), yaw=az,
              color=c.wall_color, collide=collide)

    # --- canopy roof over the -x half (underside is the 14 mm pass slot) --------------
    _span(stage, f"{prim_path}/roof", x=(c.roof_x0, c.roof_x1),
          y=(-c.roof_y, c.roof_y), z=(c.roof_z0, c.roof_z1),
          color=c.roof_color, collide=collide)

    # --- side pillars sealing the diametral plane beside the drum ---------------------
    for tag, sgn in (("p", 1.0), ("n", -1.0)):
        y0, y1 = sorted((sgn * c.pillar_y0, sgn * c.pillar_y1))
        _span(stage, f"{prim_path}/pillar_{tag}", x=(c.pillar_x0, c.pillar_x1),
              y=(y0, y1), z=(0.0, c.roof_z0), color=c.wall_color, collide=collide)

    # --- corner pedestal (the colour tile stands on top) ------------------------------
    px, py = c.ped_center
    P = c.ped_half
    _span(stage, f"{prim_path}/pedestal", x=(px - P, px + P), y=(py - P, py + P),
          z=(0.0, c.ped_h), color=c.ped_color, collide=collide)

    # --- dynamic root (NOT kinematic: the drum joint anchor must follow teleports) ----
    UsdPhysics.RigidBodyAPI.Apply(root)
    _author_mass(root, c.station_mass, (0.0, 0.0, -0.02), c.station_inertia)
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateLinearDampingAttr(0.5)
    prb.CreateAngularDampingAttr(2.0)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)
    base = _mk_material(prim_path, "base", c.base_mu_s, c.base_mu_d, "average")
    bind_physics_material(prim_path, base)
    return root


def _spawn_drum(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the carousel drum at `prim_path`: DYNAMIC compound. Local frame shares
    the station pivot (origin on the slab top plane, z=0): platter disc, centre hub,
    open-top walled bay on local +x, three yellow push pegs at 90/180/270 deg."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg

    _cyl(stage, f"{prim_path}/platter", cx=0.0, cy=0.0, r=c.platter_r,
         z0=c.platter_z0, z1=c.platter_z1, color=c.platter_color, collide=collide)
    _cyl(stage, f"{prim_path}/hub", cx=0.0, cy=0.0, r=c.hub_r,
         z0=c.platter_z1, z1=c.hub_z1, color=c.hub_color, collide=collide)

    rb, bi, t = c.bay_r, c.bay_in, c.bay_t
    zw = (c.platter_z1, c.bay_z1)
    _span(stage, f"{prim_path}/bay_xp", x=(rb + bi, rb + bi + t),
          y=(-bi - t, bi + t), z=zw, color=c.bay_color, collide=collide)
    _span(stage, f"{prim_path}/bay_xn", x=(rb - bi - t, rb - bi),
          y=(-bi - t, bi + t), z=zw, color=c.bay_color, collide=collide)
    _span(stage, f"{prim_path}/bay_yp", x=(rb - bi, rb + bi),
          y=(bi, bi + t), z=zw, color=c.bay_color, collide=collide)
    _span(stage, f"{prim_path}/bay_yn", x=(rb - bi, rb + bi),
          y=(-bi - t, -bi), z=zw, color=c.bay_color, collide=collide)

    for k, az in enumerate((90.0, 180.0, 270.0)):
        a = math.radians(az)
        _cyl(stage, f"{prim_path}/peg_{k}", cx=c.peg_rad * math.cos(a),
             cy=c.peg_rad * math.sin(a), r=c.peg_r, z0=c.platter_z1, z1=c.peg_z1,
             color=c.peg_color, collide=collide)

    UsdPhysics.RigidBodyAPI.Apply(root)
    _author_mass(root, c.drum_mass, (0.0, 0.0, 0.04), c.drum_inertia)
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)
    grip = _mk_material(prim_path, "grip", c.drum_mu_s, c.drum_mu_d, "average")
    bind_physics_material(prim_path, grip)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "station" not in _SPAWNER_CACHE:

        @configclass
        class StationSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_station)
            slab_half: float = 0.30
            slab_t: float = 0.04
            wall_r_in: float = 0.172
            wall_t: float = 0.022
            wall_z1: float = 0.118
            wall_seg: int = 12
            roof_x0: float = -0.195
            roof_x1: float = 0.005
            roof_y: float = 0.195
            roof_z0: float = 0.092
            roof_z1: float = 0.112
            pillar_x0: float = -0.020
            pillar_x1: float = 0.005
            pillar_y0: float = 0.172
            pillar_y1: float = 0.195
            ped_center: tuple = (0.265, 0.225)
            ped_half: float = 0.030
            ped_h: float = 0.095
            station_mass: float = 30.0
            station_inertia: tuple = (1.0, 1.0, 1.8)
            slab_color: tuple = (0.42, 0.40, 0.36)
            wall_color: tuple = (0.35, 0.38, 0.45)
            roof_color: tuple = (0.30, 0.33, 0.40)
            ped_color: tuple = (0.30, 0.30, 0.32)
            contact_offset: float = 0.0015
            base_mu_s: float = 0.50
            base_mu_d: float = 0.45

        @configclass
        class DrumSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_drum)
            platter_r: float = 0.160
            platter_z0: float = 0.006
            platter_z1: float = 0.018
            hub_r: float = 0.032
            hub_z1: float = 0.078
            bay_r: float = 0.098
            bay_in: float = 0.042
            bay_t: float = 0.008
            bay_z1: float = 0.078
            peg_r: float = 0.011
            peg_z1: float = 0.073
            peg_rad: float = 0.132
            drum_mass: float = 1.5
            drum_inertia: tuple = (0.010, 0.010, 0.016)
            platter_color: tuple = (0.55, 0.56, 0.60)
            hub_color: tuple = (0.45, 0.46, 0.50)
            bay_color: tuple = (0.16, 0.18, 0.22)
            peg_color: tuple = (0.85, 0.75, 0.15)
            contact_offset: float = 0.0015
            drum_mu_s: float = 0.70
            drum_mu_d: float = 0.60

        _SPAWNER_CACHE["station"] = StationSpawnerCfg
        _SPAWNER_CACHE["drum"] = DrumSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TransferCarouselSceneCfg(BaseCfg):
    """Config for `TransferCarouselScene`. The interlock contract is asserted in
    `__post_init__`: the canopy slot is too thin to load the bay inside the alcove
    but tall enough for the LOADED bay to rotate through; the drum sweep clears the
    walls/pillars by rotation-invariant margins (joint-pair collision filtering is
    not trusted either way on this stack); the delivery sector lies fully under the
    canopy; the apron slots are reachable, on the slab and clear of the sweep; and
    the drive plant is discretely stable (b*dt/I, tau_max*dt/I audited)."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    settle_lin: float = tunable(0.05)      # max parcel |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.15)    # max |FD drum rate| when judging (rad/s)
    sector_tol: float = tunable(0.35)      # |wrap(theta - pi)| delivery sector half-width (rad)
    bay_xy_tol: float = tunable(0.030)     # parcel centre box in the drum frame (m)
    bay_z_tol: float = tunable(0.016)      # parcel resting-height tolerance (m)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_deg: float = tunable(180.0)        # station yaw uniform +/- (FREE heading)
    xy_jitter: float = tunable(0.05)       # station xy jitter (+/- m)
    theta0_deg: float = tunable(40.0)      # initial drum angle uniform +/- (bay near +x)
    randomize_target: bool = tunable(True)  # sample WHICH colour is the target
    randomize_slots: bool = tunable(True)   # permute colour -> apron slot
    parcel_jitter: float = tunable(0.015)  # parcel xy jitter in its slot (+/- m)

    # --- tunable: drum drive plant ---------------------------------------------------------------
    # Discrete-stability audit (explicit external torques at 120 Hz, authored Iz=0.016):
    # b*dt/I = 0.20/(120*0.016) ~ 0.104 << 1 (monotone decay); tau_max/I*dt = 0.625 rad/s
    # per step worst case; post_step wrenches act one substep late, so solver gains must
    # keep K*dt/I well under 1 (solve uses K_w = 0.8 -> 0.42).
    tau_max: float = tunable(1.2)          # |drum_drive| clamp (N*m)
    drum_b: float = tunable(0.20)          # viscous plant damping (N*m*s/rad)
    rate_clamp: float = tunable(20.0)      # FD rate clamp (rad/s) — teleport transients

    # --- info: station (local frame: origin at the pivot on the slab top, +x = open side) --------
    base_z: float = info(0.041)            # root height: slab bottom 1 mm above ground, settles
    slab_half: float = info(0.30)
    slab_t: float = info(0.04)
    wall_r_in: float = info(0.172)         # arc wall inner face (chord planes at this radius)
    wall_t: float = info(0.022)
    wall_z1: float = info(0.118)
    wall_seg: int = info(12)
    roof_x0: float = info(-0.195)
    roof_x1: float = info(0.005)
    roof_y: float = info(0.195)
    roof_z0: float = info(0.092)           # canopy underside: the 14 mm pass slot over the drum
    roof_z1: float = info(0.112)
    pillar_x0: float = info(-0.020)
    pillar_x1: float = info(0.005)
    pillar_y0: float = info(0.172)
    pillar_y1: float = info(0.195)
    ped_center: tuple = info((0.265, 0.225))
    ped_half: float = info(0.030)
    ped_h: float = info(0.095)
    station_mass: float = info(30.0)       # heavy DYNAMIC fixture: the joint anchor must
    station_inertia: tuple = info((1.0, 1.0, 1.8))  # follow reset teleports (kinematic won't)
    # --- info: drum ------------------------------------------------------------------------------
    platter_r: float = info(0.160)
    platter_z0: float = info(0.006)        # 6 mm vertical clearance over the slab
    platter_z1: float = info(0.018)        # bay floor = platter top
    hub_r: float = info(0.032)
    hub_z1: float = info(0.078)
    bay_r: float = info(0.098)             # bay centre radius (drum-local +x)
    bay_in: float = info(0.042)            # bay inner HALF-width (interior 84 mm square)
    bay_t: float = info(0.008)
    bay_z1: float = info(0.078)            # bay wall top: roof_z0 - bay_z1 = 14 mm slot
    peg_r: float = info(0.011)
    peg_z1: float = info(0.073)
    peg_rad: float = info(0.132)
    drum_mass: float = info(1.5)
    drum_inertia: tuple = info((0.010, 0.010, 0.016))  # authored; Iz ~ platter 0.013 + parts
    # --- info: parcels / tiles / slots -----------------------------------------------------------
    parcel_size: float = info(0.055)
    parcel_mass: float = info(0.12)
    parcel_colors: tuple = info(((0.85, 0.10, 0.10), (0.10, 0.65, 0.15), (0.12, 0.30, 0.85)))
    color_names: tuple = info(("red", "green", "blue"))
    tile_side: float = info(0.070)
    tile_t: float = info(0.010)
    slot_r: float = info(0.24)             # apron slots on the slab, open half
    slot_az_deg: tuple = info((-50.0, 0.0, 50.0))
    park_xy: tuple = info(((1.7, 0.9), (1.7, -0.9)))  # off-field tile parking (world, per env)
    # --- info: materials -------------------------------------------------------------------------
    contact_offset: float = info(0.0015)
    drum_mu_s: float = info(0.70)          # grippy platter/bay: parcels ride the rotation
    drum_mu_d: float = info(0.60)
    base_mu_s: float = info(0.50)
    base_mu_d: float = info(0.45)
    # --- info: rubric weights (0.35 + 0.40 = 0.75 = the non-success cap) -------------------------
    w_load: float = info(0.35)
    w_prog: float = info(0.40)
    score_cap: float = info(0.75)

    def __post_init__(self) -> None:
        c = self
        # canopy interlock: the slot over the drum passes the LOADED bay but not a parcel
        slot = c.roof_z0 - c.bay_z1
        assert 0.010 <= slot < c.parcel_size - 0.020, \
            "canopy slot must clear the bay walls yet reject a parcel"
        assert c.roof_z0 - c.peg_z1 >= 0.015, "pegs must pass under the canopy"
        assert c.roof_z0 - (c.platter_z1 + c.parcel_size) >= 0.015, \
            "a parcel seated in the bay must pass under the canopy"
        # the parcel sits BELOW the bay wall top (fully contained, cannot be scraped off)
        assert c.platter_z1 + c.parcel_size <= c.bay_z1 - 0.004, \
            "seated parcel must ride below the bay wall top"
        # bay geometry: holds one parcel with slack, sits fully on the platter
        assert 2 * c.bay_in >= c.parcel_size + 0.020, "bay must accept the parcel with slack"
        bay_reach = math.hypot(c.bay_r + c.bay_in + c.bay_t, c.bay_in + c.bay_t)
        assert bay_reach <= c.platter_r - 0.002, "bay must sit fully on the platter"
        # rotation-invariant sweep clearances (do not trust joint collision filtering)
        sweep = max(c.platter_r, bay_reach, c.peg_rad + c.peg_r)
        assert c.wall_r_in - sweep >= 0.010, "drum sweep must clear the arc wall"
        assert c.pillar_y0 - sweep >= 0.010, "drum sweep must clear the pillars"
        assert c.platter_z0 >= 0.004, "platter must clear the slab"
        # gaps around the drum are all thinner than a parcel (no smuggling routes)
        assert c.wall_r_in - c.platter_r < c.parcel_size, "wall gap must reject a parcel"
        assert c.pillar_y0 - c.platter_r < c.parcel_size, "pillar gap must reject a parcel"
        # the whole delivery sector lies under the canopy
        x_edge = c.bay_r * math.cos(math.pi - c.sector_tol) + (c.bay_in + c.bay_t)
        assert x_edge < c.roof_x1 - 0.010, "delivery sector must sit fully under the canopy"
        # theta0 keeps the bay (and its angular extent) in the OPEN half
        half_ext = math.atan2(c.bay_in + c.bay_t, c.bay_r)
        assert math.radians(c.theta0_deg) + half_ext < math.pi / 2 - 0.15, \
            "initial bay must sit fully in the open half"
        assert math.pi - math.radians(c.theta0_deg) > c.sector_tol + 0.5, \
            "progress denominator must dominate the sector"
        # apron slots: on the slab, clear of the sweep, clear of the pedestal
        diag = c.parcel_size * math.sqrt(2) / 2
        assert c.slot_r - diag - c.parcel_jitter > sweep + 0.010, \
            "apron slots must clear the drum sweep"
        assert c.slot_r + diag + c.parcel_jitter < c.slab_half, "slots must stay on the slab"
        px, py = c.ped_center
        for az in c.slot_az_deg:
            a = math.radians(az)
            d = math.hypot(c.slot_r * math.cos(a) - px, c.slot_r * math.sin(a) - py)
            assert d > c.ped_half * math.sqrt(2) + diag + c.parcel_jitter + 0.010, \
                "slots must clear the pedestal"
        assert math.hypot(px, py) - c.ped_half * math.sqrt(2) > c.wall_r_in + c.wall_t + 0.010, \
            "pedestal must clear the arc wall"
        # plant discrete stability
        dt = 1.0 / 120.0
        iz = c.drum_inertia[2]
        assert c.drum_b * dt / iz < 0.5, "plant damping must be discretely stable"
        assert c.tau_max / iz * dt < 1.0, "per-step torque kick must stay bounded"
        assert abs(c.w_load + c.w_prog - c.score_cap) < 1e-9


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qconj(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("transfer_carousel")
class TransferCarouselScene(BaseScene):
    cfg: TransferCarouselSceneCfg

    def __init__(self, cfg: TransferCarouselSceneCfg | None = None) -> None:
        super().__init__(cfg or TransferCarouselSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        station_spawn = cls["station"](
            slab_half=c.slab_half, slab_t=c.slab_t, wall_r_in=c.wall_r_in,
            wall_t=c.wall_t, wall_z1=c.wall_z1, wall_seg=c.wall_seg,
            roof_x0=c.roof_x0, roof_x1=c.roof_x1, roof_y=c.roof_y,
            roof_z0=c.roof_z0, roof_z1=c.roof_z1,
            pillar_x0=c.pillar_x0, pillar_x1=c.pillar_x1,
            pillar_y0=c.pillar_y0, pillar_y1=c.pillar_y1,
            ped_center=c.ped_center, ped_half=c.ped_half, ped_h=c.ped_h,
            station_mass=c.station_mass, station_inertia=c.station_inertia,
            contact_offset=c.contact_offset,
            base_mu_s=c.base_mu_s, base_mu_d=c.base_mu_d)
        drum_spawn = cls["drum"](
            platter_r=c.platter_r, platter_z0=c.platter_z0, platter_z1=c.platter_z1,
            hub_r=c.hub_r, hub_z1=c.hub_z1, bay_r=c.bay_r, bay_in=c.bay_in,
            bay_t=c.bay_t, bay_z1=c.bay_z1, peg_r=c.peg_r, peg_z1=c.peg_z1,
            peg_rad=c.peg_rad, drum_mass=c.drum_mass, drum_inertia=c.drum_inertia,
            contact_offset=c.contact_offset,
            drum_mu_s=c.drum_mu_s, drum_mu_d=c.drum_mu_d)

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.20, angular_damping=0.20,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=1)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "station": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Station", spawn=station_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.base_z))),
            "drum": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Drum", spawn=drum_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.base_z))),
        }
        for k in range(3):
            out[f"parcel_{k}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Parcel_{k}",
                spawn=sim_utils.CuboidCfg(
                    size=(c.parcel_size, c.parcel_size, c.parcel_size),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.parcel_mass),
                    rigid_props=rigid, collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.50, dynamic_friction=0.45, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.parcel_colors[k])),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2 + 0.2 * k, 0.8, 0.05)))
            out[f"tile_{k}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Tile_{k}",
                spawn=sim_utils.CuboidCfg(
                    size=(c.tile_side, c.tile_side, c.tile_t),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.parcel_colors[k])),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2 + 0.2 * k, -0.8, 0.02)))
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                # Without this, external wrenches are under-applied across TGS
                # iterations and the drum plant stalls far below equilibrium.
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
        self.station: RigidObject = env.iscene["station"]
        self.drum: RigidObject = env.iscene["drum"]
        self.parcels: list[RigidObject] = [env.iscene[f"parcel_{k}"] for k in range(3)]
        self.tiles: list[RigidObject] = [env.iscene[f"tile_{k}"] for k in range(3)]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        n = env.num_envs
        dev = env.device
        # episode readbacks (verified by smoke)
        self.target_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.slot_of = torch.zeros(n, 3, dtype=torch.long, device=dev)  # parcel k -> slot
        self.theta0 = torch.zeros(n, device=dev)
        self.dref = torch.full((n,), math.pi, device=dev)  # progress denominator
        # drive plant state (post_step OWNS the drum's external-wrench slot; solve/smoke
        # write drum_drive only — never call set_external_force_and_torque on the drum)
        self.drum_drive = torch.zeros(n, device=dev)   # commanded torque (N*m, clamped)
        self.rate_fd = torch.zeros(n, device=dev)      # FD hinge rate (root_ang_vel_w is
        self._theta_prev = torch.zeros(n, device=dev)  # phantom under external wrenches)
        # latches (partial credit survives transients; success is judged live)
        self._load = torch.zeros(n, dtype=torch.bool, device=dev)
        self._prog = torch.zeros(n, device=dev)

    def _author_joints(self) -> None:
        """Per env: an unlimited Z-axis revolute joint station->drum at the shared pivot.
        Body0 is the heavy DYNAMIC station root so the anchor follows reset teleports
        (a kinematic body0 anchor stays world-fixed at the spawn pose on this stack).
        Joint-pair collision keeps its default (filtered); the drum's clearances to the
        station are geometric and rotation-invariant, so neither contact nor filtering
        between the pair is ever load-bearing."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/drum_pivot")
            j.CreateBody0Rel().SetTargets([f"{base}/Station"])
            j.CreateBody1Rel().SetTargets([f"{base}/Drum"])
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the station (free yaw + xy jitter) and the drum together
        (whole-linkage write, angle theta0), permute the three parcels over the apron
        slots, sample the target colour, stand its tile on the pedestal (others parked
        off-field), sync the FD rate reference, clear drive and latches. All discrete
        draws derive from torch.rand — the first torch.randint after manual_seed is
        degenerate on this stack."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        from isaaclab.utils.math import quat_apply

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        q_h = _qz(yaw)
        dp = torch.zeros(m, 3, device=dev)
        dp[:, 0] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        dp[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        dp[:, 2] = c.base_z
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = dp + origin
        st[:, 3:7] = q_h
        self.station.write_root_state_to_sim(st, env_ids)

        # drum: SAME pivot, relative angle theta0 (write the whole linkage together —
        # teleporting one body of a joint pair gets depenetrated back by the other)
        theta0 = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.theta0_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = dp + origin
        st[:, 3:7] = _qmul(q_h, _qz(theta0))
        self.drum.write_root_state_to_sim(st, env_ids)
        self.theta0[env_ids] = theta0
        self._theta_prev[env_ids] = theta0
        self.dref[env_ids] = (math.pi - theta0.abs()).clamp(min=c.sector_tol + 0.3)

        # target colour + colour->slot permutation (rand-derived, readback-verified)
        tgt = (torch.rand(m, device=dev) * 3).long().clamp(max=2)
        if not c.randomize_target:
            tgt = torch.zeros(m, dtype=torch.long, device=dev)
        self.target_idx[env_ids] = tgt
        if c.randomize_slots:
            perm = torch.argsort(torch.rand(m, 3, device=dev), dim=1)
        else:
            perm = torch.arange(3, device=dev).expand(m, 3).contiguous()
        self.slot_of[env_ids] = perm

        az_all = torch.tensor([math.radians(a) for a in c.slot_az_deg], device=dev)
        for k in range(3):
            az = az_all[perm[:, k]]
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = c.slot_r * torch.cos(az) \
                + (torch.rand(m, device=dev) * 2 - 1) * c.parcel_jitter
            loc[:, 1] = c.slot_r * torch.sin(az) \
                + (torch.rand(m, device=dev) * 2 - 1) * c.parcel_jitter
            loc[:, 2] = c.parcel_size / 2 + 0.003
            yaw_p = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            s = torch.zeros(m, 13, device=dev)
            s[:, 0:3] = dp + origin + quat_apply(q_h, loc)
            s[:, 3:7] = _qmul(q_h, _qz(yaw_p))
            self.parcels[k].write_root_state_to_sim(s, env_ids)

        # tiles: the target's tile stands on the pedestal; the others park off-field
        px, py = c.ped_center
        ped = torch.tensor([px, py, c.ped_h + c.tile_t / 2 + 0.001], device=dev)
        for k in range(3):
            s = torch.zeros(m, 13, device=dev)
            on_ped = tgt == k
            loc = ped.expand(m, 3)
            s[:, 0:3] = dp + origin + quat_apply(q_h, loc)
            s[:, 3:7] = q_h
            park_k = c.park_xy[k % 2]
            park = torch.tensor([park_k[0], park_k[1], c.tile_t / 2 + 0.002], device=dev)
            s[:, 0:3] = torch.where(on_ped.unsqueeze(-1), s[:, 0:3], origin + park)
            s[:, 3] = torch.where(on_ped, s[:, 3], torch.ones(m, device=dev))
            s[:, 4:7] = torch.where(on_ped.unsqueeze(-1), s[:, 4:7],
                                    torch.zeros(m, 3, device=dev))
            self.tiles[k].write_root_state_to_sim(s, env_ids)

        self.drum_drive[env_ids] = 0.0
        self.rate_fd[env_ids] = 0.0
        self._load[env_ids] = False
        self._prog[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "station": self.station.data.root_state_w[env_ids].clone(),
            "drum": self.drum.data.root_state_w[env_ids].clone(),
            "target_idx": self.target_idx[env_ids].clone(),
            "slot_of": self.slot_of[env_ids].clone(),
            "theta0": self.theta0[env_ids].clone(),
            "dref": self.dref[env_ids].clone(),
            "drive": self.drum_drive[env_ids].clone(),
            "rate_fd": self.rate_fd[env_ids].clone(),
            "theta_prev": self._theta_prev[env_ids].clone(),
            "load": self._load[env_ids].clone(),
            "prog": self._prog[env_ids].clone(),
        }
        for k in range(3):
            out[f"parcel_{k}"] = self.parcels[k].data.root_state_w[env_ids].clone()
            out[f"tile_{k}"] = self.tiles[k].data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.station.write_root_state_to_sim(state["station"], env_ids)
        self.drum.write_root_state_to_sim(state["drum"], env_ids)
        for k in range(3):
            self.parcels[k].write_root_state_to_sim(state[f"parcel_{k}"], env_ids)
            self.tiles[k].write_root_state_to_sim(state[f"tile_{k}"], env_ids)
        self.target_idx[env_ids] = state["target_idx"]
        self.slot_of[env_ids] = state["slot_of"]
        self.theta0[env_ids] = state["theta0"]
        self.dref[env_ids] = state["dref"]
        self.drum_drive[env_ids] = state["drive"]
        self.rate_fd[env_ids] = state["rate_fd"]
        self._theta_prev[env_ids] = state["theta_prev"]
        self._load[env_ids] = state["load"]
        self._prog[env_ids] = state["prog"]

    def resync_rate(self, env_ids: torch.Tensor | None = None) -> None:
        """Re-anchor the FD rate reference to the CURRENT drum angle (call after any
        manual drum teleport, once the sim buffers reflect it)."""
        th = self.theta()
        if env_ids is None:
            self._theta_prev[:] = th
            self.rate_fd[:] = 0.0
        else:
            self._theta_prev[env_ids] = th[env_ids]
            self.rate_fd[env_ids] = 0.0

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A TRANSFER STATION stands on a heavy square counter slab. Its far half is a "
            f"SEALED delivery alcove: a slate-blue arc wall around the back, side pillars, "
            f"and a canopy roof over the whole far half — there is no way to reach, drop, "
            f"or slide anything into the alcove from outside; smuggling gaps do not exist. "
            f"Set into the counter is a rotating PASS-THROUGH CAROUSEL: a grey platter "
            f"drum on a free vertical pivot, carrying one open-top BAY (a dark walled "
            f"pocket) and three YELLOW PEGS for pushing it around. The canopy clears the "
            f"drum by a {1000 * (c.roof_z0 - c.bay_z1):.0f} mm slot: the loaded bay "
            f"rotates through into the alcove, but a parcel "
            f"({1000 * c.parcel_size:.0f} mm cube) can NEVER be put into the bay while "
            f"the bay is under the canopy — loading is only possible on the open side.\n"
            f"Three parcels — one red, one green, one blue — rest on the open half of the "
            f"counter. The COLOUR TILE standing on the corner pedestal names the ONE "
            f"parcel to transfer (which colour, and which parcel sits where, varies by "
            f"episode; the drum's starting angle and the whole station's heading vary "
            f"too).\n"
            f"Goal: put the TILE-COLOURED parcel into the drum's bay while the bay faces "
            f"the open side, then rotate the carousel by its yellow pegs — roughly half a "
            f"turn — until the loaded bay sits centred inside the sealed alcove (within "
            f"about {math.degrees(c.sector_tol):.0f} degrees). Leave the other two "
            f"parcels alone: a wrong parcel riding the bay at the end fails the episode. "
            f"The episode ends settled: drum at rest with the loaded bay in the alcove, "
            f"parcels at rest, nothing held. The order is physically forced — an empty "
            f"bay rotated into the alcove cannot be loaded there, and a parcel carried "
            f"over the canopy just lands on the roof."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Read the colour tile on the corner pedestal, place the matching parcel "
            "into the carousel's open bay on the open side of the counter, then push "
            "the yellow pegs to rotate the carousel half a turn until the loaded bay "
            "sits inside the sealed alcove, and let everything come to rest."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def theta(self) -> torch.Tensor:
        """(N,) drum angle relative to the station (rad, wrapped): yaw of the relative
        quaternion. 0 = bay at the station's open +x; pi = bay centred in the alcove."""
        q = _qmul(_qconj(self.station.data.root_quat_w), self.drum.data.root_quat_w)
        w, x, y, z = q.unbind(-1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _drum_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.drum.data.root_quat_w,
                                  pos_w - self.drum.data.root_pos_w)

    def in_bay(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool: a parcel-centre world position seated in the bay (drum frame):
        centred on the bay, resting on the bay floor."""
        c = self.cfg
        loc = self._drum_local(pos_w)
        rest_z = c.platter_z1 + c.parcel_size / 2
        return ((loc[:, 0] - c.bay_r).abs() < c.bay_xy_tol) \
            & (loc[:, 1].abs() < c.bay_xy_tol) \
            & ((loc[:, 2] - rest_z).abs() < c.bay_z_tol)

    def _bay_flags(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,) target-in-bay and (N,) any-distractor-in-bay."""
        inb = torch.stack([self.in_bay(p.data.root_pos_w) for p in self.parcels], dim=1)
        onehot = torch.nn.functional.one_hot(self.target_idx, 3).bool()
        tgt_in = (inb & onehot).any(dim=1)
        dis_in = (inb & ~onehot).any(dim=1)
        return tgt_in, dis_in

    def sector_err(self) -> torch.Tensor:
        """(N,) |wrap(theta - pi)|: angular distance of the bay from alcove centre."""
        return _wrap(self.theta() - math.pi).abs()

    def settled(self) -> torch.Tensor:
        """(N,) bool: parcels + station slow, drum FD rate slow."""
        c = self.cfg
        ok = self.station.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        for p in self.parcels:
            ok = ok & (p.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
        return ok & (self.rate_fd.abs() < c.settle_omega)

    def _finite(self) -> torch.Tensor:
        ps = [self.station.data.root_pos_w, self.drum.data.root_pos_w] \
            + [p.data.root_pos_w for p in self.parcels]
        return torch.isfinite(torch.stack(ps, dim=1)).all(dim=-1).all(dim=-1)

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Drum plant: clamped external drive torque + viscous damping on the FD hinge
        rate, applied as a BODY-frame z torque (the drum's z axis stays world-vertical,
        so this is immune to the stack's wrench frame drag); then latch rubric credit.
        Owns the drum's external-wrench slot."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        dt = self.env.dt

        th = self.theta()
        fin_th = torch.isfinite(th)
        raw = _wrap(th - self._theta_prev) / dt
        self.rate_fd = torch.where(
            fin_th, raw.clamp(-c.rate_clamp, c.rate_clamp), torch.zeros_like(raw))
        self._theta_prev = torch.where(fin_th, th, self._theta_prev)

        tau = self.drum_drive.clamp(-c.tau_max, c.tau_max) - c.drum_b * self.rate_fd
        tau = torch.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
        tq = torch.zeros(n, 1, 3, device=dev)
        tq[:, 0, 2] = tau
        self.drum.set_external_force_and_torque(torch.zeros(n, 1, 3, device=dev), tq)

        self._update_latches(th)

    def _update_latches(self, th: torch.Tensor | None = None) -> None:
        c = self.cfg
        if th is None:
            th = self.theta()
        fin = self._finite()
        tgt_in, _ = self._bay_flags()
        self._load |= tgt_in & fin
        # progress is ABOARD-gated: rotating the empty drum earns nothing
        p = (1.0 - _wrap(th - math.pi).abs() / self.dref).clamp(0.0, 1.0)
        gate = tgt_in & fin & torch.isfinite(p)
        self._prog = torch.where(gate, torch.maximum(self._prog, p), self._prog)

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the TARGET parcel seated in the bay, the bay inside the delivery
        sector under the sealed canopy, no distractor in the bay, drum at rest,
        everything settled and finite — all live physical outcomes. The parcel got
        there only by riding the carousel: the sealed alcove physically excludes any
        direct placement (smoke proves this with settled constructs)."""
        self._update_latches()
        tgt_in, dis_in = self._bay_flags()
        return tgt_in & ~dis_in & (self.sector_err() < self.cfg.sector_tol) \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.35 target-ever-loaded (latched) + 0.40 * latched
        aboard-gated rotation progress, capped at 0.75; exactly 1.0 iff success()
        holds live. Doing nothing scores ~0; spinning the empty drum scores ~0; the
        seed's strategy (carry the parcel to the destination) leaves it on the
        canopy roof and scores ~0."""
        c = self.cfg
        self._update_latches()
        base = (c.w_load * self._load.float() + c.w_prog * self._prog).clamp(max=c.score_cap)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="transfer_carousel", robot="null"))
