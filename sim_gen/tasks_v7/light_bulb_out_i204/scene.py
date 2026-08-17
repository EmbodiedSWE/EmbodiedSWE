"""FloodlightEjectScene — eject the dead bulb from a recessed floodlight through its
rear service port into the disposal bin (sim_gen task `light_bulb_out_i204`).

Derived from rlbench/light_bulb_out, but STRATEGICALLY different: the seed is a direct
removal — grasp the bulb sitting in the lamp's open socket, unscrew it with a wrist
rotation and carry it to a holder (one grasp ON the goal object, one continuous wrist
rotation, one free-space transfer). Here the bulb is NEVER grasped, NEVER screwed and
NEVER carried: it sits deep inside a floodlight SHROUD whose mouth is too narrow for a
parallel jaw around the bulb and far deeper than any finger, held behind a retention
RIDGE on the shroud floor. The only interface is a rod-sized SERVICE PORT through the
back of the housing: the solver must pick up a steel PUSH ROD from the floor, insert
it through the port's countersink and guide tube, and push the bulb over the ridge by
real contact. Past the ridge the shroud floor is a downhill RAMP: gravity rolls the
bulb out of the mouth and drops it into the green DISPOSAL BIN below — the fixture's
own geometry, not the arm, performs the delivery. A solver needs a different PLAN
from the seed (acquire a tool, thread it through a rear access, push the goal object
free and let gravity deliver it) and a different code structure (tool-tip port
window + ridge-crossing + ballistic transit latches, not an unscrew-and-place).

The denial is real geometry, not a scripted flag: the mouth (66 mm across) cannot
pass the 48 mm bulb plus two jaw fingers, the seat lies 128 mm inside (finger length
~54 mm), and a thin tool entering the mouth can only press the bulb INTO its seat —
asserted in `__post_init__`, force-probed in smoke. The ridge is a real quasi-static
threshold (~0.43 N for the 50 g bulb): sub-threshold forward nudges and inward
shoves both leave the bulb seated.

success(): the bulb rests INSIDE the disposal bin (bin-frame containment window that
by construction accepts every physically-in-bin resting pose and rejects beside-bin
and rim-perched poses — asserted) AND the bulb is settled.

score() is graded and latched (credit never evaporates): 0.15 rod tip ever through
the service port + 0.25 bulb ever pushed past the ridge inside the channel + 0.25
bulb ever flown through the mouth transit window = 0.65 cap; 1.0 iff success(). The
null policy scores ~0 (the bulb spawns seated behind the ridge, the rod on the
floor).

Per-episode randomization (readback-verified in smoke): housing position + yaw, rod
spawn side/xy/yaw, bulb seat-depth jitter, bin lateral jitter. Assets are fully
procedural compound spawners (boxes, one cylinder, one sphere; one rigid body each).
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


def _material(stage, path: str, static: float = 0.6, dynamic: float = 0.5):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults)."""
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
         orient=None) -> None:
    """One box child prim (translate -> [orient] -> scale, authored once — the
    duplicate-xformOp trap is avoided by never re-authoring an existing prim's ops)."""
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


def _cyl(stage, path: str, radius, height, center, color, contact_offset: float,
         material=None) -> None:
    """One z-axis cylinder child prim."""
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


def _sphere(stage, path: str, radius, center, color, contact_offset: float,
            material=None) -> None:
    """One sphere child prim."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Sphere.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -radius),
                          Gf.Vec3f(radius, radius, radius)])
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float,
                kinematic: bool = False):
    """One rigid-body root Xform with the standard physics armor (zero sleep /
    stabilization thresholds: a sleeping body silently ignores applied wrenches, which
    the solve/smoke force probes depend on; velocity iterations 4 — the
    sphere-on-face phantom-creep fix)."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return root, pxrb


def _spawn_housing(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC floodlight housing. Local frame: origin = mouth-aperture center
    at ground level, +x = ejection direction. Channel inner 0.066 (y) x 0.070 (z),
    seat floor top z 0.220, roof underside 0.290; retention ridge (top 0.226) at
    x -0.092; downhill ramp from (-0.088, 0.220) to the mouth (0.0, 0.188); back wall
    inner face x -0.160 with a 22 mm square port hole at axis height z 0.244; square
    guide tube (inner half 0.012) back to x -0.245 with two countersink collars
    (inner half 0.018 then 0.026) back to x -0.285. One rigid body."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 12.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    shell = (0.35, 0.36, 0.40)
    inner = (0.46, 0.47, 0.52)
    amber = (0.85, 0.55, 0.10)
    port = (0.20, 0.45, 0.75)
    # ramp: top surface from (-0.088, 0.220) to (0.0, 0.188); pitch-down about y
    theta = math.atan2(0.032, 0.088)
    ramp_len = math.hypot(0.088, 0.032)
    nx, nz = math.sin(theta), math.cos(theta)  # upward surface normal
    ramp_center = (-0.044 - 0.006 * nx, 0.0, 0.204 - 0.006 * nz)
    ramp_q = (math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0)
    boxes = [
        ("pedestal", (0.100, 0.100, 0.208), (-0.128, 0.0, 0.104), shell, None),
        ("seat_floor", (0.080, 0.066, 0.012), (-0.128, 0.0, 0.214), inner, None),
        ("ridge", (0.008, 0.066, 0.012), (-0.092, 0.0, 0.220), amber, None),
        ("ramp", (ramp_len, 0.066, 0.012), ramp_center, inner, ramp_q),
        ("roof", (0.172, 0.090, 0.012), (-0.086, 0.0, 0.296), shell, None),
        ("side_py", (0.172, 0.012, 0.132), (-0.086, +0.039, 0.236), shell, None),
        ("side_ny", (0.172, 0.012, 0.132), (-0.086, -0.039, 0.236), shell, None),
        # back wall (port hole: |y| <= 0.011, z 0.233..0.255)
        ("back_bot", (0.012, 0.090, 0.063), (-0.166, 0.0, 0.2015), shell, None),
        ("back_top", (0.012, 0.090, 0.047), (-0.166, 0.0, 0.2785), shell, None),
        ("back_py", (0.012, 0.034, 0.022), (-0.166, +0.028, 0.244), shell, None),
        ("back_ny", (0.012, 0.034, 0.022), (-0.166, -0.028, 0.244), shell, None),
        # guide tube (inner half 0.012), x -0.245..-0.166
        ("tube_bot", (0.079, 0.048, 0.008), (-0.2055, 0.0, 0.228), port, None),
        ("tube_top", (0.079, 0.048, 0.008), (-0.2055, 0.0, 0.260), port, None),
        ("tube_py", (0.079, 0.008, 0.024), (-0.2055, +0.016, 0.244), port, None),
        ("tube_ny", (0.079, 0.008, 0.024), (-0.2055, -0.016, 0.244), port, None),
        # countersink collar B (inner half 0.018), x -0.265..-0.245
        ("cb_bot", (0.020, 0.060, 0.008), (-0.255, 0.0, 0.222), port, None),
        ("cb_top", (0.020, 0.060, 0.008), (-0.255, 0.0, 0.266), port, None),
        ("cb_py", (0.020, 0.008, 0.036), (-0.255, +0.022, 0.244), port, None),
        ("cb_ny", (0.020, 0.008, 0.036), (-0.255, -0.022, 0.244), port, None),
        # countersink collar A (inner half 0.026), x -0.285..-0.265
        ("ca_bot", (0.020, 0.076, 0.008), (-0.275, 0.0, 0.214), port, None),
        ("ca_top", (0.020, 0.076, 0.008), (-0.275, 0.0, 0.274), port, None),
        ("ca_py", (0.020, 0.008, 0.052), (-0.275, +0.030, 0.244), port, None),
        ("ca_ny", (0.020, 0.008, 0.052), (-0.275, -0.030, 0.244), port, None),
    ]
    for name, size, center, col, q in boxes:
        _box(stage, f"{prim_path}/{name}", size, center, col, co, material=mat,
             orient=q)
    return root


def _spawn_bin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC disposal bin: floor + 4 walls, open top. Origin = footprint
    center at ground level. Interior |x| < 0.118, |y| < 0.088, floor top 0.016,
    rim 0.146."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 2.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat", static=0.8, dynamic=0.7)
    co = cfg.contact_offset
    green = (0.15, 0.45, 0.20)
    for name, size, center in (
        ("floor", (0.260, 0.200, 0.016), (0.0, 0.0, 0.008)),
        ("wall_px", (0.012, 0.200, 0.130), (+0.124, 0.0, 0.081)),
        ("wall_nx", (0.012, 0.200, 0.130), (-0.124, 0.0, 0.081)),
        ("wall_py", (0.236, 0.012, 0.130), (0.0, +0.094, 0.081)),
        ("wall_ny", (0.236, 0.012, 0.130), (0.0, -0.094, 0.081)),
    ):
        _box(stage, f"{prim_path}/{name}", size, center, green, co, material=mat)
    return root


def _spawn_bulb(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The dead bulb: one frosted glass globe (a pure sphere — it must ROLL down the
    ramp; authored solid-sphere inertia). Origin = sphere center."""
    import omni.usd

    from pxr import Gf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    i_s = 0.4 * cfg.mass * cfg.radius * cfg.radius  # solid sphere, auditable
    mass = UsdPhysics.MassAPI(root)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(i_s, i_s, i_s))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    mat = _material(stage, f"{prim_path}/phys_mat")
    _sphere(stage, f"{prim_path}/globe", cfg.radius, (0.0, 0.0, 0.0),
            (0.95, 0.93, 0.85), cfg.contact_offset, material=mat)
    return root


def _spawn_rod(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The steel push rod: one red cylinder along local +z (length `length`, radius
    `radius`); authored diagonal inertia so the solve's wrench servo gains are
    auditable. Origin = rod center."""
    import omni.usd

    from pxr import Gf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    i_t = cfg.mass * cfg.length * cfg.length / 12.0
    i_a = 0.5 * cfg.mass * cfg.radius * cfg.radius
    mass = UsdPhysics.MassAPI(root)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(i_t, i_t, i_a))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    mat = _material(stage, f"{prim_path}/phys_mat", static=0.4, dynamic=0.35)
    _cyl(stage, f"{prim_path}/shaft", cfg.radius, cfg.length, (0.0, 0.0, 0.0),
         (0.85, 0.15, 0.12), cfg.contact_offset, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "housing" not in _SPAWNER_CACHE:

        @configclass
        class HousingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_housing)
            contact_offset: float = 0.001

        @configclass
        class BinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bin)
            contact_offset: float = 0.001

        @configclass
        class BulbSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bulb)
            mass: float = 0.05
            radius: float = 0.024
            contact_offset: float = 0.001

        @configclass
        class RodSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rod)
            mass: float = 0.06
            radius: float = 0.006
            length: float = 0.30
            contact_offset: float = 0.001

        _SPAWNER_CACHE.update(housing=HousingSpawnerCfg, bin=BinSpawnerCfg,
                              bulb=BulbSpawnerCfg, rod=RodSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class FloodlightEjectSceneCfg(BaseCfg):
    """Config for `FloodlightEjectScene`. Every honesty claim is asserted in
    `__post_init__`: the mouth really denies the jaw, the seat is really beyond the
    fingers, the ridge is a real quasi-static threshold below the solve's force cap,
    the rod really passes the port while the bulb really passes the mouth, the
    ejected bulb really clears the bin's near rim, and the bin containment window
    accepts every physically-in-bin resting pose while rejecting beside/perched."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    bin_xy_tol_x: float = tunable(0.098)  # |bulb x|, bin frame (interior 0.118 - r=0.094)
    bin_xy_tol_y: float = tunable(0.068)  # |bulb y|, bin frame (interior 0.088 - r=0.064)
    bin_z_lo: float = tunable(0.028)  # bulb center height window, bin frame: rests
    bin_z_hi: float = tunable(0.075)  # ... at 0.040; a rim perch reads 0.170
    settle_lin: float = tunable(0.08)  # max bulb |lin vel| at judging (m/s)
    settle_ang: float = tunable(4.0)  # max bulb |ang vel| at judging (rad/s)
    probe_x: float = tunable(-0.168)  # rod tip past this (housing x) = through the port
    probe_ry: float = tunable(0.020)  # ... within this of the port axis (y)
    probe_rz: float = tunable(0.030)  # ... within this of the port axis (z)
    unseat_x: float = tunable(-0.078)  # bulb center past this inside the channel = unseated
    mouth_win: tuple = tunable((0.005, 0.065))  # eject transit window, housing x
    chan_half_y: float = tunable(0.040)  # channel-band |y| for unseat/eject windows
    chan_z: tuple = tunable((0.175, 0.300))  # channel-band z for unseat window
    mouth_z: tuple = tunable((0.170, 0.300))  # transit-band z for eject window

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    hx_range: tuple = tunable((0.68, 0.76))  # housing (mouth) x position
    hy_range: tuple = tunable((-0.08, 0.08))  # housing y position
    hyaw_deg: float = tunable(20.0)  # +- housing yaw
    rod_zone: tuple = tunable((0.30, 0.22))  # |center| of the rod spawn zones (both y signs)
    rod_jitter: float = tunable(0.06)  # +- xy jitter of the rod spawn
    seat_jitter: float = tunable(0.004)  # +- bulb seat-depth (local x) jitter
    bin_jitter: float = tunable(0.008)  # +- bin lateral (local y) jitter

    # --- info: structure (the geometry the spawners author) ----------------------------------
    bulb_r: float = info(0.024)
    bulb_mass: float = info(0.05)
    rod_r: float = info(0.006)
    rod_len: float = info(0.30)
    rod_mass: float = info(0.06)
    chan_half_w: float = info(0.033)  # channel inner half-width
    seat_floor_z: float = info(0.220)  # seat floor top (local z)
    axis_z: float = info(0.244)  # bulb/port axis height (local z)
    roof_bot: float = info(0.290)
    back_inner_x: float = info(-0.160)  # back wall inner face
    hole_half: float = info(0.011)  # port hole half-aperture in the back wall
    tube_half: float = info(0.012)  # guide tube inner half-aperture
    tube_back_x: float = info(-0.245)  # guide tube rear end (collars behind to -0.285)
    port_entry_x: float = info(-0.285)  # countersink entry plane
    ridge_x: float = info(-0.092)  # ridge center (faces -0.096 / -0.088)
    ridge_h: float = info(0.006)  # ridge top above the seat floor
    seat_x: float = info(-0.128)  # nominal seated bulb center (local x)
    ramp_end_z: float = info(0.188)  # ramp top surface at the mouth (local z)
    mouth_x: float = info(0.0)
    bin_off_x: float = info(0.145)  # bin center, housing frame
    bin_inner_x: float = info(0.118)  # bin interior half-extents / heights
    bin_inner_y: float = info(0.088)
    bin_floor_top: float = info(0.016)
    bin_rim_z: float = info(0.146)
    finger_t: float = info(0.018)  # Franka finger pad thickness (embodiment denial)
    finger_len: float = info(0.054)  # Franka finger length (embodiment denial)
    contact_offset: float = info(0.001)

    def __post_init__(self) -> None:
        c = self
        bulb_d = 2 * c.bulb_r
        g = 9.81
        # -- the mouth really denies the jaw; the seat is beyond the fingers --
        assert 2 * c.chan_half_w < bulb_d + 2 * c.finger_t, \
            "mouth must not admit the bulb plus two jaw fingers"
        assert -c.seat_x - c.bulb_r > c.finger_len + 0.03, \
            "seat must lie far beyond finger reach through the mouth"
        # -- the ridge is a real quasi-static threshold, inside the solve's budget --
        f_ridge = c.bulb_mass * g * math.sqrt(
            2 * c.bulb_r * c.ridge_h - c.ridge_h ** 2) / (c.bulb_r - c.ridge_h)
        assert 0.30 < f_ridge < 0.60, f"ridge threshold {f_ridge:.2f} N out of band"
        assert c.ridge_h < c.bulb_r / 2, "ridge must be climbable by the sphere"
        # -- the rod really fits the port chain; the bulb never does --
        assert c.rod_r < c.hole_half - 0.003 < c.tube_half, "rod/hole/tube must nest"
        assert bulb_d > 2 * c.tube_half and bulb_d > 2 * c.hole_half, \
            "the bulb must never pass the port"
        # -- the rod is long enough to push the bulb past the ridge from outside --
        stroke = (c.ridge_x + 0.012 + c.bulb_r) - (-0.088 - c.bulb_r)  # spare
        assert c.rod_len > (c.port_entry_x * -1) - 0.088 + 0.06 + stroke * 0, \
            "rod must span entry-to-ridge with grasp margin"
        # -- the bulb really passes the pocket, channel and mouth --
        assert 2 * c.chan_half_w > bulb_d + 0.010, "channel must pass the bulb"
        assert c.roof_bot - c.seat_floor_z > bulb_d + 0.010, "headroom over the seat"
        assert c.roof_bot - (c.seat_floor_z + c.ridge_h) > bulb_d + 0.010, \
            "headroom over the ridge crest"
        assert (c.back_inner_x * -1) - 0.088 > bulb_d + 0.010, "pocket must hold the bulb"
        # -- the ejected bulb clears the bin's near rim --
        assert c.ramp_end_z > c.bin_rim_z + 0.020, \
            "mouth floor must overfly the bin rim"
        near_rim_x = c.bin_off_x - c.bin_inner_x - 0.012
        assert near_rim_x > c.mouth_x, "bin near wall must sit past the mouth plane"
        # -- bin containment honesty --
        rest_z = c.bin_floor_top + c.bulb_r
        assert c.bin_z_lo < rest_z < c.bin_z_hi, "resting bulb must be inside the window"
        perch_z = c.bin_rim_z + c.bulb_r
        assert perch_z > c.bin_z_hi + 0.05, "a rim-perched bulb must be above the window"
        assert c.bin_inner_x - c.bulb_r <= c.bin_xy_tol_x < c.bin_inner_x + 0.012 + c.bulb_r, \
            "x window: accept all in-bin rests, reject beside-the-wall"
        assert c.bin_inner_y - c.bulb_r <= c.bin_xy_tol_y < c.bin_inner_y + 0.012 + c.bulb_r, \
            "y window: accept all in-bin rests, reject beside-the-wall"
        # -- latch windows sit on the demonstrated path --
        assert c.unseat_x > -0.088 and c.unseat_x < c.mouth_win[0], \
            "unseat window must lie between ridge and mouth"
        assert c.mouth_z[0] < c.ramp_end_z + c.bulb_r < c.mouth_z[1], \
            "the rolling bulb's center must cross the transit window"
        assert c.probe_x > c.back_inner_x - 0.012 and c.probe_x < c.seat_x - c.bulb_r - 0.006, \
            "probe window must fire inside the port, before bulb contact"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("floodlight_eject")
class FloodlightEjectScene(BaseScene):
    cfg: FloodlightEjectSceneCfg

    def __init__(self, cfg: FloodlightEjectSceneCfg | None = None) -> None:
        super().__init__(cfg or FloodlightEjectSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        hx0 = 0.5 * (c.hx_range[0] + c.hx_range[1])

        return {
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
            "housing": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Housing",
                spawn=sp["housing"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=12.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx0, 0.0, 0.0)),
            ),
            "bin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bin",
                spawn=sp["bin"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hx0 + c.bin_off_x, 0.0, 0.0)),
            ),
            "bulb": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bulb",
                spawn=sp["bulb"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bulb_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.bulb_mass, radius=c.bulb_r,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hx0 + c.seat_x, 0.0, c.axis_z)),
            ),
            "rod": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rod",
                spawn=sp["rod"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.rod_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.rod_mass, radius=c.rod_r, length=c.rod_len,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rod_zone[0], c.rod_zone[1], c.rod_r + 0.003),
                    rot=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)),
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
                # external-wrench plant recipe: without this the solve/smoke wrench
                # servos on the rod are under-applied across TGS iterations
                "enable_external_forces_every_iteration": True,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.housing: RigidObject = env.iscene["housing"]
        self.bin: RigidObject = env.iscene["bin"]
        self.bulb: RigidObject = env.iscene["bulb"]
        self.rod: RigidObject = env.iscene["rod"]
        self.env_origins = env.iscene.env_origins
        # progress latches (post_step): rod through the port, bulb past the ridge,
        # bulb flown through the mouth transit window
        self.probe_latch = torch.zeros(n, device=dev)
        self.unseat_latch = torch.zeros(n, device=dev)
        self.eject_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: housing posed (xy + yaw sampled), bin riding the housing
        frame (+ lateral jitter), bulb SEATED in the pocket behind the ridge (depth
        jitter), rod lying flat on the floor in a sampled side zone (+ jitter + free
        yaw); latches zeroed. Discrete draws use torch.rand comparisons (the
        first-randint degeneracy)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # housing: sampled xy + yaw
        hx = c.hx_range[0] + torch.rand(m, device=dev) * (c.hx_range[1] - c.hx_range[0])
        hy = c.hy_range[0] + torch.rand(m, device=dev) * (c.hy_range[1] - c.hy_range[0])
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.hyaw_deg) / 2
        hq = torch.zeros(m, 4, device=dev)
        hq[:, 0] = torch.cos(half)
        hq[:, 3] = torch.sin(half)
        hp = torch.zeros(m, 3, device=dev)
        hp[:, 0], hp[:, 1] = hx, hy
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = hp + origin
        st[:, 3:7] = hq
        self.housing.write_root_state_to_sim(st, env_ids)

        # bin: housing frame + lateral jitter
        by = (torch.rand(m, device=dev) * 2 - 1) * c.bin_jitter
        off = torch.zeros(m, 3, device=dev)
        off[:, 0] = c.bin_off_x
        off[:, 1] = by
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = hp + quat_apply(hq, off) + origin
        st[:, 3:7] = hq
        self.bin.write_root_state_to_sim(st, env_ids)

        # bulb: seated in the pocket (depth jitter), resting on the seat floor
        sx = c.seat_x + (torch.rand(m, device=dev) * 2 - 1) * c.seat_jitter
        off = torch.zeros(m, 3, device=dev)
        off[:, 0] = sx
        off[:, 2] = c.seat_floor_z + c.bulb_r + 0.001
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = hp + quat_apply(hq, off) + origin
        st[:, 3] = 1.0
        self.bulb.write_root_state_to_sim(st, env_ids)

        # rod: flat on the floor, sampled side zone + jitter + free yaw
        side = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        rx = c.rod_zone[0] + (torch.rand(m, device=dev) * 2 - 1) * c.rod_jitter
        ry = side * c.rod_zone[1] + (torch.rand(m, device=dev) * 2 - 1) * c.rod_jitter
        ryaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        c45 = math.cos(math.pi / 4)
        rhalf = ryaw / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = rx
        st[:, 1] = ry
        st[:, 2] = c.rod_r + 0.003
        st[:, 3] = torch.cos(rhalf) * c45
        st[:, 4] = -torch.sin(rhalf) * c45
        st[:, 5] = torch.cos(rhalf) * c45
        st[:, 6] = torch.sin(rhalf) * c45
        st[:, 0:3] += origin
        self.rod.write_root_state_to_sim(st, env_ids)

        self.probe_latch[env_ids] = 0.0
        self.unseat_latch[env_ids] = 0.0
        self.eject_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"housing": self.housing, "bin": self.bin, "bulb": self.bulb,
                  "rod": self.rod}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "probe_latch": self.probe_latch[env_ids].clone(),
            "unseat_latch": self.unseat_latch[env_ids].clone(),
            "eject_latch": self.eject_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"housing": self.housing, "bin": self.bin, "bulb": self.bulb,
                  "rod": self.rod}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.probe_latch[env_ids] = state["probe_latch"]
        self.unseat_latch[env_ids] = state["unseat_latch"]
        self.eject_latch[env_ids] = state["eject_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A wall-style FLOODLIGHT stands on a grey pedestal on the floor: a deep "
            f"grey shroud whose open MOUTH ({2 * c.chan_half_w * 1000:.0f} mm wide, "
            "facing a green open-top DISPOSAL BIN on the floor below it). Deep "
            f"inside the shroud, {-c.seat_x * 1000:.0f} mm behind the mouth, a dead "
            f"frosted glass BULB ({2 * c.bulb_r * 1000:.0f} mm across) rests in a "
            "seat pocket behind a low AMBER RIDGE that runs across the shroud "
            "floor; past the ridge the floor slopes downhill to the mouth. The "
            "mouth is too narrow and the bulb too deep for any hand or gripper to "
            "reach it, and poking it from the front only presses it against the "
            "back of its seat. On the BACK of the housing, in line with the bulb, "
            "sits a BLUE SERVICE PORT: a square guide tube with a stepped "
            f"countersink, sized for the red steel PUSH ROD "
            f"({c.rod_len * 1000:.0f} mm long, {2 * c.rod_r * 1000:.0f} mm thick) "
            "lying on the floor nearby. The floodlight's spot and heading, the "
            "rod's spot and the bulb's exact seat depth change every episode — "
            "read them by looking.\n"
            "Goal: get the dead bulb OUT of the floodlight and resting INSIDE the "
            "disposal bin. Pick up the push rod, insert its tip through the rear "
            "service port, and push the bulb forward over the amber ridge; the "
            "sloped floor then rolls it out of the mouth and it falls into the bin "
            "below. Success when the bulb lies at rest inside the bin. A bulb "
            "left anywhere else — still seated, part-way along the shroud, on the "
            "floor beside the bin, or balanced on the bin's rim — does not count; "
            "the rod may be left anywhere. The bulb cannot pass the service port "
            "and the housing is bolted down; pushing from the rear port is the "
            "only way to free it."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Insert the red push rod through the floodlight's rear service port "
            "and push the dead bulb over the amber ridge so it rolls out of the "
            "mouth and falls into the green disposal bin. The bulb must end "
            "resting inside the bin."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def housing_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N, 3): world points -> housing body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.housing.data.root_quat_w,
                                  p_w - self.housing.data.root_pos_w)

    def bin_local(self) -> torch.Tensor:
        """(N, 3): bulb center in the bin body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.bin.data.root_quat_w,
                                  self.bulb.data.root_pos_w - self.bin.data.root_pos_w)

    def bulb_local(self) -> torch.Tensor:
        """(N, 3): bulb center in the housing body frame."""
        return self.housing_local(self.bulb.data.root_pos_w)

    def rod_tips_local(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Both rod end points in the housing frame, shapes (N, 3) each (the rod is
        symmetric — either end may be the working tip)."""
        from isaaclab.utils.math import quat_apply

        half = self.cfg.rod_len / 2
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        a = quat_apply(self.rod.data.root_quat_w, ez)
        p = self.rod.data.root_pos_w
        return self.housing_local(p + a * half), self.housing_local(p - a * half)

    def rod_in_port(self) -> torch.Tensor:
        """(N,) bool: either rod tip through the port hole into the channel axis
        band (past `probe_x`, on the port axis)."""
        c = self.cfg

        def hit(t: torch.Tensor) -> torch.Tensor:
            return (t[:, 0] > c.probe_x) & (t[:, 0] < c.seat_x) \
                & (t[:, 1].abs() < c.probe_ry) \
                & ((t[:, 2] - c.axis_z).abs() < c.probe_rz)

        t1, t2 = self.rod_tips_local()
        return hit(t1) | hit(t2)

    def bulb_unseated(self) -> torch.Tensor:
        """(N,) bool: bulb center past the ridge, inside the channel band."""
        c = self.cfg
        loc = self.bulb_local()
        return (loc[:, 0] > c.unseat_x) & (loc[:, 0] < c.mouth_win[1]) \
            & (loc[:, 1].abs() < c.chan_half_y) \
            & (loc[:, 2] > c.chan_z[0]) & (loc[:, 2] < c.chan_z[1])

    def bulb_in_transit(self) -> torch.Tensor:
        """(N,) bool: bulb center inside the mouth transit window (just outside the
        mouth plane, on the ballistic exit path — teleported constructions elsewhere
        never touch it)."""
        c = self.cfg
        loc = self.bulb_local()
        return (loc[:, 0] > c.mouth_win[0]) & (loc[:, 0] < c.mouth_win[1]) \
            & (loc[:, 1].abs() < c.chan_half_y) \
            & (loc[:, 2] > c.mouth_z[0]) & (loc[:, 2] < c.mouth_z[1])

    def in_bin(self) -> torch.Tensor:
        """(N,) bool, geometric: bulb inside the disposal bin — bin-frame containment
        (accepts every physically-in-bin resting pose, rejects beside-the-wall and
        rim-perched poses; asserted in __post_init__)."""
        c = self.cfg
        loc = self.bin_local()
        return (loc[:, 0].abs() <= c.bin_xy_tol_x) \
            & (loc[:, 1].abs() <= c.bin_xy_tol_y) \
            & (loc[:, 2] >= c.bin_z_lo) & (loc[:, 2] <= c.bin_z_hi)

    def settled(self) -> torch.Tensor:
        """(N,) bool: bulb at rest (thresholds sit above the GPU sphere-creep
        artifact; position windows + persistence carry the rest)."""
        c = self.cfg
        return (self.bulb.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.bulb.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch each demonstrated stage every physics substep: rod ever through the
        port, bulb ever past the ridge in the channel, bulb ever through the mouth
        transit window."""
        self.probe_latch = torch.maximum(self.probe_latch, self.rod_in_port().float())
        self.unseat_latch = torch.maximum(self.unseat_latch,
                                          self.bulb_unseated().float())
        self.eject_latch = torch.maximum(self.eject_latch,
                                         self.bulb_in_transit().float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the bulb rests inside the disposal bin, settled. The shroud
        geometry makes the chain (rod through the port -> push over the ridge ->
        gravity ejection) physically necessary."""
        return self.in_bin() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 rod ever through the port + 0.25 bulb ever past
        the ridge + 0.25 bulb ever through the mouth transit window (cap 0.65);
        1.0 iff success(). Latched — credit never evaporates; the null policy scores
        ~0 (the bulb spawns seated behind the ridge, the rod on the floor)."""
        base = (0.15 * self.probe_latch + 0.25 * self.unseat_latch
                + 0.25 * self.eject_latch).clamp(0.0, 0.65)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="floodlight_eject", robot="null",
                                      env_spacing=3.0))
