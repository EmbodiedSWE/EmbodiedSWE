"""LanternShelterScene — roll the oversized globe bulb into the lantern shelter, shove
it up onto its four-post seat, and slide the storm shutter closed behind it
(sim_gen task `light_bulb_in_i219`).

Derived from rlbench/light_bulb_in, but STRATEGICALLY different: the seed is a direct
install — grasp a bulb out of its holder, carry it through free space to the lamp,
insert it into the socket and screw it down (one grasp ON the goal object, one carried
transfer, one continuous wrist rotation). Here the goal object CANNOT BE GRASPED AT
ALL: the lantern's globe bulb is a 90 mm frosted sphere, wider than the Franka's 80 mm
parallel-jaw span (asserted in __post_init__), so the seed's grasp-carry-insert-screw
schema transfers zero. The lamp fixture is a roofed ground-level shelter whose only
opening is a floor-level doorway, so the bulb cannot be lowered in from above either —
it must be ROLLED across the floor, through the doorway, the whole way as non-prehensile
pushing. The socket is a cradle of four brass ball posts on the shelter floor: seating
the globe demands a measured SHOVE over a real ~10 mm gravity barrier (quasi-static
pushing to first contact and stopping does not seat it; the four-post geometry then
retains it behind a ~6.5 mm exit barrier, and the back wall caps overshoot so capture
is geometric). Finally a CAPTIVE SLIDING SHUTTER — a free slab riding a curb-and-rail
channel across the shelter front, no joint, no flag — must be slid across the doorway
to shutter the bulb in. A graspable 48 mm decoy bulb rests nearby: it is the WRONG
bulb — it falls straight through the post cradle (post diagonal 33.9 mm exceeds its
24+8 mm contact reach, asserted) and can never register as seated.

So the plan skeleton changes from *"grasp, carry, insert, screw"* to *"roll the
ungraspable object along the floor, thread it through a doorway, shove it over a
retention barrier onto a post seat, then close a sliding shutter"* — non-prehensile
whole-route transport plus a terminal closure stage, with the direct schema physically
rejected (jaw span) and the aerial shortcut denied (roof).

Everything is live physics: four rigid bodies (kinematic shelter; free globe, decoy
and shutter), no joints anywhere (the shutter is captured purely by curb/rail
geometry with 2 mm slack), authored friction/restitution/inertia, and every predicate
(globe in the chamber, globe seated on the posts, shutter covering the doorway) is
read from live poses in the shelter's body frame.

success(): the globe rests seated on the four posts (shelter-frame containment window
that accepts the seated pose at z ~ 48.7 mm and rejects a free-rolling globe at
z = 45 mm, the ground-resting decoy at z = 24 mm, and beside-seat poses — asserted)
AND the shutter covers the doorway (|slide offset| within tolerance, flush in its
channel, upright) AND globe + shutter are settled.

score() is graded and latched (credit never evaporates): 0.20 globe ever inside the
chamber + 0.30 globe ever seated on the posts + 0.25 shutter ever closed WHILE the
globe is seated = 0.75 cap; 1.0 iff success(). The null policy scores ~0 (the globe
and decoy spawn on the open floor outside, the shutter spawns fully open).

Per-episode randomization (readback-verified in smoke): shelter xy + yaw, shutter
open offset, globe spawn xy, decoy spawn side + xy. Assets are fully procedural
compound spawners (boxes and spheres; one rigid body each; decorations authored
idempotently). Heavy imports (isaaclab, pxr) are deferred so importing this module —
and registering the scene — stays app-free.
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


def _box(stage, path: str, size, center, color, contact_offset: float, material=None) -> None:
    """Author one box child prim (translate -> scale, authored once — the duplicate
    xformOp trap is avoided by never re-authoring an existing prim's ops)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
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


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float,
                kinematic: bool = False):
    """Author one rigid-body root Xform with the standard physics armor (zero
    sleep/stabilization thresholds: a sleeping body silently ignores applied wrenches,
    which the solve/smoke force probes depend on; velocity iterations 4 — the
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


def _spawn_shelter(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC lantern shelter: side/back walls + roof, a front wall with a
    floor-level doorway (two jambs + lintel), the four brass seat posts (spheres) on
    the floor inside, and the shutter channel hardware on the front face: two low
    curb segments, a top rail strip and two end stops. All numbers MUST match the
    LanternShelterSceneCfg info fields (asserted there). One rigid body; origin =
    seat center at ground level; the doorway faces local -x."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 10.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    wall, frame = (0.58, 0.62, 0.68), (0.30, 0.32, 0.38)
    hardware, brass = (0.22, 0.24, 0.30), (0.80, 0.62, 0.20)
    boxes = [
        # shelter shell (chamber interior x in [-0.060, 0.048], y in +-0.085)
        ("side_p", (0.132, 0.012, 0.135), (-0.006, +0.091, 0.0675), wall),
        ("side_n", (0.132, 0.012, 0.135), (-0.006, -0.091, 0.0675), wall),
        ("back", (0.012, 0.194, 0.135), (0.054, 0.0, 0.0675), wall),
        ("roof", (0.132, 0.194, 0.012), (-0.006, 0.0, 0.141), wall),
        # front wall (plane x in [-0.072, -0.060]): jambs beside the doorway, lintel
        # above it — the doorway itself is floor-level, 110 mm wide, 105 mm tall
        ("jamb_p", (0.012, 0.042, 0.135), (-0.066, +0.076, 0.0675), frame),
        ("jamb_n", (0.012, 0.042, 0.135), (-0.066, -0.076, 0.0675), frame),
        ("lintel", (0.012, 0.194, 0.030), (-0.066, 0.0, 0.120), frame),
        # shutter channel on the front face: floor curbs (outside the doorway span
        # in y — the doorway floor stays clear), top rail, two end stops. The
        # channel inner face sits at x = -0.082: 10 mm from the wall's outer face
        # for the 8 mm shutter slab (2 mm slack).
        ("curb_ny", (0.008, 0.030, 0.020), (-0.086, -0.070, 0.010), hardware),
        ("curb_py", (0.008, 0.175, 0.020), (-0.086, +0.1425, 0.010), hardware),
        ("rail", (0.008, 0.315, 0.020), (-0.086, +0.0725, 0.118), hardware),
        ("stop_n", (0.024, 0.011, 0.128), (-0.078, -0.0795, 0.064), hardware),
        ("stop_p", (0.024, 0.012, 0.128), (-0.078, +0.224, 0.064), hardware),
    ]
    for name, size, center, col in boxes:
        _box(stage, f"{prim_path}/{name}", size, center, col, co, material=mat)
    # the four-post socket cradle: brass ball posts on the shelter floor
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            _sphere(stage, f"{prim_path}/post_{'p' if sx > 0 else 'n'}{'p' if sy > 0 else 'n'}",
                    cfg.post_r, (sx * cfg.post_xy, sy * cfg.post_xy, cfg.post_z),
                    brass, co, material=mat)
    return root


def _spawn_shutter(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The free shutter slab (8 x 136 x 115 mm; origin = slab center). No joint: it is
    captured by the shelter's curb/rail channel. Authored diagonal inertia (external-
    wrench plant recipe) so the solve's force servo is auditable: kv*dt/m =
    2.0/(120*0.30) ~ 0.056 << 1."""
    import omni.usd

    from pxr import Gf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.3)
    pxrb.CreateAngularDampingAttr(0.3)
    massapi = UsdPhysics.MassAPI(root)
    massapi.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    massapi.CreateDiagonalInertiaAttr(Gf.Vec3f(7.9e-4, 3.3e-4, 4.6e-4))
    massapi.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    mat = _material(stage, f"{prim_path}/phys_mat", static=0.30, dynamic=0.25)
    _box(stage, f"{prim_path}/slab", (0.008, 0.136, 0.115), (0.0, 0.0, 0.0),
         (0.75, 0.55, 0.15), cfg.contact_offset, material=mat)
    return root


def _spawn_globe(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The oversized frosted globe bulb: one 90 mm sphere with a small brass cap disc
    on top. Origin = sphere center. Authored inertia I = 2/5 m R^2 so the rolling
    servo is auditable: kv*dt/m = 1.5/(120*0.15) ~ 0.083 << 1."""
    import omni.usd

    from pxr import Gf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.02)
    pxrb.CreateAngularDampingAttr(0.02)
    massapi = UsdPhysics.MassAPI(root)
    massapi.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    massapi.CreateDiagonalInertiaAttr(Gf.Vec3f(1.2e-4, 1.2e-4, 1.2e-4))
    massapi.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    _sphere(stage, f"{prim_path}/globe", cfg.radius, (0.0, 0.0, 0.0),
            (0.95, 0.93, 0.85), co, material=mat)
    _cyl(stage, f"{prim_path}/cap", 0.014, 0.008, (0.0, 0.0, cfg.radius + 0.002),
         (0.80, 0.62, 0.20), co, material=mat)
    return root


def _spawn_decoy(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The graspable decoy bulb: a 48 mm tinted sphere with a cap disc — the WRONG
    bulb (it falls straight through the four-post cradle). Origin = sphere center."""
    import omni.usd

    from pxr import Gf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.02)
    pxrb.CreateAngularDampingAttr(0.02)
    massapi = UsdPhysics.MassAPI(root)
    massapi.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    massapi.CreateDiagonalInertiaAttr(Gf.Vec3f(1.2e-5, 1.2e-5, 1.2e-5))
    massapi.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    _sphere(stage, f"{prim_path}/globe", cfg.radius, (0.0, 0.0, 0.0),
            (0.75, 0.82, 0.92), co, material=mat)
    _cyl(stage, f"{prim_path}/cap", 0.010, 0.006, (0.0, 0.0, cfg.radius + 0.002),
         (0.80, 0.62, 0.20), co, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "shelter" not in _SPAWNER_CACHE:

        @configclass
        class ShelterSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shelter)
            post_xy: float = 0.024
            post_r: float = 0.008
            post_z: float = 0.008
            contact_offset: float = 0.001

        @configclass
        class ShutterSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shutter)
            mass: float = 0.30
            contact_offset: float = 0.001

        @configclass
        class GlobeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_globe)
            mass: float = 0.15
            radius: float = 0.045
            contact_offset: float = 0.001

        @configclass
        class DecoySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_decoy)
            mass: float = 0.05
            radius: float = 0.024
            contact_offset: float = 0.001

        _SPAWNER_CACHE.update(shelter=ShelterSpawnerCfg, shutter=ShutterSpawnerCfg,
                              globe=GlobeSpawnerCfg, decoy=DecoySpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class LanternShelterSceneCfg(BaseCfg):
    """Config for `LanternShelterScene`. Every honesty claim is asserted in
    `__post_init__`: the globe really cannot be grasped (diameter > jaw span), the
    doorway really passes it, the decoy really falls through the post cradle, the
    seat has a real entry barrier / retention barrier / geometric overshoot cap, the
    seat window separates seated from free-rolling and decoy poses, and the closed
    shutter really seals the doorway with every channel gap far smaller than the
    globe."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    seat_xy_tol: float = tunable(0.006)  # |globe - seat center| per axis, shelter frame
    seat_z_lo: float = tunable(0.0465)  # seated-globe center height window (seated ~0.0487;
    seat_z_hi: float = tunable(0.054)  # ... free-rolling 0.045; decoy on the floor 0.024)
    enter_x: float = tunable(-0.014)  # globe center past this -> fully inside the chamber
    enter_y_tol: float = tunable(0.040)  # chamber containment half-width for `entered`
    enter_z_hi: float = tunable(0.070)  # ... and height cap (roof-top rest reads 0.192)
    close_y_tol: float = tunable(0.008)  # |shutter slide offset| for "closed"
    close_x_tol: float = tunable(0.003)  # shutter flush-in-channel window about door_x
    close_z_tol: float = tunable(0.010)  # shutter center height window about door_z
    close_tilt_deg: float = tunable(10.0)  # max shutter tilt from upright
    settle_lin: float = tunable(0.05)  # max globe |lin vel| at judging (m/s)
    settle_ang: float = tunable(1.2)  # max globe |ang vel| at judging (rad/s)
    door_settle_lin: float = tunable(0.05)  # max shutter |lin vel| at judging (m/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    house_x: tuple = tunable((0.58, 0.66))  # shelter seat-center x band (env frame)
    house_y: tuple = tunable((-0.05, 0.05))
    house_yaw_deg: float = tunable(12.0)  # +- yaw about vertical (doorway faces ~ -x)
    door_open: tuple = tunable((0.128, 0.148))  # shutter spawn slide offset (fully open)
    globe_gx: tuple = tunable((-0.34, -0.24))  # globe spawn, shelter frame
    globe_gy: tuple = tunable((-0.08, 0.08))
    decoy_dx: tuple = tunable((-0.32, -0.24))  # decoy spawn, shelter frame (side flips)
    decoy_dy: tuple = tunable((0.16, 0.24))

    # --- info: structure (the geometry the spawners author) ----------------------------------
    globe_r: float = info(0.045)  # goal globe radius (diameter 90 mm > jaw span)
    decoy_r: float = info(0.024)  # decoy radius (graspable, falls through the posts)
    jaw_span: float = info(0.080)  # Franka parallel-jaw max opening
    post_xy: float = info(0.024)  # seat posts at (+-post_xy, +-post_xy, post_z)
    post_r: float = info(0.008)
    post_z: float = info(0.008)
    front_in: float = info(-0.060)  # front wall inner face (local x)
    front_out: float = info(-0.072)  # front wall outer face
    back_in: float = info(0.048)  # back wall inner face (overshoot cap)
    side_in: float = info(0.085)  # side wall inner faces (+-y)
    doorway_half: float = info(0.055)  # doorway half-width (jamb inner faces)
    lintel_z: float = info(0.105)  # doorway clear height (lintel bottom)
    roof_bot: float = info(0.135)
    roof_top: float = info(0.147)
    rail_in_x: float = info(-0.082)  # channel inner face (curbs + rail)
    rail_out_x: float = info(-0.090)
    door_t: float = info(0.008)  # shutter slab thickness
    door_half_w: float = info(0.068)  # shutter slab half-width (along the slide)
    door_h: float = info(0.115)  # shutter slab height
    door_x: float = info(-0.077)  # shutter center x in the channel (shelter frame)
    door_z: float = info(0.0575)  # shutter center height (slab on the floor)
    stop_n_face: float = info(-0.074)  # -y end-stop inner face: push-to-stop = closed
    stop_p_face: float = info(0.218)  # +y end-stop inner face
    globe_mass: float = info(0.15)
    decoy_mass: float = info(0.05)
    door_mass: float = info(0.30)
    contact_offset: float = info(0.001)

    def __post_init__(self) -> None:
        c = self
        R, r = c.globe_r, c.post_r
        reach = R + r  # globe-center-to-post-center contact distance
        diag = math.hypot(c.post_xy, c.post_xy)  # post horizontal radius from seat axis
        seat_z = c.post_z + math.sqrt(reach**2 - diag**2)  # seated globe center ~0.0487
        peak_z = c.post_z + math.sqrt(reach**2 - c.post_xy**2)  # entry pivot peak ~0.0553
        # -- the seed's grasp schema is physically rejected; the decoy IS graspable --
        assert 2 * R > c.jaw_span + 0.008, "goal globe must exceed the jaw span"
        assert 2 * c.decoy_r < c.jaw_span - 0.020, "decoy must be easily graspable"
        # -- the doorway really passes the globe (rolling on the floor) --
        assert 2 * c.doorway_half > 2 * R + 0.015, "doorway width must pass the globe"
        assert c.lintel_z > 2 * R + 0.012, "doorway height must pass the rolling globe"
        # -- the decoy falls straight through the post cradle --
        assert diag > c.decoy_r + c.post_r + 0.001, "decoy must not reach the posts centred"
        assert c.decoy_r < c.seat_z_lo, "floor-resting decoy must be below the seat window"
        # -- the seat is a real raised rest with real barriers --
        assert diag < reach, "the globe must reach all four posts"
        assert seat_z > R + 0.003, "seated globe must sit clearly above free-rolling"
        assert 0.004 < peak_z - R < 0.020, "entry barrier must need a shove yet stay modest"
        assert peak_z - seat_z > 0.004, "retention barrier must be real"
        # -- capture is geometric: the back wall caps overshoot before the far pivot --
        assert c.back_in > R + 0.002, "the globe must fit in front of the back wall"
        assert c.back_in - R < c.post_xy, "overshoot must never reach the far pivot line"
        # -- the seat window is honest by construction --
        assert c.seat_z_lo > R + 0.001, "a free-rolling globe must read below the window"
        assert c.seat_z_lo < seat_z < c.seat_z_hi, "the seated pose must be inside the window"
        assert c.seat_xy_tol >= c.back_in - R, "the window must accept wall-capped rests"
        # -- the roof denies aerial delivery; `entered` is honest --
        assert c.roof_bot > seat_z + R + 0.010, "seated globe must clear the roof"
        assert c.enter_z_hi < c.roof_top + R - 0.05, "a roof-top globe must not read entered"
        assert c.enter_x >= c.front_in + R, "entered must mean fully past the front wall"
        assert c.enter_x < -c.seat_xy_tol and c.seat_z_hi < c.enter_z_hi \
            and c.seat_xy_tol < c.enter_y_tol, "seated must imply entered"
        assert c.side_in > R + 0.030, "the chamber must give the rolling globe room"
        # -- the shutter really seals, really slides, and push-to-stop closes it --
        slack = (c.front_out - c.rail_in_x) - c.door_t
        assert 0.0 < slack <= 0.004 < R, "channel slack must capture the slab, not the globe"
        assert c.door_half_w > c.doorway_half + c.close_y_tol + 0.004, \
            "the closed shutter must overlap the jambs"
        assert c.door_h > c.lintel_z + 0.008, "the shutter must cover the doorway height"
        assert abs(c.stop_n_face + c.door_half_w) < c.close_y_tol, \
            "pushing the shutter to the -y stop must read closed"
        assert c.stop_p_face - c.door_half_w >= c.doorway_half + 0.020, \
            "the channel must have a fully-open park"
        assert c.door_open[0] - c.door_half_w >= c.doorway_half + 0.002, \
            "the spawned shutter must leave the doorway fully open"
        assert c.door_open[1] + c.door_half_w <= c.stop_p_face, \
            "the spawned shutter must fit before the +y stop"
        # -- spawns: outside the shelter hardware, globe/decoy bands disjoint --
        assert c.globe_gx[1] + R < c.rail_out_x - 0.010, "globe spawns clear of the channel"
        assert c.decoy_dx[1] + c.decoy_r < c.rail_out_x - 0.010, "decoy spawns clear too"
        assert c.globe_gy[1] + R < c.decoy_dy[0] - c.decoy_r - 0.005, \
            "globe and decoy spawn bands must be disjoint"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("lantern_shelter")
class LanternShelterScene(BaseScene):
    cfg: LanternShelterSceneCfg

    def __init__(self, cfg: LanternShelterSceneCfg | None = None) -> None:
        super().__init__(cfg or LanternShelterSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        hx = 0.5 * (c.house_x[0] + c.house_x[1])

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
            "shelter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shelter",
                spawn=sp["shelter"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    post_xy=c.post_xy, post_r=c.post_r, post_z=c.post_z,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, 0.0, 0.0)),
            ),
            "shutter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shutter",
                spawn=sp["shutter"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.door_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.door_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hx + c.door_x, c.door_open[1], c.door_z + 0.001)),
            ),
            "globe": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Globe",
                spawn=sp["globe"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.globe_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.globe_mass, radius=c.globe_r, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hx + c.globe_gx[0], 0.0, c.globe_r + 0.002)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=sp["decoy"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.decoy_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.decoy_mass, radius=c.decoy_r, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hx + c.decoy_dx[0], c.decoy_dy[0], c.decoy_r + 0.002)),
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
                # external-wrench plant recipe: without this the solve/smoke force
                # servos on the free globe/shutter are under-applied across TGS
                # iterations (IsaacLab warns at startup; treat the warning as fatal)
                "enable_external_forces_every_iteration": True,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.shelter: RigidObject = env.iscene["shelter"]
        self.shutter: RigidObject = env.iscene["shutter"]
        self.globe: RigidObject = env.iscene["globe"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        # progress latches (post_step): globe entered, globe seated, shuttered-in
        self.enter_latch = torch.zeros(n, device=dev)
        self.seat_latch = torch.zeros(n, device=dev)
        self.shut_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: shelter at a sampled pose (xy + yaw), shutter parked fully
        open in its channel, globe and decoy resting on the open floor outside the
        doorway (decoy side flips); latches zeroed. Discrete draws use torch.rand
        comparisons (the first-randint degeneracy)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def u(lo: float, hi: float) -> torch.Tensor:
            return torch.rand(m, device=dev) * (hi - lo) + lo

        hx, hy = u(*c.house_x), u(*c.house_y)
        hyaw = u(-1.0, 1.0) * math.radians(c.house_yaw_deg)
        ch, sh = torch.cos(hyaw), torch.sin(hyaw)
        half = hyaw / 2
        qw, qz = torch.cos(half), torch.sin(half)

        def write(body, lx, ly, lz, rot: bool) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = hx + ch * lx - sh * ly
            st[:, 1] = hy + sh * lx + ch * ly
            st[:, 2] = lz
            if rot:
                st[:, 3], st[:, 6] = qw, qz
            else:
                st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        zeros = torch.zeros(m, device=dev)
        write(self.shelter, zeros, zeros, zeros, rot=True)
        write(self.shutter, zeros + c.door_x, u(*c.door_open), zeros + c.door_z + 0.001,
              rot=True)
        write(self.globe, u(*c.globe_gx), u(*c.globe_gy), zeros + c.globe_r + 0.002,
              rot=False)
        side = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        write(self.decoy, u(*c.decoy_dx), side * u(*c.decoy_dy),
              zeros + c.decoy_r + 0.002, rot=False)

        self.enter_latch[env_ids] = 0.0
        self.seat_latch[env_ids] = 0.0
        self.shut_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"shelter": self.shelter, "shutter": self.shutter,
                  "globe": self.globe, "decoy": self.decoy}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "enter_latch": self.enter_latch[env_ids].clone(),
            "seat_latch": self.seat_latch[env_ids].clone(),
            "shut_latch": self.shut_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"shelter": self.shelter, "shutter": self.shutter,
                  "globe": self.globe, "decoy": self.decoy}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.enter_latch[env_ids] = state["enter_latch"]
        self.seat_latch[env_ids] = state["seat_latch"]
        self.shut_latch[env_ids] = state["shut_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A roofed lantern SHELTER stands on the floor: grey walls and roof, and a "
            f"single floor-level DOORWAY in its front face ({2 * c.doorway_half * 1000:.0f} mm "
            f"wide, {c.lintel_z * 1000:.0f} mm tall) framed by darker jambs and a lintel. "
            "On the shelter floor inside, four BRASS BALL POSTS form the lamp's socket "
            "cradle, empty. Across the shelter's front runs a shallow channel — floor "
            "curbs, a top rail and two end stops — holding a free-sliding amber storm "
            f"SHUTTER (a {c.door_h * 1000:.0f} mm-tall slab), parked to the side so the "
            "doorway is open. Two bulbs rest on the open floor outside: the lamp's "
            f"frosted GLOBE BULB, a {2 * c.globe_r * 1000:.0f} mm sphere — too wide for "
            "the gripper's jaws — and a smaller "
            f"{2 * c.decoy_r * 1000:.0f} mm pale-blue SPARE bulb, which is the wrong "
            "size: it slips straight through between the brass posts and cannot seat. "
            "The shelter's exact spot and heading, the shutter's parked offset and both "
            "bulbs' places change every episode — read them by looking.\n"
            "Goal: roll the big globe across the floor and in through the doorway (the "
            "roof blocks any drop from above, and no grasp can hold the globe), then "
            "shove it firmly up onto the four brass posts — a gentle nudge stalls at "
            "the posts and rolls back; a firm shove pops it over and the posts hold it "
            "— and finally slide the storm shutter along its channel until it covers "
            "the doorway (pushing it to the near end stop is exactly closed). Success "
            "when the globe rests seated on all four posts, the shutter covers the "
            "doorway, and everything is at rest. A globe loose on the shelter floor, "
            "resting against a wall, or parked outside earns nothing; the small spare "
            "in the cradle earns nothing; a shutter closed on an empty shelter earns "
            "nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Roll the large frosted globe bulb across the floor, in through the "
            "lantern shelter's doorway, and shove it up onto the four brass posts so "
            "it sits seated on them, then slide the amber storm shutter along its "
            "channel to cover the doorway. Leave the globe seated with the shutter "
            "closed."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _shelter_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N, 3): a world point in the shelter body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(
            self.shelter.data.root_quat_w, p_w - self.shelter.data.root_pos_w)

    def globe_local(self) -> torch.Tensor:
        return self._shelter_local(self.globe.data.root_pos_w)

    def decoy_local(self) -> torch.Tensor:
        return self._shelter_local(self.decoy.data.root_pos_w)

    def shutter_local(self) -> torch.Tensor:
        return self._shelter_local(self.shutter.data.root_pos_w)

    def in_chamber(self) -> torch.Tensor:
        """(N,) bool, geometric: globe fully inside the shelter chamber (past the
        front wall, between the walls, on the floor — a roof-top globe reads 0.192
        and is rejected by the height cap; asserted)."""
        c = self.cfg
        loc = self.globe_local()
        return (loc[:, 0] >= c.enter_x) & (loc[:, 0] <= c.back_in) \
            & (loc[:, 1].abs() <= c.enter_y_tol) & (loc[:, 2] <= c.enter_z_hi)

    def in_seat(self) -> torch.Tensor:
        """(N,) bool, geometric: globe seated on the four posts — shelter-frame
        containment (accepts the seated pose incl. wall-capped rests, rejects
        free-rolling / beside-seat / decoy poses; asserted)."""
        c = self.cfg
        loc = self.globe_local()
        return (loc[:, 0].abs() <= c.seat_xy_tol) & (loc[:, 1].abs() <= c.seat_xy_tol) \
            & (loc[:, 2] >= c.seat_z_lo) & (loc[:, 2] <= c.seat_z_hi)

    def door_closed(self) -> torch.Tensor:
        """(N,) bool, geometric: shutter covering the doorway — centred on the slide
        within tolerance, flush in its channel, at slab height, upright."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        loc = self.shutter_local()
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        up = quat_apply(self.shutter.data.root_quat_w, ez)
        return (loc[:, 1].abs() <= c.close_y_tol) \
            & ((loc[:, 0] - c.door_x).abs() <= c.close_x_tol) \
            & ((loc[:, 2] - c.door_z).abs() <= c.close_z_tol) \
            & (up[:, 2] >= math.cos(math.radians(c.close_tilt_deg)))

    def settled(self) -> torch.Tensor:
        """(N,) bool: globe AND shutter at rest (thresholds sit above the GPU
        phantom-velocity artifact)."""
        c = self.cfg
        return (self.globe.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.globe.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang) \
            & (self.shutter.data.root_lin_vel_w.norm(dim=-1) < c.door_settle_lin)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch each demonstrated stage every physics substep: globe ever inside the
        chamber, globe ever seated on the posts, shutter ever closed WHILE the globe
        is seated (a shutter closed on an empty shelter earns nothing)."""
        self.enter_latch = torch.maximum(self.enter_latch, self.in_chamber().float())
        seated = self.in_seat()
        self.seat_latch = torch.maximum(self.seat_latch, seated.float())
        self.shut_latch = torch.maximum(
            self.shut_latch, (seated & self.door_closed()).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the globe rests seated on the four brass posts, the shutter
        covers the doorway, and both are settled. The route (roll in through the
        doorway -> shove over the entry barrier -> close the shutter) is physically
        forced: the globe exceeds the jaw span, the roof denies drops, the posts need
        a super-barrier shove, and the closed shutter's every gap is far smaller than
        the globe."""
        return self.in_seat() & self.door_closed() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.20 globe ever inside the chamber + 0.30 globe ever
        seated on the posts + 0.25 shutter ever closed while seated (cap 0.75); 1.0
        iff success(). Latched — credit never evaporates; the null policy scores ~0
        (globe and decoy spawn outside, the shutter spawns fully open)."""
        base = (0.20 * self.enter_latch + 0.30 * self.seat_latch
                + 0.25 * self.shut_latch).clamp(0.0, 0.75)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="lantern_shelter", robot="null", env_spacing=3.0))
