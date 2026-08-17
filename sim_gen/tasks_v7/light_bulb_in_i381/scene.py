"""GuardCapScene — stand the fallen filament bulb on its brass contact pad, then lower
the keyed storm-guard cage over it until the cage's notched rim seats flush inside the
curb ring (sim_gen task `light_bulb_in_i381`).

Derived from rlbench/light_bulb_in, but STRATEGICALLY different. The seed is a
socket-insertion task: grasp a small bulb out of its holder, carry it over the lamp,
lower it INTO a recessed socket and SCREW it home with a wrist rotation — the moved
object is the bulb, the goal fixture is a passive receptacle, and the terminal act is
a continuous rotation under load. Here the plan skeleton is inverted and re-keyed:

1. **The bulb is never inserted into anything.** It stands EXPOSED on a flat brass
   contact pad on an open plinth deck. The manipulation on the bulb is an
   UPRIGHTING: it spawns knocked over, lying on the floor off the plinth, and must
   be stood upright on the pad — a re-orientation + placement, not an insertion
   (no socket, no recess, nothing wraps the bulb's base).
2. **The enclosure moves, not the bulb.** The load-bearing terminal act is performed
   with a SECOND free body: a rigid storm-guard cage (octagonal walls + closed lid +
   grasp knob) that must be lowered OVER the standing bulb until its rim rests flush
   on the deck inside the curb ring. The seed moves the bulb to the fixture; here the
   fixture-like part is the payload and the bulb is the obstacle the cage must
   swallow without touching.
3. **A passive yaw key replaces the screw.** The cage cannot simply be dropped
   anywhere: two red key BRIDGES stand proud on the deck under the cage's rim
   annulus. Only when the cage's two rim NOTCHES line up over the bridges (yaw
   key, mod 180 deg) can the rim descend past them to flush; at any other yaw the rim
   rests ON the bridges, 8 mm proud — outside the 3.5 mm capped z-window (asserted).
   The screw of the seed is a driven rotation under load; the key here is a passive
   admission geometry — align first, then a straight vertical descent.
4. **Execution order is physically forced.** The cage lid is CLOSED (a solid disc
   plus knob) and the seated cage leaves no aperture that passes the bulb (largest
   under-rim gap is the 11 mm notch vs the 30 mm bulb base — asserted), so the bulb
   can only be stood BEFORE the cage is seated: stand-then-cap is the only order.
5. **A dead decoy bulb is excluded by the rubric's object identity.** An identically
   shaped burnt-out (dark) bulb lies on the other side; standing IT on the pad and
   capping it earns nothing.

So the plan skeleton changes from *"grasp bulb, carry, insert, screw"* to *"upright
the fallen bulb on an exposed pad, then align a keyed rigid enclosure by yaw and
lower it over the bulb to a flush geometric seat"* — the terminal contact-rich act
happens on a different object than the goal-defining one, guarded by a passive
alignment key instead of a driven screw.

Everything is live physics: four rigid bodies (kinematic plinth; free cage, bulb and
dead bulb), no joints anywhere, authored friction/restitution/inertia/CoM, and every
predicate (bulb standing on the pad, cage seated flush inside the curb) is read from
live poses in the plinth's body frame.

success(): the fresh bulb stands upright on the brass pad (plinth-frame containment
window that accepts the on-pad rest at root z ~6 mm and rejects deck-standing ~1 mm,
lying ~15+ mm and tilted poses) AND the cage rests capped (concentric within the curb,
rim flush within 3.5 mm — a key-misaligned cage rests 8 mm proud and is rejected —
upright) AND both are settled.

score() is graded and latched (credit never evaporates): 0.25 bulb ever standing on
the pad + 0.20 cage ever concentric and low over the STANDING bulb = 0.45 cap; 1.0 iff
success(). The null policy scores ~0 (both bulbs spawn lying on the open floor, the
cage spawns parked on the ground beside the plinth).

Per-episode randomization (readback-verified in smoke): plinth xy + yaw, bulb spawn
xy + yaw, dead-bulb side flip + xy + yaw, cage park xy + yaw. Assets are fully
procedural compound spawners (boxes, cylinders, spheres; one rigid body each). Heavy
imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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
         yaw: float | None = None) -> None:
    """Author one box child prim (translate -> orient -> scale, authored once). `yaw`
    (radians about +z) supports the octagon ring segments."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw is not None:
        half = 0.5 * float(yaw)
        sxf.AddOrientOp().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, 0.0, math.sin(half))))
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
    phantom-creep fix)."""
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


def _octagon_ring(stage, prim_path: str, name: str, flat_in: float, t: float, h: float,
                  zc: float, color, co: float, mat, skip_dirs=(), seg_len=None) -> None:
    """Eight box segments forming an octagonal ring: inner flat radius `flat_in`,
    thickness `t`, height `h`, ring centered on the body z-axis. Segment k faces the
    direction k*45 deg; `skip_dirs` lists k values authored elsewhere (notches)."""
    rc = flat_in + 0.5 * t
    length = seg_len if seg_len is not None else 2.0 * rc * math.tan(math.pi / 8.0) + 0.002
    for k in range(8):
        if k in skip_dirs:
            continue
        ang = k * math.pi / 4.0
        _box(stage, f"{prim_path}/{name}{k}", (t, length, h),
             (rc * math.cos(ang), rc * math.sin(ang), zc), color, co,
             material=mat, yaw=ang)


def _spawn_plinth(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC lamp plinth: a square deck slab (origin = deck TOP center), a
    brass contact pad disc at the center, an octagonal curb ring around the cap seat,
    and the two red key bridges on the +-y axis under the cage's rim annulus. All
    numbers MUST match the GuardCapSceneCfg info fields (asserted there)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 10.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    deck_c, curb_c = (0.42, 0.40, 0.36), (0.30, 0.32, 0.38)
    brass, key_red = (0.80, 0.62, 0.20), (0.75, 0.15, 0.12)
    # deck slab: top at local z = 0
    _box(stage, f"{prim_path}/deck", (2 * cfg.deck_half, 2 * cfg.deck_half, cfg.deck_h),
         (0.0, 0.0, -0.5 * cfg.deck_h), deck_c, co, material=mat)
    # brass contact pad at the center (the bulb's stand target)
    _cyl(stage, f"{prim_path}/pad", cfg.pad_r, cfg.pad_h, (0.0, 0.0, 0.5 * cfg.pad_h),
         brass, co, material=mat)
    # curb ring (octagon, inner flat radius curb_in)
    _octagon_ring(stage, prim_path, "curb", cfg.curb_in, cfg.curb_t, cfg.curb_h,
                  0.5 * cfg.curb_h, curb_c, co, mat)
    # two key bridges on the +-y axis (proud rails the un-notched rim rests on)
    for s, nm in ((1.0, "bridge_p"), (-1.0, "bridge_n")):
        _box(stage, f"{prim_path}/{nm}", (cfg.bridge_dx, cfg.bridge_dy, cfg.bridge_h),
             (0.0, s * cfg.bridge_y, 0.5 * cfg.bridge_h), key_red, co, material=mat)
    return root


def _spawn_cage(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The free storm-guard cage (origin = rim plane center): octagonal walls (inner
    flat radius wall_in, height wall_h) with the two +-y segments NOTCHED (bottom
    raised to notch_z — the yaw key), a closed lid disc and a grasp knob. Authored
    CoM + diagonal inertia (external-wrench plant recipe) so the solve's 6-DOF
    descent servo is auditable: kd*dt/m = 8/(120*0.25) ~ 0.27 < 1, kdw*dt/I =
    0.015/(120*6e-4) ~ 0.21 < 1."""
    import omni.usd

    from pxr import Gf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    massapi = UsdPhysics.MassAPI(root)
    massapi.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.058))
    massapi.CreateDiagonalInertiaAttr(Gf.Vec3f(8.0e-4, 8.0e-4, 6.0e-4))
    massapi.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    green, lid_c, knob_c = (0.35, 0.52, 0.40), (0.30, 0.44, 0.34), (0.20, 0.22, 0.24)
    # full wall segments: rim (z=0) to wall_h; +-y segments (k=2, k=6) are notched
    _octagon_ring(stage, prim_path, "wall", cfg.wall_in, cfg.wall_t, cfg.wall_h,
                  0.5 * cfg.wall_h, green, co, mat, skip_dirs=(2, 6))
    rc = cfg.wall_in + 0.5 * cfg.wall_t
    length = 2.0 * rc * math.tan(math.pi / 8.0) + 0.002
    nh = cfg.wall_h - cfg.notch_z
    for s, k in ((1.0, 2), (-1.0, 6)):
        _box(stage, f"{prim_path}/wall{k}", (cfg.wall_t, length, nh),
             (0.0, s * rc, cfg.notch_z + 0.5 * nh), green, co, material=mat,
             yaw=k * math.pi / 4.0)
    # closed lid + grasp knob
    _cyl(stage, f"{prim_path}/lid", cfg.lid_r, cfg.lid_h,
         (0.0, 0.0, cfg.wall_h + 0.5 * cfg.lid_h), lid_c, co, material=mat)
    _cyl(stage, f"{prim_path}/knob", cfg.knob_r, cfg.knob_h,
         (0.0, 0.0, cfg.wall_h + cfg.lid_h + 0.5 * cfg.knob_h), knob_c, co, material=mat)
    return root


def _spawn_bulb(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One filament bulb (origin = base BOTTOM center): brass base cylinder, thin
    stalk, glass globe sphere on top. `fresh` picks the frosted-white vs burnt-dark
    globe color. Authored CoM z = 0.032 (tip angle ~25 deg) + diagonal inertia."""
    import omni.usd

    from pxr import Gf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    massapi = UsdPhysics.MassAPI(root)
    massapi.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.032))
    massapi.CreateDiagonalInertiaAttr(Gf.Vec3f(5.0e-5, 5.0e-5, 1.0e-5))
    massapi.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    brass = (0.80, 0.62, 0.20)
    globe = (0.95, 0.93, 0.85) if cfg.fresh else (0.25, 0.24, 0.22)
    _cyl(stage, f"{prim_path}/base", cfg.base_r, cfg.base_h,
         (0.0, 0.0, 0.5 * cfg.base_h), brass, co, material=mat)
    _cyl(stage, f"{prim_path}/stalk", cfg.stalk_r, cfg.stalk_h,
         (0.0, 0.0, cfg.base_h + 0.5 * cfg.stalk_h), (0.55, 0.55, 0.58), co, material=mat)
    _sphere(stage, f"{prim_path}/globe", cfg.globe_r, (0.0, 0.0, cfg.globe_z),
            globe, co, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "plinth" not in _SPAWNER_CACHE:

        @configclass
        class PlinthSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plinth)
            deck_half: float = 0.13
            deck_h: float = 0.05
            pad_r: float = 0.017
            pad_h: float = 0.005
            curb_in: float = 0.064
            curb_t: float = 0.006
            curb_h: float = 0.014
            bridge_y: float = 0.052
            bridge_dx: float = 0.014
            bridge_dy: float = 0.016
            bridge_h: float = 0.008
            contact_offset: float = 0.001

        @configclass
        class CageSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cage)
            mass: float = 0.25
            wall_in: float = 0.047
            wall_t: float = 0.006
            wall_h: float = 0.110
            notch_z: float = 0.011
            lid_r: float = 0.0555
            lid_h: float = 0.006
            knob_r: float = 0.0125
            knob_h: float = 0.030
            contact_offset: float = 0.001

        @configclass
        class BulbSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bulb)
            mass: float = 0.06
            fresh: bool = True
            base_r: float = 0.015
            base_h: float = 0.020
            stalk_r: float = 0.005
            stalk_h: float = 0.025
            globe_r: float = 0.020
            globe_z: float = 0.063
            contact_offset: float = 0.001

        _SPAWNER_CACHE.update(plinth=PlinthSpawnerCfg, cage=CageSpawnerCfg,
                              bulb=BulbSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class GuardCapSceneCfg(BaseCfg):
    """Config for `GuardCapScene`. Every honesty claim is asserted in `__post_init__`:
    the yaw key is real (misaligned rim rests proud OUTSIDE the capped window, aligned
    notches clear the bridges), in-curb implies concentric, the seated cage really
    encloses the standing bulb with clearance, the closed cage passes no bulb (order
    forcing), the standing window separates on-pad from deck-standing and lying
    poses, and the spawn bands are mutually disjoint and clear of the plinth."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    stand_xy_tol: float = tunable(0.010)  # |bulb - pad axis| per axis, plinth frame
    stand_z_lo: float = tunable(0.0035)  # bulb root (base bottom) height window: on-pad
    stand_z_hi: float = tunable(0.009)  # ... ~0.006; deck-standing ~0.001; lying >= 0.015
    stand_tilt_deg: float = tunable(10.0)  # max bulb tilt from upright
    cap_xy_tol: float = tunable(0.012)  # |cage - plinth axis| per axis (in-curb => within)
    cap_z_lo: float = tunable(-0.002)  # cage rim height window: flush seat ~0.001;
    cap_z_hi: float = tunable(0.0035)  # ... key-misaligned rests proud at ~0.008
    cap_tilt_deg: float = tunable(8.0)  # max cage tilt from upright
    over_z_hi: float = tunable(0.040)  # cage rim below this while concentric -> cap latch
    settle_lin: float = tunable(0.05)  # max |lin vel| at judging (m/s), bulb + cage
    settle_ang: float = tunable(0.8)  # max |ang vel| at judging (rad/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    plinth_x: tuple = tunable((0.42, 0.50))  # plinth center x band (env frame)
    plinth_y: tuple = tunable((-0.05, 0.05))
    plinth_yaw_deg: float = tunable(12.0)  # +- yaw about vertical
    bulb_bx: tuple = tunable((-0.32, -0.23))  # fresh bulb spawn, plinth frame (lying)
    bulb_by: tuple = tunable((-0.06, 0.06))
    dead_dx: tuple = tunable((-0.32, -0.23))  # dead bulb spawn, plinth frame (side flips)
    dead_dy: tuple = tunable((0.25, 0.31))
    cage_cx: tuple = tunable((-0.08, 0.02))  # cage park, plinth frame (opposite side)
    cage_cy: tuple = tunable((0.22, 0.29))

    # --- info: structure (the geometry the spawners author) ----------------------------------
    deck_half: float = info(0.13)  # square deck half-extent; deck TOP = plinth local z 0
    deck_h: float = info(0.05)  # deck thickness (plinth spawns at world z = deck_h)
    pad_r: float = info(0.017)  # brass contact pad (stand target)
    pad_h: float = info(0.005)
    curb_in: float = info(0.064)  # curb ring inner flat radius
    curb_t: float = info(0.006)
    curb_h: float = info(0.014)
    bridge_y: float = info(0.052)  # key bridges at (0, +-bridge_y) on the deck
    bridge_dx: float = info(0.014)  # bridge tangential extent
    bridge_dy: float = info(0.016)  # bridge radial extent (annulus [0.044, 0.060])
    bridge_h: float = info(0.008)  # proud height (the misaligned rest)
    wall_in: float = info(0.047)  # cage wall inner flat radius
    wall_t: float = info(0.006)
    wall_h: float = info(0.110)  # rim (cage local z 0) to lid underside
    notch_z: float = info(0.011)  # notched +-y segments: wall bottom raised to this
    lid_r: float = info(0.0555)  # closed lid disc
    lid_h: float = info(0.006)
    knob_r: float = info(0.0125)  # grasp knob (the cage's handle)
    knob_h: float = info(0.030)
    cage_mass: float = info(0.25)
    base_r: float = info(0.015)  # bulb: brass base (graspable, 30 mm dia)
    base_h: float = info(0.020)
    stalk_r: float = info(0.005)
    stalk_h: float = info(0.025)
    globe_r: float = info(0.020)  # bulb: glass globe (top at 0.083)
    globe_z: float = info(0.063)  # globe center above base bottom
    bulb_mass: float = info(0.06)
    bulb_com_z: float = info(0.032)  # authored CoM (tip angle ~25 deg)
    jaw_span: float = info(0.080)  # Franka parallel-jaw max opening
    contact_offset: float = info(0.001)

    def __post_init__(self) -> None:
        c = self
        wall_out = c.wall_in + c.wall_t  # 0.053: cage outer flat radius
        cage_circ = wall_out / math.cos(math.pi / 8.0)  # 0.0574: cage outer corner radius
        bulb_top = c.globe_z + c.globe_r  # 0.083
        bulb_reach = bulb_top  # lying bulb's max reach from root
        # -- the yaw key is real: misaligned rests proud, aligned clears the bridges --
        assert c.bridge_h >= c.cap_z_hi + 0.004, "misaligned rim-on-bridge rest must be rejected"
        assert c.notch_z >= c.bridge_h + 0.002, "aligned notches must clear the bridges"
        assert c.bridge_y - 0.5 * c.bridge_dy <= c.wall_in - 0.003, \
            "bridge annulus must start inside the wall annulus"
        assert c.bridge_y + 0.5 * c.bridge_dy >= wall_out + 0.003, \
            "bridge annulus must end outside the wall annulus"
        ang_seg = math.pi / 8.0  # wall segment angular half-extent
        ang_bridge = math.atan(0.5 * c.bridge_dx / c.bridge_y)  # ~7.7 deg
        assert ang_bridge < ang_seg - math.radians(5.0), \
            "the bridge must fit under one notched segment with yaw slack"
        # -- the cage fits inside the curb at ANY yaw; in-curb implies concentric --
        assert cage_circ < c.curb_in - 0.004, "cage must fit inside the curb at any yaw"
        assert c.curb_in - wall_out <= c.cap_xy_tol, "fully-in-curb must read concentric"
        # -- the seated cage really encloses the standing bulb with clearance --
        assert c.wall_in >= c.stand_xy_tol + c.globe_r + c.cap_xy_tol + 0.002, \
            "cage interior must clear the standing bulb's globe"
        assert c.wall_h > bulb_top + c.pad_h + 0.010, "lid underside must clear the bulb top"
        # -- order forcing: the closed, seated cage passes no bulb --
        assert c.lid_r >= c.wall_in + 0.002, "the lid must close the top opening"
        assert min(2 * c.base_r, 2 * c.globe_r) > c.notch_z + c.cap_z_hi + 0.004, \
            "the under-rim notch gap must never pass the bulb"
        rc = c.wall_in + 0.5 * c.wall_t
        assert 2.0 * rc * math.tan(math.pi / 8.0) + 0.002 >= 2.0 * c.wall_in * math.tan(math.pi / 8.0), \
            "wall segments must close the octagon at the inner face (no side gaps)"
        # -- grasp affordances: bulb base and cage knob are graspable; cage body is not --
        assert 2 * c.base_r <= c.jaw_span - 0.020, "bulb base must be easily graspable"
        assert 2 * c.knob_r <= c.jaw_span - 0.020, "cage knob must be easily graspable"
        assert 2 * cage_circ > c.jaw_span + 0.020, "cage body must exceed the jaw span"
        # -- the standing window is honest --
        assert c.pad_r > c.base_r + 0.001, "the pad must fully support the centred base"
        assert c.stand_z_lo < c.pad_h and c.pad_h + 0.002 < c.stand_z_hi, \
            "the on-pad rest must sit inside the standing z-window"
        assert c.stand_z_lo > 0.002, "a deck-standing bulb must read below the window"
        assert c.base_r > c.stand_z_hi + 0.004, "a lying bulb must read above the window"
        assert math.tan(math.radians(c.stand_tilt_deg)) * c.bulb_com_z < c.base_r - 0.005, \
            "the tilt bound must keep the CoM well inside the base"
        assert c.stand_xy_tol + c.base_r < c.bridge_y - 0.5 * c.bridge_dy - 0.004, \
            "a standing bulb must never touch the key bridges"
        # -- the capped window is honest --
        assert c.cap_z_lo < 0.0 < 0.002 < c.cap_z_hi, "the flush seat must read capped"
        assert c.over_z_hi < c.pad_h + c.globe_z - c.globe_r - 0.004, \
            "cap-latch height must demand the walls fully surround the standing globe"
        # -- spawn bands: mutually disjoint, clear of the plinth deck, null scores 0 --
        assert c.bulb_bx[1] + bulb_reach < -c.deck_half - 0.010, \
            "fresh bulb spawns clear of the deck"
        assert c.dead_dx[1] + bulb_reach < -c.deck_half - 0.010, \
            "dead bulb spawns clear of the deck"
        assert c.bulb_by[1] + bulb_reach < c.dead_dy[0] - bulb_reach - 0.010, \
            "fresh and dead spawn bands must be disjoint"
        assert c.bulb_by[1] + bulb_reach < c.cage_cy[0] - cage_circ - 0.010, \
            "fresh bulb and cage park bands must be disjoint"
        assert c.cage_cy[0] - cage_circ > c.deck_half + 0.010, \
            "the cage parks clear of the deck"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("bulb_guard_cap")
class GuardCapScene(BaseScene):
    cfg: GuardCapSceneCfg

    def __init__(self, cfg: GuardCapSceneCfg | None = None) -> None:
        super().__init__(cfg or GuardCapSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        px = 0.5 * (c.plinth_x[0] + c.plinth_x[1])

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
            "plinth": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plinth",
                spawn=sp["plinth"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    deck_half=c.deck_half, deck_h=c.deck_h, pad_r=c.pad_r, pad_h=c.pad_h,
                    curb_in=c.curb_in, curb_t=c.curb_t, curb_h=c.curb_h,
                    bridge_y=c.bridge_y, bridge_dx=c.bridge_dx, bridge_dy=c.bridge_dy,
                    bridge_h=c.bridge_h, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, 0.0, c.deck_h)),
            ),
            "cage": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cage",
                spawn=sp["cage"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cage_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.cage_mass, wall_in=c.wall_in, wall_t=c.wall_t,
                    wall_h=c.wall_h, notch_z=c.notch_z, lid_r=c.lid_r, lid_h=c.lid_h,
                    knob_r=c.knob_r, knob_h=c.knob_h, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px + c.cage_cx[0], -c.cage_cy[0], 0.001)),
            ),
            "bulb": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bulb",
                spawn=sp["bulb"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bulb_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.bulb_mass, fresh=True, base_r=c.base_r, base_h=c.base_h,
                    stalk_r=c.stalk_r, stalk_h=c.stalk_h, globe_r=c.globe_r,
                    globe_z=c.globe_z, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px + c.bulb_bx[0], 0.0, 0.021)),
            ),
            "dead": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dead",
                spawn=sp["bulb"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bulb_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.bulb_mass, fresh=False, base_r=c.base_r, base_h=c.base_h,
                    stalk_r=c.stalk_r, stalk_h=c.stalk_h, globe_r=c.globe_r,
                    globe_z=c.globe_z, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px + c.dead_dx[0], c.dead_dy[0], 0.021)),
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
                # servos on the free cage/bulb are under-applied across TGS iterations
                "enable_external_forces_every_iteration": True,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.plinth: RigidObject = env.iscene["plinth"]
        self.cage: RigidObject = env.iscene["cage"]
        self.bulb: RigidObject = env.iscene["bulb"]
        self.dead: RigidObject = env.iscene["dead"]
        self.env_origins = env.iscene.env_origins
        # progress latches (post_step): bulb ever standing on the pad; cage ever
        # concentric and low over the STANDING bulb
        self.stand_latch = torch.zeros(n, device=dev)
        self.cap_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: plinth at a sampled pose (xy + yaw), both bulbs LYING on the
        open floor (fresh in front, dead on a flipping side), the cage parked upright
        on the ground on the opposite side; latches zeroed. A burn draw guards the
        first-post-seed-draw degeneracy; the side flip uses a rand comparison."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def u(lo: float, hi: float) -> torch.Tensor:
            return torch.rand(m, device=dev) * (hi - lo) + lo

        _ = torch.rand(m, device=dev)  # burn (first-draw degeneracy)
        px, py = u(*c.plinth_x), u(*c.plinth_y)
        pyaw = u(-1.0, 1.0) * math.radians(c.plinth_yaw_deg)
        ch, sh = torch.cos(pyaw), torch.sin(pyaw)
        half = pyaw / 2
        qw, qz = torch.cos(half), torch.sin(half)

        def write(body, lx, ly, z_abs, quat=None) -> None:
            """Place `body` at plinth-frame xy (lx, ly), absolute height z_abs, with
            world-frame quaternion `quat` (default: the plinth's yaw)."""
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = px + ch * lx - sh * ly
            st[:, 1] = py + sh * lx + ch * ly
            st[:, 2] = z_abs
            if quat is None:
                st[:, 3], st[:, 6] = qw, qz
            else:
                st[:, 3:7] = quat
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        zeros = torch.zeros(m, device=dev)

        def lying_quat(yaw: torch.Tensor) -> torch.Tensor:
            """qz(yaw) * qy(90 deg): bulb axis horizontal, pointing along yaw."""
            cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
            c45 = math.cos(math.pi / 4)
            return torch.stack([cy * c45, -sy * c45, cy * c45, sy * c45], dim=-1)

        write(self.plinth, zeros, zeros, zeros + c.deck_h)
        bulb_yaw = u(-math.pi, math.pi)
        write(self.bulb, u(*c.bulb_bx), u(*c.bulb_by), zeros + 0.021,
              quat=lying_quat(bulb_yaw))
        side = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        dead_yaw = u(-math.pi, math.pi)
        write(self.dead, u(*c.dead_dx), side * u(*c.dead_dy), zeros + 0.021,
              quat=lying_quat(dead_yaw))
        cage_yaw = u(-math.pi, math.pi)
        cy2, sy2 = torch.cos(cage_yaw / 2), torch.sin(cage_yaw / 2)
        cage_quat = torch.stack([cy2, torch.zeros_like(sy2), torch.zeros_like(sy2), sy2],
                                dim=-1)
        write(self.cage, u(*c.cage_cx), -side * u(*c.cage_cy), zeros + 0.001,
              quat=cage_quat)

        self.stand_latch[env_ids] = 0.0
        self.cap_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"plinth": self.plinth, "cage": self.cage,
                  "bulb": self.bulb, "dead": self.dead}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "stand_latch": self.stand_latch[env_ids].clone(),
            "cap_latch": self.cap_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"plinth": self.plinth, "cage": self.cage,
                  "bulb": self.bulb, "dead": self.dead}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.stand_latch[env_ids] = state["stand_latch"]
        self.cap_latch[env_ids] = state["cap_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A low square PLINTH deck stands on the floor. At its center a small "
            "BRASS PAD marks the lamp's contact point, ringed by a shallow octagonal "
            "CURB; on the deck under the ring, two proud RED KEY BRIDGES sit on "
            "opposite sides of the pad. Nearby on the open floor lie two identical "
            f"filament bulbs, knocked over: the FRESH one with a frosted white "
            f"{2 * c.globe_r * 1000:.0f} mm globe, and a BURNT-OUT one with a dark "
            "globe. An octagonal green storm-guard CAGE — open at the bottom, closed "
            "on top by a lid with a grasp knob, its rim carrying two NOTCHES — is "
            "parked upright on the ground beside the plinth. The plinth's spot and "
            "heading, both bulbs' resting places and the cage's park change every "
            "episode — read them by looking.\n"
            "Goal: stand the FRESH bulb upright on the brass pad, then pick the cage "
            "up by its knob, align its two rim notches over the two red key bridges "
            "(the cage fits down only in that alignment — at any other angle its rim "
            "lands on the bridges and sits proud), and lower it straight down over "
            "the standing bulb until the rim rests flush on the deck inside the "
            "curb. The lid is closed, so the bulb can never be put in afterwards: "
            "stand the bulb first. Success when the fresh bulb stands upright on the "
            "pad, the cage sits flush and level inside the curb around it, and "
            "everything is at rest. A bulb standing off the pad, a lying bulb, a "
            "proud or tilted cage, or the burnt-out bulb under the cage earns "
            "nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Stand the fresh frosted bulb upright on the brass pad at the center of "
            "the plinth, then align the green guard cage's rim notches with the two "
            "red key bridges and lower the cage over the bulb until its rim sits "
            "flush on the deck inside the curb. Leave the bulb standing and the cage "
            "seated."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _plinth_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N, 3): a world point in the plinth body frame (origin = deck top center)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(
            self.plinth.data.root_quat_w, p_w - self.plinth.data.root_pos_w)

    def bulb_local(self) -> torch.Tensor:
        return self._plinth_local(self.bulb.data.root_pos_w)

    def dead_local(self) -> torch.Tensor:
        return self._plinth_local(self.dead.data.root_pos_w)

    def cage_local(self) -> torch.Tensor:
        return self._plinth_local(self.cage.data.root_pos_w)

    def _up_z(self, body) -> torch.Tensor:
        """(N,): world z-component of the body's +z axis."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)[:, 2]

    def bulb_standing(self) -> torch.Tensor:
        """(N,) bool, geometric: the FRESH bulb stands upright on the brass pad —
        plinth-frame containment (root = base bottom at pad-top height, on the pad
        axis, upright; deck-standing, lying and tilted poses are rejected —
        asserted)."""
        c = self.cfg
        loc = self.bulb_local()
        return (loc[:, 0].abs() <= c.stand_xy_tol) & (loc[:, 1].abs() <= c.stand_xy_tol) \
            & (loc[:, 2] >= c.stand_z_lo) & (loc[:, 2] <= c.stand_z_hi) \
            & (self._up_z(self.bulb) >= math.cos(math.radians(c.stand_tilt_deg)))

    def cage_capped(self) -> torch.Tensor:
        """(N,) bool, geometric: the cage rests flush inside the curb — concentric
        (in-curb implies within tolerance, asserted), rim at deck level (the
        key-misaligned rim-on-bridge rest at ~8 mm is outside the window), upright."""
        c = self.cfg
        loc = self.cage_local()
        return (loc[:, 0].abs() <= c.cap_xy_tol) & (loc[:, 1].abs() <= c.cap_xy_tol) \
            & (loc[:, 2] >= c.cap_z_lo) & (loc[:, 2] <= c.cap_z_hi) \
            & (self._up_z(self.cage) >= math.cos(math.radians(c.cap_tilt_deg)))

    def cage_over(self) -> torch.Tensor:
        """(N,) bool: the cage hovers/rests concentric and low over the pad (the
        cap-latch stage: real descent over the bulb, whether or not fully seated)."""
        c = self.cfg
        loc = self.cage_local()
        return (loc[:, 0].abs() <= c.cap_xy_tol) & (loc[:, 1].abs() <= c.cap_xy_tol) \
            & (loc[:, 2] >= c.cap_z_lo) & (loc[:, 2] <= c.over_z_hi) \
            & (self._up_z(self.cage) >= math.cos(math.radians(c.cap_tilt_deg)))

    def settled(self) -> torch.Tensor:
        """(N,) bool: bulb AND cage at rest (thresholds above the GPU phantom-velocity
        artifact)."""
        c = self.cfg
        return (self.bulb.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.bulb.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang) \
            & (self.cage.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.cage.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch each demonstrated stage every physics substep: bulb ever standing on
        the pad; cage ever concentric and low over the STANDING bulb (capping an
        empty or wrong-bulb pad earns nothing)."""
        standing = self.bulb_standing()
        self.stand_latch = torch.maximum(self.stand_latch, standing.float())
        self.cap_latch = torch.maximum(
            self.cap_latch, (standing & self.cage_over()).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the fresh bulb stands upright on the brass pad, the storm-guard
        cage rests flush and level inside the curb around it, and both are settled.
        The route (stand the bulb, then key-align and lower the cage) is physically
        forced: the closed lid + no-bulb-passing gaps force stand-before-cap, and the
        key bridges force the yaw alignment before the rim can reach flush."""
        return self.bulb_standing() & self.cage_capped() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25 bulb ever standing on the pad + 0.20 cage ever
        concentric and low over the standing bulb (cap 0.45); 1.0 iff success().
        Latched — credit never evaporates; the null policy scores ~0 (both bulbs
        spawn lying on the floor, the cage parks beside the plinth)."""
        base = (0.25 * self.stand_latch + 0.20 * self.cap_latch).clamp(0.0, 0.45)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="bulb_guard_cap", robot="null", env_spacing=3.0))
