"""UmbrellaBayonetScene — unlock, hoist and BAYONET-PARK the furled umbrella runner on
its mast (sim_gen task `put_umbrella_in_umbrella_stand_i367`).

Derived from rlbench/put_umbrella_in_umbrella_stand, but STRATEGICALLY different: the
seed grasps a free umbrella and drops it tip-down into a floor stand's tube — free
transport ending in vertical insertion. Here the umbrella is a patio-style RUNNER
already CAPTIVE on its mast (a 2-DOF spawn-authored D6 joint: heave along the pole +
free yaw about it — it can never be "put in" anything; it is already in), and the goal
is a BAYONET PARK at height: extract the RED-knobbed transit pin that blocks the track
low on the mast (physics-forced first step), hoist the runner up the pole, thread its
YELLOW handle lug through the single open gap of the GREEN gallery shelf ring, TWIST
~90 deg above the shelf, and set the lug down ON the shelf ring so the runner hangs
parked hands-off. A solver needs a different PLAN (unlock -> align yaw to the keyed
gap -> hoist through -> rotate away from the gap -> lower onto the ring; releasing
anywhere un-parked lets gravity run the runner straight back down the pole) and
different code STRUCTURE (captive 2-DOF heave+yaw joint, keyed-aperture passage,
over-the-gap twist-lock — not grasp-transport-insert).

success(): runner joint height in the PARK BAND (|q - q_park| <= park_tol: the lug
resting exactly on the shelf top — a runner held hovering above the shelf, or twisted
BELOW it, is rejected), lug heading >= align_min_deg away from the gap heading (an
un-twisted runner over the gap just falls back through), the pin latch AND the pass
latch set (the state was reached by actually extracting the pin and actually hoisting
through the shelf — a runner teleported to the park pose earns nothing), and
everything PERSISTENTLY still (stillness counter-latch).

score(), latched (credit never evaporates): 0.30 once the transit pin has ever been
extracted >= pin_latch_dist from the mast axis; 0.30 once the runner has ever risen
above the shelf (q >= q_pass — with the pin seated the track jams at q_block = 0.0575,
and the shelf plate blocks the lug at every heading except the gap, so this latch is
reachable only unlocked AND gap-aligned); 1.0 iff success(). Null policy ~0 (the
runner starts parked LOW on the bottom stop).

Assets are fully procedural (compound spawners; explicit friction materials;
mass/CoM/inertia AUTHORED — custom spawn funcs apply no cfg schemas):
  - base (KINEMATIC, the D6 anchor — it never contacts the runner, so the joint's
    pair-collision filter costs nothing): a floor slab under the mast.
  - mast (KINEMATIC, separate body — its plate/pole/pin-channel contacts stay LIVE):
    round pole split at the pin channel (internal rib guides leave the pin ~1.5 mm
    of play), the GREEN gallery shelf ring at 0.52 m (square hole that passes the
    collar at ANY yaw; one open gap sector toward +x that alone passes the lug), on
    four corner posts outside the lug's swing circle.
  - runner (dynamic): square-tube collar riding the pole on the spawn-authored D6
    (heave limited [0, travel], yaw twist +/-175 deg, all else locked), YELLOW handle lug, and
    a visual-only furled-canopy cone.
  - pin (dynamic, RED knob): rests in the mast's through-channel; its protruding
    tails overlap the rising collar wall on both sides for every sampled seat depth.
  - spare (dynamic): an identical DECOY pin on the floor; moving it earns nothing.

Per-episode randomization (readback-verified in smoke): runner yaw free +/-180 deg,
pin seat depth jitter, spare pin polar position + yaw on the floor.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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
    """One USD physics material (explicit binding — the default-material ~0.5 trap)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float, material=None) -> None:
    """Author one box child prim (translate -> scale, authored once — idempotent per
    prim, the duplicate-xformOp trap)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _cyl(stage, path: str, radius, height, center, color, contact_offset: float,
         material=None) -> None:
    """Author one z-axis cylinder child prim (round pole segments — a square pole
    corner-binds a square collar bore: circumradius 0.015*sqrt(2) > bore half 0.021,
    which rotationally LOCKED the runner in forge attempts 2-6)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateAxisAttr("Z")
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateExtentAttr([Gf.Vec3f(-float(radius), -float(radius), -float(height) / 2),
                          Gf.Vec3f(float(radius), float(radius), float(height) / 2)])
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rigid_dynamic(root, mass: float, com, inertia, lin_damp: float, ang_damp: float) -> None:
    """Author the dynamic-body physics stack (mass/CoM/inertia EXPLICIT)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _spawn_base(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC floor slab under the mast — the runner joint's anchor body (Body0).
    The joint pair-collision filter silences ALL its colliders vs the runner, so the
    slab must never be a load-bearing contact for the runner (it is not: the runner's
    lowest reach stays well above the slab top)."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(40.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    _box(stage, f"{prim_path}/slab", (2 * cfg.base_half, 2 * cfg.base_half, cfg.base_h),
         (0.0, 0.0, cfg.base_h / 2), (0.45, 0.42, 0.40), cfg.contact_offset, material=mat)
    return root


def _spawn_mast(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC mast: pole split at the pin channel (rib guides INSIDE the pole
    footprint), the GREEN gallery shelf ring (E sector open — the keyed gap), corner
    posts. Separate body from Base so all its runner/pin contacts stay live."""
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
    co, cot = cfg.contact_offset, cfg.contact_offset_tight
    ph = cfg.pole_half
    grey = (0.35, 0.35, 0.38)
    green = (0.15, 0.62, 0.20)
    dark = (0.25, 0.25, 0.28)
    # --- ROUND pole (radius = pole_half), split at the pin channel band
    #     [chan_z0, chan_z1] — a round pole spins freely inside the square collar
    #     bore at every yaw (a square pole corner-binds it: forge attempts 2-6) ---
    _cyl(stage, f"{prim_path}/pole_lower", ph, cfg.chan_z0 - cfg.base_h,
         (0.0, 0.0, (cfg.base_h + cfg.chan_z0) / 2), grey, cot, material=mat)
    _cyl(stage, f"{prim_path}/pole_upper", ph, cfg.pole_top - cfg.chan_z1,
         (0.0, 0.0, (cfg.chan_z1 + cfg.pole_top) / 2), grey, cot, material=mat)
    # --- pin-channel rib guides in the y margins of the band; their corner radius
    #     hypot(rib_xh, rib_out) stays inside the collar bore so the collar can
    #     still spin while riding past the band ---
    for s, nm in ((1.0, "rib_n"), (-1.0, "rib_s")):
        _box(stage, f"{prim_path}/{nm}",
             (2 * cfg.rib_xh, cfg.rib_out - cfg.rib_inner, cfg.chan_z1 - cfg.chan_z0),
             (0.0, s * (cfg.rib_inner + cfg.rib_out) / 2, (cfg.chan_z0 + cfg.chan_z1) / 2),
             grey, cot, material=mat)
    # --- gallery shelf ring at [plate_z0, plate_z1]: square hole half `hole_half`;
    #     E sector (toward +x) OPEN — the keyed gap that alone passes the lug ---
    pt = cfg.plate_z1 - cfg.plate_z0
    zc = (cfg.plate_z0 + cfg.plate_z1) / 2
    hh, po = cfg.hole_half, cfg.plate_out
    _box(stage, f"{prim_path}/plate_n", (2 * po, po - hh, pt),
         (0.0, (hh + po) / 2, zc), green, co, material=mat)
    _box(stage, f"{prim_path}/plate_s", (2 * po, po - hh, pt),
         (0.0, -(hh + po) / 2, zc), green, co, material=mat)
    _box(stage, f"{prim_path}/plate_w", (po - hh, 2 * hh, pt),
         (-(hh + po) / 2, 0.0, zc), green, co, material=mat)
    # --- corner posts carrying the shelf (outside the lug's swing circle) ---
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            nm = f"post_{'p' if sx > 0 else 'n'}{'p' if sy > 0 else 'n'}"
            _box(stage, f"{prim_path}/{nm}",
                 (2 * cfg.post_half, 2 * cfg.post_half, cfg.plate_z0 - cfg.base_h),
                 (sx * po, sy * po, (cfg.base_h + cfg.plate_z0) / 2), dark, co, material=mat)
    return root


def _spawn_runner(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The umbrella runner: square-tube collar (4 walls) + YELLOW handle lug +
    visual-only furled-canopy cone, ONE dynamic body on a spawn-authored generic D6
    joint vs the sibling Base, frames rotated so heave rides transX (joint-local
    x == world z, limited [0, travel]) and yaw rides the TWIST DOF rotX (+/-175
    deg), all other axes locked (low > high). Mass/CoM/inertia AUTHORED."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass, (0.0, 0.0, 0.0), (3.0e-4, 3.0e-4, 2.5e-4),
                   lin_damp=0.2, ang_damp=0.3)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    ci, cwall, hh = cfg.col_in, cfg.col_out - cfg.col_in, cfg.col_hh
    cm = (ci + cfg.col_out) / 2
    blue = (0.30, 0.42, 0.75)
    # collar: +/-x walls full-y, +/-y walls between them
    _box(stage, f"{prim_path}/wall_px", (cwall, 2 * cfg.col_out, 2 * hh),
         (cm, 0.0, 0.0), blue, co, material=mat)
    _box(stage, f"{prim_path}/wall_nx", (cwall, 2 * cfg.col_out, 2 * hh),
         (-cm, 0.0, 0.0), blue, co, material=mat)
    _box(stage, f"{prim_path}/wall_py", (2 * ci, cwall, 2 * hh),
         (0.0, cm, 0.0), blue, co, material=mat)
    _box(stage, f"{prim_path}/wall_ny", (2 * ci, cwall, 2 * hh),
         (0.0, -cm, 0.0), blue, co, material=mat)
    # YELLOW handle lug along body +x
    _box(stage, f"{prim_path}/lug",
         (cfg.lug_r1 - cfg.lug_r0, 2 * cfg.lug_hw, cfg.lug_z1 - cfg.lug_z0),
         ((cfg.lug_r0 + cfg.lug_r1) / 2, 0.0, (cfg.lug_z0 + cfg.lug_z1) / 2),
         (0.95, 0.80, 0.10), co, material=mat)
    # visual-only furled canopy (no CollisionAPI — the collar is the physical proxy)
    cone = UsdGeom.Cone.Define(stage, f"{prim_path}/canopy")
    cone.CreateAxisAttr("Z")
    cone.CreateRadiusAttr(float(cfg.canopy_r))
    cone.CreateHeightAttr(float(cfg.canopy_h))
    cxf = UsdGeom.Xformable(cone.GetPrim())
    cxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(hh + cfg.canopy_h / 2)))
    cone.CreateDisplayColorAttr([Gf.Vec3f(0.60, 0.12, 0.15)])
    # generic D6 joint vs the sibling Base: heave (limited) + yaw (free)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.Joint.Define(stage, f"{prim_path}/mount")
    j.CreateBody0Rel().SetTargets([f"{base}/Base"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(False)
    # Joint frames rotated Ry(-90 deg) so joint-local X == world Z: the near-full-
    # turn yaw then rides the D6 TWIST DOF (clean revolute semantics) instead of a
    # swing axis — identity-frame rotZ is a PhysX swing/pyramid DOF, and forge
    # attempts 2-4 showed it hard-stops at ~143 deg regardless of the authored band
    # (and with no LimitAPI at all the axis is simply LOCKED). Heave = transX.
    rq = Gf.Quatf(0.70710678, 0.0, -0.70710678, 0.0)  # Ry(-90): local x -> world z
    j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in cfg.anchor_frame]))
    j.CreateLocalRot0Attr(rq)
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(rq)
    for axis in ("transY", "transZ", "rotY", "rotZ"):  # locked: low > high
        la = UsdPhysics.LimitAPI.Apply(j.GetPrim(), axis)
        la.CreateLowAttr(1.0)
        la.CreateHighAttr(-1.0)
    la = UsdPhysics.LimitAPI.Apply(j.GetPrim(), "transX")  # heave (world +z)
    la.CreateLowAttr(0.0)
    la.CreateHighAttr(float(cfg.travel))
    la = UsdPhysics.LimitAPI.Apply(j.GetPrim(), "rotX")  # yaw twist, DEGREES
    la.CreateLowAttr(-float(cfg.rot_lim_deg))
    la.CreateHighAttr(float(cfg.rot_lim_deg))
    return root


def _spawn_pin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A transit pin: square shaft along body x + RED knob cube on the +x end, ONE
    free dynamic body (no joint). CoM authored AT THE BODY ORIGIN (shaft center) so
    the heavy knob cannot silently shift it (the volume-weighted-CoM trap)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass, (0.0, 0.0, 0.0), (5.0e-6, 2.0e-5, 2.0e-5),
                   lin_damp=0.1, ang_damp=0.2)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    cot = cfg.contact_offset_tight
    _box(stage, f"{prim_path}/shaft", (2 * cfg.len_half, 2 * cfg.hw, 2 * cfg.hw),
         (0.0, 0.0, 0.0), (0.75, 0.75, 0.78), cot, material=mat)
    _box(stage, f"{prim_path}/knob", (2 * cfg.knob_half, 2 * cfg.knob_half, 2 * cfg.knob_half),
         (cfg.knob_cx, 0.0, 0.0), (0.85, 0.13, 0.13), cot, material=mat)
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
            base_half: float = 0.17
            base_h: float = 0.04
            mu_static: float = 0.5
            mu_dynamic: float = 0.4
            contact_offset: float = 0.002

        @configclass
        class MastSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_mast)
            base_h: float = 0.04
            pole_half: float = 0.015
            rib_xh: float = 0.011
            rib_out: float = 0.014
            pole_top: float = 0.88
            chan_z0: float = 0.2375
            chan_z1: float = 0.2525
            rib_inner: float = 0.0075
            plate_z0: float = 0.512
            plate_z1: float = 0.520
            hole_half: float = 0.045
            plate_out: float = 0.070
            post_half: float = 0.008
            mu_static: float = 0.5
            mu_dynamic: float = 0.4
            contact_offset: float = 0.002
            contact_offset_tight: float = 0.0005

        @configclass
        class RunnerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_runner)
            col_in: float = 0.021
            col_out: float = 0.027
            col_hh: float = 0.025
            lug_r0: float = 0.027
            lug_r1: float = 0.075
            lug_hw: float = 0.008
            lug_z0: float = 0.006
            lug_z1: float = 0.022
            canopy_r: float = 0.038
            canopy_h: float = 0.16
            mass: float = 0.20
            travel: float = 0.42
            rot_lim_deg: float = 175.0
            anchor_frame: tuple = (0.0, 0.0, 0.155)
            mu_static: float = 0.5
            mu_dynamic: float = 0.4
            contact_offset: float = 0.002

        @configclass
        class PinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pin)
            len_half: float = 0.032
            hw: float = 0.006
            knob_half: float = 0.012
            knob_cx: float = 0.044
            mass: float = 0.06
            mu_static: float = 0.5
            mu_dynamic: float = 0.4
            contact_offset_tight: float = 0.0005

        _SPAWNER_CACHE.update(base=BaseSpawnerCfg, mast=MastSpawnerCfg,
                              runner=RunnerSpawnerCfg, pin=PinSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class UmbrellaBayonetSceneCfg(BaseCfg):
    """Config for `UmbrellaBayonetScene`. `__post_init__` asserts the strategic
    honesty invariants with pre-computed geometry: the collar passes the shelf hole
    at ANY yaw while the lug passes ONLY through the keyed gap; the seated pin jams
    the track far below the shelf for every sampled seat depth (pin-first order is
    physics-forced); the park band is reachable ONLY by hoisting through the gap and
    is the unique hands-off rest at that height; the alignment threshold sits
    strictly inside the physically-supported heading range; every grasp fits a
    parallel jaw."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    q_pass: float = tunable(0.370)  # runner joint pos counted as "above the shelf" (m)
    park_tol: float = tunable(0.006)  # |q - q_park| band for the parked rest
    align_min_deg: float = tunable(55.0)  # min lug heading away from the gap heading
    pin_latch_dist: float = tunable(0.10)  # pin horizontal dist from the axis -> latched
    settle_lin: float = tunable(0.04)  # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.50)  # max runner |ang vel| when judging (rad/s)
    settle_steps_min: int = tunable(30)  # stillness must PERSIST this many steps

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    yaw_deg: float = tunable(168.0)  # uniform +/- runner yaw at reset (must clear
    #   the rotZ stops by a margin so no spawn sits pressed on a joint limit)
    pin_jit: float = tunable(0.008)  # pin seat depth jitter along +x (m)
    spare_r_lo: float = tunable(0.26)  # spare-pin polar radius band on the floor ...
    spare_r_hi: float = tunable(0.32)
    spare_ang_lo: float = tunable(100.0)  # ... and polar angle band (deg from +x;
    spare_ang_hi: float = tunable(260.0)  # keeps it out of the pin-pull corridor)

    # --- info: structure (base+mast at the env origin, NEVER teleported) ---------------------
    base_half: float = info(0.17)
    base_h: float = info(0.04)
    pole_half: float = info(0.015)  # ROUND pole radius (round so the collar can spin)
    pole_top: float = info(0.88)
    chan_z0: float = info(0.2375)  # pin channel floor (top of the lower pole section)
    chan_z1: float = info(0.2525)  # pin channel roof (bottom of the upper section)
    rib_inner: float = info(0.0075)  # rib guide inner faces at +/- this y
    rib_xh: float = info(0.011)  # rib guide half-length along x
    rib_out: float = info(0.014)  # rib guide outer faces at +/- this y
    plate_z0: float = info(0.512)  # shelf ring underside
    plate_z1: float = info(0.520)  # shelf ring top (the park seat)
    hole_half: float = info(0.045)  # shelf hole half-width (passes the collar, any yaw)
    plate_out: float = info(0.070)  # shelf ring outer half-width
    post_half: float = info(0.008)
    col_in: float = info(0.021)  # collar tube inner half-width
    col_out: float = info(0.027)  # collar tube outer half-width
    col_hh: float = info(0.025)  # collar half-height
    lug_r0: float = info(0.027)  # lug inner end (at the collar wall) ...
    lug_r1: float = info(0.075)  # ... to the lug tip radius
    lug_hw: float = info(0.008)  # lug half-width
    lug_z0: float = info(0.006)  # lug underside, body frame ...
    lug_z1: float = info(0.022)  # ... to lug top
    runner_z0: float = info(0.155)  # runner body origin height at q = 0
    travel: float = info(0.42)  # D6 transZ hard stops [0, travel]
    rot_lim_deg: float = info(175.0)  # D6 rotZ band (PhysX locks limit-less axes;
    #   175 keeps the joint off the +/-180 wrap boundary)
    runner_mass: float = info(0.20)
    pin_len_half: float = info(0.032)  # pin shaft half-length (along its x)
    pin_hw: float = info(0.006)  # pin shaft cross-section half-width
    knob_half: float = info(0.012)
    knob_cx: float = info(0.044)  # knob center on the shaft +x end
    pin_mass: float = info(0.06)
    hoist_q: float = info(0.405)  # the solve's hoist height (clearance asserted)
    pull_clear: float = info(0.14)  # the solve pulls the pin to this horizontal dist
    mu_static: float = info(0.5)
    mu_dynamic: float = info(0.4)
    contact_offset: float = info(0.002)
    contact_offset_tight: float = info(0.0005)  # pin/channel (1.5 mm play per side)

    # Derived (filled in __post_init__).
    q_park: float = field(default=None, init=False)  # lug resting on the shelf top
    q_block: float = field(default=None, init=False)  # collar jammed under the seated pin
    q_pin_rest: float = field(default=None, init=False)  # collar dropped ONTO the seated pin
    pin_rest_z: float = field(default=None, init=False)  # seated pin center height
    support_min_deg: float = field(default=None, init=False)  # min heading with shelf support

    def __post_init__(self) -> None:
        self.pin_rest_z = self.chan_z0 + self.pin_hw  # 0.2435, settled on the channel floor
        self.q_park = self.plate_z1 - (self.runner_z0 + self.lug_z0)  # 0.359
        self.q_block = (self.pin_rest_z - self.pin_hw) - (self.runner_z0 + self.col_hh)  # 0.0575
        self.q_pin_rest = (self.pin_rest_z + self.pin_hw) - (self.runner_z0 - self.col_hh)
        self.support_min_deg = math.degrees(
            math.asin((self.hole_half - self.lug_hw) / self.lug_r1))  # ~29.6

        # -- keyed aperture: collar passes at ANY yaw, lug ONLY through the gap --
        col_corner = self.col_out * math.sqrt(2.0)
        assert col_corner <= self.hole_half - 0.004, \
            f"collar corner radius {col_corner:.4f} must clear the hole at any yaw"
        assert self.lug_r1 >= self.hole_half + 0.02, \
            "lug tip must overreach the hole so the plate blocks it outside the gap"
        gap_adm = math.degrees(math.asin((self.hole_half - self.lug_hw) / self.lug_r1))
        assert gap_adm >= 20.0, f"gap admission half-angle {gap_adm:.1f} deg too tight"
        # -- alignment threshold strictly inside the physically-supported range --
        assert self.support_min_deg + 15.0 <= self.align_min_deg <= 90.0, \
            f"align_min_deg must exceed the support boundary {self.support_min_deg:.1f} deg"
        # -- pin-first order is physics-forced: the seated pin's tails overlap the
        #    rising collar wall on BOTH sides for every sampled seat depth --
        assert -self.pin_len_half + self.pin_jit <= -self.col_in - 0.002, \
            "-x pin tail must still overlap the collar wall at max seat jitter"
        assert self.pin_len_half >= self.col_out + 0.004, \
            "+x pin tail must overlap the collar wall"
        assert self.q_block <= self.q_pass - 0.25, "seated pin must jam far below the shelf"
        # -- park band honesty: unique rest, unreachable un-hoisted --
        assert self.q_park - self.park_tol > self.q_pin_rest + 0.05, \
            "park band must sit far above the collar-on-pin rest"
        assert self.q_park - self.park_tol > 0.05, "park band must sit far above the floor stop"
        assert self.q_pass <= self.q_park + 0.02, "pass latch must fire on the way to park"
        assert self.q_pass >= self.q_park - 0.005, \
            "pass latch must require the lug essentially above the shelf top"
        # -- hoist clearances --
        assert self.hoist_q <= self.travel - 0.010, "hoist height must be inside the stops"
        hoist_col_bottom = self.runner_z0 + self.hoist_q - self.col_hh
        assert hoist_col_bottom >= self.plate_z1 + 0.010, \
            "hoisted collar bottom must clear the shelf top (free twist)"
        assert self.runner_z0 + self.travel + self.col_hh <= self.pole_top - 0.02, \
            "runner at the top stop must stay on the pole"
        assert self.col_in >= self.pole_half + 0.004, \
            "collar bore must clear the ROUND pole radius at every yaw"
        assert math.hypot(self.rib_xh, self.rib_out) <= self.col_in - 0.002, \
            "rib-guide corners must stay inside the collar bore (yaw must stay free)"
        assert self.rib_out >= self.rib_inner + 0.004, "rib guides need wall thickness"
        assert self.rib_out <= self.pole_half, "rib guides stay inside the pole silhouette"
        post_corner = self.plate_out * math.sqrt(2.0) - self.post_half * math.sqrt(2.0)
        lug_corner = math.hypot(self.lug_r1, self.lug_hw)
        assert post_corner >= lug_corner + 0.008, "posts must sit outside the lug swing circle"
        # -- pin channel play vs tight contact offsets --
        assert (self.chan_z1 - self.chan_z0) - 2 * self.pin_hw >= 0.002, \
            "pin needs vertical play in the channel"
        assert 2 * self.rib_inner - 2 * self.pin_hw >= 0.002, \
            "pin needs lateral play between the rib guides"
        # -- extraction rubric: latch distance already clear of everything --
        assert self.pin_latch_dist >= self.lug_r1 + 0.02, \
            "pin latch distance must clear the lug swing circle"
        assert self.pull_clear >= self.pin_latch_dist + 0.03, \
            "the solve pulls well past the latch distance"
        # -- jaw fits (parallel jaw ~80 mm) --
        assert 2 * self.knob_half <= 0.075, "pin knob must fit the jaw"
        assert self.lug_z1 - self.lug_z0 <= 0.075, "lug must fit the jaw"
        # -- rotZ band: sampled yaw and the 90-deg park twist stay off the stops --
        assert self.yaw_deg <= self.rot_lim_deg - 5.0, \
            "sampled yaw must clear the rotZ joint stops"
        assert self.rot_lim_deg <= 178.0, "rotZ band must stay off the +/-180 wrap"
        assert 90.0 + 10.0 <= self.rot_lim_deg, "park twist must clear the rotZ stops"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("umbrella_bayonet")
class UmbrellaBayonetScene(BaseScene):
    cfg: UmbrellaBayonetSceneCfg

    def __init__(self, cfg: UmbrellaBayonetSceneCfg | None = None) -> None:
        super().__init__(cfg or UmbrellaBayonetSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        pin_kwargs = dict(
            mass_props=sim_utils.MassPropertiesCfg(mass=c.pin_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            len_half=c.pin_len_half, hw=c.pin_hw, knob_half=c.knob_half,
            knob_cx=c.knob_cx, mass=c.pin_mass,
            mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
            contact_offset_tight=c.contact_offset_tight)

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
            "base": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Base",
                spawn=sp["base"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=40.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    base_half=c.base_half, base_h=c.base_h,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "mast": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Mast",
                spawn=sp["mast"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    base_h=c.base_h, pole_half=c.pole_half, pole_top=c.pole_top,
                    rib_xh=c.rib_xh, rib_out=c.rib_out,
                    chan_z0=c.chan_z0, chan_z1=c.chan_z1, rib_inner=c.rib_inner,
                    plate_z0=c.plate_z0, plate_z1=c.plate_z1,
                    hole_half=c.hole_half, plate_out=c.plate_out, post_half=c.post_half,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset,
                    contact_offset_tight=c.contact_offset_tight),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "runner": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Runner",
                spawn=sp["runner"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.runner_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    col_in=c.col_in, col_out=c.col_out, col_hh=c.col_hh,
                    lug_r0=c.lug_r0, lug_r1=c.lug_r1, lug_hw=c.lug_hw,
                    lug_z0=c.lug_z0, lug_z1=c.lug_z1,
                    mass=c.runner_mass, travel=c.travel, rot_lim_deg=c.rot_lim_deg,
                    anchor_frame=(0.0, 0.0, c.runner_z0),
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.runner_z0)),
            ),
            "pin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pin",
                spawn=sp["pin"](**pin_kwargs),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.245)),
            ),
            "spare": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Spare",
                spawn=sp["pin"](**pin_kwargs),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.28, 0.0, 0.05)),
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
        self.base: RigidObject = env.iscene["base"]
        self.mast: RigidObject = env.iscene["mast"]
        self.runner: RigidObject = env.iscene["runner"]
        self.pin: RigidObject = env.iscene["pin"]
        self.spare: RigidObject = env.iscene["spare"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.pin_latch = torch.zeros(n, device=dev)  # pin ever extracted clear
        self.pass_latch = torch.zeros(n, device=dev)  # runner ever above the shelf
        self.still_count = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: runner written 2 mm above its bottom stop with a FREE yaw,
        pin seated in the channel with +x depth jitter, spare pin dropped at a random
        polar spot on the floor with free yaw; latches zeroed. Base/Mast (the joint
        anchor and the fixture) are NEVER teleported."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        _ = torch.rand(m, 2, device=dev)  # burn (the degenerate-first-draw trap)

        # runner: bottom stop + free yaw
        yaw_amp = math.radians(c.yaw_deg)
        half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = origin + torch.tensor([0.0, 0.0, c.runner_z0 + 0.002], device=dev)
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        self.runner.write_root_state_to_sim(st, env_ids)

        # pin: seated in the channel, +x seat-depth jitter, drops ~1.5 mm to the floor
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = torch.rand(m, device=dev) * c.pin_jit
        st[:, 2] = c.pin_rest_z + 0.0015
        st[:, 0:3] += origin
        st[:, 3] = 1.0
        self.pin.write_root_state_to_sim(st, env_ids)

        # spare: random polar spot on the floor, free yaw
        r = c.spare_r_lo + torch.rand(m, device=dev) * (c.spare_r_hi - c.spare_r_lo)
        ang = math.radians(c.spare_ang_lo) + torch.rand(m, device=dev) * math.radians(
            c.spare_ang_hi - c.spare_ang_lo)
        syaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = r * torch.cos(ang)
        st[:, 1] = r * torch.sin(ang)
        st[:, 2] = 0.05
        st[:, 0:3] += origin
        st[:, 3] = torch.cos(syaw / 2)
        st[:, 6] = torch.sin(syaw / 2)
        self.spare.write_root_state_to_sim(st, env_ids)

        for t in (self.pin_latch, self.pass_latch, self.still_count):
            t[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "runner": self.runner.data.root_state_w[env_ids].clone(),
            "pin": self.pin.data.root_state_w[env_ids].clone(),
            "spare": self.spare.data.root_state_w[env_ids].clone(),
            "latches": torch.stack([
                self.pin_latch[env_ids], self.pass_latch[env_ids],
                self.still_count[env_ids]], dim=-1).clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.runner.write_root_state_to_sim(state["runner"], env_ids)
        self.pin.write_root_state_to_sim(state["pin"], env_ids)
        self.spare.write_root_state_to_sim(state["spare"], env_ids)
        lat = state["latches"]
        (self.pin_latch[env_ids], self.pass_latch[env_ids],
         self.still_count[env_ids]) = (lat[:, 0], lat[:, 1], lat[:, 2])

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A {c.pole_top * 100:.0f} cm patio-umbrella mast (dark round pole) stands "
            f"on a floor slab. The furled umbrella is a RUNNER already captive on the "
            f"pole: a blue collar (with a dark-red furled-canopy sleeve above it) that "
            f"can only SLIDE up/down the pole and SPIN about it, carrying a YELLOW "
            f"handle lug that sticks out {c.lug_r1 * 100:.1f} cm sideways. Gravity runs "
            f"the runner to the bottom of its track whenever nothing holds it.\n"
            f"At {c.pin_rest_z * 100:.1f} cm height a steel TRANSIT PIN with a RED knob "
            f"sits in a through-channel in the pole; its protruding ends block the "
            f"collar, jamming the track a few centimetres up. The pin is not attached — "
            f"pull it straight out along the knob direction (+x) and the track is free. "
            f"An identical SPARE pin lies on the floor; it is a decoy and moving it "
            f"achieves nothing.\n"
            f"At {c.plate_z1 * 100:.0f} cm height a GREEN gallery shelf ring surrounds "
            f"the pole on four corner posts. Its square hole passes the collar at any "
            f"angle, but the long yellow lug fits through only at ONE open gap sector "
            f"(toward +x, the same side as the pin knob). Above the shelf the lug can "
            f"spin freely.\n"
            f"Goal: BAYONET-PARK the umbrella at the gallery. Required order (physically "
            f"forced): pull the red-knob pin clear of the mast; spin the runner so the "
            f"yellow lug points at the gap; hoist it up the pole until the lug rises "
            f"through the gap and clears the shelf top; TWIST it ~90 deg so the lug "
            f"overhangs the green ring; lower it so the lug rests ON the ring, and let "
            f"go. Parked correctly it hangs there hands-off; released over the gap "
            f"un-twisted it falls all the way back down. Success is judged with "
            f"everything at rest: lug seated on the shelf top, twisted well away from "
            f"the gap ({c.align_min_deg:.0f} deg or more)."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pull the red-knob transit pin out of the umbrella mast and drop it clear. "
            "Spin the umbrella runner so its yellow handle lug points at the open gap "
            "in the green gallery shelf, hoist the runner up the pole through the gap, "
            "twist it about a quarter turn, and set the lug down on the green ring so "
            "the umbrella stays parked up there on its own. The spare pin on the floor "
            "is a decoy."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def runner_q(self) -> torch.Tensor:
        """(N,) runner joint position: height above the bottom stop (base never moves,
        so the mast frame is a world translation)."""
        return (self.runner.data.root_pos_w - self.env_origins)[:, 2] - self.cfg.runner_z0

    def lug_heading_cos(self) -> torch.Tensor:
        """(N,) cosine of the lug heading vs the gap heading (+x). The lug points
        along body +x; rotate (1,0,0) by the root quat and project horizontally."""
        q = self.runner.data.root_quat_w  # (w, x, y, z)
        dx = 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2)
        dy = 2.0 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
        return dx / torch.sqrt(dx**2 + dy**2 + 1e-9)

    def pin_dist(self) -> torch.Tensor:
        """(N,) pin horizontal distance from the mast axis."""
        loc = self.pin.data.root_pos_w - self.env_origins
        return loc[:, 0:2].norm(dim=-1)

    def parked_now(self) -> torch.Tensor:
        """(N,) bool: runner joint height inside the park band."""
        c = self.cfg
        return (self.runner_q() - c.q_park).abs() <= c.park_tol

    def aligned_now(self) -> torch.Tensor:
        """(N,) bool: lug heading at least align_min_deg away from the gap heading."""
        return self.lug_heading_cos() <= math.cos(math.radians(self.cfg.align_min_deg))

    def _still_now(self) -> torch.Tensor:
        """(N,) bool: runner AND pin quiet — INSTANTANEOUS (never judge on this alone;
        the counter-latch below is what `settled()` reads)."""
        c = self.cfg
        return ((self.runner.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.runner.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
                & (self.pin.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin))

    def settled(self) -> torch.Tensor:
        """(N,) bool: stillness has PERSISTED `settle_steps_min` consecutive steps."""
        return self.still_count >= self.cfg.settle_steps_min

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Stillness counter + progress latches: pin ever extracted clear of the mast
        axis; runner ever above the shelf (physically reachable only with the pin out
        AND the lug through the keyed gap)."""
        c = self.cfg
        self.still_count = (self.still_count + 1.0) * self._still_now().float()
        self.pin_latch = torch.maximum(
            self.pin_latch, (self.pin_dist() >= c.pin_latch_dist).float())
        self.pass_latch = torch.maximum(
            self.pass_latch, (self.runner_q() >= c.q_pass).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: lug seated on the shelf (park band), twisted away from the gap,
        the pin actually extracted and the shelf actually passed (latches — a
        teleported-to-park runner earns nothing), everything persistently still."""
        return (self.parked_now() & self.aligned_now()
                & (self.pin_latch > 0.5) & (self.pass_latch > 0.5) & self.settled())

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.30 pin ever extracted clear + 0.30 runner ever above
        the shelf — latched, credit never evaporates; capped 0.60; 1.0 iff success().
        Null policy ~0 (the runner starts on the bottom stop, the pin seated)."""
        base = (0.30 * self.pin_latch + 0.30 * self.pass_latch).clamp(0.0, 0.60)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="umbrella_bayonet", robot="null", env_spacing=3.0))
