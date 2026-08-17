"""HitchTowScene — hitch the tractor to the tunnel-bound trailer with the drop pin and
TOW the coupled rig onto the dock pad (sim_gen task
`living_room_scene4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray_i391`).

Derived from libero_90/living_room_scene4_stack_the_left_bowl_on_the_right_bowl_and_
place_them_in_the_tray, but STRATEGICALLY different: the seed is a free-space
pick-and-place composition — grasp one bowl, stack it on the other (an xy/dz readout),
carry the pair into a tray's bounding box. Every subgoal is "lift object, put it
there". Here the goal object CANNOT be lifted or placed at all: the crimson TRAILER is
a rail cart parked under a low tunnel roof (its body is never reachable; only its
coupling tongue sticks out of the tunnel mouth), and the delivery must be produced by
building a MECHANICAL COUPLING and towing through it — push the blue TRACTOR cart
rearward until its overhanging coupler plate rides over the trailer tongue and stops
on the tongue's stop block (the hard stop is the alignment jig: at contact the two
square pin bores are coaxial), drop/press the red-capped shear pin down through both
bores, then pull the tractor forward so the pin — loaded in shear between the plates —
drags the trailer out of the tunnel and onto the green dock pad. A solver needs a
different PLAN (mate -> pin -> tow, force transmitted through a joint the solver
creates) and different code STRUCTURE (prismatic rail carts, lap-joint coupling,
shear-pin drag) — not stack-and-carry.

success(): trailer on the dock pad (trailer_x >= dock_lo), STILL COUPLED (the pin
threads both bores between the mated plates — "delivered hitched"), the tow latch set
(the trailer accumulated >= dist_min of forward travel WHILE coupled, with per-step
credit clamped to step_cap so a teleport jump credits ~one step — the trailer must
have been genuinely towed across the yard through the pin), and everything
PERSISTENTLY still (stillness counter-latch).

score(), latched (credit never evaporates): 0.25 once the carts have ever been MATED
(coupler plate on the stop block, bores aligned); 0.35 once the pin has ever been
COUPLED (threading both bores); capped 0.60; 1.0 iff success(). Null policy ~0 (the
trailer starts deep in the tunnel, the pin in its socket, the tractor mid-rail).

Assets are fully procedural (compound spawners; explicit friction materials;
mass/CoM/inertia AUTHORED — custom spawn funcs apply no cfg schemas):
  - bed (KINEMATIC, the rail bed slab — the D6 anchor for both carts; the joint
    pair-collision filter silences it vs the carts, which never touch it: they ride
    their joints. The free pin CAN land on it — that contact stays live).
  - yard (KINEMATIC, separate body — all its contacts stay live): the tunnel (side
    walls + LOW ROOF over the trailer park), the green dock pad, and the pin stand
    (pedestal + socket the pin starts standing in).
  - tractor (dynamic, spawn-authored D6: transX free in [l_lo, l_hi], all else
    locked): blue body crate + yellow grasp mast + rear overhanging coupler plate
    with a square bore.
  - trailer (dynamic, same D6 pattern, transX in [t_lo, t_hi]): crimson body crate
    (under the roof) + forward tongue plate with a square bore + stop block.
  - pin (dynamic, free): steel shank cylinder + RED square cap (cap inradius >
    bore circumradius: it can never fall through).

Per-episode randomization (readback-verified in smoke): trailer park depth in the
tunnel, tractor start position, pin yaw in its socket.

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
    """Author one z-axis cylinder child prim (the pin shank — round so it threads the
    square bores at any yaw and rolls to bear flat on a bore wall under shear)."""
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


def _author_cart_joint(stage, prim_path: str, anchor_z: float, lo: float, hi: float) -> None:
    """Spawn-authored generic D6 joint vs the sibling Bed: transX free in [lo, hi]
    (world x — identity joint frames), EVERY other axis locked (low > high). The cart
    rides the joint — it never contacts the bed."""
    from pxr import Gf, UsdPhysics

    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.Joint.Define(stage, f"{prim_path}/rail")
    j.CreateBody0Rel().SetTargets([f"{base}/Bed"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(False)
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(anchor_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    for axis in ("transY", "transZ", "rotX", "rotY", "rotZ"):  # locked: low > high
        la = UsdPhysics.LimitAPI.Apply(j.GetPrim(), axis)
        la.CreateLowAttr(1.0)
        la.CreateHighAttr(-1.0)
    la = UsdPhysics.LimitAPI.Apply(j.GetPrim(), "transX")
    la.CreateLowAttr(float(lo))
    la.CreateHighAttr(float(hi))


def _spawn_bed(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC rail-bed slab — the D6 anchor for both carts. The carts never touch
    it (they float on their joints); the free pin can land on it (live contact)."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(60.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    _box(stage, f"{prim_path}/slab",
         (cfg.bed_x1 - cfg.bed_x0, 2 * cfg.bed_hy, cfg.bed_top),
         ((cfg.bed_x0 + cfg.bed_x1) / 2, 0.0, cfg.bed_top / 2),
         (0.42, 0.40, 0.38), cfg.contact_offset, material=mat)
    return root


def _spawn_yard(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC yard fixtures (separate body — contacts stay live): tunnel side
    walls + back wall + LOW ROOF over the trailer park, green dock pad, pin stand
    (pedestal + 4 socket walls)."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(50.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    dark = (0.22, 0.22, 0.26)
    tx0, tx1 = cfg.tun_x0, cfg.tun_x1
    txm, txl = (tx0 + tx1) / 2, tx1 - tx0
    wz = (cfg.bed_top + cfg.roof_z1) / 2
    wh = cfg.roof_z1 - cfg.bed_top
    # tunnel side walls, back wall, roof
    for s, nm in ((1.0, "wall_n"), (-1.0, "wall_s")):
        _box(stage, f"{prim_path}/{nm}",
             (txl, cfg.wall_y1 - cfg.wall_y0, wh),
             (txm, s * (cfg.wall_y0 + cfg.wall_y1) / 2, wz), dark, co, material=mat)
    _box(stage, f"{prim_path}/wall_back", (0.03, 2 * cfg.wall_y1, wh),
         (tx0 - 0.015, 0.0, wz), dark, co, material=mat)
    _box(stage, f"{prim_path}/roof", (txl, 2 * cfg.wall_y1, cfg.roof_z1 - cfg.roof_z0),
         (txm, 0.0, (cfg.roof_z0 + cfg.roof_z1) / 2), (0.30, 0.30, 0.36), co, material=mat)
    # green dock pad (proud of the bed top; nothing ever rests on it)
    _box(stage, f"{prim_path}/dock_pad",
         (cfg.pad_x1 - cfg.pad_x0, 2 * cfg.pad_hy, cfg.pad_h),
         ((cfg.pad_x0 + cfg.pad_x1) / 2, 0.0, cfg.bed_top + cfg.pad_h / 2),
         (0.10, 0.62, 0.18), co, material=mat)
    # pin stand: pedestal + 4 socket walls around the pin tip
    sx, sy = cfg.stand_x, cfg.stand_y
    _box(stage, f"{prim_path}/pedestal", (2 * cfg.stand_half, 2 * cfg.stand_half, cfg.stand_h),
         (sx, sy, cfg.stand_h / 2), (0.50, 0.44, 0.34), co, material=mat)
    oh, wt = cfg.socket_half, cfg.socket_wall
    swz = cfg.stand_h + cfg.socket_hh
    _box(stage, f"{prim_path}/sock_e", (wt, 2 * (oh + wt), 2 * cfg.socket_hh),
         (sx + oh + wt / 2, sy, swz), (0.50, 0.44, 0.34), co, material=mat)
    _box(stage, f"{prim_path}/sock_w", (wt, 2 * (oh + wt), 2 * cfg.socket_hh),
         (sx - oh - wt / 2, sy, swz), (0.50, 0.44, 0.34), co, material=mat)
    _box(stage, f"{prim_path}/sock_n", (2 * oh, wt, 2 * cfg.socket_hh),
         (sx, sy + oh + wt / 2, swz), (0.50, 0.44, 0.34), co, material=mat)
    _box(stage, f"{prim_path}/sock_s", (2 * oh, wt, 2 * cfg.socket_hh),
         (sx, sy - oh - wt / 2, swz), (0.50, 0.44, 0.34), co, material=mat)
    return root


def _bore_plate(stage, prim_path: str, name: str, x0: float, x1: float, bore_x: float,
                bore_half: float, hy: float, zc: float, hz: float, color,
                contact_offset: float, material) -> None:
    """A horizontal plate spanning [x0, x1] x [-hy, hy] with a square through-bore of
    half-width `bore_half` centered at (bore_x, 0): 2 full-width end segments + 2
    side strips."""
    bx0, bx1 = bore_x - bore_half, bore_x + bore_half
    _box(stage, f"{prim_path}/{name}_a", (bx0 - x0, 2 * hy, 2 * hz),
         ((x0 + bx0) / 2, 0.0, zc), color, contact_offset, material=material)
    _box(stage, f"{prim_path}/{name}_b", (x1 - bx1, 2 * hy, 2 * hz),
         ((bx1 + x1) / 2, 0.0, zc), color, contact_offset, material=material)
    for s, tag in ((1.0, "n"), (-1.0, "s")):
        _box(stage, f"{prim_path}/{name}_{tag}", (2 * bore_half, hy - bore_half, 2 * hz),
             (bore_x, s * (bore_half + hy) / 2, zc), color, contact_offset, material=material)


def _spawn_tractor(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The blue TRACTOR cart: body crate + yellow grasp mast + rear overhanging
    coupler plate (UPPER lap plate) with a square bore. One dynamic body on a
    spawn-authored transX D6 vs the sibling Bed."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass, (0.04, 0.0, 0.0), (0.004, 0.004, 0.004),
                   lin_damp=0.6, ang_damp=0.6)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co, cot = cfg.contact_offset, cfg.contact_offset_tight
    blue = (0.18, 0.32, 0.78)
    _box(stage, f"{prim_path}/body",
         (cfg.body_x1 - cfg.body_x0, 2 * cfg.body_hy, 2 * cfg.body_hz),
         ((cfg.body_x0 + cfg.body_x1) / 2, 0.0, 0.0), blue, co, material=mat)
    _box(stage, f"{prim_path}/mast", (0.022, 0.022, cfg.mast_h),
         (cfg.mast_x, 0.0, cfg.body_hz + cfg.mast_h / 2), (0.95, 0.80, 0.10),
         co, material=mat)
    _bore_plate(stage, prim_path, "plate", cfg.plate_x0, cfg.body_x0, -cfg.bore_a,
                cfg.bore_half, cfg.plate_hy, cfg.plate_zc, cfg.plate_hz,
                (0.55, 0.58, 0.65), cot, mat)
    _author_cart_joint(stage, prim_path, cfg.ride_z, cfg.q_lo, cfg.q_hi)
    return root


def _spawn_trailer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The crimson TRAILER cart: body crate (lives under the tunnel roof) + forward
    tongue plate (LOWER lap plate) with a square bore + the stop block that jigs the
    mate. Same transX D6 pattern vs the sibling Bed."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass, (0.0, 0.0, 0.0), (0.003, 0.003, 0.003),
                   lin_damp=0.6, ang_damp=0.6)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co, cot = cfg.contact_offset, cfg.contact_offset_tight
    crimson = (0.72, 0.12, 0.16)
    _box(stage, f"{prim_path}/body", (2 * cfg.body_hx, 2 * cfg.body_hy, 2 * cfg.body_hz),
         (0.0, 0.0, 0.0), crimson, co, material=mat)
    _bore_plate(stage, prim_path, "tongue", cfg.body_hx, cfg.tongue_x1, cfg.bore_b,
                cfg.bore_half, cfg.tongue_hy, cfg.tongue_zc, cfg.tongue_hz,
                (0.60, 0.48, 0.30), cot, mat)
    _box(stage, f"{prim_path}/stop_block",
         (cfg.stop_x1 - cfg.body_hx, 2 * cfg.tongue_hy, cfg.stop_z1 - cfg.stop_z0),
         ((cfg.body_hx + cfg.stop_x1) / 2, 0.0, (cfg.stop_z0 + cfg.stop_z1) / 2),
         crimson, cot, material=mat)
    _author_cart_joint(stage, prim_path, cfg.ride_z, cfg.q_lo, cfg.q_hi)
    return root


def _spawn_pin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The shear pin: steel shank cylinder (body z axis, origin at shank center) +
    RED square cap. One free dynamic body; CoM authored at the body origin."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass, (0.0, 0.0, 0.0), (1.5e-5, 1.5e-5, 6.0e-6),
                   lin_damp=0.1, ang_damp=0.3)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    cot = cfg.contact_offset_tight
    _cyl(stage, f"{prim_path}/shank", cfg.shank_r, cfg.shank_len,
         (0.0, 0.0, 0.0), (0.72, 0.74, 0.78), cot, material=mat)
    _box(stage, f"{prim_path}/cap", (2 * cfg.cap_half, 2 * cfg.cap_half, cfg.cap_h),
         (0.0, 0.0, cfg.shank_len / 2 + cfg.cap_h / 2), (0.85, 0.12, 0.12),
         cot, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazy: module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bed" not in _SPAWNER_CACHE:

        @configclass
        class BedSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bed)
            bed_x0: float = -0.70
            bed_x1: float = 0.80
            bed_hy: float = 0.10
            bed_top: float = 0.04
            mu_static: float = 0.5
            mu_dynamic: float = 0.4
            contact_offset: float = 0.002

        @configclass
        class YardSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_yard)
            bed_top: float = 0.04
            tun_x0: float = -0.62
            tun_x1: float = -0.26
            wall_y0: float = 0.075
            wall_y1: float = 0.095
            roof_z0: float = 0.20
            roof_z1: float = 0.22
            pad_x0: float = 0.12
            pad_x1: float = 0.42
            pad_hy: float = 0.08
            pad_h: float = 0.005
            stand_x: float = 0.10
            stand_y: float = -0.20
            stand_half: float = 0.04
            stand_h: float = 0.10
            socket_half: float = 0.012
            socket_wall: float = 0.010
            socket_hh: float = 0.015
            mu_static: float = 0.5
            mu_dynamic: float = 0.4
            contact_offset: float = 0.002

        @configclass
        class TractorSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tractor)
            body_x0: float = -0.03
            body_x1: float = 0.11
            body_hy: float = 0.06
            body_hz: float = 0.05
            mast_x: float = 0.04
            mast_h: float = 0.10
            plate_x0: float = -0.20
            plate_zc: float = -0.030
            plate_hz: float = 0.006
            plate_hy: float = 0.03
            bore_a: float = 0.08
            bore_half: float = 0.011
            ride_z: float = 0.12
            q_lo: float = -0.14
            q_hi: float = 0.64
            mass: float = 0.8
            mu_static: float = 0.5
            mu_dynamic: float = 0.4
            contact_offset: float = 0.002
            contact_offset_tight: float = 0.0005

        @configclass
        class TrailerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_trailer)
            body_hx: float = 0.08
            body_hy: float = 0.06
            body_hz: float = 0.05
            tongue_x1: float = 0.26
            tongue_zc: float = -0.045
            tongue_hz: float = 0.006
            tongue_hy: float = 0.03
            bore_b: float = 0.22
            bore_half: float = 0.011
            stop_x1: float = 0.10
            stop_z0: float = -0.039
            stop_z1: float = -0.015
            ride_z: float = 0.12
            q_lo: float = -0.42
            q_hi: float = 0.34
            mass: float = 0.6
            mu_static: float = 0.5
            mu_dynamic: float = 0.4
            contact_offset: float = 0.002
            contact_offset_tight: float = 0.0005

        @configclass
        class PinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pin)
            shank_r: float = 0.007
            shank_len: float = 0.040
            cap_half: float = 0.018
            cap_h: float = 0.010
            mass: float = 0.05
            mu_static: float = 0.5
            mu_dynamic: float = 0.4
            contact_offset_tight: float = 0.0005

        _SPAWNER_CACHE.update(bed=BedSpawnerCfg, yard=YardSpawnerCfg,
                              tractor=TractorSpawnerCfg, trailer=TrailerSpawnerCfg,
                              pin=PinSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class HitchTowSceneCfg(BaseCfg):
    """Config for `HitchTowScene`. `__post_init__` asserts the strategic honesty
    invariants with pre-computed geometry: the stop block jigs the mate so the two
    bores are exactly coaxial at contact; the lap plates pass with clearance; the pin
    threads both bores with real slop, its cap can never fall through, and it spans
    both plates when seated; the trailer body stays under the roof over its whole
    reachable band while its tongue bore stays OUTSIDE the tunnel mouth for every
    mate position; the dock lies far enough from every spawn that the tow latch
    (dist_min of coupled travel) is strictly necessary; every grasp fits a parallel
    jaw."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    dock_lo: float = tunable(0.20)  # trailer_x at/right of this = on the dock pad
    dist_min: float = tunable(0.40)  # required coupled forward travel of the trailer (m)
    step_cap: float = tunable(0.004)  # per-step tow-travel credit clamp (m/step)
    mate_tol: float = tunable(0.010)  # |rel_dx - mate_dx| for mated_now / the mate latch
    couple_xy_tol: float = tunable(0.009)  # pin axis to EACH bore center, horizontal (m)
    couple_z_tol: float = tunable(0.008)  # pin center to the seated height (m)
    pin_tilt_max_deg: float = tunable(20.0)  # pin axis within this of vertical
    settle_lin: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.60)  # max pin |ang vel| when judging (rad/s)
    settle_steps_min: int = tunable(30)  # stillness must PERSIST this many steps

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    t_spawn_lo: float = tunable(-0.41)  # trailer park depth band in the tunnel ...
    t_spawn_hi: float = tunable(-0.37)
    l_spawn_lo: float = tunable(0.02)  # tractor start band, mid-rail ...
    l_spawn_hi: float = tunable(0.14)

    # --- info: structure (bed/yard at the env origin, NEVER teleported) ----------------------
    bed_x0: float = info(-0.70)
    bed_x1: float = info(0.80)
    bed_hy: float = info(0.10)
    bed_top: float = info(0.04)
    ride_z: float = info(0.12)  # cart body-origin height (both carts, joint-locked)
    tun_x0: float = info(-0.62)  # tunnel span ...
    tun_x1: float = info(-0.26)  # ... mouth (the trailer body must always stay left)
    wall_y0: float = info(0.075)  # tunnel side-wall inner face |y| ...
    wall_y1: float = info(0.095)
    roof_z0: float = info(0.20)  # roof underside (trailer body top + 30 mm)
    roof_z1: float = info(0.22)
    pad_x0: float = info(0.12)  # green dock pad span (visual; the rubric reads dock_lo)
    pad_x1: float = info(0.42)
    pad_hy: float = info(0.08)
    pad_h: float = info(0.005)
    stand_x: float = info(0.10)  # pin stand center ...
    stand_y: float = info(-0.20)
    stand_half: float = info(0.04)
    stand_h: float = info(0.10)  # pedestal top = socket floor
    socket_half: float = info(0.012)  # socket opening half-width (pin tip slop 5 mm)
    socket_wall: float = info(0.010)
    socket_hh: float = info(0.015)  # socket wall half-height above the floor
    # tractor
    l_body_x0: float = info(-0.03)  # body crate span, cart frame ...
    l_body_x1: float = info(0.11)
    body_hy: float = info(0.06)
    body_hz: float = info(0.05)  # both carts: body z = ride_z +/- body_hz
    mast_h: float = info(0.10)  # yellow grasp mast (22 mm square)
    plate_x0: float = info(-0.20)  # coupler plate tip (the mate contact face) ...
    plate_zc: float = info(-0.030)  # plate center, cart frame (env z 0.090)
    plate_hz: float = info(0.006)
    plate_hy: float = info(0.03)
    bore_a: float = info(0.08)  # tractor bore center at cart_x - bore_a
    l_lo: float = info(-0.14)  # tractor joint stops ...
    l_hi: float = info(0.64)
    tractor_mass: float = info(0.8)
    # trailer
    t_body_hx: float = info(0.08)
    tongue_x1: float = info(0.26)  # tongue tip, cart frame
    tongue_zc: float = info(-0.045)  # tongue center, cart frame (env z 0.075)
    tongue_hz: float = info(0.006)
    tongue_hy: float = info(0.03)
    bore_b: float = info(0.22)  # trailer bore center at cart_x + bore_b
    stop_x1: float = info(0.10)  # stop block +x face: the mate jig
    stop_z0: float = info(-0.039)  # stop block z band, cart frame ...
    stop_z1: float = info(-0.015)
    t_lo: float = info(-0.42)  # trailer joint stops ...
    t_hi: float = info(0.34)
    trailer_mass: float = info(0.6)
    # pin
    shank_r: float = info(0.007)
    shank_len: float = info(0.040)
    cap_half: float = info(0.018)  # square cap half-width (36 mm across flats)
    cap_h: float = info(0.010)
    bore_half: float = info(0.011)  # square bore half-width (both plates)
    pin_mass: float = info(0.05)
    mu_static: float = info(0.5)
    mu_dynamic: float = info(0.4)
    contact_offset: float = info(0.002)
    contact_offset_tight: float = info(0.0005)  # bores/pin/plates (4 mm real slop)

    # Derived (filled in __post_init__).
    mate_dx: float = field(default=None, init=False)  # tractor_x - trailer_x at the stop
    pin_seat_z: float = field(default=None, init=False)  # seated pin center height (env z)
    pin_socket_z: float = field(default=None, init=False)  # pin center standing in the socket
    plate_z_lo: float = field(default=None, init=False)  # upper plate underside (env z)
    plate_z_hi: float = field(default=None, init=False)  # upper plate top (env z)
    tongue_z_hi: float = field(default=None, init=False)  # lower plate top (env z)

    def __post_init__(self) -> None:
        c = self
        c.mate_dx = c.stop_x1 - c.plate_x0  # 0.30: plate tip on the stop block face
        c.plate_z_lo = c.ride_z + c.plate_zc - c.plate_hz  # 0.084
        c.plate_z_hi = c.ride_z + c.plate_zc + c.plate_hz  # 0.096
        c.tongue_z_hi = c.ride_z + c.tongue_zc + c.tongue_hz  # 0.081
        # seated: cap underside on the upper plate top; origin at shank center
        c.pin_seat_z = c.plate_z_hi - c.shank_len / 2  # 0.076
        c.pin_socket_z = c.stand_h + c.shank_len / 2  # 0.120 (tip on the socket floor)

        # -- the stop block IS the alignment jig: bores coaxial exactly at contact --
        assert abs((c.mate_dx - c.bore_a) - c.bore_b) < 1e-9, \
            "bores must be coaxial when the plate tip contacts the stop block"
        # -- lap plates pass: upper plate underside clears the tongue top --
        assert c.plate_z_lo - c.tongue_z_hi >= 0.002, "lap plates must pass with clearance"
        # -- the stop block actually catches the incoming plate tip --
        assert c.ride_z + c.stop_z0 <= c.plate_z_lo - 0.002, "stop block must start below the plate"
        assert c.ride_z + c.stop_z1 >= c.plate_z_hi + 0.002, "stop block must top out above it"
        # -- pin threads with real slop; cap can never fall through; shank spans both --
        slop = c.bore_half - c.shank_r
        assert slop >= 0.003, f"pin needs radial slop in the bore (has {slop * 1000:.1f} mm)"
        assert c.cap_half >= c.bore_half * math.sqrt(2.0) + 0.002, \
            "cap inradius must exceed the bore circumradius (cap never falls through)"
        tip_z = c.pin_seat_z - c.shank_len / 2  # 0.056
        tongue_z_lo = c.ride_z + c.tongue_zc - c.tongue_hz  # 0.069
        assert tip_z <= tongue_z_lo - 0.004, "seated shank must protrude below the lower plate"
        assert c.pin_seat_z + c.shank_len / 2 >= c.plate_z_hi - 1e-9, \
            "seated shank must fill the upper plate"
        assert tip_z >= c.bed_top + 0.010, "seated pin tip must clear the bed"
        # -- couple tolerances honest vs geometry: a pin against a bore wall is offset
        #    exactly `slop`; tolerance must cover it but stay under the bore pitch --
        assert c.couple_xy_tol >= slop + 0.003, "couple tol must cover wall-contact offsets"
        assert c.couple_xy_tol <= c.bore_half, "couple tol must stay inside the bore"
        # -- trailer body roofed over its whole reachable band; tongue bore outside --
        assert c.t_lo - c.t_body_hx >= c.tun_x0 + 0.02, "trailer at its stop stays in the tunnel"
        assert c.t_spawn_hi + c.t_body_hx <= c.tun_x1 - 0.02, \
            "trailer body must stay under the roof at the shallowest spawn"
        assert c.roof_z0 >= c.ride_z + c.body_hz + 0.02, "roof must clear the trailer body"
        assert c.wall_y0 >= c.body_hy + 0.01, "tunnel walls must clear the trailer body"
        assert c.t_lo + c.bore_b >= c.tun_x1 + 0.03, \
            "the tongue bore must sit OUTSIDE the tunnel mouth at every mate position"
        # -- mate reachable inside the tractor stops; plate/tongue pass under the roof --
        assert c.t_lo + c.mate_dx >= c.l_lo + 0.01, "mate at the deepest trailer must be reachable"
        assert c.plate_z_hi <= c.roof_z0 - 0.02, "the coupler plate passes under the roof"
        assert c.wall_y0 >= c.plate_hy + 0.01, "the coupler plate passes between the walls"
        # -- the mate never lets the tongue tip reach the tractor body --
        assert c.tongue_x1 <= c.mate_dx + c.l_body_x0 - 0.005, \
            "tongue tip must stop short of the tractor body at full mate"
        # -- dock reachable; the tow latch strictly necessary --
        assert c.dock_lo >= c.pad_x0 + 0.02 and c.dock_lo <= c.pad_x1 - c.t_body_hx * 2, \
            "dock threshold must lie on the pad"
        assert c.dock_lo + 0.05 <= c.t_hi - 0.02, "dock band must be inside the trailer stops"
        assert c.dock_lo + c.mate_dx <= c.l_hi - 0.02, "towing to dock must fit the tractor stops"
        assert c.dock_lo - c.t_spawn_hi >= c.dist_min + 0.05, \
            "dist_min must be strictly less than the shortest spawn->dock run"
        assert c.dist_min >= 0.5 * (c.dock_lo - c.t_spawn_lo), \
            "dist_min must demand most of the run be towed coupled"
        # -- per-step tow credit cap: generous vs a real tow, tiny vs a teleport jump --
        assert c.step_cap >= 3.0 * 0.12 / 120.0, "cap must clear a 0.12 m/s tow at 120 Hz"
        assert c.step_cap <= c.dist_min / 50.0, "a teleport jump must credit < 2% of dist_min"
        # -- spawn sanity --
        assert c.t_spawn_lo >= c.t_lo + 0.005 and c.t_spawn_hi <= c.tun_x1 - c.t_body_hx - 0.02
        assert c.l_spawn_lo >= c.l_lo + 0.05 and c.l_spawn_hi <= c.l_hi - 0.05
        # tractor at spawn never overlaps the trailer tongue's stop block region
        assert c.l_spawn_lo - 0.20 >= c.t_spawn_hi + c.stop_x1 + 0.03, \
            "tractor plate tip at spawn must stay clear of the stop block"
        # -- pin stand clear of the rail corridor; pin standing pose stable-ish --
        assert abs(c.stand_y) - c.stand_half >= c.bed_hy + 0.02, "stand off the rail bed"
        assert c.socket_half - c.shank_r >= 0.003, "pin tip needs slop in the socket"
        assert c.stand_h + 2 * c.socket_hh <= c.pin_socket_z + c.shank_len / 2 - 0.008, \
            "socket walls must stay below the cap (cap graspable)"
        # -- jaw fits (parallel jaw ~80 mm) --
        assert 2 * c.cap_half <= 0.075, "pin cap must fit the jaw"
        assert 0.022 <= 0.075, "mast must fit the jaw"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("hitch_tow")
class HitchTowScene(BaseScene):
    cfg: HitchTowSceneCfg

    def __init__(self, cfg: HitchTowSceneCfg | None = None) -> None:
        super().__init__(cfg or HitchTowSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
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
            "bed": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bed",
                spawn=sp["bed"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=60.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    bed_x0=c.bed_x0, bed_x1=c.bed_x1, bed_hy=c.bed_hy, bed_top=c.bed_top,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "yard": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Yard",
                spawn=sp["yard"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    bed_top=c.bed_top, tun_x0=c.tun_x0, tun_x1=c.tun_x1,
                    wall_y0=c.wall_y0, wall_y1=c.wall_y1,
                    roof_z0=c.roof_z0, roof_z1=c.roof_z1,
                    pad_x0=c.pad_x0, pad_x1=c.pad_x1, pad_hy=c.pad_hy, pad_h=c.pad_h,
                    stand_x=c.stand_x, stand_y=c.stand_y, stand_half=c.stand_half,
                    stand_h=c.stand_h, socket_half=c.socket_half,
                    socket_wall=c.socket_wall, socket_hh=c.socket_hh,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "tractor": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tractor",
                spawn=sp["tractor"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.tractor_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    body_x0=c.l_body_x0, body_x1=c.l_body_x1,
                    body_hy=c.body_hy, body_hz=c.body_hz,
                    mast_x=0.04, mast_h=c.mast_h,
                    plate_x0=c.plate_x0, plate_zc=c.plate_zc, plate_hz=c.plate_hz,
                    plate_hy=c.plate_hy, bore_a=c.bore_a, bore_half=c.bore_half,
                    ride_z=c.ride_z, q_lo=c.l_lo, q_hi=c.l_hi, mass=c.tractor_mass,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset,
                    contact_offset_tight=c.contact_offset_tight),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.08, 0.0, c.ride_z)),
            ),
            "trailer": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Trailer",
                spawn=sp["trailer"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.trailer_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    body_hx=c.t_body_hx, body_hy=c.body_hy, body_hz=c.body_hz,
                    tongue_x1=c.tongue_x1, tongue_zc=c.tongue_zc, tongue_hz=c.tongue_hz,
                    tongue_hy=c.tongue_hy, bore_b=c.bore_b, bore_half=c.bore_half,
                    stop_x1=c.stop_x1, stop_z0=c.stop_z0, stop_z1=c.stop_z1,
                    ride_z=c.ride_z, q_lo=c.t_lo, q_hi=c.t_hi, mass=c.trailer_mass,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset,
                    contact_offset_tight=c.contact_offset_tight),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.38, 0.0, c.ride_z)),
            ),
            "pin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pin",
                spawn=sp["pin"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.pin_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    shank_r=c.shank_r, shank_len=c.shank_len,
                    cap_half=c.cap_half, cap_h=c.cap_h, mass=c.pin_mass,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset_tight=c.contact_offset_tight),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_x, c.stand_y, c.pin_socket_z + 0.002)),
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
        self.bed: RigidObject = env.iscene["bed"]
        self.yard: RigidObject = env.iscene["yard"]
        self.tractor: RigidObject = env.iscene["tractor"]
        self.trailer: RigidObject = env.iscene["trailer"]
        self.pin: RigidObject = env.iscene["pin"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.mate_latch = torch.zeros(n, device=dev)  # carts ever mated at the stop
        self.pin_latch = torch.zeros(n, device=dev)  # pin ever threading both bores
        self.tow_travel = torch.zeros(n, device=dev)  # coupled forward travel (clamped/step)
        self.still_count = torch.zeros(n, device=dev)
        self._prev_tx = torch.zeros(n, device=dev)  # trailer x last step (env-local)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: trailer parked at a random depth in the tunnel, tractor at
        a random mid-rail start, pin standing in the stand socket with a random yaw
        (all joint-consistent writes); latches/odometer zeroed. Bed/Yard (the joint
        anchor and the fixtures) are NEVER teleported."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        _ = torch.rand(m, 2, device=dev)  # burn (the degenerate-first-draw trap)

        tx = c.t_spawn_lo + torch.rand(m, device=dev) * (c.t_spawn_hi - c.t_spawn_lo)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = tx
        st[:, 2] = c.ride_z
        st[:, 0:3] += origin
        st[:, 3] = 1.0
        self.trailer.write_root_state_to_sim(st, env_ids)

        lx = c.l_spawn_lo + torch.rand(m, device=dev) * (c.l_spawn_hi - c.l_spawn_lo)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = lx
        st[:, 2] = c.ride_z
        st[:, 0:3] += origin
        st[:, 3] = 1.0
        self.tractor.write_root_state_to_sim(st, env_ids)

        half = (torch.rand(m, device=dev) * 2 - 1) * math.pi / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.stand_x
        st[:, 1] = c.stand_y
        st[:, 2] = c.pin_socket_z + 0.002
        st[:, 0:3] += origin
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        self.pin.write_root_state_to_sim(st, env_ids)

        for t in (self.mate_latch, self.pin_latch, self.tow_travel, self.still_count):
            t[env_ids] = 0.0
        self._prev_tx[env_ids] = tx

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "tractor": self.tractor.data.root_state_w[env_ids].clone(),
            "trailer": self.trailer.data.root_state_w[env_ids].clone(),
            "pin": self.pin.data.root_state_w[env_ids].clone(),
            "latches": torch.stack([
                self.mate_latch[env_ids], self.pin_latch[env_ids],
                self.tow_travel[env_ids], self.still_count[env_ids],
                self._prev_tx[env_ids]], dim=-1).clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.tractor.write_root_state_to_sim(state["tractor"], env_ids)
        self.trailer.write_root_state_to_sim(state["trailer"], env_ids)
        self.pin.write_root_state_to_sim(state["pin"], env_ids)
        lat = state["latches"]
        (self.mate_latch[env_ids], self.pin_latch[env_ids], self.tow_travel[env_ids],
         self.still_count[env_ids], self._prev_tx[env_ids]) = (
            lat[:, 0], lat[:, 1], lat[:, 2], lat[:, 3], lat[:, 4])

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A straight freight rail runs along x on a low bed slab. Both carts on it "
            "can ONLY slide along the rail. At the left (-x) end a dark TUNNEL with a "
            "low roof covers the parking strip: inside it stands the CRIMSON TRAILER, "
            "a crate-shaped cart whose body is completely under the roof — it cannot "
            "be reached, grasped, lifted or pushed there. Only its coupling TONGUE (a "
            "tan horizontal plate with a square hole, carrying a small crimson stop "
            "block at its root) sticks out of the tunnel mouth into the open.\n"
            "Mid-rail stands the BLUE TRACTOR cart with a YELLOW grasp mast on top. "
            "From its rear (-x) face a grey COUPLER PLATE overhangs at a height just "
            "above the trailer tongue; the plate has the same square hole in it.\n"
            f"On a wooden pedestal beside the rail (y={c.stand_y:+.2f}) a steel SHEAR "
            "PIN with a RED square cap stands upright in a socket.\n"
            f"At the right (+x) end a GREEN DOCK PAD marks the delivery zone "
            f"(trailer center at x >= {c.dock_lo:.2f}).\n"
            "Goal: DELIVER THE TRAILER TO THE DOCK PAD, HITCHED. Required order "
            "(physically forced by the roof): (1) MATE — push the tractor rearward "
            "(-x, e.g. by its yellow mast) until its coupler plate slides over the "
            "trailer tongue and stops against the stop block; at that hard stop the "
            "two square holes are exactly aligned, one above the other. (2) HITCH — "
            "take the red-capped pin from its socket and press it straight down "
            "through both aligned holes until its cap rests on the coupler plate "
            "(4 mm of slop; it drops most of the way). (3) TOW — pull the tractor "
            "forward (+x): the pin, loaded in shear between the plates, drags the "
            "trailer out of the tunnel; keep going until the trailer body is on the "
            "green pad, then stop and let everything come to rest.\n"
            "Success is judged with everything at rest: trailer on the pad, the pin "
            "still threading BOTH holes (the rig arrives and stays hitched — pulling "
            "the pin at the end fails), and the trailer must have actually TRAVELLED "
            "the yard coupled: a trailer moved any other way scores nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Push the blue tractor cart backward until its rear coupler plate stops "
            "on the crimson trailer's tongue block, drop the red-capped pin through "
            "the two aligned holes to hitch them, then pull the tractor forward and "
            "tow the trailer out of the tunnel onto the green dock pad. The trailer "
            "must arrive at the pad towed on the hitch and still pinned at rest."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def trailer_x(self) -> torch.Tensor:
        return (self.trailer.data.root_pos_w - self.env_origins)[:, 0]

    def tractor_x(self) -> torch.Tensor:
        return (self.tractor.data.root_pos_w - self.env_origins)[:, 0]

    def rel_dx(self) -> torch.Tensor:
        """(N,) tractor_x - trailer_x (mate_dx at the stop-block contact)."""
        return self.tractor_x() - self.trailer_x()

    def tractor_bore_x(self) -> torch.Tensor:
        return self.tractor_x() - self.cfg.bore_a

    def trailer_bore_x(self) -> torch.Tensor:
        return self.trailer_x() + self.cfg.bore_b

    def pin_loc(self) -> torch.Tensor:
        """(N,3) pin center, env-local."""
        return self.pin.data.root_pos_w - self.env_origins

    def pin_up_cos(self) -> torch.Tensor:
        """(N,) cosine of the pin axis vs world up."""
        q = self.pin.data.root_quat_w  # (w, x, y, z); body z axis, world z component
        return 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)

    def mated_now(self) -> torch.Tensor:
        """(N,) bool: coupler plate at the stop block — bores aligned."""
        return (self.rel_dx() - self.cfg.mate_dx).abs() <= self.cfg.mate_tol

    def coupled_now(self) -> torch.Tensor:
        """(N,) bool, geometric: the pin shank threads BOTH bores — near-vertical,
        horizontally inside each bore's tolerance ring, at the seated depth."""
        c = self.cfg
        p = self.pin_loc()
        near_t = (p[:, 0] - self.tractor_bore_x()).abs() <= c.couple_xy_tol
        near_r = (p[:, 0] - self.trailer_bore_x()).abs() <= c.couple_xy_tol
        near_y = p[:, 1].abs() <= c.couple_xy_tol
        at_depth = (p[:, 2] - c.pin_seat_z).abs() <= c.couple_z_tol
        upright = self.pin_up_cos() >= math.cos(math.radians(c.pin_tilt_max_deg))
        return near_t & near_r & near_y & at_depth & upright

    def docked_now(self) -> torch.Tensor:
        """(N,) bool: trailer center at/right of the dock threshold."""
        return self.trailer_x() >= self.cfg.dock_lo

    def towed(self) -> torch.Tensor:
        """(N,) bool: the trailer accumulated >= dist_min of forward travel while
        coupled (per-step credit clamped — a teleport jump credits ~one step)."""
        return self.tow_travel >= self.cfg.dist_min

    def _still_now(self) -> torch.Tensor:
        """(N,) bool: both carts and the pin quiet — INSTANTANEOUS (the counter-latch
        below is what `settled()` reads; never judge on this alone)."""
        c = self.cfg
        return ((self.tractor.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.trailer.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.pin.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.pin.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang))

    def settled(self) -> torch.Tensor:
        """(N,) bool: stillness has PERSISTED `settle_steps_min` consecutive steps."""
        return self.still_count >= self.cfg.settle_steps_min

    # ----- progress latches / odometer (step-coupled) -----------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Stillness counter + progress latches + the coupled-tow odometer: forward
        trailer travel counts ONLY while coupled, clamped to step_cap per step (so a
        teleport jump credits at most one step's worth)."""
        c = self.cfg
        self.still_count = (self.still_count + 1.0) * self._still_now().float()
        self.mate_latch = torch.maximum(self.mate_latch, self.mated_now().float())
        coupled = self.coupled_now()
        self.pin_latch = torch.maximum(self.pin_latch, coupled.float())
        tx = self.trailer_x()
        fwd = (tx - self._prev_tx).clamp(0.0, c.step_cap)
        self.tow_travel = self.tow_travel + fwd * coupled.float()
        self._prev_tx = tx

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: trailer on the dock pad, STILL hitched (pin threading both
        bores), genuinely towed there coupled (odometer latch), and everything
        persistently still."""
        return self.docked_now() & self.coupled_now() & self.towed() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25 carts ever mated + 0.35 pin ever coupled —
        latched, credit never evaporates; capped 0.60; 1.0 iff success(). Null
        policy ~0 (trailer in the tunnel, pin in its socket, nothing mated)."""
        base = (0.25 * self.mate_latch + 0.35 * self.pin_latch).clamp(0.0, 0.60)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="hitch_tow", robot="null", env_spacing=3.0))
