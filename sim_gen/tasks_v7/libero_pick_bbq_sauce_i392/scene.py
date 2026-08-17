"""TareLiftScene — declutter a spring-scale freight car until it rises level with the
deck, then slide the basket (bbq sauce riding inside) out of the pit onto the goal pad.

Derived from libero/libero_pick_bbq_sauce ("pick up the bbq sauce and place it in the
basket": one target bottle among condiment distractors, one open basket, one
grasp-carry-lower, one bbox check). Kept from the seed: a bbq-sauce bottle, a basket,
loose condiment-sized objects, and a containment goal. Strategically inverted: the
bottle STARTS in the basket and is NEVER placed anywhere by hand — the judged
interaction is a WEIGHT-GATED EGRESS. The basket sits inside a freight CAR: a
spring-loaded elevator platform (vertical prismatic joint + preloaded linear drive)
in a PIT sunk into a fixed dock. Heavy ballast CANS share the basket with the bottle:

  - while ANY can is aboard, the spring is over-loaded and the car sits >= 19 mm below
    deck level, so the basket's leading face is buried below the deck edge (the SILL)
    — a horizontal push CANNOT extract it (smoke proves a sustained push advances it
    < 10 mm);
  - the car's own ROOF STRIPS overhang the basket rim by 11 mm with only a 4 mm rise
    headroom, so a FREE RISE cannot restore sill clearance either: the basket can rise
    at most ~4 mm relative to the car before pressing the strips (which also deny any
    top grasp of the side rims), and the car itself is bounded by its joint's top stop
    (arithmetic asserted in `__post_init__` with >= 20 % force margins);
  - remove the cans one by one (up through the open lane between the roof strips) into
    the discard BIN and the spring raises the car; with only basket + bottle aboard
    the preload pins the car at its top stop, where the car floor stands 3 mm PROUD of
    the deck — now a gentle push slides the basket over the sill, out of the pit, and
    onto the green GOAL PAD, the bottle riding inside on real contact the whole way.

The removal order (cans before any pushed/resting egress) is forced by physics, not
by latches: while loaded, no horizontal push extracts the basket and no free rise
restores sill clearance — unloading is the path the mechanism affords.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.10 * started  — first ballast can cleared from the basket to the bin    (latched)
  0.25 * cleared  — fraction of this episode's ballast cans now in the bin  (latched max)
  0.15 * risen    — car ever at its top stop with basket + seated bottle    (latched)
  0.20 * egressed — basket (bottle seated) ever fully out of the pit        (latched)
  1.0 iff success() — bottle seated upright in the basket pocket, basket settled on
                     the goal pad, ALL cans in the bin, everything settled and finite.
Non-success is capped at 0.70. All success clauses are live physical outcomes.

Honesty arithmetic (asserted in `__post_init__`, >= 20 % margins):
  - spring preload F0 = k * spring_target exceeds the pass load (car + basket +
    bottle) by >= 20 %: the unloaded car PINS at its top stop;
  - one can over-loads the spring by >= (sill engagement + 12 mm) of depression, and
    that depression stays <= 85 % of the joint travel (the gate never bottoms out
    vacuously at exactly one can);
  - roof-strip lane (158 mm) under-spans the basket (180 mm): the basket cannot leave
    the car upward; worst-case strip overlap (11 mm nominal - 4 mm lateral play) still
    covers the rim; the rise headroom (4 mm) cannot restore sill clearance;
  - a ballast can on its slot passes the strip lane with clearance, even yawed;
  - the goal-pad draw region (all yaws) stays on the deck, clear of pit, sill and bin.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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

G = 9.81


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4), pure torch."""
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def encode_force(mode: int, q_ref: torch.Tensor, q_now: torch.Tensor,
                 f_world: torch.Tensor) -> torch.Tensor:
    """Pre-encode a desired WORLD-frame force for `set_external_force_and_torque`.

    Some pods rotate an applied wrench by the body's rotation since its reference
    orientation (applied = R_now * R_ref^T * arg). mode 0 passes the world force
    through unchanged; mode 1 pre-encodes with R_ref * R_now^T so the applied force
    comes out as the desired world force. Callers PROBE which mode moves the body the
    right way and lock it in (`q_ref` = readback at the reference instant)."""
    if mode == 0:
        return f_world
    return _qapply(_qmul(q_ref, _qinv(q_now)), f_world)


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


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, *, center, size, color, contact_offset: float):
    """One collidable box child: translate + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(box.GetPrim(), contact_offset)
    return box.GetPrim()


def _phys_material(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _bind_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_dock(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC dock. Local frame: origin at the pit centre on the ground,
    +x toward the egress doorway and the deck. Children of one body never
    self-collide; the dock never moves (it is body0 of the car's spring joint).

    Parts: pit floor slab; a 3-sided SHROUD wall around the pit (-x back, +/-y),
    rising well above the car; the SILL support wall under the +x deck edge (the deck
    edge itself is the upper sill face); four deck slabs around the pit; the discard
    BIN walls standing on the +y deck."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(40.0)
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    c, co = cfg, cfg.contact_offset
    ix, iy = c.pit_ix, c.pit_iy                  # pit interior half spans
    wt = c.wall_t
    ox, oy = ix + wt, iy + wt                    # pit outer half spans
    dz0, dz1 = c.deck_top - c.deck_t, c.deck_top
    wz1 = c.shroud_top
    pf = c.pit_floor_top
    body, deck, shroud = c.body_color, c.deck_color, c.shroud_color
    kids = [
        # pit floor slab (full pit outer footprint)
        _box(stage, f"{prim_path}/pit_floor", center=(0.0, 0.0, pf / 2),
             size=(2 * ox, 2 * oy, pf), color=body, contact_offset=co),
        # shroud: back (-x) and +/-y walls, pit floor -> above the car top
        _box(stage, f"{prim_path}/shroud_xn",
             center=(-(ix + wt / 2), 0.0, (pf + wz1) / 2),
             size=(wt, 2 * oy, wz1 - pf), color=shroud, contact_offset=co),
        _box(stage, f"{prim_path}/shroud_yp",
             center=(0.0, iy + wt / 2, (pf + wz1) / 2),
             size=(2 * ox, wt, wz1 - pf), color=shroud, contact_offset=co),
        _box(stage, f"{prim_path}/shroud_yn",
             center=(0.0, -(iy + wt / 2), (pf + wz1) / 2),
             size=(2 * ox, wt, wz1 - pf), color=shroud, contact_offset=co),
        # sill support wall under the +x deck edge (pit floor -> deck underside)
        _box(stage, f"{prim_path}/sill",
             center=(ix + wt / 2, 0.0, (pf + dz0) / 2),
             size=(wt, 2 * oy, dz0 - pf), color=body, contact_offset=co),
        # deck slabs around the pit (z dz0..dz1); the +x slab edge is the upper sill
        _box(stage, f"{prim_path}/deck_xp",
             center=((ix + c.deck_x_max) / 2, 0.0, (dz0 + dz1) / 2),
             size=(c.deck_x_max - ix, 2 * c.deck_y_max, dz1 - dz0),
             color=deck, contact_offset=co),
        _box(stage, f"{prim_path}/deck_xn",
             center=((-c.deck_x_min - ox) / 2, 0.0, (dz0 + dz1) / 2),
             size=(c.deck_x_min - ox, 2 * c.deck_y_max, dz1 - dz0),
             color=deck, contact_offset=co),
        _box(stage, f"{prim_path}/deck_yp",
             center=((ix - ox) / 2, (oy + c.deck_y_max) / 2, (dz0 + dz1) / 2),
             size=(ix + ox, c.deck_y_max - oy, dz1 - dz0), color=deck, contact_offset=co),
        _box(stage, f"{prim_path}/deck_yn",
             center=((ix - ox) / 2, -(oy + c.deck_y_max) / 2, (dz0 + dz1) / 2),
             size=(ix + ox, c.deck_y_max - oy, dz1 - dz0), color=deck, contact_offset=co),
    ]
    # discard bin: 4 walls standing on the +y deck (floor = the deck itself)
    bx, by0, by1 = c.bin_ix, c.bin_y0, c.bin_y1
    bt, bz1 = c.bin_wall_t, c.deck_top + c.bin_wall_h
    bz = (c.deck_top + bz1) / 2
    bh = bz1 - c.deck_top
    for sgn in (1.0, -1.0):
        s = "p" if sgn > 0 else "n"
        kids.append(_box(stage, f"{prim_path}/bin_x{s}",
                         center=(sgn * (bx + bt / 2), (by0 + by1) / 2, bz),
                         size=(bt, by1 - by0 + 2 * bt, bh), color=c.bin_color,
                         contact_offset=co))
    kids.append(_box(stage, f"{prim_path}/bin_y0",
                     center=(0.0, by0 - bt / 2, bz), size=(2 * bx, bt, bh),
                     color=c.bin_color, contact_offset=co))
    kids.append(_box(stage, f"{prim_path}/bin_y1",
                     center=(0.0, by1 + bt / 2, bz), size=(2 * bx, bt, bh),
                     color=c.bin_color, contact_offset=co))
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawn_car(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC freight car + its vertical PRISMATIC spring joint to the
    sibling dock (joints must be authored at spawn — post-play joints are dead; the
    joint pair is collision-FILTERED, the USD default, so the car rides its guide
    without rubbing the pit; travel is bounded by the authored joint limits and the
    spring is a preloaded linear DriveAPI, force mode, target ABOVE the top stop).

    Local frame: origin at the centre of the floor plate TOP face. Children:
      floor  — the platform plate the basket rides on;
      back wall (-x) and two side walls (+/-y), open FRONT (+x, the egress);
      roof strips — two overhanging rails atop the side walls, leaving an open
      lane between them: cans and bottle pass, the basket does NOT."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c, co = cfg, cfg.contact_offset
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.2)
    px.CreateAngularDampingAttr(2.0)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    mat = _phys_material(stage, f"{prim_path}/physmat", c.mu_static, c.mu_dynamic)
    fx, fy, ft = c.floor_x, c.floor_y, c.floor_t
    wh, wt = c.wall_h, c.wall_t
    kids = [
        _box(stage, f"{prim_path}/floor", center=(0.0, 0.0, -ft / 2),
             size=(fx, fy, ft), color=c.floor_color, contact_offset=co),
        _box(stage, f"{prim_path}/wall_xn",
             center=(-(fx - wt) / 2, 0.0, wh / 2),
             size=(wt, fy, wh), color=c.color, contact_offset=co),
    ]
    for sgn in (1.0, -1.0):
        s = "p" if sgn > 0 else "n"
        kids.append(_box(stage, f"{prim_path}/wall_y{s}",
                         center=(0.0, sgn * (fy - wt) / 2, wh / 2),
                         size=(fx, wt, wh), color=c.color, contact_offset=co))
        # roof strip: atop the side wall, overhanging inward to the lane edge
        kids.append(_box(stage, f"{prim_path}/strip_y{s}",
                         center=(0.0, sgn * (c.lane_half + (fy / 2 - c.lane_half) / 2),
                                 wh + c.strip_t / 2),
                         size=(fx, fy / 2 - c.lane_half, c.strip_t),
                         color=c.strip_color, contact_offset=co))
    for k in kids:
        _bind_material(k, mat)

    # vertical prismatic spring joint to the sibling dock
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/spring")
    j.CreateBody0Rel().SetTargets([f"{base}/Dock"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Z")
    # joint origin: the TOP STOP pose in the dock frame -> q = 0 is deck-level ride
    j.CreateLocalPos0Attr(Gf.Vec3f(float(c.x_car), 0.0, float(c.q_top_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(float(-c.travel) - 0.0005)
    j.CreateUpperLimitAttr(0.0005)
    drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    drv.CreateTypeAttr("force")
    drv.CreateStiffnessAttr(float(c.spring_k))
    drv.CreateDampingAttr(float(c.spring_c))
    drv.CreateTargetPositionAttr(float(c.spring_target))
    return root


def _spawn_basket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC basket: floor plate, four walls, and a low square POCKET
    fence on the floor that seats the bbq-sauce bottle base. Origin at the centre
    of the floor plate (4 mm above the bottom face); CoM low (MassAPI at origin)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c, co = cfg, cfg.contact_offset
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.05)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    mat = _phys_material(stage, f"{prim_path}/physmat", c.mu_static, c.mu_dynamic)
    ho = c.out / 2                                # outer half span
    hi = ho - c.wall_t                            # interior half span
    wz = c.floor_t / 2 + c.wall_h / 2             # wall centre z (origin mid-floor)
    kids = [
        _box(stage, f"{prim_path}/floor", center=(0.0, 0.0, 0.0),
             size=(c.out, c.out, c.floor_t), color=c.color, contact_offset=co),
    ]
    for sgn in (1.0, -1.0):
        s = "p" if sgn > 0 else "n"
        kids.append(_box(stage, f"{prim_path}/wall_x{s}",
                         center=(sgn * (ho - c.wall_t / 2), 0.0, wz),
                         size=(c.wall_t, c.out, c.wall_h), color=c.color,
                         contact_offset=co))
        kids.append(_box(stage, f"{prim_path}/wall_y{s}",
                         center=(0.0, sgn * (ho - c.wall_t / 2), wz),
                         size=(2 * hi, c.wall_t, c.wall_h), color=c.color,
                         contact_offset=co))
        # pocket fences (low, on the floor, seat the bottle base)
        fo = c.pock_in / 2 + c.fence_t / 2
        fz = c.floor_t / 2 + c.fence_h / 2
        fl = c.pock_in + 2 * c.fence_t
        kids.append(_box(stage, f"{prim_path}/fence_x{s}",
                         center=(sgn * fo, 0.0, fz), size=(c.fence_t, fl, c.fence_h),
                         color=c.fence_color, contact_offset=co))
        kids.append(_box(stage, f"{prim_path}/fence_y{s}",
                         center=(0.0, sgn * fo, fz), size=(fl, c.fence_t, c.fence_h),
                         color=c.fence_color, contact_offset=co))
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC bbq-sauce bottle: square body + narrower neck cap (a
    natural parallel-jaw pinch, though the solve never grasps it). Origin at the
    BOTTOM face centre; MassAPI at origin -> low CoM (a stable rider)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c, co = cfg, cfg.contact_offset
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.05)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    mat = _phys_material(stage, f"{prim_path}/physmat", c.mu_static, c.mu_dynamic)
    kids = [
        _box(stage, f"{prim_path}/body", center=(0.0, 0.0, c.body_h / 2),
             size=(c.body_s, c.body_s, c.body_h), color=c.color, contact_offset=co),
        _box(stage, f"{prim_path}/neck", center=(0.0, 0.0, c.body_h + c.neck_h / 2),
             size=(c.neck_s, c.neck_s, c.neck_h), color=c.neck_color,
             contact_offset=co),
    ]
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "dock" not in _SPAWNER_CACHE:

        @configclass
        class DockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_dock)
            pit_ix: float = 0.104
            pit_iy: float = 0.108
            wall_t: float = 0.012
            pit_floor_top: float = 0.030
            deck_top: float = 0.120
            deck_t: float = 0.012
            shroud_top: float = 0.230
            deck_x_max: float = 0.560
            deck_x_min: float = 0.200
            deck_y_max: float = 0.300
            bin_ix: float = 0.075
            bin_y0: float = 0.130
            bin_y1: float = 0.280
            bin_wall_t: float = 0.008
            bin_wall_h: float = 0.035
            mu_static: float = 0.35
            mu_dynamic: float = 0.30
            body_color: tuple = (0.35, 0.38, 0.44)
            deck_color: tuple = (0.52, 0.55, 0.60)
            shroud_color: tuple = (0.28, 0.31, 0.38)
            bin_color: tuple = (0.20, 0.22, 0.26)
            contact_offset: float = 0.002

        @configclass
        class CarSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_car)
            floor_x: float = 0.196
            floor_y: float = 0.204
            floor_t: float = 0.010
            wall_t: float = 0.008
            wall_h: float = 0.082
            strip_t: float = 0.008
            lane_half: float = 0.079
            x_car: float = 0.002
            q_top_z: float = 0.123
            travel: float = 0.030
            spring_k: float = 250.0
            spring_c: float = 30.0
            spring_target: float = 0.0564
            mass: float = 0.45
            mu_static: float = 0.45
            mu_dynamic: float = 0.40
            color: tuple = (0.86, 0.48, 0.12)
            floor_color: tuple = (0.92, 0.60, 0.20)
            strip_color: tuple = (0.60, 0.30, 0.08)
            contact_offset: float = 0.002

        @configclass
        class BasketSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_basket)
            out: float = 0.180
            wall_t: float = 0.008
            wall_h: float = 0.070
            floor_t: float = 0.008
            pock_in: float = 0.054
            fence_t: float = 0.005
            fence_h: float = 0.012
            mass: float = 0.30
            mu_static: float = 0.45
            mu_dynamic: float = 0.40
            color: tuple = (0.55, 0.38, 0.16)
            fence_color: tuple = (0.42, 0.28, 0.10)
            contact_offset: float = 0.002

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            body_s: float = 0.045
            body_h: float = 0.100
            neck_s: float = 0.024
            neck_h: float = 0.045
            mass: float = 0.40
            mu_static: float = 0.60
            mu_dynamic: float = 0.55
            color: tuple = (0.46, 0.08, 0.05)
            neck_color: tuple = (0.88, 0.48, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["dock"] = DockSpawnerCfg
        _SPAWNER_CACHE["car"] = CarSpawnerCfg
        _SPAWNER_CACHE["basket"] = BasketSpawnerCfg
        _SPAWNER_CACHE["bottle"] = BottleSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TareLiftSceneCfg(BaseCfg):
    """Config for `TareLiftScene`. The weight gate is honest by construction: the
    force-balance and interference arithmetic below is asserted at import time."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    seat_xy_tol: float = tunable(0.012)     # bottle base offset in the basket frame (per axis)
    seat_z_lo: float = tunable(0.002)       # bottle origin z band in the basket frame
    seat_z_hi: float = tunable(0.024)
    seat_up_cos: float = tunable(0.90)      # bottle uprightness (world up)
    pad_xy_tol: float = tunable(0.055)      # basket centre offset in the pad frame (per axis)
    pad_z_lo: float = tunable(0.118)        # basket origin world-z band on the pad/deck
    pad_z_hi: float = tunable(0.140)
    pad_up_cos: float = tunable(0.95)       # basket uprightness
    bin_x_tol: float = tunable(0.070)       # can centre bands inside the bin (dock frame)
    bin_y_lo: float = tunable(0.135)
    bin_y_hi: float = tunable(0.275)
    bin_z_lo: float = tunable(0.120)
    bin_z_hi: float = tunable(0.175)
    risen_q_tol: float = tunable(0.004)     # car q at/above -tol = at the top stop
    egress_x_min: float = tunable(0.206)    # basket dock-x beyond this = fully out of the pit
    settle_speed: float = tunable(0.08)     # max |lin vel| of every judged body (m/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    dock_jitter: float = tunable(0.030)     # BUILD-time dock xy jitter (+/- m)
    dock_yaw_deg: float = tunable(12.0)     # BUILD-time dock yaw (+/- deg)
    n_cans_lo: int = tunable(2)             # ballast cans aboard per episode (inclusive)
    n_cans_hi: int = tunable(4)
    can_yaw_deg: float = tunable(8.0)       # can yaw about its slot (+/- deg)
    pad_x_lo: float = tunable(0.30)         # goal-pad draw region (dock frame)
    pad_x_hi: float = tunable(0.40)
    pad_y_amp: float = tunable(0.14)
    pad_yaw_deg: float = tunable(30.0)

    # --- info: layout (world nominal, ground z = 0) ----------------------------------------------
    dock_pos: tuple = info((0.36, 0.0))     # dock (pit centre) nominal position
    # --- info: dock structure (dock local frame, origin at pit centre, ground) -------------------
    pit_ix: float = info(0.104)             # pit interior half spans
    pit_iy: float = info(0.108)
    wall_t: float = info(0.012)
    pit_floor_top: float = info(0.030)
    deck_top: float = info(0.120)           # deck top surface = sill top
    deck_t: float = info(0.012)
    shroud_top: float = info(0.230)
    deck_x_max: float = info(0.560)
    deck_x_min: float = info(0.200)
    deck_y_max: float = info(0.300)
    bin_ix: float = info(0.075)             # bin interior half span (x)
    bin_y0: float = info(0.130)             # bin interior y band
    bin_y1: float = info(0.280)
    bin_wall_t: float = info(0.008)
    bin_wall_h: float = info(0.035)
    # --- info: freight car (origin at floor-plate top centre) ------------------------------------
    car_floor_x: float = info(0.196)
    car_floor_y: float = info(0.204)
    car_floor_t: float = info(0.010)
    car_wall_t: float = info(0.008)
    car_wall_h: float = info(0.082)
    car_strip_t: float = info(0.008)
    lane_half: float = info(0.079)          # roof-strip lane half width (open middle)
    x_car: float = info(0.002)              # car origin x in the dock frame
    q_top_z: float = info(0.123)            # car origin z at the TOP STOP (deck + 3 mm)
    travel: float = info(0.030)             # spring joint travel (top stop -> bottom stop)
    spring_k: float = info(250.0)           # N/m
    spring_c: float = info(30.0)            # N s/m
    spring_target: float = info(0.0564)     # drive target ABOVE the top stop -> preload
    car_mass: float = info(0.45)
    # --- info: basket / bottle / cans / pad ------------------------------------------------------
    bask_out: float = info(0.180)
    bask_wall_t: float = info(0.008)
    bask_wall_h: float = info(0.070)
    bask_floor_t: float = info(0.008)
    pock_in: float = info(0.054)            # bottle pocket interior span
    fence_t: float = info(0.005)
    fence_h: float = info(0.012)
    bask_mass: float = info(0.30)
    bot_s: float = info(0.045)              # bottle body square
    bot_body_h: float = info(0.100)
    bot_neck_s: float = info(0.024)
    bot_neck_h: float = info(0.045)
    bot_mass: float = info(0.40)
    can_s: float = info(0.036)              # ballast can square
    can_h: float = info(0.050)
    can_mass: float = info(0.85)
    can_color: tuple = info((0.16, 0.52, 0.52))
    pad_size: float = info(0.200)
    pad_t: float = info(0.004)
    pad_color: tuple = info((0.10, 0.62, 0.20))
    contact_offset: float = info(0.002)
    ground_mu: float = info(0.40)
    # basket can-slots (car/basket frame) and bin park spots (dock frame), rank-indexed
    can_slots: tuple = info(((-0.055, -0.055), (0.055, -0.055),
                             (-0.055, 0.055), (0.055, 0.055)))
    bin_parks: tuple = info(((-0.040, 0.168), (0.040, 0.168),
                             (-0.040, 0.240), (0.040, 0.240)))
    # rubric weights (0.10 + 0.25 + 0.15 + 0.20 = 0.70 = the non-success cap)
    w_first: float = info(0.10)
    w_clear: float = info(0.25)
    w_rise: float = info(0.15)
    w_egress: float = info(0.20)

    def __post_init__(self) -> None:
        # ---- spring force balance (the weight gate), >= 20 % margins ----------------------------
        f0 = self.spring_k * self.spring_target          # preload at the top stop
        w_pass = (self.car_mass + self.bask_mass + self.bot_mass) * G
        assert f0 >= 1.20 * w_pass, "preload must pin the unloaded car at the top stop"
        d1 = (w_pass + self.can_mass * G - f0) / self.spring_k   # 1-can depression
        lip = self.q_top_z - self.deck_top               # car floor proud height at top (3 mm)
        assert lip > 0.001, "car floor must ride proud of the deck at the top stop"
        engage = d1 - lip                                # basket bottom below sill top
        assert engage >= 0.012, "one can must bury the basket >= 12 mm below the sill"
        assert d1 <= 0.85 * self.travel, "the gate must not bottom out at exactly one can"
        # ---- the lift bypass is dead: strips cap the basket, the stop caps the car --------------
        rim = self.bask_floor_t + self.bask_wall_h       # basket rim above its bottom face
        head = self.car_wall_h - rim                     # rim -> strip underside headroom
        assert 0.003 <= head <= 0.008, "strip headroom must be a few mm only"
        # even fully risen against the strips, a 1-can-loaded basket stays below the sill
        assert (self.q_top_z - d1) + head + 0.002 <= self.deck_top - 0.004, \
            "strip headroom must not restore sill clearance under load"
        assert 2 * self.lane_half <= self.bask_out - 0.016, \
            "the strip lane must under-span the basket"
        play = (self.car_floor_y - 2 * self.car_wall_t - self.bask_out) / 2
        overlap = self.bask_out / 2 - self.lane_half
        assert overlap - play >= 0.005, "worst-case strip overlap must still cover the rim"
        # ---- cans pass the lane; slots clear pocket fences and basket walls ---------------------
        yaw = math.radians(self.can_yaw_deg)
        half = (self.can_s / 2) * (math.cos(yaw) + math.sin(yaw))
        slot = abs(self.can_slots[0][0])
        assert slot + half <= self.lane_half - 0.002, "cans must pass the strip lane"
        assert slot - half >= self.pock_in / 2 + self.fence_t + 0.001, \
            "can slots clear the pocket fences"
        assert slot + half <= self.bask_out / 2 - self.bask_wall_t - 0.002, \
            "can slots clear the basket walls"
        # ---- bottle seats in the pocket; bin holds every can ------------------------------------
        assert self.bot_s + 0.006 <= self.pock_in, "bottle base must drop into the pocket"
        diag = (self.can_s / 2) * math.sqrt(2.0)
        for px_, py_ in self.bin_parks:
            assert abs(px_) + diag <= self.bin_ix - 0.002, "bin parks clear the bin walls"
            assert self.bin_y0 + diag <= py_ + 0.0001 and py_ + diag <= self.bin_y1 + 0.0001, \
                "bin parks clear the bin y walls"
        # ---- the goal pad (any yaw) stays on the deck, clear of pit, sill and bin ---------------
        reach = (self.pad_size / 2) * (math.cos(math.radians(self.pad_yaw_deg))
                                       + math.sin(math.radians(self.pad_yaw_deg)))
        assert self.pad_x_lo - reach > self.pit_ix + self.wall_t, "pad clears the sill"
        assert self.pad_x_hi + reach < self.deck_x_max - 0.01, "pad stays on the deck (+x)"
        assert self.pad_y_amp + reach < self.deck_y_max - 0.01, "pad stays on the deck (y)"
        assert self.pad_x_lo - reach > self.bin_ix + self.bin_wall_t, "pad clears the bin"
        # egress lane: the basket path (x > sill) never meets the bin (x <= bin_ix + t)
        assert self.egress_x_min > self.pit_ix + self.wall_t + self.bask_out / 2 - 0.001, \
            "egress threshold = basket fully past the sill"
        # basket passes the doorway between the shroud walls
        assert self.bask_out / 2 + 0.006 <= self.pit_iy, "basket passes the doorway"
        # car front edge -> sill inner face gap
        gap = self.pit_ix - (self.x_car + self.car_floor_x / 2)
        assert 0.002 <= gap <= 0.008, "car floor front edge must nearly kiss the sill"
        # bin capacity sanity: 4 cans side by side with clearance
        assert self.n_cans_lo >= 2, "start always gate-closed (>= 2 cans -> bottom stop)"
        assert (self.w_first + self.w_clear + self.w_rise + self.w_egress) - 0.70 < 1e-9


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("tare_lift")
class TareLiftScene(BaseScene):
    cfg: TareLiftSceneCfg

    def __init__(self, cfg: TareLiftSceneCfg | None = None) -> None:
        super().__init__(cfg or TareLiftSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """BUILD-time dock pose randomization: the dock is kinematic body0 of the
        spawn-authored spring joint, so it must NEVER be teleported. Its pose is drawn
        HERE (`build(seed=...)` seeds the RNG before assets()); everything episodic
        (can count/arrangement, bottle yaw, goal-pad pose) happens in reset()."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        r = torch.rand(3)
        dx = c.dock_pos[0] + float(r[0] * 2 - 1) * c.dock_jitter
        dy = c.dock_pos[1] + float(r[1] * 2 - 1) * c.dock_jitter
        dyaw = math.radians(float(r[2] * 2 - 1) * c.dock_yaw_deg)
        self._dock_build = (dx, dy, dyaw)
        cosw, sinw = math.cos(dyaw), math.sin(dyaw)
        quat = (math.cos(dyaw / 2), 0.0, 0.0, math.sin(dyaw / 2))
        # car spawns at the TOP STOP in the dock frame (reset writes the loaded depth)
        cx = dx + cosw * c.x_car
        cy = dy + sinw * c.x_car

        cls = _spawner_classes()
        dock_spawn = cls["dock"](
            pit_ix=c.pit_ix, pit_iy=c.pit_iy, wall_t=c.wall_t,
            pit_floor_top=c.pit_floor_top, deck_top=c.deck_top, deck_t=c.deck_t,
            shroud_top=c.shroud_top, deck_x_max=c.deck_x_max, deck_x_min=c.deck_x_min,
            deck_y_max=c.deck_y_max, bin_ix=c.bin_ix, bin_y0=c.bin_y0, bin_y1=c.bin_y1,
            bin_wall_t=c.bin_wall_t, bin_wall_h=c.bin_wall_h,
            contact_offset=c.contact_offset)
        car_spawn = cls["car"](
            floor_x=c.car_floor_x, floor_y=c.car_floor_y, floor_t=c.car_floor_t,
            wall_t=c.car_wall_t, wall_h=c.car_wall_h, strip_t=c.car_strip_t,
            lane_half=c.lane_half, x_car=c.x_car, q_top_z=c.q_top_z, travel=c.travel,
            spring_k=c.spring_k, spring_c=c.spring_c, spring_target=c.spring_target,
            mass=c.car_mass, contact_offset=c.contact_offset)
        basket_spawn = cls["basket"](
            out=c.bask_out, wall_t=c.bask_wall_t, wall_h=c.bask_wall_h,
            floor_t=c.bask_floor_t, pock_in=c.pock_in, fence_t=c.fence_t,
            fence_h=c.fence_h, mass=c.bask_mass, contact_offset=c.contact_offset)
        bottle_spawn = cls["bottle"](
            body_s=c.bot_s, body_h=c.bot_body_h, neck_s=c.bot_neck_s,
            neck_h=c.bot_neck_h, mass=c.bot_mass, contact_offset=c.contact_offset)

        can_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.05, angular_damping=0.05,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=c.contact_offset, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.45, dynamic_friction=0.40, restitution=0.0),
        )

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.ground_mu, dynamic_friction=c.ground_mu - 0.05,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            # ORDER MATTERS: the dock must exist when the car's spring joint is
            # authored (Body0Rel targets the sibling {ENV_REGEX_NS}/Dock).
            "dock": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dock",
                spawn=dock_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(dx, dy, 0.0), rot=quat),
            ),
            "car": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Car",
                spawn=car_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, c.q_top_z),
                                                          rot=quat),
            ),
            "basket": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Basket",
                spawn=basket_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(cx, cy, c.q_top_z + 0.002 + c.bask_floor_t / 2), rot=quat),
            ),
            "bottle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle",
                spawn=bottle_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(cx, cy, c.q_top_z + 0.002 + c.bask_floor_t + 0.002), rot=quat),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=sim_utils.CuboidCfg(
                    size=(c.pad_size, c.pad_size, c.pad_t),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
                    mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    # NO collider: the pad is a goal MARKER (a judged frame), not an
                    # obstacle — even a sub-mm proud edge hard-stops the flat-bottomed
                    # basket's deck-scraping leading edge, so the basket must rest on
                    # the DECK (the pad_z band judges deck height), sliding freely.
                ),
                # top ~1 mm proud of the deck for visibility (no z-fighting)
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(dx + 0.35, dy, c.deck_top - c.pad_t / 2 + 0.001)),
            ),
        }
        for i in range(4):
            sx, sy = c.can_slots[i]
            out[f"can{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Can_" + str(i),
                spawn=sim_utils.CuboidCfg(
                    size=(c.can_s, c.can_s, c.can_h),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.can_color),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.can_mass),
                    **can_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(cx + sx, cy + sy,
                         c.q_top_z + 0.002 + c.bask_floor_t + 0.002 + c.can_h / 2),
                    rot=quat),
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
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.dock: RigidObject = env.iscene["dock"]
        self.car: RigidObject = env.iscene["car"]
        self.basket: RigidObject = env.iscene["basket"]
        self.bottle: RigidObject = env.iscene["bottle"]
        self.pad: RigidObject = env.iscene["pad"]
        self.cans: list[RigidObject] = [env.iscene[f"can{i}"] for i in range(4)]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # episode bookkeeping + latches (partial credit; success is judged live)
        self._n_cans = torch.full((n,), 4, dtype=torch.long, device=dev)
        self._frac = torch.zeros(n, device=dev)
        self._risen = torch.zeros(n, dtype=torch.bool, device=dev)
        self._egressed = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: draw the ballast count n in [n_cans_lo, n_cans_hi]; a random
        n of the 4 cans ride the basket (rank-indexed slots), the rest start parked in
        the bin. The whole freight stack (car at its loaded equilibrium depth, basket,
        bottle, cans) is written together, velocities zero; the goal pad teleports to
        a random deck pose. The dock is kinematic body0 of the spring joint and is
        never moved after build."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        dp = self.dock.data.root_pos_w[env_ids]
        dq = self.dock.data.root_quat_w[env_ids]

        # burn post-seed draws (first rand AND randint draws are degenerate across seeds)
        _ = torch.rand(8, device=dev)
        _ = torch.randint(0, 997, (4,), device=dev)

        n = torch.randint(int(c.n_cans_lo), int(c.n_cans_hi) + 1, (m,), device=dev)
        ranks = torch.argsort(torch.rand(m, 4, device=dev), dim=1)  # can i -> rank
        can_yaw = (torch.rand(m, 4, device=dev) * 2 - 1) * math.radians(c.can_yaw_deg)
        bot_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        pad_x = torch.rand(m, device=dev) * (c.pad_x_hi - c.pad_x_lo) + c.pad_x_lo
        pad_y = (torch.rand(m, device=dev) * 2 - 1) * c.pad_y_amp
        pad_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pad_yaw_deg)

        # loaded spring equilibrium depth for n cans (clamped to the bottom stop)
        f0 = c.spring_k * c.spring_target
        w = (c.car_mass + c.bask_mass + c.bot_mass + n.float() * c.can_mass) * G
        q_eq = (-(w - f0) / c.spring_k).clamp(min=-c.travel, max=0.0)

        def write(body, loc_xyz, q_local) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = dp + _qapply(dq, loc_xyz)
            st[:, 3:7] = _qmul(dq, q_local)
            body.write_root_state_to_sim(st, env_ids)

        ident = torch.zeros(m, 4, device=dev)
        ident[:, 0] = 1.0

        # car at its loaded equilibrium depth
        car_loc = torch.zeros(m, 3, device=dev)
        car_loc[:, 0] = c.x_car
        car_loc[:, 2] = c.q_top_z + q_eq
        write(self.car, car_loc, ident)

        # basket on the car floor
        bask_loc = car_loc.clone()
        bask_loc[:, 2] += 0.002 + c.bask_floor_t / 2
        write(self.basket, bask_loc, ident)

        # bottle seated in the pocket (free yaw)
        bot_loc = car_loc.clone()
        bot_loc[:, 2] += 0.002 + c.bask_floor_t + 0.002
        write(self.bottle, bot_loc, _qz(bot_yaw))

        # cans: rank < n ride the basket slots; the rest park in the bin
        slots = torch.tensor(c.can_slots, device=dev)      # (4,2) car frame
        parks = torch.tensor(c.bin_parks, device=dev)      # (4,2) dock frame
        for i in range(4):
            rk = ranks[:, i]
            aboard = rk < n
            loc = torch.zeros(m, 3, device=dev)
            sxy = slots[rk]
            pxy = parks[rk]
            loc[:, 0] = torch.where(aboard, c.x_car + sxy[:, 0], pxy[:, 0])
            loc[:, 1] = torch.where(aboard, sxy[:, 1], pxy[:, 1])
            loc[:, 2] = torch.where(
                aboard,
                c.q_top_z + q_eq + 0.002 + c.bask_floor_t + 0.002 + c.can_h / 2,
                torch.full((m,), c.deck_top + 0.002 + c.can_h / 2, device=dev))
            write(self.cans[i], loc, _qz(can_yaw[:, i]))

        # goal pad (kinematic teleport; collider-less marker, top ~1 mm proud)
        pad_loc = torch.zeros(m, 3, device=dev)
        pad_loc[:, 0] = pad_x
        pad_loc[:, 1] = pad_y
        pad_loc[:, 2] = c.deck_top - c.pad_t / 2 + 0.001
        write(self.pad, pad_loc, _qz(pad_yaw))

        self._n_cans[env_ids] = n
        self._frac[env_ids] = 0.0
        self._risen[env_ids] = False
        self._egressed[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "car": self.car.data.root_state_w[env_ids].clone(),
            "basket": self.basket.data.root_state_w[env_ids].clone(),
            "bottle": self.bottle.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "n_cans": self._n_cans[env_ids].clone(),
            "frac": self._frac[env_ids].clone(),
            "risen": self._risen[env_ids].clone(),
            "egressed": self._egressed[env_ids].clone(),
        }
        for i in range(4):
            out[f"can{i}"] = self.cans[i].data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.car.write_root_state_to_sim(state["car"], env_ids)
        self.basket.write_root_state_to_sim(state["basket"], env_ids)
        self.bottle.write_root_state_to_sim(state["bottle"], env_ids)
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        for i in range(4):
            self.cans[i].write_root_state_to_sim(state[f"can{i}"], env_ids)
        self._n_cans[env_ids] = state["n_cans"]
        self._frac[env_ids] = state["frac"]
        self._risen[env_ids] = state["risen"]
        self._egressed[env_ids] = state["egressed"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        f0 = c.spring_k * c.spring_target
        return (
            f"A gray DOCK block (position and heading vary per build) has a rectangular "
            f"PIT sunk {c.deck_top * 1000:.0f} mm below its deck, shrouded by walls on "
            f"three sides and open toward the deck on the fourth (the DOORWAY, whose "
            f"floor edge is the SILL). Inside rides an orange spring-loaded FREIGHT CAR "
            f"on a vertical rail (travel {c.travel * 1000:.0f} mm, preload "
            f"{f0:.1f} N): with only the wicker-brown BASKET and the dark-red BBQ-SAUCE "
            f"bottle aboard, the spring pins the car at its top stop, the car floor "
            f"{(c.q_top_z - c.deck_top) * 1000:.0f} mm proud of the deck. But the basket "
            f"also carries {c.n_cans_lo}-{c.n_cans_hi} heavy teal BALLAST CANS "
            f"({c.can_mass * 1000:.0f} g each; the count and arrangement vary per "
            f"episode), and each can sinks the car by >= {1000 * ((c.can_mass * G) / c.spring_k):.0f} mm — while any can is aboard, the basket's leading face is "
            f"buried below the sill and CANNOT slide out. The car's own roof strips "
            f"overhang the basket rim with only a few mm of headroom, denying any top "
            f"grasp of the rims and any free rise back to sill level: unload the cans "
            f"and the spring itself delivers the basket to deck level. The "
            f"bbq-sauce bottle stands seated in a low pocket fence on the basket floor "
            f"and pokes up through the open lane between the strips; the cans sit "
            f"around it and pass freely through the lane. On the deck stand a dark "
            f"DISCARD BIN (fixed, beside the pit) and a green GOAL PAD (position and "
            f"heading vary per episode).\n"
            f"Goal: every ballast can dropped into the discard bin, and the basket — "
            f"with the bbq-sauce bottle still seated upright in its pocket — slid out "
            f"through the doorway, over the sill, and settled centred on the goal pad "
            f"(within {c.pad_xy_tol * 1000:.0f} mm), everything at rest. A basket left "
            f"in the pit, parked off the pad, or missing its bottle does not count; "
            f"neither do cans left in the basket, on the deck, or anywhere but the bin."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Take the ballast cans out of the basket and drop them in the discard bin "
            "so the spring car rises level with the deck, then slide the basket with "
            "the bbq sauce bottle out of the pit and onto the green pad."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _dock_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the dock body frame, (N,3) -> (N,3)."""
        return _qapply(_qinv(self.dock.data.root_quat_w),
                       pos_w - self.dock.data.root_pos_w)

    def _basket_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the basket body frame, (N,3) -> (N,3)."""
        return _qapply(_qinv(self.basket.data.root_quat_w),
                       pos_w - self.basket.data.root_pos_w)

    def _pad_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the goal-pad body frame, (N,3) -> (N,3)."""
        return _qapply(_qinv(self.pad.data.root_quat_w),
                       pos_w - self.pad.data.root_pos_w)

    def _up_cos(self, body) -> torch.Tensor:
        up = torch.zeros_like(body.data.root_pos_w)
        up[:, 2] = 1.0
        return _qapply(body.data.root_quat_w, up)[:, 2]

    def car_q(self) -> torch.Tensor:
        """(N,) spring coordinate: 0 = TOP STOP (deck-level ride), -travel = bottom."""
        loc = self._dock_local(self.car.data.root_pos_w)
        return loc[:, 2] - self.cfg.q_top_z

    def bottle_seated(self) -> torch.Tensor:
        """(N,) bool: bottle base inside the basket pocket, upright."""
        c = self.cfg
        loc = self._basket_local(self.bottle.data.root_pos_w)
        return (loc[:, 0].abs() < c.seat_xy_tol) & (loc[:, 1].abs() < c.seat_xy_tol) \
            & (loc[:, 2] > c.seat_z_lo) & (loc[:, 2] < c.seat_z_hi) \
            & (self._up_cos(self.bottle) > c.seat_up_cos)

    def basket_on_pad(self) -> torch.Tensor:
        """(N,) bool: basket centred on the goal pad, at deck height, upright."""
        c = self.cfg
        loc = self._pad_local(self.basket.data.root_pos_w)
        z = self.basket.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return (loc[:, 0].abs() < c.pad_xy_tol) & (loc[:, 1].abs() < c.pad_xy_tol) \
            & (z > c.pad_z_lo) & (z < c.pad_z_hi) \
            & (self._up_cos(self.basket) > c.pad_up_cos)

    def in_bin(self, body) -> torch.Tensor:
        """(N,) bool: body centre inside the discard bin (dock frame)."""
        c = self.cfg
        loc = self._dock_local(body.data.root_pos_w)
        return (loc[:, 0].abs() < c.bin_x_tol) \
            & (loc[:, 1] > c.bin_y_lo) & (loc[:, 1] < c.bin_y_hi) \
            & (loc[:, 2] > c.bin_z_lo) & (loc[:, 2] < c.bin_z_hi)

    def cans_binned_count(self) -> torch.Tensor:
        """(N,) long: how many of the 4 cans are inside the bin."""
        return torch.stack([self.in_bin(b) for b in self.cans], dim=1).sum(dim=1)

    def cans_binned(self) -> torch.Tensor:
        """(N,) bool: ALL 4 cans inside the bin."""
        return self.cans_binned_count() == 4

    def basket_in_car(self) -> torch.Tensor:
        """(N,) bool: basket riding the car (dock frame, over the pit)."""
        c = self.cfg
        loc = self._dock_local(self.basket.data.root_pos_w)
        return ((loc[:, 0] - c.x_car).abs() < 0.030) & (loc[:, 1].abs() < 0.030) \
            & (loc[:, 2] < c.deck_top + 0.060)

    def settled(self) -> torch.Tensor:
        """(N,) bool: every judged body |lin vel| below `settle_speed`."""
        bodies = [self.car, self.basket, self.bottle] + self.cans
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in bodies], dim=1)
        return (v < self.cfg.settle_speed).all(dim=1)

    def _finite(self) -> torch.Tensor:
        bodies = [self.car, self.basket, self.bottle] + self.cans
        p = torch.stack([b.data.root_pos_w for b in bodies], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        extra = (self.cans_binned_count() - (4 - self._n_cans)).clamp(min=0)
        frac = (extra.float() / self._n_cans.float()).clamp(max=1.0)
        self._frac = torch.where(fin, torch.maximum(self._frac, frac), self._frac)
        at_top = self.car_q() >= -c.risen_q_tol
        self._risen |= at_top & self.basket_in_car() & self.bottle_seated() & fin
        bx = self._dock_local(self.basket.data.root_pos_w)[:, 0]
        self._egressed |= (bx > c.egress_x_min) & self.bottle_seated() & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: bottle seated upright in the basket pocket, basket settled
        centred on the goal pad, ALL cans in the discard bin, everything settled and
        finite. All clauses are live physical outcomes."""
        self._update_latches()
        return self.bottle_seated() & self.basket_on_pad() & self.cans_binned() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10*started + 0.25*cleared_frac + 0.15*risen +
        0.20*egressed (all latched; ~0 for doing nothing — an untouched gate latches
        nothing), capped at 0.70 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_first * (self._frac > 0).float() + c.w_clear * self._frac
                + c.w_rise * self._risen.float()
                + c.w_egress * self._egressed.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="tare_lift", robot="null"))
