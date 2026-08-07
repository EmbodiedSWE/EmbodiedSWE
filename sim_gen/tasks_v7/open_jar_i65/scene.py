"""PistonJarScene — open a sealed jar by pressing it down over a pedestal spike that
drives an internal ejector piston, then deliver the freed red ball to the dish.

Derived from rlbench/open_jar ("open_jar": grasp the jar's lid, UNSCREW it — one
continuous rotation of the gripped lid about the jar axis — lift it off, done). Here
the closure cannot be unscrewed, gripped, pried or shaken off at all, and no part of
the opening is a rotation. The lid is a bare plate recessed 4 mm below the rim inside
a collar pocket (nothing to pinch: 2 mm annular gap), and it is SEALED to the jar by
a physical weld (a PhysX breakable fixed joint, breakForce ~10 N): gravity, carrying,
shaking, even resting the whole 0.45 kg jar on the spike (~4.5 N through the seal)
cannot pop it. Opening runs through the jar's own PRESS-TO-EJECT mechanism:

  1. the jar's bottom plate has a 36 mm square through-hole; inside the bore, a
     captive ejector PISTON (prismatic joint, 80 mm travel) rests just above the
     hole, carrying the red payload ball in a shallow tray;
  2. the fixed pedestal carries a vertical ORANGE SPIKE sized to pass through the
     bottom hole; sleeving the jar over the spike and PRESSING DOWN drives the
     piston up relative to the jar (descent is converted to content ascent);
  3. the piston presses the ball into the lid's underside; past ~10 N the seal
     SHEARS (irreversibly), the rising ball shoves the lid out of the collar pocket
     (the lid's offset centre of mass tips it off the rim), and when the jar bottoms
     out on the pedestal base the ball sits in the piston tray ABOVE the rim,
     presented for a pinch grasp;
  4. the freed ball is placed in the walled blue dish.

Direction-sensitivity of the seal is geometric, not scripted: the lid hovers 0.5 mm
above a supporting shoulder, so a press from OUTSIDE (down onto the lid top) is
carried by shoulder CONTACT and does not load the weld — only a push from INSIDE
(the ball, driven by the piston) loads the weld in tension. success()/score() judge
only physical outcomes (poses, containment, settledness).

Assets are fully procedural (compound-spawner pattern; the prismatic joint and the
breakable weld are authored at spawn — post-play joints are dead on this stack):
  - pedestal: heavy DYNAMIC compound (12 kg): base slab 170x170x30, square spike
    16x16 to z 106 with an 8x8 pilot tip to z 116 (the entry funnel: 8 mm tip into
    the 36 mm hole).
  - jar (0.45 kg): 64x64 outer square tube; bottom plate z 0..8 with the central
    36x36 hole; bore 48x48 walls z 8..92 (wall top = the lid SHOULDER); mouth
    collar z 92..102, inner 56x56 (rim top 102). Local origin: footprint centre on
    the ground.
  - piston (60 g): 46x46x10 plate + 6 mm tray fences (pocket 34x34), on a Z
    prismatic joint to the jar (limits [-1, +80] mm; rests hanging on the lower
    limit 2 mm above the bottom plate). Piston<->jar collision keeps the USD
    joint-pair default (FILTERED) — alignment comes from the joint, the limits are
    the stops, and every load-bearing contact (spike->piston, piston->ball,
    ball->lid) is between UNjointed pairs and live.
  - lid (40 g): bare 52x52x6 plate welded (breakable fixed joint, collision
    ENABLED) 0.5 mm above the shoulder; centre of mass offset 10 mm so a
    centre-lifted lid tips off the rim instead of balancing on the ball.
  - ball: RED 30 mm sphere, 30 g, in the piston tray under the sealed lid.
  - dish: DYNAMIC blue walled tray (120x120 base, 14 mm fences, 1.2 kg).

Per-episode randomization (readback-verifiable): pedestal xy + yaw, jar xy + free
yaw, dish xy + yaw (the jar/piston/lid/ball stack is written coherently).

Rubric (0..1; latched partial credit anchored in the demonstrated solve trajectory):
  0.15 * engaged_ever   — the spike ever inside the jar through the bottom hole
  0.25 * opened_ever    — the lid ever physically clear of the mouth (irreversible
                          in real play: the weld must shear first)
  0.20 * presented_ever — the ball ever riding high in the bore above the shoulder
                          (physically requires the lid gone AND the piston driven
                          up), order-coupled on opened_ever
  1.0 iff success()     — red ball at rest inside the dish pocket, the jar mouth
                          OPEN (lid clear), everything still (consecutive-step
                          counter) and finite. Non-success capped at 0.60.

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
    out[:, 1:] = -out[:, 1:]
    return out


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat([torch.zeros_like(q[:, :1]), v], dim=-1)
    return _qmul(_qmul(q, qv), _qinv(q))[:, 1:]


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qx(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 1] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


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


def _add_box(stage, path: str, *, center, size, color, collide: Callable):
    """One box child: translate + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _rigid_dynamic(root, mass: float, *, com=None, lin_damp=0.05, ang_damp=0.05):
    """Author RigidBody + explicit Mass (+ CoM — MassAPI mass alone leaves the CoM
    at the body ORIGIN on this stack) + PhysX armor."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    if com is not None:
        m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)


def _bind_mat(prim_path: str, mat_path: str, static: float, dynamic: float) -> None:
    """Bind an explicit physics material (custom-spawner colliders otherwise get
    ~0.5 friction silently — the sliding spike/hole/piston path needs LOW mu)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    sim_utils.spawn_rigid_body_material(
        mat_path,
        sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic,
                                       restitution=0.0))
    bind_physics_material(prim_path, mat_path)


def _spawn_pedestal(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Heavy DYNAMIC pedestal (a joint anchored to a teleported kinematic body0
    stays world-fixed on this stack, and heavy-dynamic is the corpus convention for
    rock-solid fixtures): base slab + square spike + 8 mm pilot tip. Local frame:
    origin at the footprint centre on the ground."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, 12.0, com=(0.0, 0.0, 0.020), lin_damp=0.5, ang_damp=0.5)
    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/base", center=(0.0, 0.0, c.base_h / 2),
             size=(2 * c.base_half, 2 * c.base_half, c.base_h),
             color=c.base_color, collide=collide)
    shaft_h = c.spike_shaft_top - c.base_h
    _add_box(stage, f"{prim_path}/spike",
             center=(0.0, 0.0, c.base_h + shaft_h / 2),
             size=(c.spike_w, c.spike_w, shaft_h), color=c.spike_color, collide=collide)
    tip_h = c.spike_top - c.spike_shaft_top
    _add_box(stage, f"{prim_path}/tip",
             center=(0.0, 0.0, c.spike_shaft_top + tip_h / 2),
             size=(c.tip_w, c.tip_w, tip_h), color=c.spike_color, collide=collide)
    _bind_mat(f"{prim_path}/spike", f"{prim_path}/slipMat", 0.10, 0.08)
    _bind_mat(f"{prim_path}/tip", f"{prim_path}/slipMat2", 0.10, 0.08)
    return root


def _spawn_jar(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The jar: DYNAMIC square tube, origin at the footprint centre on the ground.
    Children: 4 bottom strips leaving the central through-hole, 4 bore walls
    (interior 48x48, top = the lid shoulder), 4 collar boxes (interior 56x56,
    rim top above). Low-mu material on the bottom strips (the hole edges ride the
    spike during the 90 mm press stroke)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, float(c.jar_mass), com=(0.0, 0.0, 0.045),
                   lin_damp=0.2, ang_damp=0.4)
    collide = _make_collide(c.contact_offset)
    oh, bh = c.outer_half, c.bore_half           # 0.032, 0.024
    hh, bt = c.hole_half, c.bottom_t             # 0.018, 0.008
    # bottom plate: two full-width x strips + two y strips between them
    sx = (oh - hh)                               # 0.014 strip width
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/bot_x{'p' if sgn > 0 else 'n'}",
                 center=(sgn * (hh + sx / 2), 0.0, bt / 2),
                 size=(sx, 2 * oh, bt), color=c.body_color, collide=collide)
        _add_box(stage, f"{prim_path}/bot_y{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * (hh + sx / 2), bt / 2),
                 size=(2 * hh, sx, bt), color=c.body_color, collide=collide)
        _bind_mat(f"{prim_path}/bot_x{'p' if sgn > 0 else 'n'}",
                  f"{prim_path}/slipX{'p' if sgn > 0 else 'n'}", 0.10, 0.08)
        _bind_mat(f"{prim_path}/bot_y{'p' if sgn > 0 else 'n'}",
                  f"{prim_path}/slipY{'p' if sgn > 0 else 'n'}", 0.10, 0.08)
    # bore walls z bt..shoulder (x pair full length, y pair between)
    wt = oh - bh                                 # 0.008
    wh = c.shoulder_z - bt
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/wall_x{'p' if sgn > 0 else 'n'}",
                 center=(sgn * (bh + wt / 2), 0.0, bt + wh / 2),
                 size=(wt, 2 * oh, wh), color=c.body_color, collide=collide)
        _add_box(stage, f"{prim_path}/wall_y{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * (bh + wt / 2), bt + wh / 2),
                 size=(2 * bh, wt, wh), color=c.body_color, collide=collide)
    # mouth collar z shoulder..rim (interior widens to 2*collar_in)
    ct = oh - c.collar_in                        # 0.004
    ch = c.rim_z - c.shoulder_z
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/col_x{'p' if sgn > 0 else 'n'}",
                 center=(sgn * (c.collar_in + ct / 2), 0.0, c.shoulder_z + ch / 2),
                 size=(ct, 2 * oh, ch), color=c.trim_color, collide=collide)
        _add_box(stage, f"{prim_path}/col_y{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * (c.collar_in + ct / 2), c.shoulder_z + ch / 2),
                 size=(2 * c.collar_in, ct, ch), color=c.trim_color, collide=collide)
    return root


def _spawn_piston(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The ejector piston: plate + 4 tray fences (pocket keeps the riding ball
    centred), plus the Z PRISMATIC joint to the sibling jar (authored at spawn).
    Body origin at the plate BOTTOM centre. Rests hanging on the lower limit."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, float(c.piston_mass), com=(0.0, 0.0, 0.006),
                   lin_damp=0.2, ang_damp=0.2)
    collide = _make_collide(c.contact_offset)
    ph, pt = c.plate_half, c.plate_t             # 0.023, 0.010
    _add_box(stage, f"{prim_path}/plate", center=(0.0, 0.0, pt / 2),
             size=(2 * ph, 2 * ph, pt), color=c.piston_color, collide=collide)
    _bind_mat(f"{prim_path}/plate", f"{prim_path}/slipMat", 0.30, 0.25)
    fh, ft = c.fence_h, c.fence_t
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/fence_x{'p' if sgn > 0 else 'n'}",
                 center=(sgn * (ph - ft / 2), 0.0, pt + fh / 2),
                 size=(ft, 2 * ph, fh), color=c.piston_color, collide=collide)
        _add_box(stage, f"{prim_path}/fence_y{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * (ph - ft / 2), pt + fh / 2),
                 size=(2 * (ph - ft), ft, fh), color=c.piston_color, collide=collide)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/slide")
    j.CreateBody0Rel().SetTargets([f"{base}/Jar"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.rest_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-0.001)
    j.CreateUpperLimitAttr(float(c.stroke))
    # viscous drag on the slide (a real ejector piston is not frictionless): a
    # pure damper (no stiffness, target velocity 0). Invisible to the slow press
    # (0.04 m/s x 2 N.s/m ~ 0.1 N) but it stops the piston FREE-FALLING its
    # stroke when the jar is inverted and hammering the ball into the lid with
    # an impulse spike that could shear the seal without any press.
    drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    drv.CreateTypeAttr("force")
    drv.CreateStiffnessAttr(0.0)
    drv.CreateDampingAttr(float(c.damping))
    drv.CreateTargetVelocityAttr(0.0)
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The seal lid: one bare plate (nothing to pinch), CoM offset +x so a
    centre-lifted lid TIPS off the rim, welded to the sibling jar by a BREAKABLE
    fixed joint with collision ENABLED (after the shear the free lid must collide
    with the jar it used to seal). Authored 0.5 mm above the shoulder: an outside
    down-press is carried by shoulder contact, only an inside up-push loads the
    weld."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, float(c.lid_mass), com=(float(c.lid_com_x), 0.0, 0.0),
                   lin_damp=0.1, ang_damp=0.1)
    collide = _make_collide(c.lid_contact_offset)
    _add_box(stage, f"{prim_path}/plate", center=(0.0, 0.0, 0.0),
             size=(2 * c.lid_half, 2 * c.lid_half, c.lid_t),
             color=c.lid_color, collide=collide)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/seal")
    j.CreateBody0Rel().SetTargets([f"{base}/Jar"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.lid_seat_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateBreakForceAttr(float(c.break_force))
    j.CreateBreakTorqueAttr(float(c.break_torque))
    j.CreateCollisionEnabledAttr(True)
    return root


def _spawn_dish(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The delivery dish: DYNAMIC walled tray (base plate + 4 fences)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, 1.2, com=(0.0, 0.0, 0.006), lin_damp=0.3, ang_damp=0.3)
    collide = _make_collide(c.contact_offset)
    dh, dt = c.dish_half, c.dish_t
    _add_box(stage, f"{prim_path}/base", center=(0.0, 0.0, dt / 2),
             size=(2 * dh, 2 * dh, dt), color=c.dish_color, collide=collide)
    fh, ft = c.dish_fence_h, c.dish_fence_t
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/fence_x{'p' if sgn > 0 else 'n'}",
                 center=(sgn * (dh - ft / 2), 0.0, dt + fh / 2),
                 size=(ft, 2 * dh, fh), color=c.dish_color, collide=collide)
        _add_box(stage, f"{prim_path}/fence_y{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * (dh - ft / 2), dt + fh / 2),
                 size=(2 * (dh - ft), ft, fh), color=c.dish_color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pedestal" not in _SPAWNER_CACHE:

        @configclass
        class PedestalSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pedestal)
            base_half: float = 0.085
            base_h: float = 0.030
            spike_w: float = 0.016
            tip_w: float = 0.008
            spike_shaft_top: float = 0.106
            spike_top: float = 0.116
            base_color: tuple = (0.25, 0.25, 0.28)
            spike_color: tuple = (0.95, 0.55, 0.10)
            contact_offset: float = 0.002

        @configclass
        class JarSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_jar)
            outer_half: float = 0.032
            bore_half: float = 0.024
            hole_half: float = 0.018
            bottom_t: float = 0.008
            shoulder_z: float = 0.092
            collar_in: float = 0.028
            rim_z: float = 0.102
            jar_mass: float = 0.45
            body_color: tuple = (0.16, 0.45, 0.42)
            trim_color: tuple = (0.10, 0.30, 0.28)
            contact_offset: float = 0.002

        @configclass
        class PistonSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_piston)
            plate_half: float = 0.023
            plate_t: float = 0.010
            fence_h: float = 0.006
            fence_t: float = 0.006
            rest_z: float = 0.010
            stroke: float = 0.080
            damping: float = 2.0
            piston_mass: float = 0.06
            piston_color: tuple = (0.90, 0.80, 0.20)
            contact_offset: float = 0.002

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            lid_half: float = 0.026
            lid_t: float = 0.006
            lid_seat_z: float = 0.0955
            lid_mass: float = 0.04
            lid_com_x: float = 0.010
            break_force: float = 10.0
            break_torque: float = 0.5
            lid_color: tuple = (0.78, 0.78, 0.80)
            lid_contact_offset: float = 0.001

        @configclass
        class DishSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_dish)
            dish_half: float = 0.060
            dish_t: float = 0.008
            dish_fence_h: float = 0.014
            dish_fence_t: float = 0.006
            dish_color: tuple = (0.15, 0.30, 0.85)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["pedestal"] = PedestalSpawnerCfg
        _SPAWNER_CACHE["jar"] = JarSpawnerCfg
        _SPAWNER_CACHE["piston"] = PistonSpawnerCfg
        _SPAWNER_CACHE["lid"] = LidSpawnerCfg
        _SPAWNER_CACHE["dish"] = DishSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PistonJarSceneCfg(BaseCfg):
    """Config for `PistonJarScene`. The stack geometry (jar frame, z up from the
    footprint): bottom plate 0..8 with the 36x36 hole; piston plate rests 10..20
    (tray fences to 26); ball centre 35; bore walls to the SHOULDER at 92; lid
    welded 92.5..98.5; collar to the RIM at 102. Spike tip at pedestal z 116; the
    jar pressed to the pedestal base (z 30) drives the piston plate top to 96 —
    ball centre 111, nine millimetres proud of the rim."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    covered_xy: float = tunable(0.030)   # lid centre within this box of the jar axis...
    covered_z_lo: float = tunable(0.070)  # ...and this z window (jar frame) = mouth COVERED
    covered_z_hi: float = tunable(0.135)  # (seated lid centre: 0.0955; flat on the rim: 0.105)
    engage_xy: float = tunable(0.020)    # spike tip within this of the jar axis...
    engage_z_lo: float = tunable(0.000)  # ...and inside this z window = spike ENGAGED
    engage_z_hi: float = tunable(0.100)
    present_xy: float = tunable(0.026)   # ball centre near the jar axis...
    present_z_lo: float = tunable(0.090)  # ...above the shoulder = ball PRESENTED (final: 0.111)
    present_z_hi: float = tunable(0.160)
    injar_z_hi: float = tunable(0.075)   # ball centre below this (jar frame) = still captive
    dish_xy: float = tunable(0.042)      # ball centre within this box of the dish axis...
    dish_z_lo: float = tunable(0.010)    # ...and this z window (dish frame) = IN the dish
    dish_z_hi: float = tunable(0.045)    # (rest: 0.023)
    dish_up_min: float = tunable(0.90)   # dish up-axis z component (tray upright)
    settle_speed: float = tunable(0.05)  # instantaneous |lin vel| gate (m/s)
    still_steps: int = tunable(60)       # consecutive still substeps before "still" (0.5 s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    ped_jitter: float = tunable(0.050)   # pedestal xy jitter (+/- m)
    ped_yaw_deg: float = tunable(25.0)   # pedestal yaw (+/- deg)
    jar_jitter: float = tunable(0.040)   # jar xy jitter (+/- m)
    jar_yaw_deg: float = tunable(180.0)  # jar free yaw (+/- deg)
    dish_jitter: float = tunable(0.040)  # dish xy jitter (+/- m)
    dish_yaw_deg: float = tunable(180.0)  # dish yaw (+/- deg)

    # --- info: layout (world nominal) ------------------------------------------------------------
    ped_pos: tuple = info((0.46, 0.10))  # pedestal origin on the ground
    jar_pos: tuple = info((0.16, -0.20))  # jar footprint centre
    dish_pos: tuple = info((0.16, 0.24))  # dish centre
    # --- info: pedestal --------------------------------------------------------------------------
    base_half: float = info(0.085)
    base_h: float = info(0.030)          # pedestal base top = the press end stop
    spike_w: float = info(0.016)
    spike_top: float = info(0.116)       # spike tip height (pedestal frame)
    # --- info: jar stack (jar frame) -------------------------------------------------------------
    outer_half: float = info(0.032)
    bore_half: float = info(0.024)
    hole_half: float = info(0.018)
    bottom_t: float = info(0.008)
    shoulder_z: float = info(0.092)      # bore wall top = lid support shoulder
    collar_in: float = info(0.028)
    rim_z: float = info(0.102)
    jar_mass: float = info(0.45)         # 4.5 N through the seal when parked — under threshold
    piston_rest_z: float = info(0.010)   # piston plate bottom at rest
    piston_stroke: float = info(0.080)
    piston_damping: float = info(2.0)    # slide drag (N.s/m): no free-fall hammer inverted
    plate_t: float = info(0.010)
    lid_seat_z: float = info(0.0955)     # lid centre, welded (0.5 mm above the shoulder)
    lid_half: float = info(0.026)
    lid_t: float = info(0.006)
    break_force: float = info(10.0)      # the seal: PhysX breakForce (N), irreversible
    break_torque: float = info(0.5)
    ball_r: float = info(0.015)
    ball_start_z: float = info(0.036)    # ball centre at rest in the tray
    # --- info: dish ------------------------------------------------------------------------------
    dish_half: float = info(0.060)
    dish_fence_h: float = info(0.014)
    dish_t: float = info(0.008)
    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.25 + 0.20 = 0.60 = the non-success cap)
    w_engage: float = info(0.15)
    w_open: float = info(0.25)
    w_present: float = info(0.20)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("piston_jar")
class PistonJarScene(BaseScene):
    cfg: PistonJarSceneCfg

    def __init__(self, cfg: PistonJarSceneCfg | None = None) -> None:
        super().__init__(cfg or PistonJarSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        jx, jy = c.jar_pos

        # spawn order matters: the piston/lid joints reference the sibling Jar prim.
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "pedestal": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pedestal",
                spawn=cls["pedestal"](),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.ped_pos[0], c.ped_pos[1], 0.0)),
            ),
            "jar": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Jar",
                spawn=cls["jar"](jar_mass=c.jar_mass),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(jx, jy, 0.0)),
            ),
            "piston": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Piston",
                spawn=cls["piston"](rest_z=c.piston_rest_z, stroke=c.piston_stroke,
                                    damping=c.piston_damping),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(jx, jy, c.piston_rest_z)),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=cls["lid"](lid_seat_z=c.lid_seat_z, break_force=c.break_force,
                                 break_torque=c.break_torque),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(jx, jy, c.lid_seat_z)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BallRed",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.08, 0.08)),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.3,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=1),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.5, dynamic_friction=0.4, restitution=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(jx, jy, c.ball_start_z)),
            ),
            "dish": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dish",
                spawn=cls["dish"](),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.dish_pos[0], c.dish_pos[1], 0.0)),
            ),
        }
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
        self.pedestal: RigidObject = env.iscene["pedestal"]
        self.jar: RigidObject = env.iscene["jar"]
        self.piston: RigidObject = env.iscene["piston"]
        self.lid: RigidObject = env.iscene["lid"]
        self.ball: RigidObject = env.iscene["ball"]
        self.dish: RigidObject = env.iscene["dish"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches (partial credit survives transients; success is judged live)
        self._engaged_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._opened_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._presented_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        # consecutive-still counter (instantaneous stillness passes at oscillation
        # turning points — latch a counter instead)
        self._still_count = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the pedestal (xy + yaw), the SEALED jar stack
        (jar + piston + lid + ball written coherently: same yaw, offsets on the jar
        axis), and the dish (xy + yaw); clear the latches and the still counter.
        NOTE: a sheared seal is IRREVERSIBLE across reset (PhysX breakable joints
        do not re-weld) — batteries must order intact-seal probes first."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def place(body, nominal, jitter, yaw_deg, dz=0.0, quat=None):
            yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(yaw_deg)
            q = _qz(yaw) if quat is None else quat
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = nominal[0]
            st[:, 1] = nominal[1]
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * jitter
            st[:, 2] = dz
            st[:, 0:3] += origin
            st[:, 3:7] = q
            body.write_root_state_to_sim(st, env_ids)
            return st[:, 0:3] - origin, q

        place(self.pedestal, c.ped_pos, c.ped_jitter, c.ped_yaw_deg)
        jar_p, jar_q = place(self.jar, c.jar_pos, c.jar_jitter, c.jar_yaw_deg)
        # the sealed stack rides the jar: offsets are ON the jar axis (yaw-invariant
        # positions), orientations share the jar yaw (weld/prismatic-consistent)
        for body, dz, share_q in ((self.piston, c.piston_rest_z, True),
                                  (self.lid, c.lid_seat_z, True),
                                  (self.ball, c.ball_start_z, False)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = jar_p + origin
            st[:, 2] += dz
            st[:, 3:7] = jar_q if share_q else _qz(torch.zeros(m, device=dev))
            body.write_root_state_to_sim(st, env_ids)
        place(self.dish, c.dish_pos, c.dish_jitter, c.dish_yaw_deg)

        self._engaged_ever[env_ids] = False
        self._opened_ever[env_ids] = False
        self._presented_ever[env_ids] = False
        self._still_count[env_ids] = 0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pedestal": self.pedestal.data.root_state_w[env_ids].clone(),
            "jar": self.jar.data.root_state_w[env_ids].clone(),
            "piston": self.piston.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "dish": self.dish.data.root_state_w[env_ids].clone(),
            "engaged_ever": self._engaged_ever[env_ids].clone(),
            "opened_ever": self._opened_ever[env_ids].clone(),
            "presented_ever": self._presented_ever[env_ids].clone(),
            "still_count": self._still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.pedestal.write_root_state_to_sim(state["pedestal"], env_ids)
        self.jar.write_root_state_to_sim(state["jar"], env_ids)
        self.piston.write_root_state_to_sim(state["piston"], env_ids)
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        self.dish.write_root_state_to_sim(state["dish"], env_ids)
        self._engaged_ever[env_ids] = state["engaged_ever"]
        self._opened_ever[env_ids] = state["opened_ever"]
        self._presented_ever[env_ids] = state["presented_ever"]
        self._still_count[env_ids] = state["still_count"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "On the ground stand three things. (1) A dark PEDESTAL block "
            "(170 mm square, 30 mm tall) carrying a vertical ORANGE SPIKE (16 mm "
            "square, tip 116 mm up, with a slimmer pilot tip). (2) A teal square "
            "JAR (64 mm wide, 102 mm tall) standing mouth-up, SEALED: recessed "
            "4 mm below its rim sits a bare light-grey LID with only a 2 mm gap "
            "around it — there is nothing to pinch, and the lid is factory-sealed "
            "in place (it does NOT lift, pry, twist or shake off; the seal holds "
            "many times the weight of everything in the scene). Inside, under the "
            "lid, a RED BALL (30 mm) rides in the tray of a yellow ejector PISTON, "
            "and the jar's BOTTOM plate has a 36 mm square through-hole leading to "
            "that piston. (3) A blue walled DISH (120 mm square, 14 mm walls). All "
            "three positions and headings vary per episode.\n"
            "Goal: get the RED BALL to rest inside the blue dish. The only way in "
            "is the jar's press-to-eject mechanism: pick the jar up, carry it over "
            "the pedestal, align the bottom hole with the orange spike, and PRESS "
            "the jar straight down so the spike passes through the hole and drives "
            "the internal piston up. Simply resting the jar on the spike is NOT "
            "enough — the seal needs a firm sustained push (roughly ten newtons, "
            "about twice the jar's weight) transmitted through the ball into the "
            "lid before it shears. Keep pressing until the jar bottoms out on the "
            "pedestal base: the popped lid tips off the rim and the red ball ends "
            "riding above the rim, presented in the piston tray. Pick the ball and "
            "set it down inside the blue dish. Success: the jar mouth stands open "
            "(lid clear of it) and the red ball rests still inside the dish walls."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the sealed jar's bottom hole down over the orange spike and "
            "press the jar firmly to the pedestal base so the internal piston pops "
            "the lid and raises the red ball, then place the red ball inside the "
            "blue dish."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _jar_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the (live-read) jar frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.jar.data.root_quat_w,
                                  pos_w - self.jar.data.root_pos_w)

    def _dish_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.dish.data.root_quat_w,
                                  pos_w - self.dish.data.root_pos_w)

    def spike_tip_w(self) -> torch.Tensor:
        """(N,3) the spike tip point in world (live pedestal pose)."""
        n = self.env.num_envs
        tip = torch.zeros(n, 3, device=self.env.device)
        tip[:, 2] = self.cfg.spike_top
        return self.pedestal.data.root_pos_w + _qapply(
            self.pedestal.data.root_quat_w, tip)

    def covered(self) -> torch.Tensor:
        """(N,) bool: the LID physically covers the jar mouth (lid centre inside
        the mouth box in the jar frame — seated, or resting anywhere on/over the
        mouth). jar_open = NOT covered."""
        c = self.cfg
        loc = self._jar_local(self.lid.data.root_pos_w)
        return (loc[:, 0].abs() < c.covered_xy) & (loc[:, 1].abs() < c.covered_xy) \
            & (loc[:, 2] > c.covered_z_lo) & (loc[:, 2] < c.covered_z_hi)

    def jar_open(self) -> torch.Tensor:
        return ~self.covered()

    def engaged(self) -> torch.Tensor:
        """(N,) bool: the spike tip is INSIDE the jar, through the bottom hole."""
        c = self.cfg
        loc = self._jar_local(self.spike_tip_w())
        return (loc[:, 0].abs() < c.engage_xy) & (loc[:, 1].abs() < c.engage_xy) \
            & (loc[:, 2] > c.engage_z_lo) & (loc[:, 2] < c.engage_z_hi)

    def presented(self) -> torch.Tensor:
        """(N,) bool: the ball rides high in the bore, above the shoulder — the
        piston-up presentation (physically impossible with the lid welded: a
        captive ball centre cannot exceed shoulder - ball_r)."""
        c = self.cfg
        loc = self._jar_local(self.ball.data.root_pos_w)
        return (loc[:, 0].abs() < c.present_xy) & (loc[:, 1].abs() < c.present_xy) \
            & (loc[:, 2] > c.present_z_lo) & (loc[:, 2] < c.present_z_hi)

    def ball_in_jar(self) -> torch.Tensor:
        """(N,) bool: the ball is still captive in the lower bore."""
        c = self.cfg
        loc = self._jar_local(self.ball.data.root_pos_w)
        return (loc[:, 0].abs() < c.bore_half) & (loc[:, 1].abs() < c.bore_half) \
            & (loc[:, 2] > 0.005) & (loc[:, 2] < c.injar_z_hi)

    def ball_in_dish(self) -> torch.Tensor:
        """(N,) bool: ball centre inside the dish pocket (dish frame), dish upright."""
        c = self.cfg
        loc = self._dish_local(self.ball.data.root_pos_w)
        n = self.env.num_envs
        ez = torch.zeros(n, 3, device=self.env.device)
        ez[:, 2] = 1.0
        up = _qapply(self.dish.data.root_quat_w, ez)
        return (loc[:, 0].abs() < c.dish_xy) & (loc[:, 1].abs() < c.dish_xy) \
            & (loc[:, 2] > c.dish_z_lo) & (loc[:, 2] < c.dish_z_hi) \
            & (up[:, 2] > c.dish_up_min)

    def still(self) -> torch.Tensor:
        """(N,) bool: everything has been instantaneously-still for `still_steps`
        consecutive substeps."""
        return self._still_count >= self.cfg.still_steps

    def _inst_still(self) -> torch.Tensor:
        c = self.cfg
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in (self.ball, self.lid, self.jar,
                                   self.piston, self.dish)], dim=1)
        return (v < c.settle_speed).all(dim=1)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in
                         (self.pedestal, self.jar, self.piston,
                          self.lid, self.ball, self.dish)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        """Credit latches (idempotent — safe to call repeatedly). presented is
        order-coupled on opened (it is physically downstream of the shear)."""
        fin = self._finite()
        self._engaged_ever |= self.engaged() & fin
        self._opened_ever |= self.jar_open() & fin
        self._presented_ever |= self._opened_ever & self.presented() & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()
        inst = self._inst_still()
        self._still_count = torch.where(inst, self._still_count + 1,
                                        torch.zeros_like(self._still_count))

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the red ball at rest inside the dish pocket AND the jar
        mouth open (lid physically clear of it), everything still (counter) and
        finite — all live physical outcomes. A ball teleported to the dish past a
        still-sealed jar does NOT succeed: the open-mouth clause is load-bearing."""
        self._update_latches()
        return self.ball_in_dish() & self.jar_open() & self.still() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15*engaged_ever + 0.25*opened_ever +
        0.20*presented_ever (latched; presented coupled on opened), capped at
        0.60 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_engage * self._engaged_ever.float()
                + c.w_open * self._opened_ever.float()
                + c.w_present * self._presented_ever.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="piston_jar", robot="null"))
