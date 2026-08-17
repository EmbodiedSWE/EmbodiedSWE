"""UtensilBalanceScene — weigh a hidden counterweight on a two-pan balance scale by
placing kitchen utensils in its empty pan until the beam sits level (sim_gen task
`track_spoon_i160`).

Derived from pick_place/track_spoon, but STRATEGICALLY different: the seed starts with
a spoon ALREADY rigidly grasped in the closed Franka gripper and rewards dense per-step
position+rotation tracking of a prescribed free-space waypoint path toward a basket —
pure transport fidelity of a held payload, with nothing to discover and no mechanism in
the loop. Here the judged skill is INTERACTIVE MASS IDENTIFICATION THROUGH A COMPLIANT
MECHANISM: a two-pan balance scale holds a hidden counterweight in one hanging pan (the
three candidate weights look identical — mass is invisible to any camera), and exactly
one of three kitchen utensils (brass fork 25 g, silver spoon 45 g, dark ladle 75 g) has
the same mass. The solver must USE the scale — load a utensil into the empty pan, READ
the beam's settled tilt (level within the tolerance only for the matching mass; every
wrong pairing pins the beam at its +/-28 deg stop), swap utensils as needed — and leave
the scene with the match in the empty pan, the beam level, and the two rejected
utensils laid flat on the floor away from the scale. There is no prescribed path, no
per-step tracking, and the payload placement is only the *input* to a mechanism whose
settled RESPONSE is what is judged. A solver therefore needs a different plan
(hypothesis testing against a physical readout, not trajectory following) and different
code (mechanism-settling predicates and a discovery loop instead of a waypoint
follower).

Judged on the settled state, in body frames:
  success() iff  the hidden ACTIVE weight rests in one pan  AND  the matching utensil
  rests in the OTHER pan  AND  the beam is level (|pitch| <= level_deg)  AND  both
  non-matching utensils lie flat on the floor clear of the scale  AND  the two unused
  (parked) weights stay far from the scale  AND  everything is at rest.
score() is latched every physics substep: 0.20 * a utensil was ever loaded in the free
pan while the weight sat in its pan  +  0.30 * the match-vs-weight arrangement was
ever level and quiet  +  0.20 * success() ever held, capped at 0.70; exactly 1.0 iff
success() holds now. Doing nothing scores ~0 (the weight alone pins the beam at its
stop — visibly, since the keel pointer swings with the beam).

Assets are fully procedural (no external meshes):
  - stand: DYNAMIC (25 kg, never teleported after reset — spawn-authored joint anchors
    on a teleported KINEMATIC body0 stay world-fixed on this stack) base slab + column
    + clevis, xy + yaw randomized per episode;
  - beam: 0.36 m bar on a spawn-authored revolute joint (axis Y) at the clevis,
    limits +/- stop_deg; 0.40 kg with its authored CoM 10 mm BELOW the pivot (a keel,
    drawn as a red pointer fin) so the empty beam is stable and level, and
    tan(theta) = dm * L / (M_bar * d_keel) maps mass error to tilt: 20 g of mismatch
    (the smallest wrong pairing) overwhelms the keel and pins the beam at the stop
    with a 1.5x torque margin, while the 6 deg level band corresponds to < 3 g;
  - pans: two hanging pans, each a walled tray on yoke rods under a spawn-authored
    revolute joint (axis Y) at a beam tip. HANGING pans make the beam torque depend
    only on the total load in the pan, not on where the load sits in the tray — the
    readout measures MASS, not placement luck; pan CoM 105 mm below its pivot keeps
    the tray plumb under off-centre loads;
  - utensils: fork/spoon/ladle — flat-headed, distinct sizes/colours, 25/45/75 g;
    handles attach at head-top level so a lying utensil's handle floats ~10 mm above
    the floor for a parallel-jaw pinch; all three fit fully inside a tray;
  - weights: three IDENTICAL-looking steel cylinders whose authored masses mirror the
    utensils (25/45/75 g). Per episode one is the ACTIVE counterweight resting in a
    random pan; the other two are parked ~2 m away and must stay away.
Masses are authored via MassAPI inside the custom spawners (custom spawn funcs ignore
cfg mass_props on this stack) and read back in smoke via get_masses().

Per-episode randomization (verified by readback in smoke): stand xy + yaw, the match
identity (which weight/utensil mass is active), which pan holds the weight, and the
utensils' floor slots (permutation + xy jitter + free yaw). Heavy imports (isaaclab,
pxr) are deferred so importing this module — and registering the scene — stays
app-free.
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


# ----- small torch quaternion helpers (wxyz) ---------------------------------------------------
def _qz(yaw: torch.Tensor) -> torch.Tensor:
    """(m,) yaw -> (m, 4) wxyz quaternion about world z."""
    q = torch.zeros(yaw.shape[0], 4, device=yaw.device)
    q[:, 0] = torch.cos(yaw / 2)
    q[:, 3] = torch.sin(yaw / 2)
    return q


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[:, :1], -q[:, 1:]], dim=1)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=1)


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ----- custom compound spawners ----------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _cyl(stage, path: str, radius: float, height: float, center, color,
         contact_offset: float) -> None:
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
    _collide(seg.GetPrim(), contact_offset)


def _rigid_dynamic(root, *, mass: float, com, lin_damp: float, ang_damp: float,
                   pos_iters: int = 16, vel_iters: int = 4) -> None:
    """Author a dynamic rigid body with EXPLICIT mass and CoM (MassAPI-only mass leaves
    the CoM at the body origin on this stack — always author both)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(int(pos_iters))
    px.CreateSolverVelocityIterationCountAttr(int(vel_iters))
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _revolute(stage, joint_path: str, body0_path: str, body1_path: str, *, axis: str,
              pos0, pos1, lo_deg: float, hi_deg: float) -> None:
    """Spawn-authored revolute joint (i79 pattern): body1's root origin sits AT the
    pivot; joint pairs are collision-filtered by PhysX."""
    from pxr import Gf, UsdPhysics

    j = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    j.CreateBody0Rel().SetTargets([body0_path])
    j.CreateBody1Rel().SetTargets([body1_path])
    j.CreateCollisionEnabledAttr(False)
    j.CreateAxisAttr(axis)
    j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in pos0]))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(*[float(v) for v in pos1]))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(float(lo_deg))
    j.CreateUpperLimitAttr(float(hi_deg))


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC (heavy) scale stand: base slab + column + clevis block. Local origin at
    the FLOOR under the column axis. Dynamic — never kinematic — because the beam's
    spawn-authored joint anchors would stay world-fixed if a kinematic body0 were ever
    teleported (reset teleports the stand)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, mass=cfg.mass, com=(0.0, 0.0, 0.04), lin_damp=2.0, ang_damp=2.0)

    co = cfg.contact_offset
    _box(stage, f"{prim_path}/base", (cfg.base_x, cfg.base_y, cfg.base_t),
         (0.0, 0.0, cfg.base_t / 2), cfg.base_color, co)
    _box(stage, f"{prim_path}/column", (cfg.col_w, cfg.col_w, cfg.col_top - cfg.base_t),
         (0.0, 0.0, (cfg.col_top + cfg.base_t) / 2), cfg.col_color, co)
    _box(stage, f"{prim_path}/clevis", (0.030, 0.030, 0.022),
         (0.0, 0.0, cfg.col_top + 0.011), cfg.col_color, co)
    return root


def _stand_spawner_cfg(*, base_x: float, base_y: float, base_t: float, col_w: float,
                       col_top: float, mass: float, base_color: tuple, col_color: tuple,
                       contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "stand" not in _SPAWNER_CACHE:

        @configclass
        class BalanceStandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            base_x: float = 0.34
            base_y: float = 0.22
            base_t: float = 0.024
            col_w: float = 0.05
            col_top: float = 0.27
            mass: float = 25.0
            base_color: tuple = (0.35, 0.33, 0.30)
            col_color: tuple = (0.45, 0.43, 0.40)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["stand"] = BalanceStandSpawnerCfg

    return _SPAWNER_CACHE["stand"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        base_x=base_x, base_y=base_y, base_t=base_t, col_w=col_w, col_top=col_top,
        mass=mass, base_color=base_color, col_color=col_color,
        contact_offset=contact_offset,
    )


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Balance beam. Local origin AT the pivot; the bar runs along local x, and a red
    keel/pointer fin hangs below the origin (it is also the authored CoM offset that
    gives the beam its restoring stiffness). A spawn-authored revolute joint (axis Y)
    ties the origin to the sibling Stand's clevis with +/- stop_deg limits."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, mass=cfg.mass, com=(0.0, 0.0, -cfg.keel_d),
                   lin_damp=0.2, ang_damp=cfg.ang_damp)

    co = cfg.contact_offset
    _box(stage, f"{prim_path}/bar", (cfg.bar_len, 0.024, 0.016), (0.0, 0.0, 0.0),
         cfg.color, co)
    _box(stage, f"{prim_path}/keel", (0.010, 0.006, 0.060), (0.0, 0.0, -0.038),
         cfg.keel_color, co)

    base = prim_path.rsplit("/", 1)[0]
    _revolute(stage, f"{prim_path}/pivot", f"{base}/Stand", prim_path, axis="Y",
              pos0=(0.0, 0.0, cfg.pivot_h), pos1=(0.0, 0.0, 0.0),
              lo_deg=-cfg.stop_deg, hi_deg=cfg.stop_deg)
    return root


def _beam_spawner_cfg(*, bar_len: float, mass: float, keel_d: float, pivot_h: float,
                      stop_deg: float, ang_damp: float, color: tuple, keel_color: tuple,
                      contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "beam" not in _SPAWNER_CACHE:

        @configclass
        class BalanceBeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            bar_len: float = 0.36
            mass: float = 0.40
            keel_d: float = 0.010
            pivot_h: float = 0.30
            stop_deg: float = 28.0
            ang_damp: float = 12.0
            color: tuple = (0.55, 0.38, 0.16)
            keel_color: tuple = (0.85, 0.10, 0.08)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["beam"] = BalanceBeamSpawnerCfg

    return _SPAWNER_CACHE["beam"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        bar_len=bar_len, mass=mass, keel_d=keel_d, pivot_h=pivot_h, stop_deg=stop_deg,
        ang_damp=ang_damp, color=color, keel_color=keel_color,
        contact_offset=contact_offset,
    )


def _spawn_pan(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Hanging pan. Local origin AT its pivot (the beam tip); yoke rods drop to a
    walled square tray whose floor top sits `hang_depth` below the pivot. Authored CoM
    well below the pivot keeps the tray plumb under off-centre loads. A spawn-authored
    revolute joint (axis Y) ties the origin to the sibling Beam at (tip_x, 0, 0)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, mass=cfg.mass, com=(0.0, 0.0, -cfg.hang_depth + 0.005),
                   lin_damp=0.1, ang_damp=cfg.ang_damp)

    co = cfg.contact_offset
    h, th, wt, wh, ft = cfg.hang_depth, cfg.tray_half, cfg.wall_t, cfg.wall_h, cfg.floor_t
    # yoke rods on the +/- y sides (the tray's x-span stays open for utensil handles)
    for s in (1.0, -1.0):
        _box(stage, f"{prim_path}/rod_{'p' if s > 0 else 'n'}",
             (0.007, 0.007, h - 0.006), (0.0, s * (th + 0.0035), -(h - 0.006) / 2),
             cfg.rod_color, co)
    _box(stage, f"{prim_path}/floor", (2 * th, 2 * th, ft),
         (0.0, 0.0, -h - ft / 2), cfg.color, co)
    for s in (1.0, -1.0):
        _box(stage, f"{prim_path}/wall_x{'p' if s > 0 else 'n'}", (wt, 2 * th, wh),
             (s * (th - wt / 2), 0.0, -h + wh / 2), cfg.color, co)
        _box(stage, f"{prim_path}/wall_y{'p' if s > 0 else 'n'}",
             (2 * th - 2 * wt, wt, wh),
             (0.0, s * (th - wt / 2), -h + wh / 2), cfg.color, co)

    base = prim_path.rsplit("/", 1)[0]
    _revolute(stage, f"{prim_path}/pivot", f"{base}/Beam", prim_path, axis="Y",
              pos0=(cfg.tip_x, 0.0, 0.0), pos1=(0.0, 0.0, 0.0),
              lo_deg=-75.0, hi_deg=75.0)
    return root


def _pan_spawner_cfg(*, tip_x: float, hang_depth: float, tray_half: float, wall_t: float,
                     wall_h: float, floor_t: float, mass: float, ang_damp: float,
                     color: tuple, rod_color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pan" not in _SPAWNER_CACHE:

        @configclass
        class BalancePanSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pan)
            tip_x: float = 0.16
            hang_depth: float = 0.110
            tray_half: float = 0.0775
            wall_t: float = 0.005
            wall_h: float = 0.022
            floor_t: float = 0.005
            mass: float = 0.25
            ang_damp: float = 6.0
            color: tuple = (0.72, 0.72, 0.75)
            rod_color: tuple = (0.40, 0.40, 0.44)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["pan"] = BalancePanSpawnerCfg

    return _SPAWNER_CACHE["pan"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        tip_x=tip_x, hang_depth=hang_depth, tray_half=tray_half, wall_t=wall_t,
        wall_h=wall_h, floor_t=floor_t, mass=mass, ang_damp=ang_damp, color=color,
        rod_color=rod_color, contact_offset=contact_offset,
    )


def _spawn_utensil(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Kitchen utensil lying flat: a head block at the -x end (with an optional rim
    that makes the spoon 'bowl' and ladle 'cup' read distinctly) and a thinner handle
    attached at HEAD-TOP level, so the lying handle floats above the floor for a
    parallel-jaw pinch. Local origin at the centre of the overall length, z = floor
    plane. Authored CoM over the head keeps the cantilevered handle airborne."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    head_cx = -cfg.total_l / 2 + cfg.head_l / 2
    _rigid_dynamic(root, mass=cfg.mass, com=(head_cx, 0.0, 0.006),
                   lin_damp=0.1, ang_damp=1.0, vel_iters=1)

    co = cfg.contact_offset
    hh = cfg.head_h
    _box(stage, f"{prim_path}/head", (cfg.head_l, cfg.head_w, hh),
         (head_cx, 0.0, hh / 2), cfg.color, co)
    if cfg.rim_h > 0.0:
        rt = 0.004
        for s in (1.0, -1.0):
            _box(stage, f"{prim_path}/rim_x{'p' if s > 0 else 'n'}",
                 (rt, cfg.head_w, cfg.rim_h),
                 (head_cx + s * (cfg.head_l - rt) / 2, 0.0, hh + cfg.rim_h / 2),
                 cfg.color, co)
            _box(stage, f"{prim_path}/rim_y{'p' if s > 0 else 'n'}",
                 (cfg.head_l - 2 * rt, rt, cfg.rim_h),
                 (head_cx, s * (cfg.head_w - rt) / 2, hh + cfg.rim_h / 2),
                 cfg.color, co)
    handle_l = cfg.total_l - cfg.head_l
    _box(stage, f"{prim_path}/handle", (handle_l, cfg.handle_w, cfg.handle_t),
         (cfg.head_l / 2, 0.0, cfg.head_h + cfg.handle_t / 2 - 0.004),
         cfg.color, co)
    return root


def _utensil_spawner_cfg(*, total_l: float, head_l: float, head_w: float, head_h: float,
                         rim_h: float, handle_w: float, handle_t: float, mass: float,
                         color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "utensil" not in _SPAWNER_CACHE:

        @configclass
        class BalanceUtensilSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_utensil)
            total_l: float = 0.13
            head_l: float = 0.05
            head_w: float = 0.05
            head_h: float = 0.010
            rim_h: float = 0.0
            handle_w: float = 0.015
            handle_t: float = 0.008
            mass: float = 0.045
            color: tuple = (0.8, 0.8, 0.85)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["utensil"] = BalanceUtensilSpawnerCfg

    return _SPAWNER_CACHE["utensil"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        total_l=total_l, head_l=head_l, head_w=head_w, head_h=head_h, rim_h=rim_h,
        handle_w=handle_w, handle_t=handle_t, mass=mass, color=color,
        contact_offset=contact_offset,
    )


def _spawn_weight(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Counterweight: a steel cylinder, origin at its BOTTOM face centre. All three
    weights are visually identical — only the authored mass differs."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, mass=cfg.mass, com=(0.0, 0.0, cfg.height / 2),
                   lin_damp=0.1, ang_damp=1.0)
    _cyl(stage, f"{prim_path}/body", cfg.radius, cfg.height,
         (0.0, 0.0, cfg.height / 2), cfg.color, cfg.contact_offset)
    return root


def _weight_spawner_cfg(*, radius: float, height: float, mass: float, color: tuple,
                        contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "weight" not in _SPAWNER_CACHE:

        @configclass
        class BalanceWeightSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_weight)
            radius: float = 0.024
            height: float = 0.036
            mass: float = 0.045
            color: tuple = (0.52, 0.52, 0.56)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["weight"] = BalanceWeightSpawnerCfg

    return _SPAWNER_CACHE["weight"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        radius=radius, height=height, mass=mass, color=color,
        contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class UtensilBalanceCfg(BaseCfg):
    """Config for `UtensilBalanceScene`. Honesty knobs asserted in `__post_init__`:
    the smallest wrong pairing pins the beam at its stop with margin, the level band
    is several times tighter than the smallest wrong mass gap, every utensil fits a
    tray, handles are pinch-graspable, and the swinging pans clear the column and
    base at the stops."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    level_deg: float = tunable(6.0)  # |beam pitch| <= this counts as LEVEL
    settle_lin: float = tunable(0.05)  # max |lin vel| of movables when judging (m/s)
    settle_omega: float = tunable(0.06)  # max |beam ang vel| when judging (rad/s)
    floor_z_max: float = tunable(0.015)  # utensil origin below this = flat on the FLOOR
    clear_r: float = tunable(0.24)  # rejected utensils farther than this from the stand
    parked_r: float = tunable(0.50)  # unused weights farther than this from the stand
    in_margin: float = tunable(0.005)  # xy margin inside the tray for containment

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    stand_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the stand (m)
    stand_yaw_deg: float = tunable(15.0)  # uniform +/- stand yaw (judge in stand frame)
    slot_jitter: float = tunable(0.02)  # +/- xy jitter of each utensil floor slot (m)

    # --- tunable: placement ------------------------------------------------------------------
    stand_pos: tuple = tunable((0.45, 0.0))  # stand centre, WORLD xy nominal
    slot_x: float = tunable(0.12)  # utensil floor slots: nominal x (WORLD)
    slot_ys: tuple = tunable((-0.16, 0.0, 0.16))  # utensil floor slots: nominal y
    park_xys: tuple = tunable(((2.0, -0.5), (2.0, 0.0), (2.0, 0.5)))  # weight depot

    # --- info: stand -------------------------------------------------------------------------
    base_x: float = info(0.34)
    base_y: float = info(0.22)
    base_t: float = info(0.024)
    col_w: float = info(0.05)
    col_top: float = info(0.27)
    pivot_h: float = info(0.30)  # beam pivot height above the floor
    stand_mass: float = info(25.0)
    # --- info: beam (the readout) ------------------------------------------------------------
    arm_l: float = info(0.16)  # pivot -> pan-pivot arm length
    bar_len: float = info(0.36)
    beam_mass: float = info(0.40)
    keel_d: float = info(0.010)  # authored CoM depth below the pivot (restoring keel)
    stop_deg: float = info(28.0)  # joint limits: the beam's hard stops
    beam_ang_damp: float = info(12.0)  # near-critical: settles in ~1.5 swings
    # --- info: pans --------------------------------------------------------------------------
    hang_depth: float = info(0.110)  # pan pivot -> tray floor TOP
    tray_half: float = info(0.0775)  # tray outer half-extent (square)
    wall_t: float = info(0.005)
    wall_h: float = info(0.022)
    floor_t: float = info(0.005)
    pan_mass: float = info(0.25)
    pan_ang_damp: float = info(6.0)
    # --- info: utensils (fork, spoon, ladle) — masses mirror the weights ---------------------
    ute_names: tuple = info(("fork", "spoon", "ladle"))
    ute_masses: tuple = info((0.025, 0.045, 0.075))
    ute_total_l: tuple = info((0.134, 0.122, 0.130))
    ute_head_l: tuple = info((0.048, 0.052, 0.058))
    ute_head_w: tuple = info((0.044, 0.050, 0.056))
    ute_rim_h: tuple = info((0.0, 0.008, 0.016))
    ute_colors: tuple = info(((0.72, 0.55, 0.20), (0.80, 0.80, 0.85), (0.24, 0.26, 0.30)))
    head_h: float = info(0.010)
    handle_w: float = info(0.015)
    handle_t: float = info(0.008)
    # --- info: weights (visually identical, masses mirror the utensils) ---------------------
    wt_radius: float = info(0.024)
    wt_height: float = info(0.036)
    wt_color: tuple = info((0.52, 0.52, 0.56))
    jaw_span: float = info(0.080)  # the Franka parallel jaw (embodiment argument)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    tray_inner: float = field(default=None, init=False)  # tray inner span

    def __post_init__(self) -> None:
        g_ignore = 9.81  # cancels in every ratio below; named for clarity
        self.tray_inner = 2 * (self.tray_half - self.wall_t)
        keel = self.beam_mass * self.keel_d
        stop = math.radians(self.stop_deg)
        gaps = [abs(a - b) for i, a in enumerate(self.ute_masses)
                for b in list(self.ute_masses)[i + 1:]]
        dm_min = min(gaps)

        # every wrong pairing PINS the beam at its stop with torque margin
        assert dm_min * self.arm_l * math.cos(stop) >= 1.25 * keel * math.sin(stop), (
            "smallest wrong mass gap must pin the beam at the stop")
        # the level band is far tighter than the smallest wrong gap
        dm_level = math.tan(math.radians(self.level_deg)) * keel / self.arm_l
        assert dm_min >= 4.0 * dm_level, "level band must cleanly separate wrong pairings"
        # the lightest weight ALONE pins the beam (the reset state is visibly loaded)
        assert min(self.ute_masses) * self.arm_l * math.cos(stop) >= 1.4 * keel * math.sin(stop), (
            "the weight alone must pin the beam at its stop")
        # every utensil fits fully inside a tray; so does a weight
        assert max(self.ute_total_l) <= self.tray_inner - 0.006, "utensils must fit the tray"
        assert 2 * self.wt_radius <= self.tray_inner - 0.02, "weight must fit the tray"
        # lying utensil handles float and are pinch-graspable
        handle_bot = self.head_h - 0.004
        assert handle_bot >= 0.005, "lying handle must float clear of the floor"
        assert self.handle_w + 0.01 <= self.jaw_span, "handle must fit the parallel jaw"
        # pans clear the column and the base slab at the stops
        assert (self.arm_l * math.cos(stop) - self.tray_half - self.col_w / 2) >= 0.02, (
            "swinging pan must clear the column")
        low_tray = self.pivot_h - self.arm_l * math.sin(stop) - self.hang_depth - self.floor_t
        assert low_tray >= self.base_t + 0.04, "lowest tray must clear the base slab"
        # the floor slots are clear of the stand by more than the clear_r clause + jitters
        min_slot = min(math.hypot(self.slot_x - self.stand_pos[0], sy - self.stand_pos[1])
                       for sy in self.slot_ys)
        assert min_slot >= self.clear_r + self.slot_jitter + self.stand_jitter + 0.01, (
            "utensil floor slots must satisfy the clear_r clause at spawn")
        # masses strictly increasing with real gaps (identity is discoverable)
        ms = list(self.ute_masses)
        assert all(b - a >= 0.019 for a, b in zip(ms, ms[1:])), "mass gaps must be real"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("utensil_balance")
class UtensilBalanceScene(BaseScene):
    cfg: UtensilBalanceCfg

    def __init__(self, cfg: UtensilBalanceCfg | None = None) -> None:
        super().__init__(cfg or UtensilBalanceCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sx, sy = c.stand_pos
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
            # NOTE: dict order matters — joint body0 targets must exist when the
            # dependent spawner runs: Stand before Beam before the pans.
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=_stand_spawner_cfg(
                    base_x=c.base_x, base_y=c.base_y, base_t=c.base_t, col_w=c.col_w,
                    col_top=c.col_top, mass=c.stand_mass,
                    base_color=(0.35, 0.33, 0.30), col_color=(0.45, 0.43, 0.40),
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, 0.0)),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=_beam_spawner_cfg(
                    bar_len=c.bar_len, mass=c.beam_mass, keel_d=c.keel_d,
                    pivot_h=c.pivot_h, stop_deg=c.stop_deg, ang_damp=c.beam_ang_damp,
                    color=(0.55, 0.38, 0.16), keel_color=(0.85, 0.10, 0.08),
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, c.pivot_h)),
            ),
        }
        for name, tip in (("pan_l", -c.arm_l), ("pan_r", c.arm_l)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + ("PanL" if tip < 0 else "PanR"),
                spawn=_pan_spawner_cfg(
                    tip_x=tip, hang_depth=c.hang_depth, tray_half=c.tray_half,
                    wall_t=c.wall_t, wall_h=c.wall_h, floor_t=c.floor_t,
                    mass=c.pan_mass, ang_damp=c.pan_ang_damp,
                    color=(0.72, 0.72, 0.75), rod_color=(0.40, 0.40, 0.44),
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx + tip, sy, c.pivot_h)),
            )
        for j in range(3):
            out[f"ute_{j}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"Ute{j}",
                spawn=_utensil_spawner_cfg(
                    total_l=c.ute_total_l[j], head_l=c.ute_head_l[j],
                    head_w=c.ute_head_w[j], head_h=c.head_h, rim_h=c.ute_rim_h[j],
                    handle_w=c.handle_w, handle_t=c.handle_t, mass=c.ute_masses[j],
                    color=c.ute_colors[j], contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_x, c.slot_ys[j], 0.003)),
            )
            out[f"wt_{j}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"Wt{j}",
                spawn=_weight_spawner_cfg(
                    radius=c.wt_radius, height=c.wt_height, mass=c.ute_masses[j],
                    color=c.wt_color, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.park_xys[j][0], c.park_xys[j][1], 0.001)),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.stand: RigidObject = env.iscene["stand"]
        self.beam: RigidObject = env.iscene["beam"]
        self.pans: list[RigidObject] = [env.iscene["pan_l"], env.iscene["pan_r"]]
        self.utes: list[RigidObject] = [env.iscene[f"ute_{j}"] for j in range(3)]
        self.wts: list[RigidObject] = [env.iscene[f"wt_{j}"] for j in range(3)]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self._stand_xy = torch.tensor(self.cfg.stand_pos, device=dev).repeat(n, 1)
        self._stand_yaw = torch.zeros(n, device=dev)
        self._match = torch.zeros(n, dtype=torch.long, device=dev)  # active mass index
        self._side = torch.ones(n, device=dev)  # +1: weight in PanR (+x); -1: PanL
        self._engaged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._balanced = torch.zeros(n, dtype=torch.bool, device=dev)
        self._succ_ever = torch.zeros(n, dtype=torch.bool, device=dev)

    # ----- reset ------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the whole scale LINKAGE consistently (stand with xy
        jitter + yaw, beam LEVEL at the pivot, pans plumb at the tips — the beam then
        tips to its stop under the counterweight during the first settle), draw the
        match identity and the weight's pan with `torch.rand` (the first randint after
        a manual seed is degenerate on this stack), scatter the utensils over permuted
        floor slots, park the two unused weights at the depot, and clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        sxy = torch.tensor(c.stand_pos, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.stand_jitter
        syaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.stand_yaw_deg)
        qs = _qz(syaw)
        cy, sy = torch.cos(syaw), torch.sin(syaw)
        self._stand_xy[env_ids] = sxy
        self._stand_yaw[env_ids] = syaw

        def sframe(lx, ly):
            """(m, 2) world xy of a stand-local point."""
            if not torch.is_tensor(lx):
                lx = torch.full((m,), float(lx), device=dev)
            if not torch.is_tensor(ly):
                ly = torch.full((m,), float(ly), device=dev)
            return torch.stack([sxy[:, 0] + cy * lx - sy * ly,
                                sxy[:, 1] + sy * lx + cy * ly], dim=1)

        def write(body, wxy: torch.Tensor, z: float, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = wxy
            st[:, 2] = z
            st[:, 3:7] = quat
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- the linkage, written as a consistent whole (level beam, plumb pans) ---
        write(self.stand, sxy, 0.0, qs)
        write(self.beam, sxy, c.pivot_h, qs)
        write(self.pans[0], sframe(-c.arm_l, 0.0), c.pivot_h, qs)
        write(self.pans[1], sframe(c.arm_l, 0.0), c.pivot_h, qs)

        # --- hidden identity: which mass is active, and which pan holds it ---
        match = (torch.rand(m, device=dev) * 3).floor().long().clamp_(0, 2)
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           -torch.ones(m, device=dev), torch.ones(m, device=dev))
        self._match[env_ids] = match
        self._side[env_ids] = side

        pan_xy = sframe(side * c.arm_l, 0.0)
        wt_z = c.pivot_h - c.hang_depth + 0.004
        for j in range(3):
            act = match == j
            park = torch.tensor(c.park_xys[j], device=dev).expand(m, 2)
            wxy = torch.where(act.unsqueeze(1), pan_xy, park)
            z = torch.where(act, torch.full((m,), wt_z, device=dev),
                            torch.full((m,), 0.001, device=dev))
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = wxy
            st[:, 2] = z
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            self.wts[j].write_root_state_to_sim(st, env_ids)

        # --- utensils over permuted floor slots, xy jitter + free yaw ---
        perm = torch.stack([torch.randperm(3, device=dev) for _ in range(m)])  # (m, 3)
        slot_y = torch.tensor(c.slot_ys, device=dev)
        for j in range(3):
            ys = slot_y[perm[:, j]]
            jx = (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            jy = (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            uyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            wxy = torch.stack([torch.full((m,), c.slot_x, device=dev) + jx, ys + jy], dim=1)
            write(self.utes[j], wxy, 0.003, _qz(uyaw))

        self._engaged[env_ids] = False
        self._balanced[env_ids] = False
        self._succ_ever[env_ids] = False

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "stand_xy": self._stand_xy[env_ids].clone(),
            "stand_yaw": self._stand_yaw[env_ids].clone(),
            "match": self._match[env_ids].clone(),
            "side": self._side[env_ids].clone(),
            "engaged": self._engaged[env_ids].clone(),
            "balanced": self._balanced[env_ids].clone(),
            "succ_ever": self._succ_ever[env_ids].clone(),
        }
        for k, pan in enumerate(self.pans):
            out[f"pan_{k}"] = pan.data.root_state_w[env_ids].clone()
        for j in range(3):
            out[f"ute_{j}"] = self.utes[j].data.root_state_w[env_ids].clone()
            out[f"wt_{j}"] = self.wts[j].data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.beam.write_root_state_to_sim(state["beam"], env_ids)
        for k, pan in enumerate(self.pans):
            pan.write_root_state_to_sim(state[f"pan_{k}"], env_ids)
        for j in range(3):
            self.utes[j].write_root_state_to_sim(state[f"ute_{j}"], env_ids)
            self.wts[j].write_root_state_to_sim(state[f"wt_{j}"], env_ids)
        self._stand_xy[env_ids] = state["stand_xy"]
        self._stand_yaw[env_ids] = state["stand_yaw"]
        self._match[env_ids] = state["match"]
        self._side[env_ids] = state["side"]
        self._engaged[env_ids] = state["engaged"]
        self._balanced[env_ids] = state["balanced"]
        self._succ_ever[env_ids] = state["succ_ever"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A two-pan BALANCE SCALE stands near ({c.stand_pos[0]:.2f}, "
            f"{c.stand_pos[1]:.2f}) (its xy and yaw change per episode): a heavy base, "
            f"a column, and a {c.bar_len:.2f} m wooden beam pivoting {c.pivot_h:.2f} m "
            f"up, with a red pointer fin under the pivot. From each beam tip a walled "
            f"square tray ({c.tray_inner * 100:.1f} cm inside, "
            f"{c.wall_h * 1000:.0f} mm walls) HANGS on a free swing joint, its floor "
            f"about {c.pivot_h - c.hang_depth:.2f} m up. The beam tilts up to "
            f"{c.stop_deg:.0f} deg each way and is LEVEL only when the two pans carry "
            f"equal mass (within ~3 g); any wrong load pins it at the stop. One pan "
            f"holds a steel COUNTERWEIGHT cylinder. Three identical-looking spare "
            f"weights are parked ~2 m away and are irrelevant except that they must "
            f"stay there. On the floor in front of the scale lie three kitchen "
            f"utensils (positions shuffle per episode): a brass FORK (25 g), a silver "
            f"SPOON (45 g), and a dark LADLE (75 g). The counterweight's mass equals "
            f"exactly ONE of those three — which one is hidden and changes per "
            f"episode; all weights look alike, so the only way to find out is to USE "
            f"the scale: set a utensil in the empty pan and read the beam.\n"
            f"Goal: leave the matching utensil resting in the empty pan (counterweight "
            f"still in its pan) with the beam LEVEL within {c.level_deg:.0f} deg, and "
            f"lay the two rejected utensils flat on the floor at least "
            f"{c.clear_r:.2f} m from the scale's column (not on its base). Everything "
            f"must be at rest, and the parked weights must remain at least "
            f"{c.parked_r:.2f} m away. Utensil handles float above the floor for a "
            f"pinch grasp; every utensil fits fully inside a tray."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Find which utensil weighs the same as the counterweight sitting in the "
            "balance scale's pan: place utensils in the empty pan and watch the beam, "
            "which sits level only for an equal-mass load. Leave the matching utensil "
            "in the empty pan with the beam level, and lay the other two utensils "
            "flat on the floor well away from the scale."
        )

    # ----- readings / rubric ------------------------------------------------------------------
    def beam_pitch(self) -> torch.Tensor:
        """(N,) signed beam angle about the pivot axis relative to the LIVE stand
        (positive = the +x tip, PanR's side, tilts DOWN)."""
        qr = _quat_mul(_quat_conj(self.stand.data.root_quat_w),
                       self.beam.data.root_quat_w)
        return _wrap(2.0 * torch.atan2(qr[:, 2], qr[:, 0]))

    def in_pan(self, body: RigidObject, pan_k: int) -> torch.Tensor:
        """(N,) bool: body root inside pan k's tray volume, in the PAN's body frame
        (the pan hangs plumb at any beam angle, so this works mid-tilt too)."""
        from isaaclab.utils.math import quat_rotate_inverse

        c = self.cfg
        pan = self.pans[pan_k]
        d = quat_rotate_inverse(pan.data.root_quat_w,
                                body.data.root_pos_w - pan.data.root_pos_w)
        lim = c.tray_half - c.wall_t - c.in_margin
        return ((d[:, 0].abs() <= lim) & (d[:, 1].abs() <= lim)
                & (d[:, 2] >= -c.hang_depth - 0.008) & (d[:, 2] <= -c.hang_depth + 0.065))

    def _stand_dist(self, body: RigidObject) -> torch.Tensor:
        p = body.data.root_pos_w - self.env_origins
        sp = self.stand.data.root_pos_w - self.env_origins
        return (p[:, 0:2] - sp[:, 0:2]).norm(dim=-1)

    def ute_on_floor_clear(self, j: int) -> torch.Tensor:
        """(N,) bool: utensil j flat on the FLOOR (not on the base slab — the slab top
        alone lifts the origin past floor_z_max) and clear of the stand."""
        p = self.utes[j].data.root_pos_w - self.env_origins
        return ((p[:, 2] <= self.cfg.floor_z_max)
                & (self._stand_dist(self.utes[j]) >= self.cfg.clear_r)
                & ~self.in_pan(self.utes[j], 0) & ~self.in_pan(self.utes[j], 1))

    def _gather3(self, per_j: list[torch.Tensor]) -> torch.Tensor:
        """(N,) select per_j[match[i]][i] from three (N,) tensors."""
        stacked = torch.stack(per_j, dim=1)  # (N, 3)
        return stacked.gather(1, self._match.unsqueeze(1)).squeeze(1)

    def wt_in_pans(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,), (N,): the ACTIVE weight is in PanL / PanR."""
        in_l = self._gather3([self.in_pan(w, 0) for w in self.wts])
        in_r = self._gather3([self.in_pan(w, 1) for w in self.wts])
        return in_l, in_r

    def match_in_pans(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,), (N,): the MATCH utensil is in PanL / PanR."""
        in_l = self._gather3([self.in_pan(u, 0) for u in self.utes])
        in_r = self._gather3([self.in_pan(u, 1) for u in self.utes])
        return in_l, in_r

    def arrangement(self) -> torch.Tensor:
        """(N,) bool: active weight in one pan AND the match utensil in the OTHER."""
        wl, wr = self.wt_in_pans()
        ml, mr = self.match_in_pans()
        return (wl & mr) | (wr & ml)

    def level(self) -> torch.Tensor:
        return self.beam_pitch().abs() <= math.radians(self.cfg.level_deg)

    def others_on_floor(self) -> torch.Tensor:
        """(N,) bool: both NON-match utensils flat on the floor clear of the scale."""
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for j in range(3):
            ok &= self.ute_on_floor_clear(j) | (self._match == j)
        return ok

    def parked_far(self) -> torch.Tensor:
        """(N,) bool: both UNUSED weights farther than parked_r from the stand."""
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for j in range(3):
            ok &= (self._stand_dist(self.wts[j]) >= self.cfg.parked_r) | (self._match == j)
        return ok

    def settled(self) -> torch.Tensor:
        c = self.cfg
        ok = self.beam.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega
        for b in (*self.pans, *self.utes, *self.wts):
            ok &= b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        return ok

    def _update_latches(self) -> None:
        wl, wr = self.wt_in_pans()
        ute_l = torch.zeros_like(wl)
        ute_r = torch.zeros_like(wr)
        for u in self.utes:
            ute_l |= self.in_pan(u, 0)
            ute_r |= self.in_pan(u, 1)
        self._engaged |= (wl & ute_r) | (wr & ute_l)
        quiet = self.beam.data.root_ang_vel_w.norm(dim=-1) < 0.15
        self._balanced |= self.arrangement() & self.level() & quiet
        self._succ_ever |= self._success_now()

    def _success_now(self) -> torch.Tensor:
        return (self.arrangement() & self.level() & self.others_on_floor()
                & self.parked_far() & self.settled())

    # ----- step-coupled bookkeeping (every substep) -------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """The mechanism is fully passive (gravity + joints do the work) — post_step
        only latches partial credit so transient progress keeps its score."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: hidden weight resting in one pan, the MATCHING utensil resting
        in the other, beam LEVEL, both rejected utensils flat on the floor clear of
        the scale, unused weights still parked far away, everything at rest. Physical
        outcomes only — judged in body frames on the settled state."""
        self._update_latches()
        return self._success_now()

    def score(self) -> torch.Tensor:
        """Latched, monotone. 0.20 a utensil ever loaded opposite the seated weight;
        +0.30 the match-vs-weight arrangement ever level and quiet; +0.20 success()
        ever held; capped at 0.70. Exactly 1.0 iff success() holds NOW."""
        self._update_latches()
        base = (0.20 * self._engaged.float() + 0.30 * self._balanced.float()
                + 0.20 * self._succ_ever.float()).clamp(max=0.70)
        return torch.where(self._success_now(), torch.ones_like(base), base)


register_env("simgen", lambda: EnvCfg(scene="utensil_balance", robot="null"))
