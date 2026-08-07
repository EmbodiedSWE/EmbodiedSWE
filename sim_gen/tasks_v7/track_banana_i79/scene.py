"""BananaLineScene — harvest CLAMPED bananas from a clothespin line by SQUEEZING the
clamps open over a staged crate (sim_gen task `track_banana_i79`, scene `banana_line`,
env `simgen.banana_line`).

Derived from pick_place/track_banana, but STRATEGICALLY different: the seed starts with
the banana ALREADY IN THE HAND and grades tracking of a prescribed free-space waypoint
path — pure transport of a rigidly held object. Here the judged bananas can NEVER be
grasped or carried at all: each hangs from a spring-loaded clothespin CLAMP on a gantry
line, its stem knob form-closed in the clamp's pinch slot (a roof blocks lifting out,
converged pad tips block forward escape, a rear stop blocks backward escape, and the
knob cannot pass the 14 mm closed gap — 12 N yanks in any direction provably fail).
The ONLY way to free a banana is to actuate the MECHANISM: squeeze the clamp's two
tail paddles together (a native parallel-jaw close), which scissors the pads apart
until the knob falls STRAIGHT DOWN by gravity. Dropping a banana anywhere but into the
crate BRUISES it — an irreversible latched fail — so the mobile crate must be staged
under each clamp BEFORE that clamp is opened, and afterwards the loaded crate must be
pushed along the floor into the marked depot zone. One of the three hanging bananas is
GREEN (unripe) and must be LEFT HANGING; which station it hangs at is randomized, so
the release decision is per-episode. No prescribed path, nothing held, nothing
tracked: the plan is stage -> actuate -> catch -> re-stage -> actuate -> catch ->
deliver, with an irreversible failure mode the seed does not have.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - gantry: heavy DYNAMIC compound (28 kg — must NOT be kinematic: on this stack a
    joint anchored to a teleported kinematic body0 stays world-fixed at the spawn
    pose). Base slab + two posts + beam + 3 bracket plates; per station: two axle
    pins, a rear stop, the bracket plate roof over the knob (5 mm clearance), and
    two knob-level CURBS (front + rear, z above the lever sweep) that cage the
    knob plate in y — a yank-jolted lever cannot let it slide off the pad ends.
  - 6 levers (2 per station): DYNAMIC scissor arms on spawn-authored VERTICAL
    revolute joints (axis Z, joint authored inside the lever spawner against the
    sibling gantry prim). A spring-damper plant in `post_step` re-centers each lever
    to the CLOSED pose; solve/smoke actuate through the `lever_drive` buffer
    (clamped z-torques — z-torque is immune to this stack's wrench frame drag).
  - 3 bananas: DYNAMIC compound (body + tip + 8x8 stem + a flat 8-gon 34 mm knob
    PLATE — flat faces defeat snatch camming; see the knob_w note below), two
    YELLOW + one GREEN; hung by the knob resting on the pad tops.
  - crate: DYNAMIC open box (interior 210 x 150 mm, walls 85 mm) — the only
    free-transportable object; it is staged under stations and pushed to the depot.
  - depot: KINEMATIC flat marker slab (jointless — safe to re-pose per reset).

Per-episode randomization (readback-verifiable): gantry xy jitter + yaw, the GREEN
banana's station (0/1/2) + shuffled yellows, per-banana free hang yaw + stem-slot
jitter, crate spawn xy + free yaw, depot xy.

Rubric (0..1, latched partial credit that never evaporates):
  0.25 * yellow A ever settled INSIDE the crate (crate-frame containment)
  0.25 * yellow B ever settled INSIDE the crate
  0.20 * delivery ever: crate inside the depot zone WITH both yellows aboard
  cap 0.70; exactly 1.0 iff success(): both yellows in the crate, the crate in the
  depot, the GREEN banana still hanging at its clamp, all clamps re-closed, nothing
  ever bruised (no banana ever touched the open floor), and everything settled.

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


# ----- custom compound spawners ---------------------------------------------------------------

_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable, yaw: float = 0.0) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw != 0.0:
        xf.AddRotateZOp().Set(math.degrees(yaw))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _slick_material(stage, root_path: str):
    """Author a low-friction physics material under `root_path` (polished clamp faces /
    smooth waxed stem knob). Captivity of the hanging banana is FORM closure (roof,
    stops, converged slot) — never friction — so the release mechanism is allowed to
    be slick by design; without it PhysX's ~0.5 default lets a knob corner tilt-jam
    on a pad's top edge as the pads slide apart (yaw-dependent, observed)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, f"{root_path}/slickMat")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(0.05)
    api.CreateDynamicFrictionAttr(0.04)
    api.CreateRestitutionAttr(0.0)
    return mat


def _bind_slick(stage, mat, *prim_paths: str) -> None:
    from pxr import UsdShade

    for p in prim_paths:
        prim = stage.GetPrimAtPath(p)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _root_xform(prim_path: str, translation, orientation):
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


def _rigid_dynamic(root, *, mass: float, lin_damp: float, ang_damp: float,
                   pos_iters: int = 16, vel_iters: int = 4) -> None:
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateSolverPositionIterationCountAttr(pos_iters)
    pxrb.CreateSolverVelocityIterationCountAttr(vel_iters)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)


def _spawn_gantry(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the gantry: heavy DYNAMIC compound. Local origin at the floor under the
    beam midline; brackets extend local +y; stations along local x at cfg.stations.
    Per station: bracket plate (its underside is the knob ROOF), two axle pins the
    levers hinge on, and a rear stop blocking backward knob/stem escape."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_dynamic(root, mass=float(cfg.gantry_mass), lin_damp=0.5, ang_damp=0.5)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    grey = (0.42, 0.42, 0.46)
    dark = (0.25, 0.25, 0.28)
    # base slab (kept clear of the crate corridor: front edge at slab_y_max)
    slab_cy = (c.slab_y_min + c.slab_y_max) / 2
    _add_box(stage, f"{prim_path}/slab",
             center=(0.0, slab_cy, c.slab_t / 2),
             size=(c.slab_len, c.slab_y_max - c.slab_y_min, c.slab_t), color=dark,
             collide=collide)
    for sgn, nm in ((1.0, "post_p"), (-1.0, "post_m")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(sgn * c.post_x, -0.030, (c.beam_z_top + c.slab_t) / 2),
                 size=(0.040, 0.040, c.beam_z_top - c.slab_t), color=grey, collide=collide)
    _add_box(stage, f"{prim_path}/beam",
             center=(0.0, -0.030, (c.beam_z_top + c.beam_z_bot) / 2),
             size=(2 * c.post_x + 0.040, 0.050, c.beam_z_top - c.beam_z_bot),
             color=grey, collide=collide)
    for i, sx in enumerate(c.stations):
        # bracket plate: the ROOF — underside at roof_z, 5 mm above the hanging knob
        # top; wide enough (plate_x) to cover the knob's worst lateral excursion when
        # a lever is jolted to its hard stop (knob edge reaches |x - sx| ~ 52 mm)
        _add_box(stage, f"{prim_path}/plate_{i}",
                 center=(sx, (c.plate_y0 + c.plate_y1) / 2, c.roof_z + c.plate_t / 2),
                 size=(c.plate_x, c.plate_y1 - c.plate_y0, c.plate_t), color=grey,
                 collide=collide)
        # knob-level CURBS: wide lips under the roof at KNOB height only (z from
        # curb_z_bot, entirely ABOVE the 0.292 lever-sweep ceiling, so they can never
        # touch the mechanism). They cage the knob plate in y: without them a yank
        # that transiently jolts ONE lever open ~7 deg parts the touching tip nubs
        # enough for the 8 mm stem, and the knob slides forward off the pad ends (or
        # diagonally around the rear stop) and falls — observed as the -x escape.
        for y0, y1, nm in ((c.curb_f_y0, c.curb_f_y1, "curb_f"),
                           (c.stop_y0, c.stop_y1, "curb_r")):
            _add_box(stage, f"{prim_path}/{nm}_{i}",
                     center=(sx, (y0 + y1) / 2, (c.roof_z + c.curb_z_bot) / 2),
                     size=(c.plate_x, y1 - y0, c.roof_z - c.curb_z_bot), color=dark,
                     collide=collide)
        # two axle pins descending from the plate through the lever hubs
        for s, nm in ((1.0, "l"), (-1.0, "r")):
            _add_box(stage, f"{prim_path}/pin_{i}{nm}",
                     center=(sx - s * c.pivot_dx, c.pivot_y, (c.roof_z + c.lever_z0) / 2),
                     size=(0.010, 0.010, c.roof_z - c.lever_z0), color=dark,
                     collide=collide)
        # rear stop: blocks backward escape of knob and stem out of the pinch slot
        _add_box(stage, f"{prim_path}/stop_{i}",
                 center=(sx, (c.stop_y0 + c.stop_y1) / 2, (c.roof_z + c.stop_z_bot) / 2),
                 size=(0.036, c.stop_y1 - c.stop_y0, c.roof_z - c.stop_z_bot),
                 color=dark, collide=collide)
    return root


def _spawn_lever(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one clamp lever: DYNAMIC scissor arm, root origin AT ITS PIVOT (the
    joint anchor), hand = +1 (left, pivot on -x side of the slot) or -1 (right,
    mirrored). Authors its own VERTICAL revolute joint against the sibling gantry.
    Toward the slot centre is +hand * x in lever frame."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_dynamic(root, mass=float(cfg.mass), lin_damp=0.05, ang_damp=float(cfg.ang_damp),
                   pos_iters=32, vel_iters=4)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    s = float(c.hand)
    red = (0.78, 0.16, 0.12)
    dark = (0.50, 0.10, 0.08)
    zt = c.thick
    _add_box(stage, f"{prim_path}/hub", center=(0.0, 0.0, 0.0),
             size=(0.024, 0.024, zt), color=dark, collide=collide)
    # spine from tail to pad along local y
    _add_box(stage, f"{prim_path}/spine", center=(s * 0.004, 0.008, 0.0),
             size=(0.012, 0.150, zt), color=red, collide=collide)
    # pad block: inner face at rel x = s*(pivot_dx - gap/2)
    pad_in = c.pivot_dx - c.gap_closed / 2
    _add_box(stage, f"{prim_path}/pad", center=(s * (pad_in - 0.012), c.pad_ry, 0.0),
             size=(0.024, c.pad_len, zt), color=red, collide=collide)
    # converged tip nub: forward stop of the pinch slot
    nub_in = c.pivot_dx - c.nub_gap / 2
    _add_box(stage, f"{prim_path}/nub", center=(s * (nub_in - 0.012), c.nub_ry, 0.0),
             size=(0.024, 0.010, zt), color=dark, collide=collide)
    # tail paddle: the squeeze handle (outer faces of the pair span 2*(pivot_dx+0.004))
    _add_box(stage, f"{prim_path}/tail", center=(-s * 0.002, -c.tail_ry, 0.0),
             size=(0.012, 0.050, zt), color=red, collide=collide)
    # polished clamp: every working face is slick so release is friction-independent
    mat = _slick_material(stage, prim_path)
    _bind_slick(stage, mat, f"{prim_path}/hub", f"{prim_path}/spine",
                f"{prim_path}/pad", f"{prim_path}/nub", f"{prim_path}/tail")
    # vertical revolute joint to the sibling gantry, axis Z, anchored at this pivot
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/pivot")
    j.CreateBody0Rel().SetTargets([f"{base}/Gantry"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(False)
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(float(c.px), float(c.py), float(c.pz)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    lim = float(c.limit_deg)
    j.CreateLowerLimitAttr(-3.0 if s > 0 else -lim)
    j.CreateUpperLimitAttr(lim if s > 0 else 3.0)
    return root


def _spawn_banana(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a banana: DYNAMIC compound — body + tip block + 8x8 stem + a flat 8-gon
    knob PLATE. Root origin at the BODY CENTRE; the knob top is at local z = knob_z_top."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_dynamic(root, mass=float(cfg.mass), lin_damp=0.08, ang_damp=0.30)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/body", center=(0.0, 0.0, 0.0),
             size=(c.body_l, c.body_w, c.body_h), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/tipa", center=(c.body_l / 2, 0.0, 0.004),
             size=(0.024, 0.026, 0.026), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/tipb", center=(-c.body_l / 2, 0.0, 0.004),
             size=(0.024, 0.026, 0.026), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/stem",
             center=(0.0, 0.0, (c.body_h / 2 + c.knob_z_bot) / 2),
             size=(c.stem_w, c.stem_w, c.knob_z_bot - c.body_h / 2), color=(0.35, 0.28, 0.10),
             collide=collide)
    kz = (c.knob_z_bot + c.knob_z_top) / 2
    kt = c.knob_z_top - c.knob_z_bot
    _add_box(stage, f"{prim_path}/knob_a", center=(0.0, 0.0, kz),
             size=(c.knob_w, c.knob_w, kt), color=(0.35, 0.28, 0.10), collide=collide)
    _add_box(stage, f"{prim_path}/knob_b", center=(0.0, 0.0, kz), yaw=math.pi / 4,
             size=(c.knob_w, c.knob_w, kt), color=(0.35, 0.28, 0.10), collide=collide)
    # smooth waxed stem/knob (the hanging interface only; the BODY keeps default
    # friction so it behaves normally on the crate floor)
    mat = _slick_material(stage, prim_path)
    _bind_slick(stage, mat, f"{prim_path}/stem", f"{prim_path}/knob_a",
                f"{prim_path}/knob_b")
    return root


def _spawn_crate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the crate: DYNAMIC open box, root origin at the floor centre of its
    footprint. Interior in_x * in_y, wall height wall_h above the floor plate."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_dynamic(root, mass=float(cfg.mass), lin_damp=0.10, ang_damp=0.50)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.floor_t / 2),
             size=(c.in_x + 2 * c.wall_t, c.in_y + 2 * c.wall_t, c.floor_t),
             color=c.color, collide=collide)
    for sgn, nm in ((1.0, "wall_xp"), (-1.0, "wall_xm")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(sgn * (c.in_x + c.wall_t) / 2, 0.0, (c.wall_h + c.floor_t) / 2),
                 size=(c.wall_t, c.in_y + 2 * c.wall_t, c.wall_h - c.floor_t + 0.001),
                 color=c.color, collide=collide)
    for sgn, nm in ((1.0, "wall_yp"), (-1.0, "wall_ym")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(0.0, sgn * (c.in_y + c.wall_t) / 2, (c.wall_h + c.floor_t) / 2),
                 size=(c.in_x, c.wall_t, c.wall_h - c.floor_t + 0.001),
                 color=c.color, collide=collide)
    return root


def _spawn_depot(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the depot marker: KINEMATIC paint zone — NO collider (a 3 mm proud step
    would wall off the pushed crate's 10 mm floor face; delivery is judged by xy)."""
    from pxr import Gf, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    c = cfg
    box = UsdGeom.Cube.Define(stage, f"{prim_path}/pad")
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0008))
    xf.AddScaleOp().Set(Gf.Vec3f(float(c.size_x), float(c.size_y), 0.0016))
    box.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.15, 0.60)])
    return root


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "gantry" not in _SPAWNER_CACHE:

        @configclass
        class GantrySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gantry)
            gantry_mass: float = 28.0
            stations: tuple = (-0.22, 0.0, 0.22)
            slab_len: float = 0.70
            slab_y_min: float = -0.11
            slab_y_max: float = 0.045
            slab_t: float = 0.020
            post_x: float = 0.30
            beam_z_bot: float = 0.28
            beam_z_top: float = 0.34
            plate_y0: float = -0.005
            plate_y1: float = 0.193
            plate_t: float = 0.012
            plate_x: float = 0.110
            curb_f_y0: float = 0.185  # front knob-curb y band (rear curb reuses the
            curb_f_y1: float = 0.193  # rear stop's band); z from curb_z_bot to roof_z
            curb_z_bot: float = 0.295  # 3 mm above the 0.292 lever tops: no contact gen
            roof_z: float = 0.307
            lever_z0: float = 0.272
            pivot_dx: float = 0.028
            pivot_y: float = 0.100
            stop_y0: float = 0.122    # pulled back 6 mm so the 34 mm knob plate
            stop_y1: float = 0.130    # does not author in contact with the stop
            stop_z_bot: float = 0.260
            contact_offset: float = 0.0012

        @configclass
        class LeverSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lever)
            hand: float = 1.0
            px: float = 0.0
            py: float = 0.100
            pz: float = 0.282
            mass: float = 0.60
            ang_damp: float = 2.0
            thick: float = 0.020
            gap_closed: float = 0.014
            nub_gap: float = 0.0
            pivot_dx: float = 0.028
            pad_ry: float = 0.053
            pad_len: float = 0.036
            nub_ry: float = 0.076
            tail_ry: float = 0.045
            limit_deg: float = 20.0
            contact_offset: float = 0.0012

        @configclass
        class BananaSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_banana)
            mass: float = 0.12
            color: tuple = (0.92, 0.80, 0.12)
            body_l: float = 0.105
            body_w: float = 0.034
            body_h: float = 0.036
            stem_w: float = 0.008
            knob_w: float = 0.034
            knob_z_bot: float = 0.075
            knob_z_top: float = 0.083
            contact_offset: float = 0.0012

        @configclass
        class CrateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_crate)
            mass: float = 0.60
            color: tuple = (0.15, 0.35, 0.75)
            in_x: float = 0.210
            in_y: float = 0.150
            wall_t: float = 0.012
            wall_h: float = 0.085
            floor_t: float = 0.010
            contact_offset: float = 0.0015

        @configclass
        class DepotSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_depot)
            size_x: float = 0.30
            size_y: float = 0.26
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(gantry=GantrySpawnerCfg, lever=LeverSpawnerCfg,
                              banana=BananaSpawnerCfg, crate=CrateSpawnerCfg,
                              depot=DepotSpawnerCfg)
    return _SPAWNER_CACHE


def _qz(yaw: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(yaw.shape[0], 4, device=yaw.device)
    q[:, 0] = torch.cos(yaw / 2)
    q[:, 3] = torch.sin(yaw / 2)
    return q


def _yaw_of(q: torch.Tensor) -> torch.Tensor:
    return torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                       1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BananaLineSceneCfg(BaseCfg):
    """Config for `BananaLineScene`. Honesty is asserted in __post_init__: the knob
    cannot pass the closed slot in any yaw but falls freely through the open slot in
    any yaw; the stem hangs free in the closed slot; the roof gap is smaller than the
    rise needed to lift the knob clear of the pads; the spring holds the declared
    yank torque at the single-lever escape angle with margin while the drive can
    still open past the release angle; the closed tail pair fits the Franka jaw; the
    crate rim passes under the hanging bananas; the depot admits the crate."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    depot_tol: float = tunable(0.055)     # crate centre within this of the depot centre
    hang_z_tol: float = tunable(0.020)    # green root z within this of the hang height
    hang_xy_tol: float = tunable(0.050)   # green root xy within this of its live hang point
    closed_deg: float = tunable(4.0)      # all levers within this of closed = "released"
    in_margin: float = tunable(0.015)     # crate-frame containment margin off the walls
    bruise_z: float = tunable(0.050)      # banana root below this outside the crate = bruised
    bruise_r: float = tunable(0.170)      # "outside the crate" = further than this from its axis
    settle_speed: float = tunable(0.05)   # max |lin vel| of judged bodies when judging (m/s)
    settle_omega: float = tunable(0.60)   # max |ang vel| of judged bodies when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    gantry_jitter: float = tunable(0.030)  # gantry footprint xy jitter (+/- m)
    gantry_yaw_deg: float = tunable(20.0)  # gantry yaw, uniform (+/- deg)
    ban_jitter_x: float = tunable(0.002)   # stem-in-slot x jitter (+/- m)
    ban_jitter_y: float = tunable(0.003)   # stem-in-slot y jitter (+/- m)
    crate_jitter: float = tunable(0.050)   # crate spawn xy jitter (+/- m)
    depot_jitter: float = tunable(0.050)   # depot centre xy jitter (+/- m)

    # --- tunable: plant (difficulty dials) ------------------------------------------------------
    spring_k: float = tunable(5.0)        # lever centering spring (N*m/rad)
    spring_d: float = tunable(0.12)       # lever damping (N*m*s/rad)
    spring_preload: float = tunable(0.5)  # constant CLOSING torque per lever (N*m) — the
    #   clothespin's spring is pre-tensioned at the closed pose, so an impulsive yank on
    #   the banana cannot ratchet the slick pads open (holds ~2.3x the 12 N yank torque)
    drive_max: float = tunable(2.6)       # |lever_drive| clamp per lever (N*m)
    yank_hold: float = tunable(12.0)      # lateral yank (N) the closed clamp must hold

    # --- info: layout (single Franka base at the origin) ----------------------------------------
    gantry_pos: tuple = info((0.45, 0.08))   # gantry origin (before jitter); brackets face +y
    crate_pos: tuple = info((0.16, -0.22))   # crate spawn (before jitter)
    depot_pos: tuple = info((0.04, -0.42))   # depot centre (before jitter)

    # --- info: gantry / clamp geometry (mirrors the spawner defaults) ---------------------------
    stations: tuple = info((-0.22, 0.0, 0.22))
    pivot_dx: float = info(0.028)     # pivot lateral offset from the station axis (m)
    pivot_y: float = info(0.100)      # pivot y in gantry frame (m)
    stem_y: float = info(0.153)       # hang point y in gantry frame (mid pad span)
    pad_arm: float = info(0.053)      # effective pad-face lever arm about the pivot (m)
    gap_closed: float = info(0.014)   # closed pad gap (m)
    nub_gap: float = info(0.0)        # tip nubs TOUCH at 0 deg: the preload rests the
    #   levers on the nub-on-nub stop at ~0 deg, so the closed pad gap never narrows
    #   onto the hanging stem (a diagonal stem needs 11.3 mm; the gap stays 14 mm)
    lever_z0: float = info(0.272)     # lever plate bottom (m)
    lever_z1: float = info(0.292)     # lever plate top = pad top the knob rests on (m)
    roof_z: float = info(0.307)       # bracket plate underside over the knob (m)
    curb_f_y0: float = info(0.185)    # front knob-curb inner face (gantry y); with the
    #   rear curb (over the rear stop) it cages the knob plate in y at knob height
    #   ONLY (z >= 0.295, above the lever sweep — the mechanism is never touched)
    limit_deg: float = info(20.0)     # lever opening hard stop (deg)
    open_deg: float = info(17.5)      # commanded opening angle for release (deg)
    tail_span: float = info(0.064)    # closed outer span of the tail paddle pair (m)
    tail_arm: float = info(0.045)     # tail paddle lever arm about the pivot (m)

    # --- info: banana / crate / depot ------------------------------------------------------------
    ban_mass: float = info(0.12)
    body_l: float = info(0.105)
    body_h: float = info(0.036)
    stem_w: float = info(0.008)
    knob_w: float = info(0.034)       # knob box side; 8-gon max extent = knob_w / cos(22.5 deg)
    #   The knob is a wide FLAT PLATE (34 x 34 x 8 mm 8-gon). Anti-yank by shape:
    #   (a) flat plate on flat pad tops — a downward snatch load has NO opening
    #       component (a chunky knob's bottom corner would cam the slick pads apart);
    #   (b) a SINGLE lever kicked to its 20 deg hard stop opens only 32.1 mm < 34 mm,
    #       so a one-sided jolt can never drop the plate (asserted in __post_init__);
    #   (c) the roof gap (7 mm) blocks tipping the plate edgewise into the slot;
    #   (d) knob-level curbs fore and aft of the pad span cage the plate in y — a
    #       jolted-open lever parts the tip nubs enough for the stem, but the plate
    #       can neither slide forward off the pad ends nor around the rear stop.
    #   Release drives BOTH levers to 17.5 deg: 43.2 mm > the worst-yaw 36.8 mm.
    knob_z_bot: float = info(0.075)   # knob bottom in banana frame (root = body centre)
    knob_z_top: float = info(0.083)
    yellow: tuple = info((0.92, 0.80, 0.12))
    green: tuple = info((0.25, 0.62, 0.18))
    crate_in_x: float = info(0.210)
    crate_in_y: float = info(0.150)
    crate_wall_h: float = info(0.085)
    crate_floor_t: float = info(0.010)
    crate_mass: float = info(0.60)
    depot_size: tuple = info((0.30, 0.26))
    contact_offset: float = info(0.0012)

    # rubric weights (0.25 + 0.25 + 0.20 = 0.70 = the non-success cap)
    w_yellow: float = info(0.25)
    w_deliver: float = info(0.20)

    # Derived (filled in __post_init__).
    hang_root_z: float = field(default=None, init=False)   # banana root z when hanging
    knob_max: float = field(default=None, init=False)      # knob worst-yaw extent
    open_gap: float = field(default=None, init=False)      # pad gap at open_deg

    def __post_init__(self) -> None:
        self.hang_root_z = self.lever_z1 - self.knob_z_bot + 0.0015
        self.knob_max = self.knob_w / math.cos(math.radians(22.5))
        self.open_gap = self.gap_closed + 2 * self.pad_arm * math.sin(math.radians(self.open_deg))
        # captivity: the knob passes the closed slot in NO yaw; the stem hangs free in ANY yaw
        assert self.knob_w > self.gap_closed + 0.008, "knob passes the closed slot"
        assert self.stem_w * math.sqrt(2) < self.gap_closed - 0.002, "stem binds in the slot"
        assert self.stem_w * math.sqrt(2) > self.nub_gap, "stem escapes through the tip nubs"
        # release: at open_deg the slot passes the knob at its WORST yaw with margin
        assert self.open_gap > self.knob_max + 0.008, "open slot does not pass the knob"
        # no upward escape: the roof gap is far less than the rise that clears the pads
        assert self.roof_z - (self.lever_z1 + (self.knob_z_top - self.knob_z_bot)) \
            < (self.lever_z1 - self.lever_z0) - 0.004, "roof admits lifting the knob clear"
        # the front curb clears the hanging knob (worst yaw + slot jitter) yet stops it
        # while still supported by the pads (pad span ends at pivot_y + 0.071)
        assert self.stem_y + self.ban_jitter_y + self.knob_max / 2 \
            < self.curb_f_y0 - 0.004, "front curb pinches the hanging knob"
        assert self.curb_f_y0 - self.knob_w / 2 < self.pivot_y + 0.071, \
            "curb lets the knob leave the pad span before stopping it"
        # no single-lever escape: even at the hard stop, one lever alone opens the slot
        # less than the plate's face-on width — a yank-jolted lever cannot drop it
        assert self.gap_closed + self.pad_arm * math.sin(math.radians(self.limit_deg)) \
            < self.knob_w - 0.001, "one jolted lever could pass the knob plate"
        # the spring (preload + rate) holds the declared yank at the single-lever escape
        # angle (1.4x margin): one lever alone must retreat the FULL (knob - gap) for the
        # knob to pass
        th_esc = math.asin((self.knob_w - self.gap_closed) / self.pad_arm)
        assert self.spring_preload + self.spring_k * th_esc \
            > 1.4 * self.yank_hold * self.pad_arm, "spring cannot hold the declared yank"
        # ...while the drive can still push past the release angle with margin
        assert self.drive_max > self.spring_preload \
            + 1.3 * self.spring_k * math.radians(self.open_deg), \
            "drive cannot reach the release angle"
        assert self.open_deg < self.limit_deg - 2.0, "release angle too close to the hard stop"
        # embodiment: the closed tail pair fits inside the 80 mm Franka jaw stroke
        assert self.tail_span < 0.078, "tail pair exceeds the jaw stroke"
        # staging: the crate rim passes under the hanging bananas; the body falls inside
        body_bot = self.hang_root_z - self.body_h / 2
        assert body_bot > self.crate_wall_h + 0.030, "crate cannot pass under the hang line"
        assert self.crate_in_x > self.body_l + 0.040, "crate interior too short for a banana"
        # the depot pad admits the whole crate footprint at the success tolerance
        assert self.depot_size[0] / 2 > self.depot_tol + 0.05, "depot too small"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("banana_line")
class BananaLineScene(BaseScene):
    cfg: BananaLineSceneCfg

    def __init__(self, cfg: BananaLineSceneCfg | None = None) -> None:
        super().__init__(cfg or BananaLineSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        rigid = sim_utils.RigidBodyPropertiesCfg()
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
            "gantry": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gantry",
                spawn=sp["gantry"](mass_props=sim_utils.MassPropertiesCfg(mass=28.0),
                                   rigid_props=rigid, stations=c.stations,
                                   contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.gantry_pos[0], c.gantry_pos[1], 0.0)),
            ),
        }
        for i, sx in enumerate(c.stations):
            for s, nm in ((1.0, "l"), (-1.0, "r")):
                out[f"lever_{i}{nm}"] = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/Lever_" + f"{i}{nm}",
                    spawn=sp["lever"](mass_props=sim_utils.MassPropertiesCfg(mass=0.60),
                                      rigid_props=rigid, hand=s,
                                      px=sx - s * c.pivot_dx, py=c.pivot_y, pz=0.282,
                                      gap_closed=c.gap_closed, nub_gap=c.nub_gap,
                                      pivot_dx=c.pivot_dx, pad_ry=c.pad_arm,
                                      tail_ry=c.tail_arm, limit_deg=c.limit_deg,
                                      contact_offset=c.contact_offset),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(c.gantry_pos[0] + sx - s * c.pivot_dx,
                             c.gantry_pos[1] + c.pivot_y, 0.282)),
                )
        for j, col in ((0, c.yellow), (1, c.yellow), (2, c.green)):
            out[f"ban_{j}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Banana_" + str(j),
                spawn=sp["banana"](mass_props=sim_utils.MassPropertiesCfg(mass=c.ban_mass),
                                   rigid_props=rigid, mass=c.ban_mass, color=col,
                                   body_l=c.body_l, body_h=c.body_h, stem_w=c.stem_w,
                                   knob_w=c.knob_w, knob_z_bot=c.knob_z_bot,
                                   knob_z_top=c.knob_z_top,
                                   contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.gantry_pos[0] + c.stations[j], c.gantry_pos[1] + c.stem_y,
                         c.hang_root_z)),
            )
        out["crate"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Crate",
            spawn=sp["crate"](mass_props=sim_utils.MassPropertiesCfg(mass=c.crate_mass),
                              rigid_props=rigid, mass=c.crate_mass, in_x=c.crate_in_x,
                              in_y=c.crate_in_y, wall_h=c.crate_wall_h,
                              floor_t=c.crate_floor_t),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.crate_pos[0], c.crate_pos[1], 0.001)),
        )
        out["depot"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Depot",
            spawn=sp["depot"](mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                              rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                  kinematic_enabled=True),
                              size_x=c.depot_size[0], size_y=c.depot_size[1]),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.depot_pos[0], c.depot_pos[1], 0.0)),
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.gantry: RigidObject = env.iscene["gantry"]
        self.levers: list[RigidObject] = [env.iscene[f"lever_{i}{nm}"]
                                          for i in range(3) for nm in ("l", "r")]
        self.bananas: list[RigidObject] = [env.iscene[f"ban_{j}"] for j in range(3)]
        self.crate: RigidObject = env.iscene["crate"]
        self.depot: RigidObject = env.iscene["depot"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # actuation buffer: raw z-torque per lever, clamped to +/- drive_max in post_step
        self.lever_drive = torch.zeros(n, 6, device=dev)
        # per-episode layout (env-local)
        self._gantry_xy = torch.tensor(self.cfg.gantry_pos, device=dev).repeat(n, 1)
        self._gantry_yaw = torch.zeros(n, device=dev)
        self._depot_xy = torch.tensor(self.cfg.depot_pos, device=dev).repeat(n, 1)
        self._ban_station = torch.tensor([0, 1, 2], device=dev).repeat(n, 1)  # (n, 3)
        # latches
        self._in_crate_ever = torch.zeros(n, 3, dtype=torch.bool, device=dev)
        self._delivered = torch.zeros(n, dtype=torch.bool, device=dev)
        self._bruised = torch.zeros(n, 3, dtype=torch.bool, device=dev)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the (dynamic) gantry with xy jitter + yaw, place all six
        levers CLOSED consistently with their joints, hang the three bananas with the
        GREEN one at a random station (yellows shuffled over the rest, free hang yaw +
        stem-slot jitter), spawn the crate (xy jitter + free yaw) and the depot marker
        (xy jitter); clear latches and drives."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        gxy = torch.tensor(c.gantry_pos, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.gantry_jitter
        gyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.gantry_yaw_deg)
        qg = _qz(gyaw)
        cy, sy = torch.cos(gyaw), torch.sin(gyaw)

        def gframe(lx, ly):
            """(m, 2) world xy of a gantry-local point."""
            if not torch.is_tensor(lx):
                lx = torch.full((m,), float(lx), device=dev)
            if not torch.is_tensor(ly):
                ly = torch.full((m,), float(ly), device=dev)
            return torch.stack([gxy[:, 0] + cy * lx - sy * ly,
                                gxy[:, 1] + sy * lx + cy * ly], dim=1)

        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = gxy
        st[:, 3:7] = qg
        st[:, 0:3] += origin
        self.gantry.write_root_state_to_sim(st, env_ids)
        self._gantry_xy[env_ids] = gxy
        self._gantry_yaw[env_ids] = gyaw

        for i, sx in enumerate(c.stations):
            for k, s in ((0, 1.0), (1, -1.0)):
                st = torch.zeros(m, 13, device=dev)
                st[:, 0:2] = gframe(sx - s * c.pivot_dx, c.pivot_y)
                st[:, 2] = 0.282
                st[:, 3:7] = qg
                st[:, 0:3] += origin
                self.levers[2 * i + k].write_root_state_to_sim(st, env_ids)

        # green banana (asset 2) at a random station; yellows shuffled over the rest
        perm = torch.stack([torch.randperm(3, device=dev) for _ in range(m)])  # (m, 3)
        self._ban_station[env_ids] = perm
        for j in range(3):
            sx = torch.tensor(c.stations, device=dev)[perm[:, j]]
            jx = (torch.rand(m, device=dev) * 2 - 1) * c.ban_jitter_x
            jy = (torch.rand(m, device=dev) * 2 - 1) * c.ban_jitter_y
            byaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = gframe(sx + jx, c.stem_y + jy)
            st[:, 2] = c.hang_root_z
            st[:, 3:7] = _qz(gyaw + byaw)
            st[:, 0:3] += origin
            self.bananas[j].write_root_state_to_sim(st, env_ids)

        cxy = torch.tensor(c.crate_pos, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.crate_jitter
        cyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = cxy
        st[:, 2] = 0.001
        st[:, 3:7] = _qz(cyaw)
        st[:, 0:3] += origin
        self.crate.write_root_state_to_sim(st, env_ids)

        dxy = torch.tensor(c.depot_pos, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.depot_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = dxy
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.depot.write_root_state_to_sim(st, env_ids)
        self._depot_xy[env_ids] = dxy

        self.lever_drive[env_ids] = 0.0
        self._in_crate_ever[env_ids] = False
        self._delivered[env_ids] = False
        self._bruised[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "gantry": self.gantry.data.root_state_w[env_ids].clone(),
            "crate": self.crate.data.root_state_w[env_ids].clone(),
            "depot": self.depot.data.root_state_w[env_ids].clone(),
            "gantry_xy": self._gantry_xy[env_ids].clone(),
            "gantry_yaw": self._gantry_yaw[env_ids].clone(),
            "depot_xy": self._depot_xy[env_ids].clone(),
            "ban_station": self._ban_station[env_ids].clone(),
            "lever_drive": self.lever_drive[env_ids].clone(),
            "in_crate_ever": self._in_crate_ever[env_ids].clone(),
            "delivered": self._delivered[env_ids].clone(),
            "bruised": self._bruised[env_ids].clone(),
        }
        for k, lv in enumerate(self.levers):
            out[f"lever_{k}"] = lv.data.root_state_w[env_ids].clone()
        for j, bn in enumerate(self.bananas):
            out[f"ban_{j}"] = bn.data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.gantry.write_root_state_to_sim(state["gantry"], env_ids)
        self.crate.write_root_state_to_sim(state["crate"], env_ids)
        self.depot.write_root_state_to_sim(state["depot"], env_ids)
        for k, lv in enumerate(self.levers):
            lv.write_root_state_to_sim(state[f"lever_{k}"], env_ids)
        for j, bn in enumerate(self.bananas):
            bn.write_root_state_to_sim(state[f"ban_{j}"], env_ids)
        self._gantry_xy[env_ids] = state["gantry_xy"]
        self._gantry_yaw[env_ids] = state["gantry_yaw"]
        self._depot_xy[env_ids] = state["depot_xy"]
        self._ban_station[env_ids] = state["ban_station"]
        self.lever_drive[env_ids] = state["lever_drive"]
        self._in_crate_ever[env_ids] = state["in_crate_ever"]
        self._delivered[env_ids] = state["delivered"]
        self._bruised[env_ids] = state["bruised"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A heavy gantry line stands near ({c.gantry_pos[0]:.2f}, {c.gantry_pos[1]:.2f}) "
            f"(xy and yaw change per episode). Three RED clothespin CLAMPS hang from its "
            f"bracket arms about {c.lever_z1:.2f} m up, one per station "
            f"({', '.join(f'{s:+.2f}' for s in c.stations)} m along the beam). Each clamp is a "
            f"pair of spring-loaded scissor levers: their PADS pinch a "
            f"{c.gap_closed * 1000:.0f} mm slot in front, and their TAIL paddles stick out "
            f"behind (outer span {c.tail_span * 1000:.0f} mm). A banana hangs from each clamp "
            f"by its stem: the stem knob rests ON the pads and cannot pass the closed slot, "
            f"a roof plate blocks lifting it out, and stops block sliding it out — pulling "
            f"on a banana in any direction fails. SQUEEZING a clamp's two tail paddles "
            f"together scissors its pads apart until the knob falls straight down. Two "
            f"bananas are YELLOW (ripe) and one is GREEN (unripe); which station the green "
            f"one hangs at changes per episode. A BLUE open crate "
            f"({c.crate_in_x * 100:.0f} x {c.crate_in_y * 100:.0f} cm inside, "
            f"{c.crate_wall_h * 100:.0f} cm walls) sits on the floor, and a MAGENTA depot "
            f"pad is marked on the floor away from the gantry (both move per episode).\n"
            f"Goal: harvest ONLY the two yellow bananas. Slide the crate along the floor "
            f"until it sits under a yellow banana's clamp, squeeze that clamp's tail "
            f"paddles so the banana drops INTO the crate, and release the clamp; repeat "
            f"for the other yellow banana; then push the loaded crate until it rests "
            f"centred on the depot pad (within {c.depot_tol * 100:.0f} cm) and leave "
            f"everything at rest with every clamp re-closed. The green banana must remain "
            f"HANGING on its clamp. A banana that falls anywhere but into the crate is "
            f"bruised — that cannot be undone and the task is failed."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the blue crate under each YELLOW banana's clothespin clamp in turn "
            "and squeeze that clamp's rear tail paddles together so the banana drops "
            "into the crate, then push the loaded crate onto the magenta depot pad. "
            "Leave the green banana hanging, re-close every clamp, and never let a "
            "banana fall on the floor — a dropped banana is bruised for good."
        )

    # ----- readings / rubric ---------------------------------------------------------------------
    def lever_angles(self) -> torch.Tensor:
        """(N, 6) signed lever angle relative to the gantry (0 = closed; opening is
        +hand). Uses the LIVE gantry yaw (the gantry is dynamic)."""
        gy = _yaw_of(self.gantry.data.root_quat_w)
        return torch.stack([_wrap(_yaw_of(lv.data.root_quat_w) - gy)
                            for lv in self.levers], dim=1)

    def hang_points(self) -> torch.Tensor:
        """(N, 3, 2) LIVE world xy (env-local) of each banana's assigned hang point."""
        c = self.cfg
        gp = self.gantry.data.root_pos_w - self.env_origins
        gy = _yaw_of(self.gantry.data.root_quat_w)
        cy, sy = torch.cos(gy), torch.sin(gy)
        sx = torch.tensor(c.stations, device=self.env.device)[self._ban_station]  # (N, 3)
        lx, ly = sx, torch.full_like(sx, c.stem_y)
        return torch.stack([gp[:, 0:1] + cy.unsqueeze(1) * lx - sy.unsqueeze(1) * ly,
                            gp[:, 1:2] + sy.unsqueeze(1) * lx + cy.unsqueeze(1) * ly], dim=2)

    def in_crate(self) -> torch.Tensor:
        """(N, 3) bool: banana root inside the crate volume, in the crate's body frame
        (below the rim — a rim-perched banana reads too high and is rejected)."""
        from isaaclab.utils.math import quat_rotate_inverse

        c = self.cfg
        qc = self.crate.data.root_quat_w
        pc = self.crate.data.root_pos_w
        out = []
        for bn in self.bananas:
            d = quat_rotate_inverse(qc, bn.data.root_pos_w - pc)
            out.append((d[:, 0].abs() <= c.crate_in_x / 2 - c.in_margin)
                       & (d[:, 1].abs() <= c.crate_in_y / 2 - c.in_margin)
                       & (d[:, 2] >= c.crate_floor_t) & (d[:, 2] <= c.crate_wall_h))
        return torch.stack(out, dim=1)

    def hanging(self) -> torch.Tensor:
        """(N, 3) bool: banana still hanging at its assigned clamp (root at hang
        height, xy at the live hang point)."""
        c = self.cfg
        hp = self.hang_points()
        out = []
        for j, bn in enumerate(self.bananas):
            p = bn.data.root_pos_w - self.env_origins
            out.append(((p[:, 0:2] - hp[:, j]).norm(dim=-1) <= c.hang_xy_tol)
                       & ((p[:, 2] - c.hang_root_z).abs() <= c.hang_z_tol))
        return torch.stack(out, dim=1)

    def crate_in_depot(self) -> torch.Tensor:
        p = self.crate.data.root_pos_w - self.env_origins
        return (p[:, 0:2] - self._depot_xy).norm(dim=-1) <= self.cfg.depot_tol

    def clamps_closed(self) -> torch.Tensor:
        return (self.lever_angles().abs() < math.radians(self.cfg.closed_deg)).all(dim=1)

    def bruised(self) -> torch.Tensor:
        """(N,) any banana ever touched down outside the crate (latched)."""
        return self._bruised.any(dim=1)

    def _still(self) -> torch.Tensor:
        c = self.cfg
        ok = ((self.crate.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
              & (self.crate.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega))
        for bn in self.bananas:
            ok &= ((bn.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                   & (bn.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega))
        return ok

    def _update_latches(self) -> None:
        c = self.cfg
        inc = self.in_crate()
        self._in_crate_ever |= inc
        # bruise: LOW and away from the crate axis (a banana inside the crate is low
        # but near the axis; a hanging banana is high)
        pc = self.crate.data.root_pos_w - self.env_origins
        for j, bn in enumerate(self.bananas):
            p = bn.data.root_pos_w - self.env_origins
            low = p[:, 2] < c.bruise_z
            away = (p[:, 0:2] - pc[:, 0:2]).norm(dim=-1) > c.bruise_r
            self._bruised[:, j] |= low & away
        self._delivered |= (self.crate_in_depot() & inc[:, 0] & inc[:, 1])

    # ----- step-coupled mechanics (every substep) -------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Clamp plant: per lever, spring back to CLOSED + damping + the clamped
        `lever_drive` buffer, applied as a pure z-torque (immune to the wrench frame
        drag — levers only ever yaw). Then the latches."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        ang = self.lever_angles()
        drive = self.lever_drive.clamp(-c.drive_max, c.drive_max)
        zero3 = torch.zeros(n, 1, 3, device=dev)
        all_ids = torch.arange(n, device=dev)
        for k, lv in enumerate(self.levers):
            w = lv.data.root_ang_vel_w[:, 2]
            # opening direction is +z for even (left) levers, -z for odd (right) ones;
            # the constant preload always presses toward CLOSED (nub-on-nub stop)
            open_sign = 1.0 if k % 2 == 0 else -1.0
            tau = (-c.spring_k * ang[:, k] - c.spring_d * w
                   - c.spring_preload * open_sign + drive[:, k])
            t = torch.zeros(n, 1, 3, device=dev)
            t[:, 0, 2] = tau
            lv.set_external_force_and_torque(zero3, t, env_ids=all_ids, is_global=True)
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: both YELLOW bananas settled inside the crate, the crate settled
        inside the depot zone, the GREEN banana still hanging at its clamp, every
        clamp re-closed, no banana ever bruised, everything at rest. Physical
        outcomes only."""
        self._update_latches()
        inc = self.in_crate()
        return (inc[:, 0] & inc[:, 1] & ~inc[:, 2]
                & self.hanging()[:, 2]
                & self.crate_in_depot()
                & self.clamps_closed()
                & ~self.bruised()
                & self._still())

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25 per yellow banana ever settled INSIDE the crate
        + 0.20 for the delivery config ever held (crate in the depot with both yellows
        aboard) — latched, ~0 for doing nothing, capped 0.70 — exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_yellow * self._in_crate_ever[:, 0].float()
                + c.w_yellow * self._in_crate_ever[:, 1].float()
                + c.w_deliver * self._delivered.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="banana_line", robot="null"))
