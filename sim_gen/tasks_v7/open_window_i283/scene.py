"""SashVentScene — prop a gravity-loaded sash window open on its one-way pawl, then
pass the red parcel through the propped gap onto the outside delivery tray
(sim_gen task `open_window_i283`).

Derived from rlbench/open_window, but STRATEGICALLY different: the seed rotates a
handle and swings a hinged casement panel open — the articulation act IS the goal and
the panel stays wherever it is swung. Here the window is a DOUBLE-HUNG VERTICAL SASH
that is gravity-loaded shut (a dead-man mechanism: release it anywhere un-propped and
it slides closed on its own), the frame carries a one-way GRAVITY PAWL (an orange tab
on a revolute joint, hard stop at horizontal, free to swing up), and opening the
window is only a MEANS: the goal is to deliver the RED parcel through the propped gap
onto the walled delivery tray on the far side of the wall, leaving the BLUE distractor
inside. A solver needs a different PLAN — exploit the passive ratchet (lift the sash
PAST the pawl so the rising lift-rail cams the tab aside, wait for the tab to
gravity-return underneath, then SET THE SASH DOWN ON the tab; merely holding the
window open achieves nothing because the hand is then not free for the parcel) and
then use the created aperture for a through-the-wall transit — and different code
STRUCTURE (prismatic dead-man + revolute pawl + aperture-transit gating, not
handle-turn + hinge-angle).

success(): the RED parcel at rest on the outside tray floor (frame-frame membership:
x in the tray band beyond the wall, |y| inside the fences, z at tray rest height —
the z band rejects a parcel on the fence, stacked on the distractor, or on the ground
outside), the parcel having actually TRANSITED the window aperture (`through_latch`:
it crossed the wall plane BELOW the fixed upper pane while the sash was open — a
parcel lobbed over the 0.8 m wall or carried around it never sets this), the sash
still OPEN (q >= q_open_min, only reachable propped: the pawl rest gives q_prop =
0.1265 while the tallest wedgeable object, a 6 cm cube, props only q = 0.0585), and
everything PERSISTENTLY still (stillness counter-latch).

score(), latched (credit never evaporates): 0.30 once the sash has ever been open AND
still for a 20-step streak (a teleported-open sash free-falls immediately — the
streak never fires un-propped); 0.25 once the parcel crosses the wall plane through
the aperture while open; 0.30 once the parcel rests in the tray (gated on the
through-latch and openness); 1.0 iff success(). Null policy ~0 (sash starts closed).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - frame (KINEMATIC, never teleported — both joints anchor to it): a 1.1 x 0.8 m
    wall standing on the ground with a 0.42 x 0.50 m window opening, a fixed
    light-blue upper pane (lower edge z = 0.375), a through-wall sill shelf whose
    outside half is the GREEN delivery tray (end + side fences), and the pawl hinge
    post on the right jamb.
  - sash (dynamic): glass-blue panel + protruding YELLOW lift rail (the grasp/push
    feature), riding a spawn-authored prismatic joint (axis Z, stops [0, 0.22]).
    Gravity returns it to the closed stop from any un-propped height.
  - pawl tab (dynamic): orange tab on a spawn-authored revolute joint (axis X,
    limits [-85 deg (swung up), 0 deg (horizontal hard stop)]); its CoM is outboard
    of the hinge so gravity parks it on the horizontal stop, where it carries the
    propped sash's weight through the joint limit.
  - parcel (RED 6 cm cube) and distractor (BLUE 6 cm cube), on the inside floor.
Friction materials are bound explicitly everywhere (the default-material trap);
masses/CoM/inertia are AUTHORED in the custom spawners (custom spawn funcs apply no
cfg schemas).

Per-episode randomization (readback-verified in smoke): the two cubes occupy 2 of 3
floor slots (shuffled) with xy jitter and free yaw.

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


def _spawn_frame(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC wall + window frame + sill/tray + pawl post. Body origin at the
    ground under the wall centerline (inside face x = 0, outside is -x)."""
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
    co = cfg.contact_offset
    wt = cfg.wall_t  # wall x extent: [-wt, 0]
    hy, hz = cfg.wall_half_y, cfg.wall_h
    oy, z0, z1 = cfg.open_half_y, cfg.open_z0, cfg.open_z1
    grey = (0.55, 0.53, 0.50)
    dark = (0.40, 0.38, 0.36)
    # --- wall slabs framing the opening ---
    _box(stage, f"{prim_path}/wall_left", (wt, hy - oy, hz), (-wt / 2, -(oy + hy) / 2, hz / 2),
         grey, co, material=mat)
    _box(stage, f"{prim_path}/wall_right", (wt, hy - oy, hz), (-wt / 2, (oy + hy) / 2, hz / 2),
         grey, co, material=mat)
    _box(stage, f"{prim_path}/wall_below", (wt, 2 * oy, cfg.sill_z0),
         (-wt / 2, 0.0, cfg.sill_z0 / 2), grey, co, material=mat)
    _box(stage, f"{prim_path}/wall_above", (wt, 2 * oy, hz - z1),
         (-wt / 2, 0.0, (hz + z1) / 2), grey, co, material=mat)
    # --- fixed upper pane (recessed toward the outside face) ---
    _box(stage, f"{prim_path}/upper_pane", (0.016, 2 * oy, z1 - cfg.pane_z0),
         (-wt + 0.008, 0.0, (z1 + cfg.pane_z0) / 2), (0.62, 0.78, 0.90), co, material=mat)
    # --- through-wall sill shelf (top = sill_top) ---
    sx0, sx1 = cfg.sill_x_out, cfg.sill_x_in  # x extent [-sx0, +sx1]
    _box(stage, f"{prim_path}/sill", (sx0 + sx1, 2 * cfg.sill_half_y, cfg.sill_top - cfg.sill_z0),
         ((sx1 - sx0) / 2, 0.0, (cfg.sill_top + cfg.sill_z0) / 2), dark, co, material=mat)
    # --- GREEN delivery tray pad on the outside half + fences (recessed into the
    #     sill: top only `pad_proud` above the sill top, so the slid parcel crosses a
    #     sub-millimetre step, not a wall) ---
    pt = cfg.tray_pad_t
    _box(stage, f"{prim_path}/tray_pad", (sx0 - wt, 2 * cfg.tray_half_y, pt),
         (-(sx0 + wt) / 2, 0.0, cfg.sill_top + cfg.pad_proud - pt / 2),
         (0.15, 0.62, 0.20), co, material=mat)
    _box(stage, f"{prim_path}/tray_end", (0.012, 2 * cfg.sill_half_y, cfg.fence_h),
         (-sx0 - 0.006, 0.0, cfg.sill_top + cfg.fence_h / 2), dark, co, material=mat)
    for s, nm in ((-1.0, "tray_side_n"), (1.0, "tray_side_p")):
        _box(stage, f"{prim_path}/{nm}", (sx0 - wt, cfg.sill_half_y - cfg.tray_half_y, cfg.fence_h),
             (-(sx0 + wt) / 2, s * (cfg.tray_half_y + cfg.sill_half_y) / 2,
              cfg.sill_top + cfg.fence_h / 2), dark, co, material=mat)
    # --- pawl hinge post on the right jamb, inside face ---
    _box(stage, f"{prim_path}/pawl_post", (0.05, 0.055, 0.08),
         (0.055, cfg.hinge_y + 0.0275, cfg.hinge_z), dark, co, material=mat)
    return root


def _spawn_sash(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The sash: glass-blue panel + protruding YELLOW lift rail, ONE dynamic rigid
    body on a spawn-authored prismatic joint (axis Z, stops [0, travel]) against the
    sibling Frame. The joint is the track (frame-sash collision filtered — nothing
    rubs); gravity is the return spring. The joint is authored IN THE SPAWNER so it
    exists before the physics parse. Mass/CoM/inertia AUTHORED."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.012, -0.008, -0.025))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.007, 0.023, 0.016))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(cfg.lin_damping))
    pxrb.CreateAngularDampingAttr(2.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    # panel: body origin at panel center
    _box(stage, f"{prim_path}/panel", (cfg.panel_t, cfg.panel_w, cfg.panel_h),
         (0.0, 0.0, 0.0), (0.35, 0.52, 0.70), co, material=mat)
    # lift rail: protrudes +x at the panel bottom, right edge stops at lip_y_hi
    lip_cy = (cfg.lip_y_lo + cfg.lip_y_hi) / 2 - 0.0  # panel frame y center
    _box(stage, f"{prim_path}/lip",
         (cfg.lip_depth, cfg.lip_y_hi - cfg.lip_y_lo, cfg.lip_t),
         (cfg.panel_t / 2 + cfg.lip_depth / 2, lip_cy, -(cfg.panel_h - cfg.lip_t) / 2),
         (0.95, 0.80, 0.10), co, material=mat)
    # Prismatic track to the sibling Frame (pair collision FILTERED by the joint).
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/track")
    j.CreateBody0Rel().SetTargets([f"{base}/Frame"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(False)
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in cfg.anchor_frame]))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(float(cfg.travel))
    return root


def _spawn_tab(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The gravity pawl: an orange tab, ONE dynamic body on a spawn-authored revolute
    joint (axis X) against the sibling Frame; limits [-swing_deg (up), 0 (horizontal
    hard stop)]. Body origin AT THE HINGE; the box child extends inboard (-y), so the
    CoM sits outboard of nothing — gravity torque presses the tab onto the 0-degree
    stop, where it carries the propped sash through the joint limit."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, -float(cfg.length) / 2, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(5.0e-5, 1.0e-5, 5.0e-5))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.30)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    _box(stage, f"{prim_path}/tab", (cfg.thick_x, cfg.length, cfg.thick_z),
         (0.0, -float(cfg.length) / 2, 0.0), (0.92, 0.35, 0.10),
         cfg.contact_offset, material=mat)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Frame"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(False)
    j.CreateAxisAttr("X")
    j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in cfg.anchor_frame]))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(cfg.swing_deg))
    j.CreateUpperLimitAttr(0.0)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazy: module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "frame" not in _SPAWNER_CACHE:

        @configclass
        class FrameSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_frame)
            wall_t: float = 0.03
            wall_half_y: float = 0.55
            wall_h: float = 0.80
            open_half_y: float = 0.21
            open_z0: float = 0.12
            open_z1: float = 0.62
            pane_z0: float = 0.375
            sill_z0: float = 0.08
            sill_top: float = 0.12
            sill_x_out: float = 0.16
            sill_x_in: float = 0.14
            sill_half_y: float = 0.24
            tray_half_y: float = 0.20
            tray_pad_t: float = 0.0025
            pad_proud: float = 0.0005
            fence_h: float = 0.07
            hinge_y: float = 0.155
            hinge_z: float = 0.24
            mu_static: float = 0.5
            mu_dynamic: float = 0.4
            contact_offset: float = 0.002

        @configclass
        class SashSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_sash)
            panel_t: float = 0.028
            panel_w: float = 0.40
            panel_h: float = 0.26
            lip_depth: float = 0.05
            lip_t: float = 0.03
            lip_y_lo: float = -0.20
            lip_y_hi: float = 0.10
            travel: float = 0.22
            mass: float = 1.2
            lin_damping: float = 1.0
            anchor_frame: tuple = (0.014, 0.0, 0.2515)
            mu_static: float = 0.5
            mu_dynamic: float = 0.4
            contact_offset: float = 0.002

        @configclass
        class TabSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tab)
            length: float = 0.08
            thick_x: float = 0.03
            thick_z: float = 0.016
            swing_deg: float = 85.0
            mass: float = 0.05
            anchor_frame: tuple = (0.055, 0.155, 0.24)
            mu_static: float = 0.5
            mu_dynamic: float = 0.4
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(frame=FrameSpawnerCfg, sash=SashSpawnerCfg, tab=TabSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SashVentSceneCfg(BaseCfg):
    """Config for `SashVentScene`. `__post_init__` asserts the strategic honesty
    invariants with pre-computed force-balance/sweep geometry (the dead-man pattern):
    the pawl can gravity-return UNDER the held sash; the propped gap passes the
    parcel with margin while the closed sash seals it out; the openness threshold
    rejects every wedgeable substitute prop; the tray z band rejects
    fence/stacked/ground rests; every grasp fits a parallel jaw."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    q_open_min: float = tunable(0.11)  # sash joint pos counted as OPEN (m); pawl rest
    # gives 0.1265, a 6 cm cube wedged under the sash gives only 0.0585
    tray_x_lo: float = tunable(-0.155)  # parcel center x band on the tray ...
    tray_x_hi: float = tunable(-0.055)  # ... (frame frame; wall outside face at -0.03)
    tray_y_tol: float = tunable(0.19)  # parcel center |y| inside the side fences
    tray_z_tol: float = tunable(0.020)  # |parcel z - tray rest z| (rejects fence/stack)
    through_x0: float = tunable(-0.045)  # transit slab: parcel center x inside ...
    through_x1: float = tunable(0.015)  # ... (the wall slab itself, x in [-0.03, 0])
    through_z_max: float = tunable(0.35)  # ... and BELOW the fixed pane (no over-wall)
    settle_lin: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(1.0)  # max parcel |ang vel| when judging (rad/s)
    tab_settle_ang: float = tunable(0.6)  # max tab |ang vel| when judging (rad/s)
    open_streak_min: int = tunable(20)  # open+still steps before the open latch fires
    settle_steps_min: int = tunable(30)  # stillness must PERSIST this many steps

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    slot_x: float = tunable(0.45)  # cube staging row x (inside floor)
    slot_ys: tuple = tunable((-0.25, 0.0, 0.25))  # 3 slots; 2 used, shuffled
    slot_jitter: float = tunable(0.03)  # uniform +/- xy jitter per cube at reset (m)
    yaw_deg: float = tunable(180.0)  # uniform +/- yaw per cube at reset

    # --- info: structure (frame body at the env origin, NEVER teleported) --------------------
    wall_t: float = info(0.03)  # wall x extent [-0.03, 0]; inside is +x
    wall_half_y: float = info(0.55)
    wall_h: float = info(0.80)
    open_half_y: float = info(0.21)  # window opening y half-width
    open_z0: float = info(0.12)  # opening bottom = sill top
    open_z1: float = info(0.62)  # opening top
    pane_z0: float = info(0.375)  # fixed upper pane lower edge
    sill_top: float = info(0.12)
    sill_x_out: float = info(0.16)  # sill reaches x = -0.16 outside ...
    sill_x_in: float = info(0.14)  # ... and x = +0.14 inside
    sill_half_y: float = info(0.24)
    tray_half_y: float = info(0.20)  # inner faces of the tray side fences
    tray_pad_t: float = info(0.0025)
    pad_proud: float = info(0.0005)  # pad top above the sill top (sub-mm step)
    fence_h: float = info(0.07)
    seal_gap: float = info(0.0015)  # closed sash bottom sits this above the sill
    panel_t: float = info(0.028)
    panel_w: float = info(0.40)
    panel_h: float = info(0.26)
    lip_depth: float = info(0.05)
    lip_t: float = info(0.03)
    lip_y_lo: float = info(-0.20)
    lip_y_hi: float = info(0.10)  # lift-rail right edge (clears the pawl's up-swing)
    travel: float = info(0.22)  # prismatic hard stops [0, travel]
    sash_mass: float = info(1.2)
    hinge_y: float = info(0.155)
    hinge_z: float = info(0.24)
    tab_len: float = info(0.08)
    tab_t: float = info(0.016)
    tab_thick_x: float = info(0.03)
    tab_swing_deg: float = info(85.0)
    tab_mass: float = info(0.05)
    cube: float = info(0.06)  # parcel/distractor edge
    cube_mass: float = info(0.15)
    lift_q_hold: float = info(0.21)  # the solve's hold height (clearance asserted)
    mu_static: float = info(0.5)
    mu_dynamic: float = info(0.4)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    sash_center0: tuple = field(default=None, init=False)  # sash body center, closed
    q_prop: float = field(default=None, init=False)  # sash joint pos resting on the pawl
    tray_rest_z: float = field(default=None, init=False)  # parcel center on the tray pad
    closed_bottom: float = field(default=None, init=False)  # sash bottom edge, closed

    def __post_init__(self) -> None:
        self.closed_bottom = self.sill_top + self.seal_gap  # 0.1215
        self.sash_center0 = (self.panel_t / 2, 0.0, self.closed_bottom + self.panel_h / 2)
        tab_top = self.hinge_z + self.tab_t / 2  # 0.248
        self.q_prop = tab_top - self.closed_bottom  # 0.1265
        self.tray_rest_z = self.sill_top + self.pad_proud + self.cube / 2  # 0.1505

        # -- dead-man honesty: the openness threshold is only reachable propped --
        assert self.q_open_min <= self.q_prop - 0.012, "open band must include the pawl rest"
        wedge_q = (self.sill_top + self.cube) - self.closed_bottom  # cube under the sash
        assert self.q_open_min >= wedge_q + 0.03, \
            f"a wedged cube (q={wedge_q:.4f}) must fall far below q_open_min"
        # -- pawl return clearance UNDER the held sash: the tab's return sweep within
        #    the lip's y-extent stays below the held lip bottom (single-arm feasibility)
        r_eff = self.tab_len + self.tab_t / 2
        dy = self.hinge_y - self.lip_y_hi
        assert r_eff > dy + 0.01, "tab must overlap the lip in y (the prop contact)"
        sweep_top = self.hinge_z + math.sqrt(r_eff**2 - dy**2)
        lip_bottom_held = self.closed_bottom + self.lift_q_hold
        assert lip_bottom_held >= sweep_top + 3 * self.contact_offset + 0.004, \
            f"held lip {lip_bottom_held:.4f} must clear the tab return sweep {sweep_top:.4f}"
        assert self.lift_q_hold <= self.travel - 0.008, "hold height must be inside the stops"
        assert self.closed_bottom + self.travel + self.panel_h <= self.open_z1 + 0.005, \
            "fully-raised sash must stay inside the opening"
        # -- prop contact overlap and first-contact clearance --
        assert self.lip_y_hi - (self.hinge_y - self.tab_len) >= 0.02, \
            "lip-tab overlap must be a real contact patch"
        assert self.hinge_z - self.tab_t / 2 >= self.closed_bottom + self.lip_t + 0.04, \
            "tab must sit well above the closed lift rail"
        # -- the aperture: propped gap passes the parcel, closed sash seals it out --
        gap_min = self.closed_bottom + self.q_open_min - self.sill_top
        assert gap_min >= self.cube + 0.045, "open band must pass the parcel with margin"
        assert self.closed_bottom <= self.sill_top + 0.01, "closed sash must seal the gap"
        assert 2 * self.open_half_y >= self.cube + 0.10, "opening width must pass the parcel"
        assert self.through_z_max <= self.pane_z0 - 0.02, \
            "the through band must sit below the fixed pane"
        # -- tray membership honesty --
        span_lo = -self.sill_x_out + 0.006 + self.cube / 2  # against the end fence
        span_hi = -self.wall_t - self.cube / 2  # against the wall outside face
        assert self.tray_x_lo <= span_lo - 0.005 and self.tray_x_hi >= span_hi + 0.004, \
            "tray x band must cover the physical rest range"
        assert self.tray_x_hi <= self.through_x0 - 0.008, \
            "tray band lies beyond the transit slab (a teleported-to-tray parcel never transits)"
        assert self.through_x0 <= -self.wall_t - 0.01 and self.through_x1 >= 0.01, \
            "transit slab must bracket the wall so a sliding parcel cannot skip it"
        assert self.sill_x_out - self.wall_t >= self.cube + 0.05, "tray must fit the parcel"
        assert self.tray_half_y - self.cube / 2 >= self.tray_y_tol - 0.02, \
            "y tolerance must be honest vs the fences"
        for wrong in (self.tray_rest_z + self.cube,  # stacked on the distractor
                      self.sill_top + self.fence_h + self.cube / 2,  # perched on a fence
                      self.cube / 2):  # on the ground outside
            assert abs(wrong - self.tray_rest_z) > self.tray_z_tol + 0.005, \
                f"tray z band must reject resting height {wrong:+.4f}"
        # -- jaw fits (parallel jaw ~80 mm) --
        assert self.cube <= 0.075, "parcel must fit the jaw"
        assert self.lip_t <= 0.075, "lift rail must fit the jaw"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("sash_vent")
class SashVentScene(BaseScene):
    cfg: SashVentSceneCfg

    def __init__(self, cfg: SashVentSceneCfg | None = None) -> None:
        super().__init__(cfg or SashVentSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()

        out: dict[str, Any] = {
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
            "frame": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Frame",
                spawn=spawners["frame"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=60.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    wall_t=c.wall_t, wall_half_y=c.wall_half_y, wall_h=c.wall_h,
                    open_half_y=c.open_half_y, open_z0=c.open_z0, open_z1=c.open_z1,
                    pane_z0=c.pane_z0, sill_z0=0.08, sill_top=c.sill_top,
                    sill_x_out=c.sill_x_out, sill_x_in=c.sill_x_in,
                    sill_half_y=c.sill_half_y, tray_half_y=c.tray_half_y,
                    tray_pad_t=c.tray_pad_t, pad_proud=c.pad_proud, fence_h=c.fence_h,
                    hinge_y=c.hinge_y, hinge_z=c.hinge_z,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "sash": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Sash",
                spawn=spawners["sash"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.sash_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    panel_t=c.panel_t, panel_w=c.panel_w, panel_h=c.panel_h,
                    lip_depth=c.lip_depth, lip_t=c.lip_t,
                    lip_y_lo=c.lip_y_lo, lip_y_hi=c.lip_y_hi,
                    travel=c.travel, mass=c.sash_mass, lin_damping=1.0,
                    anchor_frame=c.sash_center0,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.sash_center0),
            ),
            "tab": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tab",
                spawn=spawners["tab"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.tab_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    length=c.tab_len, thick_x=c.tab_thick_x, thick_z=c.tab_t,
                    swing_deg=c.tab_swing_deg, mass=c.tab_mass,
                    anchor_frame=(0.055, c.hinge_y, c.hinge_z),
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.055, c.hinge_y, c.hinge_z)),
            ),
        }
        for nm, color, y0 in (("parcel", (0.85, 0.13, 0.13), c.slot_ys[0]),
                              ("distractor", (0.13, 0.28, 0.88), c.slot_ys[1])):
            out[nm] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + nm.capitalize(),
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube, c.cube, c.cube),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.10),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_static, dynamic_friction=c.mu_dynamic,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_x, y0, c.cube / 2 + 0.003)),
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
        self.frame: RigidObject = env.iscene["frame"]
        self.sash: RigidObject = env.iscene["sash"]
        self.tab: RigidObject = env.iscene["tab"]
        self.parcel: RigidObject = env.iscene["parcel"]
        self.distractor: RigidObject = env.iscene["distractor"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.open_latch = torch.zeros(n, device=dev)  # sash ever open+still (streak)
        self.through_latch = torch.zeros(n, device=dev)  # parcel ever crossed, open
        self.tray_latch = torch.zeros(n, device=dev)  # parcel ever at rest in the tray
        self.open_streak = torch.zeros(n, device=dev)
        self.tray_streak = torch.zeros(n, device=dev)
        self.still_count = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sash written closed (2 mm above the stop — it settles onto
        it), tab horizontal at its hinge, cubes on 2 of 3 shuffled floor slots with
        xy jitter and free yaw; latches zeroed. The frame is NEVER teleported."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = origin + torch.tensor(
            [c.sash_center0[0], c.sash_center0[1], c.sash_center0[2] + 0.002], device=dev)
        st[:, 3] = 1.0
        self.sash.write_root_state_to_sim(st, env_ids)

        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = origin + torch.tensor([0.055, c.hinge_y, c.hinge_z], device=dev)
        st[:, 3] = 1.0
        self.tab.write_root_state_to_sim(st, env_ids)

        # cubes: 2 of the 3 slots, shuffled (rand-argsort — the first-randint trap)
        perm = torch.rand(m, len(c.slot_ys), device=dev).argsort(dim=1)
        ys = torch.tensor(c.slot_ys, device=dev)
        yaw_amp = math.radians(c.yaw_deg)
        for i, body in enumerate((self.parcel, self.distractor)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.slot_x
            st[:, 1] = ys[perm[:, i]]
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            st[:, 2] = c.cube / 2 + 0.003
            st[:, 0:3] += origin
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            body.write_root_state_to_sim(st, env_ids)

        for t in (self.open_latch, self.through_latch, self.tray_latch,
                  self.open_streak, self.tray_streak, self.still_count):
            t[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "sash": self.sash.data.root_state_w[env_ids].clone(),
            "tab": self.tab.data.root_state_w[env_ids].clone(),
            "parcel": self.parcel.data.root_state_w[env_ids].clone(),
            "distractor": self.distractor.data.root_state_w[env_ids].clone(),
            "latches": torch.stack([
                self.open_latch[env_ids], self.through_latch[env_ids],
                self.tray_latch[env_ids], self.open_streak[env_ids],
                self.tray_streak[env_ids], self.still_count[env_ids]], dim=-1).clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.sash.write_root_state_to_sim(state["sash"], env_ids)
        self.tab.write_root_state_to_sim(state["tab"], env_ids)
        self.parcel.write_root_state_to_sim(state["parcel"], env_ids)
        self.distractor.write_root_state_to_sim(state["distractor"], env_ids)
        lat = state["latches"]
        (self.open_latch[env_ids], self.through_latch[env_ids], self.tray_latch[env_ids],
         self.open_streak[env_ids], self.tray_streak[env_ids],
         self.still_count[env_ids]) = (lat[:, 0], lat[:, 1], lat[:, 2],
                                       lat[:, 3], lat[:, 4], lat[:, 5])

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A {2 * c.wall_half_y * 100:.0f} cm wide, {c.wall_h * 100:.0f} cm tall wall "
            f"stands on the floor; you are on its near side. It has a window: the upper "
            f"half is a FIXED light-blue pane, the lower half is a sliding glass SASH "
            f"({c.panel_w * 100:.0f} cm wide) with a protruding YELLOW lift rail along its "
            f"bottom edge. The sash slides only vertically and is gravity-loaded: released "
            f"un-propped at any height it slides fully closed on its own. On the right "
            f"jamb, just inside the opening, an ORANGE pawl tab sticks out horizontally at "
            f"{c.hinge_z * 100:.0f} cm height. The tab is a one-way ratchet: it swings UP "
            f"freely (the rising lift rail pushes it aside on the way past) and falls back "
            f"to horizontal on its own, but it cannot swing below horizontal — a sash "
            f"lowered onto it from above rests on the tab, holding the window open "
            f"(~{c.q_prop * 100:.1f} cm of travel, a {(c.closed_bottom + c.q_prop - c.sill_top) * 100:.1f} cm "
            f"gap above the sill). The window sill runs THROUGH the wall; its far half is "
            f"a GREEN delivery tray enclosed by low fences.\n"
            f"On the floor on your side lie two {c.cube * 100:.0f} cm cubes: one RED (the "
            f"parcel) and one BLUE (a distractor). Their spots change every episode.\n"
            f"Goal: deliver the RED parcel onto the green tray on the far side of the "
            f"wall, passing it THROUGH the window gap — and leave the window propped open "
            f"on the pawl. Required order (physically forced): first lift the sash past "
            f"the pawl, let the tab fall back underneath, and set the sash down on it; "
            f"only then can the parcel be slid across the sill and through the gap. The "
            f"closed sash seals the gap; anything wedged under the sash instead of the "
            f"pawl leaves it too low to count as open. The blue cube must NOT go to the "
            f"tray (it earns nothing there). Success is judged with everything at rest: "
            f"red parcel on the tray floor, window still propped open."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lift the sash window past the orange pawl, let the pawl drop back, and set "
            "the sash down on it so the window stays propped open. Then push the red "
            "cube through the open gap onto the green tray on the far side of the wall. "
            "The blue cube stays inside; a closed or un-propped window fails."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def sash_q(self) -> torch.Tensor:
        """(N,) sash joint position: its height above the closed pose (the frame is
        never moved, so the frame frame is a world translation)."""
        c = self.cfg
        return (self.sash.data.root_pos_w - self.env_origins)[:, 2] - c.sash_center0[2]

    def tab_angle(self) -> torch.Tensor:
        """(N,) tab hinge angle in rad (0 = horizontal stop, negative = swung up).
        The tab only ever rotates about x, so the quat is (cos t/2, sin t/2, 0, 0)."""
        q = self.tab.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 1], q[:, 0])

    def parcel_local(self) -> torch.Tensor:
        """(N, 3) parcel center in the frame frame."""
        return self.parcel.data.root_pos_w - self.env_origins

    def sash_open(self) -> torch.Tensor:
        """(N,) bool: sash high enough to count as open (only reachable propped)."""
        return self.sash_q() >= self.cfg.q_open_min

    def parcel_through_now(self) -> torch.Tensor:
        """(N,) bool: parcel center inside the WALL SLAB, in the window aperture band
        (below the fixed pane, inside the jambs) — the transit evidence. An
        over-the-wall arc crosses the slab only ABOVE the pane band; an
        around-the-wall carry crosses it only OUTSIDE the jambs; neither fires."""
        c = self.cfg
        loc = self.parcel_local()
        return ((loc[:, 0] > c.through_x0) & (loc[:, 0] < c.through_x1)
                & (loc[:, 1].abs() < c.open_half_y)
                & (loc[:, 2] < c.through_z_max) & (loc[:, 2] > 0.05))

    def parcel_in_tray(self) -> torch.Tensor:
        """(N,) bool: parcel center inside the tray membership window (z band rejects
        fence-perch, stacked and ground rests)."""
        c = self.cfg
        loc = self.parcel_local()
        return ((loc[:, 0] >= c.tray_x_lo) & (loc[:, 0] <= c.tray_x_hi)
                & (loc[:, 1].abs() <= c.tray_y_tol)
                & ((loc[:, 2] - c.tray_rest_z).abs() <= c.tray_z_tol))

    def _sash_still(self) -> torch.Tensor:
        return self.sash.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin

    def _parcel_still(self) -> torch.Tensor:
        c = self.cfg
        return ((self.parcel.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.parcel.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang))

    def _still_now(self) -> torch.Tensor:
        """(N,) bool: sash, parcel AND tab quiet — INSTANTANEOUS (never judge on this
        alone; the counter-latch below is what `settled()` reads)."""
        tab_ok = self.tab.data.root_ang_vel_w.norm(dim=-1) < self.cfg.tab_settle_ang
        return self._sash_still() & self._parcel_still() & tab_ok

    def settled(self) -> torch.Tensor:
        """(N,) bool: stillness has PERSISTED `settle_steps_min` consecutive steps."""
        return self.still_count >= self.cfg.settle_steps_min

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Stillness counter + progress latches: sash open AND still for a streak (a
        teleported-open sash free-falls — its velocity breaks the streak within ~2
        steps); parcel through the aperture while open (instantaneous — physically
        only possible through the gap); parcel at rest in the tray while open, gated
        on the through-latch (a parcel that never transited earns no tray credit)."""
        c = self.cfg
        self.still_count = (self.still_count + 1.0) * self._still_now().float()
        open_now = self.sash_open()
        self.open_streak = (self.open_streak + 1.0) * (open_now & self._sash_still()).float()
        self.open_latch = torch.maximum(
            self.open_latch, (self.open_streak >= c.open_streak_min).float())
        self.through_latch = torch.maximum(
            self.through_latch, (self.parcel_through_now() & open_now).float())
        tray_now = (self.parcel_in_tray() & open_now & self._parcel_still()
                    & self._sash_still() & (self.through_latch > 0.5))
        self.tray_streak = (self.tray_streak + 1.0) * tray_now.float()
        self.tray_latch = torch.maximum(
            self.tray_latch, (self.tray_streak >= c.open_streak_min).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: red parcel at rest in the tray, having actually transited the
        aperture, with the sash still propped open and everything persistently
        still."""
        return (self.parcel_in_tray() & (self.through_latch > 0.5)
                & self.sash_open() & self.settled())

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.30 sash ever open+still (streak-latched) + 0.25
        parcel ever through the aperture while open + 0.30 parcel ever at rest in
        the tray (gated on the through-latch) — all latched, credit never evaporates;
        capped 0.85; 1.0 iff success(). Null policy ~0 (the sash starts closed)."""
        base = (0.30 * self.open_latch + 0.25 * self.through_latch
                + 0.30 * self.tray_latch).clamp(0.0, 0.85)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="sash_vent", robot="null", env_spacing=3.0))
