"""MoatBridgeScene — bridge the moat with the channel plank, then roll the oversized
ball across into the green dock (sim_gen task `approach_grasp_screwdriver_i192`).

Derived from pick_place/approach_grasp_screwdriver, but STRATEGICALLY different: the
seed is a prehensile pick-and-place — approach a screwdriver lying in tabletop
clutter, close the parallel jaw on it, lift it and carry it toward a basket; the
whole plan is one grasp affordance plus free-space transport of a rigidly held
object over open, always-traversable space. Here the judged object CANNOT be
grasped (a 100 mm ball vs the 80 mm Franka jaw) and — more importantly — the space
between its start and its goal is NOT traversable at floor level: a moat (a sheer
120 mm-deep gap between two raised platforms) separates them, and a ball that goes
in can never come out (the moat is deeper than the ball, its walls are vertical,
and the ball affords no grasp). The solver must first BUILD INFRASTRUCTURE: lay the
blue channel plank across the moat as a bridge (a rested two-support construct found
by contact, spanning both rims), and only then move the payload — non-prehensile
rolling up the plank's end ramp, along its guard-railed channel, over the moat, and
into the green three-walled dock on the far platform. The plan is
build-then-traverse with a physically enforced order (crossing before bridging =
irreversible loss in the moat), not approach-grasp-carry; the plank is a TOOL whose
placement is load-bearing (it must span both rims AND be aligned with the dock,
whose lateral position changes every episode, as does the moat width).

success() (all live, judged on physical poses):
  - the ball's centre is inside the dock interior (well past the mouth plane and
    between the walls),
  - the ball rests AT FAR-PLATFORM HEIGHT (centre z within `dock_z_tol` of
    top + ball_r — kills fly-through/hover and under-the-dock-in-the-moat states),
  - the ball is settled (lin + ang velocity thresholds).
score() = latched stage credit anchored in the demonstrated solution:
  0.10 * best ball x-progress toward the dock mouth (gated ON-platform/bridge height
  so moat wandering earns nothing) + 0.30 * BRIDGED (the plank seen resting level at
  rim height with its deck overlapping BOTH rims, near-still) + 0.30 * CROSSED (the
  ball seen past the far rim at platform height) + 0.15 * DOCKED (ball inside the
  dock, slow), capped at 0.85; exactly 1.0 iff success(). Doing nothing scores ~0.

Assets are fully procedural (no external files):
  - near/far platforms: KINEMATIC raised slabs (520 x 800 x 120 mm) with low guard
    rails on every edge EXCEPT the moat-facing one; the far platform is re-posed per
    episode so the moat width varies;
  - plank: DYNAMIC blue channel plank — 300 x 160 x 12 mm deck, two 8 mm-thick x
    20 mm-tall side rails (the graspable feature: a parallel jaw pinches a rail from
    above), and a short chamfer climb ramp at each end (the ball rolls up onto the
    deck);
  - ball: DYNAMIC orange sphere, r = 50 mm (STRICTLY wider than the 80 mm jaw);
  - dock: KINEMATIC green three-walled corral (interior 145 x 150 mm, walls 45 mm)
    open toward the moat, with a bright floor mark (visual only).

Per-episode randomization (verified by readback in smoke): moat width (the far
platform and dock shift together), the dock's lateral offset along the far rim, the
ball spawn xy on the near platform, and the plank spawn xy + free yaw (with a
ball/plank keep-out and a deterministic fallback). Heavy imports (isaaclab, pxr) are
deferred so importing this module — and registering the scene — stays app-free.
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

_RAMP_TH = 0.003  # climb-ramp plate thickness (m)


def _ramp_len(deck_t: float, ang_deg: float) -> float:
    """Ramp plate length such that the plate's LOWER foot corner lands 0.5 mm ABOVE
    the plank's underside plane — the plank always rests on its deck (broad, stable
    face contact), never on the ramp corners.

    The pitch must be STEEP: a rolling ball of radius r meets an inclined plane
    tangentially at height r*(1 - cos a); the ramp's exposed foot corner sits at
    0.5 mm + plate_t*cos a above the floor. Only when the tangent height exceeds the
    corner height does the ball contact the inclined FACE first (a climbing normal).
    On a shallow ramp the ball hits the plate's end-face corner, whose PhysX contact
    normal is horizontal — an unclimbable phantom wall (observed on the forge: a
    9.8-degree ramp with a 3.5 mm foot corner walled a 100 mm ball indefinitely).
    At 35 degrees the tangent height is 9.1 mm vs a 3.0 mm corner: face-first."""
    a = math.radians(ang_deg)
    return (deck_t - 0.0005 - _RAMP_TH * math.cos(a)) / math.sin(a)


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


def _box(stage, path: str, size, center, color, contact_offset: float | None,
         pitch_y_deg: float = 0.0) -> None:
    """A colored box prim; collides iff `contact_offset` is not None; optional pitch
    about the local y axis (for the plank's climb ramps)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if pitch_y_deg:
        h = math.radians(pitch_y_deg) / 2
        sxf.AddOrientOp().Set(Gf.Quatf(math.cos(h), Gf.Vec3f(0.0, math.sin(h), 0.0)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(seg.GetPrim(), contact_offset)


def _phys_material(stage, path: str, static: float, dynamic: float) -> Any:
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


def _rigid_dynamic(root, mass: float, lin_damp: float, ang_damp: float,
                   vel_iters: int = 1) -> None:
    """Dynamic rigid-body armor on a compound root: EXPLICIT mass (custom spawners
    apply no cfg schemas — density mass is a trap), damping, no sleeping while we
    judge velocities, the depenetration cap, and the solver iterations (velocity
    iterations 4 kill the GPU ball-creep artifact)."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(int(vel_iters))
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _rigid_kinematic(root) -> None:
    from pxr import UsdPhysics

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(50.0)


def _spawn_platform(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a KINEMATIC raised platform at `prim_path`. Origin = slab centre.
    Slab + guard rails on the back edge (side sign `back_sign`) and both y edges;
    the moat-facing edge stays open."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_kinematic(root)

    L, W, H = cfg.plat_l, cfg.plat_w, cfg.plat_h
    rt, rh = cfg.rail_t, cfg.rail_h
    _box(stage, f"{prim_path}/slab", (L, W, H), (0.0, 0.0, 0.0), cfg.color, cfg.contact_offset)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.friction, cfg.friction - 0.05)
    _bind_material(stage.GetPrimAtPath(f"{prim_path}/slab"), mat)
    rz = H / 2 + rh / 2
    # back rail (the edge away from the moat)
    _box(stage, f"{prim_path}/rail_back", (rt, W, rh),
         (cfg.back_sign * (L / 2 - rt / 2), 0.0, rz), cfg.rail_color, cfg.contact_offset)
    for sgn, nm in ((1.0, "rail_left"), (-1.0, "rail_right")):
        _box(stage, f"{prim_path}/{nm}", (L - rt, rt, rh),
             (0.0, sgn * (W / 2 - rt / 2), rz), cfg.rail_color, cfg.contact_offset)
    return root


def _spawn_plank(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC channel plank at `prim_path`. Origin = deck centre (mid
    thickness). Deck + two side rails (the graspable pinch feature) + a shallow
    climb ramp at each end whose foot lands ~0.5 mm above the plank's underside."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.plank_mass, lin_damp=0.05, ang_damp=0.10)

    dl, dw, dt = cfg.deck_l, cfg.deck_w, cfg.deck_t
    _box(stage, f"{prim_path}/deck", (dl, dw, dt), (0.0, 0.0, 0.0), cfg.color,
         cfg.contact_offset)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.friction, cfg.friction - 0.1)
    _bind_material(stage.GetPrimAtPath(f"{prim_path}/deck"), mat)
    # side rails on the deck top edges
    rt, rh = cfg.prail_t, cfg.prail_h
    for sgn, nm in ((1.0, "rail_l"), (-1.0, "rail_r")):
        _box(stage, f"{prim_path}/{nm}", (dl, rt, rh),
             (0.0, sgn * (dw / 2 - rt / 2), dt / 2 + rh / 2), cfg.rail_color,
             cfg.contact_offset)
        _bind_material(stage.GetPrimAtPath(f"{prim_path}/{nm}"), mat)
    # climb ramps: a short STEEP chamfer plate at each end (see _ramp_len for why
    # steep); the plate's lower foot corner stays 0.5 mm above the deck underside.
    rw, rth = dw - 2 * rt, _RAMP_TH
    ang = cfg.ramp_deg
    a = math.radians(ang)
    rl = _ramp_len(dt, ang)
    for sgn, nm in ((1.0, "ramp_pos"), (-1.0, "ramp_neg")):
        # top-edge start point (sgn*dl/2, dt/2); ramp direction (sgn*cos a, -sin a);
        # centre = top-surface midpoint - normal * (thickness/2)
        mx = sgn * dl / 2 + sgn * (rl / 2) * math.cos(a)
        mz = dt / 2 - (rl / 2) * math.sin(a)
        nx, nz = sgn * math.sin(a), math.cos(a)
        _box(stage, f"{prim_path}/{nm}", (rl, rw, rth),
             (mx - nx * rth / 2, 0.0, mz - nz * rth / 2), cfg.color,
             cfg.contact_offset, pitch_y_deg=sgn * ang)
        _bind_material(stage.GetPrimAtPath(f"{prim_path}/{nm}"), mat)
    return root


def _spawn_dock(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC green dock at `prim_path`. Origin = interior centre AT
    FLOOR LEVEL (the far-platform top). Back wall (+x), two side walls, open mouth
    toward -x (the moat), plus a bright visual-only floor mark."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_kinematic(root)

    hx, hy = cfg.dock_half_x, cfg.dock_half_y  # interior half extents
    wt, wh = cfg.wall_t, cfg.wall_h
    _box(stage, f"{prim_path}/wall_back", (wt, 2 * hy + 2 * wt, wh),
         (hx + wt / 2, 0.0, wh / 2), cfg.color, cfg.contact_offset)
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        _box(stage, f"{prim_path}/{nm}", (2 * hx + wt, wt, wh),
             (wt / 2, sgn * (hy + wt / 2), wh / 2), cfg.color, cfg.contact_offset)
    # visual-only floor mark, 1 mm proud, no collider
    _box(stage, f"{prim_path}/mark", (2 * hx - 0.004, 2 * hy - 0.004, 0.001),
         (0.0, 0.0, 0.0015), cfg.mark_color, None)
    return root


def _platform_spawner_cfg(*, plat_l: float, plat_w: float, plat_h: float, rail_t: float,
                          rail_h: float, back_sign: float, color: tuple, rail_color: tuple,
                          friction: float, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "platform" not in _SPAWNER_CACHE:

        @configclass
        class PlatformSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_platform)
            plat_l: float = 0.52
            plat_w: float = 0.80
            plat_h: float = 0.12
            rail_t: float = 0.012
            rail_h: float = 0.030
            back_sign: float = -1.0
            color: tuple = (0.60, 0.60, 0.62)
            rail_color: tuple = (0.45, 0.45, 0.48)
            friction: float = 0.8
            contact_offset: float = 0.002

        _SPAWNER_CACHE["platform"] = PlatformSpawnerCfg

    return _SPAWNER_CACHE["platform"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        plat_l=plat_l, plat_w=plat_w, plat_h=plat_h, rail_t=rail_t, rail_h=rail_h,
        back_sign=back_sign, color=color, rail_color=rail_color, friction=friction,
        contact_offset=contact_offset,
    )


def _plank_spawner_cfg(*, deck_l: float, deck_w: float, deck_t: float, prail_t: float,
                       prail_h: float, ramp_deg: float, plank_mass: float, color: tuple,
                       rail_color: tuple, friction: float, contact_offset: float) -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "plank" not in _SPAWNER_CACHE:

        @configclass
        class PlankSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plank)
            deck_l: float = 0.30
            deck_w: float = 0.16
            deck_t: float = 0.012
            prail_t: float = 0.008
            prail_h: float = 0.020
            ramp_deg: float = 35.0
            plank_mass: float = 0.35
            color: tuple = (0.15, 0.35, 0.85)
            rail_color: tuple = (0.10, 0.22, 0.55)
            friction: float = 0.8
            contact_offset: float = 0.002

        _SPAWNER_CACHE["plank"] = PlankSpawnerCfg

    return _SPAWNER_CACHE["plank"](
        deck_l=deck_l, deck_w=deck_w, deck_t=deck_t, prail_t=prail_t, prail_h=prail_h,
        ramp_deg=ramp_deg, plank_mass=plank_mass, color=color, rail_color=rail_color,
        friction=friction, contact_offset=contact_offset,
    )


def _dock_spawner_cfg(*, dock_half_x: float, dock_half_y: float, wall_t: float,
                      wall_h: float, color: tuple, mark_color: tuple,
                      contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "dock" not in _SPAWNER_CACHE:

        @configclass
        class DockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_dock)
            dock_half_x: float = 0.0725
            dock_half_y: float = 0.075
            wall_t: float = 0.010
            wall_h: float = 0.045
            color: tuple = (0.10, 0.65, 0.20)
            mark_color: tuple = (0.30, 0.90, 0.35)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["dock"] = DockSpawnerCfg

    return _SPAWNER_CACHE["dock"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        dock_half_x=dock_half_x, dock_half_y=dock_half_y, wall_t=wall_t, wall_h=wall_h,
        color=color, mark_color=mark_color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class MoatBridgeSceneCfg(BaseCfg):
    """Config for `MoatBridgeScene`. Honesty knobs asserted in `__post_init__`: the
    ball is strictly wider than the Franka jaw (grasp/carry impossible — the moat can
    never be flown over), the moat is deeper than the ball (falling in is
    irreversible), the plank always spans the widest moat with real overlap, the
    channel and dock admit the ball with margin, and the plank rail is pinchable."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    dock_x_lo: float = tunable(-0.045)  # ball centre x, dock frame, must exceed (mouth at
    # -dock_half_x = -72.5 mm; a ball fully inside sits at x >= -22.5 mm — 27 mm margin)
    dock_x_hi: float = tunable(0.055)  # ... and stay short of this (back wall stops at +22.5)
    dock_y_tol: float = tunable(0.045)  # |ball centre y - dock centre y| below this
    dock_z_tol: float = tunable(0.020)  # |ball centre z - (top + r)| below this (rest height)
    settle_lin: float = tunable(0.06)  # max |lin vel| when judging (m/s, above GPU creep)
    settle_ang: float = tunable(1.5)  # max |ang vel| when judging (rad/s)
    span_margin: float = tunable(0.012)  # deck end must overlap each rim by more than this
    span_z_tol: float = tunable(0.012)  # plank deck-centre z within this of rim + deck_t/2
    span_tilt_deg: float = tunable(12.0)  # plank deck normal within this of world-up
    crossed_z_tol: float = tunable(0.030)  # ball at far-platform height within this (latch)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    gap_min: float = tunable(0.15)  # moat width range (m)
    gap_max: float = tunable(0.20)
    dock_y_amp: float = tunable(0.22)  # dock lateral offset amplitude along the far rim (m)
    ball_x_rng: tuple = tunable((-0.44, -0.20))  # ball spawn box on the near platform
    ball_y_amp: float = tunable(0.28)
    plank_x_rng: tuple = tunable((-0.30, -0.21))  # plank spawn box (free yaw)
    plank_y_amp: float = tunable(0.16)
    keepout: float = tunable(0.17)  # min ball-centre to plank-long-axis distance at spawn

    # --- info: structure ----------------------------------------------------------------------
    plat_l: float = info(0.52)  # platform slab length (x)
    plat_w: float = info(0.80)  # platform slab width (y)
    plat_h: float = info(0.12)  # platform height = moat depth
    rail_t: float = info(0.012)  # platform guard rail thickness / height
    rail_h: float = info(0.030)
    ball_r: float = info(0.050)  # ball radius — diameter STRICTLY wider than the jaw
    ball_mass: float = info(0.30)
    jaw_max: float = info(0.080)  # Franka parallel-jaw max opening (embodiment honesty)
    deck_l: float = info(0.30)  # plank deck (x) — spans gap_max with real overlap
    deck_w: float = info(0.16)
    deck_t: float = info(0.012)
    prail_t: float = info(0.008)  # plank side rails (the pinch-grasp feature)
    prail_h: float = info(0.020)
    ramp_deg: float = info(35.0)  # chamfer-ramp pitch — STEEP so a rolling ball meets
    # the inclined FACE (tangent height r*(1-cos a) = 9.1 mm > 3.0 mm corner), not the
    # plate's end-face corner whose horizontal contact normal walls the ball
    plank_mass: float = info(0.35)
    dock_half_x: float = info(0.0725)  # dock interior half extents (mouth at -x)
    dock_half_y: float = info(0.075)
    wall_t: float = info(0.010)
    wall_h: float = info(0.045)
    dock_mouth_setback: float = info(0.095)  # dock mouth plane distance past the far rim
    friction_plat: float = info(0.8)
    friction_plank: float = info(0.8)
    friction_ball: float = info(0.6)
    contact_offset: float = info(0.002)  # small: the 12 mm deck step and rails must be real
    plat_color: tuple = info((0.60, 0.60, 0.62))
    prail_color: tuple = info((0.45, 0.45, 0.48))
    plank_color: tuple = info((0.15, 0.35, 0.85))
    plank_rail_color: tuple = info((0.10, 0.22, 0.55))
    ball_color: tuple = info((0.95, 0.45, 0.05))
    dock_color: tuple = info((0.10, 0.65, 0.20))
    mark_color: tuple = info((0.30, 0.90, 0.35))

    # Derived (filled in __post_init__).
    plank_half: float = field(default=None, init=False)  # full half-length incl. ramps

    def __post_init__(self) -> None:
        c = self
        c.plank_half = c.deck_l / 2 + _ramp_len(c.deck_t, c.ramp_deg) * math.cos(
            math.radians(c.ramp_deg))
        assert 2 * c.ball_r > c.jaw_max + 0.015, (
            "the ball must be strictly wider than the Franka jaw — grasping/carrying it "
            "must be impossible, so the moat is a real obstacle")
        assert c.plat_h > 2 * c.ball_r + 0.015, (
            "the moat must be deeper than the ball — falling in must be irreversible")
        assert c.deck_l > c.gap_max + 2 * (c.span_margin + 0.025), (
            "the deck must span the widest moat with real overlap on both rims")
        assert c.deck_w - 2 * c.prail_t > 2 * c.ball_r + 0.02, (
            "the plank channel must admit the ball with margin")
        assert 2 * c.dock_half_y > 2 * c.ball_r + 0.03, "the dock must admit the ball"
        assert c.prail_t < c.jaw_max - 0.03, "the plank rail must be pinchable by the jaw"
        assert 2 * c.plank_half < c.plat_l - 0.02, "the plank must fit on the near platform"
        assert c.dock_y_amp + c.dock_half_y + c.wall_t < c.plat_w / 2 - c.rail_t - 0.02, (
            "the dock must stay inside the far platform's guard rails at any offset")
        assert c.ball_x_rng[1] < -2 * c.ball_r, (
            "the ball must spawn clearly on the near platform (never at the goal side)")
        assert c.dock_x_lo > -c.dock_half_x + 0.02, (
            "the accept window must sit strictly INSIDE the dock mouth plane")


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("moat_bridge")
class MoatBridgeScene(BaseScene):
    cfg: MoatBridgeSceneCfg

    def __init__(self, cfg: MoatBridgeSceneCfg | None = None) -> None:
        super().__init__(cfg or MoatBridgeSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        g0 = (c.gap_min + c.gap_max) / 2
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0, dynamic_friction=0.9, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "near_plat": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/NearPlat",
                spawn=_platform_spawner_cfg(
                    plat_l=c.plat_l, plat_w=c.plat_w, plat_h=c.plat_h, rail_t=c.rail_t,
                    rail_h=c.rail_h, back_sign=-1.0, color=c.plat_color,
                    rail_color=c.prail_color, friction=c.friction_plat,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-c.plat_l / 2, 0.0, c.plat_h / 2)),
            ),
            "far_plat": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/FarPlat",
                spawn=_platform_spawner_cfg(
                    plat_l=c.plat_l, plat_w=c.plat_w, plat_h=c.plat_h, rail_t=c.rail_t,
                    rail_h=c.rail_h, back_sign=+1.0, color=c.plat_color,
                    rail_color=c.prail_color, friction=c.friction_plat,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(g0 + c.plat_l / 2, 0.0, c.plat_h / 2)),
            ),
            "dock": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dock",
                spawn=_dock_spawner_cfg(
                    dock_half_x=c.dock_half_x, dock_half_y=c.dock_half_y, wall_t=c.wall_t,
                    wall_h=c.wall_h, color=c.dock_color, mark_color=c.mark_color,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(g0 + c.dock_mouth_setback + c.dock_half_x, 0.0, c.plat_h)),
            ),
            "plank": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plank",
                spawn=_plank_spawner_cfg(
                    deck_l=c.deck_l, deck_w=c.deck_w, deck_t=c.deck_t, prail_t=c.prail_t,
                    prail_h=c.prail_h, ramp_deg=c.ramp_deg, plank_mass=c.plank_mass,
                    color=c.plank_color, rail_color=c.plank_rail_color,
                    friction=c.friction_plank, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.26, 0.10, c.plat_h + c.deck_t / 2 + 0.001)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,  # kills GPU rolling creep
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.05,
                        sleep_threshold=0.0, stabilization_threshold=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.friction_ball,
                        dynamic_friction=c.friction_ball - 0.1, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ball_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.32, -0.10, c.plat_h + c.ball_r + 0.002)),
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
        self.near_plat: RigidObject = env.iscene["near_plat"]
        self.far_plat: RigidObject = env.iscene["far_plat"]
        self.dock: RigidObject = env.iscene["dock"]
        self.plank: RigidObject = env.iscene["plank"]
        self.ball: RigidObject = env.iscene["ball"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.gap = torch.full((n,), 0.175, device=dev)  # per-episode moat width
        self.x_spawn = torch.full((n,), -0.32, device=dev)  # ball spawn x (progress base)
        self.mouth_x = torch.full((n,), 0.27, device=dev)  # dock mouth plane x (progress goal)
        self.prog_latch = torch.zeros(n, device=dev)
        self.bridged_latch = torch.zeros(n, device=dev)
        self.crossed_latch = torch.zeros(n, device=dev)
        self.docked_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the moat width (the far platform and the dock shift
        together), the dock's lateral offset, the ball spawn and the plank spawn
        (free yaw) with a ball/plank keep-out (resample + deterministic fallback);
        zero the latches, record the progress baseline/goal."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        g = c.gap_min + torch.rand(m, device=dev) * (c.gap_max - c.gap_min)
        self.gap[env_ids] = g

        def write(body, xy: torch.Tensor, z, quat: torch.Tensor | None = None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            if quat is None:
                st[:, 3] = 1.0
            else:
                st[:, 3:7] = quat
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- platforms: near fixed, far sets the moat width ---
        near_xy = torch.stack([torch.full((m,), -c.plat_l / 2, device=dev),
                               torch.zeros(m, device=dev)], dim=-1)
        write(self.near_plat, near_xy, c.plat_h / 2)
        far_xy = torch.stack([g + c.plat_l / 2, torch.zeros(m, device=dev)], dim=-1)
        write(self.far_plat, far_xy, c.plat_h / 2)

        # --- dock: rides the far platform, lateral offset random ---
        dock_y = (torch.rand(m, device=dev) * 2 - 1) * c.dock_y_amp
        dock_x = g + c.dock_mouth_setback + c.dock_half_x
        write(self.dock, torch.stack([dock_x, dock_y], dim=-1), c.plat_h)

        # --- plank: spawn box on the near platform, free yaw ---
        px = c.plank_x_rng[0] + torch.rand(m, device=dev) * (c.plank_x_rng[1]
                                                             - c.plank_x_rng[0])
        py = (torch.rand(m, device=dev) * 2 - 1) * c.plank_y_amp
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        half = yaw / 2
        quat = torch.zeros(m, 4, device=dev)
        quat[:, 0] = torch.cos(half)
        quat[:, 3] = torch.sin(half)
        plank_xy = torch.stack([px, py], dim=-1)
        # The deck underside is the plank's lowest surface (ramp corners stay 0.5 mm
        # above it): spawn 1 mm proud and settle by contact.
        write(self.plank, plank_xy, c.plat_h + c.deck_t / 2 + 0.001, quat)

        # --- ball: spawn box with keep-out from the plank's long axis ---
        u = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)  # plank long axis

        def _sample_ball() -> torch.Tensor:
            bx = c.ball_x_rng[0] + torch.rand(m, device=dev) * (c.ball_x_rng[1]
                                                                - c.ball_x_rng[0])
            by = (torch.rand(m, device=dev) * 2 - 1) * c.ball_y_amp
            return torch.stack([bx, by], dim=-1)

        def _seg_dist(b_xy: torch.Tensor) -> torch.Tensor:
            w = b_xy - plank_xy
            t = (w * u).sum(-1).clamp(-c.plank_half, c.plank_half)
            return (w - u * t.unsqueeze(-1)).norm(dim=-1)

        ball_xy = _sample_ball()
        for _ in range(12):
            bad = _seg_dist(ball_xy) < c.keepout
            if not bad.any():
                break
            ball_xy = torch.where(bad.unsqueeze(-1), _sample_ball(), ball_xy)
        bad = _seg_dist(ball_xy) < c.keepout
        if bad.any():  # deterministic fallback: the spawn-box corner farthest from the plank
            corners = torch.tensor(
                [[c.ball_x_rng[0], -c.ball_y_amp], [c.ball_x_rng[0], c.ball_y_amp],
                 [c.ball_x_rng[1], -c.ball_y_amp], [c.ball_x_rng[1], c.ball_y_amp]],
                device=dev)  # (4, 2)
            d = (corners.unsqueeze(0) - plank_xy.unsqueeze(1)).norm(dim=-1)  # (m, 4)
            pick = corners[d.argmax(dim=1)]
            ball_xy = torch.where(bad.unsqueeze(-1), pick, ball_xy)
        write(self.ball, ball_xy, c.plat_h + c.ball_r + 0.002)

        # --- baselines + latches ---
        self.x_spawn[env_ids] = ball_xy[:, 0]
        self.mouth_x[env_ids] = dock_x - c.dock_half_x
        self.prog_latch[env_ids] = 0.0
        self.bridged_latch[env_ids] = 0.0
        self.crossed_latch[env_ids] = 0.0
        self.docked_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {nm: getattr(self, nm).data.root_state_w[env_ids].clone()
               for nm in ("near_plat", "far_plat", "dock", "plank", "ball")}
        for nm in ("gap", "x_spawn", "mouth_x", "prog_latch", "bridged_latch",
                   "crossed_latch", "docked_latch"):
            out[nm] = getattr(self, nm)[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm in ("near_plat", "far_plat", "dock", "plank", "ball"):
            getattr(self, nm).write_root_state_to_sim(state[nm], env_ids)
        for nm in ("gap", "x_spawn", "mouth_x", "prog_latch", "bridged_latch",
                   "crossed_latch", "docked_latch"):
            getattr(self, nm)[env_ids] = state[nm]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"Two flat gray platforms, each {c.plat_h * 1000:.0f} mm high with low guard "
            f"rails on their outer edges, face each other across an open MOAT — a sheer "
            f"gap {c.gap_min * 1000:.0f}-{c.gap_max * 1000:.0f} mm wide (it varies per "
            f"episode) whose floor lies {c.plat_h * 1000:.0f} mm below the platform tops. "
            f"On the NEAR platform sit an ORANGE BALL ({2 * c.ball_r * 1000:.0f} mm across) "
            f"and a BLUE PLANK ({2 * c.plank_half * 1000:.0f} mm long) that has raised side "
            f"rails forming a channel and a short climb ramp at each end. On the FAR "
            f"platform stands a GREEN DOCK: three low walls around a bright green floor "
            f"mark, its open mouth facing the moat; its position along the far edge changes "
            f"per episode.\n"
            f"Goal: get the orange ball resting inside the green dock on the far platform, "
            f"and leave it at rest. The ball is {2 * c.ball_r * 1000:.0f} mm wide and the "
            f"gripper opens only {c.jaw_max * 1000:.0f} mm, so the ball can NEVER be grasped "
            f"or carried — it can only be rolled, and the moat is deeper than the ball, so "
            f"a ball rolled into the moat is LOST FOR GOOD. First lay the blue plank across "
            f"the moat as a bridge (grip a side rail; the deck must rest on BOTH rims, lined "
            f"up with the dock's mouth), then roll the ball up the plank's end ramp, along "
            f"the channel between its rails, across the moat, and into the dock until it "
            f"rests against the walls. The bridge must be built BEFORE the ball approaches "
            f"the gap — there is no way back from the moat floor."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lay the blue plank across the moat so its deck rests on both rims in line "
            "with the green dock, then roll the orange ball over the bridge into the dock "
            "and leave it at rest. The ball is too wide to grasp; if it falls into the "
            "moat it is lost."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _ball_pos(self) -> torch.Tensor:
        """(N,3) ball centre, env-local."""
        return self.ball.data.root_pos_w - self.env_origins

    def _rims(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,), (N,): near rim x (gap start) and far rim x (gap end), env-local,
        read back from the platform bodies."""
        c = self.cfg
        near = (self.near_plat.data.root_pos_w - self.env_origins)[:, 0] + c.plat_l / 2
        far = (self.far_plat.data.root_pos_w - self.env_origins)[:, 0] - c.plat_l / 2
        return near, far

    def _dock_rel(self) -> torch.Tensor:
        """(N,3) ball centre relative to the dock origin (interior centre, floor level)."""
        return self.ball.data.root_pos_w - self.dock.data.root_pos_w

    # ----- predicates -------------------------------------------------------------------------
    def bridged_now(self) -> torch.Tensor:
        """(N,) bool: the plank rests as a bridge — deck near rim height, near level,
        its deck ends overlapping BOTH rims (in x), and the plank near-still."""
        from isaaclab.utils.math import matrix_from_quat

        c = self.cfg
        near, far = self._rims()
        p = self.plank.data.root_pos_w - self.env_origins
        r = matrix_from_quat(self.plank.data.root_quat_w)
        ax = r[:, 0, 0] * (c.deck_l / 2)  # world-x extent of the deck half-length
        end_lo = p[:, 0] - ax.abs()
        end_hi = p[:, 0] + ax.abs()
        spans = (end_lo < near - c.span_margin) & (end_hi > far + c.span_margin)
        z_ok = (p[:, 2] - (c.plat_h + c.deck_t / 2)).abs() < c.span_z_tol
        level = r[:, 2, 2] > math.cos(math.radians(c.span_tilt_deg))
        still = self.plank.data.root_lin_vel_w.norm(dim=-1) < 0.15
        return spans & z_ok & level & still

    def in_dock(self) -> torch.Tensor:
        """(N,) bool: ball centre inside the dock interior (past the mouth plane with
        margin, between the walls) at far-platform rest height."""
        c = self.cfg
        d = self._dock_rel()
        inside = ((d[:, 0] > c.dock_x_lo) & (d[:, 0] < c.dock_x_hi)
                  & (d[:, 1].abs() < c.dock_y_tol))
        z_ok = (d[:, 2] - c.ball_r).abs() < c.dock_z_tol  # dock origin is at floor level
        return inside & z_ok

    def settled(self) -> torch.Tensor:
        """(N,) bool: ball lin AND ang velocity below thresholds."""
        return ((self.ball.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin)
                & (self.ball.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang))

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch ball x-progress (gated at platform/bridge height so the moat earns
        nothing), the bridge construct, the crossing, and the docking each physics
        substep, so transient progress keeps its credit."""
        c = self.cfg
        b = self._ball_pos()
        _near, far = self._rims()
        high = b[:, 2] > c.plat_h - 0.02  # on a platform or the bridge, not in the moat
        prog = ((b[:, 0] - self.x_spawn) / (self.mouth_x - self.x_spawn).clamp(min=0.05))
        self.prog_latch = torch.maximum(self.prog_latch,
                                        torch.where(high, prog.clamp(0.0, 1.0),
                                                    torch.zeros_like(prog)))
        self.bridged_latch = torch.maximum(self.bridged_latch, self.bridged_now().float())
        crossed = ((b[:, 0] > far + 0.02)
                   & ((b[:, 2] - (c.plat_h + c.ball_r)).abs() < c.crossed_z_tol))
        self.crossed_latch = torch.maximum(self.crossed_latch, crossed.float())
        slow = self.ball.data.root_lin_vel_w.norm(dim=-1) < 0.12
        self.docked_latch = torch.maximum(self.docked_latch,
                                          (self.in_dock() & slow).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: ball inside the dock at far-platform rest height, settled."""
        return self.in_dock() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 * latched height-gated x-progress + 0.30 * bridged
        + 0.30 * crossed + 0.15 * docked, capped at 0.85; exactly 1.0 iff success().
        Doing nothing scores ~0; a ball lost to the moat can never latch crossing or
        docking (both require platform height), so the moat caps an episode at
        whatever was honestly latched before the fall."""
        base = (0.10 * self.prog_latch + 0.30 * self.bridged_latch
                + 0.30 * self.crossed_latch + 0.15 * self.docked_latch).clamp(0.0, 0.85)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="moat_bridge", robot="null"))
