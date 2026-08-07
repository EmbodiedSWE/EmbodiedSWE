"""SkywayBridgeScene — repair the broken elevated track by seating a bridge span
across its gap, then release the captive ball so GRAVITY carries it down the skyway
into the walled catch basin. Derived from rlbench/hit_ball_with_queue but the entire
plan is inverted: the robot never propels or carries the payload to the goal.

Seed (rlbench/hit_ball_with_queue): grasp a cue stick and STRIKE a free ball across an
open surface into a pocket — tool-mediated impulse transfer, one ballistic contact,
aim + momentum decide the outcome. Here striking is exactly the losing move:

- The ball (90 mm — WIDER than a Franka parallel-jaw span, so it can never be grasped
  or carried) waits on an elevated LAUNCH DECK behind a small detent ridge. The only
  productive contact with it is a slow nudge over the ridge; hitting it hard simply
  launches it into the BROKEN part of the track (smoke's seed-strategy check).
- The elevated track has an open GAP mid-span. A ball released before the gap is
  bridged falls through to the floor and is UNRECOVERABLE: it cannot be grasped
  (diameter > jaw), and the basin's walls (>= 12 cm, vertical) cannot be rolled up —
  so execution order is geometry-forced: bridge FIRST, release SECOND.
- The real work is INFRASTRUCTURE: pick the correct yellow BRIDGE SPAN (16 cm — a
  same-looking 9 cm DECOY cannot reach the gap's support tabs and falls through) by
  its gantry carry-handle, and SEAT it across the gap onto recessed support tabs
  between lateral guide walls. Seating is a real contact placement: the span must
  bear on both tabs, upright and axis-aligned, or the ball ride dislodges/misses.
- After release the ball is HANDS-OFF: it rolls down the first slope, crosses the
  seated bridge (the span carries the moving load), rolls down the second slope,
  flies off the launch lip over the basin's low near wall and settles inside.

So a solver needs a different PLAN (choose + place a structural span under placement
tolerances, then a gentle uphill-side release; no aiming, no impulse, no carrying —
the payload's whole journey is passive) and different CODE STRUCTURE (a placement
controller + a nudge, instead of a grasp-stick + swing/strike controller).

Assets are fully procedural (hockey/pen_holder-pattern compound spawners; child
colliders of one body never self-collide):
  - skyway: ONE KINEMATIC compound — launch deck (top 0.26 m) with back wall, side
    rails and the detent ridge; a 6.4 deg down-slope; the GAP with two recessed
    support TABS (top 0.2155 m, bearing 35 mm each) flanked by orange lateral guide
    walls (slot width 0.19 m -> +/-10 mm lateral placement tolerance); a second
    down-slope ending in a launch lip; and the wall-enclosed catch BASIN (floor top
    0.03 m, near wall 0.12 m under the flight path, other walls 0.30 m).
    Origin at the GAP CENTRE on the floor; local +x ("u") points downhill.
  - bridge: DYNAMIC yellow channel span, 0.16 x 0.17 m floor + side rails + a
    gantry handle (12 mm bar, 0.115 m above the span floor — above the rolling
    ball's crown, and a clean parallel-jaw pinch).
  - decoy: IDENTICAL construction, but 0.09 m long — shorter than the 0.11 m
    tab-tip-to-tab-tip opening: it can bear on NEITHER tab and falls through.
  - ball: DYNAMIC red 90 mm sphere (> 80 mm jaw span, asserted below).

Per-episode randomization (readback-verifiable): whole-assembly yaw + xy offset,
Bernoulli bridge/decoy ground-slot swap + per-span xy jitter + free yaw, ball spawn
jitter on the deck.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.30 * seated   — the BRIDGE (not the decoy) ever seated on both tabs: centred
                    within tolerance, upright, axis-aligned, at bearing height, still
                    (latched)
  0.35 * crossed  — the ball ever over the gap at track height WHILE the bridge is
                    seated under it (latched; only a real ride across earns this)
  1.0 iff success() — ball inside the basin, at rest. Non-success cap 0.65.

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

FRANKA_JAW_SPAN = 0.080  # Franka parallel-jaw max opening (m) — the ungraspability bound


# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             orient=None) -> None:
    """Author one box collider (optionally rotated: orient = wxyz quaternion)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


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


def _spawn_skyway(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the skyway: ONE KINEMATIC compound. Origin at the GAP CENTRE on the
    floor, local +x (u) pointing downhill. Deck -> slope 1 -> gap (tabs + guides) ->
    slope 2 -> launch lip -> walled basin."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg

    def sloped_plate(tag: str, u0: float, z0: float, u1: float, z1: float,
                     width: float, color) -> None:
        """A floor plate whose TOP face runs (u0, z0) -> (u1, z1), plus side rails,
        both pitched about y."""
        alpha = math.atan2(z0 - z1, u1 - u0)  # descending toward +u
        length = math.hypot(u1 - u0, z0 - z1)
        h, hh = math.cos(alpha / 2), math.sin(alpha / 2)
        q = (h, 0.0, hh, 0.0)  # pitch about +y: +x -> downhill, +z -> tilted normal
        n = (math.sin(alpha), 0.0, math.cos(alpha))
        mid_u, mid_z = (u0 + u1) / 2, (z0 + z1) / 2
        pc = (mid_u - n[0] * c.floor_t / 2, 0.0, mid_z - n[2] * c.floor_t / 2)
        _add_box(stage, f"{prim_path}/{tag}_floor", center=pc,
                 size=(length, c.chan_w, c.floor_t), color=color, collide=collide,
                 orient=q)
        off = c.floor_t / 2 + c.rail_h / 2
        for sgn, nm in ((1.0, "l"), (-1.0, "r")):
            rc = (mid_u + n[0] * off, sgn * (c.chan_w / 2 + c.rail_t / 2),
                  mid_z + n[2] * off)
            _add_box(stage, f"{prim_path}/{tag}_rail_{nm}", center=rc,
                     size=(length, c.rail_t, c.rail_h), color=c.rail_color,
                     collide=collide, orient=q)

    # --- launch deck (solid block to the floor) + rails + back wall + detent ridge ---
    du0, du1, dt = c.deck_u0, c.deck_u1, c.deck_top
    _add_box(stage, f"{prim_path}/deck", center=((du0 + du1) / 2, 0.0, dt / 2),
             size=(du1 - du0, c.chan_w, dt), color=c.deck_color, collide=collide)
    for sgn, nm in ((1.0, "l"), (-1.0, "r")):
        _add_box(stage, f"{prim_path}/deck_rail_{nm}",
                 center=((du0 + du1) / 2, sgn * (c.chan_w / 2 + c.rail_t / 2),
                         dt + c.rail_h / 2),
                 size=(du1 - du0, c.rail_t, c.rail_h), color=c.rail_color,
                 collide=collide)
    _add_box(stage, f"{prim_path}/deck_back",
             center=(du0 - c.rail_t / 2, 0.0, dt + 0.04),
             size=(c.rail_t, c.chan_w + 2 * c.rail_t, 0.08), color=c.rail_color,
             collide=collide)
    _add_box(stage, f"{prim_path}/ridge",
             center=(c.ridge_u, 0.0, dt + c.ridge_h / 2),
             size=(c.ridge_t, c.chan_w, c.ridge_h), color=c.ridge_color,
             collide=collide)

    # --- slope 1: deck edge down to the gap's uphill lip ---
    sloped_plate("s1", du1, dt, c.seg1_u1, c.seg1_top1, c.chan_w, c.track_color)
    _add_box(stage, f"{prim_path}/s1_block",
             center=((du1 + c.seg1_u1 + 0.02) / 2, 0.0, (c.seg1_top1 - 0.02) / 2),
             size=(c.seg1_u1 - du1 - 0.02, c.chan_w, c.seg1_top1 - 0.02),
             color=c.deck_color, collide=collide)

    # --- the gap: recessed support tabs + lateral guide walls ---
    tab_len = c.tab_u_outer - c.gap_half
    for sgn, nm in ((-1.0, "up"), (1.0, "dn")):
        uc = sgn * (c.gap_half + tab_len / 2)
        _add_box(stage, f"{prim_path}/tab_{nm}",
                 center=(uc, 0.0, c.tab_top - c.tab_t / 2),
                 size=(tab_len, c.guide_hw * 2, c.tab_t), color=c.tab_color,
                 collide=collide)
        for sv, vn in ((1.0, "l"), (-1.0, "r")):
            _add_box(stage, f"{prim_path}/guide_{nm}_{vn}",
                     center=(uc, sv * (c.guide_hw + c.rail_t / 2),
                             c.tab_top + c.guide_h / 2),
                     size=(tab_len, c.rail_t, c.guide_h), color=c.tab_color,
                     collide=collide)

    # --- slope 2: gap's downhill lip to the launch lip ---
    sloped_plate("s2", c.seg2_u0, c.seg2_top0, c.seg2_u1, c.seg2_top1, c.chan_w,
                 c.track_color)
    _add_box(stage, f"{prim_path}/s2_block",
             center=((c.seg2_u0 + 0.02 + c.seg2_u1) / 2, 0.0,
                     (c.seg2_top1 - 0.02) / 2),
             size=(c.seg2_u1 - c.seg2_u0 - 0.02, c.chan_w, c.seg2_top1 - 0.02),
             color=c.deck_color, collide=collide)

    # --- catch basin: floor + near (low) / far / side walls ---
    b0, b1, hw = c.basin_u0, c.basin_u1, c.basin_hw
    _add_box(stage, f"{prim_path}/basin_floor",
             center=((b0 + b1) / 2, 0.0, c.basin_floor_top / 2),
             size=(b1 - b0, 2 * hw, c.basin_floor_top), color=c.basin_color,
             collide=collide)
    _add_box(stage, f"{prim_path}/basin_near",
             center=(b0 - c.wall_t / 2, 0.0, c.near_wall_h / 2),
             size=(c.wall_t, 2 * hw + 2 * c.wall_t, c.near_wall_h),
             color=c.basin_wall_color, collide=collide)
    _add_box(stage, f"{prim_path}/basin_far",
             center=(b1 + c.wall_t / 2, 0.0, c.wall_h / 2),
             size=(c.wall_t, 2 * hw + 2 * c.wall_t, c.wall_h),
             color=c.basin_wall_color, collide=collide)
    for sgn, nm in ((1.0, "l"), (-1.0, "r")):
        _add_box(stage, f"{prim_path}/basin_side_{nm}",
                 center=((b0 + b1) / 2, sgn * (hw + c.wall_t / 2), c.wall_h / 2),
                 size=(b1 - b0 + 2 * c.wall_t, c.wall_t, c.wall_h),
                 color=c.basin_wall_color, collide=collide)
    return root


def _spawn_span(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a channel span (bridge or decoy): DYNAMIC — floor plate + side rails +
    gantry carry-handle (posts + bar). Origin at the FLOOR PLATE centre (so the
    explicit MassAPI mass keeps the CoM low)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.50)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    c = cfg
    ln, wd, ft = c.span_len, c.span_w, c.span_floor_t
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, 0.0),
             size=(ln, wd, ft), color=c.color, collide=collide)
    for sgn, nm in ((1.0, "l"), (-1.0, "r")):
        _add_box(stage, f"{prim_path}/rail_{nm}",
                 center=(0.0, sgn * (wd / 2 - c.span_rail_t / 2),
                         ft / 2 + c.span_rail_h / 2),
                 size=(ln, c.span_rail_t, c.span_rail_h), color=c.color,
                 collide=collide)
        _add_box(stage, f"{prim_path}/post_{nm}",
                 center=(0.0, sgn * (wd / 2 - c.span_rail_t / 2),
                         ft / 2 + c.span_rail_h + c.post_h / 2),
                 size=(c.bar_t, c.bar_t, c.post_h), color=c.color, collide=collide)
    bar_z = ft / 2 + c.span_rail_h + c.post_h + c.bar_t / 2
    _add_box(stage, f"{prim_path}/bar", center=(0.0, 0.0, bar_z),
             size=(c.bar_t, wd, c.bar_t), color=c.color, collide=collide)
    return root


def _spawn_ball(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the ball: DYNAMIC sphere; LOW damping (it must roll the whole skyway on
    gravity alone), sleep thresholds zeroed (judged for stillness)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.02)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    r = float(cfg.radius)
    sph = UsdGeom.Sphere.Define(stage, f"{prim_path}/ball")
    sph.CreateRadiusAttr(r)
    sph.CreateExtentAttr([Gf.Vec3f(-r, -r, -r), Gf.Vec3f(r, r, r)])
    sph.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    _make_collide(cfg)(sph.GetPrim())
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "skyway" not in _SPAWNER_CACHE:

        @configclass
        class SkywaySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_skyway)
            deck_u0: float = -0.54
            deck_u1: float = -0.36
            deck_top: float = 0.26
            ridge_u: float = -0.375
            ridge_h: float = 0.008
            ridge_t: float = 0.012
            seg1_u1: float = -0.09
            seg1_top1: float = 0.2295
            seg2_u0: float = 0.09
            seg2_top0: float = 0.2255
            seg2_u1: float = 0.33
            seg2_top1: float = 0.1984
            gap_half: float = 0.055
            tab_u_outer: float = 0.09
            tab_top: float = 0.2155
            tab_t: float = 0.02
            guide_hw: float = 0.095
            guide_h: float = 0.05
            chan_w: float = 0.16
            rail_t: float = 0.015
            rail_h: float = 0.04
            floor_t: float = 0.02
            basin_u0: float = 0.35
            basin_u1: float = 0.63
            basin_hw: float = 0.13
            basin_floor_top: float = 0.03
            near_wall_h: float = 0.12
            wall_h: float = 0.30
            wall_t: float = 0.015
            deck_color: tuple = (0.42, 0.45, 0.50)
            track_color: tuple = (0.62, 0.64, 0.68)
            rail_color: tuple = (0.30, 0.33, 0.38)
            tab_color: tuple = (0.90, 0.55, 0.10)
            ridge_color: tuple = (0.15, 0.15, 0.18)
            basin_color: tuple = (0.20, 0.55, 0.25)
            basin_wall_color: tuple = (0.16, 0.40, 0.20)
            contact_offset: float = 0.002

        @configclass
        class SpanSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_span)
            span_len: float = 0.16
            span_w: float = 0.17
            span_floor_t: float = 0.012
            span_rail_t: float = 0.012
            span_rail_h: float = 0.03
            post_h: float = 0.073
            bar_t: float = 0.012
            color: tuple = (0.93, 0.80, 0.12)
            contact_offset: float = 0.002

        @configclass
        class BallSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ball)
            radius: float = 0.045
            color: tuple = (0.85, 0.10, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(skyway=SkywaySpawnerCfg, span=SpanSpawnerCfg,
                              ball=BallSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SkywayBridgeSceneCfg(BaseCfg):
    """Config for `SkywayBridgeScene`. The interlocks are metric: the ball (90 mm) is
    wider than the Franka jaw span (80 mm, asserted) so it can never be grasped or
    carried; the basin's walls (>= 0.12 m, vertical) cannot be rolled up from the
    floor; the decoy span (0.09 m) is shorter than the tab-tip opening (0.11 m) so it
    can bear on neither tab; the guide slot (0.19 m for the 0.17 m span) makes the
    lateral seating tolerance +/-10 mm; a released ball reaches the basin ONLY over a
    seated bridge."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.80)  # max |ang vel| when judging (rad/s)
    seat_u_tol: float = tunable(0.020)  # bridge CoM |u| seating tolerance (m):
    # at 0.02 the short end still bears ~5 mm on its tab; beyond, the span tips
    seat_v_tol: float = tunable(0.030)  # bridge CoM |v| seating tolerance (m): the
    # guide slot physically caps |v| at ~0.010 for any state at bearing height
    seat_z_tol: float = tunable(0.008)  # bridge CoM height tolerance about seat_z (m)
    seat_align: float = tunable(0.95)  # min |axis alignment| cosines for "seated"
    cross_z_min: float = tunable(0.22)  # ball CoM height over the gap that counts

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_max_deg: float = tunable(20.0)  # whole-assembly yaw (+/- deg)
    xy_jitter: float = tunable(0.05)  # whole-assembly xy offset (+/- m)
    swap_slots: bool = tunable(True)  # Bernoulli bridge/decoy ground-slot swap
    span_jitter: float = tunable(0.03)  # per-span ground xy jitter (+/- m)
    span_yaw_deg: float = tunable(180.0)  # per-span free yaw (+/- deg)
    ball_u_jitter: float = tunable(0.025)  # ball deck-slot u jitter (+/- m)
    ball_v_jitter: float = tunable(0.020)  # ball deck-slot v jitter (+/- m)

    # --- info: layout (assembly-local; one Franka base at u=-0.15, v=-0.45) ----------------------
    slot_a: tuple = info((0.02, -0.38))  # span ground slot A (downhill side)
    slot_b: tuple = info((-0.32, -0.38))  # span ground slot B (uphill side)
    ball_slot_u: float = info(-0.44)  # ball deck slot (behind the ridge)

    # --- info: skyway structure (assembly frame: origin at gap centre, +u downhill) --------------
    deck_u0: float = info(-0.54)
    deck_u1: float = info(-0.36)
    deck_top: float = info(0.26)
    ridge_u: float = info(-0.375)
    ridge_h: float = info(0.008)
    seg1_u1: float = info(-0.09)
    seg1_top1: float = info(0.2295)
    seg2_u0: float = info(0.09)
    seg2_top0: float = info(0.2255)
    seg2_u1: float = info(0.33)
    seg2_top1: float = info(0.1984)
    gap_half: float = info(0.055)  # tab-tip half-opening: 0.11 m clear span
    tab_u_outer: float = info(0.09)
    tab_top: float = info(0.2155)
    guide_hw: float = info(0.095)  # guide-wall inner half-gap (lateral funnel)
    guide_h: float = info(0.05)
    chan_w: float = info(0.16)  # track channel floor width
    rail_t: float = info(0.015)
    rail_h: float = info(0.04)
    basin_u0: float = info(0.35)
    basin_u1: float = info(0.63)
    basin_hw: float = info(0.13)
    basin_floor_top: float = info(0.03)
    near_wall_h: float = info(0.12)
    wall_h: float = info(0.30)

    # --- info: spans -----------------------------------------------------------------------------
    bridge_len: float = info(0.16)  # bears 25+ mm on each tab when centred
    decoy_len: float = info(0.09)  # < 0.11 m opening: bears on NEITHER tab
    span_w: float = info(0.17)
    span_floor_t: float = info(0.012)
    span_rail_h: float = info(0.03)
    span_mass: float = info(0.45)
    span_color: tuple = info((0.93, 0.80, 0.12))  # both spans yellow: pick by LENGTH

    # --- info: ball ------------------------------------------------------------------------------
    ball_r: float = info(0.045)  # 90 mm dia > 80 mm jaw span: never graspable
    ball_mass: float = info(0.15)
    ball_color: tuple = info((0.85, 0.10, 0.10))

    contact_offset: float = info(0.002)
    # rubric weights (0.30 + 0.35 = 0.65 = the non-success cap)
    w_seat: float = info(0.30)
    w_cross: float = info(0.35)

    def __post_init__(self) -> None:
        assert 2 * self.ball_r > FRANKA_JAW_SPAN, \
            "ball must be wider than the Franka jaw span (ungraspable by design)"
        assert self.decoy_len < 2 * self.gap_half, \
            "decoy must be shorter than the tab-tip opening (must fall through)"
        assert self.bridge_len > 2 * self.tab_u_outer - 0.03, \
            "bridge must bear on both tabs when centred"
        # bridge CoM height when seated on the tabs (floor-plate centre)
        self.seat_z: float = self.tab_top + self.span_floor_t / 2


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("skyway_bridge")
class SkywayBridgeScene(BaseScene):
    cfg: SkywayBridgeSceneCfg

    def __init__(self, cfg: SkywayBridgeSceneCfg | None = None) -> None:
        super().__init__(cfg or SkywayBridgeSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        skyway_spawn = spawners["skyway"](
            mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            deck_u0=c.deck_u0, deck_u1=c.deck_u1, deck_top=c.deck_top,
            ridge_u=c.ridge_u, ridge_h=c.ridge_h,
            seg1_u1=c.seg1_u1, seg1_top1=c.seg1_top1,
            seg2_u0=c.seg2_u0, seg2_top0=c.seg2_top0,
            seg2_u1=c.seg2_u1, seg2_top1=c.seg2_top1,
            gap_half=c.gap_half, tab_u_outer=c.tab_u_outer, tab_top=c.tab_top,
            guide_hw=c.guide_hw, guide_h=c.guide_h,
            chan_w=c.chan_w, rail_t=c.rail_t, rail_h=c.rail_h,
            basin_u0=c.basin_u0, basin_u1=c.basin_u1, basin_hw=c.basin_hw,
            basin_floor_top=c.basin_floor_top, near_wall_h=c.near_wall_h,
            wall_h=c.wall_h, contact_offset=c.contact_offset,
        )

        def span_spawn(length):
            return spawners["span"](
                mass_props=sim_utils.MassPropertiesCfg(mass=c.span_mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                span_len=length, span_w=c.span_w, span_floor_t=c.span_floor_t,
                span_rail_h=c.span_rail_h, color=c.span_color,
                contact_offset=c.contact_offset,
            )

        ball_spawn = spawners["ball"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            radius=c.ball_r, color=c.ball_color, contact_offset=c.contact_offset,
        )
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "skyway": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Skyway",
                spawn=skyway_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "bridge": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bridge",
                spawn=span_spawn(c.bridge_len),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_a[0], c.slot_a[1], c.span_floor_t / 2 + 0.002)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=span_spawn(c.decoy_len),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_b[0], c.slot_b[1], c.span_floor_t / 2 + 0.002)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=ball_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.ball_slot_u, 0.0, c.deck_top + c.ball_r + 0.002)),
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
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.skyway: RigidObject = env.iscene["skyway"]
        self.bridge: RigidObject = env.iscene["bridge"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.ball: RigidObject = env.iscene["ball"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._seated_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._crossed_ever = torch.zeros(n, dtype=torch.bool, device=dev)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: the whole skyway re-posed (yaw + xy offset), the ball
        re-placed on the deck behind the ridge (jitter), bridge/decoy randomly
        ASSIGNED to the two ground slots (+ jitter + free yaw), latches cleared."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- skyway (kinematic): yaw + xy offset ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_max_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = (torch.rand(m, 2, device=dev) * 2 - 1) * c.xy_jitter
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.skyway.write_root_state_to_sim(st, env_ids)
        s_pos, s_quat = st[:, 0:3].clone(), st[:, 3:7].clone()

        # --- ball: deck slot behind the ridge, jittered, expressed in assembly frame ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.ball_slot_u \
            + (torch.rand(m, device=dev) * 2 - 1) * c.ball_u_jitter
        loc[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.ball_v_jitter
        loc[:, 2] = c.deck_top + c.ball_r + 0.002
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = s_pos + quat_apply(s_quat, loc)
        st[:, 3] = 1.0
        self.ball.write_root_state_to_sim(st, env_ids)

        # --- spans: Bernoulli slot swap + jitter + free yaw, flat on the ground ---
        if c.swap_slots:
            swap = torch.rand(m, device=dev) < 0.5
        else:
            swap = torch.zeros(m, dtype=torch.bool, device=dev)
        slot_a = torch.tensor(c.slot_a, device=dev).expand(m, 2)
        slot_b = torch.tensor(c.slot_b, device=dev).expand(m, 2)
        for body, xy in ((self.bridge, torch.where(swap.unsqueeze(1), slot_b, slot_a)),
                         (self.decoy, torch.where(swap.unsqueeze(1), slot_a, slot_b))):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0:2] = xy + (torch.rand(m, 2, device=dev) * 2 - 1) * c.span_jitter
            loc[:, 2] = c.span_floor_t / 2 + 0.002
            syaw = yaw + (torch.rand(m, device=dev) * 2 - 1) \
                * math.radians(c.span_yaw_deg)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = s_pos + quat_apply(s_quat, loc)
            st[:, 3], st[:, 6] = torch.cos(syaw / 2), torch.sin(syaw / 2)
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._seated_ever[env_ids] = False
        self._crossed_ever[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "skyway": self.skyway.data.root_state_w[env_ids].clone(),
            "bridge": self.bridge.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "seated_ever": self._seated_ever[env_ids].clone(),
            "crossed_ever": self._crossed_ever[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.skyway.write_root_state_to_sim(state["skyway"], env_ids)
        self.bridge.write_root_state_to_sim(state["bridge"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        self._seated_ever[env_ids] = state["seated_ever"]
        self._crossed_ever[env_ids] = state["crossed_ever"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"An elevated BALL RUN (a gray 'skyway') stands on the floor: a launch "
            f"DECK (top {c.deck_top * 100:.0f} cm high) holds a RED ball "
            f"({2 * c.ball_r * 1000:.0f} mm — too wide for the gripper jaws, it can "
            f"only be pushed, never grasped or carried) behind a small dark detent "
            f"ridge; from the deck a walled track slopes down, but the track is "
            f"BROKEN: an open GAP ({2 * c.gap_half * 100:.0f} cm clear span) "
            f"interrupts it, marked by ORANGE support tabs recessed just below track "
            f"level on both sides, flanked by short orange guide walls. Past the "
            f"gap the track slopes on and ends at a lip above a GREEN walled catch "
            f"BASIN. On the floor beside the skyway lie TWO yellow channel-shaped "
            f"spans, each with a raised carry HANDLE bar across its top (the bar is "
            f"{12:.0f} mm thick — pinch it with the jaws): one is "
            f"{c.bridge_len * 100:.0f} cm long (the BRIDGE — long enough to rest on "
            f"both orange tabs), the other {c.decoy_len * 100:.0f} cm (a DECOY — "
            f"too short to reach either tab; it falls straight through the gap). "
            f"They look identical except for LENGTH, and which lies where changes "
            f"between episodes.\n"
            f"Goal: the red ball must end up resting INSIDE the green basin. The "
            f"only way it can get there is by ROLLING the whole track, so: FIRST "
            f"pick the LONGER yellow span up by its handle and seat it across the "
            f"gap — lowered between the orange guide walls onto both orange tabs, "
            f"level and aligned with the track (the guides leave about +/-1 cm of "
            f"side play; its channel must line up so the ball can run through under "
            f"the handle) — and only THEN gently push the red ball over the dark "
            f"ridge so it rolls down the track, across the seated bridge, and drops "
            f"into the basin. Order matters: a ball released before the bridge is "
            f"seated falls through the gap to the floor and is UNRECOVERABLE (it "
            f"cannot be grasped, and the basin walls cannot be rolled up). Placing "
            f"the short decoy instead, leaving the bridge misseated (an end off its "
            f"tab, tilted, or skewed so the channel is blocked), striking the ball "
            f"hard from the deck, or a ball resting anywhere except inside the "
            f"basin — all failure."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Seat the longer yellow span by its handle across the gap in the "
            "elevated track so it rests on both orange tabs, then gently push the "
            "red ball over the ridge so it rolls down the track, across the bridge, "
            "and comes to rest inside the green basin. Do not release the ball "
            "before the bridge is seated."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def _local(self, body: RigidObject) -> torch.Tensor:
        """(N, 3) body CoM position in the assembly frame (u downhill, v lateral)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - self.skyway.data.root_pos_w
        return quat_apply_inverse(self.skyway.data.root_quat_w, rel)

    def _span_axes_ok(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: span upright (local z up) and length-axis aligned with the
        track (either way round — the span is fore-aft symmetric)."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        ex = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        up_w = quat_apply(body.data.root_quat_w, ez)
        x_w = quat_apply(body.data.root_quat_w, ex)
        x_a = quat_apply_inverse(self.skyway.data.root_quat_w, x_w)
        return (up_w[:, 2] > c.seat_align) & (x_a[:, 0].abs() > c.seat_align)

    def _bridge_seated(self, require_still: bool = True) -> torch.Tensor:
        """(N,) bool: the BRIDGE bearing on both tabs — CoM centred within tolerance,
        at bearing height, upright and axis-aligned (and still, unless the caller is
        checking the under-load state while the ball rides it)."""
        c = self.cfg
        loc = self._local(self.bridge)
        pose_ok = ((loc[:, 0].abs() < c.seat_u_tol)
                   & (loc[:, 1].abs() < c.seat_v_tol)
                   & ((loc[:, 2] - c.seat_z).abs() < c.seat_z_tol)
                   & self._span_axes_ok(self.bridge))
        if not require_still:
            return pose_ok
        still = self.bridge.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return pose_ok & still

    def _in_basin(self) -> torch.Tensor:
        """(N,) bool: ball CoM inside the basin volume, resting-height band."""
        c = self.cfg
        loc = self._local(self.ball)
        return ((loc[:, 0] > c.basin_u0 + 0.01) & (loc[:, 0] < c.basin_u1 - 0.01)
                & (loc[:, 1].abs() < c.basin_hw - 0.015)
                & (loc[:, 2] > c.basin_floor_top + c.ball_r - 0.02)
                & (loc[:, 2] < c.basin_floor_top + c.ball_r + 0.05))

    def _over_gap(self) -> torch.Tensor:
        """(N,) bool: ball CoM over the gap AT TRACK HEIGHT (a ball on the floor
        under the gap reads z ~ 0.045 and never qualifies)."""
        c = self.cfg
        loc = self._local(self.ball)
        return (loc[:, 0].abs() < c.gap_half + 0.01) & (loc[:, 2] > c.cross_z_min)

    def _update_latches(self) -> None:
        self._seated_ever |= self._bridge_seated(require_still=True)
        # only a ride across a bridge that is seated UNDER the ball counts
        self._crossed_ever |= self._over_gap() \
            & self._bridge_seated(require_still=False)

    # ----- step-coupled bookkeeping (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No plant here (the skyway is kinematic and jointless) — just latch rubric
        progress every step so transient achievements keep credit."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: ball resting inside the basin, at rest. Physical outcome only —
        by construction the basin is reachable only over the seated bridge (the ball
        can never be grasped, the walls never rolled up)."""
        c = self.cfg
        self._update_latches()
        still = ((self.ball.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                 & (self.ball.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega))
        return self._in_basin() & still

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30*seated + 0.35*crossed — both latched, ~0 for
        doing nothing, non-success cap 0.65 — and exactly 1.0 iff success() holds."""
        c = self.cfg
        self._update_latches()
        base = (c.w_seat * self._seated_ever.float()
                + c.w_cross * self._crossed_ever.float()).clamp(max=0.65)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="skyway_bridge", robot="null"))
