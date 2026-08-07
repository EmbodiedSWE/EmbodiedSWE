"""CamPressScene — depress a red push button by ROTATING a captive stair-cam wheel
(sim_gen task `push_button_i35`).

Derived from rlbench/push_button, but STRATEGICALLY different: the seed is one
fingertip primitive — reach an exposed button and PRESS it along its axis until it
bottoms out. Here the very same red button exists (a follower riding in a square
guide sleeve) but pressing it is PHYSICALLY IMPOSSIBLE: the follower's foot rests on
the top step of a spiral staircase machined into a heavy rotatable CAM WHEEL, so an
axial push just drives the load path wheel -> pedestal -> ground and nothing moves
(the smoke battery pushes with 30 N and proves depth stays ~0). The only way the
button goes down is to OPERATE THE MACHINE: rotate the wheel by its rim pegs in the
one direction the geometry allows, so the staircase turns under the foot and the
follower drops riser by riser — 8 steps, 70 mm of travel — until the wheel's
underside pin arrests against the frame's stop post with the foot parked on the
lowest step. The press is thereby TRANSMITTED, not performed: the hand never touches
the button in the nominal solution, and every millimetre of button travel is a
side-effect of wheel rotation.

Plan-level contrast with the seed: the seed's whole skill (axial press on the judged
part) is nullified and scored ~0; the judged part is moved only INDIRECTLY through a
mechanism; the wrong rotation direction is blocked by a headwall + stop post (so the
solver must READ the wheel's chirality, which is randomized, and rotate the correct
way); progress is PHYSICALLY latched by the risers (the foot cannot climb back up a
10 mm riser by itself), and the machine is self-locking at every step, so success
must persist hands-off.

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide):
  - tower (KINEMATIC compound): base plate, pedestal, centre post with a retaining
    cap (captures the wheel hub: <= 5 mm lift, no dismount), a square guide SLEEVE
    held over the stair track by an outboard column + arm, and two STOP POSTS on the
    base at +/-50 deg that arrest the wheel's underside pin at each end of travel.
  - disc_a / disc_b (DYNAMIC compounds): the cam wheel in its two mirror-image
    chiralities (s = +1 descends clockwise-from-top, s = -1 counter-clockwise). Hub
    ring (rides the centre post), circular deck, 8-step spiral staircase (tops
    0.171 -> 0.101, 10 mm risers), a tall headwall before step 0 (blocks reverse
    rotation under the foot), an underside stop pin, and 4 blue rim PEGS (the
    intended handles). Exactly one disc is MOUNTED per episode; the other is
    depoted far away on the ground.
  - follower (DYNAMIC compound): the button — BALL foot (a sphere cannot perch on
    a step's leading edge, so every riser drop is crisp; too big for the sleeve
    aperture, so it can neither fall through nor be pulled out), cylindrical stem
    riding in the sleeve, red head cap well above the sleeve.

Per-episode randomization (readback-verifiable): tower yaw FREE (+/-180 deg) + xy
jitter; wheel CHIRALITY (which mirror disc is mounted, 50/50) — this flips the
required rotation direction, the staircase handedness, and the stop-post that ends
the travel; the wheel's parked angle jitters a few degrees.

Rubric (0..1; latched partial credit anchored in the demonstrated solve):
  0.20 * d >= 25 mm ever (foot on step 3+, wheel seated, follower in sleeve, slow)
  0.20 * d >= 45 mm ever (step 5+, same guards)
  0.30 * d >= 66 mm ever (step 7, same guards)
  capped at 0.70; exactly 1.0 iff success(): depth >= 66 mm AND the wheel is parked
  at its low stop (wrap-aware angle check) AND the wheel is seated on the pedestal
  AND the follower is in the sleeve, everything settled and finite. Null policy ~0;
  pressing the button scores ~0; rotating the wrong way scores ~0; dismount
  cheats (follower free-fall onto the sleeve's head-catch) latch nothing because
  the wheel-seated guard fails.

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


def _add_box(stage, path: str, *, center, size, color, collide: Callable, yaw_deg=None):
    """One box child: translate (+ optional z-rotation) + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw_deg is not None:
        xf.AddRotateZOp().Set(float(yaw_deg))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _add_sphere(stage, path: str, *, center, radius, color, collide: Callable):
    """One sphere child: translate only, displayColor, collider."""
    from pxr import Gf, UsdGeom

    r = float(radius)
    sph = UsdGeom.Sphere.Define(stage, path)
    sph.CreateRadiusAttr(r)
    sph.CreateExtentAttr([Gf.Vec3f(-r, -r, -r), Gf.Vec3f(r, r, r)])
    xf = UsdGeom.Xformable(sph.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sph.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(sph.GetPrim())
    return sph.GetPrim()


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable):
    """One z-axis cylinder child: translate only, displayColor, collider."""
    from pxr import Gf, UsdGeom

    r, h = float(radius), float(height)
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(r)
    cyl.CreateHeightAttr(h)
    cyl.CreateAxisAttr("Z")
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    return cyl.GetPrim()


def _apply_dynamic(root, mass: float) -> None:
    """Make a compound root a DYNAMIC rigid body with an explicit total mass."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateLinearDampingAttr(0.05)
    prb.CreateAngularDampingAttr(0.05)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)


def _spawn_tower(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the frame tower at `prim_path`: KINEMATIC compound. Local frame:
    origin on the ground under the wheel axis, +z up, the guide sleeve at local
    angle 0 deg over (sleeve_x, 0)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    # base plate + pedestal + centre post + retaining cap
    _add_box(stage, f"{prim_path}/base", center=(0.0, 0.0, c.base_t / 2),
             size=(c.base_size, c.base_size, c.base_t), color=c.body_color,
             collide=collide)
    ped_h = c.ped_top - c.base_t
    _add_cyl(stage, f"{prim_path}/pedestal", center=(0.0, 0.0, c.base_t + ped_h / 2),
             radius=c.ped_r, height=ped_h, color=c.body_color, collide=collide)
    post_h = c.post_top - c.base_t
    _add_cyl(stage, f"{prim_path}/post", center=(0.0, 0.0, c.base_t + post_h / 2),
             radius=c.post_r, height=post_h, color=c.cap_color, collide=collide)
    _add_cyl(stage, f"{prim_path}/cap", center=(0.0, 0.0, c.cap_bot + c.cap_t / 2),
             radius=c.cap_r, height=c.cap_t, color=c.cap_color, collide=collide)
    # guide sleeve: 4 walls around the square aperture over (sleeve_x, 0)
    zc = (c.sleeve_z0 + c.sleeve_z1) / 2
    zh = c.sleeve_z1 - c.sleeve_z0
    off = c.aperture / 2 + c.sleeve_t / 2
    for sy in (1.0, -1.0):
        t = "p" if sy > 0 else "n"
        _add_box(stage, f"{prim_path}/sleeve_y{t}",
                 center=(c.sleeve_x, sy * off, zc),
                 size=(c.aperture + 2 * c.sleeve_t, c.sleeve_t, zh),
                 color=c.sleeve_color, collide=collide)
    for sx in (1.0, -1.0):
        t = "p" if sx > 0 else "n"
        _add_box(stage, f"{prim_path}/sleeve_x{t}",
                 center=(c.sleeve_x + sx * off, 0.0, zc),
                 size=(c.sleeve_t, c.aperture, zh),
                 color=c.sleeve_color, collide=collide)
    # outboard column (inner face at x = col_in_x) + arm to the sleeve's outer wall
    _add_box(stage, f"{prim_path}/column",
             center=(c.col_in_x + c.sleeve_t / 2, 0.0, (c.base_t + c.sleeve_z1) / 2),
             size=(c.sleeve_t, c.aperture + 2 * c.sleeve_t, c.sleeve_z1 - c.base_t),
             color=c.sleeve_color, collide=collide)
    arm_x0 = c.sleeve_x + c.aperture / 2 + c.sleeve_t - 0.002   # overlap the sleeve wall
    arm_x1 = c.col_in_x + c.sleeve_t
    _add_box(stage, f"{prim_path}/arm",
             center=((arm_x0 + arm_x1) / 2, 0.0, c.sleeve_z1 - 0.025),
             size=(arm_x1 - arm_x0, c.aperture + 2 * c.sleeve_t, 0.050),
             color=c.sleeve_color, collide=collide)
    # stop posts at local angles +/- post_deg (radial x tangential x height, yawed)
    for sgn in (1.0, -1.0):
        t = "p" if sgn > 0 else "n"
        a = math.radians(sgn * c.post_deg)
        _add_box(stage, f"{prim_path}/stop_{t}",
                 center=(c.stop_r * math.cos(a), c.stop_r * math.sin(a),
                         c.base_t + c.stop_size[2] / 2),
                 size=c.stop_size, color=c.stop_color, collide=collide,
                 yaw_deg=sgn * c.post_deg)
    return root


def _spawn_disc(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one cam wheel at `prim_path`: DYNAMIC compound. Local frame: origin on
    the wheel axis AT GROUND LEVEL of the mounted pose (so mounting = write the
    tower's root pose + yaw). Chirality s=+1: staircase descends in the -z rotation
    direction (clockwise from top); s=-1 is the mirror image."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _apply_dynamic(root, cfg.mass)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    s = 1.0 if c.chirality >= 0 else -1.0
    # hub ring: 8 yawed boxes forming an octagonal bore around the centre post
    hub_rc = c.hub_in_r + c.hub_rad_t / 2
    hub_tan = 2 * hub_rc * math.tan(math.pi / 8) + 0.004    # overlap neighbours
    for i in range(8):
        a = math.radians(45.0 * i)
        _add_box(stage, f"{prim_path}/hub_{i}",
                 center=(hub_rc * math.cos(a), hub_rc * math.sin(a),
                         c.hub_bot + c.hub_h / 2),
                 size=(c.hub_rad_t, hub_tan, c.hub_h), color=c.deck_color,
                 collide=collide, yaw_deg=45.0 * i)
    # deck: 12 sector boxes -> a continuous annular table under the staircase
    for i in range(12):
        a = math.radians(30.0 * i)
        _add_box(stage, f"{prim_path}/deck_{i}",
                 center=(c.track_r * math.cos(a), c.track_r * math.sin(a),
                         (c.deck_bot + c.deck_top) / 2),
                 size=(c.deck_rad, c.deck_tan, c.deck_top - c.deck_bot),
                 color=c.deck_color, collide=collide, yaw_deg=30.0 * i)
    # spiral staircase: 8 steps, tops descending by `riser` per `step_pitch_deg`
    for k in range(c.n_steps):
        phi = s * (c.step0_deg + c.step_pitch_deg * k)
        top = c.step0_top - c.riser * k
        a = math.radians(phi)
        _add_box(stage, f"{prim_path}/step_{k}",
                 center=(c.track_r * math.cos(a), c.track_r * math.sin(a),
                         (c.deck_top + top) / 2),
                 size=(c.step_rad, c.step_tan, top - c.deck_top),
                 color=c.step_color, collide=collide, yaw_deg=phi)
    # headwall just before step 0: taller than the follower's rest foot height, so
    # rotating the wrong way jams the wall against the sleeved foot
    phi = -s * c.step0_deg
    a = math.radians(phi)
    _add_box(stage, f"{prim_path}/headwall",
             center=(c.track_r * math.cos(a), c.track_r * math.sin(a),
                     (c.deck_top + c.headwall_top) / 2),
             size=(c.step_rad, c.step_tan, c.headwall_top - c.deck_top),
             color=c.wall_color, collide=collide, yaw_deg=phi)
    # underside stop pin (arrests on the tower's stop posts at each end of travel)
    phi = s * c.pin_deg_s + 180.0
    a = math.radians(phi)
    _add_box(stage, f"{prim_path}/pin",
             center=(c.pin_r * math.cos(a), c.pin_r * math.sin(a), c.pin_zc),
             size=c.pin_size, color=c.pin_color, collide=collide, yaw_deg=phi)
    # 4 blue rim pegs: the intended fingertip handles
    for i in range(4):
        a = math.radians(45.0 + 90.0 * i)
        _add_cyl(stage, f"{prim_path}/peg_{i}",
                 center=(c.peg_r * math.cos(a), c.peg_r * math.sin(a),
                         c.deck_top + c.peg_h / 2),
                 radius=c.peg_rad, height=c.peg_h, color=c.peg_color, collide=collide)
    return root


def _spawn_follower(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the button follower at `prim_path`: DYNAMIC compound. Local frame:
    origin at the BOTTOM POLE OF THE BALL FOOT (so root z in the tower frame is
    directly the cam-follower height). The foot is a SPHERE on purpose: a ball
    cannot perch on a step's leading top edge — it rolls off the instant the edge
    passes under its centre, so every riser drop happens crisply."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _apply_dynamic(root, cfg.mass)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_sphere(stage, f"{prim_path}/foot", center=(0.0, 0.0, c.foot_r),
                radius=c.foot_r, color=c.foot_color, collide=collide)
    stem_z0 = c.foot_r  # bury the stem base in the ball (same body: no self-collision)
    _add_cyl(stage, f"{prim_path}/stem",
             center=(0.0, 0.0, (stem_z0 + c.stem_top) / 2),
             radius=c.stem_r, height=c.stem_top - stem_z0, color=c.stem_color,
             collide=collide)
    _add_box(stage, f"{prim_path}/head",
             center=(0.0, 0.0, c.stem_top + c.head_t / 2),
             size=(c.head_w, c.head_w, c.head_t), color=c.head_color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tower" not in _SPAWNER_CACHE:

        @configclass
        class TowerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tower)
            base_size: float = 0.36
            base_t: float = 0.020
            ped_r: float = 0.055
            ped_top: float = 0.070
            post_r: float = 0.024
            post_top: float = 0.095
            cap_r: float = 0.042
            cap_bot: float = 0.095
            cap_t: float = 0.012
            sleeve_x: float = 0.110
            aperture: float = 0.034
            sleeve_t: float = 0.012
            sleeve_z0: float = 0.220
            sleeve_z1: float = 0.310
            col_in_x: float = 0.1735
            post_deg: float = 50.0
            stop_r: float = 0.150
            stop_size: tuple = (0.020, 0.024, 0.040)
            body_color: tuple = (0.35, 0.36, 0.40)
            sleeve_color: tuple = (0.42, 0.44, 0.50)
            cap_color: tuple = (0.20, 0.20, 0.22)
            stop_color: tuple = (0.55, 0.20, 0.20)
            contact_offset: float = 0.0015

        @configclass
        class DiscSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_disc)
            chirality: float = 1.0
            mass: float = 0.8
            hub_in_r: float = 0.030
            hub_rad_t: float = 0.012
            hub_bot: float = 0.070
            hub_h: float = 0.020
            track_r: float = 0.110
            deck_rad: float = 0.110
            deck_tan: float = 0.078
            deck_bot: float = 0.078
            deck_top: float = 0.090
            n_steps: int = 8
            step_pitch_deg: float = 34.0
            step0_deg: float = 17.0
            step0_top: float = 0.171
            riser: float = 0.010
            step_rad: float = 0.050
            step_tan: float = 0.066
            headwall_top: float = 0.185
            pin_deg_s: float = 136.0
            pin_r: float = 0.150
            pin_zc: float = 0.062
            pin_size: tuple = (0.016, 0.016, 0.032)
            peg_r: float = 0.148
            peg_rad: float = 0.009
            peg_h: float = 0.050
            deck_color: tuple = (0.58, 0.44, 0.24)
            step_color: tuple = (0.74, 0.64, 0.44)
            wall_color: tuple = (0.28, 0.22, 0.16)
            pin_color: tuple = (0.10, 0.10, 0.10)
            peg_color: tuple = (0.15, 0.25, 0.85)
            contact_offset: float = 0.0015

        @configclass
        class FollowerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_follower)
            mass: float = 0.15
            foot_r: float = 0.0225
            stem_r: float = 0.0145
            stem_top: float = 0.232
            head_w: float = 0.050
            head_t: float = 0.015
            foot_color: tuple = (0.20, 0.20, 0.20)
            stem_color: tuple = (0.75, 0.75, 0.78)
            head_color: tuple = (0.80, 0.08, 0.08)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["tower"] = TowerSpawnerCfg
        _SPAWNER_CACHE["disc"] = DiscSpawnerCfg
        _SPAWNER_CACHE["follower"] = FollowerSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CamPressSceneCfg(BaseCfg):
    """Config for `CamPressScene`. The load-bearing invariants — the follower can
    only descend when the staircase rotates under it, cannot be pressed through the
    wheel, cannot be extracted from the sleeve, and the wheel cannot be dismounted
    or over-rotated — are all geometric and asserted below."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    d1: float = tunable(0.025)             # depth latch 1 (>= step 3 of 10 mm risers)
    d2: float = tunable(0.045)             # depth latch 2 (>= step 5)
    d3: float = tunable(0.066)             # depth latch 3 / success depth (step 7 only)
    at_low_tol_deg: float = tunable(10.0)  # wheel parked-at-low-stop tolerance
    settle_lin: float = tunable(0.05)      # judge gate: max lin vel (m/s)
    settle_ang: float = tunable(0.30)      # judge gate: max wheel ang vel (rad/s)
    latch_slow: float = tunable(0.10)      # follower speed gate for the depth latches
    seat_xy: float = tunable(0.012)        # wheel-seated: axis offset tol (m)
    seat_z: float = tunable(0.010)         # wheel-seated: height tol (m)
    seat_up: float = tunable(0.995)        # wheel-seated: min up-vector z
    sleeve_xy: float = tunable(0.020)      # follower-in-sleeve: xy tol about the aperture,
                                           # measured at the BALL (root = ball bottom pole):
                                           # bore play + stem tilt cocks the ball up to
                                           # ~10 mm off-axis while genuinely in the bore;
                                           # a follower OUT of the bore sits >= 30 mm off

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_deg: float = tunable(180.0)        # tower yaw uniform +/- (FREE heading)
    xy_jitter: float = tunable(0.05)       # tower xy jitter (+/- m)
    random_chirality: bool = tunable(True)  # randomize which mirror wheel is mounted
    park_jitter_deg: float = tunable(4.0)  # wheel initial park angle jitter (+/- deg)

    # --- info: tower (local frame: origin on the ground under the wheel axis) --------------------
    tower_pos: tuple = info((0.0, 0.0))
    base_size: float = info(0.36)
    base_t: float = info(0.020)
    ped_r: float = info(0.055)
    ped_top: float = info(0.070)           # pedestal top = hub seat height
    post_r: float = info(0.024)
    post_top: float = info(0.095)
    cap_r: float = info(0.042)
    cap_bot: float = info(0.095)           # cap underside: limits hub lift to 5 mm
    cap_t: float = info(0.012)
    sleeve_x: float = info(0.110)          # sleeve axis over (sleeve_x, 0), local angle 0
    aperture: float = info(0.034)          # square guide bore (foot and head are bigger)
    sleeve_t: float = info(0.012)
    sleeve_z0: float = info(0.220)         # above the ball foot's rest crown
    sleeve_z1: float = info(0.310)
    col_in_x: float = info(0.1735)         # column inner face (wheel rim swings clear)
    post_deg: float = info(50.0)           # stop posts at local angles +/- this
    stop_r: float = info(0.150)
    stop_size: tuple = info((0.020, 0.024, 0.040))
    # --- info: cam wheel -------------------------------------------------------------------------
    disc_mass: float = info(0.8)
    hub_in_r: float = info(0.030)
    hub_rad_t: float = info(0.012)
    hub_bot: float = info(0.070)
    hub_h: float = info(0.020)
    track_r: float = info(0.110)           # stair track radius = sleeve_x
    deck_rad: float = info(0.110)
    deck_tan: float = info(0.078)
    deck_bot: float = info(0.078)
    deck_top: float = info(0.090)
    n_steps: int = info(8)
    step_pitch_deg: float = info(34.0)
    step0_deg: float = info(17.0)          # step k centre at s*(step0 + pitch*k)
    step0_top: float = info(0.171)         # rest height of the follower foot
    riser: float = info(0.010)
    step_rad: float = info(0.050)
    step_tan: float = info(0.066)
    headwall_top: float = info(0.185)      # > foot rest height, < sleeve_z0
    pin_deg_s: float = info(136.0)         # pin at disc angle s*this + 180
    pin_r: float = info(0.150)
    pin_zc: float = info(0.062)
    pin_size: tuple = info((0.016, 0.016, 0.032))
    peg_r: float = info(0.148)
    peg_rad: float = info(0.009)
    peg_h: float = info(0.050)
    psi_start_deg: float = info(17.0)      # park magnitude: psi_start = -s*(this +/- jitter)
    psi_lo_deg: float = info(106.0)        # wrapped LOW-stop park: psi_lo = s*this (== -s*254;
                                           # contact offsets arrest ~4 deg before rigid geometry)
    depot: tuple = info((1.4, 1.4))        # unmounted wheel's parking spot (env frame)
    depot_z: float = info(-0.040)
    # --- info: follower --------------------------------------------------------------------------
    follower_mass: float = info(0.15)
    foot_r: float = info(0.0225)           # BALL foot (cannot perch on step edges)
    stem_r: float = info(0.0145)
    stem_top: float = info(0.232)          # head underside (local, from foot bottom)
    head_w: float = info(0.050)
    head_t: float = info(0.015)
    contact_offset: float = info(0.0015)
    # rubric weights (0.20 + 0.20 + 0.30 = 0.70 = the non-success cap)
    w1: float = info(0.20)
    w2: float = info(0.20)
    w3: float = info(0.30)

    # Derived (filled in __post_init__).
    full_depth: float = field(default=0.0, init=False)
    foot_half_deg: float = field(default=0.0, init=False)
    step_half_deg: float = field(default=0.0, init=False)
    contact_gap_deg: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.full_depth = self.riser * (self.n_steps - 1)
        self.foot_half_deg = math.degrees(math.atan(self.foot_r / self.track_r))
        self.step_half_deg = math.degrees(math.atan((self.step_tan / 2) / self.track_r))
        self.contact_gap_deg = math.degrees(
            math.atan((self.pin_size[1] / 2) / self.pin_r)
            + math.atan((self.stop_size[1] / 2) / self.stop_r))

        # Depth thresholds bracket unique steps of the physical ladder.
        assert 2 * self.riser < self.d1 <= 3 * self.riser, "d1 must mean step 3+"
        assert 4 * self.riser < self.d2 <= 5 * self.riser, "d2 must mean step 5+"
        assert 6 * self.riser < self.d3 <= self.full_depth, "d3 must mean the last step only"
        # The wheel is captured: <= 5 mm lift (less than a riser), no dismount.
        cap_gap = self.cap_bot - (self.hub_bot + self.hub_h)
        assert 0.002 < cap_gap < self.riser, "cap must stop the hub before a riser height"
        assert (self.post_top - self.hub_bot) > cap_gap + 0.010, \
            "hub cannot clear the centre post within the allowed lift"
        assert self.hub_in_r > self.post_r + 0.004, "hub must ride the post with play"
        # The follower is captured: nothing passes the aperture except the stem.
        assert 2 * self.foot_r > self.aperture + 0.008, "foot must not pass the sleeve bore"
        assert self.head_w > self.aperture + 0.008, "head must not pass the sleeve bore"
        assert 2 * self.stem_r < self.aperture - 0.004, "stem must slide freely in the bore"
        assert self.step0_top + 2 * self.foot_r < self.sleeve_z0 - 0.002, \
            "ball crown must ride below the sleeve (also blocks extraction)"
        # Head stays above the sleeve through full travel (never enters the bore).
        assert (self.step0_top - self.full_depth) + self.stem_top > self.sleeve_z1 + 0.02, \
            "head underside must stay above the sleeve at full depth"
        # Without the wheel the head catches on the sleeve top DEEPER than d3 — that
        # free-fall depth must be disarmed by the wheel-seated guard, not geometry.
        assert self.step0_top - (self.sleeve_z1 - self.stem_top) > self.d3, \
            "head-catch depth exceeds d3: latches MUST require the wheel seated"
        # Rotating hardware passes under the fixed hardware.
        assert self.headwall_top < self.sleeve_z0 - 0.004, "headwall must pass under the sleeve"
        assert self.deck_top + self.peg_h < self.sleeve_z0 - 0.004, \
            "pegs must pass under the sleeve arm plane"
        step_corner = math.hypot(self.track_r + self.step_rad / 2, self.step_tan / 2)
        deck_corner = math.hypot(self.track_r + self.deck_rad / 2, self.deck_tan / 2)
        assert max(step_corner, deck_corner, self.peg_r + self.peg_rad,
                   self.pin_r + self.pin_size[0] / 2) < self.col_in_x - 0.002, \
            "every rotating corner must clear the column's inner face"
        assert self.deck_bot > self.base_t + self.stop_size[2] + 0.004, \
            "deck must pass over the stop posts"
        assert self.pin_zc - self.pin_size[2] / 2 < self.base_t + self.stop_size[2] - 0.010, \
            "pin must overlap the stop posts by >= 10 mm in z"
        assert self.pin_zc + self.pin_size[2] / 2 <= self.deck_bot, "pin must hang under the deck"
        # Travel bookkeeping: the pin arrests the wheel with the foot on step 7.
        arrest_mag = (self.pin_deg_s + 180.0) - self.post_deg - self.contact_gap_deg
        assert abs((360.0 - self.psi_lo_deg) - arrest_mag) < self.at_low_tol_deg / 2, \
            "declared low park must match the pin/post arrest angle"
        step7_c = self.step0_deg + self.step_pitch_deg * (self.n_steps - 1)
        # Ball-foot contact bookkeeping: the ball leaves step 6 the instant its
        # trailing edge passes under the ball CENTRE, i.e. well before arrest even
        # with a few degrees of pin/post scatter; and at arrest the contact point
        # (directly under the centre) is still on step 7, clear of its far edge.
        step6_edge = (self.step0_deg + self.step_pitch_deg * (self.n_steps - 2)) \
            + self.step_half_deg
        assert arrest_mag - 4.0 > step6_edge + 2.0, \
            "ball must drop off step 6 before any arrest-angle scatter"
        assert arrest_mag < step7_c + self.step_half_deg - 2.0, \
            "ball contact point must stay on step 7 at arrest"
        # Reverse rotation is blocked twice, without contact at spawn.
        back_gap = (360.0 - self.post_deg) - ((self.pin_deg_s + 180.0) - self.psi_start_deg) \
            - self.contact_gap_deg
        assert 0.5 < back_gap < 8.0, "pin must clear the far post at spawn yet block reverse"
        hw_edge = (self.psi_start_deg + self.step0_deg) - self.step_half_deg  # headwall near edge
        assert hw_edge - self.foot_half_deg > self.park_jitter_deg + 1.0, \
            "headwall must clear the foot at any jittered park"
        # Foot centred on step 0 at park, within jitter.
        assert self.park_jitter_deg + self.foot_half_deg < self.step_half_deg, \
            "foot must start fully on step 0"
        # Stair track is the sleeve axis.
        assert abs(self.track_r - self.sleeve_x) < 1e-9


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


def _qconj(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _wrap_deg(a: torch.Tensor) -> torch.Tensor:
    return (a + 180.0) % 360.0 - 180.0


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("cam_press")
class CamPressScene(BaseScene):
    cfg: CamPressSceneCfg

    def __init__(self, cfg: CamPressSceneCfg | None = None) -> None:
        super().__init__(cfg or CamPressSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        tower_spawn = cls["tower"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            base_size=c.base_size, base_t=c.base_t, ped_r=c.ped_r, ped_top=c.ped_top,
            post_r=c.post_r, post_top=c.post_top, cap_r=c.cap_r, cap_bot=c.cap_bot,
            cap_t=c.cap_t, sleeve_x=c.sleeve_x, aperture=c.aperture,
            sleeve_t=c.sleeve_t, sleeve_z0=c.sleeve_z0, sleeve_z1=c.sleeve_z1,
            col_in_x=c.col_in_x, post_deg=c.post_deg, stop_r=c.stop_r,
            stop_size=c.stop_size, contact_offset=c.contact_offset)

        def disc_spawn(s: float):
            return cls["disc"](
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                chirality=s, mass=c.disc_mass, hub_in_r=c.hub_in_r,
                hub_rad_t=c.hub_rad_t, hub_bot=c.hub_bot, hub_h=c.hub_h,
                track_r=c.track_r, deck_rad=c.deck_rad, deck_tan=c.deck_tan,
                deck_bot=c.deck_bot, deck_top=c.deck_top, n_steps=c.n_steps,
                step_pitch_deg=c.step_pitch_deg, step0_deg=c.step0_deg,
                step0_top=c.step0_top, riser=c.riser, step_rad=c.step_rad,
                step_tan=c.step_tan, headwall_top=c.headwall_top,
                pin_deg_s=c.pin_deg_s, pin_r=c.pin_r, pin_zc=c.pin_zc,
                pin_size=c.pin_size, peg_r=c.peg_r, peg_rad=c.peg_rad, peg_h=c.peg_h,
                contact_offset=c.contact_offset)

        follower_spawn = cls["follower"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass=c.follower_mass, foot_r=c.foot_r, stem_r=c.stem_r,
            stem_top=c.stem_top, head_w=c.head_w, head_t=c.head_t,
            contact_offset=c.contact_offset)

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "tower": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tower", spawn=tower_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tower_pos[0], c.tower_pos[1], 0.0))),
            "disc_a": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/DiscA", spawn=disc_spawn(1.0),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.4, 1.4, 0.2))),
            "disc_b": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/DiscB", spawn=disc_spawn(-1.0),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.4, -1.4, 0.2))),
            "follower": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Follower", spawn=follower_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.4, 1.4, 0.05))),
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
        self.tower: RigidObject = env.iscene["tower"]
        self.discs: dict[int, RigidObject] = {
            1: env.iscene["disc_a"], -1: env.iscene["disc_b"]}
        self.follower: RigidObject = env.iscene["follower"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke)
        self.chir = torch.ones(n, dtype=torch.long, device=dev)     # +1 -> disc_a mounted
        self.psi_start = torch.zeros(n, device=dev)                 # park angle (deg)
        self.tower_yaw = torch.zeros(n, device=dev)
        # latches (partial credit survives regressions; success is judged live)
        self._l1 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l2 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l3 = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the tower (free yaw + xy jitter), pick a chirality,
        MOUNT that mirror wheel on the pedestal at a jittered park angle, depot the
        other wheel far away, drop the follower into the sleeve onto step 0, clear
        latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        self.tower_yaw[env_ids] = yaw
        q_yaw = _qz(yaw)
        tp = torch.zeros(m, 3, device=dev)
        tp[:, 0] = c.tower_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        tp[:, 1] = c.tower_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = tp + origin
        st[:, 3:7] = q_yaw
        self.tower.write_root_state_to_sim(st, env_ids)

        s = torch.where(torch.rand(m, device=dev) < 0.5,
                        torch.ones(m, dtype=torch.long, device=dev),
                        -torch.ones(m, dtype=torch.long, device=dev)) \
            if c.random_chirality else torch.ones(m, dtype=torch.long, device=dev)
        self.chir[env_ids] = s
        jit = (torch.rand(m, device=dev) * 2 - 1) * c.park_jitter_deg
        psi0 = -s.float() * (c.psi_start_deg + jit)
        self.psi_start[env_ids] = psi0

        for sign, _disc in self.discs.items():
            active = s == sign
            st = torch.zeros(m, 13, device=dev)
            # mounted: on the tower axis, 1 mm proud, yawed to the park angle
            st[:, 0:3] = tp + origin
            st[:, 2] += 0.001
            st[:, 3:7] = _qz(yaw + torch.deg2rad(psi0))
            # depot: flat on the ground far away
            dp = torch.zeros(m, 13, device=dev)
            dp[:, 0] = c.depot[0]
            dp[:, 1] = float(sign) * c.depot[1]
            dp[:, 2] = c.depot_z
            dp[:, 3] = 1.0
            dp[:, 0:3] += origin
            out = torch.where(active.unsqueeze(-1), st, dp)
            _disc.write_root_state_to_sim(out, env_ids)

        # follower: dropped 2 mm above step 0, stem in the sleeve bore
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = tp[:, 0] + c.sleeve_x * cy
        st[:, 1] = tp[:, 1] + c.sleeve_x * sy
        st[:, 2] = c.step0_top + 0.002
        st[:, 3:7] = q_yaw
        st[:, 0:3] += origin
        self.follower.write_root_state_to_sim(st, env_ids)

        self._l1[env_ids] = False
        self._l2[env_ids] = False
        self._l3[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "tower": self.tower.data.root_state_w[env_ids].clone(),
            "discs": {k: d.data.root_state_w[env_ids].clone()
                      for k, d in self.discs.items()},
            "follower": self.follower.data.root_state_w[env_ids].clone(),
            "chir": self.chir[env_ids].clone(),
            "psi_start": self.psi_start[env_ids].clone(),
            "tower_yaw": self.tower_yaw[env_ids].clone(),
            "l1": self._l1[env_ids].clone(), "l2": self._l2[env_ids].clone(),
            "l3": self._l3[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.tower.write_root_state_to_sim(state["tower"], env_ids)
        for k, d in self.discs.items():
            d.write_root_state_to_sim(state["discs"][k], env_ids)
        self.follower.write_root_state_to_sim(state["follower"], env_ids)
        self.chir[env_ids] = state["chir"]
        self.psi_start[env_ids] = state["psi_start"]
        self.tower_yaw[env_ids] = state["tower_yaw"]
        self._l1[env_ids] = state["l1"]
        self._l2[env_ids] = state["l2"]
        self._l3[env_ids] = state["l3"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A PRESS MACHINE stands on a base plate: a red PUSH BUTTON rides in a "
            f"square guide sleeve, its red cap sticking up in plain view. The button "
            f"CANNOT be pressed directly — its foot rests on the TOP STEP of a "
            f"spiral staircase cut into a heavy CAM WHEEL that turns on the "
            f"machine's centre post, so pushing on the cap just loads the wheel "
            f"against its pedestal. Four BLUE PEGS stand on the wheel's rim: "
            f"pushing a peg sideways turns the wheel. The staircase descends in "
            f"only ONE direction (which direction is mirrored on some machines — "
            f"look at the staircase); turning the wrong way jams a tall headwall "
            f"against the button's foot and an underside pin against a stop post "
            f"almost immediately. Turning the RIGHT way carries the steps under the "
            f"button foot so the button drops one {c.riser * 1000:.0f} mm riser per "
            f"{c.step_pitch_deg:.0f} deg of rotation — {c.n_steps} steps, "
            f"{c.full_depth * 1000:.0f} mm of button travel in about "
            f"{c.step_pitch_deg * (c.n_steps - 1):.0f} deg — until the pin arrests "
            f"on the other stop post with the foot on the LOWEST step. The wheel is "
            f"captive (a cap on the centre post stops it from lifting) and the "
            f"button is captive in its sleeve (foot and cap are both wider than the "
            f"bore). The machine's position, heading, staircase handedness and "
            f"parked wheel angle vary every episode; read the machine, do not "
            f"memorize it.\n"
            f"Goal: get the red button FULLY DOWN and leave it there — rotate the "
            f"cam wheel by its pegs in the descending direction until the wheel "
            f"arrests at its low stop and the button has dropped its full "
            f"{c.full_depth * 1000:.0f} mm of travel. Final state: wheel parked at "
            f"the low stop, still seated on its pedestal, button fully depressed in "
            f"its sleeve, everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Depress the red button fully by rotating the cam wheel: push the blue "
            "rim pegs to turn the wheel in the staircase's descending direction "
            "until it arrests at the low stop and the button is all the way down."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _tower_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.tower.data.root_quat_w,
                                  pos_w - self.tower.data.root_pos_w)

    def _disc_state(self):
        """Per-env root state of the ACTIVE (mounted-chirality) wheel."""
        a, b = self.discs[1].data, self.discs[-1].data
        m = (self.chir > 0).unsqueeze(-1)
        pos = torch.where(m, a.root_pos_w, b.root_pos_w)
        quat = torch.where(m, a.root_quat_w, b.root_quat_w)
        lin = torch.where(m, a.root_lin_vel_w, b.root_lin_vel_w)
        ang = torch.where(m, a.root_ang_vel_w, b.root_ang_vel_w)
        return pos, quat, lin, ang

    def psi_deg(self) -> torch.Tensor:
        """(N,) active wheel's yaw RELATIVE to the tower, degrees in (-180, 180]."""
        _, quat, _, _ = self._disc_state()
        qr = _qmul(_qconj(self.tower.data.root_quat_w), quat)
        w, x, y, z = qr.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return torch.rad2deg(yaw)

    def depth(self) -> torch.Tensor:
        """(N,) button travel: rest foot height minus current foot height (tower frame)."""
        loc = self._tower_local(self.follower.data.root_pos_w)
        return self.cfg.step0_top - loc[:, 2]

    def disc_seated(self) -> torch.Tensor:
        """(N,) bool: active wheel upright on the pedestal, hub on the centre post."""
        c = self.cfg
        pos, quat, _, _ = self._disc_state()
        loc = self._tower_local(pos)
        w, x, y, z = quat.unbind(-1)
        up_z = 1 - 2 * (x * x + y * y)
        return (loc[:, 0:2].norm(dim=-1) < c.seat_xy) \
            & (loc[:, 2].abs() < c.seat_z) & (up_z > c.seat_up)

    def in_sleeve(self) -> torch.Tensor:
        """(N,) bool: follower stem riding in the guide bore (tower frame xy)."""
        c = self.cfg
        loc = self._tower_local(self.follower.data.root_pos_w)
        dx = loc[:, 0] - c.sleeve_x
        return (dx * dx + loc[:, 1] * loc[:, 1]).sqrt() < c.sleeve_xy

    def at_low(self) -> torch.Tensor:
        """(N,) bool: active wheel parked at its LOW stop (wrap-aware)."""
        c = self.cfg
        tgt = self.chir.float() * c.psi_lo_deg
        return _wrap_deg(self.psi_deg() - tgt).abs() < c.at_low_tol_deg

    def settled(self) -> torch.Tensor:
        """(N,) bool: follower and active wheel below the velocity gates."""
        c = self.cfg
        _, _, lin, ang = self._disc_state()
        fl = self.follower.data.root_lin_vel_w.norm(dim=-1)
        return (fl < c.settle_lin) & (lin.norm(dim=-1) < c.settle_lin) \
            & (ang.norm(dim=-1) < c.settle_ang)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.follower.data.root_pos_w,
                         self.discs[1].data.root_pos_w,
                         self.discs[-1].data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        d = self.depth()
        slow = self.follower.data.root_lin_vel_w.norm(dim=-1) < c.latch_slow
        ok = self.disc_seated() & self.in_sleeve() & slow & self._finite()
        self._l1 |= (d >= c.d1) & ok
        self._l2 |= (d >= c.d2) & ok
        self._l3 |= (d >= c.d3) & ok

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: button at full depth WITH the wheel parked at its low stop,
        seated on the pedestal, follower in the sleeve, everything settled and
        finite. All clauses are live: the only physical route to this state is the
        full one-way stair descent, because the follower cannot be pressed through
        the wheel, extracted from the sleeve, or latched deep without the wheel
        seated under it."""
        self._update_latches()
        return (self.depth() >= self.cfg.d3) & self.at_low() & self.disc_seated() \
            & self.in_sleeve() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20/0.20/0.30 for ever reaching 25/45/66 mm of
        guarded button travel (latched — risers make regression physically
        impossible anyway), capped at 0.70; exactly 1.0 iff success() holds live.
        Doing nothing scores ~0; pressing the button (the seed's whole skill)
        scores ~0; rotating the wrong way scores ~0."""
        c = self.cfg
        self._update_latches()
        base = (c.w1 * self._l1.float() + c.w2 * self._l2.float()
                + c.w3 * self._l3.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="cam_press", robot="null"))
