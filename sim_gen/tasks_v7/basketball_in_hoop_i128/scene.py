"""CraterRunScene — roll the oversized ball UP the switchback ramp into the elevated
crater basin (sim_gen task `basketball_in_hoop_i128`).

Derived from rlbench/basketball_in_hoop, but STRATEGICALLY different: the seed is a
grasp-and-drop — pick the ball off its stand, carry it above the hoop and let gravity
finish the job (one grasp, one transport, one release; gravity is the DELIVERY
mechanism and the judged region sits directly below the release point). Here the
delivery-by-gravity plan is dead on arrival and gravity is the ADVERSARY instead:

- the ball is 90 mm across — wider than the Franka parallel jaw span (80 mm, asserted
  in `__post_init__`) — so it cannot be grasped, carried or dropped at all; the only
  contact strategy that exists is PUSHING;
- the goal is a walled CRATER BASIN raised ~0.18 m above the floor at the top of a
  switchback ramp: a lower incline, a flat 180-degree turn pocket, an upper incline
  running back the other way, and a short summit landing ending in a low retaining
  lip in front of the sunken basin;
- the ball therefore has to be CONVEYED the whole way by sustained pushing — up the
  first incline, U-turned through the pocket, up the second incline, along the
  landing and over the lip — with gravity pulling it back down the whole climb: a
  released ball mid-incline rolls straight back to the foot bay (proved in smoke).

A solver needs a different PLAN from the seed (continuous closed-loop conveyance
along a constrained 3D path against gravity, with a direction reversal, instead of
one pick-carry-release) and a different CODE STRUCTURE (path-following push control
instead of a grasp pose + a release point). The ramp's chirality flips randomly per
episode (the turn is at the +x or the -x end of the lower leg), so the route itself
must be read from the scene, not memorized.

success(): the ball rests INSIDE the crater basin — canonical-frame containment
window that by construction accepts every physically-possible resting pose inside
the basin and rejects on-the-landing, on-the-lip, rim-perch and on-the-floor poses
(asserted in `__post_init__`) — with the ball settled.

score() is graded and latched (credit never evaporates): 0.15 for each demonstrated
stage of the climb — upper half of incline A, the turn pocket, upper half of incline
B, the summit landing (cap 0.60) — and 1.0 iff success(). The null policy scores ~0
(the ball spawns in the flat foot bay at the ramp's base).

Per-episode randomization (readback-verified in smoke): ramp chirality (a mirrored
twin structure swaps into the workspace), structure yaw + xy jitter, ball start
position jitter in the foot bay. Assets are fully procedural compound spawners
(boxes + one sphere; one rigid body each; both structures are kinematic). Heavy
imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
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


def _material(stage, path: str, static: float = 0.6, dynamic: float = 0.5):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, contact_offset: float, material) -> None:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float, material=None,
         orient=None) -> None:
    """Author one box child prim (translate -> orient -> scale, authored once — the
    duplicate-xformOp trap is avoided by never re-authoring an existing prim's ops)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _sphere(stage, path: str, radius, center, color, contact_offset: float,
            material=None) -> None:
    """One sphere child prim."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Sphere.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -radius),
                          Gf.Vec3f(radius, radius, radius)])
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float,
                kinematic: bool = False):
    """Author one rigid-body root Xform with the standard physics armor (zero
    sleep/stabilization thresholds: a sleeping body silently ignores applied wrenches,
    which the solve/smoke force probes depend on; velocity iterations 4 — the
    sphere-on-face phantom-creep fix)."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return root, pxrb


def _ramp_boxes(c: Any, m: float) -> list[tuple]:
    """The full box list of one ramp structure, canonical geometry mirrored by
    `m` in y (m=+1 canonical, m=-1 mirrored twin). Each entry:
    (name, size, center, color, orient-or-None). All numbers derive from the cfg's
    measured constants so the rubric windows and the geometry cannot drift apart."""
    a = math.atan(c.grade)  # incline angle
    ca2, sa2 = math.cos(a / 2), math.sin(a / 2)
    ls = 2 * c.leg_half / math.cos(a) + 0.012  # slope-length of an incline slab (+lap)
    slab_t, wall_t = 0.04, c.wall_t
    lane_w = 2 * (c.lane_half + wall_t)  # floor width under walls too

    green = (0.13, 0.45, 0.20)
    blue = (0.15, 0.30, 0.65)
    grey = (0.55, 0.57, 0.60)
    red = (0.80, 0.15, 0.12)
    white = (0.92, 0.92, 0.90)

    yb = c.laneB_y  # upper-lane center y (canonical)
    wa = c.lane_half + wall_t / 2  # lane-A wall center |y|
    wb_lo, wb_hi = yb - wa, yb + wa  # lane-B wall center y

    boxes: list[tuple] = []
    # --- foot bay (flat, low): deck + two side walls + end stop -------------------------
    fx = (c.foot_x0 + c.foot_x1) / 2
    boxes += [
        ("foot_deck", (c.foot_x1 - c.foot_x0, lane_w, c.deck_h),
         (fx, 0.0, c.deck_h / 2), blue, None),
        ("foot_wall_p", (c.foot_x1 - c.foot_x0, wall_t, c.deck_h + c.wall_h),
         (fx, +wa, (c.deck_h + c.wall_h) / 2), grey, None),
        ("foot_wall_n", (c.foot_x1 - c.foot_x0, wall_t, c.deck_h + c.wall_h),
         (fx, -wa, (c.deck_h + c.wall_h) / 2), grey, None),
        ("foot_end", (wall_t, lane_w, c.deck_h + c.wall_h),
         (c.foot_x0 - wall_t / 2, 0.0, (c.deck_h + c.wall_h) / 2), grey, None),
    ]
    # --- incline A (lane y=0, rises toward +x): slab + two walls ------------------------
    # surface: z = legA_mid + grade*x for x in [-leg_half, +leg_half]
    legA_mid = (c.deck_h + c.pocket_h) / 2
    nx, nz = -math.sin(a), math.cos(a)  # unit normal of a +x-rising slope
    qA = (ca2, 0.0, -sa2, 0.0)  # R_y(-a): local +x maps to (cos a, 0, +sin a)
    boxes += [
        # slab center = surface midpoint - normal * slab_t/2 (top face on the surface)
        ("legA_slab", (ls, lane_w, slab_t),
         (-nx * slab_t / 2, 0.0, legA_mid - nz * slab_t / 2), green, qA),
        # wall center = surface midpoint + normal * wall_h/2 (base on the surface)
        ("legA_wall_p", (ls, wall_t, c.wall_h),
         (nx * c.wall_h / 2, +wa, legA_mid + nz * c.wall_h / 2), grey, qA),
        ("legA_wall_n", (ls, wall_t, c.wall_h),
         (nx * c.wall_h / 2, -wa, legA_mid + nz * c.wall_h / 2), grey, qA),
    ]
    # --- turn pocket (flat, mid height): deck + end wall + two side walls ---------------
    px_mid = (c.pocket_x0 + c.pocket_x1) / 2
    pocket_w = wb_hi - (-wa) + wall_t  # spans lane A's -y wall to lane B's +y wall
    pocket_yc = (wb_hi + -wa) / 2
    pk_top = c.pocket_h + c.wall_h
    boxes += [
        ("pocket_deck", (c.pocket_x1 - c.pocket_x0, pocket_w, 0.04),
         (px_mid, pocket_yc, c.pocket_h - 0.02), green, None),
        ("pocket_end", (wall_t, pocket_w, pk_top),
         (c.pocket_x1 + wall_t / 2, pocket_yc, pk_top / 2), grey, None),
        ("pocket_wall_a", (c.pocket_x1 - c.pocket_x0 + 2 * wall_t, wall_t, pk_top),
         (px_mid, -wa, pk_top / 2), grey, None),
        ("pocket_wall_b", (c.pocket_x1 - c.pocket_x0 + 2 * wall_t, wall_t, pk_top),
         (px_mid, wb_hi, pk_top / 2), grey, None),
    ]
    # --- incline B (lane y=laneB_y, rises toward -x): slab + two walls ------------------
    legB_mid = (c.pocket_h + c.land_h) / 2
    qB = (ca2, 0.0, +sa2, 0.0)  # R_y(+a): local +x maps to (cos a, 0, -sin a)
    # incline B's upward normal is the y-mirror-in-x of incline A's: (-nx, 0, nz)
    boxes += [
        ("legB_slab", (ls, lane_w, slab_t),
         (+nx * slab_t / 2, yb, legB_mid - nz * slab_t / 2), green, qB),
        ("legB_wall_lo", (ls, wall_t, c.wall_h),
         (-nx * c.wall_h / 2, wb_lo, legB_mid + nz * c.wall_h / 2), grey, qB),
        ("legB_wall_hi", (ls, wall_t, c.wall_h),
         (-nx * c.wall_h / 2, wb_hi, legB_mid + nz * c.wall_h / 2), grey, qB),
    ]
    # --- summit landing (flat, high): deck + two side walls -----------------------------
    lx = (c.land_x0 + c.land_x1) / 2
    boxes += [
        ("land_deck", (c.land_x1 - c.land_x0, lane_w, 0.06),
         (lx, yb, c.land_h - 0.03), green, None),
        ("land_wall_lo", (c.land_x1 - c.land_x0, wall_t, c.wall_h),
         (lx, wb_lo, c.land_h + c.wall_h / 2), grey, None),
        ("land_wall_hi", (c.land_x1 - c.land_x0, wall_t, c.wall_h),
         (lx, wb_hi, c.land_h + c.wall_h / 2), grey, None),
    ]
    # --- crater basin (sunken, elevated): floor + far wall + side walls + entry lip -----
    bx0 = c.basin_cx - c.basin_half_x  # inner faces
    bx1 = c.basin_cx + c.basin_half_x
    floor_x0, floor_x1 = bx0 - wall_t, c.land_x0 + 0.01
    fxc = (floor_x0 + floor_x1) / 2
    boxes += [
        ("basin_floor", (floor_x1 - floor_x0, lane_w + 0.02, c.basin_floor_h),
         (fxc, yb, c.basin_floor_h / 2), white, None),
        ("basin_far", (wall_t, lane_w + 0.02, c.rim_top),
         (bx0 - wall_t / 2, yb, c.rim_top / 2), red, None),
        ("basin_side_lo", (floor_x1 - floor_x0 + wall_t, wall_t, c.rim_top),
         (fxc, wb_lo, c.rim_top / 2), red, None),
        ("basin_side_hi", (floor_x1 - floor_x0 + wall_t, wall_t, c.rim_top),
         (fxc, wb_hi, c.rim_top / 2), red, None),
        ("basin_lip", (0.02, 2 * c.lane_half, c.lip_top - c.basin_floor_h),
         (bx1 + 0.01, yb, (c.lip_top + c.basin_floor_h) / 2), red, None),
    ]
    # mirror in y (boxes are y-symmetric; pitch-about-y orients are mirror-invariant)
    return [(nm, sz, (cx, m * cy, cz), col, q) for nm, sz, (cx, cy, cz), col, q in boxes]


def _spawn_ramp(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One KINEMATIC switchback-ramp structure (foot bay -> incline A -> turn pocket ->
    incline B -> summit landing -> lip -> crater basin), chirality by `cfg.mirror`."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 30.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    for name, size, center, color, orient in _ramp_boxes(cfg, cfg.mirror):
        _box(stage, f"{prim_path}/{name}", size, center, color, cfg.contact_offset,
             material=mat, orient=orient)
    return root


def _spawn_ball(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The oversized ball: one orange sphere (wider than the Franka jaw span —
    asserted in the scene cfg — so pushing is the only contact strategy)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.1)
    pxrb.CreateAngularDampingAttr(0.3)
    mat = _material(stage, f"{prim_path}/phys_mat")
    _sphere(stage, f"{prim_path}/ball", cfg.radius, (0.0, 0.0, 0.0),
            (0.95, 0.45, 0.10), cfg.contact_offset, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "ramp" not in _SPAWNER_CACHE:

        @configclass
        class RampSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ramp)
            mirror: float = 1.0
            contact_offset: float = 0.002
            # geometry constants copied from the scene cfg at assets() time
            grade: float = 0.2
            leg_half: float = 0.20
            lane_half: float = 0.07
            wall_t: float = 0.02
            wall_h: float = 0.07
            deck_h: float = 0.03
            pocket_h: float = 0.11
            land_h: float = 0.19
            laneB_y: float = 0.16
            foot_x0: float = -0.36
            foot_x1: float = -0.20
            pocket_x0: float = 0.20
            pocket_x1: float = 0.37
            land_x0: float = -0.355
            land_x1: float = -0.20
            basin_cx: float = -0.43
            basin_half_x: float = 0.075
            basin_floor_h: float = 0.13
            rim_top: float = 0.25
            lip_top: float = 0.198

        @configclass
        class BallSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ball)
            mass: float = 0.25
            radius: float = 0.045
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(ramp=RampSpawnerCfg, ball=BallSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CraterRunSceneCfg(BaseCfg):
    """Config for `CraterRunScene`. The push-only premise and every rubric window's
    honesty are asserted in `__post_init__`: the ball cannot be grasped, the basin
    window accepts every physically-in-basin resting pose while rejecting
    landing/lip/rim/floor poses, the lip retains a settled ball, and the channel
    walls retain a rolling ball."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    basin_xy_tol: float = tunable(0.045)  # |ball - basin center| per canonical axis
    basin_z_lo: float = tunable(0.155)  # ball-center height window inside the basin:
    basin_z_hi: float = tunable(0.205)  # rest = 0.175; landing rest reads 0.235
    settle_lin: float = tunable(0.05)  # max ball |lin vel| at judging (m/s)
    settle_ang: float = tunable(1.5)  # max ball |ang vel| at judging (rad/s)
    on_track_z: float = tunable(0.05)  # stage windows: |ball_z - (floor+r)| below this

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    yaw_jitter_deg: float = tunable(12.0)  # +- structure yaw jitter
    pos_jitter: float = tunable(0.03)  # +- structure xy jitter
    ball_x0: float = tunable(-0.31)  # ball start window in the foot bay (canonical)
    ball_x1: float = tunable(-0.24)
    ball_y_jitter: float = tunable(0.025)

    # --- info: structure (the geometry the spawner authors) ----------------------------------
    grade: float = info(0.2)  # incline rise/run (11.3 deg)
    leg_half: float = info(0.20)  # incline half-run (x in [-0.20, 0.20])
    lane_half: float = info(0.07)  # channel half-width (wall inner faces)
    wall_t: float = info(0.02)
    wall_h: float = info(0.07)  # wall height above the local floor
    deck_h: float = info(0.03)  # foot-bay floor top
    pocket_h: float = info(0.11)  # turn-pocket floor top
    land_h: float = info(0.19)  # summit-landing floor top
    laneB_y: float = info(0.16)  # upper-lane center y (canonical; mirrored by m)
    foot_x0: float = info(-0.36)
    foot_x1: float = info(-0.20)
    pocket_x0: float = info(0.20)
    pocket_x1: float = info(0.37)  # pocket interior x range
    land_x0: float = info(-0.355)
    land_x1: float = info(-0.20)
    basin_cx: float = info(-0.43)  # basin center x (canonical); center y = laneB_y
    basin_half_x: float = info(0.075)  # basin inner half-extent in x
    basin_floor_h: float = info(0.13)  # basin floor top
    rim_top: float = info(0.25)  # far/side rim wall top
    lip_top: float = info(0.198)  # entry lip top (8 mm proud of the landing)
    ball_r: float = info(0.045)
    ball_mass: float = info(0.25)
    jaw_span: float = info(0.080)  # Franka parallel-jaw max opening
    park_xy: tuple = info((2.2, 2.2))  # depot for the inactive chirality twin
    contact_offset: float = info(0.002)

    def __post_init__(self) -> None:
        c = self
        r = c.ball_r
        # -- the push-only premise is real: the ball cannot be grasped --
        assert 2 * r > c.jaw_span + 0.008, "ball must exceed the Franka jaw span"
        # -- the basin window is honest by construction --
        rest_z = c.basin_floor_h + r  # settled in-basin ball center height
        assert c.basin_z_lo < rest_z < c.basin_z_hi, "in-basin rest must be inside"
        assert c.basin_half_x - r <= c.basin_xy_tol, \
            "every physically-in-basin x must be inside the window"
        assert c.lane_half - r <= c.basin_xy_tol, \
            "every physically-in-basin y must be inside the window"
        assert c.land_h + r > c.basin_z_hi + 0.02, "a landing rest must read above"
        assert c.lip_top + r > c.basin_z_hi + 0.02, "a lip-top rest must read above"
        assert c.rim_top + r > c.basin_z_hi + 0.02, "a rim perch must read above"
        assert r < c.basin_z_lo, "a floor-level ball must read below"
        landing_rest_x = c.basin_cx + c.basin_half_x + 0.02 + r  # against the lip
        assert abs(landing_rest_x - c.basin_cx) > c.basin_xy_tol + 0.01, \
            "a ball resting against the lip on the landing must be outside in x"
        # -- the lip retains a settled in-basin ball, the walls retain a rolling one --
        assert c.lip_top - rest_z > 0.02, "lip must sit above the settled ball center"
        assert c.wall_h > r + 0.02, "channel walls must top the rolling ball's center"
        assert c.rim_top - rest_z > 0.05, "rim must comfortably retain the ball"
        # -- the channel and the turn actually pass the ball --
        assert 2 * c.lane_half > 2 * r + 0.04, "channel must pass the ball freely"
        assert c.pocket_x1 - c.pocket_x0 > 2 * r + 0.06, "turn pocket depth too tight"
        assert c.laneB_y - 2 * c.lane_half - c.wall_t > -1e-9, "lanes must not overlap"
        # -- gravity is adversarial: the incline is steep enough to reject a parked ball
        assert c.grade > 0.12, "incline must roll a released ball back down"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("crater_run")
class CraterRunScene(BaseScene):
    cfg: CraterRunSceneCfg

    def __init__(self, cfg: CraterRunSceneCfg | None = None) -> None:
        super().__init__(cfg or CraterRunSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        geo = {k: getattr(c, k) for k in (
            "grade", "leg_half", "lane_half", "wall_t", "wall_h", "deck_h",
            "pocket_h", "land_h", "laneB_y", "foot_x0", "foot_x1", "pocket_x0",
            "pocket_x1", "land_x0", "land_x1", "basin_cx", "basin_half_x",
            "basin_floor_h", "rim_top", "lip_top")}

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
        }
        for name, mirror, init_xy in (
                ("ramp_r", 1.0, (0.0, 0.0)),
                ("ramp_l", -1.0, c.park_xy)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.capitalize(),
                spawn=sp["ramp"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mirror=mirror, contact_offset=c.contact_offset, **geo),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(init_xy[0], init_xy[1], 0.0)),
            )
        out["ball"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Ball",
            spawn=sp["ball"](
                mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass=c.ball_mass, radius=c.ball_r, contact_offset=c.contact_offset),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=((c.ball_x0 + c.ball_x1) / 2, 0.0, c.deck_h + c.ball_r + 0.003)),
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
                # external-wrench plant recipe: without this the solve/smoke force
                # servos on the ball are under-applied across TGS iterations
                "enable_external_forces_every_iteration": True,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.ramp_r: RigidObject = env.iscene["ramp_r"]
        self.ramp_l: RigidObject = env.iscene["ramp_l"]
        self.ball: RigidObject = env.iscene["ball"]
        self.env_origins = env.iscene.env_origins
        # per-env chirality: +1 -> ramp_r active (turn on the +x end, upper lane +y),
        # -1 -> ramp_l active (mirrored)
        self.mirror = torch.ones(n, device=dev)
        # progress latches (post_step): upper half of incline A, turn pocket, upper
        # half of incline B, summit landing
        self.climbA_latch = torch.zeros(n, device=dev)
        self.turn_latch = torch.zeros(n, device=dev)
        self.climbB_latch = torch.zeros(n, device=dev)
        self.land_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample chirality (torch.rand comparison — the first-randint
        degeneracy), pose the active structure at the workspace with yaw + xy jitter,
        park the twin in the depot, drop the ball in the foot bay with start jitter;
        latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        self.mirror[env_ids] = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        mir = self.mirror[env_ids]

        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_jitter_deg) / 2
        jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.pos_jitter
        active = torch.zeros(m, 13, device=dev)
        active[:, 0:2] = jit
        active[:, 3] = torch.cos(half)
        active[:, 6] = torch.sin(half)
        active[:, 0:3] += origin
        parked = torch.zeros(m, 13, device=dev)
        parked[:, 0] = c.park_xy[0]
        parked[:, 1] = c.park_xy[1]
        parked[:, 3] = 1.0
        parked[:, 0:3] += origin
        r_active = mir > 0
        self.ramp_r.write_root_state_to_sim(
            torch.where(r_active.unsqueeze(1), active, parked), env_ids)
        self.ramp_l.write_root_state_to_sim(
            torch.where(r_active.unsqueeze(1), parked, active), env_ids)

        # ball: canonical foot-bay slot -> world through the active structure pose
        bx = c.ball_x0 + torch.rand(m, device=dev) * (c.ball_x1 - c.ball_x0)
        by = (torch.rand(m, device=dev) * 2 - 1) * c.ball_y_jitter
        cy, sy = torch.cos(2 * half), torch.sin(2 * half)
        ly = mir * by  # canonical -> local (mirror)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = active[:, 0] - origin[:, 0] + cy * bx - sy * ly
        st[:, 1] = active[:, 1] - origin[:, 1] + sy * bx + cy * ly
        st[:, 2] = c.deck_h + c.ball_r + 0.003
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.ball.write_root_state_to_sim(st, env_ids)

        for latch in (self.climbA_latch, self.turn_latch, self.climbB_latch,
                      self.land_latch):
            latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"ramp_r": self.ramp_r, "ramp_l": self.ramp_l, "ball": self.ball}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in bodies.items()},
            "mirror": self.mirror[env_ids].clone(),
            "latches": torch.stack([self.climbA_latch[env_ids],
                                    self.turn_latch[env_ids],
                                    self.climbB_latch[env_ids],
                                    self.land_latch[env_ids]], dim=1),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"ramp_r": self.ramp_r, "ramp_l": self.ramp_l, "ball": self.ball}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.mirror[env_ids] = state["mirror"]
        lt = state["latches"]
        self.climbA_latch[env_ids] = lt[:, 0]
        self.turn_latch[env_ids] = lt[:, 1]
        self.climbB_latch[env_ids] = lt[:, 2]
        self.land_latch[env_ids] = lt[:, 3]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A switchback BALL RAMP stands on the floor: a walled channel that climbs "
            "in two green inclined legs connected by a flat turn pocket, ending at a "
            "raised summit. The route, start to finish: a flat BLUE FOOT BAY at floor "
            "level holds a single large ORANGE BALL "
            f"({2 * c.ball_r * 1000:.0f} mm across — wider than a parallel-jaw "
            "gripper opens, so it cannot be grasped, only rolled by pushing); the "
            "bay opens onto the LOWER GREEN INCLINE, which climbs to a flat grey-"
            "walled TURN POCKET where the channel U-turns 180 degrees into the "
            "UPPER GREEN INCLINE running back the opposite way; that climbs to a "
            "short flat SUMMIT LANDING which ends at a low RED LIP in front of the "
            "CRATER BASIN — a sunken white-floored well ringed by RED rim walls, "
            f"its floor {c.basin_floor_h * 1000:.0f} mm above the ground. The ramp's "
            "handedness is mirrored at random between episodes (the turn pocket may "
            "be at either end of the lower leg): follow the channel with your eyes "
            "from the blue bay to know which way it climbs. The grey channel walls "
            "are low enough to reach over from either side.\n"
            "Goal: roll the orange ball from the blue foot bay all the way up the "
            "ramp — up the lower incline, around the turn pocket, up the upper "
            "incline, along the summit landing — and push it over the low red lip "
            "so it drops into the crater basin; leave it resting inside. The ball "
            "rolls back downhill whenever it is left unsupported on an incline, so "
            "keep it controlled; the flat bay, turn pocket and landing are safe "
            "places to pause. A ball left anywhere on the ramp, balanced on a wall "
            "or rim, resting against the lip on the landing side, or knocked off "
            "onto the floor scores nothing at the end — only a ball at rest inside "
            "the crater basin is success."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Roll the large orange ball from the blue foot bay up the switchback "
            "ramp — up the lower incline, around the flat turn pocket, up the upper "
            "incline — then push it over the low red lip so it drops into the "
            "raised crater basin and rests inside. The ball is too wide to grasp; "
            "it must be pushed the whole way, and it must end at rest inside the "
            "basin."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _active_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos_w (N,3), quat_w (N,4)) of the per-env ACTIVE structure."""
        r_active = (self.mirror > 0).unsqueeze(1)
        pos = torch.where(r_active, self.ramp_r.data.root_pos_w,
                          self.ramp_l.data.root_pos_w)
        quat = torch.where(r_active, self.ramp_r.data.root_quat_w,
                           self.ramp_l.data.root_quat_w)
        return pos, quat

    def ball_canon(self) -> torch.Tensor:
        """(N, 3): ball center in the CANONICAL structure frame (active structure's
        local frame with y mirrored back, so one set of region windows serves both
        chiralities)."""
        from isaaclab.utils.math import quat_apply_inverse

        pos, quat = self._active_pose()
        loc = quat_apply_inverse(quat, self.ball.data.root_pos_w - pos)
        loc = loc.clone()
        loc[:, 1] = self.mirror * loc[:, 1]
        return loc

    def canon_to_world(self, canon: torch.Tensor) -> torch.Tensor:
        """(N, 3) canonical points -> world (the solve's waypoint transform)."""
        from isaaclab.utils.math import quat_apply

        pos, quat = self._active_pose()
        loc = canon.clone()
        loc[:, 1] = self.mirror * loc[:, 1]
        return pos + quat_apply(quat, loc)

    # ----- region predicates (canonical frame) ------------------------------------------------
    def _regions(self) -> dict[str, torch.Tensor]:
        c = self.cfg
        p = self.ball_canon()
        cx, cy, cz = p[:, 0], p[:, 1], p[:, 2]
        r, zt = c.ball_r, c.on_track_z
        legA = c.deck_h / 2 + c.pocket_h / 2 + c.grade * cx  # incline-A floor at cx
        legB = c.pocket_h / 2 + c.land_h / 2 - c.grade * cx  # incline-B floor at cx
        in_laneA = cy.abs() <= c.lane_half + 0.01
        in_laneB = (cy - c.laneB_y).abs() <= c.lane_half + 0.01
        return {
            "climbA": in_laneA & (cx >= 0.0) & (cx <= c.pocket_x0 + 0.01)
            & ((cz - (legA + r)).abs() <= zt),
            "turn": (cx >= c.pocket_x0 + 0.01) & (cx <= c.pocket_x1 - 0.005)
            & (cy >= -c.lane_half - 0.01) & (cy <= c.laneB_y + c.lane_half + 0.01)
            & ((cz - (c.pocket_h + r)).abs() <= zt),
            "climbB": in_laneB & (cx <= 0.0) & (cx >= c.land_x1 - 0.01)
            & ((cz - (legB + r)).abs() <= zt),
            "landing": in_laneB & (cx <= c.land_x1) & (cx >= c.land_x0 - 0.005)
            & ((cz - (c.land_h + r)).abs() <= zt),
        }

    def in_basin(self) -> torch.Tensor:
        """(N,) bool, geometric: ball center inside the crater-basin containment
        window (accepts every physically-in-basin resting pose, rejects landing /
        lip / rim / floor poses; asserted in __post_init__)."""
        c = self.cfg
        p = self.ball_canon()
        return ((p[:, 0] - c.basin_cx).abs() <= c.basin_xy_tol) \
            & ((p[:, 1] - c.laneB_y).abs() <= c.basin_xy_tol) \
            & (p[:, 2] >= c.basin_z_lo) & (p[:, 2] <= c.basin_z_hi)

    def settled(self) -> torch.Tensor:
        """(N,) bool: ball at rest."""
        c = self.cfg
        return (self.ball.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.ball.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch each demonstrated climb stage every physics substep: ball ever on
        the upper half of incline A, ever in the turn pocket, ever on the upper half
        of incline B, ever on the summit landing."""
        reg = self._regions()
        self.climbA_latch = torch.maximum(self.climbA_latch, reg["climbA"].float())
        self.turn_latch = torch.maximum(self.turn_latch, reg["turn"].float())
        self.climbB_latch = torch.maximum(self.climbB_latch, reg["climbB"].float())
        self.land_latch = torch.maximum(self.land_latch, reg["landing"].float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the ball rests inside the crater basin. The climb is physically
        necessary: the ball cannot be grasped (wider than the jaw span) and the basin
        is sunken behind rim walls at the top of the only channel that reaches it."""
        return self.in_basin() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 per latched climb stage (upper incline A, turn
        pocket, upper incline B, summit landing; cap 0.60); 1.0 iff success().
        Latched — credit never evaporates; the null policy scores ~0 (the ball
        spawns in the flat foot bay at the base of the ramp)."""
        base = (0.15 * (self.climbA_latch + self.turn_latch + self.climbB_latch
                        + self.land_latch)).clamp(0.0, 0.60)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="crater_run", robot="null", env_spacing=6.0))
