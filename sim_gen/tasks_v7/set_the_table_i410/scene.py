"""WobblyBistroScene — shore the wobbly bistro table's short leg with a wedge shim,
THEN set the table (plate + cup on their marked seats). Derived from
rlbench/set_the_table but the plan is inverted: placement is trivial — what the task
is ABOUT is diagnosing and repairing the furniture's stability first, because the
unrepaired table sheds everything you put on it.

Seed (rlbench/set_the_table): pick five utensils out of a holder and arrange them at
their designated flat spots — N independent pick-and-places, no mechanism, the table
is a passive perfect surface. Here the TABLE ITSELF is the mechanism:

- The bistro table is a free DYNAMIC body with 4 legs; ONE leg is 25 mm SHORT. Its
  centre of mass is biased slightly toward the opposite corner, so the empty table
  rests LEVEL on its three long legs with the short leg visibly HOVERING ~25 mm off
  the floor: the fault is invisible in the tabletop and visible only in that gap.
- Both marked seats (red ring = plate, pale-blue disc = cup) lie on the SHORT-LEG
  half of the top, OUTSIDE the three grounded feet's support hull. The rocking
  pivot is the hull EDGE through the two adjacent long feet's pad corners (the
  30 mm pads push it leg_w/sqrt(2) ~ 21 mm past the leg-centre diagonal); the
  moment either dish is set down on its seat, the combined table+dish CoM crosses
  that hull edge with >= 1.25x margin (asserted): the table rocks until the short
  leg lands and the top pitches to ~7.2 deg.
- The tabletop is POLISHED (slick physics material, pair-averaged with the equally
  slick dish bases): at 7.2 deg, tan(7.2) >= 2.2x the static friction (asserted), so
  the dish accelerates off the seat and off the table. Setting the table FIRST is
  the losing move — the physics-forced order is SHIM FIRST, PLACE SECOND.
- The repair: an orange RAMP WEDGE lies on the floor. Slid tip-first into the gap
  under the hovering short leg (a floor-level PUSH, no grasp needed), its ramp takes
  the leg's load as the table tries to rock: the table becomes a 4-point stance and
  the top stays level (within 1.0 deg) under load. The wedge's ramp slope
  (tan ~ 0.18) self-locks under load against friction (asserted), so the shored
  state persists with nothing holding it.
- Success is PHYSICAL state only: top level within 1.0 deg, plate settled on the red
  ring seat, cup settled on the blue disc seat (both judged in the TABLE's body
  frame, xy AND resting z), everything still. No wedge clause: with both seats
  loaded, a level top is only physically reachable with the short leg shored (the
  wedge parked ON the table as a counterweight is asserted insufficient).

So a solver needs a different PLAN (find the hovering leg, shim it from the floor,
and only then place dishes — repair-then-place instead of arrange-N-items) and
different CODE STRUCTURE (a floor-level wedge push with a level/height readback loop
plus two placements, instead of a list of pick-and-place targets).

Assets are fully procedural (compound box/cylinder spawners; child colliders of one
body never self-collide):
  - table: DYNAMIC compound — 0.34 x 0.34 x 0.02 m top slab (slick top), 4 legs
    (30 x 30 mm) at (+/-0.14, +/-0.14); legs 0.15 m long except the SHORT leg at
    body-frame corner (+0.14, +0.14): 0.125 m (25 mm stub). Explicit MassAPI CoM
    biased 6 mm (perpendicular) past the leg-centre diagonal away from the short
    corner. Seat markers are zero-collision visual discs on the top.
  - wedge: DYNAMIC orange ramp — 0.11 x 0.06 m footprint, ramp top rising 0.010 ->
    0.030 m tip-to-heel (slope 10.3 deg), built from base + pitched ramp plate +
    heel wall boxes. Pushable at floor level; too short to prop the slab.
  - plate: DYNAMIC white cylinder D60 x 14 mm (fits the 80 mm Franka jaw span).
  - cup: DYNAMIC blue cylinder D45 x 70 mm (upright tumbler; slides, never tips,
    at the fault angle: r/h_com = tan 32.7 deg >> tan 7.2 deg).

Per-episode randomization (readback-verifiable): table xy jitter + free 360 deg yaw
(so the short corner points anywhere in the world — find it by the hover gap), and
independent floor spawn slots (angle band + radial jitter + free yaw) for wedge,
plate and cup.

Rubric (0..1; stage credit latched on 30-step (0.25 s) streaks so it is earned by
SUSTAINED states — longer than the pre-tip window on an unshored table — and
survives later probes):
  0.30 * shored_ever — the wedge ever supporting the short leg: leg tip over the
                       ramp within the wedge footprint, at bearing height, top
                       level, wedge still (latched)
  0.20 * plate_ever  — plate ever settled on its seat with the top level (latched)
  0.20 * cup_ever    — cup ever settled on its seat with the top level (latched)
  1.0 iff success()  — both dishes seated + top level + all still. Cap 0.70 else.

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

FRANKA_JAW_SPAN = 0.080  # Franka parallel-jaw max opening (m) — graspability bound
G = 9.81


# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


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


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable | None,
             orient=None) -> None:
    """Author one box (collider unless collide is None: pure visual marker)."""
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
    if collide is not None:
        collide(box.GetPrim())


def _add_cyl(stage, path: str, *, center, radius, height, color,
             collide: Callable | None) -> None:
    """Author one z-axis cylinder (collider unless collide is None)."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(cyl.GetPrim())


def _slick_material(stage, root_path: str, static: float, dynamic: float):
    """Low-friction physics material (the polished tabletop / glazed dish bases).
    PhysX pair-AVERAGES friction, so both the top face and the dish bodies carry it:
    dish-on-top contact is slick-on-slick at the authored value."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, f"{root_path}/slickMat")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(static))
    api.CreateDynamicFrictionAttr(float(dynamic))
    api.CreateRestitutionAttr(0.0)
    return mat


def _bind_mat(stage, mat, *prim_paths: str) -> None:
    from pxr import UsdShade

    for p in prim_paths:
        prim = stage.GetPrimAtPath(p)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rigid_armor(root, *, mass: float, com, lin_damp: float, ang_damp: float) -> None:
    """DYNAMIC body boilerplate: explicit mass AND CoM (MassAPI mass alone leaves
    the CoM at the body origin), damping, zeroed sleep thresholds (judged for
    stillness), depenetration cap."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    mapi = UsdPhysics.MassAPI.Apply(root)
    mapi.CreateMassAttr(float(mass))
    mapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)


def _spawn_table(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the bistro table: ONE DYNAMIC compound. Origin at the TOP SLAB CENTRE.
    Top slab (slick top) + 4 legs (leg 0 at (+hx,+hy) is the SHORT one) + two
    zero-collision seat marker discs on the top face."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_armor(root, mass=c.table_mass, com=c.table_com,
                 lin_damp=0.02, ang_damp=0.05)
    collide = _make_collide(c.contact_offset)

    _add_box(stage, f"{prim_path}/top", center=(0.0, 0.0, 0.0),
             size=(c.top_side, c.top_side, c.top_t), color=c.top_color,
             collide=collide)
    hx = hy = c.leg_xy
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        length = c.leg_len - (c.stub if i == 0 else 0.0)
        zc = -c.top_t / 2 - length / 2
        _add_box(stage, f"{prim_path}/leg_{i}",
                 center=(sx * hx, sy * hy, zc),
                 size=(c.leg_w, c.leg_w, length), color=c.leg_color,
                 collide=collide)
    # seat markers: PURE VISUAL (no collision), 0.4 mm proud of the top face
    zm = c.top_t / 2 + 0.0004
    _add_cyl(stage, f"{prim_path}/seat_plate_marker",
             center=(c.seat_plate[0], c.seat_plate[1], zm),
             radius=c.plate_r + 0.007, height=0.0008, color=c.marker_plate_color,
             collide=None)
    _add_cyl(stage, f"{prim_path}/seat_cup_marker",
             center=(c.seat_cup[0], c.seat_cup[1], zm),
             radius=c.cup_r + 0.005, height=0.0008, color=c.marker_cup_color,
             collide=None)
    mat = _slick_material(stage, prim_path, c.mu_slick_s, c.mu_slick_d)
    _bind_mat(stage, mat, f"{prim_path}/top")
    return root


def _spawn_wedge(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the ramp wedge: DYNAMIC — flat base + PITCHED ramp plate (top surface
    rises h0 at the tip -> H at the heel along +x) + heel wall. Origin at the centre
    of the base's bottom face (z=0 on the floor)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_armor(root, mass=c.wedge_mass, com=(0.0, 0.0, 0.010),
                 lin_damp=0.20, ang_damp=0.20)
    collide = _make_collide(c.contact_offset)
    L, W, h0, H = c.wedge_len, c.wedge_w, c.wedge_h0, c.wedge_hh
    _add_box(stage, f"{prim_path}/base", center=(0.0, 0.0, 0.004),
             size=(L, W, 0.008), color=c.wedge_color, collide=collide)
    alpha = math.atan2(H - h0, L)  # ramp pitch
    t = 0.008
    mid = (0.0, 0.0, (h0 + H) / 2)
    n = (-math.sin(alpha), 0.0, math.cos(alpha))
    ctr = (mid[0] + n[0] * t / 2, 0.0, mid[2] - n[2] * t / 2)
    q = (math.cos(alpha / 2), 0.0, -math.sin(alpha / 2), 0.0)  # +x rises by +alpha
    _add_box(stage, f"{prim_path}/ramp", center=ctr,
             size=(L / math.cos(alpha), W, t), color=c.wedge_color,
             collide=collide, orient=q)
    _add_box(stage, f"{prim_path}/heel", center=(L / 2 - 0.010, 0.0, H / 2 - 0.004),
             size=(0.020, W, H - 0.008), color=c.wedge_color, collide=collide)
    return root


def _spawn_dish(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a dish (plate or cup): DYNAMIC single cylinder, slick (glazed base) so
    dish-on-top friction pair-averages to the slick value. Origin at the cylinder
    CENTRE."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_armor(root, mass=c.mass, com=(0.0, 0.0, 0.0),
                 lin_damp=0.05, ang_damp=0.05)
    collide = _make_collide(c.contact_offset)
    _add_cyl(stage, f"{prim_path}/body", center=(0.0, 0.0, 0.0),
             radius=c.radius, height=c.height, color=c.color, collide=collide)
    mat = _slick_material(stage, prim_path, c.mu_slick_s, c.mu_slick_d)
    _bind_mat(stage, mat, f"{prim_path}/body")
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "table" not in _SPAWNER_CACHE:

        @configclass
        class TableSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_table)
            top_side: float = 0.34
            top_t: float = 0.02
            leg_xy: float = 0.14
            leg_w: float = 0.03
            leg_len: float = 0.15
            stub: float = 0.025
            table_mass: float = 0.8
            table_com: tuple = (-0.004243, -0.004243, -0.05)
            seat_plate: tuple = (0.125, 0.045)
            seat_cup: tuple = (0.060, 0.132)
            plate_r: float = 0.030
            cup_r: float = 0.0225
            mu_slick_s: float = 0.055
            mu_slick_d: float = 0.045
            top_color: tuple = (0.45, 0.30, 0.18)
            leg_color: tuple = (0.20, 0.16, 0.12)
            marker_plate_color: tuple = (0.85, 0.10, 0.10)
            marker_cup_color: tuple = (0.55, 0.75, 0.95)
            contact_offset: float = 0.002

        @configclass
        class WedgeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_wedge)
            wedge_len: float = 0.11
            wedge_w: float = 0.06
            wedge_h0: float = 0.010
            wedge_hh: float = 0.030
            wedge_mass: float = 0.05
            wedge_color: tuple = (0.95, 0.55, 0.08)
            contact_offset: float = 0.002

        @configclass
        class DishSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_dish)
            radius: float = 0.030
            height: float = 0.014
            mass: float = 0.20
            color: tuple = (0.95, 0.95, 0.92)
            mu_slick_s: float = 0.055
            mu_slick_d: float = 0.045
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(table=TableSpawnerCfg, wedge=WedgeSpawnerCfg,
                              dish=DishSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class WobblyBistroSceneCfg(BaseCfg):
    """Config for `WobblyBistroScene`. The interlocks are metric and asserted at
    import time: the rocking pivot is the support-hull EDGE through the grounded
    feet's pad corners (leg_w/sqrt(2) past the leg-centre diagonal), and EACH dish
    seated alone pushes the combined CoM >= 1.25x past it (so an unshored placement
    always rocks the table); both dishes together out-tip even
    the wedge parked on the far corner as ballast (>= 1.5x); at the full fault angle
    tan(theta) >= 2.2x the slick static friction (a tipped table SHEDS its dishes);
    at the level tolerance tan(tol) <= 0.5x the slick dynamic friction (a level
    table KEEPS them); the wedge ramp self-locks under load; and the wedge is too
    short to prop the slab corner even at full tilt."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    level_tol_deg: float = tunable(1.0)  # top tilt from world-up for "level"
    seat_tol: float = tunable(0.035)  # dish centre xy distance from its seat (table frame)
    seat_z_tol: float = tunable(0.012)  # dish resting-height band about the top face (m)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.40)  # max table |ang vel| when judging (rad/s)
    shore_h_slack: float = tunable(0.0035)  # ramp-height slack under the leg (m)
    shore_gap_max: float = tunable(0.006)  # max leg-tip-to-ramp vertical gap (m)
    latch_streak: int = tunable(30)  # consecutive steps (0.25 s) before a stage
    # latches: longer than the ~20-step window in which a dish freshly landed on an
    # UNSHORED table still reads seated+level+still before the tip passes 1 deg

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    table_xy_jitter: float = tunable(0.05)  # table centre xy jitter (+/- m)
    table_yaw_deg: float = tunable(180.0)  # table free yaw (+/- deg): short corner anywhere
    slot_angles_deg: tuple = tunable((180.0, 300.0, 60.0))  # wedge/plate/cup floor slots
    slot_angle_jitter_deg: float = tunable(25.0)
    slot_radii: tuple = tunable((0.34, 0.42, 0.42))
    slot_r_jitter: float = tunable(0.03)

    # --- info: table structure (body frame: origin at TOP SLAB CENTRE) ---------------------------
    top_side: float = info(0.34)
    top_t: float = info(0.02)
    leg_xy: float = info(0.14)  # legs at (+/-leg_xy, +/-leg_xy); leg 0 (+,+) is SHORT
    leg_w: float = info(0.03)
    leg_len: float = info(0.15)
    stub: float = info(0.025)  # the short leg's deficit = unloaded hover gap
    table_mass: float = info(0.8)
    com_bias: float = info(0.006)  # CoM offset PERPENDICULAR past the leg-centre diagonal
    table_com_z: float = info(-0.05)
    seat_plate: tuple = info((0.125, 0.045))  # red ring seat (past the support hull)
    seat_cup: tuple = info((0.060, 0.132))  # pale-blue disc seat (past the support hull)
    mu_slick_s: float = info(0.055)  # polished top / glazed dish base (pair-averaged)
    mu_slick_d: float = info(0.045)

    # --- info: wedge (origin at base bottom centre; ramp rises toward +x heel) -------------------
    wedge_len: float = info(0.11)
    wedge_w: float = info(0.06)
    wedge_h0: float = info(0.010)  # ramp top height at the tip
    wedge_hh: float = info(0.030)  # ramp top height at the heel
    wedge_mass: float = info(0.05)

    # --- info: dishes ----------------------------------------------------------------------------
    plate_r: float = info(0.030)
    plate_h: float = info(0.014)
    plate_mass: float = info(0.30)  # stoneware — heavy enough to out-tip the hull alone
    cup_r: float = info(0.0225)
    cup_h: float = info(0.070)
    cup_mass: float = info(0.25)  # ceramic tumbler — heavy enough to out-tip alone

    contact_offset: float = info(0.002)
    # rubric weights (0.30 + 0.20 + 0.20 = 0.70 = the non-success cap)
    w_shore: float = info(0.30)
    w_plate: float = info(0.20)
    w_cup: float = info(0.20)

    def __post_init__(self) -> None:
        # rocking geometry: the free diagonal runs through legs 1 and 2; the short
        # leg 0 sits at perpendicular lever arm ell from it
        self.ell: float = 2 * self.leg_xy / math.sqrt(2.0)  # 0.198
        self.theta_full: float = math.atan2(self.stub, self.ell)  # full fault angle
        d_p = (self.seat_plate[0] + self.seat_plate[1]) / math.sqrt(2.0)
        d_c = (self.seat_cup[0] + self.seat_cup[1]) / math.sqrt(2.0)
        restore = self.table_mass * self.com_bias
        # the rocking pivot is the support-hull EDGE through the two grounded
        # feet's pad corners nearest the short leg: x + y = leg_w in the body
        # frame, i.e. leg_w/sqrt(2) PAST the leg-centre diagonal
        d_hull = self.leg_w / math.sqrt(2.0)
        # (1) EACH dish seated alone pushes the combined table+dish CoM >= 1.25x
        # past the hull edge — an unshored single placement already rocks
        for m_d, d_d, who in ((self.plate_mass, d_p, "plate"),
                              (self.cup_mass, d_c, "cup")):
            net = (m_d * d_d - restore) / (m_d + self.table_mass)
            assert net > 1.25 * d_hull, f"{who} alone must tip the unshored table"
        # (2) both dishes out-tip even the wedge parked on the far corner as ballast
        d_ballast = self.top_side / math.sqrt(2.0)
        net_all = ((self.plate_mass * d_p + self.cup_mass * d_c - restore
                    - self.wedge_mass * d_ballast)
                   / (self.plate_mass + self.cup_mass + self.table_mass
                      + self.wedge_mass))
        assert net_all > 1.5 * d_hull, \
            "wedge-as-counterweight must NOT rescue an unshored table"
        # (3) at full tilt the slick top SHEDS the dishes...
        assert math.tan(self.theta_full) > 2.2 * self.mu_slick_s, \
            "tipped table must shed its dishes"
        # ...(4) and at the level tolerance it KEEPS them
        assert math.tan(math.radians(self.level_tol_deg)) < 0.5 * self.mu_slick_d, \
            "level table must hold its dishes"
        # (5) wedge ramp self-locks under load (leg-side friction alone, mu ~0.5 avg)
        ramp_slope = (self.wedge_hh - self.wedge_h0) / self.wedge_len
        assert ramp_slope < 0.5 * 0.5, "wedge ramp must self-lock under load"
        # (6) wedge tip passes freely under the hovering leg
        assert self.wedge_h0 + 0.004 < self.stub, "wedge tip must enter the gap"
        # (7) the wedge cannot prop the slab: even fully tipped, the slab's lowest
        # corner bottom stays above the wedge's tallest stable stance (its length)
        slab_corner_drop = (self.top_side / 2) * math.sqrt(2.0) \
            * math.sin(self.theta_full)
        slab_bottom = self.leg_len - slab_corner_drop  # corner bottom height, tipped
        assert self.wedge_len < slab_bottom - 0.004, \
            "wedge must be too short to prop the slab corner"
        # (8) shim band: heel height overshoot stays within ~1.5x the level tol,
        # and the level band is wide enough to contain the whole heel plateau
        over = math.degrees(math.atan2(self.wedge_hh - self.stub, self.ell))
        assert over < 1.6 * self.level_tol_deg, "full insertion must stay near level"
        # (9) seats: on the top with margin, distinct beyond tolerance + dish radii
        for (sx, sy), r in ((self.seat_plate, self.plate_r),
                            (self.seat_cup, self.cup_r)):
            assert max(abs(sx), abs(sy)) + r < self.top_side / 2 - 0.012, \
                "seat too close to the table edge"
        sep = math.hypot(self.seat_plate[0] - self.seat_cup[0],
                         self.seat_plate[1] - self.seat_cup[1])
        assert sep > self.seat_tol + max(self.plate_r, self.cup_r), \
            "seats must be distinct beyond the seat tolerance"
        assert sep > self.plate_r + self.cup_r + 0.015, \
            "both dishes must fit their seats simultaneously"
        # (10) embodiment: both dishes fit the Franka jaw span
        assert 2 * self.plate_r < FRANKA_JAW_SPAN - 0.015, "plate must fit the jaws"
        assert 2 * self.cup_r < FRANKA_JAW_SPAN - 0.015, "cup must fit the jaws"
        # derived layout constants
        self.table_z0: float = self.top_t / 2 + self.leg_len + 0.003  # spawn root z
        self.top_face_z: float = self.top_t / 2  # local z of the top face
        # short-leg tip point, table body frame
        self.leg_tip_local: tuple = (self.leg_xy, self.leg_xy,
                                     -self.top_t / 2 - (self.leg_len - self.stub))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("wobbly_bistro")
class WobblyBistroScene(BaseScene):
    cfg: WobblyBistroSceneCfg

    def __init__(self, cfg: WobblyBistroSceneCfg | None = None) -> None:
        super().__init__(cfg or WobblyBistroSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        table_spawn = spawners["table"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.table_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            top_side=c.top_side, top_t=c.top_t, leg_xy=c.leg_xy, leg_w=c.leg_w,
            leg_len=c.leg_len, stub=c.stub, table_mass=c.table_mass,
            table_com=(-c.com_bias / math.sqrt(2.0), -c.com_bias / math.sqrt(2.0),
                       c.table_com_z),
            seat_plate=c.seat_plate, seat_cup=c.seat_cup,
            plate_r=c.plate_r, cup_r=c.cup_r,
            mu_slick_s=c.mu_slick_s, mu_slick_d=c.mu_slick_d,
            contact_offset=c.contact_offset,
        )
        wedge_spawn = spawners["wedge"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.wedge_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            wedge_len=c.wedge_len, wedge_w=c.wedge_w, wedge_h0=c.wedge_h0,
            wedge_hh=c.wedge_hh, wedge_mass=c.wedge_mass,
            contact_offset=c.contact_offset,
        )
        plate_spawn = spawners["dish"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.plate_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            radius=c.plate_r, height=c.plate_h, mass=c.plate_mass,
            color=(0.95, 0.95, 0.92), mu_slick_s=c.mu_slick_s,
            mu_slick_d=c.mu_slick_d, contact_offset=c.contact_offset,
        )
        cup_spawn = spawners["dish"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.cup_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            radius=c.cup_r, height=c.cup_h, mass=c.cup_mass,
            color=(0.12, 0.25, 0.75), mu_slick_s=c.mu_slick_s,
            mu_slick_d=c.mu_slick_d, contact_offset=c.contact_offset,
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
            "table": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BistroTable",
                spawn=table_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.table_z0)),
            ),
            "wedge": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Wedge",
                spawn=wedge_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.34, 0.0, 0.002)),
            ),
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate",
                spawn=plate_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.21, -0.36, c.plate_h / 2 + 0.003)),
            ),
            "cup": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cup",
                spawn=cup_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.21, 0.36, c.cup_h / 2 + 0.003)),
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
        self.table: RigidObject = env.iscene["table"]
        self.wedge: RigidObject = env.iscene["wedge"]
        self.plate: RigidObject = env.iscene["plate"]
        self.cup: RigidObject = env.iscene["cup"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # streak counters + latches: stage credit survives later probes
        self._shore_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._plate_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._cup_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._shored_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._plate_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._cup_ever = torch.zeros(n, dtype=torch.bool, device=dev)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: table re-posed (xy jitter + FREE yaw — the short corner
        can point anywhere), wedge/plate/cup on independent floor slots (angle band
        + radial jitter + free yaw), streaks/latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- table: jittered centre, free yaw, long legs 3 mm off the floor ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.table_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = (torch.rand(m, 2, device=dev) * 2 - 1) * c.table_xy_jitter
        st[:, 2] = c.table_z0
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        t_xy = st[:, 0:2].clone()
        st[:, 0:3] += origin
        self.table.write_root_state_to_sim(st, env_ids)

        # --- wedge / plate / cup: floor slots around the table (world angles) ---
        zs = (0.002, c.plate_h / 2 + 0.003, c.cup_h / 2 + 0.003)
        for body, ang0, r0, z in zip((self.wedge, self.plate, self.cup),
                                     c.slot_angles_deg, c.slot_radii, zs):
            ang = torch.deg2rad(
                ang0 + (torch.rand(m, device=dev) * 2 - 1) * c.slot_angle_jitter_deg)
            r = r0 + (torch.rand(m, device=dev) * 2 - 1) * c.slot_r_jitter
            byaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = t_xy[:, 0] + r * torch.cos(ang)
            st[:, 1] = t_xy[:, 1] + r * torch.sin(ang)
            st[:, 2] = z
            st[:, 3], st[:, 6] = torch.cos(byaw / 2), torch.sin(byaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        for t in (self._shore_streak, self._plate_streak, self._cup_streak):
            t[env_ids] = 0
        for t in (self._shored_ever, self._plate_ever, self._cup_ever):
            t[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "table": self.table.data.root_state_w[env_ids].clone(),
            "wedge": self.wedge.data.root_state_w[env_ids].clone(),
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "cup": self.cup.data.root_state_w[env_ids].clone(),
            "streaks": torch.stack([self._shore_streak[env_ids],
                                    self._plate_streak[env_ids],
                                    self._cup_streak[env_ids]], dim=1).clone(),
            "latches": torch.stack([self._shored_ever[env_ids],
                                    self._plate_ever[env_ids],
                                    self._cup_ever[env_ids]], dim=1).clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.table.write_root_state_to_sim(state["table"], env_ids)
        self.wedge.write_root_state_to_sim(state["wedge"], env_ids)
        self.plate.write_root_state_to_sim(state["plate"], env_ids)
        self.cup.write_root_state_to_sim(state["cup"], env_ids)
        self._shore_streak[env_ids] = state["streaks"][:, 0]
        self._plate_streak[env_ids] = state["streaks"][:, 1]
        self._cup_streak[env_ids] = state["streaks"][:, 2]
        self._shored_ever[env_ids] = state["latches"][:, 0]
        self._plate_ever[env_ids] = state["latches"][:, 1]
        self._cup_ever[env_ids] = state["latches"][:, 2]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A small square bistro TABLE (dark wood, top "
            f"{c.top_side * 100:.0f} x {c.top_side * 100:.0f} cm, standing about "
            f"{(c.top_t + c.leg_len) * 100:.0f} cm tall) stands free on the floor. "
            f"It is FAULTY: one of its four legs is {c.stub * 1000:.0f} mm too "
            f"short. The empty table still rests level on its three good legs, so "
            f"the fault shows only as a visible ~{c.stub * 1000:.0f} mm GAP between "
            f"the short leg's foot and the floor — walk around and find the "
            f"hovering leg. The tabletop is polished and very slippery, and both "
            f"dinner seats marked on it — a RED RING (for the white plate) and a "
            f"PALE-BLUE DISC (for the blue cup) — lie on the short leg's half of "
            f"the top: set ANY dish down there while the leg is unsupported and its "
            f"weight rocks the table onto the short leg (~"
            f"{math.degrees(c.theta_full):.0f} deg of tilt) and the dish slides "
            f"straight off. On the floor around the table lie an ORANGE WEDGE ramp "
            f"({c.wedge_len * 100:.0f} cm long, rising to {c.wedge_hh * 1000:.0f} "
            f"mm at its tall heel end), a WHITE PLATE (a {2 * c.plate_r * 100:.0f} "
            f"cm disc) and a BLUE CUP (a {2 * c.cup_r * 100:.0f} cm wide, "
            f"{c.cup_h * 100:.0f} cm tall tumbler).\n"
            f"Goal: set the table so it STAYS set. First SHORE THE SHORT LEG: "
            f"slide the orange wedge along the floor, thin tip first, into the gap "
            f"under the hovering foot until the foot sits over the thick part of "
            f"the ramp (the ramp self-locks under load; pushed all the way in it "
            f"still leaves the top acceptably level). Then place the WHITE PLATE "
            f"flat on the RED RING and stand the BLUE CUP upright on the PALE-BLUE "
            f"DISC. Success requires the final state to hold on its own: tabletop "
            f"level within {c.level_tol_deg:.0f} deg, each dish resting centred on "
            f"its own marker (within {c.seat_tol * 100:.0f} cm), everything still. "
            f"Dishes placed before shoring simply slide off — shim first, set "
            f"second. Swapped dishes (plate on the blue disc, cup on the red ring), "
            f"a dish elsewhere on the table or on the floor, or a still-tilted "
            f"table all fail."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the orange wedge under the bistro table's short hovering leg "
            "until the leg is supported, then set the white plate on the red ring "
            "and stand the blue cup on the pale-blue disc. The top must end level "
            "with both dishes resting on their own markers; placing dishes before "
            "shoring the leg tips the table and they slide off."
        )

    # ----- readings ------------------------------------------------------------------------------
    def tilt_deg(self) -> torch.Tensor:
        """(N,) tabletop tilt from world-up, degrees."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(self.table.data.root_quat_w, ez)
        return torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0)))

    def level(self) -> torch.Tensor:
        """(N,) bool: top within level_tol_deg of world-up."""
        return self.tilt_deg() < self.cfg.level_tol_deg

    def leg_tip_w(self) -> torch.Tensor:
        """(N, 3) world position of the SHORT leg's foot centre."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        p = torch.tensor(self.cfg.leg_tip_local, device=self.env.device).expand(n, 3)
        return self.table.data.root_pos_w + quat_apply(self.table.data.root_quat_w, p)

    def _in_table_frame(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        rel = pos_w - self.table.data.root_pos_w
        return quat_apply_inverse(self.table.data.root_quat_w, rel)

    def shored(self) -> torch.Tensor:
        """(N,) bool: the wedge supporting the short leg — leg foot over the ramp
        inside the wedge footprint, the ramp under the foot's BEARING EDGE (the
        30 mm foot meets the incline heel-side-edge first) at supporting height,
        the foot down on it (small vertical gap), wedge upright & still, top
        level. The geometric transcription of 'the shim is in and bearing'."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        tip_w = self.leg_tip_w()
        rel = tip_w - self.wedge.data.root_pos_w
        loc = quat_apply_inverse(self.wedge.data.root_quat_w, rel)
        in_x = (loc[:, 0] > -c.wedge_len / 2 + 0.005) \
            & (loc[:, 0] < c.wedge_len / 2 - 0.003)
        in_y = loc[:, 1].abs() < c.wedge_w / 2 - 0.004
        slope = (c.wedge_hh - c.wedge_h0) / c.wedge_len
        x_eval = (loc[:, 0] + c.leg_w / 2 - 0.002).clamp(max=c.wedge_len / 2 - 0.003)
        ramp_h = c.wedge_h0 + (x_eval + c.wedge_len / 2) * slope
        h_ok = ramp_h > c.stub - c.shore_h_slack
        gap_ok = (loc[:, 2] - ramp_h) < c.shore_gap_max
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        w_up = quat_apply(self.wedge.data.root_quat_w, ez)
        upright = w_up[:, 2] > 0.97
        still = self.wedge.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return in_x & in_y & h_ok & gap_ok & upright & still & self.level()

    def _dish_seated(self, body: RigidObject, seat: tuple, half_h: float) -> torch.Tensor:
        """(N,) bool: dish centre within seat_tol of its OWN seat (table frame xy),
        RESTING on the top face (table-frame z band), upright, dish still, top
        level."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        loc = self._in_table_frame(body.data.root_pos_w)
        seat_t = torch.tensor(seat, device=loc.device).expand(n, 2)
        xy_ok = (loc[:, 0:2] - seat_t).norm(dim=-1) < c.seat_tol
        z_expect = c.top_face_z + half_h
        z_ok = (loc[:, 2] - z_expect).abs() < c.seat_z_tol
        ez = torch.tensor([0.0, 0.0, 1.0], device=loc.device).expand(n, 3)
        up = quat_apply(body.data.root_quat_w, ez)
        upright = up[:, 2] > 0.95
        still = body.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return xy_ok & z_ok & upright & still & self.level()

    def plate_seated(self) -> torch.Tensor:
        return self._dish_seated(self.plate, self.cfg.seat_plate, self.cfg.plate_h / 2)

    def cup_seated(self) -> torch.Tensor:
        return self._dish_seated(self.cup, self.cfg.seat_cup, self.cfg.cup_h / 2)

    def table_still(self) -> torch.Tensor:
        c = self.cfg
        return (self.table.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.table.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega)

    def _update_latches(self) -> None:
        c = self.cfg
        for streak, ever, now in (
                (self._shore_streak, self._shored_ever, self.shored()),
                (self._plate_streak, self._plate_ever, self.plate_seated()),
                (self._cup_streak, self._cup_ever, self.cup_seated())):
            streak[:] = torch.where(now, streak + 1,
                                    torch.zeros_like(streak))
            ever |= streak >= c.latch_streak

    # ----- step-coupled bookkeeping (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No plant (the table is passive) — just run the streak latches so stage
        credit is earned by SUSTAINED physical states and survives later probes."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: the set table HOLDS — top level, plate settled on the red
        ring, cup settled on the blue disc, table still. Physical outcome only: with
        both seats loaded on the short-leg half, a level top is only reachable with
        the short leg actually shored (asserted moment margins)."""
        return (self.plate_seated() & self.cup_seated() & self.table_still()
                & self.level())

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.30*shored + 0.20*plate + 0.20*cup (streak-latched;
        ~0 for the null policy; non-success cap 0.70) and exactly 1.0 iff
        success()."""
        c = self.cfg
        self._update_latches()
        base = (c.w_shore * self._shored_ever.float()
                + c.w_plate * self._plate_ever.float()
                + c.w_cup * self._cup_ever.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="wobbly_bistro", robot="null"))
