"""DumpHopperScene — carry the BASKET under the caged milk hopper, then tip the hopper
by its lever so the carton slides out and drops in (sim_gen task
`living_room_scene2_pick_up_the_milk_and_put_it_in_the_basket_i177`).

Derived from libero_90/living_room_scene2 "pick up the milk and put it in the basket",
but STRATEGICALLY inverted: the seed's whole plan is to grasp the ITEM — lift the milk
off a cluttered table, carry it, release it inside an open basket; success is a
bounding-box containment readout and the milk is free-standing and graspable from the
first frame. Here the milk carton can NEVER be grasped: it lies inside a roofed cage on
a tipping tray (a real revolute dump mechanism on a pedestal), whose only opening is a
low front slot — too short for the carton to stand through, too far under the roof to
lift anything out. The solver must instead (a) pick up and carry the EMPTY BASKET and
set it on the catch mark under the hopper's spout, then (b) press the yellow lever
paddle down and HOLD it, tipping the tray about its hinge until the carton slides out
through the slot under gravity, falls off the discharge edge, and lands in the basket,
then (c) release the lever (the back-heavy tray returns to level on its own). The plan
inverts the seed (move the CONTAINER to the cargo, and deliver the cargo by actuating a
mechanism + gravity, hands never on the item), and the load-bearing interactions —
holding a lever through its arc while a sliding delivery completes, and catching a
ballistic drop — have no counterpart in the seed's grasp-carry-drop.

Strategy vs what was read while building (spring_bay i38, pen_holder exemplar, the
seed): spring_bay's task is an elastic compress-then-seat — press a can against a
spring through a growing force and release so stored energy pins it; its end state is a
PRELOADED mechanism and the object is manipulated directly throughout. Here there is no
elastic element anywhere, the mechanism is a plain gravity-return dump hinge, the item
is UNTOUCHABLE by design, and the mechanism's end state is unstressed (back at level) —
what is judged is ordinary containment that can only be reached by operating the
mechanism over a correctly pre-placed container. pen_holder is many-object tip-up
insertion into a carriable cup with no mechanism at all.

success(): the milk carton rests INSIDE the basket (basket-frame x/y windows, center
below the rim), the basket stands upright ON the floor, carton and basket settled, and
the red juice carton (decoy) is nowhere in/over the basket. score(): stateless,
monotone along the solution — 0.15 basket upright on the catch mark, 0.45 the milk
carton released from the hopper cage (a physical, persistent fact), 0.70 released AND
inside the basket, combined by max and capped below 1.0; 1.0 iff success(). Null
policy scores ~0 (the basket spawns well off the catch mark, the milk starts caged).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - station (dynamic, 30 kg — heavy DYNAMIC fixture, never kinematic, so the hinge
    anchor follows the reset teleport): ground slab + pedestal column + a GREEN
    visual-only catch mark painted on the floor at the spout side.
  - tray (dynamic, 1.2 kg, authored CoM + diagonal inertia): the tipping hopper — a
    slick floor plate, two side walls, a back wall, a roof, a front lintel leaving a
    100 mm tall discharge slot, and the YELLOW lever paddle sticking out beside the
    spout. RevoluteJoint (axis = station y) at the plate's mid-line, hinge 0.36 m up,
    limits 0..35 deg; the authored CoM sits behind the hinge so gravity holds the tray
    LEVEL against the lower limit and returns it there when the lever is released.
    Joint-pair collision stays FILTERED (the USD default): travel is bounded by the
    joint limits alone.
  - milk carton (target): 65 x 65 x 120 mm white box, 350 g, spawned LYING inside the
    cage (a lying carton fits under the slot lintel; a standing one does not).
  - juice carton (decoy): identical red box standing on the floor; it must stay out of
    the basket.
  - basket (dynamic, 0.5 kg): open-top box, 300 mm square, 150 mm walls, 8 mm wall
    thickness (a parallel jaw grips the rim), high-grip interior.

Per-episode randomization (readback-verified in smoke): station xy jitter + FREE yaw,
milk pose inside the cage (x/y jitter + yaw), basket and decoy on mirrored jittered
arcs around the station (side, bearing, radius, free yaw). Heavy imports (isaaclab,
pxr) are deferred so importing this module — and registering the scene — stays
app-free.
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


# ----- geometry constants (single source of truth: spawners + cfg asserts + rubric) ------------
H_HINGE = 0.360  # hinge axis height above the ground (station frame)
TL_HALF = 0.150  # tray plate half-length (station/tray x; +x = discharge)
TW_HALF = 0.100  # cage inner half-width (y)
PLATE_T = 0.012  # tray plate thickness (top surface = tray-frame z 0, through the hinge)
WALL_T = 0.010
CAGE_H = 0.150  # wall/roof underside height above the plate top
ROOF_T = 0.012
OPEN_H = 0.100  # discharge slot height above the plate top (lintel underside)
LINTEL_X = 0.145  # lintel center x (inner face 0.140)
PAD_X1 = 0.270  # lever paddle tip x (tray frame)
PAD_Y = (-0.175, -0.105)  # paddle y extent — OUTBOARD of the discharge stream
TILT_MAX_DEG = 35.0  # hinge upper limit (dump stop)
COM_X = -0.055  # authored tray CoM x: back-heavy -> gravity-return to level
MILK_W = 0.065  # carton square cross-section
MILK_H = 0.120  # carton long dimension
BK_HALF = 0.150  # basket outer half-width
BK_WALL_T = 0.008
BK_WALL_H = 0.150  # wall height above the basket floor top
BK_PLATE_T = 0.012
BK_IN_HALF = BK_HALF - BK_WALL_T / 2 - BK_WALL_T / 2  # inner half-width = 0.142
ZONE_X = 0.345  # catch-mark center (station frame, on the ground)
SLAB_HALF_X = 0.130  # station ground-slab half-x (basket at the zone must clear it)


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (N,3) by quaternions q (N,4), wxyz."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def encode_force(mode: int, q_ref: torch.Tensor, q_now: torch.Tensor,
                 f_world: torch.Tensor) -> torch.Tensor:
    """Pre-encode a desired WORLD-frame wrench for `set_external_force_and_torque`.

    Some pods rotate an applied wrench by the body's rotation since its reference
    orientation (applied = R_now * R_ref^T * arg). mode 0 passes the world vector
    through unchanged; mode 1 pre-encodes with R_ref * R_now^T. NOTE the lever torque
    this task needs is ALONG the hinge axis, and the tray only ever rotates ABOUT that
    axis, so the drag rotation leaves it invariant — the fallback exists for safety."""
    if mode == 0:
        return f_world
    return _qapply(_qmul(q_ref, _qinv(q_now)), f_world)


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


def _friction_material(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None, collide: bool = True) -> None:
    """Author one box child prim (translate -> scale, authored once — idempotent per
    prim, the duplicate-xformOp trap). `collide=False` -> pure visual (the catch mark)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if not collide:
        return
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_station(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the hopper station as ONE heavy DYNAMIC compound body (never kinematic:
    the revolute hinge's anchor must follow the reset teleport — a kinematic body0's
    anchor stays world-fixed at the spawn pose). Origin = station center on the
    ground; +x = discharge/spout direction."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    # ZERO sleep/stabilization: a sleeping station would freeze the hinge anchor.
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    body = _friction_material(stage, f"{prim_path}/body_mat", cfg.mu_body_s, cfg.mu_body_d)
    gray = (0.45, 0.48, 0.50)
    dark = (0.30, 0.32, 0.35)
    green = (0.10, 0.75, 0.20)
    # ground slab (stability footprint; short on +x so a basket at the zone clears it)
    _box(stage, f"{prim_path}/slab", (2 * SLAB_HALF_X, 0.26, 0.030), (0.0, 0.0, 0.015),
         gray, 0.0015, material=body)
    # pedestal column up to just under the tray plate
    _box(stage, f"{prim_path}/column", (0.080, 0.160, 0.310), (0.0, 0.0, 0.185),
         dark, 0.0015, material=body)
    # GREEN catch mark on the floor under the spout — VISUAL ONLY (no collider)
    _box(stage, f"{prim_path}/catch_mark", (0.16, 0.20, 0.002), (ZONE_X, 0.0, 0.001),
         green, 0.0, collide=False)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the tipping hopper tray: slick floor plate, cage (side walls + back wall
    + roof + front lintel leaving the discharge slot), and the YELLOW lever paddle —
    one dynamic compound body with AUTHORED CoM (back of the hinge: gravity-return to
    level) and diagonal inertia, plus the RevoluteJoint into the sibling station.
    Origin = the hinge point (plate top surface passes through it)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass_props.mass))
    # Authored CoM (MassAPI mass alone would pin the CoM at the body origin = the
    # hinge, killing the gravity-return) + diagonal inertia (box estimates).
    mass.CreateCenterOfMassAttr(Gf.Vec3f(float(COM_X), 0.0, 0.02))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.008, 0.013, 0.014))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    pxrb.CreateAngularDampingAttr(1.0)  # soften the swing into / back from the stop
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    slick = _friction_material(stage, f"{prim_path}/slick_mat", cfg.mu_slide_s, cfg.mu_slide_d)
    body = _friction_material(stage, f"{prim_path}/body_mat", cfg.mu_body_s, cfg.mu_body_d)
    steel = (0.55, 0.62, 0.70)
    yellow = (0.95, 0.85, 0.10)
    # slick floor plate (top surface = tray z 0, through the hinge axis)
    _box(stage, f"{prim_path}/plate", (2 * TL_HALF, 0.22, PLATE_T), (0.0, 0.0, -PLATE_T / 2),
         steel, 0.0015, material=slick)
    # cage side walls (inner faces |y| = TW_HALF)
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/wall_{tag}", (2 * TL_HALF, WALL_T, CAGE_H),
             (0.0, sy * (TW_HALF + WALL_T / 2), CAGE_H / 2), steel, 0.0015, material=body)
    # back wall (inner face x = -0.140)
    _box(stage, f"{prim_path}/back", (WALL_T, 0.22, CAGE_H),
         (-TL_HALF + 0.005, 0.0, CAGE_H / 2), steel, 0.0015, material=body)
    # roof (no reach-in from above; the cage's only opening is the front slot)
    _box(stage, f"{prim_path}/roof", (2 * TL_HALF, 0.22, ROOF_T),
         (0.0, 0.0, CAGE_H + ROOF_T / 2), steel, 0.0015, material=body)
    # front lintel: leaves a discharge slot OPEN_H tall x 2*TW_HALF wide
    _box(stage, f"{prim_path}/lintel", (WALL_T, 0.22, CAGE_H - OPEN_H),
         (LINTEL_X, 0.0, OPEN_H + (CAGE_H - OPEN_H) / 2), steel, 0.0015, material=slick)
    # YELLOW lever paddle, outboard of the discharge stream (press it DOWN)
    _box(stage, f"{prim_path}/paddle",
         (PAD_X1 - TL_HALF, PAD_Y[1] - PAD_Y[0], PLATE_T),
         ((TL_HALF + PAD_X1) / 2, (PAD_Y[0] + PAD_Y[1]) / 2, -PLATE_T / 2),
         yellow, 0.0015, material=body)

    # revolute hinge to the sibling station, axis = station y, at the plate mid-line
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Station"])
    j.CreateBody1Rel().SetTargets([prim_path])
    # joint-pair collision stays FILTERED (the USD default): the tray may sweep
    # through the column; its travel is bounded by the joint limits alone.
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(H_HINGE)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-0.5)  # rest: gravity holds the back-heavy tray here (level)
    j.CreateUpperLimitAttr(float(TILT_MAX_DEG))  # the dump stop
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (explicit @configclass subclasses
    of RigidObjectSpawnerCfg, defined lazily so the module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "station" not in _SPAWNER_CACHE:

        @configclass
        class StationSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_station)
            mu_body_s: float = 0.40
            mu_body_d: float = 0.35

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            mu_body_s: float = 0.35
            mu_body_d: float = 0.30
            mu_slide_s: float = 0.10
            mu_slide_d: float = 0.08

        _SPAWNER_CACHE.update(station=StationSpawnerCfg, tray=TraySpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DumpHopperSceneCfg(BaseCfg):
    """Config for `DumpHopperScene`. The honesty knobs are asserted in `__post_init__`:
    the cage physically bars every path but the slot (a lying carton fits under the
    lintel, a standing one does not, the roof bars lifting), the slick plate genuinely
    slides at the dump tilt, the lever force sits in an arm's comfortable band, the
    paddle's sweep clears the basket rim, and the basket spawns off the catch mark so
    the null policy scores ~0."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    in_xy_max: float = tunable(0.115)  # |x|,|y| of the carton center, basket frame
    in_z_win: tuple = tunable((0.010, 0.120))  # carton center z above the basket floor top
    basket_up_max_deg: float = tunable(10.0)  # basket axis vs world up
    basket_z_win: tuple = tunable((0.004, 0.032))  # basket root height (rest: 0.012)
    settle_lin: float = tunable(0.05)  # settle gates (m/s, rad/s)
    settle_ang: float = tunable(1.0)
    zone_x_win: tuple = tunable((0.290, 0.400))  # catch-mark window (station frame)
    zone_y_max: float = tunable(0.080)
    decoy_xy_max: float = tunable(0.160)  # decoy-in/over-basket violation box
    decoy_z_win: tuple = tunable((-0.020, 0.250))
    cage_box: tuple = tunable((-0.140, 0.160, 0.100, -0.020, 0.150))  # x0,x1,|y|,z0,z1
    free_box: tuple = tunable((-0.200, 0.220, 0.150, -0.100, 0.210))  # released = OUTSIDE

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    stn_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the station (m)
    stn_yaw_max: float = tunable(180.0)  # uniform +/- station yaw (deg; FREE heading)
    milk_x_win: tuple = tunable((-0.075, -0.020))  # carton spawn x in the tray frame
    milk_y_max: float = tunable(0.045)  # carton spawn |y| in the tray frame
    milk_yaw_max: float = tunable(15.0)  # carton spawn yaw about its lying pose (deg)
    arc_bearing: tuple = tunable((30.0, 70.0))  # basket/decoy bearing band off +x (deg)
    arc_radius: tuple = tunable((0.42, 0.50))  # ... radius band around the station (m)

    # --- info: structure -----------------------------------------------------------------------
    milk_w: float = info(MILK_W)
    milk_h: float = info(MILK_H)
    milk_mass: float = info(0.35)
    juice_mass: float = info(0.30)
    basket_mass: float = info(0.50)
    tray_mass: float = info(1.2)
    stn_mass: float = info(30.0)  # heavy dynamic fixture (hinge anchor follows teleports)
    mu_milk_s: float = info(0.35)
    mu_milk_d: float = info(0.30)
    mu_slide_s: float = info(0.10)  # tray plate + lintel (the carton must slide out)
    mu_slide_d: float = info(0.08)
    mu_body_s: float = info(0.35)
    mu_body_d: float = info(0.30)
    mu_basket_s: float = info(0.55)  # high-grip basket (kills the landing slide)
    mu_basket_d: float = info(0.50)
    mu_ground_s: float = info(0.60)
    mu_ground_d: float = info(0.50)

    def __post_init__(self) -> None:
        # -- captivity by construction: the slot is the cage's only exit ----------------------
        assert OPEN_H >= MILK_W + 0.030, "a LYING carton must pass the slot with margin"
        assert OPEN_H < MILK_H - 0.015, "a STANDING carton must NOT pass the slot"
        assert CAGE_H < MILK_H + 0.035, "roof too high: a carton could be stood up inside"
        assert 2 * TW_HALF >= MILK_H + 0.06, "slot wide enough for a yawed lying carton"
        # -- the slick plate genuinely slides at the dump tilt --------------------------------
        mu_comb = (self.mu_slide_s + self.mu_milk_s) / 2  # PhysX default combine: average
        assert math.tan(math.radians(TILT_MAX_DEG)) > 2.5 * mu_comb, \
            "dump tilt must beat the plate friction angle by a wide margin"
        # -- gravity-return is real and the lever force is comfortable ------------------------
        bias = self.tray_mass * 9.81 * abs(COM_X)
        assert 0.3 <= bias <= 1.5, f"tray gravity-return bias {bias:.2f} N*m out of band"
        f_lever = (bias + self.milk_mass * 9.81 * 0.09) / PAD_X1
        assert 1.5 <= f_lever <= 10.0, f"lever press force {f_lever:.1f} N out of band"
        # -- the paddle's sweep clears the basket rim -----------------------------------------
        rim = BK_PLATE_T + BK_WALL_H
        tip_z = H_HINGE - PAD_X1 * math.sin(math.radians(TILT_MAX_DEG))
        assert tip_z >= rim + 0.030, "paddle at full depression must clear the basket rim"
        edge_z = H_HINGE - TL_HALF * math.sin(math.radians(TILT_MAX_DEG))
        assert edge_z >= rim + 0.050, "discharge edge must clear the basket rim"
        # -- the paddle sits outboard of the discharge stream ---------------------------------
        yaw = math.radians(self.milk_yaw_max)
        milk_edge = self.milk_y_max + (MILK_H * math.sin(yaw) + MILK_W * math.cos(yaw)) / 2
        assert PAD_Y[1] < -milk_edge - 0.005, "paddle must sit outside the carton's path"
        # -- the basket swallows the carton and fits the jaw ----------------------------------
        assert 2 * BK_IN_HALF > math.hypot(MILK_W, MILK_H) + 0.10, \
            "basket interior must swallow a tumbled carton with margin"
        assert BK_WALL_T < 0.020, "basket wall must fit a parallel jaw"
        # -- rubric windows consistent with geometry ------------------------------------------
        assert self.in_xy_max <= BK_IN_HALF - 0.020
        assert self.in_z_win[1] <= BK_WALL_H - 0.025, "in-basket must mean BELOW the rim"
        # -- basket at the zone clears the station slab ---------------------------------------
        assert self.zone_x_win[0] - BK_HALF >= SLAB_HALF_X + 0.005, \
            "a basket anywhere in the zone must clear the station slab"
        # -- null policy scores 0: spawns start outside the catch mark ------------------------
        y_min = self.arc_radius[0] * math.sin(math.radians(self.arc_bearing[0]))
        assert y_min > self.zone_y_max + 0.08, \
            "basket spawn arc must start well outside the catch mark"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("dump_hopper")
class DumpHopperScene(BaseScene):
    cfg: DumpHopperSceneCfg

    def __init__(self, cfg: DumpHopperSceneCfg | None = None) -> None:
        super().__init__(cfg or DumpHopperSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        carton_rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            max_depenetration_velocity=0.5, sleep_threshold=0.0,
            stabilization_threshold=0.0, linear_damping=0.05, angular_damping=0.1)
        carton_coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0)
        carton_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_milk_s, dynamic_friction=c.mu_milk_d, restitution=0.0)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_ground_s, dynamic_friction=c.mu_ground_d,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            # NOTE: the station MUST spawn before the tray (the hinge targets it).
            "station": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Station",
                spawn=spawners["station"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.stn_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mu_body_s=0.40, mu_body_d=0.35),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=spawners["tray"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.tray_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mu_body_s=c.mu_body_s, mu_body_d=c.mu_body_d,
                    mu_slide_s=c.mu_slide_s, mu_slide_d=c.mu_slide_d),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, H_HINGE)),
            ),
            "milk": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Milk",
                spawn=sim_utils.CuboidCfg(
                    size=(c.milk_w, c.milk_w, c.milk_h),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.95, 0.95, 0.97)),
                    physics_material=carton_mat, rigid_props=carton_rigid,
                    collision_props=carton_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.milk_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.05, 0.0, H_HINGE + MILK_W / 2 + 0.003),
                    rot=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)),
            ),
            "juice": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Juice",
                spawn=sim_utils.CuboidCfg(
                    size=(c.milk_w, c.milk_w, c.milk_h),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.85, 0.12, 0.12)),
                    physics_material=carton_mat, rigid_props=carton_rigid,
                    collision_props=carton_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.juice_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, -0.55, c.milk_h / 2 + 0.003)),
            ),
            "basket": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Basket",
                spawn=spawners.get("basket", self._basket_spawner())(
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.basket_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mu_basket_s=c.mu_basket_s, mu_basket_d=c.mu_basket_d),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.55, BK_PLATE_T + 0.003)),
            ),
        }
        return out

    @staticmethod
    def _basket_spawner() -> Any:
        """Basket compound-spawner cfg class (lazy, cached beside the others)."""
        from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
        from isaaclab.sim.utils import clone
        from isaaclab.utils import configclass

        if "basket" not in _SPAWNER_CACHE:

            @configclass
            class BasketSpawnerCfg(RigidObjectSpawnerCfg):
                func: Callable = clone(_spawn_basket)
                mu_basket_s: float = 0.55
                mu_basket_d: float = 0.50

            _SPAWNER_CACHE["basket"] = BasketSpawnerCfg
        return _SPAWNER_CACHE["basket"]

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # External-wrench hinge plant: without this the lever torque is
                # under-applied across TGS iterations and the tray stalls short.
                "enable_external_forces_every_iteration": True,
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
        self.station: RigidObject = env.iscene["station"]
        self.tray: RigidObject = env.iscene["tray"]
        self.milk: RigidObject = env.iscene["milk"]
        self.juice: RigidObject = env.iscene["juice"]
        self.basket: RigidObject = env.iscene["basket"]
        self.env_origins = env.iscene.env_origins

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter + free-yaw the station (heavy DYNAMIC teleport — the
        hinge anchor follows), write the tray CONSISTENTLY level in the station's new
        frame, lay the carton inside the cage with jitter + yaw, and drop basket and
        decoy on mirrored jittered arcs (side, bearing, radius, free yaw)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # --- station: xy jitter + free yaw (write the WHOLE linkage together) ---
        sxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.stn_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.stn_yaw_max)
        q_stn = _qz(yaw)
        zeros = torch.zeros(m, device=dev)
        stn_pos = torch.cat([sxy, zeros.unsqueeze(-1)], dim=-1)
        write(self.station, stn_pos, q_stn)

        # --- tray: LEVEL at the hinge, in the station's new frame ---
        hinge_local = torch.tensor([0.0, 0.0, H_HINGE], device=dev).expand(m, 3)
        write(self.tray, stn_pos + _qapply(q_stn, hinge_local), q_stn)

        # --- milk: LYING inside the cage (tray frame x/y jitter + yaw) ---
        mx = c.milk_x_win[0] + torch.rand(m, device=dev) * (c.milk_x_win[1] - c.milk_x_win[0])
        my = (torch.rand(m, device=dev) * 2 - 1) * c.milk_y_max
        milk_local = torch.stack(
            [mx, my, torch.full((m,), H_HINGE + MILK_W / 2 + 0.003, device=dev)], dim=-1)
        milk_local[:, 2] -= H_HINGE  # express in station frame at hinge height...
        milk_local[:, 2] += H_HINGE  # (kept explicit: plate top IS hinge height)
        psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.milk_yaw_max)
        half_pi = torch.full((m,), math.pi / 2, device=dev)
        q_milk = _qmul(q_stn, _qmul(_qz(psi), _qy(half_pi)))  # local +z -> station +x
        write(self.milk, stn_pos + _qapply(q_stn, milk_local), q_milk)

        # --- basket + decoy: mirrored jittered arcs around the station ---
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.tensor(-1.0, device=dev), torch.tensor(1.0, device=dev))
        b0, b1 = (math.radians(v) for v in c.arc_bearing)
        r0, r1 = c.arc_radius
        for body, sgn, z0, quat_fn in (
                (self.basket, side, BK_PLATE_T + 0.003, None),
                (self.juice, -side, MILK_H / 2 + 0.003, None)):
            bear = (b0 + torch.rand(m, device=dev) * (b1 - b0)) * sgn
            rad = r0 + torch.rand(m, device=dev) * (r1 - r0)
            local = torch.stack([rad * torch.cos(bear), rad * torch.sin(bear),
                                 torch.full((m,), z0, device=dev)], dim=-1)
            byaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            write(body, stn_pos + _qapply(q_stn, local), _qmul(q_stn, _qz(byaw)))

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {nm: getattr(self, nm).data.root_state_w[env_ids].clone()
                for nm in ("station", "tray", "milk", "juice", "basket")}

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm in ("station", "tray", "milk", "juice", "basket"):
            getattr(self, nm).write_root_state_to_sim(state[nm], env_ids)

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A hopper station stands on the floor: a gray pedestal "
            f"({H_HINGE * 100:.0f} cm tall) carrying a steel tipping tray — a roofed "
            f"cage whose ONLY opening is a low discharge slot ({OPEN_H * 100:.0f} cm "
            f"tall, {2 * TW_HALF * 100:.0f} cm wide) at one end, the spout. A WHITE "
            f"milk carton ({c.milk_w * 100:.1f} x {c.milk_w * 100:.1f} x "
            f"{c.milk_h * 100:.0f} cm) lies inside the cage, visible through the slot; "
            f"it cannot be lifted out (the roof is in the way) and it cannot stand up "
            f"through the slot — the cage only releases it when the tray is TIPPED. A "
            f"YELLOW lever paddle sticks out beside the spout at tray height: pressing "
            f"it down and holding it tips the tray about its hinge (up to "
            f"{TILT_MAX_DEG:.0f} deg) so the carton slides out through the slot and "
            f"falls off the spout edge; when released, the back-heavy tray returns "
            f"level on its own. A GREEN catch mark is painted on the floor under the "
            f"spout, where the carton lands. An empty open-top BASKET "
            f"({2 * BK_HALF * 100:.0f} cm square, {BK_WALL_H * 100:.0f} cm walls, "
            f"thin rim a parallel jaw can grip) and a RED juice carton stand on the "
            f"floor on opposite sides of the station. The station's position and "
            f"heading, the carton's pose in the cage, and the basket and juice "
            f"positions change every episode — read the scene by looking.\n"
            f"Goal: get the WHITE milk carton into the basket. The carton cannot be "
            f"grasped, so work the delivery: first pick up the empty basket by its rim "
            f"and set it upright on the GREEN catch mark under the spout, then press "
            f"the YELLOW paddle down and hold it until the milk slides out and drops "
            f"into the basket, then let go. Finish with the milk resting INSIDE the "
            f"basket, the basket standing upright on the floor, and everything at "
            f"rest. The RED juice carton must stay out of the basket: any part of it "
            f"in or over the basket fails the task. Tipping the tray before the "
            f"basket is in place just dumps the milk on the floor."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Set the empty basket upright on the green catch mark under the hopper's "
            "spout, then press the yellow lever down and hold it so the white milk "
            "carton slides out of the cage and drops into the basket; release the "
            "lever and leave the milk resting inside the upright basket. Keep the red "
            "juice carton out of the basket."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def stn_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> station frame."""
        return _qapply(_qinv(self.station.data.root_quat_w),
                       pos_w - self.station.data.root_pos_w)

    def tray_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> tray frame (origin = hinge, plate top = z 0)."""
        return _qapply(_qinv(self.tray.data.root_quat_w),
                       pos_w - self.tray.data.root_pos_w)

    def basket_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> basket frame (origin = floor-plate top center)."""
        return _qapply(_qinv(self.basket.data.root_quat_w),
                       pos_w - self.basket.data.root_pos_w)

    def tilt_deg(self) -> torch.Tensor:
        """(N,) tray tip angle in degrees (positive = spout end dipping): the tray's
        +x axis expressed in the station frame — readback of the hinge's actual travel."""
        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        ax = _qapply(_qinv(self.station.data.root_quat_w),
                     _qapply(self.tray.data.root_quat_w, ex))
        return torch.rad2deg(torch.atan2(-ax[:, 2], ax[:, 0]))

    def milk_in_cage(self) -> torch.Tensor:
        """(N,) bool: carton center inside the cage volume (tray frame)."""
        x0, x1, ym, z0, z1 = self.cfg.cage_box
        loc = self.tray_local(self.milk.data.root_pos_w)
        return (loc[:, 0] >= x0) & (loc[:, 0] <= x1) & (loc[:, 1].abs() <= ym) \
            & (loc[:, 2] >= z0) & (loc[:, 2] <= z1)

    def milk_released(self) -> torch.Tensor:
        """(N,) bool: carton center OUTSIDE the expanded cage volume — the physical,
        persistent fact that the hopper has genuinely let the carton go."""
        x0, x1, ym, z0, z1 = self.cfg.free_box
        loc = self.tray_local(self.milk.data.root_pos_w)
        inside = (loc[:, 0] >= x0) & (loc[:, 0] <= x1) & (loc[:, 1].abs() <= ym) \
            & (loc[:, 2] >= z0) & (loc[:, 2] <= z1)
        return ~inside

    def in_basket(self, body) -> torch.Tensor:
        """(N,) bool: body center inside the basket interior, BELOW the rim."""
        c = self.cfg
        loc = self.basket_local(body.data.root_pos_w)
        return (loc[:, 0].abs() <= c.in_xy_max) & (loc[:, 1].abs() <= c.in_xy_max) \
            & (loc[:, 2] >= c.in_z_win[0]) & (loc[:, 2] <= c.in_z_win[1])

    def basket_upright_on_floor(self) -> torch.Tensor:
        """(N,) bool: basket standing upright, resting on the ground."""
        c = self.cfg
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        up = _qapply(self.basket.data.root_quat_w, ez)
        upright = up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(c.basket_up_max_deg))
        z = (self.basket.data.root_pos_w - self.env_origins)[:, 2]
        return upright & (z >= c.basket_z_win[0]) & (z <= c.basket_z_win[1])

    def basket_in_zone(self) -> torch.Tensor:
        """(N,) bool: basket upright on the floor, centered on the catch mark."""
        c = self.cfg
        loc = self.stn_local(self.basket.data.root_pos_w)
        return self.basket_upright_on_floor() \
            & (loc[:, 0] >= c.zone_x_win[0]) & (loc[:, 0] <= c.zone_x_win[1]) \
            & (loc[:, 1].abs() <= c.zone_y_max)

    def decoy_out(self) -> torch.Tensor:
        """(N,) bool: the red juice carton is nowhere in/over the basket."""
        c = self.cfg
        loc = self.basket_local(self.juice.data.root_pos_w)
        inside = (loc[:, 0].abs() <= c.decoy_xy_max) & (loc[:, 1].abs() <= c.decoy_xy_max) \
            & (loc[:, 2] >= c.decoy_z_win[0]) & (loc[:, 2] <= c.decoy_z_win[1])
        return ~inside

    def settled(self, body) -> torch.Tensor:
        return (body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: milk carton resting inside the basket (below the rim), basket
        upright on the floor, both settled, decoy nowhere in/over the basket."""
        return self.in_basket(self.milk) & self.basket_upright_on_floor() \
            & self.settled(self.milk) & self.settled(self.basket) & self.decoy_out()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: max of 0.15 (basket upright on the catch mark), 0.45
        (the carton genuinely released from the hopper cage — a persistent physical
        fact), 0.70 (released AND inside the basket); 1.0 iff success(). Stateless
        and monotone along the intended solution (place basket -> tip -> catch ->
        settle); the null policy scores ~0 (basket spawns off the mark, milk caged)."""
        zone = self.basket_in_zone().float()
        rel = self.milk_released().float()
        inb = (self.milk_released() & self.in_basket(self.milk)).float()
        base = torch.maximum(0.15 * zone, torch.maximum(0.45 * rel, 0.70 * inb))
        return torch.where(self.success(), base.new_tensor(1.0), base)


def _spawn_basket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the open-top basket: floor plate + four walls, one dynamic compound
    body, high-grip interior. Origin = floor-plate TOP center (rest root z 0.012)."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.1)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    grip = _friction_material(stage, f"{prim_path}/grip_mat",
                              cfg.mu_basket_s, cfg.mu_basket_d)
    brown = (0.55, 0.38, 0.18)
    _box(stage, f"{prim_path}/floor", (2 * BK_HALF, 2 * BK_HALF, BK_PLATE_T),
         (0.0, 0.0, -BK_PLATE_T / 2), brown, 0.0015, material=grip)
    for tag, cx, cy, sx, sy in (
            ("xp", BK_HALF - BK_WALL_T / 2, 0.0, BK_WALL_T, 2 * BK_HALF),
            ("xm", -(BK_HALF - BK_WALL_T / 2), 0.0, BK_WALL_T, 2 * BK_HALF),
            ("yp", 0.0, BK_HALF - BK_WALL_T / 2, 2 * BK_HALF - 2 * BK_WALL_T, BK_WALL_T),
            ("ym", 0.0, -(BK_HALF - BK_WALL_T / 2), 2 * BK_HALF - 2 * BK_WALL_T, BK_WALL_T)):
        _box(stage, f"{prim_path}/wall_{tag}", (sx, sy, BK_WALL_H),
             (cx, cy, BK_WALL_H / 2), brown, 0.0015, material=grip)
    return root


register_env("simgen", lambda: EnvCfg(scene="dump_hopper", robot="null", env_spacing=3.0))
