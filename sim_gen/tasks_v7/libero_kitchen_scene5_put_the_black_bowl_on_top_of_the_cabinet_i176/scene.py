"""BallastRockerScene — ballast the cabinet's rocking top tray, THEN put the black bowl on it.

Derived from libero_90 kitchen_scene5 "put the black bowl on top of the cabinet" (a
Franka picks a black bowl off the table and sets it on the cabinet's FIXED flat top).
Here the cabinet's top is NOT a fixed surface: it is a SEE-SAW TRAY on a central
hinge between two pylons, deliberately unbalanced (its centre of mass sits on the
socket side), resting tilted against its +rest_deg hinge stop with the fenced ballast
SOCKET end down and the green-marked BOWL ZONE end raised. The seed's whole plan —
carry the bowl to the top and set it down — is a live trap: the bowl's weight on the
raised zone out-torques the tray's own bias, the tray pivots to its -tip_deg stop
(steeper than the friction angle) and DUMPS the bowl onto the floor, then rocks back
empty. The only winning plan is a counterweight interlock: FIRST seat the heavy steel
ballast cube into the fenced socket on the lowered end (its torque pins the tray
against the rest stop with ~5x margin), and only THEN place the black bowl upright on
the zone at the raised end. The seed needs one pick-and-place onto a static shelf;
this task needs torque reasoning about a passive mechanism, a second object whose
placement is a physical precondition, and an execution order forced by gravity — the
mechanism is never actuated directly, it is STABILIZED by loading it.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - cabinet: KINEMATIC pedestal at a FIXED pose (it anchors the tray's hinge; jointed
    mechanisms are never teleported): a solid 32 x 26 x 26 cm box plus two hinge
    pylons rising to the hinge line.
  - tray: DYNAMIC plate (23 x 42 x 1 cm, 0.30 kg) with its root origin ON the hinge
    line (tray centre); authored CoM offset toward the socket end (the tilt bias);
    children: 4 RED socket fences (8.5 cm square interior) at the socket end, a low
    dark CLEAT strip guarding the inboard edge of the bowl zone (catches downhill
    creep at rest; does NOT retain the bowl when the tray tips the other way), and a
    GREEN visual-only zone patch. Bind-time revolute joint cabinet->tray about +X,
    limits [-tip_deg, +rest_deg]; both stops are pure joint limits.
  - ballast: steel-blue cube (5 cm, 0.9 kg) — the counterweight.
  - bowl: BLACK hexagonal cup (3 crossed base boxes + 6 wall boxes, ~12 cm across,
    4.5 cm tall, 0.18 kg) — the seed's object.

Torque budget (about the hinge, at rest angle): tray bias 0.13 Nm holds the empty
tray at +rest_deg; the bowl on the zone applies 0.26 Nm (2.0x the bias -> always
tips); the ballast in the socket applies 1.30 Nm (with the bias, 5.5x the bowl ->
always holds, including drop transients).

Per-episode randomization (readback-verifiable): Bernoulli left/right slot swap of
ballast and bowl on the floor + per-object xy jitter + free yaw.

Rubric (0..1; partial credit latched so transient achievements keep credit):
  0.10 * carry_ballast — running max of the ballast's approach to the socket point
  0.25 * seated        — ballast ever settled in the socket with the tray at rest
                         (latched; the interlock stage)
  0.10 * carry_bowl    — running max of the bowl's approach to the zone point
  0.25 * placed        — bowl ever settled upright in the zone while the ballasted
                         tray rests at its stop (latched)
  1.0 iff success()    — ballast seated, bowl upright in the zone, tray at the rest
                         stop, everything still. Non-success cap 0.85. Null ~0 (the
                         empty tray already rests at its stop: no stage is free).

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
# One rigid body per object, several child colliders, authored with raw pxr APIs; only
# `isaaclab.sim.utils.clone` is borrowed (regex-resolve + per-env replication).

_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable | None,
             rot_z_deg: float = 0.0) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if rot_z_deg:
        xf.AddRotateZOp().Set(float(rot_z_deg))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
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


def _spawn_cabinet(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the cabinet: KINEMATIC solid pedestal + two hinge pylons. Local origin
    at the centre of the footprint at ground level; the hinge line runs along +X
    between the pylon tops — there is NO fixed top surface, only the tray."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    _add_box(stage, f"{prim_path}/body",
             center=(0.0, 0.0, c.cab_h / 2),
             size=(c.cab_dx, c.cab_dy, c.cab_h), color=c.color, collide=collide)
    for sgn, nm in ((-1.0, "pylon_a"), (1.0, "pylon_b")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(sgn * c.pylon_x, 0.0, c.cab_h + c.pylon_h / 2),
                 size=(c.pylon_t, c.pylon_w, c.pylon_h),
                 color=c.pylon_color, collide=collide)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the rocking tray: DYNAMIC plate, root origin ON the hinge line (plate
    centre). Children: plate, 4 socket fences (socket end), inboard cleat (zone end),
    and a GREEN visual-only zone patch (no collision). Mass AND centre of mass are
    authored explicitly — the CoM offset toward the socket end IS the tilt bias.
    Sleep/stabilization thresholds zeroed (it must respond the instant it is loaded)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass_props.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, float(cfg.com_y), 0.0))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.40)  # damp the rock-back after a dump
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    c = cfg
    _add_box(stage, f"{prim_path}/plate",
             center=(0.0, 0.0, 0.0),
             size=(c.tray_dx, c.tray_dy, c.tray_t), color=c.color, collide=collide)
    top = c.tray_t / 2
    # socket fences: 8.5 cm square interior centred at (0, sock_y)
    half_in = c.sock_inner / 2
    for sgn, nm in ((-1.0, "fence_near"), (1.0, "fence_far")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(0.0, c.sock_y + sgn * (half_in + c.fence_t / 2), top + c.fence_h / 2),
                 size=(c.sock_inner + 2 * c.fence_t, c.fence_t, c.fence_h),
                 color=c.fence_color, collide=collide)
    for sgn, nm in ((-1.0, "fence_l"), (1.0, "fence_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(sgn * (half_in + c.fence_t / 2), c.sock_y, top + c.fence_h / 2),
                 size=(c.fence_t, c.sock_inner, c.fence_h),
                 color=c.fence_color, collide=collide)
    # cleat: low strip on the zone side's INBOARD edge — catches downhill creep while
    # the tray rests zone-up; the outboard edge stays free so a tipping tray dumps.
    _add_box(stage, f"{prim_path}/cleat",
             center=(0.0, c.cleat_y, top + c.cleat_h / 2),
             size=(c.tray_dx - 0.02, c.cleat_t, c.cleat_h),
             color=c.cleat_color, collide=collide)
    # green zone patch: VISUAL ONLY (no collision), flush with the plate top
    _add_box(stage, f"{prim_path}/zone_patch",
             center=(0.0, c.zone_y, top + 0.0006),
             size=(0.12, 0.11, 0.001), color=(0.15, 0.65, 0.20), collide=None)
    return root


def _spawn_ballast(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the ballast: DYNAMIC steel-blue cube, root at its centre."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.20)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    s = cfg.side
    _add_box(stage, f"{prim_path}/cube", center=(0.0, 0.0, 0.0),
             size=(s, s, s), color=cfg.color, collide=collide)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the black bowl: DYNAMIC hexagonal cup. Base = 3 crossed flat boxes (a
    near-disc), walls = 6 boxes on the hexagon flats. Root origin at the centre of
    the base's BOTTOM face (root z == support height when the bowl stands upright)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.30)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    c = cfg
    for k in range(3):
        _add_box(stage, f"{prim_path}/base_{k}",
                 center=(0.0, 0.0, c.base_t / 2),
                 size=(c.base_len, c.base_w, c.base_t), color=c.color,
                 collide=collide, rot_z_deg=60.0 * k)
    for k in range(6):
        phi = math.radians(60.0 * k)
        _add_box(stage, f"{prim_path}/wall_{k}",
                 center=(c.wall_r * math.cos(phi), c.wall_r * math.sin(phi),
                         c.base_t + c.wall_h / 2),
                 size=(c.wall_t, c.wall_w, c.wall_h), color=c.color,
                 collide=collide, rot_z_deg=60.0 * k)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cabinet" not in _SPAWNER_CACHE:

        @configclass
        class CabinetSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cabinet)
            cab_dx: float = 0.32
            cab_dy: float = 0.26
            cab_h: float = 0.26
            pylon_x: float = 0.145
            pylon_t: float = 0.03
            pylon_w: float = 0.04
            pylon_h: float = 0.125
            color: tuple = (0.45, 0.30, 0.15)
            pylon_color: tuple = (0.30, 0.20, 0.10)
            contact_offset: float = 0.002

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            tray_dx: float = 0.23
            tray_dy: float = 0.42
            tray_t: float = 0.010
            com_y: float = -0.045
            sock_y: float = -0.15
            sock_inner: float = 0.085
            fence_t: float = 0.008
            fence_h: float = 0.020
            cleat_y: float = 0.085
            cleat_t: float = 0.008
            cleat_h: float = 0.014
            zone_y: float = 0.15
            color: tuple = (0.55, 0.55, 0.58)
            fence_color: tuple = (0.75, 0.12, 0.10)
            cleat_color: tuple = (0.15, 0.15, 0.15)
            contact_offset: float = 0.002

        @configclass
        class BallastSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ballast)
            side: float = 0.05
            color: tuple = (0.30, 0.40, 0.60)
            contact_offset: float = 0.002

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            base_len: float = 0.10
            base_w: float = 0.052
            base_t: float = 0.010
            wall_r: float = 0.0475
            wall_t: float = 0.010
            wall_w: float = 0.055
            wall_h: float = 0.035
            color: tuple = (0.04, 0.04, 0.04)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(cabinet=CabinetSpawnerCfg, tray=TraySpawnerCfg,
                              ballast=BallastSpawnerCfg, bowl=BowlSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BallastRockerSceneCfg(BaseCfg):
    """Config for `BallastRockerScene`. The interlock is purely gravitational: the
    unbalanced tray rests zone-end-up; a bowl loaded on the zone without ballast
    out-torques the bias 2:1 and the tray dumps it; the seated ballast out-torques
    the bowl 5.5:1 and pins the tray at its rest stop."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    rest_tol_deg: float = tunable(5.0)  # tray counts "at rest stop" within this of rest_deg
    settle_speed: float = tunable(0.05)  # max |lin vel| (ballast AND bowl) when judging (m/s)
    settle_omega: float = tunable(0.40)  # max tray |ang vel| when judging (rad/s)
    sock_xy_tol: float = tunable(0.030)  # ballast centre within this of the socket centre
    sock_z_lo: float = tunable(0.015)  # ballast-centre z band, tray frame (seated ~0.030)
    sock_z_hi: float = tunable(0.045)
    zone_x_tol: float = tunable(0.060)  # bowl root within this of the zone centre (tray x)
    zone_y_tol: float = tunable(0.055)  # ... and this along the tray (tray y)
    zone_z_lo: float = tunable(-0.010)  # bowl-root z band, tray frame (resting ~0.006)
    zone_z_hi: float = tunable(0.040)
    up_min: float = tunable(0.88)  # min body-z . world-z for "upright" (28 deg > 12 deg tilt)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    slot_jitter: float = tunable(0.03)  # per-object spawn xy jitter (+/- m)
    swap_slots: bool = tunable(True)  # Bernoulli ballast/bowl spawn-slot swap (demo sets False)
    yaw_deg: float = tunable(180.0)  # free yaw on ballast and bowl (+/- deg)

    # --- info: layout (single Franka base at the origin; socket and zone within reach) ----------
    cab_pos: tuple = info((0.50, 0.0))  # cabinet footprint centre (fixture is FIXED: it
    # anchors the tray's hinge, and jointed mechanisms are never teleported — the
    # randomization lives in the two floor objects)
    slot_a: tuple = info((0.32, 0.22))  # floor spawn slot A (robot side, left)
    slot_b: tuple = info((0.32, -0.22))  # floor spawn slot B (robot side, right)

    # --- info: cabinet / hinge structure --------------------------------------------------------
    cab_dx: float = info(0.32)
    cab_dy: float = info(0.26)
    cab_h: float = info(0.26)
    pylon_x: float = info(0.145)
    pylon_t: float = info(0.03)
    pylon_w: float = info(0.04)
    pylon_h: float = info(0.125)
    hinge_z: float = info(0.375)  # hinge line height (along +x through the pylon tops)
    cab_color: tuple = info((0.45, 0.30, 0.15))

    # --- info: tray + mechanism -----------------------------------------------------------------
    tray_dx: float = info(0.23)  # across the hinge (x)
    tray_dy: float = info(0.42)  # along the see-saw (y): socket end -y, zone end +y
    tray_t: float = info(0.010)
    tray_mass: float = info(0.30)
    tray_com_y: float = info(-0.045)  # authored CoM offset: the tilt bias (socket side)
    rest_deg: float = info(12.0)  # upper joint limit: rest stop (zone end UP)
    tip_deg: float = info(35.0)  # lower joint limit: dump stop (zone end DOWN, > friction angle)
    sock_y: float = info(-0.15)  # socket centre along the tray (tray frame)
    sock_inner: float = info(0.085)  # fence interior (square)
    fence_h: float = info(0.020)
    cleat_y: float = info(0.085)  # inboard-edge cleat guarding the zone
    zone_y: float = info(0.15)  # bowl-zone centre along the tray (tray frame)

    # --- info: objects --------------------------------------------------------------------------
    ballast_side: float = info(0.05)
    ballast_mass: float = info(0.90)
    ballast_color: tuple = info((0.30, 0.40, 0.60))  # steel blue
    bowl_r_out: float = info(0.060)  # hexagon corner reach (~12 cm wide)
    bowl_h: float = info(0.045)
    bowl_mass: float = info(0.18)

    contact_offset: float = info(0.002)
    # rubric weights (0.10 + 0.25 + 0.10 + 0.25 = 0.70 <= the 0.85 non-success cap)
    w_carry_bal: float = info(0.10)
    w_seat: float = info(0.25)
    w_carry_bowl: float = info(0.10)
    w_place: float = info(0.25)

    # Derived (filled in __post_init__).
    sock_goal: tuple = field(default=None, init=False)  # world rest-pose ballast-centre point
    zone_goal: tuple = field(default=None, init=False)  # world rest-pose bowl-root point

    def __post_init__(self) -> None:
        self.sock_goal = self.rest_point(self.sock_y, self.ballast_side / 2 + self.tray_t / 2)
        self.zone_goal = self.rest_point(self.zone_y, self.tray_t / 2 + 0.001)

    def rest_point(self, y_local: float, z_local: float) -> tuple:
        """World (env-local) point of tray-frame (0, y_local, z_local) with the tray
        at its REST angle — the anchor for goals, teleport hovers and probe placements."""
        phi = math.radians(self.rest_deg)
        cx, cy = self.cab_pos
        return (
            cx,
            cy + y_local * math.cos(phi) - z_local * math.sin(phi),
            self.hinge_z + y_local * math.sin(phi) + z_local * math.cos(phi),
        )


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("ballast_rocker")
class BallastRockerScene(BaseScene):
    cfg: BallastRockerSceneCfg

    def __init__(self, cfg: BallastRockerSceneCfg | None = None) -> None:
        super().__init__(cfg or BallastRockerSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        cx, cy = c.cab_pos
        cabinet_spawn = spawners["cabinet"](
            mass_props=sim_utils.MassPropertiesCfg(mass=15.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            cab_dx=c.cab_dx, cab_dy=c.cab_dy, cab_h=c.cab_h, pylon_x=c.pylon_x,
            pylon_t=c.pylon_t, pylon_w=c.pylon_w, pylon_h=c.pylon_h,
            color=c.cab_color, contact_offset=c.contact_offset,
        )
        tray_spawn = spawners["tray"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.tray_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            tray_dx=c.tray_dx, tray_dy=c.tray_dy, tray_t=c.tray_t, com_y=c.tray_com_y,
            sock_y=c.sock_y, sock_inner=c.sock_inner, fence_h=c.fence_h,
            cleat_y=c.cleat_y, zone_y=c.zone_y, contact_offset=c.contact_offset,
        )
        ballast_spawn = spawners["ballast"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.ballast_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            side=c.ballast_side, color=c.ballast_color, contact_offset=c.contact_offset,
        )
        bowl_spawn = spawners["bowl"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            contact_offset=c.contact_offset,
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
            "cabinet": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet",
                spawn=cabinet_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, 0.0)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=tray_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(cx, cy, c.hinge_z),
                    rot=(math.cos(math.radians(c.rest_deg) / 2),
                         math.sin(math.radians(c.rest_deg) / 2), 0.0, 0.0)),
            ),
            "ballast": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ballast",
                spawn=ballast_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_a[0], c.slot_a[1], c.ballast_side / 2 + 0.002)),
            ),
            "bowl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl",
                spawn=bowl_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_b[0], c.slot_b[1], 0.003)),
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
        self.cabinet: RigidObject = env.iscene["cabinet"]
        self.tray: RigidObject = env.iscene["tray"]
        self.ballast: RigidObject = env.iscene["ballast"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.env_origins = env.iscene.env_origins
        self._author_hinge()
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._carry_bal_max = torch.zeros(n, device=dev)
        self._seated = torch.zeros(n, dtype=torch.bool, device=dev)
        self._carry_bowl_max = torch.zeros(n, device=dev)
        self._placed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._d0_bal = torch.full((n,), 0.5, device=dev)
        self._d0_bowl = torch.full((n,), 0.5, device=dev)

    def _author_hinge(self) -> None:
        """Per env: a +X revolute joint cabinet->tray on the hinge line, limits
        [-tip_deg, +rest_deg] (positive rotation about +x raises the +y zone end;
        +rest_deg = rest stop, -tip_deg = dump stop), joint-pair collision disabled."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/tray_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Cabinet"])
            j.CreateBody1Rel().SetTargets([f"{base}/Tray"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.hinge_z)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-float(c.tip_deg))
            j.CreateUpperLimitAttr(float(c.rest_deg))

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: fixture re-asserted, tray re-posed at its REST angle (pure
        joint-coordinate re-pose of the follower about the unchanged hinge; gravity
        then keeps it on the rest stop), ballast and bowl randomly ASSIGNED to the two
        floor slots (+ xy jitter, free yaw), latches cleared, spawn distances recorded."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        cx, cy = c.cab_pos

        # --- fixture (kinematic, fixed) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = cx, cy
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.cabinet.write_root_state_to_sim(st, env_ids)

        # --- tray: at the rest stop (roll +rest_deg about the hinge x-axis) ---
        half = math.radians(c.rest_deg) / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = cx, cy, c.hinge_z
        st[:, 3], st[:, 4] = math.cos(half), math.sin(half)
        st[:, 0:3] += origin
        self.tray.write_root_state_to_sim(st, env_ids)

        # --- ballast + bowl: Bernoulli slot swap + xy jitter + free yaw, on the floor ---
        if c.swap_slots:
            swap = torch.rand(m, device=dev) < 0.5
        else:
            swap = torch.zeros(m, dtype=torch.bool, device=dev)
        slot_a = torch.tensor(c.slot_a, device=dev).expand(m, 2)
        slot_b = torch.tensor(c.slot_b, device=dev).expand(m, 2)
        bal_xy = torch.where(swap.unsqueeze(1), slot_b, slot_a) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        bowl_xy = torch.where(swap.unsqueeze(1), slot_a, slot_b) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        for body, xy, z0 in ((self.ballast, bal_xy, c.ballast_side / 2 + 0.002),
                             (self.bowl, bowl_xy, 0.003)):
            half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg) / 2
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z0
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- latches + carry references ---
        sock = torch.tensor(c.sock_goal, device=dev)
        zone = torch.tensor(c.zone_goal, device=dev)
        p_bal = torch.cat([bal_xy, torch.full((m, 1), c.ballast_side / 2, device=dev)], dim=1)
        p_bowl = torch.cat([bowl_xy, torch.full((m, 1), 0.003, device=dev)], dim=1)
        self._d0_bal[env_ids] = (p_bal - sock).norm(dim=-1).clamp(min=0.05)
        self._d0_bowl[env_ids] = (p_bowl - zone).norm(dim=-1).clamp(min=0.05)
        self._carry_bal_max[env_ids] = 0.0
        self._seated[env_ids] = False
        self._carry_bowl_max[env_ids] = 0.0
        self._placed[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cabinet": self.cabinet.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "ballast": self.ballast.data.root_state_w[env_ids].clone(),
            "bowl": self.bowl.data.root_state_w[env_ids].clone(),
            "carry_bal": self._carry_bal_max[env_ids].clone(),
            "seated": self._seated[env_ids].clone(),
            "carry_bowl": self._carry_bowl_max[env_ids].clone(),
            "placed": self._placed[env_ids].clone(),
            "d0_bal": self._d0_bal[env_ids].clone(),
            "d0_bowl": self._d0_bowl[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cabinet.write_root_state_to_sim(state["cabinet"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.ballast.write_root_state_to_sim(state["ballast"], env_ids)
        self.bowl.write_root_state_to_sim(state["bowl"], env_ids)
        self._carry_bal_max[env_ids] = state["carry_bal"]
        self._seated[env_ids] = state["seated"]
        self._carry_bowl_max[env_ids] = state["carry_bowl"]
        self._placed[env_ids] = state["placed"]
        self._d0_bal[env_ids] = state["d0_bal"]
        self._d0_bowl[env_ids] = state["d0_bowl"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A BROWN wooden cabinet ({c.cab_dx * 100:.0f} x {c.cab_dy * 100:.0f} cm "
            f"footprint, {c.cab_h * 100:.0f} cm tall) stands on the floor. Its top is "
            f"NOT a fixed surface: a GRAY see-saw tray ({c.tray_dx * 100:.0f} x "
            f"{c.tray_dy * 100:.0f} cm plate) rocks on a central hinge between two "
            f"pylons at {c.hinge_z * 100:.0f} cm height, free to pivot side to side "
            f"between two stops ({c.rest_deg:.0f} deg one way, {c.tip_deg:.0f} deg the "
            f"other). The tray is unbalanced and currently rests tilted "
            f"{c.rest_deg:.0f} deg with one end LOW and one end HIGH. The LOW end "
            f"carries a RED-fenced square socket ({c.sock_inner * 100:.1f} cm interior, "
            f"{c.fence_h * 100:.0f} cm fences). The HIGH end carries the target zone: a "
            f"GREEN patch on the tray top, guarded on its hinge side by a low dark "
            f"cleat strip. On the floor in front of the cabinet stand two objects — a "
            f"steel-blue BALLAST cube ({c.ballast_side * 100:.0f} cm, "
            f"{c.ballast_mass:.1f} kg) and a BLACK hexagonal bowl "
            f"(~{2 * c.bowl_r_out * 100:.0f} cm wide, {c.bowl_h * 100:.1f} cm tall, "
            f"{c.bowl_mass * 1000:.0f} g); which stands left and which stands right "
            f"changes per episode.\n"
            f"BEWARE: the bowl's weight on the raised zone out-torques the empty "
            f"tray's bias — the tray pivots to its {c.tip_deg:.0f} deg stop (steeper "
            f"than friction can hold) and DUMPS the bowl onto the floor, then rocks "
            f"back. The ballast cube seated in the red socket out-torques the bowl "
            f"~5:1 and pins the tray at its rest stop.\n"
            f"Goal: FIRST seat the ballast cube inside the red-fenced socket on the "
            f"tray's low end (its centre within {c.sock_xy_tol * 100:.0f} cm of the "
            f"socket centre — the fences funnel it), THEN place the BLACK bowl UPRIGHT "
            f"on the green zone at the raised end (bowl centre within "
            f"{c.zone_y_tol * 100:.1f} cm of the zone centre along the tray and "
            f"{c.zone_x_tol * 100:.0f} cm across it), and leave everything at rest "
            f"with the tray on its rest stop. The order is forced by physics: the "
            f"unballasted tray dumps the bowl. A bowl on the floor or inboard of the "
            f"cleat, a ballast resting outside the socket, an upside-down bowl, or a "
            f"tipped tray does not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Seat the steel-blue ballast cube inside the red-fenced socket on the low "
            "end of the cabinet's rocking top tray, then set the black bowl upright "
            "on the green zone at the tray's raised end. Placing the bowl before the "
            "ballast tips the tray and dumps the bowl."
        )

    # ----- readings / rubric ----------------------------------------------------------------------
    def tray_deg(self) -> torch.Tensor:
        """(N,) tray roll in DEG about the hinge +x axis (+rest_deg = rest stop with
        the zone end up; -tip_deg = dump stop). The tray only ever rotates about the
        hinge, so the root quat is (cos t/2, sin t/2, 0, 0)."""
        q = self.tray.data.root_quat_w
        return torch.rad2deg(2.0 * torch.atan2(q[:, 1], q[:, 0]))

    def tray_at_rest(self) -> torch.Tensor:
        """(N,) bool: tray within `rest_tol_deg` of its rest stop."""
        return (self.tray_deg() - self.cfg.rest_deg).abs() <= self.cfg.rest_tol_deg

    def _tray_local(self, body: RigidObject) -> torch.Tensor:
        """(N,3) body root position in the TRAY'S BODY FRAME (the rubric lives there
        so the socket/zone checks track the tray through any tilt)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - self.tray.data.root_pos_w
        return quat_apply_inverse(self.tray.data.root_quat_w, rel)

    def _upright(self, body: RigidObject) -> torch.Tensor:
        q = body.data.root_quat_w
        r33 = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        return r33 >= self.cfg.up_min

    def _still(self, body: RigidObject) -> torch.Tensor:
        return body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def _tray_still(self) -> torch.Tensor:
        return self.tray.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_omega

    def ballast_in_socket(self) -> torch.Tensor:
        """(N,) bool, geometric (tray frame): ballast centre inside the fenced socket."""
        c = self.cfg
        p = self._tray_local(self.ballast)
        return ((p[:, 0].abs() <= c.sock_xy_tol)
                & ((p[:, 1] - c.sock_y).abs() <= c.sock_xy_tol)
                & (p[:, 2] >= c.sock_z_lo) & (p[:, 2] <= c.sock_z_hi))

    def ballast_seated(self) -> torch.Tensor:
        """(N,) bool: ballast in the socket, tray resting still on its rest stop,
        ballast still — the interlock stage, judged as a settled physical outcome."""
        return (self.ballast_in_socket() & self.tray_at_rest()
                & self._tray_still() & self._still(self.ballast))

    def bowl_in_zone(self) -> torch.Tensor:
        """(N,) bool, geometric (tray frame): bowl root inside the zone box, upright,
        with the tray at its rest stop. The z band rejects hovers and stacks; the y
        band rejects the cleat's inboard side and the tray tip."""
        c = self.cfg
        p = self._tray_local(self.bowl)
        return ((p[:, 0].abs() <= c.zone_x_tol)
                & ((p[:, 1] - c.zone_y).abs() <= c.zone_y_tol)
                & (p[:, 2] >= c.zone_z_lo) & (p[:, 2] <= c.zone_z_hi)
                & self._upright(self.bowl) & self.tray_at_rest())

    def _update_latches(self) -> None:
        c = self.cfg
        dev = self.env.device
        sock = torch.tensor(c.sock_goal, device=dev)
        zone = torch.tensor(c.zone_goal, device=dev)
        d_bal = (self.ballast.data.root_pos_w - self.env_origins - sock).norm(dim=-1)
        d_bowl = (self.bowl.data.root_pos_w - self.env_origins - zone).norm(dim=-1)
        carry_bal = torch.nan_to_num((1.0 - d_bal / self._d0_bal).clamp(0.0, 1.0))
        carry_bowl = torch.nan_to_num((1.0 - d_bowl / self._d0_bowl).clamp(0.0, 1.0))
        self._carry_bal_max = torch.maximum(self._carry_bal_max, carry_bal)
        self._carry_bowl_max = torch.maximum(self._carry_bowl_max, carry_bowl)
        seated = self.ballast_seated()
        self._seated |= seated
        # placed latches only on the FULL physical outcome (ballasted tray at rest
        # carrying the settled bowl) — a bowl transiting the zone volume on an
        # unballasted tray about to dump it earns nothing here.
        self._placed |= seated & self.bowl_in_zone() & self._still(self.bowl)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No driven mechanics: gravity owns the tray's two stops. Latch."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: ballast seated in the socket, BLACK bowl upright in the zone,
        tray resting still on its rest stop, everything still. Physical outcomes only
        (the ballasted hinge demonstrably carries both loads)."""
        self._update_latches()
        return self.ballast_seated() & self.bowl_in_zone() & self._still(self.bowl)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10 * ballast carry + 0.25 * seated + 0.10 * bowl
        carry + 0.25 * placed — all latched, ~0 for doing nothing, capped 0.85 — and
        exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_carry_bal * self._carry_bal_max
                + c.w_seat * self._seated.float()
                + c.w_carry_bowl * self._carry_bowl_max
                + c.w_place * self._placed.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="ballast_rocker", robot="null"))
