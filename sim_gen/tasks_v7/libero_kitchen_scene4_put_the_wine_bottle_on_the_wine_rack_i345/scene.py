"""MissingRailRackScene — REPAIR the display rack (seat its missing crossbar), THEN
rack the wine bottle. Derived from libero_90 "put the wine bottle on the wine rack",
but the goal support surface DOES NOT EXIST at reset: the rack's near crossbar lies
loose on the floor, and until it is seated in its two bracket slots a bottle laid at
the goal pose simply tips through the missing-rail gap and falls to the floor.

Scene: a floor-standing DISPLAY RACK — two side towers carrying one FIXED far rail
(with a pair of dark CHOCK blocks marking the display saddle) and, on the near side,
two empty open-top BRACKET slots (guide walls fore and aft on each tower top). A
loose CROSSBAR (square-section wooden bar, the rack's missing near rail) lies on the
floor on one side; a green glass WINE BOTTLE stands on the floor on the other side.

Goal (two stages, the order physically forced):
  1. REPAIR — seat the crossbar into BOTH bracket slots so it spans the towers,
     dropping between each slot's guide walls onto the tower tops; seated, its top
     surface is level with the fixed rail's top: the rack now has two parallel
     horizontal rails 14 cm apart.
  2. RACK — lay the bottle HORIZONTAL across the two rails, centred on the chock
     saddle (body between the chocks, axis square across the rails, either end may
     face the towers' near side). The bottle rests bridging fixed rail + crossbar.

The missing-rail gap is the physical order-forcer: the bottle's success x-window
(|x| <= 25 mm) lies entirely on the near side of the fixed rail, so without the
crossbar every goal-window pose is supported on ONE side only and tips through —
the seed's plan (put the bottle straight on the rack) cannot terminate in the goal
state. The rubric additionally makes the repair load-bearing by conjunct: success
requires the BAR seated in its brackets, so bridging the bottle across the fixed
rail and a mis-laid bar (resting anywhere else on the towers) fails.

A solver therefore needs a different PLAN from the seed (fetch and install a rack
component to CREATE the support surface, then place across two rails into a saddle
— versus lower-onto-an-existing-surface) and a different code structure (a
two-object, conjunct rubric with a repair stage — versus a single bbox test over a
finished rack's top).

Assets are fully procedural (compound spawner for the rack, raw pxr authoring;
plain cuboid for the bar; compound body+neck bottle):
  - rack: KINEMATIC compound — 2 towers + fixed far rail + 2 chocks + 4 bracket
    guide walls. Origin at the base centre on the floor; local +x points FRONT
    (toward the robot): the empty brackets are the NEAR rail position (x=+0.07),
    the fixed rail the FAR one (x=-0.07).
  - bar: DYNAMIC cuboid 25 x 380 x 25 mm (long axis local +y), 0.25 kg.
  - bottle: DYNAMIC compound — body cylinder (55 mm dia, 190 mm) + neck cylinder
    (20 mm dia, 70 mm) along local +z, origin at the BODY centre (MassAPI CoM
    there, so the racked rest is symmetric about the body).

Per-episode randomization (readback-verifiable): rack xy jitter + yaw; a Bernoulli
side sign puts the bar on one side of the centreline and the bottle on the other,
each with xy jitter; bar yaw jitter; bottle free yaw.

Rubric (0..1; partial progress latched every step so credit never evaporates):
  0.10 * bar_lift    — bar origin ever above 0.08 m (a null policy never)
  0.30 * bar_seat    — bar ever seated in its brackets (pose windows, rack frame)
  0.10 * bottle_lift — bottle origin ever above 0.155 m (stand is 0.095)
  0.10 * cradle      — bottle ever in loose saddle bands WHILE the bar is seated
  1.0 iff success()  — bar seated AND bottle resting horizontal across both rails
                       centred on the saddle AND both bodies settled. Non-success
                       cap 0.60.

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


def _add_box(stage, path: str, *, center, size, color, collide: Callable) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable) -> None:
    """One z-axis cylinder collider (PhysX convex-approximates the gprim)."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr("Z")
    r, h = float(radius), float(height) / 2
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h), Gf.Vec3f(r, r, h)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())


def _phys_material(stage, path: str, static: float, dynamic: float):
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


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the rack: KINEMATIC compound. Origin at the base centre on the floor;
    local +x = front (toward the robot). Two towers, the FIXED far rail with its two
    chock blocks (the display saddle), and the 4 bracket guide walls of the EMPTY
    near-rail slots on the tower tops. Grippy material everywhere (nothing here is
    meant to slide)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(20.0)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    grip = _phys_material(stage, f"{prim_path}/grip", c.mu_static, c.mu_dynamic)

    def box(name: str, center, size, color) -> None:
        _add_box(stage, f"{prim_path}/{name}", center=center, size=size,
                 color=color, collide=collide)
        _bind_material(stage.GetPrimAtPath(f"{prim_path}/{name}"), grip)

    # towers
    for sgn, nm in ((1.0, "l"), (-1.0, "r")):
        box(f"tower_{nm}", (0.0, sgn * c.tower_y, c.tower_h / 2),
            (c.tower_x, c.tower_t, c.tower_h), c.rack_color)
    # fixed FAR rail (top level with a seated bar's top) + its two chocks
    box("rail_far", (-c.rail_x, 0.0, c.tower_h + c.bar_s / 2),
        (c.bar_s, c.bar_len, c.bar_s), c.rail_color)
    for sgn, nm in ((1.0, "l"), (-1.0, "r")):
        box(f"chock_{nm}", (-c.rail_x, sgn * c.chock_y, c.tower_h + c.bar_s + c.chock_h / 2),
            (c.bar_s, c.chock_t, c.chock_h), c.chock_color)
    # bracket guide walls of the EMPTY near-rail slots (open-top, slot centred at
    # x=+rail_x, width slot_w, floor = the tower top)
    for wx, wnm in ((c.rail_x - c.slot_w / 2 - c.wall_t / 2, "f"),
                    (c.rail_x + c.slot_w / 2 + c.wall_t / 2, "n")):
        for sgn, tnm in ((1.0, "l"), (-1.0, "r")):
            box(f"wall_{wnm}{tnm}", (wx, sgn * c.tower_y, c.tower_h + c.wall_h / 2),
                (c.wall_t, c.tower_t, c.wall_h), c.chock_color)
    return root


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the bottle: DYNAMIC compound (body + neck along local +z), origin at
    the BODY centre — MassAPI puts the CoM there, so the horizontal racked rest is
    symmetric about the body regardless of neck direction. Damping + solver
    iterations so the cylinder-on-rails rest settles (GPU cylinder-creep guard)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.30)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    collide = _make_collide(cfg.contact_offset)
    grip = _phys_material(stage, f"{prim_path}/grip", cfg.mu_static, cfg.mu_dynamic)
    c = cfg
    _add_cyl(stage, f"{prim_path}/body", center=(0.0, 0.0, 0.0),
             radius=c.r_body, height=c.h_body, color=c.body_color, collide=collide)
    _add_cyl(stage, f"{prim_path}/neck",
             center=(0.0, 0.0, c.h_body / 2 + c.h_neck / 2),
             radius=c.r_neck, height=c.h_neck, color=c.body_color, collide=collide)
    for child in ("body", "neck"):
        _bind_material(stage.GetPrimAtPath(f"{prim_path}/{child}"), grip)
    return root


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rack" not in _SPAWNER_CACHE:

        @configclass
        class RackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            tower_x: float = 0.20
            tower_t: float = 0.024
            tower_y: float = 0.15
            tower_h: float = 0.12
            rail_x: float = 0.07
            bar_s: float = 0.025
            bar_len: float = 0.38
            chock_y: float = 0.045
            chock_t: float = 0.012
            chock_h: float = 0.012
            slot_w: float = 0.045
            wall_t: float = 0.012
            wall_h: float = 0.035
            rack_color: tuple = (0.45, 0.30, 0.16)
            rail_color: tuple = (0.72, 0.55, 0.30)
            chock_color: tuple = (0.16, 0.12, 0.08)
            mu_static: float = 0.60
            mu_dynamic: float = 0.50
            contact_offset: float = 0.002

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            r_body: float = 0.0275
            h_body: float = 0.190
            r_neck: float = 0.010
            h_neck: float = 0.070
            mass: float = 0.45
            body_color: tuple = (0.10, 0.32, 0.12)
            mu_static: float = 0.60
            mu_dynamic: float = 0.50
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(rack=RackSpawnerCfg, bottle=BottleSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class MissingRailRackSceneCfg(BaseCfg):
    """Config for `MissingRailRackScene`. The interlocks are metric and asserted in
    __post_init__: the success x-window lies entirely on the near side of the fixed
    rail (so a bottle at any goal-window pose WITHOUT the crossbar is one-sided and
    tips through), the slot admits the bar with ~10 mm of play per end, the seated
    bar's top is level with the fixed rail's top by construction, the chock gap
    admits the body with ~11 mm of play per side, and both graspable bodies fit the
    80 mm Franka jaw."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.60)  # max |ang vel| when judging (rad/s)
    bar_x_tol: float = tunable(0.014)  # bar seated: |x - rail_x| bound (m)
    bar_y_tol: float = tunable(0.040)  # bar seated: |y| bound (ends stay over towers)
    bar_z_tol: float = tunable(0.008)  # bar seated: |z - seat_z| bound (m)
    bar_align_deg: float = tunable(8.0)  # bar axis within this of the rack y-axis
    x_tol: float = tunable(0.025)  # bottle racked: |x| bound (between the rails)
    y_tol: float = tunable(0.030)  # bottle racked: |y| bound (in the chock saddle)
    z_tol: float = tunable(0.012)  # bottle racked: |z - rest_z| bound (m)
    level_max_deg: float = tunable(15.0)  # bottle axis within this of horizontal
    align_max_deg: float = tunable(25.0)  # bottle axis within this of the rack x-axis
    cradle_x: float = tunable(0.045)  # cradle latch: loose |x| band
    cradle_y: float = tunable(0.050)  # cradle latch: loose |y| band
    cradle_z: float = tunable(0.020)  # cradle latch: loose z band
    cradle_level_deg: float = tunable(30.0)  # cradle latch: loose level band
    bar_lift_z: float = tunable(0.080)  # bar lift latch (floor rest is 0.0125)
    bottle_lift_z: float = tunable(0.155)  # bottle lift latch (stand is 0.095)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    rack_jitter: float = tunable(0.03)  # rack xy jitter (+/- m)
    rack_yaw_deg: float = tunable(12.0)  # rack yaw about front-facing (+/- deg)
    bar_jitter: float = tunable(0.025)  # bar spawn xy jitter (+/- m)
    bar_yaw_deg: float = tunable(25.0)  # bar spawn yaw jitter (+/- deg)
    bottle_jitter: float = tunable(0.03)  # bottle spawn xy jitter (+/- m)
    swap_sides: bool = tunable(True)  # Bernoulli side sign (bar one side, bottle other)

    # --- info: layout (single Franka base at the origin; radii ~0.2-0.7 m) ----------------------
    rack_x: float = info(0.62)  # rack centre distance from the base
    bar_slot: tuple = info((0.26, 0.18))  # bar spawn (x, +/-y by the side sign)
    bottle_slot: tuple = info((0.28, 0.18))  # bottle spawn (x, -/+y opposite side)

    # --- info: rack structure --------------------------------------------------------------------
    tower_x: float = info(0.20)  # tower footprint along rack x
    tower_t: float = info(0.024)  # tower thickness (y)
    tower_y: float = info(0.15)  # tower centres at y = +/- tower_y
    tower_h: float = info(0.12)  # tower height = bracket slot floor
    rail_x: float = info(0.07)  # rails at x = -rail_x (fixed far) / +rail_x (brackets)
    chock_y: float = info(0.045)  # chock centres at y = +/- chock_y on the far rail
    chock_t: float = info(0.012)
    chock_h: float = info(0.012)
    slot_w: float = info(0.045)  # bracket slot width along x (bar 0.025 -> ~10 mm play)
    wall_t: float = info(0.012)
    wall_h: float = info(0.035)  # guide walls rise to tower_h + wall_h
    rack_color: tuple = info((0.45, 0.30, 0.16))
    rail_color: tuple = info((0.72, 0.55, 0.30))
    chock_color: tuple = info((0.16, 0.12, 0.08))

    # --- info: bar + bottle ----------------------------------------------------------------------
    bar_s: float = info(0.025)  # square section — fits the 80 mm Franka jaw
    bar_len: float = info(0.38)
    bar_mass: float = info(0.25)
    bar_color: tuple = info((0.72, 0.55, 0.30))  # same wood as the fixed rail
    r_body: float = info(0.0275)  # 55 mm body — fits the 80 mm Franka jaw
    h_body: float = info(0.190)
    r_neck: float = info(0.010)
    h_neck: float = info(0.070)
    bottle_mass: float = info(0.45)
    body_color: tuple = info((0.10, 0.32, 0.12))
    mu_static: float = info(0.60)  # grippy: rests must hold, nothing needs to slide
    mu_dynamic: float = info(0.50)

    contact_offset: float = info(0.002)
    # rubric weights (0.10 + 0.30 + 0.10 + 0.10 = 0.60 = the non-success cap)
    w_bar_lift: float = info(0.10)
    w_bar_seat: float = info(0.30)
    w_bottle_lift: float = info(0.10)
    w_cradle: float = info(0.10)

    # Derived (filled in __post_init__).
    rail_top_z: float = field(default=0.0, init=False)  # top plane of both rails
    seat_z: float = field(default=0.0, init=False)  # seated bar origin height
    rest_z: float = field(default=0.0, init=False)  # racked bottle origin height
    stand_z: float = field(default=0.0, init=False)  # standing bottle origin height
    bar_floor_z: float = field(default=0.0, init=False)  # bar-on-floor origin height

    def __post_init__(self) -> None:
        self.rail_top_z = self.tower_h + self.bar_s  # seated bar top == fixed rail top
        self.seat_z = self.tower_h + self.bar_s / 2
        self.rest_z = self.rail_top_z + self.r_body
        self.stand_z = self.h_body / 2
        self.bar_floor_z = self.bar_s / 2
        # --- honesty-by-construction assertions -------------------------------------------------
        assert 2 * self.r_body <= 0.078, "bottle body too wide for the Franka jaw"
        assert self.bar_s <= 0.078, "bar too wide for the Franka jaw"
        # the missing-rail gap forces the order: every goal-window pose is on the
        # near side of the fixed rail, so without the bar it is one-side-supported
        assert self.x_tol < self.rail_x - self.bar_s / 2, \
            "goal x-window overlaps the fixed rail: seed strategy might balance"
        # both rails end up under the body across the whole goal window: even at the
        # worst window offset the body still reaches past the far rail's INNER edge
        assert self.x_tol + self.rail_x - self.bar_s / 2 <= self.h_body / 2 - 0.005, \
            "body cannot bridge both rails across the goal x-window"
        # a single-rail balance rest (origin over a rail) is outside the x-window
        assert self.rail_x >= self.x_tol + 0.025, "single-rail balance too close to x_tol"
        # bar fits the slot with real play, and the walls retain it once seated
        assert self.slot_w >= self.bar_s + 0.016, "bracket slot play too tight"
        assert self.wall_h >= self.bar_s + 0.008, "guide walls too low to retain the bar"
        assert self.bar_len / 2 >= self.tower_y + self.tower_t / 2 + 0.005, \
            "bar too short to span both towers"
        assert self.bar_y_tol < self.bar_len / 2 - (self.tower_y - self.tower_t / 2), \
            "bar_y_tol admits an end slipped off its tower"
        # chock saddle admits the body with play; a body parked OUTSIDE the chocks
        # is beyond the y-window
        gap = 2 * (self.chock_y - self.chock_t / 2)
        assert gap >= 2 * self.r_body + 0.016, "chock gap too tight for the body"
        assert self.chock_y + self.chock_t / 2 + self.r_body > self.y_tol + 0.02, \
            "a rest against the chock outer face would pass the y-window"
        # the horizontal neck clears the chock tops and the guide-wall tops
        assert self.rest_z - self.r_neck >= self.rail_top_z + self.chock_h + 0.004, \
            "racked neck fouls the chocks"
        assert self.rest_z - self.r_neck >= self.tower_h + self.wall_h + 0.004, \
            "racked neck fouls the bracket walls"
        # lift latches sit between the floor states and the goal states
        assert self.bar_floor_z + 0.03 < self.bar_lift_z < self.seat_z, "bar lift latch"
        assert self.stand_z + 0.03 < self.bottle_lift_z < self.rest_z, "bottle lift latch"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("missing_rail_rack")
class MissingRailRackScene(BaseScene):
    cfg: MissingRailRackSceneCfg

    def __init__(self, cfg: MissingRailRackSceneCfg | None = None) -> None:
        super().__init__(cfg or MissingRailRackSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        rack_spawn = spawners["rack"](
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            tower_x=c.tower_x, tower_t=c.tower_t, tower_y=c.tower_y, tower_h=c.tower_h,
            rail_x=c.rail_x, bar_s=c.bar_s, bar_len=c.bar_len,
            chock_y=c.chock_y, chock_t=c.chock_t, chock_h=c.chock_h,
            slot_w=c.slot_w, wall_t=c.wall_t, wall_h=c.wall_h,
            rack_color=c.rack_color, rail_color=c.rail_color, chock_color=c.chock_color,
            mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
            contact_offset=c.contact_offset,
        )
        bottle_spawn = spawners["bottle"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.bottle_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            r_body=c.r_body, h_body=c.h_body, r_neck=c.r_neck, h_neck=c.h_neck,
            mass=c.bottle_mass, body_color=c.body_color,
            mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
            contact_offset=c.contact_offset,
        )
        bar_spawn = sim_utils.CuboidCfg(
            size=(c.bar_s, c.bar_len, c.bar_s),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.bar_color),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=c.mu_static, dynamic_friction=c.mu_dynamic,
                restitution=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=c.contact_offset, rest_offset=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.10, angular_damping=0.30,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=c.bar_mass),
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
            "rack": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=rack_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_x, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),  # yaw pi
            ),
            "bar": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Crossbar",
                spawn=bar_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bar_slot[0], c.bar_slot[1], c.bar_floor_z + 0.002)),
            ),
            "bottle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle",
                spawn=bottle_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bottle_slot[0], -c.bottle_slot[1], c.stand_z + 0.002)),
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
        self.rack: RigidObject = env.iscene["rack"]
        self.bar: RigidObject = env.iscene["bar"]
        self.bottle: RigidObject = env.iscene["bottle"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements
        self._bar_lift_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._bar_seat_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._bottle_lift_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._cradle_ever = torch.zeros(n, dtype=torch.bool, device=dev)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: rack re-posed (xy jitter + yaw about front-facing); a
        Bernoulli side sign puts the loose bar on one side of the centreline and the
        bottle on the other (both with jitter; bar yaw jitter, bottle free yaw);
        latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(m, device=dev)  # burn the degenerate first post-seed draw

        # --- rack (kinematic): front faces the robot (yaw pi) + jitter/yaw ---
        yaw = math.pi + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rack_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.rack_x
        st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.rack_jitter
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.rack.write_root_state_to_sim(st, env_ids)

        # --- side sign: bar on one side of the centreline, bottle on the other ---
        if c.swap_sides:
            s_sgn = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        else:
            s_sgn = torch.ones(m, device=dev)

        # --- bar: lying flat on the floor, long axis ~along world y + yaw jitter ---
        byaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.bar_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.bar_slot[0]
        st[:, 1] = s_sgn * c.bar_slot[1]
        st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.bar_jitter
        st[:, 2] = c.bar_floor_z + 0.002
        st[:, 3], st[:, 6] = torch.cos(byaw / 2), torch.sin(byaw / 2)
        st[:, 0:3] += origin
        self.bar.write_root_state_to_sim(st, env_ids)

        # --- bottle: standing upright on the opposite side + jitter + free yaw ---
        oyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.bottle_slot[0]
        st[:, 1] = -s_sgn * c.bottle_slot[1]
        st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.bottle_jitter
        st[:, 2] = c.stand_z + 0.002
        st[:, 3], st[:, 6] = torch.cos(oyaw / 2), torch.sin(oyaw / 2)
        st[:, 0:3] += origin
        self.bottle.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._bar_lift_ever[env_ids] = False
        self._bar_seat_ever[env_ids] = False
        self._bottle_lift_ever[env_ids] = False
        self._cradle_ever[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "rack": self.rack.data.root_state_w[env_ids].clone(),
            "bar": self.bar.data.root_state_w[env_ids].clone(),
            "bottle": self.bottle.data.root_state_w[env_ids].clone(),
            "bar_lift_ever": self._bar_lift_ever[env_ids].clone(),
            "bar_seat_ever": self._bar_seat_ever[env_ids].clone(),
            "bottle_lift_ever": self._bottle_lift_ever[env_ids].clone(),
            "cradle_ever": self._cradle_ever[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.rack.write_root_state_to_sim(state["rack"], env_ids)
        self.bar.write_root_state_to_sim(state["bar"], env_ids)
        self.bottle.write_root_state_to_sim(state["bottle"], env_ids)
        self._bar_lift_ever[env_ids] = state["bar_lift_ever"]
        self._bar_seat_ever[env_ids] = state["bar_seat_ever"]
        self._bottle_lift_ever[env_ids] = state["bottle_lift_ever"]
        self._cradle_ever[env_ids] = state["cradle_ever"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden DISPLAY RACK stands on the floor in front of the robot: two "
            f"side TOWERS ({c.tower_h * 100:.0f} cm tall, {2 * c.tower_y * 100:.0f} cm "
            f"apart) carrying ONE fixed horizontal RAIL across their tops on the FAR "
            f"side, with two small dark CHOCK blocks on that rail marking the display "
            f"saddle. The rack's NEAR rail is MISSING: on each tower top, at the near "
            f"rail position ({2 * c.rail_x * 100:.0f} cm in front of the fixed rail), "
            f"an EMPTY open-top BRACKET SLOT ({c.slot_w * 1000:.0f} mm wide, between "
            f"two short dark guide walls) waits for it. The missing CROSSBAR — a "
            f"{c.bar_s * 1000:.0f} mm square wooden bar, {c.bar_len * 100:.0f} cm "
            f"long, the same light wood as the fixed rail — lies on the floor on one "
            f"side; a green glass WINE BOTTLE ({2 * c.r_body * 1000:.0f} mm body, "
            f"{(c.h_body + c.h_neck) * 100:.0f} cm long with a "
            f"{2 * c.r_neck * 1000:.0f} mm neck) stands on the floor on the other "
            f"side.\n"
            f"Goal: REPAIR the rack, then RACK the bottle. First seat the crossbar "
            f"into BOTH bracket slots so it spans the two towers — dropped in from "
            f"above between each slot's guide walls, it rests on the tower tops with "
            f"its top level with the fixed rail. Then lay the bottle HORIZONTAL "
            f"across the two rails, centred on the chock saddle: body between the "
            f"two chocks, axis square across the rails (the neck may point toward "
            f"either rail), resting bridging the fixed rail and the crossbar. The "
            f"order is physical, not conventional: laid at the goal spot before the "
            f"crossbar is seated, the bottle is supported on one side only and tips "
            f"through the missing-rail gap to the floor. A bottle standing on the "
            f"floor or on the rack, lying off the saddle (outside the chocks or off "
            f"the rails), or bridging onto a crossbar that is NOT seated in its "
            f"brackets — all failure."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Seat the loose wooden crossbar into the rack's two empty bracket slots "
            "so it spans the towers, then lay the wine bottle horizontally across "
            "the fixed rail and the crossbar, centred between the two chock blocks. "
            "The bottle falls through if the crossbar is not seated first."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def _rack_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N, 3) world points in the rack frame (+x front, +y across, z up)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.rack.data.root_quat_w,
                                  pos_w - self.rack.data.root_pos_w)

    def _bar_loc(self) -> torch.Tensor:
        return self._rack_local(self.bar.data.root_pos_w)

    def _bottle_loc(self) -> torch.Tensor:
        return self._rack_local(self.bottle.data.root_pos_w)

    def _bar_align(self) -> torch.Tensor:
        """(N,) |cos| between the bar's long axis (local +y) and the rack's y-axis."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ey = torch.tensor([0.0, 1.0, 0.0], device=self.env.device).expand(n, 3)
        a_w = quat_apply(self.bar.data.root_quat_w, ey)
        r_w = quat_apply(self.rack.data.root_quat_w, ey)
        return (a_w * r_w).sum(dim=-1).abs().clamp(max=1.0)

    def _bottle_axis_local(self) -> torch.Tensor:
        """(N, 3) the bottle's +z axis (body->neck) expressed in the rack frame."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        a_w = quat_apply(self.bottle.data.root_quat_w, ez)
        return quat_apply_inverse(self.rack.data.root_quat_w, a_w)

    def bar_seated(self) -> torch.Tensor:
        """(N,) bool, geometric: the bar seated in its brackets — origin at the
        near-rail position within tolerances, long axis along the rack y-axis, in
        the rack frame."""
        c = self.cfg
        loc = self._bar_loc()
        x_ok = (loc[:, 0] - c.rail_x).abs() <= c.bar_x_tol
        y_ok = loc[:, 1].abs() <= c.bar_y_tol
        z_ok = (loc[:, 2] - c.seat_z).abs() <= c.bar_z_tol
        a_ok = self._bar_align() >= math.cos(math.radians(c.bar_align_deg))
        return x_ok & y_ok & z_ok & a_ok

    def bottle_racked(self) -> torch.Tensor:
        """(N,) bool, geometric: the bottle resting horizontal across the two rails
        centred on the chock saddle — origin in the goal window at rail rest height,
        axis level and square across the rails, in the rack frame."""
        c = self.cfg
        loc = self._bottle_loc()
        a = self._bottle_axis_local()
        x_ok = loc[:, 0].abs() <= c.x_tol
        y_ok = loc[:, 1].abs() <= c.y_tol
        z_ok = (loc[:, 2] - c.rest_z).abs() <= c.z_tol
        level_ok = a[:, 2].abs() <= math.sin(math.radians(c.level_max_deg))
        align_ok = a[:, 0].abs() >= math.cos(math.radians(c.align_max_deg))
        return x_ok & y_ok & z_ok & level_ok & align_ok

    def _cradled_loose(self) -> torch.Tensor:
        """(N,) bool: loose saddle bands (the cradle latch), gated on the bar being
        seated at that instant — bridging onto a mis-laid bar earns nothing."""
        c = self.cfg
        loc = self._bottle_loc()
        a = self._bottle_axis_local()
        x_ok = loc[:, 0].abs() <= c.cradle_x
        y_ok = loc[:, 1].abs() <= c.cradle_y
        z_ok = (loc[:, 2] - c.rest_z).abs() <= c.cradle_z
        level_ok = a[:, 2].abs() <= math.sin(math.radians(c.cradle_level_deg))
        return x_ok & y_ok & z_ok & level_ok & self.bar_seated()

    def _settled(self) -> torch.Tensor:
        c = self.cfg
        return ((self.bottle.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                & (self.bottle.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega)
                & (self.bar.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                & (self.bar.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega))

    def _update_latches(self) -> None:
        c = self.cfg
        bar_z = (self.bar.data.root_pos_w - self.env_origins)[:, 2]
        bot_z = (self.bottle.data.root_pos_w - self.env_origins)[:, 2]
        self._bar_lift_ever |= bar_z >= c.bar_lift_z
        self._bar_seat_ever |= self.bar_seated()
        self._bottle_lift_ever |= bot_z >= c.bottle_lift_z
        self._cradle_ever |= self._cradled_loose()

    # ----- step-coupled bookkeeping (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No plant (the rack is kinematic and jointless) — just latch rubric
        progress every step so transient achievements keep credit."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: the crossbar seated in its brackets AND the bottle resting
        horizontal across both rails centred on the chock saddle AND both bodies at
        rest. Physical, settled outcomes only."""
        self._update_latches()
        return self.bar_seated() & self.bottle_racked() & self._settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10*bar_lift + 0.30*bar_seat + 0.10*bottle_lift +
        0.10*cradle (loose bands, gated on the bar being seated) — all latched, ~0
        for doing nothing, capped 0.60 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_bar_lift * self._bar_lift_ever.float()
                + c.w_bar_seat * self._bar_seat_ever.float()
                + c.w_bottle_lift * self._bottle_lift_ever.float()
                + c.w_cradle * self._cradle_ever.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="missing_rail_rack", robot="null"))
