"""CoinOpWasherScene — pay-to-unlock laundromat washer: drop the BRASS token into
the pay-lever's coin tray (tipping the gravity rocker and lifting its lock tab off
the sliding door), slide the door open, get the laundry ball out of the sealed
chamber, and deliver it into the collection basket. The gray SLUG is too light to
tip the lever — physics, not the rubric, rejects it.

Derived from the RLBench `open_washing_machine` seed but STRATEGICALLY DIFFERENT
(see TASK.md): the seed's whole plan is ONE pull on ONE articulation whose end
pose IS the goal. Here there are TWO articulations (a gravity rocker lever and a
sliding door) and neither one's end pose is the goal: the door is LOCKED at spawn
by the rocker's tab, and the only way to unlock it is to load the correct payload
(the heavy token, not the light slug) into the rocker's coin tray so gravity tips
it past its stop and swings the tab clear. Opening the door is then merely
instrumental — the scored object is the free laundry ball sealed inside the
chamber, which must end up resting in the external basket, with the token still
riding the paid-down tray. Shoving the locked door harder does NOT help: the tab
sits below the rocker pivot, so door contact torques the rocker INTO its locked
stop (verified by a strong-shove smoke probe).

Mechanics (compound rigid bodies via custom spawn funcs + spawn-authored joints):
  - housing: ONE kinematic compound (raised chamber floor, three walls, front wall
    with a doorway, lintel, roof, and the pay-post column) — never teleported, so
    it is a safe anchor for both joints;
  - door: dynamic compound (plate + handle bar) on a PrismaticJoint (housing ->
    door, local +y, limits [0, stroke]); linear damping parks it where released;
  - rocker: dynamic compound (beam, lock tab finger, coin tray pocket) on a
    RevoluteJoint (housing -> rocker, axis x, limits [paid_deg, locked_deg]);
    authored CoM offset toward the tab side makes gravity hold it LOCKED; the
    token's tray torque is ~3x the holding torque, the slug's ~0.4x;
  - token / slug: flat cylinders (same size, very different mass); ball: the
    laundry ball sealed in the chamber; basket: dynamic open box (teleported at
    reset for randomization).
`post_step` owns the wrench slots (`door_f`, `ball_f`, `rocker_tau`) and latches
rubric progress.

Rubric (graded 0..1, anchored in the demonstrated solve.py trajectory):
  success() = token resting in the coin tray (ROCKER body frame) AND rocker at
  its paid stop AND ball resting inside the basket (BASKET body frame, z-window
  rejects rim perching) AND everything settled.
  score() = 1.0 iff success(); else 0.15*token_latch (token seen at rest in the
  tray) + 0.15*paid_latch (rocker seen at the paid stop WITH the token aboard —
  a hand-press on the empty lever earns nothing) + 0.15*open_latch (door slid
  past open_thresh — geometry says this needs the tab lifted) + 0.15*out_latch
  (ball seen at rest outside the chamber on the table) + 0.15*in_basket (live).
  ~0 for the null policy; all latches pass slow-speed gates.

Per-episode randomization (readback-verified in smoke): token, slug and basket
table poses, ball start pose inside the chamber, door initial crack (0..2 mm —
far below the ~4 mm tab gap and the 60 mm open latch).

Heavy imports (isaaclab, pxr) are deferred so importing this module stays
app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable
from robobench.core.registries import ENVS

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- USD authoring helpers -----------------------------------------------------------------------
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable | None):
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(box.GetPrim())
    return box.GetPrim()


# ----- compound spawn funcs ------------------------------------------------------------------------
def _spawn_housing(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The washer housing: KINEMATIC compound. Local frame: origin at the footprint
    centre ON the table top, front face toward +x. Children: raised chamber floor,
    back/left/right walls, front wall split around the doorway (left panel, right
    panel, lintel, and the under-door sill is the floor's own front edge), roof
    (no top extraction), and the free-standing pay-post column that visually
    carries the rocker pivot."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    collide = _make_collide(c.contact_offset)
    hd, hw, t = c.depth / 2, c.width / 2, c.wall_t
    fh, wh = c.floor_h, c.roof_z  # floor top, wall/roof base height
    body = c.body_color
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, fh / 2),
             size=(c.depth, c.width, fh), color=(0.35, 0.36, 0.40), collide=collide)
    _add_box(stage, f"{prim_path}/back", center=(-hd + t / 2, 0.0, fh + (wh - fh) / 2),
             size=(t, c.width, wh - fh), color=body, collide=collide)
    for tag, sy in (("side_l", -1.0), ("side_r", 1.0)):
        _add_box(stage, f"{prim_path}/{tag}", center=(0.0, sy * (hw - t / 2), fh + (wh - fh) / 2),
                 size=(c.depth, t, wh - fh), color=body, collide=collide)
    # front wall around the doorway (doorway: |y| < door_open_w/2, floor_h..open_top)
    ow, ot = c.door_open_w / 2, c.open_top
    for tag, y0, y1 in (("front_l", -hw, -ow), ("front_r", ow, hw)):
        _add_box(stage, f"{prim_path}/{tag}",
                 center=(hd - t / 2, (y0 + y1) / 2, fh + (wh - fh) / 2),
                 size=(t, y1 - y0, wh - fh), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/lintel", center=(hd - t / 2, 0.0, ot + (wh - ot) / 2),
             size=(t, 2 * ow, wh - ot), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/roof", center=(0.0, 0.0, wh + c.roof_t / 2),
             size=(c.depth, c.width, c.roof_t), color=c.roof_color, collide=collide)
    # pay-post column (free-standing, outside the door's swept volume)
    _add_box(stage, f"{prim_path}/post",
             center=(c.post_x, c.post_y, c.post_h / 2),
             size=(c.post_w, c.post_w, c.post_h), color=c.post_color, collide=collide)
    return root


def _spawn_door(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The sliding door: dynamic compound (plate + handle bar) on a spawn-authored
    PrismaticJoint to the sibling housing (axis local +y, limits [0, stroke]).
    Linear damping parks it where released — no spring."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(c.lin_damping))
    pxrb.CreateAngularDampingAttr(5.0)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/plate", center=(0.0, 0.0, 0.0),
             size=(c.plate_t, c.plate_w, c.plate_h), color=c.plate_color, collide=collide)
    _add_box(stage, f"{prim_path}/handle",
             center=(c.plate_t / 2 + c.handle_len / 2, 0.0, c.handle_dz),
             size=(c.handle_len, c.handle_w, c.handle_h), color=c.handle_color, collide=collide)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/rail")
    j.CreateBody0Rel().SetTargets([f"{base}/Housing"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Y")
    j.CreateCollisionEnabledAttr(False)
    j.CreateLocalPos0Attr(Gf.Vec3f(float(c.anchor_x), 0.0, float(c.anchor_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(float(c.stroke))
    return root


def _spawn_rocker(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The pay lever: dynamic compound (beam along y, lock-tab finger hanging near
    the -y end, coin-tray pocket on the +y arm) on a spawn-authored RevoluteJoint
    to the sibling housing (axis x, limits [paid_deg, locked_deg]). Body origin =
    pivot. MassAPI authors mass, an explicit CoM offset toward the tab side (the
    gravity hold), and a diagonal inertia (custom spawners apply no cfg schemas)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(c.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, float(c.com_y), float(c.com_z)))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(8e-4, 3e-4, 8e-4))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.2)
    pxrb.CreateAngularDampingAttr(float(c.ang_damping))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    grey = c.beam_color
    # beam along y (body frame; origin = pivot)
    _add_box(stage, f"{prim_path}/beam", center=(0.0, 0.016, 0.0),
             size=(0.016, 0.192, 0.016), color=grey, collide=collide)
    # lock tab: reaches back in -x to overlap the door plane, hangs below the beam
    _add_box(stage, f"{prim_path}/tab", center=(float(c.tab_cx), float(c.tab_by), -0.024),
             size=(0.044, 0.012, 0.032), color=c.tab_color, collide=collide)
    # coin tray pocket on the +y arm: floor + four low walls (open top)
    ty = float(c.tray_by)
    _add_box(stage, f"{prim_path}/tray_floor", center=(0.0, ty, 0.011),
             size=(0.070, 0.070, 0.006), color=c.tray_color, collide=collide)
    for tag, cx, cy, sx, sy in (
        ("tray_f", 0.032, 0.0, 0.006, 0.070), ("tray_b", -0.032, 0.0, 0.006, 0.070),
        ("tray_l", 0.0, -0.032, 0.058, 0.006), ("tray_r", 0.0, 0.032, 0.058, 0.006),
    ):
        _add_box(stage, f"{prim_path}/{tag}", center=(cx, ty + cy, 0.024),
                 size=(sx, sy, 0.020), color=c.tray_color, collide=collide)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/pivot")
    j.CreateBody0Rel().SetTargets([f"{base}/Housing"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("X")
    j.CreateCollisionEnabledAttr(False)
    j.CreateLocalPos0Attr(Gf.Vec3f(float(c.anchor_x), float(c.anchor_y), float(c.anchor_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(float(c.paid_deg))
    j.CreateUpperLimitAttr(float(c.locked_deg))
    return root


def _spawn_basket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The collection basket: dynamic open-top box (floor + 4 walls). Dynamic so
    reset() can teleport it (randomization); it is NOT jointed."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(2.0)
    pxrb.CreateAngularDampingAttr(2.0)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    o, t, h = c.outer, c.wall_t, c.wall_h
    col = c.color
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, t / 2),
             size=(o, o, t), color=col, collide=collide)
    for tag, cx, cy, sx, sy in (
        ("w_f", o / 2 - t / 2, 0.0, t, o), ("w_b", -o / 2 + t / 2, 0.0, t, o),
        ("w_l", 0.0, -o / 2 + t / 2, o - 2 * t, t), ("w_r", 0.0, o / 2 - t / 2, o - 2 * t, t),
    ):
        _add_box(stage, f"{prim_path}/{tag}", center=(cx, cy, t + h / 2),
                 size=(sx, sy, h), color=col, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "housing" not in _SPAWNER_CACHE:

        @configclass
        class HousingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_housing)
            depth: float = 0.22
            width: float = 0.24
            wall_t: float = 0.012
            floor_h: float = 0.020
            door_open_w: float = 0.110
            open_top: float = 0.110
            roof_z: float = 0.150
            roof_t: float = 0.012
            post_x: float = 0.150
            post_y: float = 0.148
            post_w: float = 0.030
            post_h: float = 0.125
            body_color: tuple = (0.85, 0.86, 0.90)
            roof_color: tuple = (0.30, 0.32, 0.38)
            post_color: tuple = (0.45, 0.30, 0.15)
            contact_offset: float = 0.002

        @configclass
        class DoorSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_door)
            plate_t: float = 0.010
            plate_w: float = 0.130
            plate_h: float = 0.100
            handle_len: float = 0.034
            handle_w: float = 0.060
            handle_h: float = 0.016
            handle_dz: float = 0.010
            mass: float = 0.40
            lin_damping: float = 4.0
            anchor_x: float = 0.117
            anchor_z: float = 0.065
            stroke: float = 0.080
            plate_color: tuple = (0.20, 0.45, 0.70)
            handle_color: tuple = (0.15, 0.15, 0.18)
            contact_offset: float = 0.002

        @configclass
        class RockerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rocker)
            mass: float = 0.10
            com_y: float = -0.015
            com_z: float = -0.005
            ang_damping: float = 2.0
            tab_cx: float = -0.016
            tab_by: float = -0.071
            tray_by: float = 0.080
            paid_deg: float = -20.0
            locked_deg: float = 8.0
            anchor_x: float = 0.150
            anchor_y: float = 0.148
            anchor_z: float = 0.145
            beam_color: tuple = (0.70, 0.70, 0.74)
            tab_color: tuple = (0.85, 0.20, 0.15)
            tray_color: tuple = (0.90, 0.70, 0.15)
            contact_offset: float = 0.002

        @configclass
        class BasketSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_basket)
            outer: float = 0.106
            wall_t: float = 0.008
            wall_h: float = 0.045
            mass: float = 0.25
            color: tuple = (0.55, 0.40, 0.20)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["housing"] = HousingSpawnerCfg
        _SPAWNER_CACHE["door"] = DoorSpawnerCfg
        _SPAWNER_CACHE["rocker"] = RockerSpawnerCfg
        _SPAWNER_CACHE["basket"] = BasketSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class CoinOpWasherSceneCfg(BaseCfg):
    """Config for `CoinOpWasherScene`. Geometry and the torque budget are derived
    and ASSERTED once in `__post_init__` so the scene, smoke AND solver read the
    same numbers."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    open_thresh: float = tunable(0.060)  # door slide past this latches "opened" (m)
    paid_thresh_deg: float = tunable(-16.0)  # rocker angle below this = paid (deg)
    locked_thresh_deg: float = tunable(2.0)  # rocker angle above this = still locked (deg)
    out_x: float = tunable(0.100)  # ball env-x past this (on the table) = out of chamber
    slow_gate: float = tunable(0.10)  # m/s: bodies count as slow below this (latch gates)
    settle_lin: float = tunable(0.05)  # max |lin vel| when judging success (m/s)

    # --- tunable: randomization ------------------------------------------------------------------
    crack_max: float = tunable(0.002)  # door initial slide sampled in [0, this] (m)
    token_x_range: tuple = tunable((0.20, 0.32))
    token_y_range: tuple = tunable((0.03, 0.15))
    slug_y_range: tuple = tunable((-0.15, -0.03))  # same x band as the token
    ball_x_range: tuple = tunable((-0.110, -0.020))  # env-local, inside the chamber
    ball_y_range: tuple = tunable((-0.055, 0.055))
    basket_x_range: tuple = tunable((0.20, 0.32))
    basket_y_range: tuple = tunable((-0.36, -0.26))

    # --- tunable: plant --------------------------------------------------------------------------
    token_mass: float = tunable(0.060)  # ~3.2x the rocker holding torque at the tray arm
    slug_mass: float = tunable(0.008)  # ~0.4x — physically cannot tip the lever
    rocker_mass: float = tunable(0.10)
    rocker_com_y: float = tunable(-0.015)  # gravity-hold CoM offset (tab side)
    door_mass: float = tunable(0.40)
    door_damping: float = tunable(4.0)
    ball_mass: float = tunable(0.060)

    # --- info: structure (env-local coordinates) -------------------------------------------------
    table_center: tuple = info((0.10, 0.0, 0.36))
    table_size: tuple = info((1.00, 1.00, 0.08))  # top at z = 0.40
    housing_pos: tuple = info((-0.05, 0.0))  # housing origin on the table top
    housing_depth: float = info(0.22)
    housing_width: float = info(0.24)
    wall_t: float = info(0.012)
    floor_h: float = info(0.020)
    door_open_w: float = info(0.110)  # doorway width in the front wall
    open_top: float = info(0.110)  # doorway top (above table top)
    roof_z: float = info(0.150)
    door_plate_w: float = info(0.130)
    door_plate_h: float = info(0.100)
    door_plate_t: float = info(0.010)
    door_standoff: float = info(0.002)  # gap between wall face and door plate
    door_stroke: float = info(0.080)
    door_z0: float = info(0.065)  # door body-origin height above the table top
    handle_len: float = info(0.034)
    handle_w: float = info(0.060)
    pivot: tuple = info((0.100, 0.148, 0.145))  # rocker pivot, env-local xy + z above table
    tab_by: float = info(-0.071)  # tab centre y in the ROCKER body frame (= lever arm)
    tab_z_top: float = info(-0.008)  # tab top in rocker body frame
    tab_z_bot: float = info(-0.040)  # tab bottom in rocker body frame
    tray_by: float = info(0.080)  # tray centre y in the rocker body frame (= token arm)
    tray_floor_top: float = info(0.014)  # rocker body frame
    tray_inner: float = info(0.058)
    tray_wall_top: float = info(0.034)
    paid_deg: float = info(-20.0)
    locked_deg: float = info(8.0)
    post_y: float = info(0.148)
    post_x_off: float = info(0.040)  # post/pivot x offset from the housing face
    post_w: float = info(0.030)
    post_h: float = info(0.125)
    token_r: float = info(0.015)
    token_h: float = info(0.008)
    ball_r: float = info(0.0225)
    basket_outer: float = info(0.106)
    basket_wall_t: float = info(0.008)
    basket_wall_h: float = info(0.045)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    table_top_z: float = field(default=None, init=False)
    face_x: float = field(default=None, init=False)  # front wall outer plane (env x)
    door_x: float = field(default=None, init=False)  # door body-origin x (env)
    door_z: float = field(default=None, init=False)  # door body-origin z (env)
    door_lead_y: float = field(default=None, init=False)  # door leading (+y) edge, shut
    pivot_env: tuple = field(default=None, init=False)  # rocker pivot (env, absolute z)
    tray_drop_z: float = field(default=None, init=False)  # free-air token drop height (env z)
    window_y: tuple = field(default=None, init=False)  # exposed doorway span at full stroke
    ball_z0: float = field(default=None, init=False)  # ball rest height on the chamber floor

    def __post_init__(self) -> None:
        c = self
        self.table_top_z = c.table_center[2] + c.table_size[2] / 2
        self.face_x = c.housing_pos[0] + c.housing_depth / 2
        self.door_x = self.face_x + c.door_standoff + c.door_plate_t / 2
        self.door_z = self.table_top_z + c.door_z0
        self.door_lead_y = c.door_plate_w / 2
        self.pivot_env = (c.pivot[0], c.pivot[1], self.table_top_z + c.pivot[2])
        self.tray_drop_z = self.pivot_env[2] + c.tray_floor_top + 0.030
        # exposed doorway window at full stroke: from -doorway/2 to door trailing edge
        self.window_y = (-c.door_open_w / 2, -c.door_plate_w / 2 + c.door_stroke)
        self.ball_z0 = self.table_top_z + c.floor_h + c.ball_r + 0.002

        # ---- torque budget (the pay mechanism) ----
        g = 9.81
        tau_hold = c.rocker_mass * g * abs(c.rocker_com_y)
        tau_token = c.token_mass * g * c.tray_by * math.cos(math.radians(abs(c.paid_deg)))
        tau_slug = c.slug_mass * g * c.tray_by
        assert tau_token > 2.5 * tau_hold, (tau_token, tau_hold)
        assert tau_slug < 0.6 * tau_hold, (tau_slug, tau_hold)
        # door shove torque sign: tab centre below the pivot -> +tau_x = INTO the locked stop
        tab_zc = (c.tab_z_top + c.tab_z_bot) / 2
        assert tab_zc < -0.02, tab_zc

        # ---- lock geometry ----
        a = abs(c.tab_by)
        door_top = self.door_z + c.door_plate_h / 2
        tab_bot_locked = self.pivot_env[2] + c.tab_z_bot - a * math.sin(math.radians(c.locked_deg))
        tab_bot_paid = self.pivot_env[2] + c.tab_z_bot + a * math.sin(math.radians(-c.paid_deg))
        assert door_top - tab_bot_locked > 0.015, (door_top, tab_bot_locked)  # locked overlap
        assert tab_bot_paid - door_top > 0.010, (tab_bot_paid, door_top)  # paid clearance
        tab_near = c.pivot[1] + c.tab_by - 0.006  # tab near (-y) face, env y
        assert tab_near - self.door_lead_y > c.crack_max + 0.002, (tab_near, self.door_lead_y)

        # ---- ball passage ----
        win_w = self.window_y[1] - self.window_y[0]
        assert win_w > 2 * c.ball_r + 0.010, win_w  # exposed window admits the ball
        assert c.open_top - c.floor_h > 2 * c.ball_r + 0.010  # doorway height
        door_bot = self.door_z - c.door_plate_h / 2
        assert door_bot - self.table_top_z < 2 * c.ball_r - 0.010  # no squeeze under the door
        assert c.open_thresh < c.door_stroke - 0.010  # open latch reachable before the stop

        # ---- clearances around the pay post ----
        handle_max_y = c.handle_w / 2 + c.door_stroke
        assert c.post_y - c.post_w / 2 - handle_max_y > 0.010  # handle never hits the post
        beam_bot_worst = self.pivot_env[2] - 0.008 - 0.015 * math.sin(math.radians(-c.paid_deg))
        assert beam_bot_worst - (self.table_top_z + c.post_h) > 0.004  # beam clears the post

        # ---- spawn bands don't collide ----
        assert c.token_y_range[0] - c.slug_y_range[1] > 2 * c.token_r + 0.005
        assert c.slug_y_range[0] - c.basket_y_range[1] > c.basket_outer / 2 + c.token_r + 0.005
        assert c.tray_inner > 2 * c.token_r + 0.010  # a flat drop lands in the pocket
        assert c.ball_x_range[1] + c.ball_r < c.housing_pos[0] + c.housing_depth / 2 - c.wall_t
        assert abs(c.ball_y_range[0]) + c.ball_r < c.housing_width / 2 - c.wall_t


# ----- scene ---------------------------------------------------------------------------------------
class CoinOpWasherScene(BaseScene):
    cfg: CoinOpWasherSceneCfg

    def __init__(self, cfg: CoinOpWasherSceneCfg | None = None) -> None:
        super().__init__(cfg or CoinOpWasherSceneCfg())

    # ----- assets --------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, kinematic table, the housing compound FIRST (both
        spawn-authored joints target their sibling), then door + rocker + basket
        compounds, the token/slug cylinders and the laundry ball."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        dyn = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, solver_position_iteration_count=32,
            solver_velocity_iteration_count=4, linear_damping=0.2, angular_damping=0.2,
            sleep_threshold=0.0, stabilization_threshold=0.0)

        housing_spawn = spawners["housing"](
            depth=c.housing_depth, width=c.housing_width, wall_t=c.wall_t, floor_h=c.floor_h,
            door_open_w=c.door_open_w, open_top=c.open_top, roof_z=c.roof_z,
            post_x=c.housing_depth / 2 + c.post_x_off, post_y=c.post_y, post_w=c.post_w,
            post_h=c.post_h, contact_offset=c.contact_offset)
        door_spawn = spawners["door"](
            plate_t=c.door_plate_t, plate_w=c.door_plate_w, plate_h=c.door_plate_h,
            handle_len=c.handle_len, handle_w=c.handle_w, mass=c.door_mass,
            lin_damping=c.door_damping, stroke=c.door_stroke,
            anchor_x=self.cfg.door_x - c.housing_pos[0], anchor_z=c.door_z0,
            contact_offset=c.contact_offset)
        rocker_spawn = spawners["rocker"](
            mass=c.rocker_mass, com_y=c.rocker_com_y, tab_by=c.tab_by, tray_by=c.tray_by,
            paid_deg=c.paid_deg, locked_deg=c.locked_deg,
            anchor_x=c.pivot[0] - c.housing_pos[0], anchor_y=c.pivot[1], anchor_z=c.pivot[2],
            contact_offset=c.contact_offset)
        basket_spawn = spawners["basket"](
            outer=c.basket_outer, wall_t=c.basket_wall_t, wall_h=c.basket_wall_h,
            contact_offset=c.contact_offset)

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "table": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=c.table_size, rigid_props=kin, collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.48, 0.35, 0.20))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.table_center)),
            "housing": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Housing", spawn=housing_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.housing_pos[0], c.housing_pos[1], c.table_top_z))),
            "door": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Door", spawn=door_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.door_x, 0.0, c.door_z))),
            "rocker": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rocker", spawn=rocker_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=self.cfg.pivot_env)),
            "basket": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Basket", spawn=basket_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.26, -0.31, c.table_top_z))),
            "token": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Token",
                spawn=sim_utils.CylinderCfg(
                    radius=c.token_r, height=c.token_h, axis="Z", rigid_props=dyn,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.token_mass),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.65, 0.13))),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.26, 0.09, c.table_top_z + c.token_h / 2 + 0.002))),
            "slug": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Slug",
                spawn=sim_utils.CylinderCfg(
                    radius=c.token_r, height=c.token_h, axis="Z", rigid_props=dyn,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.slug_mass),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.60, 0.60, 0.62))),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.26, -0.09, c.table_top_z + c.token_h / 2 + 0.002))),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5, solver_position_iteration_count=32,
                        solver_velocity_iteration_count=4, linear_damping=0.05,
                        angular_damping=0.05, sleep_threshold=0.0, stabilization_threshold=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.90, 0.95))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.06, 0.0, c.ball_z0))),
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
        n, dev = env.num_envs, env.device
        self.door: RigidObject = env.iscene["door"]
        self.rocker: RigidObject = env.iscene["rocker"]
        self.basket: RigidObject = env.iscene["basket"]
        self.token: RigidObject = env.iscene["token"]
        self.slug: RigidObject = env.iscene["slug"]
        self.ball: RigidObject = env.iscene["ball"]
        self.env_origins = env.iscene.env_origins
        # Episode state (readback targets for smoke).
        self.crack0 = torch.zeros(n, device=dev)
        self.token_start = torch.zeros(n, 3, device=dev)
        self.slug_start = torch.zeros(n, 3, device=dev)
        self.ball_start = torch.zeros(n, 3, device=dev)
        self.basket_start = torch.zeros(n, 3, device=dev)
        # Rubric latches.
        self.token_latch = torch.zeros(n, device=dev)
        self.paid_latch = torch.zeros(n, device=dev)
        self.open_latch = torch.zeros(n, device=dev)
        self.out_latch = torch.zeros(n, device=dev)
        # Wrench slots (solve/smoke write; post_step consumes — never call
        # set_external_force_and_torque directly).
        self.door_f = torch.zeros(n, device=dev)  # force on the door along +y (N)
        self.ball_f = torch.zeros(n, 3, device=dev)  # force on the ball (N)
        self.rocker_tau = torch.zeros(n, device=dev)  # probe torque about x (N*m)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample token/slug/basket table poses, the ball's chamber
        pose and the door crack; park the rocker exactly at its locked stop; clear
        latches and drives. torch.rand comparisons only (first randint after
        manual_seed is degenerate on this stack)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def band(lo: float, hi: float) -> torch.Tensor:
            return lo + torch.rand(m, device=dev) * (hi - lo)

        crack = band(0.0, c.crack_max)
        tx, ty = band(*c.token_x_range), band(*c.token_y_range)
        sx, sy = band(*c.token_x_range), band(*c.slug_y_range)
        bx, by = band(*c.ball_x_range), band(*c.ball_y_range)
        kx, ky = band(*c.basket_x_range), band(*c.basket_y_range)

        self.crack0[env_ids] = crack
        for latch in (self.token_latch, self.paid_latch, self.open_latch, self.out_latch):
            latch[env_ids] = 0.0
        self.door_f[env_ids] = 0.0
        self.ball_f[env_ids] = 0.0
        self.rocker_tau[env_ids] = 0.0

        def write(body, px, py, pz, quat=None):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1], st[:, 2] = px, py, pz
            if quat is None:
                st[:, 3] = 1.0
            else:
                st[:, 3:7] = quat
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # door: shut + sampled crack along +y
        write(self.door, torch.full((m,), c.door_x, device=dev), crack,
              torch.full((m,), c.door_z, device=dev))
        # rocker: exactly at the locked stop (rotation +locked_deg about x)
        half = math.radians(c.locked_deg) / 2
        rq = torch.zeros(m, 4, device=dev)
        rq[:, 0], rq[:, 1] = math.cos(half), math.sin(half)
        write(self.rocker, torch.full((m,), c.pivot_env[0], device=dev),
              torch.full((m,), c.pivot_env[1], device=dev),
              torch.full((m,), c.pivot_env[2], device=dev), rq)
        zc = c.table_top_z + c.token_h / 2 + 0.002
        write(self.token, tx, ty, torch.full((m,), zc, device=dev))
        write(self.slug, sx, sy, torch.full((m,), zc, device=dev))
        write(self.ball, bx, by, torch.full((m,), c.ball_z0, device=dev))
        write(self.basket, kx, ky, torch.full((m,), c.table_top_z + 0.001, device=dev))

        self.token_start[env_ids] = torch.stack([tx, ty, torch.full_like(tx, zc)], dim=1)
        self.slug_start[env_ids] = torch.stack([sx, sy, torch.full_like(sx, zc)], dim=1)
        self.ball_start[env_ids] = torch.stack([bx, by, torch.full_like(bx, c.ball_z0)], dim=1)
        self.basket_start[env_ids] = torch.stack(
            [kx, ky, torch.full_like(kx, c.table_top_z + 0.001)], dim=1)

    # ----- readings ------------------------------------------------------------------------------
    def door_open(self) -> torch.Tensor:
        """(N,) door slide in m (0 = shut; the joint caps it at `door_stroke`)."""
        return self.door.data.root_pos_w[:, 1] - self.env_origins[:, 1]

    def rocker_angle(self) -> torch.Tensor:
        """(N,) rocker angle about x in rad (+ = locked side, - = paid side)."""
        q = self.rocker.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 1], q[:, 0])

    def _local(self, body_from: RigidObject, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(body_from.data.root_quat_w,
                                  pos_w - body_from.data.root_pos_w)

    def token_in_tray(self) -> torch.Tensor:
        """(N,) bool: token centre inside the coin-tray pocket, ROCKER body frame
        (rides the tipping lever); z window rejects rim perching."""
        c = self.cfg
        p = self._local(self.rocker, self.token.data.root_pos_w)
        half = c.tray_inner / 2 - 0.004
        return (p[:, 0].abs() < half) & ((p[:, 1] - c.tray_by).abs() < half) \
            & (p[:, 2] > c.tray_floor_top - 0.004) & (p[:, 2] < c.tray_wall_top)

    def paid_now(self) -> torch.Tensor:
        return self.rocker_angle() < math.radians(self.cfg.paid_thresh_deg)

    def ball_in_basket(self) -> torch.Tensor:
        """(N,) bool: ball centre inside the basket's inner box (BASKET body
        frame); z window rejects rim perching and fly-throughs."""
        c = self.cfg
        p = self._local(self.basket, self.ball.data.root_pos_w)
        half = c.basket_outer / 2 - c.basket_wall_t
        return (p[:, 0].abs() < half - 0.002) & (p[:, 1].abs() < half - 0.002) \
            & (p[:, 2] > c.basket_wall_t + 0.010) & (p[:, 2] < c.basket_wall_t + 0.042)

    def ball_out(self) -> torch.Tensor:
        """(N,) bool: ball on the table beyond the housing face (out of chamber)."""
        c = self.cfg
        p = self.ball.data.root_pos_w - self.env_origins
        return (p[:, 0] > c.out_x) & (p[:, 2] < c.table_top_z + 0.10)

    def settled(self) -> torch.Tensor:
        c = self.cfg
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for b in (self.ball, self.basket, self.token, self.rocker, self.door):
            ok &= b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        return ok

    def success(self) -> torch.Tensor:
        """(N,) bool: token resting in the tray + rocker at the paid stop + ball
        resting inside the basket + everything settled (current physical state)."""
        return self.token_in_tray() & self.paid_now() & self.ball_in_basket() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 1.0 iff success(); else the latch ladder
        0.15*(token + paid + open + out) + 0.15*in_basket (live). ~0 for the null
        policy; latched credit never evaporates under correct behavior."""
        partial = 0.15 * (self.token_latch + self.paid_latch + self.open_latch
                          + self.out_latch) + 0.15 * self.ball_in_basket().float()
        return torch.where(self.success(), torch.ones_like(partial), partial)

    # ----- step-coupled mechanics (every substep) -------------------------------------------------
    def post_step(self) -> None:
        """Apply the wrench buffers (owns the slots), then latch rubric progress
        through slow-speed gates."""
        n, dev = self.env.num_envs, self.env.device
        f = torch.zeros(n, 1, 3, device=dev)
        f[:, 0, 1] = self.door_f
        self.door.set_external_force_and_torque(f, torch.zeros(n, 1, 3, device=dev))
        # ball_f is WORLD-frame by contract; the wrench call applies in the BODY
        # frame and a ROLLING sphere's body frame revolves once per pi*d of travel,
        # so re-encode per step (rolling-ball frame-drag catastrophe).
        from isaaclab.utils.math import quat_apply_inverse

        fb = torch.zeros(n, 1, 3, device=dev)
        fb[:, 0, :] = quat_apply_inverse(self.ball.data.root_quat_w, self.ball_f)
        self.ball.set_external_force_and_torque(fb, torch.zeros(n, 1, 3, device=dev))
        tq = torch.zeros(n, 1, 3, device=dev)
        tq[:, 0, 0] = self.rocker_tau
        self.rocker.set_external_force_and_torque(torch.zeros(n, 1, 3, device=dev), tq)

        c = self.cfg
        slow_tok = self.token.data.root_lin_vel_w.norm(dim=-1) < c.slow_gate
        slow_rock = self.rocker.data.root_ang_vel_w.norm(dim=-1) < 2.0
        tok = self.token_in_tray() & slow_tok & slow_rock
        self.token_latch = torch.maximum(self.token_latch, tok.float())
        paid = self.paid_now() & self.token_in_tray() & slow_rock
        self.paid_latch = torch.maximum(self.paid_latch, paid.float())
        g = self.door_open()
        opened = (g >= c.open_thresh) & torch.isfinite(g)
        self.open_latch = torch.maximum(self.open_latch, opened.float())
        out = self.ball_out() & (self.ball.data.root_lin_vel_w.norm(dim=-1) < c.slow_gate)
        self.out_latch = torch.maximum(self.out_latch, out.float())

    # ----- state (full, restorable) ---------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self._bodies().items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("crack0", "token_start", "slug_start", "ball_start",
                               "basket_start", "token_latch", "paid_latch", "open_latch",
                               "out_latch", "door_f", "ball_f", "rocker_tau")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    def _bodies(self) -> dict[str, Any]:
        return {"door": self.door, "rocker": self.rocker, "basket": self.basket,
                "token": self.token, "slug": self.slug, "ball": self.ball}

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A coin-operated laundromat washer stands on a wooden table: a pale housing "
            f"with a roof, and in its front face a blue SLIDING DOOR with a black handle "
            f"bar. A white laundry ball ({c.ball_r * 2000:.0f} mm) sits sealed inside the "
            f"chamber. Beside the door, a brown pay-post carries a grey rocker LEVER: its "
            f"near arm ends in a red LOCK TAB hanging in front of the door's leading edge "
            f"(the door is locked — pushing it only presses the tab harder into its stop), "
            f"and its far arm carries a small yellow COIN TRAY. On the table lie two flat "
            f"coins of identical size: a heavy BRASS token and a light grey slug — only "
            f"the token is heavy enough to tip the lever and swing the tab clear. A brown "
            f"collection basket also sits on the table. Coin, slug, ball, basket and a "
            f"tiny door crack are re-randomized every episode.\n"
            f"Goal: pay first — put the BRASS token into the coin tray so the lever tips "
            f"to its paid stop and unlocks the door; slide the door open by its handle; "
            f"get the laundry ball out of the chamber through the doorway; and leave it "
            f"resting inside the basket, with the token still sitting in the tray. The "
            f"slug cannot pay; ball-in-basket without a paid lever does not count."
        )

    def instruction(self) -> str:
        return (
            "Drop the brass token (not the grey slug) into the pay-lever's coin tray to "
            "unlock the sliding door, slide the door open, take the laundry ball out of "
            "the chamber and put it in the basket."
        )


# Guarded registration: the forge may import this module under two names.
if "coinop_washer" not in SCENES.list():
    SCENES.register("coinop_washer", CoinOpWasherScene)
if "simgen.coinop_washer" not in ENVS.list():
    register_env("simgen", lambda: EnvCfg(scene="coinop_washer", robot="null"))
