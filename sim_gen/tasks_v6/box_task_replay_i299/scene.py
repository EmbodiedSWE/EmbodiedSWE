"""FlapPostboxScene — push both parcels through a one-way flap door into a sealed drop box.

Derived from box_task/box_task_replay (a bimanual robot picks a soda can and a scented
candle off a table and PLACES them down into an OPEN cardboard box), but the container
relation is INVERTED from open-top placement to sealed push-through delivery: the
receiving box here is CLOSED on every face — walls, jambs, lintel and a roof — and its
only entry is a swinging FLAP DOOR (a letterbox / cat-door) hanging from a hinge above
the doorway, held shut by gravity against two outward stop nubs. Nothing can be lowered,
dropped, or placed into this box from above: to deliver a parcel the solver must slide
it across the raised porch and PUSH it against the flap until the flap yields inward,
the parcel crosses the sill, and it falls into the interior pit 70 mm below — after
which the flap swings shut again behind it. The seed's own plan (put the items into an
open container) is present as a trap: an open-top gray collection crate stands beside
the porch and parcels placed there are worth exactly nothing. A solver replaying the
seed's strategy — pick up, carry, lower into the open box — either scores zero (crate)
or finds no opening to lower into (postbox); the only winning plan is a planar
push-through-the-flap delivery, a contact interaction the seed never needs.

Assets are fully procedural (pen_holder-pattern compound spawners; child colliders of
one body never self-collide):
  - postbox fixture: KINEMATIC compound at a FIXED pose (it anchors a joint; the
    randomization lives in the movable objects) — raised porch (top at sill height,
    70 mm), front wall split into sill / two jambs / lintel around a 130 x 100 mm
    doorway, two side walls, back wall, roof.
  - flap: DYNAMIC thin plate (122 x 80 x 4 mm, 30 g), origin ON the hinge line above
    the doorway (12 mm below the lintel, so the plate's top corner clears the lintel
    corner and can never wedge); bind-time revolute joint fixture->flap about +Y,
    limits +/-85 deg, joint-pair collision disabled. Gravity is the return spring:
    the flap hangs shut at ~0 deg with no drives. Sleep thresholds zeroed.
  - stopbar: KINEMATIC pair of outward nubs flanking the doorway on a SEPARATE body
    (the hinge disables postbox<->flap pair collision, so the one-way stop must be a
    third body): the flap rests against them and can only swing INWARD.
  - parcels: a BLUE cylindrical can (52 mm dia, 65 mm tall) and a YELLOW box parcel
    (42 x 42 x 65 mm) — the seed's soda can and candle re-cast as procedural parcels.
  - decoy crate: KINEMATIC open-top crate (interior ~196 mm square, 100 mm walls) on
    the floor beside the porch; its side SWAPS per episode.

Per-episode randomization (readback-verifiable): Bernoulli spawn-slot swap of the two
parcels + per-parcel xy jitter + free yaw on the box parcel, and the decoy crate's
side (Bernoulli left/right) + xy jitter.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.05/parcel * approach   — running max of progress from the parcel's spawn distance
                             toward the doorway centre (~0 for doing nothing)
  0.20/parcel * delivered  — parcel ever inside the postbox interior (latched bool;
                             the interior is reachable ONLY through the flap)
  0.15 * shut-with-load    — both parcels inside while the flap hangs shut (latched)
  1.0 iff success()        — BOTH parcels inside the interior, flap hanging shut
                             (|angle| <= closed_tol_deg), parcels and flap at rest.
                             Non-success cap 0.85.

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


def _add_box(stage, path: str, *, center, size, color, collide: Callable) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
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


def _spawn_postbox(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the postbox fixture: KINEMATIC compound. Local origin at the centre of the
    front wall's OUTER face at ground level; the box interior extends toward +x, the
    porch extends toward -x. Front wall = sill + two jambs + lintel around the doorway;
    two outward stop nubs flank the doorway (they catch the flap: inward swing only)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    t = c.t
    w_out = c.in_w + 2 * t          # outer width (y)
    d_out = c.in_d + 2 * t          # outer depth (x)
    jamb_w = (w_out - c.door_w) / 2
    body, porch = c.color, c.porch_color
    # porch (raised approach surface, top flush with the sill top)
    _add_box(stage, f"{prim_path}/porch",
             center=(-c.porch_d / 2, 0.0, c.sill_h / 2),
             size=(c.porch_d, c.porch_w, c.sill_h), color=porch, collide=collide)
    # front wall: sill / jambs / lintel around the doorway
    _add_box(stage, f"{prim_path}/sill",
             center=(t / 2, 0.0, c.sill_h / 2),
             size=(t, w_out, c.sill_h), color=body, collide=collide)
    for sgn, nm in ((1.0, "jamb_l"), (-1.0, "jamb_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(t / 2, sgn * (c.door_w / 2 + jamb_w / 2),
                         c.sill_h + c.door_h / 2),
                 size=(t, jamb_w, c.door_h), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/lintel",
             center=(t / 2, 0.0, (c.sill_h + c.door_h + c.in_h) / 2),
             size=(t, w_out, c.in_h - c.sill_h - c.door_h), color=body, collide=collide)
    # side walls, back wall, roof: the box is sealed everywhere but the doorway
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(d_out / 2, sgn * (c.in_w / 2 + t / 2), c.in_h / 2),
                 size=(d_out, t, c.in_h), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/wall_back",
             center=(d_out - t / 2, 0.0, c.in_h / 2),
             size=(t, w_out, c.in_h), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/roof",
             center=(d_out / 2, 0.0, c.in_h + t / 2),
             size=(d_out, w_out + 0.004, t), color=body, collide=collide)
    return root


def _spawn_stopbar(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the flap's outward stop: KINEMATIC pair of nubs proud of the doorway's
    outer face, flanking the parcel lane. A SEPARATE body from the postbox on purpose:
    the flap's joint disables postbox<->flap pair collision, so the one-way stop must
    live on a third body that DOES collide with the flap."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    for sgn, nm in ((1.0, "nub_l"), (-1.0, "nub_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(-0.0095, sgn * cfg.nub_y, cfg.nub_z),
                 size=(0.009, 0.012, 0.012), color=cfg.color, collide=collide)
    return root


def _spawn_flap(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the flap: DYNAMIC thin plate hanging from its origin (the hinge line).
    The plate extends -z from the origin, its faces normal to +/-x. Sleep and
    stabilization thresholds zeroed (it must respond the instant a parcel touches it)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.25)  # pendulum settle: hangs shut without ringing
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    # plate centred ON the hinge axis (x = 0): its CoM hangs exactly under the hinge,
    # so the gravity rest angle is exactly 0 deg (an offset plate rests ~3 deg ajar
    # and flickers across the closed tolerance — measured on the forge)
    _add_box(stage, f"{prim_path}/plate",
             center=(0.0, 0.0, -cfg.plate_l / 2),
             size=(cfg.plate_t, cfg.plate_w, cfg.plate_l),
             color=cfg.color, collide=collide)
    return root


def _spawn_crate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the decoy crate: KINEMATIC open-top box (floor + 4 walls). Origin at the
    centre of the crate's ground footprint."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    _add_box(stage, f"{prim_path}/floor",
             center=(0.0, 0.0, c.floor_t / 2),
             size=(c.out_w, c.out_w, c.floor_t), color=c.color, collide=collide)
    for sgn, nm in ((1.0, "wall_xp"), (-1.0, "wall_xm")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(sgn * (c.out_w / 2 - c.t / 2), 0.0, c.floor_t + c.wall_h / 2),
                 size=(c.t, c.out_w, c.wall_h), color=c.color, collide=collide)
    for sgn, nm in ((1.0, "wall_yp"), (-1.0, "wall_ym")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(0.0, sgn * (c.out_w / 2 - c.t / 2), c.floor_t + c.wall_h / 2),
                 size=(c.out_w - 2 * c.t, c.t, c.wall_h), color=c.color, collide=collide)
    return root


def _spawn_block(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the box parcel: DYNAMIC single cuboid. Origin at the cuboid centre."""
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
    _add_box(stage, f"{prim_path}/body",
             center=(0.0, 0.0, 0.0),
             size=(cfg.side, cfg.side, cfg.height), color=cfg.color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "postbox" not in _SPAWNER_CACHE:

        @configclass
        class PostboxSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_postbox)
            in_w: float = 0.28
            in_d: float = 0.30
            in_h: float = 0.24
            t: float = 0.012
            sill_h: float = 0.07
            door_w: float = 0.13
            door_h: float = 0.10
            porch_d: float = 0.30
            porch_w: float = 0.38
            color: tuple = (0.62, 0.10, 0.08)
            porch_color: tuple = (0.35, 0.30, 0.28)
            contact_offset: float = 0.002

        @configclass
        class FlapSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_flap)
            plate_t: float = 0.004
            plate_w: float = 0.122
            plate_l: float = 0.080
            color: tuple = (0.92, 0.92, 0.90)
            contact_offset: float = 0.002

        @configclass
        class StopbarSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stopbar)
            nub_y: float = 0.058
            nub_z: float = 0.080
            color: tuple = (0.25, 0.25, 0.25)
            contact_offset: float = 0.002

        @configclass
        class CrateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_crate)
            out_w: float = 0.22
            wall_h: float = 0.10
            floor_t: float = 0.012
            t: float = 0.012
            color: tuple = (0.55, 0.55, 0.55)
            contact_offset: float = 0.002

        @configclass
        class BlockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_block)
            side: float = 0.042
            height: float = 0.065
            color: tuple = (0.90, 0.75, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(postbox=PostboxSpawnerCfg, flap=FlapSpawnerCfg,
                              stopbar=StopbarSpawnerCfg, crate=CrateSpawnerCfg,
                              block=BlockSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class FlapPostboxSceneCfg(BaseCfg):
    """Config for `FlapPostboxScene`. The interlock is architectural: the postbox is
    sealed on every face except a 130 x 100 mm doorway covered by a gravity-shut flap,
    so containment is reachable ONLY by displacing the flap with a pushed parcel.
    The flap hangs to 7 mm above the sill (no parcel sneaks under: min parcel dimension
    is 42 mm) and parcels fall 70 mm into the interior pit, dropping below the flap's
    swing so it always falls shut again behind them."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    closed_tol_deg: float = tunable(6.0)  # flap counts as shut within this of hanging 0 deg
    settle_speed: float = tunable(0.05)  # max parcel |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.50)  # max flap |ang vel| when judging (rad/s)
    inside_z_max: float = tunable(0.15)  # parcel centre below this = in the pit, not on roof

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    slot_jitter: float = tunable(0.03)  # per-parcel spawn xy jitter (+/- m)
    swap_slots: bool = tunable(True)  # Bernoulli can/block spawn-slot swap (demo sets False)
    block_yaw_deg: float = tunable(180.0)  # free yaw on the box parcel (+/- deg)
    crate_jitter: float = tunable(0.03)  # decoy crate xy jitter (+/- m)
    swap_crate: bool = tunable(True)  # Bernoulli crate side swap (demo sets False)

    # --- info: layout (single Franka base near the origin; porch and door within reach) ---------
    box_pos: tuple = info((0.50, 0.0))  # front wall outer face centre (fixture is FIXED:
    # it anchors the flap's revolute joint, and jointed mechanisms are never teleported —
    # the randomization lives in the parcels and the decoy crate)
    slot_a: tuple = info((0.27, 0.065))  # parcel spawn slot A (on the porch)
    slot_b: tuple = info((0.31, -0.065))  # parcel spawn slot B (on the porch)
    crate_pos_y: float = info(0.34)  # decoy crate |y| (side is Bernoulli), x below
    crate_pos_x: float = info(0.28)

    # --- info: postbox structure -----------------------------------------------------------------
    wall_t: float = info(0.012)
    in_w: float = info(0.28)  # interior width (y)
    in_d: float = info(0.30)  # interior depth (x)
    in_h: float = info(0.24)  # roof underside height
    sill_h: float = info(0.07)  # doorway bottom = porch top: parcels slide straight across
    door_w: float = info(0.13)
    door_h: float = info(0.10)
    porch_d: float = info(0.30)
    porch_w: float = info(0.38)
    hinge_z: float = info(0.158)  # hinge line height: 12 mm below the lintel bottom, so
    # the plate's top corner (~4 mm swing radius) clears the lintel corner by > 3 mm
    # even with both contact offsets — the flap must never wedge against the wall
    flap_t: float = info(0.004)
    flap_w: float = info(0.122)
    flap_l: float = info(0.080)  # hangs to z 0.078: 8 mm above the sill top
    flap_mass: float = info(0.03)  # light: a ~0.3 N push at parcel height displaces it
    flap_limit_deg: float = info(85.0)  # symmetric revolute limits (sign-convention hedge);
    # the one-way behaviour is owned by the stop nubs on the SEPARATE stopbar body
    # (the hinge disables postbox<->flap pair collision, so fixture nubs would be inert)
    nub_y: float = info(0.058)
    nub_z: float = info(0.080)
    box_color: tuple = info((0.62, 0.10, 0.08))  # red postbox
    flap_color: tuple = info((0.92, 0.92, 0.90))  # white flap

    # --- info: parcels ---------------------------------------------------------------------------
    can_r: float = info(0.026)  # blue cylindrical parcel (the seed's soda can, recast)
    can_h: float = info(0.065)
    can_mass: float = info(0.10)
    can_color: tuple = info((0.10, 0.35, 0.80))
    block_side: float = info(0.042)  # yellow box parcel (the seed's candle, recast)
    block_h: float = info(0.065)
    block_mass: float = info(0.08)
    block_color: tuple = info((0.90, 0.75, 0.10))

    # --- info: decoy crate -----------------------------------------------------------------------
    crate_out_w: float = info(0.22)
    crate_wall_h: float = info(0.10)
    crate_floor_t: float = info(0.012)
    crate_color: tuple = info((0.55, 0.55, 0.55))

    contact_offset: float = info(0.002)
    # rubric weights (2*0.05 + 2*0.20 + 0.15 = 0.65 <= the 0.85 non-success cap)
    w_app: float = info(0.05)  # per parcel
    w_in: float = info(0.20)  # per parcel
    w_shut: float = info(0.15)

    # Derived (filled in __post_init__).
    door_center: tuple = field(default=None, init=False)  # doorway centre, env-local (x, y, z)
    inside_lo: tuple = field(default=None, init=False)  # interior containment box, env-local
    inside_hi: tuple = field(default=None, init=False)

    def __post_init__(self) -> None:
        bx, by = self.box_pos
        self.door_center = (bx, by, self.sill_h + self.door_h / 2)
        m = 0.004  # margin inside the interior faces
        self.inside_lo = (bx + self.wall_t + m, by - self.in_w / 2 + m, -0.02)
        self.inside_hi = (bx + self.wall_t + self.in_d - m, by + self.in_w / 2 - m,
                          self.inside_z_max)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("flap_postbox")
class FlapPostboxScene(BaseScene):
    cfg: FlapPostboxSceneCfg

    def __init__(self, cfg: FlapPostboxSceneCfg | None = None) -> None:
        super().__init__(cfg or FlapPostboxSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        bx, by = c.box_pos
        postbox_spawn = spawners["postbox"](
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            in_w=c.in_w, in_d=c.in_d, in_h=c.in_h, t=c.wall_t, sill_h=c.sill_h,
            door_w=c.door_w, door_h=c.door_h, porch_d=c.porch_d, porch_w=c.porch_w,
            color=c.box_color, contact_offset=c.contact_offset,
        )
        flap_spawn = spawners["flap"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.flap_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            plate_t=c.flap_t, plate_w=c.flap_w, plate_l=c.flap_l,
            color=c.flap_color, contact_offset=c.contact_offset,
        )
        stopbar_spawn = spawners["stopbar"](
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            nub_y=c.nub_y, nub_z=c.nub_z, contact_offset=c.contact_offset,
        )
        crate_spawn = spawners["crate"](
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            out_w=c.crate_out_w, wall_h=c.crate_wall_h, floor_t=c.crate_floor_t,
            color=c.crate_color, contact_offset=c.contact_offset,
        )
        block_spawn = spawners["block"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.block_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            side=c.block_side, height=c.block_h, color=c.block_color,
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
            "postbox": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Postbox",
                spawn=postbox_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(bx, by, 0.0)),
            ),
            "flap": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Flap",
                spawn=flap_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(bx, by, c.hinge_z)),
            ),
            "stopbar": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stopbar",
                spawn=stopbar_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(bx, by, 0.0)),
            ),
            "crate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Crate",
                spawn=crate_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.crate_pos_x, c.crate_pos_y, 0.0)),
            ),
            "can": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Can",
                spawn=sim_utils.CylinderCfg(
                    radius=c.can_r, height=c.can_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.10,
                        angular_damping=0.30,  # a loose cylinder is a roller: settle, not ring
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.can_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.4, dynamic_friction=0.35, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.can_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_a[0], c.slot_a[1], c.sill_h + c.can_h / 2 + 0.002)),
            ),
            "block": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Block",
                spawn=block_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_b[0], c.slot_b[1], c.sill_h + c.block_h / 2 + 0.002)),
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
        self.postbox: RigidObject = env.iscene["postbox"]
        self.flap: RigidObject = env.iscene["flap"]
        self.stopbar: RigidObject = env.iscene["stopbar"]
        self.crate: RigidObject = env.iscene["crate"]
        self.can: RigidObject = env.iscene["can"]
        self.block: RigidObject = env.iscene["block"]
        self.env_origins = env.iscene.env_origins
        self._author_hinge()
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._app_max = torch.zeros(n, 2, device=dev)  # approach to the doorway, running max
        self._in = torch.zeros(n, 2, dtype=torch.bool, device=dev)  # ever inside the postbox
        self._shut_loaded = torch.zeros(n, dtype=torch.bool, device=dev)  # both in + flap shut
        self._d0 = torch.full((n, 2), 0.25, device=dev)  # spawn distance to the doorway

    def _author_hinge(self) -> None:
        """Per env: a +Y revolute joint postbox->flap on the hinge line above the
        doorway, symmetric limits +/-flap_limit_deg (sign-convention hedge — the
        one-way stop is the pair of colliding nubs), joint-pair collision disabled."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/flap_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Postbox"])
            j.CreateBody1Rel().SetTargets([f"{base}/Flap"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.hinge_z)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-float(c.flap_limit_deg))
            j.CreateUpperLimitAttr(float(c.flap_limit_deg))

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: parcels randomly ASSIGNED to the two porch slots (+ xy jitter,
        free yaw on the box parcel), decoy crate side Bernoulli + jitter, flap re-posed
        hanging shut (pure joint-coordinate re-pose of the follower about the unchanged
        hinge — the proven safe articulated re-pose), fixture re-asserted, latches
        cleared, spawn distances recorded."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        bx, by = c.box_pos

        # --- fixture (kinematic, fixed) + flap hanging shut at the hinge ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = bx, by
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.postbox.write_root_state_to_sim(st, env_ids)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = bx, by, c.hinge_z
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.flap.write_root_state_to_sim(st, env_ids)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = bx, by
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.stopbar.write_root_state_to_sim(st, env_ids)

        # --- decoy crate: Bernoulli side + xy jitter ---
        if c.swap_crate:
            side = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        else:
            side = torch.ones(m, device=dev)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.crate_pos_x + (torch.rand(m, device=dev) * 2 - 1) * c.crate_jitter
        st[:, 1] = side * c.crate_pos_y + (torch.rand(m, device=dev) * 2 - 1) * c.crate_jitter
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.crate.write_root_state_to_sim(st, env_ids)

        # --- parcels: Bernoulli slot swap + xy jitter, standing on the porch ---
        if c.swap_slots:
            swap = torch.rand(m, device=dev) < 0.5
        else:
            swap = torch.zeros(m, dtype=torch.bool, device=dev)
        slot_a = torch.tensor(c.slot_a, device=dev).expand(m, 2)
        slot_b = torch.tensor(c.slot_b, device=dev).expand(m, 2)
        can_xy = torch.where(swap.unsqueeze(1), slot_b, slot_a) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        blk_xy = torch.where(swap.unsqueeze(1), slot_a, slot_b) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = can_xy
        st[:, 2] = c.sill_h + c.can_h / 2 + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.can.write_root_state_to_sim(st, env_ids)
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.block_yaw_deg) / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = blk_xy
        st[:, 2] = c.sill_h + c.block_h / 2 + 0.002
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.block.write_root_state_to_sim(st, env_ids)

        # --- latches + spawn distances (approach progress is measured from these) ---
        door = torch.tensor(c.door_center, device=dev)
        for k, xy, z in ((0, can_xy, c.sill_h + c.can_h / 2),
                         (1, blk_xy, c.sill_h + c.block_h / 2)):
            p = torch.cat([xy, torch.full((m, 1), z, device=dev)], dim=1)
            self._d0[env_ids, k] = (p - door).norm(dim=-1).clamp(min=0.05)
        self._app_max[env_ids] = 0.0
        self._in[env_ids] = False
        self._shut_loaded[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "postbox": self.postbox.data.root_state_w[env_ids].clone(),
            "flap": self.flap.data.root_state_w[env_ids].clone(),
            "stopbar": self.stopbar.data.root_state_w[env_ids].clone(),
            "crate": self.crate.data.root_state_w[env_ids].clone(),
            "can": self.can.data.root_state_w[env_ids].clone(),
            "block": self.block.data.root_state_w[env_ids].clone(),
            "app_max": self._app_max[env_ids].clone(),
            "in": self._in[env_ids].clone(),
            "shut_loaded": self._shut_loaded[env_ids].clone(),
            "d0": self._d0[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.postbox.write_root_state_to_sim(state["postbox"], env_ids)
        self.flap.write_root_state_to_sim(state["flap"], env_ids)
        self.stopbar.write_root_state_to_sim(state["stopbar"], env_ids)
        self.crate.write_root_state_to_sim(state["crate"], env_ids)
        self.can.write_root_state_to_sim(state["can"], env_ids)
        self.block.write_root_state_to_sim(state["block"], env_ids)
        self._app_max[env_ids] = state["app_max"]
        self._in[env_ids] = state["in"]
        self._shut_loaded[env_ids] = state["shut_loaded"]
        self._d0[env_ids] = state["d0"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A RED drop box (a sealed postbox, ~{(c.in_w + 2 * c.wall_t) * 100:.0f} cm "
            f"wide x {(c.in_d + 2 * c.wall_t) * 100:.0f} cm deep x "
            f"{(c.in_h + c.wall_t) * 100:.0f} cm tall, closed on every side INCLUDING the "
            f"top) stands at the far edge of a raised dark porch (top {c.sill_h * 100:.0f} cm "
            f"above the floor). Its only opening is a {c.door_w * 100:.0f} x "
            f"{c.door_h * 100:.0f} cm doorway at porch level in the wall facing the porch, "
            f"covered by a WHITE FLAP hinged along the doorway's top edge: the flap hangs "
            f"shut under gravity, swings INWARD when something is pushed against it, and "
            f"falls shut again by itself — it cannot swing outward. On the porch stand two "
            f"parcels (their positions swap between episodes): a BLUE cylindrical can "
            f"({2 * c.can_r * 100:.1f} cm across, {c.can_h * 100:.1f} cm tall) and a YELLOW "
            f"box parcel ({c.block_side * 100:.1f} cm square, {c.block_h * 100:.1f} cm "
            f"tall). Beside the porch, on the floor (left or right, changing per episode), "
            f"sits an open-top GRAY crate — it is a decoy: parcels placed in it count for "
            f"nothing.\n"
            f"Goal: BOTH parcels must end up INSIDE the red drop box with the white flap "
            f"hanging fully shut (within {c.closed_tol_deg:.0f} deg) and everything at "
            f"rest. The box cannot be opened, lifted, or loaded from above: the only way "
            f"in is to slide each parcel across the porch and push it THROUGH the flap "
            f"until it drops inside (inside the box the floor is {c.sill_h * 100:.0f} cm "
            f"below the doorway, so delivered parcels fall clear and the flap closes "
            f"behind them). Either parcel may go first. Parcels left on the porch, "
            f"leaning on the flap, stuck in the doorway, on the box roof, on the floor, "
            f"or in the gray crate do not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Push the blue can and the yellow parcel one at a time across the porch and "
            "in through the white swinging flap of the red drop box, so that both end up "
            "inside the box and the flap hangs shut. Do not put them in the gray crate."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def flap_angle(self) -> torch.Tensor:
        """(N,) hinge angle in rad (0 = hanging shut). The flap only ever rotates about
        the hinge +y axis, so the root quat is (cos t/2, 0, sin t/2, 0)."""
        q = self.flap.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 2], q[:, 0])

    def flap_shut(self) -> torch.Tensor:
        """(N,) bool: flap hanging within `closed_tol_deg` of shut."""
        return self.flap_angle().abs() <= math.radians(self.cfg.closed_tol_deg)

    def _inside(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body's origin inside the postbox interior (env-local axis-aligned
        box — the fixture is fixed and never rotated). The interior is architecturally
        reachable only through the flap doorway; the z clause excludes the roof."""
        c = self.cfg
        p = body.data.root_pos_w - self.env_origins
        lo = torch.tensor(c.inside_lo, device=p.device)
        hi = torch.tensor(c.inside_hi, device=p.device)
        return ((p >= lo) & (p <= hi)).all(dim=-1)

    def _parcels_still(self) -> torch.Tensor:
        c = self.cfg
        return ((self.can.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                & (self.block.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed))

    def _update_latches(self) -> None:
        c = self.cfg
        door = torch.tensor(c.door_center, device=self.env.device)
        in_now = []
        for k, body in ((0, self.can), (1, self.block)):
            p = body.data.root_pos_w - self.env_origins
            d = (p - door).norm(dim=-1)
            app = (1.0 - d / self._d0[:, k]).clamp(0.0, 1.0)
            app = torch.nan_to_num(app, nan=0.0, posinf=0.0, neginf=0.0)
            self._app_max[:, k] = torch.maximum(self._app_max[:, k], app)
            inside = self._inside(body)
            self._in[:, k] |= inside
            in_now.append(inside)
        self._shut_loaded |= in_now[0] & in_now[1] & self.flap_shut()

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No driven mechanics: gravity is the flap's return spring. Latch progress."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: BOTH parcels inside the postbox interior, flap hanging shut, flap
        and parcels at rest. Physical outcomes only (settled poses, real containment —
        the interior is reachable only through the flap)."""
        c = self.cfg
        self._update_latches()
        flap_still = self.flap.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega
        return (self._inside(self.can) & self._inside(self.block) & self.flap_shut()
                & flap_still & self._parcels_still())

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.05/parcel * doorway approach + 0.20/parcel *
        delivered-inside + 0.15 * shut-with-both-loaded — all latched, ~0 for doing
        nothing, capped 0.85 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_app * self._app_max.sum(dim=1)
                + c.w_in * self._in.float().sum(dim=1)
                + c.w_shut * self._shut_loaded.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="flap_postbox", robot="null"))
