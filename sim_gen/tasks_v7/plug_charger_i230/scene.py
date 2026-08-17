"""ShutterGarageScene — open the sliding shutter, push the charger brick through the
doorway onto the contact plate, then slide the shutter closed again (sim_gen task
`plug_charger_i230`).

Derived from maniskill/plug_charger, but STRATEGICALLY different: the seed is a
terminal precision INSERTION — pick a free charger off the table, align its two
prongs with two slightly-enlarged holes in a fixed base, and press it in; the episode
ends the instant the charger pose sits in a bbox around the base, and nothing else in
the scene is judged. Here nothing is ever grasped-aligned-pressed into a hole, and
the judged goal is the state of a MECHANISM bracketing the object move:

  1. a roofed charging GARAGE stands on the floor: its only opening is a front
     doorway, and that doorway is covered at reset by a SLIDING SHUTTER — a free
     slab captive in rails (floor ridge + hanging lip + end stops), with a knob.
     The shutter must first be SLID ASIDE (either direction) far enough to clear
     the doorway;
  2. the charger brick must then be driven THROUGH the doorway ALONG THE FLOOR —
     the roof forbids dropping it in, the walls forbid every other approach —
     prongs-first until its two brass prongs touch the copper contact plate at the
     back of the bay (a floor-level push transit, not a pick-align-press);
  3. finally the shutter must be SLID CLOSED again so it covers the doorway, sealing
     the charger inside.

The execution order is MECHANISM-ENFORCED in both directions: a closed shutter
physically blocks the doorway (the slab overlaps the frame on all sides, so pushing
the brick at it just presses slab against frame), and a brick that is not yet fully
inside stands in the shutter's swept lane and physically blocks re-closing. A solver
therefore needs a different PLAN (operate a mechanism, route an object through the
opening it guards, restore the mechanism) and a different code structure (a shutter-
travel latch, a gated doorway-transit latch, a docking predicate, and a shutter-
closed predicate that is part of success) — not "align prongs with holes and push".

success() iff, with brick AND shutter settled (|v| < settle_lin):
  - the brick is DOCKED: prong tips within `seat_tol` of the contact plate (dock
    frame), prong axis into the bay within `seat_align_deg`, brick upright, on the
    bay floor, laterally inside the bay;
  - the shutter is CLOSED: slab centred on the doorway within `door_closed_tol`,
    still captive in its rails.

score() is latched every physics substep (credit never evaporates):
  0.20 * best shutter-opening travel (fraction of the travel that fully clears the
         doorway, gated on the slab being captive in its rails)
+ 0.35 * best doorway-transit progress of the PRONG TIP toward the plate (gated on
         the brick actually being in the doorway/bay lane at floor level)
+ 0.15 * docked (prongs at the plate, settled — latched once reached),
capped at 0.85; exactly 1.0 iff success(). Doing nothing scores ~0; the seed's
whole strategy (insert, episode over) leaves the shutter open and caps at 0.70.

Assets are fully procedural (no external files):
  - garage: KINEMATIC compound, 390 x 174 x 110 mm overall: facade with a 95 x 75 mm
    doorway, 144 x 105 mm roofed bay behind it, copper contact plate at the back,
    and a front rail channel (floor ridge, hanging lip under a soffit, end stops)
    holding the shutter. Repositioned per episode (kinematic root).
  - shutter: DYNAMIC yellow slab 130 x 12 x 90 mm with a knob (16 mm stem, 36 mm
    cap), 0.40 kg, standing in the rail channel, covering the doorway at reset.
  - brick: DYNAMIC white charger brick 60 x 90 x 50 mm, 0.30 kg, with two brass
    prongs (8 mm proud) on one end face; it starts loose on the floor in front.
Contact offsets are explicit and small (1 mm): the default ~2 cm offset would eat
the 3 mm rail clearances and the doorway margins.

Per-episode randomization (verified by readback in smoke): garage xy jitter + yaw
(the push direction must be read from the scene), brick spawn xy + free yaw.
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


def _box(stage, path: str, size, center, color, contact_offset: float) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _rigid_dynamic(root, mass: float) -> None:
    """Dynamic rigid-body armor on a compound root: mass (PhysX derives inertia from
    the child colliders), damping so parts settle promptly, no sleeping while we
    judge velocities, and the depenetration cap."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.10)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _spawn_garage(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC charging garage at `prim_path`. Origin = centre of the
    facade's OUTER face at floor level; local +y points INWARD (doorway faces -y),
    x lateral, z up. Facade with a doorway cutout, roofed bay with side walls and
    the copper back plate, and the front rail channel (soffit + hanging lip + floor
    ridge + end stops) that keeps the shutter captive. Everything axis-aligned
    boxes."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(8.0)

    co = cfg.contact_offset
    body_c, plate_c, rail_c = cfg.body_color, cfg.plate_color, cfg.rail_color
    hw = cfg.facade_half_w
    tf, H = cfg.facade_t, cfg.facade_h
    dwh, dh = cfg.doorway_w / 2, cfg.doorway_h
    bhw, wt = cfg.bay_half_w, cfg.wall_t
    y_plate0 = tf + cfg.bay_d  # plate inner face
    y_back = y_plate0 + cfg.plate_t
    ch_y0 = -cfg.channel_d  # channel outer face
    lip_t = cfg.lip_t

    # facade (with doorway cutout) --------------------------------------------------------
    seg_w = hw - dwh
    _box(stage, f"{prim_path}/facade_l", (seg_w, tf, H),
         (-(dwh + seg_w / 2), tf / 2, H / 2), body_c, co)
    _box(stage, f"{prim_path}/facade_r", (seg_w, tf, H),
         (dwh + seg_w / 2, tf / 2, H / 2), body_c, co)
    _box(stage, f"{prim_path}/header", (2 * dwh, tf, H - dh),
         (0.0, tf / 2, (dh + H) / 2), body_c, co)
    # bay: side walls + copper back plate + roof ------------------------------------------
    for tag, sx in (("wall_l", -1.0), ("wall_r", 1.0)):
        _box(stage, f"{prim_path}/{tag}", (wt, y_back - tf, H),
             (sx * (bhw + wt / 2), (tf + y_back) / 2, H / 2), body_c, co)
    _box(stage, f"{prim_path}/plate", (2 * (bhw + wt), cfg.plate_t, H),
         (0.0, (y_plate0 + y_back) / 2, H / 2), plate_c, co)
    _box(stage, f"{prim_path}/roof", (2 * hw, y_back, cfg.roof_t),
         (0.0, y_back / 2, H + cfg.roof_t / 2), body_c, co)
    # front rail channel -------------------------------------------------------------------
    _box(stage, f"{prim_path}/soffit", (2 * hw, cfg.channel_d, cfg.soffit_t),
         (0.0, ch_y0 / 2, cfg.soffit_z0 + cfg.soffit_t / 2), rail_c, co)
    _box(stage, f"{prim_path}/lip", (2 * hw, lip_t, cfg.soffit_z0 - cfg.lip_z0),
         (0.0, ch_y0 + lip_t / 2, (cfg.lip_z0 + cfg.soffit_z0) / 2), rail_c, co)
    # floor ridge, interrupted at the doorway (the brick must pass at floor level;
    # the 130 mm slab always overlaps at least one segment of the 95 mm gap)
    for tag, sx in (("ridge_l", -1.0), ("ridge_r", 1.0)):
        _box(stage, f"{prim_path}/{tag}", (seg_w, lip_t, cfg.ridge_h),
             (sx * (dwh + seg_w / 2), ch_y0 + lip_t / 2, cfg.ridge_h / 2), rail_c, co)
    for tag, sx in (("stop_l", -1.0), ("stop_r", 1.0)):
        _box(stage, f"{prim_path}/{tag}", (cfg.stop_t, cfg.channel_d + tf, H),
             (sx * (cfg.stop_x + cfg.stop_t / 2), (ch_y0 + tf) / 2, H / 2), rail_c, co)
    return root


def _spawn_shutter(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC sliding shutter at `prim_path`. Origin = slab centre;
    local -y = outward (knob side): slab + knob stem + knob cap."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass)

    co = cfg.contact_offset
    _box(stage, f"{prim_path}/slab", (cfg.door_slab_w, cfg.door_t, cfg.door_h),
         (0.0, 0.0, 0.0), cfg.color, co)
    _box(stage, f"{prim_path}/stem", (cfg.knob_stem, cfg.knob_len, cfg.knob_stem),
         (0.0, -(cfg.door_t + cfg.knob_len) / 2, cfg.knob_dz), cfg.knob_color, co)
    _box(stage, f"{prim_path}/cap", (cfg.knob_cap_w, cfg.knob_cap_t, cfg.knob_cap_w),
         (0.0, -(cfg.door_t / 2 + cfg.knob_len + cfg.knob_cap_t / 2), cfg.knob_dz),
         cfg.knob_color, co)
    return root


def _spawn_brick(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC charger brick at `prim_path`. Origin = body centre; local
    +y = prong direction: body + two brass prongs proud of the +y face."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass)

    co = cfg.contact_offset
    _box(stage, f"{prim_path}/body", (cfg.brick_w, cfg.brick_l, cfg.brick_h),
         (0.0, 0.0, 0.0), cfg.color, co)
    for tag, sx in (("prong_l", -1.0), ("prong_r", 1.0)):
        _box(stage, f"{prim_path}/{tag}", (cfg.prong_w, cfg.prong_len, cfg.prong_w),
             (sx * cfg.prong_dx, (cfg.brick_l + cfg.prong_len) / 2, cfg.prong_dz),
             cfg.prong_color, co)
    return root


def _garage_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "garage" not in _SPAWNER_CACHE:

        @configclass
        class GarageSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_garage)
            facade_half_w: float = 0.195
            facade_t: float = 0.012
            facade_h: float = 0.100
            doorway_w: float = 0.095
            doorway_h: float = 0.075
            bay_half_w: float = 0.072
            wall_t: float = 0.012
            bay_d: float = 0.105
            plate_t: float = 0.012
            roof_t: float = 0.010
            channel_d: float = 0.030
            lip_t: float = 0.008
            lip_z0: float = 0.079
            soffit_z0: float = 0.094
            soffit_t: float = 0.010
            ridge_h: float = 0.012
            stop_x: float = 0.1825
            stop_t: float = 0.012
            body_color: tuple = (0.45, 0.48, 0.55)
            plate_color: tuple = (0.78, 0.45, 0.20)
            rail_color: tuple = (0.30, 0.32, 0.38)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["garage"] = GarageSpawnerCfg

    return _SPAWNER_CACHE["garage"](
        mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        **kw,
    )


def _shutter_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "shutter" not in _SPAWNER_CACHE:

        @configclass
        class ShutterSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shutter)
            door_slab_w: float = 0.130
            door_t: float = 0.012
            door_h: float = 0.090
            knob_stem: float = 0.016
            knob_len: float = 0.020
            knob_cap_w: float = 0.036
            knob_cap_t: float = 0.010
            knob_dz: float = 0.005
            mass: float = 0.40
            color: tuple = (0.88, 0.70, 0.12)
            knob_color: tuple = (0.20, 0.20, 0.22)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["shutter"] = ShutterSpawnerCfg

    return _SPAWNER_CACHE["shutter"](
        mass_props=sim_utils.MassPropertiesCfg(mass=kw["mass"]),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        **kw,
    )


def _brick_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "brick" not in _SPAWNER_CACHE:

        @configclass
        class BrickSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_brick)
            brick_w: float = 0.060
            brick_l: float = 0.090
            brick_h: float = 0.050
            prong_w: float = 0.010
            prong_len: float = 0.008
            prong_dx: float = 0.014
            prong_dz: float = 0.005
            mass: float = 0.30
            color: tuple = (0.92, 0.91, 0.88)
            prong_color: tuple = (0.75, 0.68, 0.30)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["brick"] = BrickSpawnerCfg

    return _SPAWNER_CACHE["brick"](
        mass_props=sim_utils.MassPropertiesCfg(mass=kw["mass"]),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        **kw,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ShutterGarageSceneCfg(BaseCfg):
    """Config for `ShutterGarageScene`. Honesty knobs asserted in `__post_init__`:
    the closed shutter genuinely seals the doorway (entry is impossible), the open
    travel genuinely clears it, the doorway passes the brick with real clearance,
    the roof forbids top-loading, and a docked brick sits clear of the shutter's
    swept lane (so the shutter can close on a docked brick and ONLY then)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    seat_tol: float = tunable(0.008)  # prong tips within this of the contact plate (m)
    seat_align_deg: float = tunable(30.0)  # prong axis within this of the bay's inward axis
    seat_xy_half: float = tunable(0.030)  # brick centre within this of the bay axis, laterally (m)
    brick_upright_deg: float = tunable(20.0)  # brick +z within this of world-up when docked
    door_closed_tol: float = tunable(0.012)  # slab centre within this of the doorway centre (m)
    settle_lin: float = tunable(0.05)  # max |lin vel| (brick AND shutter) when judging (m/s)
    door_open_dead: float = tunable(0.020)  # travel latch dead-band (no credit for rattle) (m)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    dock_jitter: float = tunable(0.04)  # uniform +/- xy jitter of the garage at reset (m)
    dock_yaw_deg: float = tunable(30.0)  # uniform +/- garage yaw at reset (deg)
    brick_lat: float = tunable(0.10)  # brick spawn: uniform +/- lateral range (dock frame) (m)
    brick_y0: float = tunable(-0.24)  # brick spawn: dock-frame y range (in front of the rails)
    brick_y1: float = tunable(-0.15)
    brick_yaw_deg: float = tunable(180.0)  # uniform +/- brick yaw at reset (free)

    # --- tunable: placement ------------------------------------------------------------------
    dock_pos: tuple = tunable((0.0, 0.08))  # garage origin (facade outer face centre), nominal

    # --- info: garage structure (dock frame: origin = facade outer face centre at floor, ----
    # +y INWARD through the doorway, x lateral, z up) -----------------------------------------
    facade_half_w: float = info(0.195)
    facade_t: float = info(0.012)
    facade_h: float = info(0.100)
    doorway_w: float = info(0.095)  # doorway spans |x| < this/2, z < doorway_h
    doorway_h: float = info(0.075)
    bay_half_w: float = info(0.072)  # bay interior half width
    wall_t: float = info(0.012)
    bay_d: float = info(0.105)  # bay interior depth (facade inner face -> plate inner face)
    plate_t: float = info(0.012)
    roof_t: float = info(0.010)
    channel_d: float = info(0.030)  # rail channel outer face at y = -this
    lip_t: float = info(0.008)
    lip_z0: float = info(0.079)  # hanging lip bottom edge
    soffit_z0: float = info(0.094)
    soffit_t: float = info(0.010)
    ridge_h: float = info(0.012)
    stop_x: float = info(0.1825)  # end-stop inner faces at +/- this
    stop_t: float = info(0.012)
    # --- info: shutter structure -------------------------------------------------------------
    door_slab_w: float = info(0.130)  # > doorway_w: the closed slab overlaps the frame
    door_t: float = info(0.012)
    door_h: float = info(0.090)  # > doorway_h: overlaps the header too
    door_gap: float = info(0.003)  # slab-to-facade running clearance
    knob_stem: float = info(0.016)
    knob_len: float = info(0.020)
    knob_cap_w: float = info(0.036)
    knob_cap_t: float = info(0.010)
    knob_dz: float = info(0.005)
    door_mass: float = info(0.40)
    # --- info: brick structure ---------------------------------------------------------------
    brick_w: float = info(0.060)
    brick_l: float = info(0.090)
    brick_h: float = info(0.050)
    prong_w: float = info(0.010)
    prong_len: float = info(0.008)
    prong_dx: float = info(0.014)
    prong_dz: float = info(0.005)
    brick_mass: float = info(0.30)
    # --- info: colors + misc -----------------------------------------------------------------
    body_color: tuple = info((0.45, 0.48, 0.55))
    plate_color: tuple = info((0.78, 0.45, 0.20))  # copper contact plate
    rail_color: tuple = info((0.30, 0.32, 0.38))
    door_color: tuple = info((0.88, 0.70, 0.12))  # safety yellow
    knob_color: tuple = info((0.20, 0.20, 0.22))
    brick_color: tuple = info((0.92, 0.91, 0.88))
    prong_color: tuple = info((0.75, 0.68, 0.30))  # brass
    contact_offset: float = info(0.001)  # explicit: default ~2 cm would eat the 3 mm clearances

    # Derived (filled in __post_init__).
    plate_y: float = field(default=None, init=False)  # plate inner face (dock frame)
    tip_half: float = field(default=None, init=False)  # brick origin -> prong tip, local +y
    y_seat_tip: float = field(default=None, init=False)  # tip y when docked (== plate_y)
    y_seat_min: float = field(default=None, init=False)  # tip y at the seat tolerance edge
    x_door_clear: float = field(default=None, init=False)  # |slab centre x| that clears the doorway
    x_door_max: float = field(default=None, init=False)  # |slab centre x| at the end stops
    door_yc: float = field(default=None, init=False)  # slab centre y in the channel
    door_zc: float = field(default=None, init=False)  # slab centre z when standing

    def __post_init__(self) -> None:
        self.plate_y = self.facade_t + self.bay_d
        self.tip_half = self.brick_l / 2 + self.prong_len
        self.y_seat_tip = self.plate_y
        self.y_seat_min = self.plate_y - self.seat_tol
        self.x_door_clear = self.doorway_w / 2 + self.door_slab_w / 2
        self.x_door_max = self.stop_x - self.door_slab_w / 2
        self.door_yc = -(self.door_gap + self.door_t / 2)
        self.door_zc = self.door_h / 2

        assert self.doorway_w >= self.brick_w + 0.020, (
            "doorway must pass the brick with real lateral clearance")
        assert self.doorway_h >= self.brick_h + 0.015, (
            "doorway must pass the brick with real headroom")
        assert (self.door_slab_w - self.doorway_w) / 2 >= self.door_closed_tol + 0.004, (
            "a closed-within-tolerance slab must still overlap the doorway frame")
        assert self.door_h >= self.doorway_h + 0.010, (
            "the slab must overlap the header (no climbing over a closed shutter)")
        assert self.door_h >= self.lip_z0 + 0.008, (
            "the hanging lip must retain the slab top (shutter cannot tip out)")
        assert self.door_slab_w >= self.doorway_w + 0.010, (
            "the slab must always overlap at least one floor-ridge segment "
            "(the ridge is interrupted at the doorway so the brick can pass)")
        assert self.x_door_max >= self.x_door_clear + 0.004, (
            "the rail travel must let the slab fully clear the doorway")
        assert self.bay_half_w >= self.brick_w / 2 + 0.008, (
            "bay must hold the brick with lateral clearance")
        assert self.plate_y >= self.facade_t + self.brick_l + self.prong_len + 0.004, (
            "a docked brick must sit entirely behind the facade")
        assert self.plate_y - self.tip_half - self.brick_l / 2 >= 0.004, (
            "a docked brick must be clear of the shutter's swept lane (rear face "
            "behind the facade outer plane) so the shutter can close on it")
        assert self.doorway_h < self.facade_h, "doorway must sit under a real header"
        assert self.seat_xy_half <= self.bay_half_w - self.brick_w / 2, (
            "docking tolerance must stay inside the physical bay walls")
        assert self.door_open_dead < self.x_door_clear, "travel latch must be reachable"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("shutter_garage")
class ShutterGarageScene(BaseScene):
    cfg: ShutterGarageSceneCfg

    def __init__(self, cfg: ShutterGarageSceneCfg | None = None) -> None:
        super().__init__(cfg or ShutterGarageSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        garage_kw = dict(
            facade_half_w=c.facade_half_w, facade_t=c.facade_t, facade_h=c.facade_h,
            doorway_w=c.doorway_w, doorway_h=c.doorway_h, bay_half_w=c.bay_half_w,
            wall_t=c.wall_t, bay_d=c.bay_d, plate_t=c.plate_t, roof_t=c.roof_t,
            channel_d=c.channel_d, lip_t=c.lip_t, lip_z0=c.lip_z0, soffit_z0=c.soffit_z0,
            soffit_t=c.soffit_t, ridge_h=c.ridge_h, stop_x=c.stop_x, stop_t=c.stop_t,
            body_color=c.body_color, plate_color=c.plate_color, rail_color=c.rail_color,
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
            "garage": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Garage",
                spawn=_garage_spawner_cfg(**garage_kw),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.dock_pos[0], c.dock_pos[1], 0.0005)),
            ),
            "shutter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shutter",
                spawn=_shutter_spawner_cfg(
                    door_slab_w=c.door_slab_w, door_t=c.door_t, door_h=c.door_h,
                    knob_stem=c.knob_stem, knob_len=c.knob_len, knob_cap_w=c.knob_cap_w,
                    knob_cap_t=c.knob_cap_t, knob_dz=c.knob_dz, mass=c.door_mass,
                    color=c.door_color, knob_color=c.knob_color,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.dock_pos[0], c.dock_pos[1] + c.door_yc, c.door_zc + 0.0005)),
            ),
            "brick": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Brick",
                spawn=_brick_spawner_cfg(
                    brick_w=c.brick_w, brick_l=c.brick_l, brick_h=c.brick_h,
                    prong_w=c.prong_w, prong_len=c.prong_len, prong_dx=c.prong_dx,
                    prong_dz=c.prong_dz, mass=c.brick_mass, color=c.brick_color,
                    prong_color=c.prong_color, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.dock_pos[0], c.dock_pos[1] - 0.20, c.brick_h / 2 + 0.001)),
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
        self.garage: RigidObject = env.iscene["garage"]
        self.shutter: RigidObject = env.iscene["shutter"]
        self.brick: RigidObject = env.iscene["brick"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # progress latches
        self.door_latch = torch.zeros(n, device=dev)
        self.entry_latch = torch.zeros(n, device=dev)
        self.dock_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: garage with xy jitter + yaw (the push direction moves with
        it), shutter written CLOSED in its rails (covering the doorway), brick loose
        on the floor in front with xy jitter + free yaw; latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- garage: xy jitter + yaw (kinematic root) ---
        dock_xy = torch.tensor(c.dock_pos, device=dev).expand(m, 2).clone()
        dock_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.dock_jitter
        dock_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.dock_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = dock_xy
        st[:, 2] = 0.0005
        st[:, 3] = torch.cos(dock_yaw / 2)
        st[:, 6] = torch.sin(dock_yaw / 2)
        st[:, 0:3] += origin
        self.garage.write_root_state_to_sim(st, env_ids)

        ca, sa = torch.cos(dock_yaw), torch.sin(dock_yaw)

        def to_world(lx: torch.Tensor, ly: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return dock_xy[:, 0] + ca * lx - sa * ly, dock_xy[:, 1] + sa * lx + ca * ly

        # --- shutter: CLOSED, standing in its rails ---
        lx = torch.zeros(m, device=dev)
        ly = torch.full((m,), c.door_yc, device=dev)
        wx, wy = to_world(lx, ly)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = wx, wy
        st[:, 2] = c.door_zc + 0.0005
        st[:, 3] = torch.cos(dock_yaw / 2)
        st[:, 6] = torch.sin(dock_yaw / 2)
        st[:, 0:3] += origin
        self.shutter.write_root_state_to_sim(st, env_ids)

        # --- brick: loose on the floor in front, xy jitter + free yaw ---
        lx = (torch.rand(m, device=dev) * 2 - 1) * c.brick_lat
        ly = c.brick_y0 + torch.rand(m, device=dev) * (c.brick_y1 - c.brick_y0)
        wx, wy = to_world(lx, ly)
        byaw = dock_yaw + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.brick_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = wx, wy
        st[:, 2] = c.brick_h / 2 + 0.001
        st[:, 3] = torch.cos(byaw / 2)
        st[:, 6] = torch.sin(byaw / 2)
        st[:, 0:3] += origin
        self.brick.write_root_state_to_sim(st, env_ids)

        # --- latches ---
        self.door_latch[env_ids] = 0.0
        self.entry_latch[env_ids] = 0.0
        self.dock_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "garage": self.garage.data.root_state_w[env_ids].clone(),
            "shutter": self.shutter.data.root_state_w[env_ids].clone(),
            "brick": self.brick.data.root_state_w[env_ids].clone(),
            "door_latch": self.door_latch[env_ids].clone(),
            "entry_latch": self.entry_latch[env_ids].clone(),
            "dock_latch": self.dock_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.garage.write_root_state_to_sim(state["garage"], env_ids)
        self.shutter.write_root_state_to_sim(state["shutter"], env_ids)
        self.brick.write_root_state_to_sim(state["brick"], env_ids)
        self.door_latch[env_ids] = state["door_latch"]
        self.entry_latch[env_ids] = state["entry_latch"]
        self.dock_latch[env_ids] = state["dock_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A blue-gray charging GARAGE ({2 * c.facade_half_w * 1000:.0f} mm wide, "
            f"{(c.channel_d + c.plate_y + c.plate_t) * 1000:.0f} mm deep, "
            f"{(c.facade_h + c.roof_t) * 1000:.0f} mm tall) stands on the floor. Its only "
            f"opening is a {c.doorway_w * 1000:.0f} x {c.doorway_h * 1000:.0f} mm doorway "
            f"in the front face — the bay behind it is roofed and walled, with a COPPER "
            f"CONTACT PLATE as its back wall. The doorway is covered by a YELLOW SLIDING "
            f"SHUTTER (a {c.door_slab_w * 1000:.0f} x {c.door_h * 1000:.0f} mm slab with a "
            f"dark knob) that runs sideways in rails along the front of the garage; it can "
            f"slide either direction until it hits an end stop, and it overlaps the doorway "
            f"frame, so nothing passes the doorway while it is closed. A WHITE CHARGER "
            f"BRICK ({c.brick_w * 1000:.0f} x {c.brick_l * 1000:.0f} x "
            f"{c.brick_h * 1000:.0f} mm) with two BRASS PRONGS on one end lies loose on "
            f"the floor in front of the garage (position and heading vary; the garage's "
            f"own heading varies too — read directions from the scene).\n"
            f"Goal, in the only order the mechanism permits: (1) slide the shutter aside "
            f"(either direction) until the doorway is clear; (2) move the charger brick "
            f"through the doorway along the floor, prongs first and upright, until both "
            f"prongs touch the copper plate at the back (within "
            f"{c.seat_tol * 1000:.0f} mm); (3) slide the shutter back until it is centred "
            f"over the doorway again (within {c.door_closed_tol * 1000:.0f} mm), sealing "
            f"the charger inside. The roof and walls make the doorway the only way in, "
            f"and a brick left sticking out of the doorway blocks the shutter from "
            f"closing.\n"
            f"Judged only when the brick and the shutter have settled: success needs the "
            f"brick docked prongs-against-the-plate (prong axis within "
            f"{c.seat_align_deg:.0f} deg of straight-in, upright, on the bay floor, "
            f"laterally within {c.seat_xy_half * 1000:.0f} mm of the bay axis) AND the "
            f"shutter closed. A brick jammed against the closed shutter, left in the "
            f"doorway, docked backwards (prongs facing out), or docked with the shutter "
            f"still open counts as incomplete."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the yellow shutter aside, push the white charger brick through the "
            "doorway prongs-first until its prongs touch the copper plate at the back, "
            "then slide the shutter closed over the doorway again."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _dock_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> garage (dock) frame (origin = facade outer face
        centre at floor level, +y inward)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.garage.data.root_quat_w,
                                  p_w - self.garage.data.root_pos_w)

    def _yaw(self, body) -> torch.Tensor:
        q = body.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    def _up_z(self, body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)[:, 2]

    def tip_point_w(self) -> torch.Tensor:
        """(N,3) world position of the point midway between the two prong tips."""
        from isaaclab.utils.math import quat_apply

        ey = torch.tensor([0.0, 1.0, 0.0], device=self.env.device).expand(self.env.num_envs, 3)
        axis = quat_apply(self.brick.data.root_quat_w, ey)
        return self.brick.data.root_pos_w + axis * self.cfg.tip_half

    def brick_dock(self) -> torch.Tensor:
        """(N,3) brick origin in the dock frame."""
        return self._dock_local(self.brick.data.root_pos_w)

    def tip_dock(self) -> torch.Tensor:
        """(N,3) prong-tip midpoint in the dock frame."""
        return self._dock_local(self.tip_point_w())

    def shutter_dock(self) -> torch.Tensor:
        """(N,3) shutter slab centre in the dock frame."""
        return self._dock_local(self.shutter.data.root_pos_w)

    def prong_axis_in(self) -> torch.Tensor:
        """(N,) cosine between the brick's prong axis and the dock's inward (+y) axis."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        ey = torch.tensor([0.0, 1.0, 0.0], device=self.env.device).expand(self.env.num_envs, 3)
        axis_w = quat_apply(self.brick.data.root_quat_w, ey)
        axis_d = quat_apply_inverse(self.garage.data.root_quat_w, axis_w)
        return axis_d[:, 1]

    # ----- predicates -------------------------------------------------------------------------
    def shutter_in_rails(self) -> torch.Tensor:
        """(N,) bool: the slab still captive in its rail channel (y at the channel
        line, standing at slab height) — a shutter somehow out of its rails guards
        nothing and earns nothing."""
        c = self.cfg
        loc = self.shutter_dock()
        return ((loc[:, 1] - c.door_yc).abs() < 0.008) & ((loc[:, 2] - c.door_zc).abs() < 0.015)

    def shutter_closed(self) -> torch.Tensor:
        """(N,) bool: slab centred over the doorway within `door_closed_tol`, captive
        in its rails (the slab then overlaps the frame on all sides: sealed)."""
        loc = self.shutter_dock()
        return (loc[:, 0].abs() < self.cfg.door_closed_tol) & self.shutter_in_rails()

    def docked(self) -> torch.Tensor:
        """(N,) bool, geometric: prong tips at the copper plate (dock frame, within
        `seat_tol`), prong axis inward within `seat_align_deg`, brick upright, on the
        bay floor, laterally inside the bay."""
        c = self.cfg
        b = self.brick_dock()
        t = self.tip_dock()
        at_plate = (t[:, 1] > c.y_seat_min) & (t[:, 1] < c.plate_y + 0.006)
        aligned = self.prong_axis_in() > math.cos(math.radians(c.seat_align_deg))
        upright = self._up_z(self.brick) > math.cos(math.radians(c.brick_upright_deg))
        on_floor = (b[:, 2] > c.brick_h / 2 - 0.010) & (b[:, 2] < c.brick_h / 2 + 0.015)
        lateral = b[:, 0].abs() < c.seat_xy_half
        return at_plate & aligned & upright & on_floor & lateral

    def settled(self) -> torch.Tensor:
        """(N,) bool: brick AND shutter |lin vel| below `settle_lin`."""
        c = self.cfg
        return ((self.brick.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.shutter.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin))

    # ----- graded progress --------------------------------------------------------------------
    def door_open_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: the slab's opening travel |x| as a fraction of the travel
        that fully clears the doorway, past a dead-band (no credit for rattle),
        gated on the slab being captive in its rails. Zero at reset."""
        c = self.cfg
        loc = self.shutter_dock()
        frac = (loc[:, 0].abs() - c.door_open_dead) / (c.x_door_clear - c.door_open_dead)
        return frac.clamp(0.0, 1.0) * self.shutter_in_rails().float()

    def entry_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: the prong-tip midpoint's progress through the doorway
        toward the plate (tip y / seat line), gated on the brick actually being in
        the doorway/bay lane at floor level (a brick on the roof, in the rail
        channel beside the doorway, or waved through the air earns nothing)."""
        c = self.cfg
        b = self.brick_dock()
        t = self.tip_dock()
        in_lane = (b[:, 0].abs() < 0.055) & (b[:, 2] < 0.055) & (b[:, 2] > 0.010)
        frac = t[:, 1] / c.y_seat_min
        return frac.clamp(0.0, 1.0) * in_lane.float()

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch shutter-opening travel, doorway-transit progress, and settled
        docking each physics substep, so transient progress keeps its credit."""
        self.door_latch = torch.maximum(self.door_latch, self.door_open_frac())
        self.entry_latch = torch.maximum(self.entry_latch, self.entry_frac())
        dock_now = self.docked() \
            & (self.brick.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin)
        self.dock_latch = torch.maximum(self.dock_latch, dock_now.float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: brick docked prongs-on-plate + shutter closed over the doorway,
        both settled."""
        return self.docked() & self.shutter_closed() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.20 * latched shutter travel + 0.35 * latched
        doorway transit + 0.15 * docked, capped at 0.85; exactly 1.0 iff success().
        Doing nothing scores ~0; the seed's strategy (insert, episode over — the
        shutter never re-closed) caps at 0.70."""
        base = (0.20 * self.door_latch + 0.35 * self.entry_latch
                + 0.15 * self.dock_latch).clamp(0.0, 0.85)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="shutter_garage", robot="null"))
