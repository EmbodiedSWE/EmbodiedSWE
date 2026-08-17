"""GaugeAdapterScene — the charger does NOT fit the wall outlet: seat the green
travel ADAPTER in the outlet first, then plug the charger into the adapter's top
sockets (sim_gen task `plug_charger_i337`).

Derived from maniskill/plug_charger, but STRATEGICALLY different: the seed is a
terminal DIRECT insertion — pick the free charger off the table, align its two
prongs with the two (slightly enlarged) holes of a fixed base, push it in, done.
Here the seed's entire plan is GEOMETRICALLY EXCLUDED and the task is MEDIATED
mating through a gauge converter:

  - a kinematic OUTLET STATION stands on the floor: a gray panel above a shelf,
    with two wide-gauge slots (20 mm wide, 22 mm tall, at +/-22 mm centers,
    separated by a divider) cut into the panel flush with the shelf top;
  - the WHITE CHARGER's two brass prongs are NARROW gauge: 10 mm square at
    +/-12 mm centers. Each prong overlaps the outlet's center divider by 5 mm, and
    the 24 mm prong spacing can never match the 44 mm slot spacing at ANY planar
    pose — the charger physically cannot enter the wall outlet (asserted in
    `__post_init__`, force-proven in smoke);
  - the GREEN ADAPTER is the converter: two wide-gauge fingers (14 mm, +/-22 mm
    centers) on its front face fit the outlet slots, and two narrow-gauge sockets
    (16 mm square bores, 20 mm deep, +/-12 mm centers, funneled mouths) open in
    its top deck and fit the charger prongs;
  - the ORANGE DECOY adapter is identical outside but its top sockets are 8 mm —
    narrower than the prongs: it can be seated but can never complete the chain
    (also asserted, also force-proven in smoke).

Goal: the assembled two-interface CHAIN — adapter seated flush in the outlet
(fingers deep in both slots, body face on the panel) AND charger plugged
prongs-down into the adapter's top sockets (body resting on the deck), everything
settled. No execution order is required: adapter-first is the natural route, but a
pre-coupled pair slid in together is equally valid — the defining constraint is
that direct wall mating is impossible, so a solver needs a different PLAN
(diagnose the gauge mismatch, select the compatible converter, assemble the chain
through two heterogeneous mates: a horizontal slide-to-flush and a vertical
drop-press) and a different code structure (two relative-frame mating predicates
composed into a chain, plus a physical-impossibility argument) — not "align the
plug with the wall and push".

success() iff, with adapter AND charger AND decoy settled (|v| < settle_lin):
  - SEATED: the green adapter's finger tips deep in the outlet slots (dock-frame
    readback past `seat_y_min`), body centred laterally, standing on the shelf,
    yaw-aligned and upright;
  - MATED: both charger prong tips inside the green adapter's top bores (adapter-
    frame readback: below the deck by more than `mate_depth_min`, inside the bore
    footprints, opposite x signs), prong axis pointing down the bores.

score() is latched every physics substep (credit never evaporates):
  0.35 * best gated seating progress of the adapter fingers into the slots
+ 0.30 * best gated descent progress of the charger prongs into the bores
(max partial 0.65); exactly 1.0 iff success(). Doing nothing scores ~0; pressing
the charger at the wall outlet (the seed's whole strategy) scores ~0 because the
prongs never pass the panel face.

Assets are fully procedural (no external files):
  - outlet station: KINEMATIC compound, 240 mm wide, 160 mm tall panel with the
    twin slots (30 mm deep, back-stopped) and a 90 mm shelf whose top is flush
    with the slot floors at 60 mm height;
  - green adapter: DYNAMIC compound 64 x 50 x 46 mm body, two front fingers, two
    16 mm top bores with 20 mm funnel mouths, 0.35 kg;
  - orange decoy: same body/fingers, 8 mm top bores with 10 mm mouths, 0.35 kg;
  - charger: DYNAMIC white brick 50 x 40 x 32 mm, 0.25 kg, two brass prongs
    (10 mm square, 18 mm proud) on one end face.
Contact offsets are explicit and small (1 mm): the default ~2 cm offset would eat
the 2-3 mm mating clearances.

Per-episode randomization (verified by readback in smoke): station xy jitter +
yaw (insertion directions must be read from the scene), the two adapters SWAP
sides at random with xy jitter + free yaw, charger spawn xy + free yaw.
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
    """Dynamic rigid-body armor on a compound root: mass (PhysX derives inertia
    from the child colliders), damping so parts settle promptly, no sleeping while
    we judge velocities, and the depenetration cap."""
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


def _spawn_station(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC outlet station at `prim_path`. Origin = centre of the
    panel's FRONT face at floor level; local +y INTO the panel (insertion
    direction), x lateral, z up. Panel with two wide-gauge slots (divider between
    them, back-stopped), plinth below, header above, and the shelf whose top is
    flush with the slot floors. Everything axis-aligned boxes."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(10.0)

    co = cfg.contact_offset
    hw = cfg.panel_w / 2
    t = cfg.panel_t
    z_sh = cfg.shelf_top
    z_top = z_sh + cfg.slot_h
    s_in = cfg.slot_cx - cfg.slot_w / 2  # slot inner edge (divider half width)
    s_out = cfg.slot_cx + cfg.slot_w / 2  # slot outer edge
    pc, sc, dc = cfg.panel_color, cfg.shelf_color, cfg.slot_back_color

    # plinth (below the slot floor) + header (above the slots)
    _box(stage, f"{prim_path}/plinth", (2 * hw, t, z_sh), (0.0, t / 2, z_sh / 2), pc, co)
    _box(stage, f"{prim_path}/header", (2 * hw, t, cfg.panel_h - z_top),
         (0.0, t / 2, (z_top + cfg.panel_h) / 2), pc, co)
    # slot-level blocks: outer cheeks + centre divider (the gauge gate)
    _box(stage, f"{prim_path}/cheek_l", (hw - s_out, t, cfg.slot_h),
         (-(s_out + hw) / 2, t / 2, (z_sh + z_top) / 2), pc, co)
    _box(stage, f"{prim_path}/cheek_r", (hw - s_out, t, cfg.slot_h),
         ((s_out + hw) / 2, t / 2, (z_sh + z_top) / 2), pc, co)
    _box(stage, f"{prim_path}/divider", (2 * s_in, t, cfg.slot_h),
         (0.0, t / 2, (z_sh + z_top) / 2), pc, co)
    # slot back stop (slots are slot_d deep, the panel is thicker)
    _box(stage, f"{prim_path}/back", (2 * s_out, t - cfg.slot_d, cfg.slot_h),
         (0.0, (cfg.slot_d + t) / 2, (z_sh + z_top) / 2), dc, co)
    # shelf: top flush with the slot floors
    _box(stage, f"{prim_path}/shelf", (2 * hw, cfg.shelf_d, cfg.shelf_t),
         (0.0, -cfg.shelf_d / 2, z_sh - cfg.shelf_t / 2), sc, co)
    # shelf legs (visual grounding)
    for tag, sx in (("leg_l", -1.0), ("leg_r", 1.0)):
        _box(stage, f"{prim_path}/{tag}", (0.012, 0.012, z_sh - cfg.shelf_t),
             (sx * (hw - 0.010), -cfg.shelf_d + 0.008, (z_sh - cfg.shelf_t) / 2), sc, co)
    return root


def _spawn_adapter(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a DYNAMIC gauge adapter at `prim_path`. Origin = body centre; local
    +y = finger (insertion) direction, +z up. Solid lower block, two front
    fingers, and a two-level top block forming two square bores (width `bore_w`,
    funnel mouths `mouth_w`) at +/-`bore_cx`."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass)

    co = cfg.contact_offset
    hx, hy, hz = cfg.body_w / 2, cfg.body_d / 2, cfg.body_h / 2
    col = cfg.color
    z_deck = hz
    z_bore0 = z_deck - cfg.bore_d          # bore floor level
    z_mouth0 = z_deck - cfg.mouth_d        # funnel-mouth level

    # solid lower block (everything below the bore floors)
    _box(stage, f"{prim_path}/base", (2 * hx, 2 * hy, cfg.body_h - cfg.bore_d),
         (0.0, 0.0, (-hz + z_bore0) / 2), col, co)

    def hole_level(tag: str, w: float, z0: float, z1: float) -> None:
        """One z-level of the top block with two square holes of width w at
        +/-bore_cx (hole y-span |y| < w/2)."""
        th = z1 - z0
        zc = (z0 + z1) / 2
        i_in = cfg.bore_cx - w / 2
        i_out = cfg.bore_cx + w / 2
        _box(stage, f"{prim_path}/{tag}_f", (2 * hx, hy - w / 2, th),
             (0.0, (w / 2 + hy) / 2, zc), col, co)
        _box(stage, f"{prim_path}/{tag}_b", (2 * hx, hy - w / 2, th),
             (0.0, -(w / 2 + hy) / 2, zc), col, co)
        _box(stage, f"{prim_path}/{tag}_l", (hx - i_out, w, th),
             (-(i_out + hx) / 2, 0.0, zc), col, co)
        _box(stage, f"{prim_path}/{tag}_r", (hx - i_out, w, th),
             ((i_out + hx) / 2, 0.0, zc), col, co)
        _box(stage, f"{prim_path}/{tag}_c", (2 * i_in, w, th), (0.0, 0.0, zc), col, co)

    hole_level("lvA", cfg.bore_w, z_bore0, z_mouth0)
    hole_level("lvB", cfg.mouth_w, z_mouth0, z_deck)

    # wide-gauge fingers on the +y face, lifted `finger_lift` above the body
    # bottom so they enter the slot band clear of the slot-floor edge (the body
    # rides the shelf whose top is coplanar with the slot floors — a flush
    # finger bottom would catch the plinth's front face)
    for tag, sx in (("finger_l", -1.0), ("finger_r", 1.0)):
        _box(stage, f"{prim_path}/{tag}", (cfg.finger_w, cfg.finger_len, cfg.finger_h),
             (sx * cfg.finger_cx, hy + cfg.finger_len / 2,
              -hz + cfg.finger_lift + cfg.finger_h / 2),
             cfg.finger_color, co)
    return root


def _spawn_charger(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC charger at `prim_path`. Origin = body centre; local +y =
    prong direction: body + two brass prongs proud of the +y face at mid height."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass)

    co = cfg.contact_offset
    _box(stage, f"{prim_path}/body", (cfg.body_w, cfg.body_d, cfg.body_h),
         (0.0, 0.0, 0.0), cfg.color, co)
    for tag, sx in (("prong_l", -1.0), ("prong_r", 1.0)):
        _box(stage, f"{prim_path}/{tag}", (cfg.prong_w, cfg.prong_len, cfg.prong_w),
             (sx * cfg.prong_cx, (cfg.body_d + cfg.prong_len) / 2, 0.0),
             cfg.prong_color, co)
    return root


def _station_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "station" not in _SPAWNER_CACHE:

        @configclass
        class StationSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_station)
            panel_w: float = 0.240
            panel_t: float = 0.040
            panel_h: float = 0.160
            shelf_top: float = 0.060
            shelf_d: float = 0.090
            shelf_t: float = 0.008
            slot_w: float = 0.020
            slot_h: float = 0.022
            slot_cx: float = 0.022
            slot_d: float = 0.030
            panel_color: tuple = (0.45, 0.48, 0.55)
            shelf_color: tuple = (0.35, 0.37, 0.42)
            slot_back_color: tuple = (0.15, 0.15, 0.18)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["station"] = StationSpawnerCfg

    return _SPAWNER_CACHE["station"](
        mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        **kw,
    )


def _adapter_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "adapter" not in _SPAWNER_CACHE:

        @configclass
        class AdapterSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_adapter)
            body_w: float = 0.064
            body_d: float = 0.050
            body_h: float = 0.046
            bore_w: float = 0.016
            mouth_w: float = 0.020
            bore_d: float = 0.020
            mouth_d: float = 0.005
            bore_cx: float = 0.012
            finger_w: float = 0.014
            finger_h: float = 0.018
            finger_len: float = 0.026
            finger_cx: float = 0.022
            finger_lift: float = 0.002
            mass: float = 0.35
            color: tuple = (0.20, 0.62, 0.30)
            finger_color: tuple = (0.70, 0.64, 0.30)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["adapter"] = AdapterSpawnerCfg

    return _SPAWNER_CACHE["adapter"](
        mass_props=sim_utils.MassPropertiesCfg(mass=kw["mass"]),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        **kw,
    )


def _charger_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "charger" not in _SPAWNER_CACHE:

        @configclass
        class ChargerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_charger)
            body_w: float = 0.050
            body_d: float = 0.040
            body_h: float = 0.032
            prong_w: float = 0.010
            prong_len: float = 0.018
            prong_cx: float = 0.012
            mass: float = 0.25
            color: tuple = (0.92, 0.91, 0.88)
            prong_color: tuple = (0.75, 0.68, 0.30)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["charger"] = ChargerSpawnerCfg

    return _SPAWNER_CACHE["charger"](
        mass_props=sim_utils.MassPropertiesCfg(mass=kw["mass"]),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        **kw,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class GaugeAdapterSceneCfg(BaseCfg):
    """Config for `GaugeAdapterScene`. Honesty knobs asserted in `__post_init__`:
    the charger's prongs genuinely cannot enter the wall outlet at any planar pose
    (spacing + divider interference), the decoy's sockets genuinely refuse the
    prongs, the correct adapter's interfaces genuinely admit their mates with real
    clearance, and a seated adapter leaves its deck sockets open to the sky."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    seat_y_min: float = tunable(0.021)  # finger-tip midpoint y (dock) beyond this = seated (m)
    seat_x_tol: float = tunable(0.010)  # adapter origin |x| (dock) when seated (m)
    seat_z_tol: float = tunable(0.008)  # adapter origin z band around shelf-riding height (m)
    seat_align_deg: float = tunable(10.0)  # adapter +y within this of dock +y when seated
    mate_depth_min: float = tunable(0.012)  # prong tips below the deck by more than this (m)
    mate_align_deg: float = tunable(15.0)  # prong axis within this of straight-down-the-bores
    settle_lin: float = tunable(0.05)  # max |lin vel| (adapter, decoy, charger) judging (m/s)
    seat_gate_y0: float = tunable(0.004)  # seat credit gate: finger tips past the panel face (m)
    mate_gate_dz: float = tunable(0.004)  # mate credit gate: tips below deck by this (m)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    dock_jitter: float = tunable(0.04)  # uniform +/- xy jitter of the station at reset (m)
    dock_yaw_deg: float = tunable(25.0)  # uniform +/- station yaw at reset (deg)
    adapter_lx: float = tunable(0.16)  # adapters spawn at dock-local x = +/- this (sides SWAP)
    adapter_ly: float = tunable(-0.20)  # adapters spawn at dock-local y = this
    adapter_jitter: float = tunable(0.03)  # uniform +/- xy jitter per adapter (m)
    charger_lat: float = tunable(0.10)  # charger spawn: uniform +/- dock-local x range (m)
    charger_y0: float = tunable(-0.36)  # charger spawn: dock-local y range
    charger_y1: float = tunable(-0.26)
    free_yaw_deg: float = tunable(180.0)  # uniform +/- yaw for adapters and charger (free)

    # --- tunable: placement ------------------------------------------------------------------
    dock_pos: tuple = tunable((0.0, 0.10))  # station origin (panel front face centre), nominal

    # --- info: station structure (dock frame: origin = panel front face centre at floor, ----
    # +y INTO the panel, x lateral, z up) ------------------------------------------------------
    panel_w: float = info(0.240)
    panel_t: float = info(0.040)
    panel_h: float = info(0.160)
    shelf_top: float = info(0.060)  # shelf top surface, flush with the slot floors
    shelf_d: float = info(0.090)  # shelf protrudes to y = -this
    shelf_t: float = info(0.008)
    slot_w: float = info(0.020)  # each outlet slot: width (x)
    slot_h: float = info(0.022)  # height (z), from the shelf top
    slot_cx: float = info(0.022)  # slot centres at x = +/- this
    slot_d: float = info(0.030)  # depth (y) to the back stop
    # --- info: adapter structure (origin = body centre, +y = finger direction) ---------------
    ad_body_w: float = info(0.064)
    ad_body_d: float = info(0.050)
    ad_body_h: float = info(0.046)
    bore_w: float = info(0.016)  # correct adapter's socket bores (square)
    mouth_w: float = info(0.020)  # funnel mouth width (top mouth_d of the bore)
    bore_d: float = info(0.020)  # bore depth below the deck
    mouth_d: float = info(0.005)
    bore_cx: float = info(0.012)  # bores at local x = +/- this (charger gauge)
    decoy_bore_w: float = info(0.008)  # decoy's sockets: NARROWER than the prongs
    decoy_mouth_w: float = info(0.010)
    finger_w: float = info(0.014)
    finger_h: float = info(0.018)
    finger_len: float = info(0.026)
    finger_cx: float = info(0.022)  # fingers at local x = +/- this (outlet gauge)
    finger_lift: float = info(0.002)  # finger bottoms this far above the body bottom
    adapter_mass: float = info(0.35)
    # --- info: charger structure (origin = body centre, +y = prong direction) ----------------
    ch_body_w: float = info(0.050)
    ch_body_d: float = info(0.040)
    ch_body_h: float = info(0.032)
    prong_w: float = info(0.010)
    prong_len: float = info(0.018)
    prong_cx: float = info(0.012)  # prongs at local x = +/- this (charger gauge)
    charger_mass: float = info(0.25)
    # --- info: colors + misc -----------------------------------------------------------------
    panel_color: tuple = info((0.45, 0.48, 0.55))
    shelf_color: tuple = info((0.35, 0.37, 0.42))
    slot_back_color: tuple = info((0.15, 0.15, 0.18))
    adapter_color: tuple = info((0.20, 0.62, 0.30))  # GREEN = correct converter
    decoy_color: tuple = info((0.85, 0.45, 0.15))  # ORANGE = narrow-socket decoy
    finger_color: tuple = info((0.70, 0.64, 0.30))
    charger_color: tuple = info((0.92, 0.91, 0.88))
    prong_color: tuple = info((0.75, 0.68, 0.30))  # brass
    contact_offset: float = info(0.001)  # explicit: default ~2 cm would eat the clearances

    # Derived (filled in __post_init__).
    ad_z_ride: float = field(default=None, init=False)  # adapter origin z riding the shelf
    ad_z_floor: float = field(default=None, init=False)  # adapter origin z on the ground
    ch_z_floor: float = field(default=None, init=False)  # charger origin z on the ground
    deck_dz: float = field(default=None, init=False)  # adapter origin -> deck plane (local +z)
    finger_tip_dy: float = field(default=None, init=False)  # adapter origin -> finger tip (+y)
    finger_tip_dz: float = field(default=None, init=False)  # adapter origin -> finger centre z
    prong_tip_dy: float = field(default=None, init=False)  # charger origin -> prong tip (+y)
    seat_tip_y: float = field(default=None, init=False)  # finger-tip y when flush-seated
    mate_tip_dz: float = field(default=None, init=False)  # tip below deck when fully mated

    def __post_init__(self) -> None:
        self.ad_z_ride = self.shelf_top + self.ad_body_h / 2
        self.ad_z_floor = self.ad_body_h / 2
        self.ch_z_floor = self.ch_body_h / 2
        self.deck_dz = self.ad_body_h / 2
        self.finger_tip_dy = self.ad_body_d / 2 + self.finger_len
        self.finger_tip_dz = -self.ad_body_h / 2 + self.finger_lift + self.finger_h / 2
        self.prong_tip_dy = self.ch_body_d / 2 + self.prong_len
        self.seat_tip_y = self.finger_len  # body face flush at y=0 -> tips at finger_len
        self.mate_tip_dz = self.prong_len  # body resting on the deck -> tips prong_len deep

        s_in = self.slot_cx - self.slot_w / 2  # divider half width (0.012)
        s_out = self.slot_cx + self.slot_w / 2
        # --- the strategic crux: the charger CANNOT enter the wall outlet ---
        p_in = self.prong_cx - self.prong_w / 2  # prong inner edge (0.007)
        assert s_in - p_in >= 0.004, (
            "each prong must overlap the outlet divider by >= 4 mm (both prongs "
            "simultaneously blocked at the panel face)")
        # one prong centred in a slot puts the other prong's centre on the divider:
        other = 2 * self.prong_cx - self.slot_cx  # signed x of the other prong
        assert abs(other) + self.prong_w / 2 <= s_in, (
            "with one prong centred in a slot, the other prong must land fully on "
            "the divider (single-prong sneak insertion also blocked)")
        # both prongs in different slots at once needs the prongs' inner gap to
        # clear the divider: 2*p_in >= 2*s_in. Require the opposite with margin.
        assert 2 * p_in <= 2 * s_in - 0.004, (
            "the gap between the prongs' inner edges must be narrower than the "
            "divider (the prong pair can never straddle it into both slots)")
        # --- the decoy refuses the prongs, the correct adapter admits them ---
        assert self.decoy_bore_w <= self.prong_w - 0.001, (
            "decoy sockets must be narrower than the prongs")
        assert self.decoy_mouth_w <= self.prong_w + 0.001, (
            "decoy funnel mouths must not admit a prong either")
        assert self.bore_w >= self.prong_w + 0.004, (
            "correct sockets must admit the prongs with >= 2 mm/side clearance")
        assert self.mouth_w > self.bore_w, "funnel mouths must be wider than the bores"
        assert self.bore_cx == self.prong_cx, "socket gauge must match the prong gauge"
        # --- the adapter fits the outlet with real clearance ---
        assert self.slot_w >= self.finger_w + 0.004, (
            "outlet slots must admit the fingers with >= 2 mm/side clearance")
        # fingers ride `finger_lift` above the slot floor (the body rests on the
        # shelf, whose top is coplanar with the slot floors — a flush finger
        # bottom would catch the slot-floor front edge) and need top headroom too
        assert self.finger_lift >= 0.0015, "fingers must clear the slot-floor edge"
        assert self.slot_h >= self.finger_lift + self.finger_h + 0.0015, (
            "slots need finger headroom above the lifted fingers")
        assert self.finger_cx == self.slot_cx, "finger gauge must match the outlet gauge"
        assert self.ad_body_h > self.slot_h + 0.010, (
            "the adapter BODY must not fit into the slots (only the fingers enter; "
            "the panel face is the flush stop)")
        assert self.finger_len <= self.slot_d - 0.003, (
            "flush stop must be body-face-on-panel, not finger-tip-on-back-wall")
        assert self.prong_len <= self.bore_d - 0.001, (
            "mate stop must be charger-body-on-deck, not prong-tip-on-bore-floor")
        assert self.mate_tip_dz >= self.mate_depth_min + 0.004, (
            "a flush-mated charger must clear the depth threshold with margin")
        assert self.seat_tip_y >= self.seat_y_min + 0.003, (
            "a flush-seated adapter must clear the seat threshold with margin")
        assert self.shelf_d >= self.ad_body_d + 0.030, (
            "the shelf must stage the whole adapter body in front of the panel")
        # a seated adapter's deck stands proud of the header line only if the body
        # is taller than slot_h + header start; with body 46 mm on a 60 mm shelf the
        # deck is at 106 mm and the header starts at 82 mm, 40 mm BEHIND the deck
        # (panel plane y>=0 vs body y<0) — the bores open to the sky. Assert the
        # geometric fact that matters: the seated body sits fully in front of the
        # panel face, so nothing above it can shadow the bores.
        assert self.finger_len <= self.slot_d, "seated body face at y=0 (in front of panel)"
        assert self.bore_d <= self.ad_body_h, "bores inside body"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("gauge_adapter")
class GaugeAdapterScene(BaseScene):
    cfg: GaugeAdapterSceneCfg

    def __init__(self, cfg: GaugeAdapterSceneCfg | None = None) -> None:
        super().__init__(cfg or GaugeAdapterSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        station_kw = dict(
            panel_w=c.panel_w, panel_t=c.panel_t, panel_h=c.panel_h,
            shelf_top=c.shelf_top, shelf_d=c.shelf_d, shelf_t=c.shelf_t,
            slot_w=c.slot_w, slot_h=c.slot_h, slot_cx=c.slot_cx, slot_d=c.slot_d,
            panel_color=c.panel_color, shelf_color=c.shelf_color,
            slot_back_color=c.slot_back_color, contact_offset=c.contact_offset,
        )
        adapter_kw = dict(
            body_w=c.ad_body_w, body_d=c.ad_body_d, body_h=c.ad_body_h,
            bore_d=c.bore_d, mouth_d=c.mouth_d, bore_cx=c.bore_cx,
            finger_w=c.finger_w, finger_h=c.finger_h, finger_len=c.finger_len,
            finger_cx=c.finger_cx, finger_lift=c.finger_lift, mass=c.adapter_mass,
            finger_color=c.finger_color, contact_offset=c.contact_offset,
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
            "station": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Station",
                spawn=_station_spawner_cfg(**station_kw),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.dock_pos[0], c.dock_pos[1], 0.0005)),
            ),
            "adapter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Adapter",
                spawn=_adapter_spawner_cfg(
                    bore_w=c.bore_w, mouth_w=c.mouth_w, color=c.adapter_color,
                    **adapter_kw),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-c.adapter_lx, c.dock_pos[1] + c.adapter_ly,
                         c.ad_z_floor + 0.0015)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=_adapter_spawner_cfg(
                    bore_w=c.decoy_bore_w, mouth_w=c.decoy_mouth_w, color=c.decoy_color,
                    **adapter_kw),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.adapter_lx, c.dock_pos[1] + c.adapter_ly,
                         c.ad_z_floor + 0.0015)),
            ),
            "charger": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Charger",
                spawn=_charger_spawner_cfg(
                    body_w=c.ch_body_w, body_d=c.ch_body_d, body_h=c.ch_body_h,
                    prong_w=c.prong_w, prong_len=c.prong_len, prong_cx=c.prong_cx,
                    mass=c.charger_mass, color=c.charger_color,
                    prong_color=c.prong_color, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.dock_pos[0], c.dock_pos[1] - 0.30,
                         c.ch_z_floor + 0.0015)),
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
        self.station: RigidObject = env.iscene["station"]
        self.adapter: RigidObject = env.iscene["adapter"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.charger: RigidObject = env.iscene["charger"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # progress latches
        self.seat_latch = torch.zeros(n, device=dev)
        self.mate_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: station with xy jitter + yaw, the two adapters SWAP sides
        at random (with jitter + free yaw), charger loose in front with jitter +
        free yaw; latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # burn a draw (the degenerate-first-randint trap) before the discrete swap
        _ = torch.rand(m, 4, device=dev)

        # --- station: xy jitter + yaw (kinematic root) ---
        dock_xy = torch.tensor(c.dock_pos, device=dev).expand(m, 2).clone()
        dock_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.dock_jitter
        dock_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.dock_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = dock_xy
        st[:, 2] = 0.0005
        st[:, 3] = torch.cos(dock_yaw / 2)
        st[:, 6] = torch.sin(dock_yaw / 2)
        st[:, 0:3] += origin
        self.station.write_root_state_to_sim(st, env_ids)

        ca, sa = torch.cos(dock_yaw), torch.sin(dock_yaw)

        def to_world(lx: torch.Tensor, ly: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return dock_xy[:, 0] + ca * lx - sa * ly, dock_xy[:, 1] + sa * lx + ca * ly

        # --- adapters: sides swap at random, jitter + free yaw, resting on the ground ---
        side = torch.where(torch.rand(m, device=dev) < 0.5, 1.0,
                           -torch.ones(m, device=dev))
        yaw_amp = math.radians(c.free_yaw_deg)
        for body, sgn in ((self.adapter, side), (self.decoy, -side)):
            lx = sgn * c.adapter_lx + (torch.rand(m, device=dev) * 2 - 1) * c.adapter_jitter
            ly = c.adapter_ly + (torch.rand(m, device=dev) * 2 - 1) * c.adapter_jitter
            wx, wy = to_world(lx, ly)
            byaw = dock_yaw + (torch.rand(m, device=dev) * 2 - 1) * yaw_amp
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1] = wx, wy
            st[:, 2] = c.ad_z_floor + 0.0015
            st[:, 3] = torch.cos(byaw / 2)
            st[:, 6] = torch.sin(byaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- charger: loose on the ground in front, jitter + free yaw ---
        lx = (torch.rand(m, device=dev) * 2 - 1) * c.charger_lat
        ly = c.charger_y0 + torch.rand(m, device=dev) * (c.charger_y1 - c.charger_y0)
        wx, wy = to_world(lx, ly)
        byaw = dock_yaw + (torch.rand(m, device=dev) * 2 - 1) * yaw_amp
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = wx, wy
        st[:, 2] = c.ch_z_floor + 0.0015
        st[:, 3] = torch.cos(byaw / 2)
        st[:, 6] = torch.sin(byaw / 2)
        st[:, 0:3] += origin
        self.charger.write_root_state_to_sim(st, env_ids)

        # --- latches ---
        self.seat_latch[env_ids] = 0.0
        self.mate_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "station": self.station.data.root_state_w[env_ids].clone(),
            "adapter": self.adapter.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "charger": self.charger.data.root_state_w[env_ids].clone(),
            "seat_latch": self.seat_latch[env_ids].clone(),
            "mate_latch": self.mate_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.station.write_root_state_to_sim(state["station"], env_ids)
        self.adapter.write_root_state_to_sim(state["adapter"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.charger.write_root_state_to_sim(state["charger"], env_ids)
        self.seat_latch[env_ids] = state["seat_latch"]
        self.mate_latch[env_ids] = state["mate_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A gray OUTLET STATION stands on the floor: a {c.panel_w * 1000:.0f} mm wide, "
            f"{c.panel_h * 1000:.0f} mm tall panel with a WALL OUTLET — two dark slots, "
            f"each {c.slot_w * 1000:.0f} mm wide and {c.slot_h * 1000:.0f} mm tall, "
            f"{2 * c.slot_cx * 1000:.0f} mm apart centre-to-centre, separated by a solid "
            f"divider — cut into it just above a shelf whose top is flush with the slot "
            f"floors ({c.shelf_top * 1000:.0f} mm high, protruding "
            f"{c.shelf_d * 1000:.0f} mm). On the floor in front lie three loose objects "
            f"(positions, sides and headings vary; the station's heading varies too — "
            f"read directions from the scene):\n"
            f"  - a WHITE CHARGER ({c.ch_body_w * 1000:.0f} x {c.ch_body_d * 1000:.0f} x "
            f"{c.ch_body_h * 1000:.0f} mm) with two BRASS PRONGS "
            f"({c.prong_w * 1000:.0f} mm square, {2 * c.prong_cx * 1000:.0f} mm apart) on "
            f"one end face. Its prong gauge does NOT match the wall outlet: each prong "
            f"overlaps the outlet's centre divider, so the charger physically cannot be "
            f"plugged into the wall no matter how it is turned;\n"
            f"  - a GREEN ADAPTER (a {c.ad_body_w * 1000:.0f} x {c.ad_body_d * 1000:.0f} x "
            f"{c.ad_body_h * 1000:.0f} mm block): two wide-gauge brass FINGERS on its "
            f"front face match the wall outlet, and two {c.bore_w * 1000:.0f} mm square "
            f"SOCKETS with funneled mouths open in its top deck and match the charger's "
            f"prongs;\n"
            f"  - an ORANGE DECOY adapter, identical outside, but its top sockets are only "
            f"{c.decoy_bore_w * 1000:.0f} mm — narrower than the prongs — so it can never "
            f"take the charger.\n"
            f"Goal: assemble the charging chain. Put the GREEN adapter on the shelf and "
            f"slide it fingers-first into the wall outlet until its body sits flush "
            f"against the panel (fingers deep in both slots, body centred and aligned), "
            f"and plug the CHARGER prongs-down into the green adapter's top sockets until "
            f"its body rests on the deck (both prongs at least "
            f"{c.mate_depth_min * 1000:.0f} mm below the deck, pointing down the bores). "
            f"No particular order is required — but the wall outlet takes only one adapter "
            f"at a time, and only the green one completes the chain. Success is judged "
            f"with everything settled: adapter seated in the outlet AND charger plugged "
            f"into the adapter. A charger pressed at the wall outlet, a seated decoy, a "
            f"charger perched on a deck without its prongs in the sockets, or a mated "
            f"pair left loose on the floor all count as incomplete."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "The white charger's prongs do not fit the wall outlet, and the orange "
            "adapter's sockets are too narrow. Slide the green adapter along the shelf "
            "into the wall outlet until it sits flush against the panel, and plug the "
            "charger prongs-down into the green adapter's top sockets until its body "
            "rests on the deck."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _dock_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> station (dock) frame (origin = panel front face
        centre at floor level, +y into the panel)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.station.data.root_quat_w,
                                  p_w - self.station.data.root_pos_w)

    def _body_axis(self, body, axis: tuple) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        v = torch.tensor(axis, device=self.env.device, dtype=torch.float32)
        return quat_apply(body.data.root_quat_w, v.expand(self.env.num_envs, 3))

    def adapter_dock(self) -> torch.Tensor:
        """(N,3) green adapter origin in the dock frame."""
        return self._dock_local(self.adapter.data.root_pos_w)

    def decoy_dock(self) -> torch.Tensor:
        return self._dock_local(self.decoy.data.root_pos_w)

    def charger_dock(self) -> torch.Tensor:
        return self._dock_local(self.charger.data.root_pos_w)

    def _finger_tip_w(self, body) -> torch.Tensor:
        """(N,3) world midpoint between an adapter's two finger tips."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        p = torch.tensor([0.0, c.finger_tip_dy, c.finger_tip_dz],
                         device=self.env.device)
        return body.data.root_pos_w + quat_apply(
            body.data.root_quat_w, p.expand(self.env.num_envs, 3))

    def finger_tip_dock(self) -> torch.Tensor:
        """(N,3) green adapter finger-tip midpoint in the dock frame."""
        return self._dock_local(self._finger_tip_w(self.adapter))

    def decoy_tip_dock(self) -> torch.Tensor:
        return self._dock_local(self._finger_tip_w(self.decoy))

    def prong_tips_w(self) -> torch.Tensor:
        """(N,2,3) world positions of the charger's two prong tips."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        pts = torch.tensor([[-c.prong_cx, c.prong_tip_dy, 0.0],
                            [c.prong_cx, c.prong_tip_dy, 0.0]], device=self.env.device)
        q = self.charger.data.root_quat_w[:, None, :].expand(n, 2, 4).reshape(-1, 4)
        loc = pts[None].expand(n, 2, 3).reshape(-1, 3)
        return (self.charger.data.root_pos_w[:, None, :]
                + quat_apply(q, loc).reshape(n, 2, 3))

    def prong_tips_dock(self) -> torch.Tensor:
        """(N,2,3) prong tips in the dock frame (for the direct-wall probes)."""
        tips = self.prong_tips_w()
        return torch.stack(
            [self._dock_local(tips[:, 0]), self._dock_local(tips[:, 1])], dim=1)

    def prong_tips_adapter(self) -> torch.Tensor:
        """(N,2,3) charger prong tips in the GREEN ADAPTER's body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        tips = self.prong_tips_w()
        n = tips.shape[0]
        q = self.adapter.data.root_quat_w[:, None, :].expand(n, 2, 4).reshape(-1, 4)
        d = (tips - self.adapter.data.root_pos_w[:, None, :]).reshape(-1, 3)
        return quat_apply_inverse(q, d).reshape(n, 2, 3)

    def adapter_align(self) -> torch.Tensor:
        """(N,) cosine between the green adapter's +y (finger axis) and dock +y."""
        from isaaclab.utils.math import quat_apply_inverse

        a = self._body_axis(self.adapter, (0.0, 1.0, 0.0))
        return quat_apply_inverse(self.station.data.root_quat_w, a)[:, 1]

    def prong_axis_down(self) -> torch.Tensor:
        """(N,) cosine between the charger's prong axis (+y) and MINUS the green
        adapter's up axis (straight down the bores)."""
        pa = self._body_axis(self.charger, (0.0, 1.0, 0.0))
        au = self._body_axis(self.adapter, (0.0, 0.0, 1.0))
        return -(pa * au).sum(dim=-1)

    # ----- predicates -------------------------------------------------------------------------
    def adapter_seated(self) -> torch.Tensor:
        """(N,) bool, geometric: green adapter flush in the outlet — finger tips
        deep in the slots (dock frame), body centred laterally, riding the shelf,
        yaw-aligned, upright."""
        c = self.cfg
        a = self.adapter_dock()
        t = self.finger_tip_dock()
        deep = (t[:, 1] > c.seat_y_min) & (t[:, 1] < c.slot_d + 0.004)
        centred = a[:, 0].abs() < c.seat_x_tol
        on_shelf = (a[:, 2] - c.ad_z_ride).abs() < c.seat_z_tol
        aligned = self.adapter_align() > math.cos(math.radians(c.seat_align_deg))
        upright = self._body_axis(self.adapter, (0.0, 0.0, 1.0))[:, 2] > \
            math.cos(math.radians(c.seat_align_deg))
        return deep & centred & on_shelf & aligned & upright

    def charger_mated(self) -> torch.Tensor:
        """(N,) bool, geometric: both charger prong tips inside the GREEN adapter's
        top bores (adapter frame): below the deck by more than `mate_depth_min`,
        above the bore floors, inside the bore footprints with opposite x signs,
        prong axis pointing down the bores."""
        c = self.cfg
        tp = self.prong_tips_adapter()  # (N,2,3)
        z_max = c.deck_dz - c.mate_depth_min
        z_min = c.deck_dz - c.bore_d - 0.005
        deep = (tp[:, :, 2] < z_max) & (tp[:, :, 2] > z_min)
        in_y = tp[:, :, 1].abs() < c.bore_w / 2 + 0.002
        ax = tp[:, :, 0]
        in_x = ((ax.abs() > c.bore_cx - c.bore_w / 2 - 0.002)
                & (ax.abs() < c.bore_cx + c.bore_w / 2 + 0.002))
        opposite = (ax[:, 0] * ax[:, 1]) < 0
        down = self.prong_axis_down() > math.cos(math.radians(c.mate_align_deg))
        return deep.all(dim=1) & in_y.all(dim=1) & in_x.all(dim=1) & opposite & down

    def settled(self) -> torch.Tensor:
        """(N,) bool: adapter, decoy AND charger |lin vel| below `settle_lin`."""
        c = self.cfg
        return ((self.adapter.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.decoy.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.charger.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin))

    # ----- graded progress --------------------------------------------------------------------
    def seat_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: green adapter finger-tip progress into the slots, from
        just past the panel face (`seat_gate_y0`) to the seat line, gated on the
        adapter actually riding the shelf, roughly centred, aligned and upright (an
        adapter waved through the air or parked beside the outlet earns nothing)."""
        c = self.cfg
        a = self.adapter_dock()
        t = self.finger_tip_dock()
        gate = ((a[:, 2] - c.ad_z_ride).abs() < 0.012) \
            & (a[:, 0].abs() < 0.020) \
            & (self.adapter_align() > math.cos(math.radians(25.0))) \
            & (self._body_axis(self.adapter, (0.0, 0.0, 1.0))[:, 2] > math.cos(
                math.radians(20.0)))
        frac = (t[:, 1] - c.seat_gate_y0) / (c.seat_y_min - c.seat_gate_y0)
        return frac.clamp(0.0, 1.0) * gate.float()

    def mate_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: charger prong descent into the green adapter's bores,
        from `mate_gate_dz` below the deck to the depth threshold, gated on both
        tips being inside the bore footprints with the prong axis pointing down
        (a charger perched on the deck or hovering above it earns nothing)."""
        c = self.cfg
        tp = self.prong_tips_adapter()
        in_y = (tp[:, :, 1].abs() < c.bore_w / 2 + 0.002).all(dim=1)
        ax = tp[:, :, 0]
        in_x = ((ax.abs() > c.bore_cx - c.bore_w / 2 - 0.002)
                & (ax.abs() < c.bore_cx + c.bore_w / 2 + 0.002)).all(dim=1)
        opposite = (ax[:, 0] * ax[:, 1]) < 0
        down = self.prong_axis_down() > math.cos(math.radians(30.0))
        gate = in_y & in_x & opposite & down
        depth = c.deck_dz - tp[:, :, 2].max(dim=1).values  # shallower tip governs
        frac = (depth - c.mate_gate_dz) / (c.mate_depth_min - c.mate_gate_dz)
        return frac.clamp(0.0, 1.0) * gate.float()

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch seating and mating progress each physics substep, so transient
        progress keeps its credit."""
        self.seat_latch = torch.maximum(self.seat_latch, self.seat_frac())
        self.mate_latch = torch.maximum(self.mate_latch, self.mate_frac())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: green adapter seated in the outlet + charger mated into its
        top sockets, everything settled — the assembled charging chain."""
        return self.adapter_seated() & self.charger_mated() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.35 * latched seating progress + 0.30 * latched
        mating progress (max partial 0.65); exactly 1.0 iff success(). Doing
        nothing scores ~0; pressing the charger at the wall outlet (the seed's
        whole strategy) scores ~0 (the prongs never pass the panel face); a seated
        decoy earns nothing (seat credit tracks the GREEN adapter)."""
        base = (0.35 * self.seat_latch + 0.30 * self.mate_latch).clamp(0.0, 0.65)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="gauge_adapter", robot="null"))
