"""HaspPinScene — pin the latch: align the bar's eye over the platform bore, then drop
the headed pin down THROUGH BOTH holes to link them (sim_gen task `peg_insertion_side_i2`).

Derived from maniskill/peg_insertion_side, but STRATEGICALLY different: the seed is a
single terminal insertion — grasp a peg lying on the table, align it horizontally, and
translate it into a tight hole in ONE fixed block; its checker is pure relative-bbox
containment of the peg in the block's frame, so the hole passage pre-exists and the
task ends the instant the peg pose is deep enough. Here the judged outcome is a
FASTENED LINKAGE between two bodies, and the passage the fastener needs DOES NOT EXIST
at reset — the solver must first CREATE it: lay the free bar on the elevated platform
so the square hole in the bar's eye stacks over the platform's square bore (a
plate-on-plate alignment sub-goal with its own tolerance), and only then drop the
headed pin down through the two aligned holes until its head seats on the eye and its
tip hangs clear through into the open gap under the plate. Execution order is REQUIRED
and physically enforced: the pin head (34 mm) is wider than both holes (26 mm), so a
pin dropped into the bare platform bore first leaves a captive head sticking up that
the bar's eye can never pass over — the seed's whole strategy ("put the peg in the
fixed block's hole, done") is exactly that out-of-order end state and is rejected by
the rubric (smoke check). A red decoy pin with a 30 mm shaft cannot enter either hole
(identity control). A solver therefore needs a different PLAN (two-body alignment
first, vertical gravity-assisted fastening second, correct-fastener selection) and a
different code structure (a seat predicate on body A, a through-linkage predicate
tying the fastener to BOTH bodies, and an order-latched score) — not a peg-pose bbox.

Judged in the STAND's body frame (kinematic, xy + free yaw randomized, so the bore
axis and hole orientation must be read from the scene). success() iff:
  - the bar is SEATED: eye centre within `align_tol` of the bore axis, bar resting
    flat on the plate top, upright;
  - the pin is LINKED: near-vertical, shaft passing through the bar's eye (axis point
    at the eye plane inside the hole, in the BAR frame), tip below the plate underside
    by `through_margin` and inside the bore cross-section (STAND frame), and the HEAD
    seated at the eye top (this clause kills the axis-line loophole of a pin standing
    on the ground in the gap under the plate);
  - bar and pin settled.
score() is latched every physics substep: 0.15 * best bar approach toward the bore
axis (normalized by the episode's own spawn distance) + 0.25 * bar seated + 0.45 *
best pin-through depth (gated on the tip actually being inside the EYE hole with the
bar seated — pin credit is impossible before the passage exists, which is the order
latch), capped at 0.85; exactly 1.0 iff success(). Doing nothing scores ~0.

Assets are fully procedural (no external files):
  - stand: KINEMATIC gray platform — top plate 140 x 140 x 12 mm, top face 80 mm up,
    with a 26 mm square bore through its centre, carried by two side legs (open front
    and back: the gap below the bore is visible and the pin tip hangs into it).
  - bar: GREEN flat bar (dynamic) — a 64 x 64 x 16 mm eye plate with a matching 26 mm
    square hole, a 96 x 26 mm handle, and a raised darker grip block (34 x 26 x 28 mm)
    mid-handle for a clean jaw grasp.
  - pin: BLUE steel pin (dynamic) — 18 mm shaft, 72 mm long, with a 34 mm round head:
    4 mm radial clearance in the holes, head cannot pass, tip hangs 40+ mm through.
  - decoy: RED pin, 30 mm shaft — too fat to enter either hole.
Contact offsets are explicit and small (1 mm): the default ~2 cm offset would eat the
4 mm working clearance in the holes.

Per-episode randomization (verified by readback in smoke): stand xy + free yaw, bar
xy + free yaw, pin xy + free yaw (lying flat), decoy xy + free yaw, with batched
keep-out resampling so nothing spawns intersecting. Heavy imports (isaaclab, pxr) are
deferred so importing this module — and registering the scene — stays app-free.
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


def _cyl(stage, path: str, radius: float, height: float, center_z: float, color,
         contact_offset: float) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    seg.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(center_z)))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _rigid_dynamic(root, mass: float) -> None:
    """Dynamic rigid-body armor on a compound root: mass (PhysX derives inertia from
    the child colliders), damping so light parts settle promptly, no sleeping while
    we judge velocities, and the depenetration cap."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.10)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC stand at `prim_path`. Origin = the BORE AXIS at table
    level; the top plate (with the square bore through its centre) rides on two side
    legs at local y = +/- so the gap below the bore is open along local x."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(8.0)

    co, color = cfg.contact_offset, cfg.color
    half_out = cfg.plate_out / 2
    hh = cfg.hole_half
    pt, pb = cfg.plate_top, cfg.plate_top - cfg.plate_t
    pz = (pt + pb) / 2
    leg_h = pb
    # two legs at local +/- y, plate-length along x
    for tag, sgn in (("leg_yp", 1.0), ("leg_yn", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (cfg.plate_out, cfg.leg_w, leg_h),
             (0.0, sgn * (half_out - cfg.leg_w / 2), leg_h / 2), cfg.leg_color, co)
    # top plate as four boxes around the central square bore
    side = half_out - hh  # width of the y+/y- full-length slabs
    _box(stage, f"{prim_path}/plate_yp", (cfg.plate_out, side, cfg.plate_t),
         (0.0, hh + side / 2, pz), color, co)
    _box(stage, f"{prim_path}/plate_yn", (cfg.plate_out, side, cfg.plate_t),
         (0.0, -(hh + side / 2), pz), color, co)
    _box(stage, f"{prim_path}/plate_xp", (side, 2 * hh, cfg.plate_t),
         (hh + side / 2, 0.0, pz), color, co)
    _box(stage, f"{prim_path}/plate_xn", (side, 2 * hh, cfg.plate_t),
         (-(hh + side / 2), 0.0, pz), color, co)
    return root


def _spawn_bar(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC bar at `prim_path`. Origin = the EYE CENTRE at
    mid-thickness; the square eye hole is four boxes, the handle extends along local
    +x with the raised grip block on top mid-handle."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass)

    co, color = cfg.contact_offset, cfg.color
    eh = cfg.eye_out / 2
    hh = cfg.hole_half
    t = cfg.bar_t
    side = eh - hh
    _box(stage, f"{prim_path}/eye_yp", (cfg.eye_out, side, t), (0.0, hh + side / 2, 0.0), color, co)
    _box(stage, f"{prim_path}/eye_yn", (cfg.eye_out, side, t), (0.0, -(hh + side / 2), 0.0), color, co)
    _box(stage, f"{prim_path}/eye_xp", (side, 2 * hh, t), (hh + side / 2, 0.0, 0.0), color, co)
    _box(stage, f"{prim_path}/eye_xn", (side, 2 * hh, t), (-(hh + side / 2), 0.0, 0.0), color, co)
    _box(stage, f"{prim_path}/handle", (cfg.handle_len, cfg.handle_w, t),
         (eh + cfg.handle_len / 2, 0.0, 0.0), color, co)
    _box(stage, f"{prim_path}/grip", (cfg.block_len, cfg.handle_w, cfg.block_h),
         (cfg.block_x, 0.0, t / 2 + cfg.block_h / 2), cfg.block_color, co)
    return root


def _spawn_pin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a DYNAMIC headed pin at `prim_path`. Origin = the SHAFT CENTRE, axis =
    local +z, tip at local -shaft_len/2, head on top."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass)

    co = cfg.contact_offset
    _cyl(stage, f"{prim_path}/shaft", cfg.shaft_r, cfg.shaft_len, 0.0, cfg.color, co)
    _cyl(stage, f"{prim_path}/head", cfg.head_r, cfg.head_h,
         cfg.shaft_len / 2 + cfg.head_h / 2, cfg.head_color, co)
    return root


def _stand_spawner_cfg(*, plate_out: float, plate_t: float, plate_top: float,
                       hole_half: float, leg_w: float, color: tuple, leg_color: tuple,
                       contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "stand" not in _SPAWNER_CACHE:

        @configclass
        class HaspStandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            plate_out: float = 0.140
            plate_t: float = 0.012
            plate_top: float = 0.080
            hole_half: float = 0.013
            leg_w: float = 0.016
            color: tuple = (0.45, 0.45, 0.48)
            leg_color: tuple = (0.30, 0.30, 0.33)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["stand"] = HaspStandSpawnerCfg

    return _SPAWNER_CACHE["stand"](
        mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        plate_out=plate_out, plate_t=plate_t, plate_top=plate_top, hole_half=hole_half,
        leg_w=leg_w, color=color, leg_color=leg_color, contact_offset=contact_offset,
    )


def _bar_spawner_cfg(*, eye_out: float, hole_half: float, bar_t: float, handle_len: float,
                     handle_w: float, block_len: float, block_h: float, block_x: float,
                     mass: float, color: tuple, block_color: tuple,
                     contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bar" not in _SPAWNER_CACHE:

        @configclass
        class HaspBarSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bar)
            eye_out: float = 0.064
            hole_half: float = 0.013
            bar_t: float = 0.016
            handle_len: float = 0.096
            handle_w: float = 0.026
            block_len: float = 0.034
            block_h: float = 0.028
            block_x: float = 0.080
            mass: float = 0.12
            color: tuple = (0.10, 0.60, 0.25)
            block_color: tuple = (0.05, 0.35, 0.15)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["bar"] = HaspBarSpawnerCfg

    return _SPAWNER_CACHE["bar"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        eye_out=eye_out, hole_half=hole_half, bar_t=bar_t, handle_len=handle_len,
        handle_w=handle_w, block_len=block_len, block_h=block_h, block_x=block_x,
        mass=mass, color=color, block_color=block_color, contact_offset=contact_offset,
    )


def _pin_spawner_cfg(*, key: str, shaft_r: float, shaft_len: float, head_r: float,
                     head_h: float, mass: float, color: tuple, head_color: tuple,
                     contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pin" not in _SPAWNER_CACHE:

        @configclass
        class HeadedPinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pin)
            shaft_r: float = 0.009
            shaft_len: float = 0.072
            head_r: float = 0.017
            head_h: float = 0.010
            mass: float = 0.08
            color: tuple = (0.25, 0.45, 0.90)
            head_color: tuple = (0.20, 0.35, 0.75)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["pin"] = HeadedPinSpawnerCfg

    return _SPAWNER_CACHE["pin"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        shaft_r=shaft_r, shaft_len=shaft_len, head_r=head_r, head_h=head_h,
        mass=mass, color=color, head_color=head_color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class HaspPinSceneCfg(BaseCfg):
    """Config for `HaspPinScene`. Honesty knobs asserted in `__post_init__`: the pin
    threads both holes with real clearance, the decoy physically cannot, the head
    physically cannot pass (which is what enforces bar-before-pin ordering AND kills
    the drop-it-all-the-way-through degenerate), and a seated pin's tip hangs free of
    the ground so the head genuinely carries it."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    align_tol: float = tunable(0.010)  # bar eye centre within this of the bore axis, per axis (m)
    seat_z_tol: float = tunable(0.006)  # bar underside within this of the plate top (m)
    bar_upright_deg: float = tunable(15.0)  # bar +z within this of world-up when seated
    pin_upright_deg: float = tunable(25.0)  # pin axis within this of world-up when linked
    tip_xy_tol: float = tunable(0.010)  # pin tip within this of the bore axis, per axis (m)
    through_margin: float = tunable(0.008)  # tip below the plate underside by at least this (m)
    head_seat_low: float = tunable(0.010)  # head underside no more than this BELOW the eye top (m)
    head_seat_high: float = tunable(0.015)  # ... and no more than this ABOVE it (head truly seated)
    settle_lin: float = tunable(0.05)  # max |lin vel| (bar AND pin) when judging (m/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    stand_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the stand at reset (m)
    stand_yaw_deg: float = tunable(180.0)  # uniform +/- stand yaw (free — read it from the scene)
    bar_jitter: float = tunable(0.04)  # uniform +/- xy jitter of the bar spawn (m)
    bar_yaw_deg: float = tunable(180.0)  # uniform +/- bar yaw (free)
    pin_jitter: float = tunable(0.04)  # uniform +/- xy jitter of the pin spawn (m)
    pin_yaw_deg: float = tunable(180.0)  # uniform +/- pin lying-axis yaw (free)
    decoy_jitter: float = tunable(0.04)  # uniform +/- xy jitter of the decoy spawn (m)

    # --- tunable: placement ------------------------------------------------------------------
    stand_pos: tuple = tunable((0.02, 0.14))  # bore axis, nominal
    bar_pos: tuple = tunable((-0.20, -0.10))  # bar eye centre, nominal
    pin_pos: tuple = tunable((0.16, -0.16))  # pin shaft centre, nominal
    decoy_pos: tuple = tunable((-0.22, 0.16))  # decoy shaft centre, nominal

    # --- info: structure ---------------------------------------------------------------------
    plate_out: float = info(0.140)  # stand top plate outer square (m)
    plate_t: float = info(0.012)
    plate_top: float = info(0.080)  # plate top face above the table
    hole_half: float = info(0.013)  # square hole half-width — SAME in plate and bar eye (26 mm)
    leg_w: float = info(0.016)
    eye_out: float = info(0.064)  # bar eye plate outer square
    bar_t: float = info(0.016)  # bar thickness (flat underside, coplanar eye + handle)
    handle_len: float = info(0.096)
    handle_w: float = info(0.026)
    block_len: float = info(0.034)  # raised grip block (jaw target), mid-handle so the
    block_h: float = info(0.028)  # seated bar's COM stays over the plate
    block_x: float = info(0.080)  # block centre along bar local x (from the eye centre)
    bar_mass: float = info(0.12)
    pin_shaft_r: float = info(0.009)  # 18 mm shaft in the 26 mm holes (4 mm/side clearance)
    pin_shaft_len: float = info(0.072)
    pin_head_r: float = info(0.017)  # 34 mm head > 26 mm holes: cannot pass (order + captivity)
    pin_head_h: float = info(0.010)
    pin_mass: float = info(0.08)
    decoy_shaft_r: float = info(0.015)  # 30 mm shaft > 26 mm holes: cannot enter
    decoy_head_r: float = info(0.020)
    decoy_mass: float = info(0.10)
    stand_color: tuple = info((0.45, 0.45, 0.48))
    stand_leg_color: tuple = info((0.30, 0.30, 0.33))
    bar_color: tuple = info((0.10, 0.60, 0.25))
    bar_block_color: tuple = info((0.05, 0.35, 0.15))
    pin_color: tuple = info((0.25, 0.45, 0.90))
    pin_head_color: tuple = info((0.20, 0.35, 0.75))
    decoy_color: tuple = info((0.85, 0.15, 0.12))
    decoy_head_color: tuple = info((0.65, 0.10, 0.08))
    # Explicit small offsets: the ~2 cm default would eat the 4 mm hole clearance.
    contact_offset: float = info(0.001)
    # Spawn keep-out radii (batched rejection resampling at reset): bar reach ~0.135
    # (eye centre -> handle end) + stand plate half-diagonal ~0.099 + margin.
    keepout_stand_bar: float = info(0.26)
    keepout_stand_pin: float = info(0.20)
    keepout_bar_pin: float = info(0.24)
    keepout_pin_decoy: float = info(0.13)

    # Derived (filled in __post_init__).
    plate_bot: float = field(default=None, init=False)  # plate underside height
    eye_top_seated: float = field(default=None, init=False)  # bar eye top when seated
    pin_half: float = field(default=None, init=False)  # shaft half-length
    through_z: float = field(default=None, init=False)  # tip below this = through the plate

    def __post_init__(self) -> None:
        self.plate_bot = self.plate_top - self.plate_t
        self.eye_top_seated = self.plate_top + self.bar_t
        self.pin_half = self.pin_shaft_len / 2
        self.through_z = self.plate_bot - self.through_margin

        assert 2 * self.hole_half - 2 * self.pin_shaft_r >= 0.006, (
            "pin must thread the holes with real clearance")
        assert 2 * self.decoy_shaft_r >= 2 * self.hole_half + 0.003, (
            "decoy must be physically unable to enter the holes")
        assert 2 * self.pin_head_r >= 2 * self.hole_half + 0.006, (
            "pin head must be unable to pass the holes (order enforcement + captivity)")
        assert self.pin_shaft_len >= self.bar_t + self.plate_t + self.through_margin + 0.020, (
            "seated pin must protrude well through both plates")
        assert self.eye_top_seated - self.pin_shaft_len >= 0.004, (
            "seated pin tip must hang free of the ground (the head carries the pin)")
        assert self.align_tol <= 2 * (self.hole_half - self.pin_shaft_r) + 0.002, (
            "seat tolerance must stay within the physically linkable hole offset")
        assert self.through_z > self.pin_half - 0.030, "through threshold sanity"
        assert self.block_x + self.block_len / 2 <= self.eye_out / 2 + self.handle_len, (
            "grip block must sit on the handle")


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("hasp_pin_link")
class HaspPinScene(BaseScene):
    cfg: HaspPinSceneCfg

    def __init__(self, cfg: HaspPinSceneCfg | None = None) -> None:
        super().__init__(cfg or HaspPinSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
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
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=_stand_spawner_cfg(
                    plate_out=c.plate_out, plate_t=c.plate_t, plate_top=c.plate_top,
                    hole_half=c.hole_half, leg_w=c.leg_w, color=c.stand_color,
                    leg_color=c.stand_leg_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.stand_pos[0], c.stand_pos[1], 0.0)),
            ),
            "bar": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bar",
                spawn=_bar_spawner_cfg(
                    eye_out=c.eye_out, hole_half=c.hole_half, bar_t=c.bar_t,
                    handle_len=c.handle_len, handle_w=c.handle_w, block_len=c.block_len,
                    block_h=c.block_h, block_x=c.block_x, mass=c.bar_mass,
                    color=c.bar_color, block_color=c.bar_block_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bar_pos[0], c.bar_pos[1], c.bar_t / 2 + 0.002)),
            ),
            "pin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pin",
                spawn=_pin_spawner_cfg(
                    key="pin", shaft_r=c.pin_shaft_r, shaft_len=c.pin_shaft_len,
                    head_r=c.pin_head_r, head_h=c.pin_head_h, mass=c.pin_mass,
                    color=c.pin_color, head_color=c.pin_head_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pin_pos[0], c.pin_pos[1], c.pin_head_r + 0.002),
                    rot=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0),
                ),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=_pin_spawner_cfg(
                    key="decoy", shaft_r=c.decoy_shaft_r, shaft_len=c.pin_shaft_len,
                    head_r=c.decoy_head_r, head_h=c.pin_head_h, mass=c.decoy_mass,
                    color=c.decoy_color, head_color=c.decoy_head_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.decoy_pos[0], c.decoy_pos[1], c.decoy_head_r + 0.002),
                    rot=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0),
                ),
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
        self.stand: RigidObject = env.iscene["stand"]
        self.bar: RigidObject = env.iscene["bar"]
        self.pin: RigidObject = env.iscene["pin"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.d0 = torch.full((n,), 0.30, device=dev)  # bar-spawn -> bore-axis distance
        self.approach_latch = torch.zeros(n, device=dev)
        self.seat_latch = torch.zeros(n, device=dev)
        self.pin_latch = torch.zeros(n, device=dev)

    def _sample_clear(self, m: int, nominal: tuple, jitter: float,
                      keepouts: list[tuple[torch.Tensor, float]]) -> torch.Tensor:
        """(m,2) jittered xy around `nominal`, resampled (12 tries, batched) until
        outside every (centre, radius) keep-out — nothing spawns intersecting."""
        dev = self.env.device
        base = torch.tensor(nominal, device=dev).expand(m, 2)
        xy = base + (torch.rand(m, 2, device=dev) * 2 - 1) * jitter
        for _ in range(12):
            bad = torch.zeros(m, dtype=torch.bool, device=dev)
            for ctr, rad in keepouts:
                bad |= (xy - ctr).norm(dim=-1) < rad
            if not bad.any():
                break
            k = int(bad.sum())
            xy[bad] = base[bad] + (torch.rand(k, 2, device=dev) * 2 - 1) * jitter
        return xy

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: stand with xy jitter + free yaw (the bore axis and hole
        orientation move), bar flat on the table with xy + free yaw, pin and decoy
        lying flat with xy + free yaw — all keep-out resampled; latches zeroed and the
        approach baseline `d0` captured."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def yaw_quat_state(xy: torch.Tensor, z: float, yaw: torch.Tensor,
                           lying: bool = False) -> torch.Tensor:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            half = yaw / 2
            if lying:  # q = qz(yaw) * qy(90 deg): local +z -> horizontal
                c45 = math.cos(math.pi / 4)
                st[:, 3] = torch.cos(half) * c45
                st[:, 4] = -torch.sin(half) * c45
                st[:, 5] = torch.cos(half) * c45
                st[:, 6] = torch.sin(half) * c45
            else:
                st[:, 3] = torch.cos(half)
                st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            return st

        # --- stand: xy jitter + free yaw ---
        stand_xy = torch.tensor(c.stand_pos, device=dev).expand(m, 2).clone()
        stand_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.stand_jitter
        stand_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.stand_yaw_deg)
        self.stand.write_root_state_to_sim(yaw_quat_state(stand_xy, 0.0, stand_yaw), env_ids)

        # --- bar: flat on the table, keep-out from the stand ---
        bar_xy = self._sample_clear(m, c.bar_pos, c.bar_jitter,
                                    [(stand_xy, c.keepout_stand_bar)])
        bar_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.bar_yaw_deg)
        self.bar.write_root_state_to_sim(
            yaw_quat_state(bar_xy, c.bar_t / 2 + 0.002, bar_yaw), env_ids)

        # --- pin: lying flat, keep-out from stand + bar ---
        pin_xy = self._sample_clear(m, c.pin_pos, c.pin_jitter,
                                    [(stand_xy, c.keepout_stand_pin),
                                     (bar_xy, c.keepout_bar_pin)])
        pin_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pin_yaw_deg)
        self.pin.write_root_state_to_sim(
            yaw_quat_state(pin_xy, c.pin_head_r + 0.002, pin_yaw, lying=True), env_ids)

        # --- decoy: lying flat, keep-out from everything ---
        dec_xy = self._sample_clear(m, c.decoy_pos, c.decoy_jitter,
                                    [(stand_xy, c.keepout_stand_pin),
                                     (bar_xy, c.keepout_bar_pin),
                                     (pin_xy, c.keepout_pin_decoy)])
        dec_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pin_yaw_deg)
        self.decoy.write_root_state_to_sim(
            yaw_quat_state(dec_xy, c.decoy_head_r + 0.002, dec_yaw, lying=True), env_ids)

        # --- baselines + latches ---
        self.d0[env_ids] = (bar_xy - stand_xy).norm(dim=-1).clamp(min=0.05)
        self.approach_latch[env_ids] = 0.0
        self.seat_latch[env_ids] = 0.0
        self.pin_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "bar": self.bar.data.root_state_w[env_ids].clone(),
            "pin": self.pin.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "d0": self.d0[env_ids].clone(),
            "approach_latch": self.approach_latch[env_ids].clone(),
            "seat_latch": self.seat_latch[env_ids].clone(),
            "pin_latch": self.pin_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.bar.write_root_state_to_sim(state["bar"], env_ids)
        self.pin.write_root_state_to_sim(state["pin"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.d0[env_ids] = state["d0"]
        self.approach_latch[env_ids] = state["approach_latch"]
        self.seat_latch[env_ids] = state["seat_latch"]
        self.pin_latch[env_ids] = state["pin_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A gray elevated platform stands on the floor: a "
            f"{c.plate_out * 1000:.0f} x {c.plate_out * 1000:.0f} mm top plate carried "
            f"{c.plate_top * 1000:.0f} mm up on two dark side legs, with a "
            f"{2 * c.hole_half * 1000:.0f} mm SQUARE BORE straight through the middle of the "
            f"plate — the gap under the plate is open on two sides, so the bore is a true "
            f"through-hole. A GREEN flat bar lies loose on the floor: one end is a square "
            f"eye plate with a {2 * c.hole_half * 1000:.0f} mm square hole matching the bore, "
            f"the other end is a handle with a raised dark-green grip block. A BLUE steel pin "
            f"({2 * c.pin_shaft_r * 1000:.0f} mm shaft, {c.pin_shaft_len * 1000:.0f} mm long, "
            f"with a {2 * c.pin_head_r * 1000:.0f} mm round head) lies on the floor, and so "
            f"does a RED pin whose {2 * c.decoy_shaft_r * 1000:.0f} mm shaft is TOO THICK to "
            f"fit either hole — the red pin is a decoy.\n"
            f"Goal: fasten the bar to the platform. First lay the bar flat on top of the "
            f"plate so the hole in its eye lines up over the plate's bore (within about "
            f"{c.align_tol * 1000:.0f} mm). Then take the BLUE pin, hold it vertical, and "
            f"lower it down through both aligned holes until its wide head seats on the "
            f"bar's eye and its tip hangs through into the open gap under the plate.\n"
            f"Order matters: the pin's head is wider than the holes, so a pin dropped into "
            f"the bare bore first sticks up and the bar's eye can never pass over it — seat "
            f"the bar BEFORE inserting the pin. Judged only when settled: bar seated over "
            f"the bore with the blue pin's shaft passing through BOTH holes and its head "
            f"resting on the eye. A misaligned bar, a pin resting on top, a pin in the bore "
            f"without the bar, or the red decoy anywhere count for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lay the green bar on top of the gray platform so the hole in its eye end lines "
            "up with the square bore through the plate, then lower the blue pin vertically "
            "through both holes until its head seats on the bar. Seat the bar before "
            "inserting the pin, and do not use the thick red decoy pin."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _stand_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> stand body frame (origin = bore axis at table level)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.stand.data.root_quat_w,
                                  p_w - self.stand.data.root_pos_w)

    def _bar_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> bar body frame (origin = eye centre, mid-thickness)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.bar.data.root_quat_w, p_w - self.bar.data.root_pos_w)

    def _body_up(self, body) -> torch.Tensor:
        """(N,3) the body's local +z in world frame."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)

    def _z_rel(self, body) -> torch.Tensor:
        """(N,) body origin height above the env-origin table plane."""
        return (body.data.root_pos_w - self.env_origins)[:, 2]

    # ----- predicates -------------------------------------------------------------------------
    def bar_seated(self) -> torch.Tensor:
        """(N,) bool: the bar's eye centred over the bore (stand frame, per-axis
        `align_tol`), resting flat on the plate top, upright."""
        c = self.cfg
        loc = self._stand_local(self.bar.data.root_pos_w)
        near = (loc[:, 0].abs() < c.align_tol) & (loc[:, 1].abs() < c.align_tol)
        bar_bot = self._z_rel(self.bar) - c.bar_t / 2
        on_plate = (bar_bot - c.plate_top).abs() < c.seat_z_tol
        upright = self._body_up(self.bar)[:, 2] >= math.cos(math.radians(c.bar_upright_deg))
        return near & on_plate & upright

    def _pin_ends_w(self, body) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(axis, tip, head_bottom) world tensors for a headed-pin body."""
        axis = self._body_up(body)
        pos = body.data.root_pos_w
        return axis, pos - axis * self.cfg.pin_half, pos + axis * self.cfg.pin_half

    def _linked(self, body) -> torch.Tensor:
        """(N,) bool: `body`'s shaft genuinely fastens bar to stand — near-vertical,
        tip through the plate bore (stand frame) below the underside, shaft passing
        through the bar's eye (axis point at the eye plane, BAR frame), and the head
        seated at the eye top (kills the pin-standing-in-the-gap axis-line loophole)."""
        c = self.cfg
        axis, tip_w, head_w = self._pin_ends_w(body)
        upright = axis[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(c.pin_upright_deg))
        tip_loc = self._stand_local(tip_w)
        tip_z = (tip_w - self.env_origins)[:, 2]
        in_bore = ((tip_loc[:, 0].abs() < c.tip_xy_tol) & (tip_loc[:, 1].abs() < c.tip_xy_tol)
                   & (tip_z < c.through_z) & (tip_z > 0.002))
        # shaft through the eye: axis point at the bar's mid-plane height, in bar frame
        bar_z = self._z_rel(self.bar)
        pin_z = self._z_rel(body)
        t = (bar_z - pin_z) / axis[:, 2].clamp(min=0.1)
        p_at = body.data.root_pos_w + axis * t.unsqueeze(-1)
        eye_loc = self._bar_local(p_at)
        in_eye = (eye_loc[:, 0].abs() < c.hole_half) & (eye_loc[:, 1].abs() < c.hole_half)
        # head genuinely carried by the eye top
        head_z = (head_w - self.env_origins)[:, 2]
        head_seated = ((head_z > c.eye_top_seated - c.head_seat_low)
                       & (head_z < c.eye_top_seated + c.head_seat_high))
        return upright & in_bore & in_eye & head_seated

    def pin_linked(self) -> torch.Tensor:
        """(N,) bool: the BLUE pin fastens bar to stand (the decoy is judged nowhere)."""
        return self._linked(self.pin)

    def settled(self) -> torch.Tensor:
        """(N,) bool: bar AND pin |lin vel| below `settle_lin`."""
        return ((self.bar.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin)
                & (self.pin.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin))

    # ----- graded progress --------------------------------------------------------------------
    def approach_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: bar eye progress toward the bore axis, normalized by the
        episode's own spawn distance."""
        d = (self.bar.data.root_pos_w - self.stand.data.root_pos_w)[:, :2].norm(dim=-1)
        return (1.0 - d / self.d0).clamp(0.0, 1.0)

    def pin_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: pin-through depth — tip descent from the eye top to through
        the plate underside — gated on the bar being SEATED and the tip actually inside
        the EYE hole (bar frame). Before the passage exists this is identically 0: the
        order latch. Waving the pin in the air or dropping it into the bare bore earns
        nothing."""
        c = self.cfg
        axis, tip_w, _ = self._pin_ends_w(self.pin)
        tip_loc_bar = self._bar_local(tip_w)
        tip_z = (tip_w - self.env_origins)[:, 2]
        gate = (self.bar_seated()
                & (axis[:, 2] > math.cos(math.radians(c.pin_upright_deg + 20)))
                & (tip_loc_bar[:, 0].abs() < c.hole_half)
                & (tip_loc_bar[:, 1].abs() < c.hole_half)
                & (tip_z < c.eye_top_seated + 0.002))
        depth = (c.eye_top_seated + 0.002 - tip_z) / (c.eye_top_seated - c.through_z)
        return depth.clamp(0.0, 1.0) * gate.float()

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch best bar approach, bar seating, and pin-through depth each physics
        substep, so transient progress keeps its credit."""
        self.approach_latch = torch.maximum(self.approach_latch, self.approach_frac())
        seated_now = (self.bar_seated()
                      & (self.bar.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin))
        self.seat_latch = torch.maximum(self.seat_latch, seated_now.float())
        self.pin_latch = torch.maximum(self.pin_latch, self.pin_frac())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: bar seated over the bore + blue pin linking both bodies, settled."""
        return self.bar_seated() & self.pin_linked() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 * latched bar approach + 0.25 * bar seated +
        0.45 * latched pin-through depth (order-gated), capped at 0.85; exactly 1.0 iff
        success(). Doing nothing scores ~0; the seed's strategy (pin into the fixed
        block's hole, nothing else) earns no seat and no pin credit."""
        base = (0.15 * self.approach_latch + 0.25 * self.seat_latch
                + 0.45 * self.pin_latch).clamp(0.0, 0.85)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="hasp_pin_link", robot="null"))
