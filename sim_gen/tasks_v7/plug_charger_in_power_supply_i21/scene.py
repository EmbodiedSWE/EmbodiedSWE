"""KeyholeUnplugScene — free the locked blue plug from the power strip's keyhole
socket and lay it in the tray, disturbing nothing else (sim_gen task
`plug_charger_in_power_supply_i21`).

Derived from rlbench/plug_charger_in_power_supply, but STRATEGICALLY different: the
seed is a terminal INSERTION — pick a free charger off the table, align its prongs,
and push it INTO a fixed wall socket; the episode ends the instant the charger's pose
sits in the socket, and nothing else in the scene is judged. Here the whole plan is
inverted and a mechanism is added: the judged plug STARTS captive inside the supply
(the seed's goal state is this task's RESET state), and the goal is a guarded
DISASSEMBLY + relocation under side-conditions:

  1. the blue plug is locked in the strip by a KEYHOLE LATCH — its wide foot is
     captive under the strip's lid, whose opening over the channel is a narrow slot
     that only the thin neck passes; the plug must first be SLID along the channel
     (contact-guided by the cavity walls and slot rails) to the wide square opening
     at the channel's far end,
  2. then LIFTED straight out through that opening (the only place the foot fits),
  3. then carried away and LAID FLAT inside an open tray,
  4. while the black decoy plug stays seated in its own socket and the free-standing
     strip (a DYNAMIC body resting on the floor — nothing anchors it) stays where it
     was: a straight upward yank on a still-locked plug physically hoists the whole
     strip (the foot bears on the lid underside), which the rubric rejects.

So a solver needs a different PLAN (observe the slot direction from the wide opening,
slide-unlock, extract, transport, stow, preserve the rest of the scene) and a
different code structure (an unlock-progress latch, an extraction latch, a stow
predicate, and preservation constraints on two untouched bodies) — not "align prongs
with a hole and push". Insertion appears nowhere in the goal; success() judges only
settled physical outcomes.

success() iff, settled (strip AND both plugs |v| < settle_lin):
  - the BLUE plug rests inside the tray (tray-frame |x|,|y| < in_tray_xy, origin
    height in the tray's interior band — a plug on the rim, leaning outside, or on
    the floor next to the tray all fail);
  - the BLACK plug is still seated at its own lock seat (strip frame);
  - the strip is undisturbed: position drift < strip_drift_xy, yaw drift <
    strip_drift_yaw_deg, still upright.

score() is latched every physics substep (credit never evaporates):
  0.25 * best unlock-slide progress (blue foot's travel from the lock seat toward
         the wide opening, gated on the foot actually being inside its own channel)
+ 0.30 * extracted (blue foot clear above the lid top, or the plug fully off the
         strip footprint)
+ 0.30 * stowed (blue plug settled inside the tray),
capped at 0.85; exactly 1.0 iff success(). Doing nothing scores ~0 — and because
the reset state IS the seed's goal state, the seed's whole strategy scores ~0 here.

Assets are fully procedural (no external files):
  - strip: DYNAMIC gray power strip, 220 x 110 x 30 mm, free-standing (1.5 kg). Two
    identical keyhole sockets at local x = +/-45 mm: each is an internal channel
    (30 mm wide, 62 mm long, 16 mm tall) under a 6 mm lid; the lid opening is a
    20 mm slot over most of the channel that widens into a 30 mm square opening at
    the channel's local +y end.
  - plugs: DYNAMIC, one BLUE and one BLACK, geometrically identical: 24 mm square
    x 10 mm foot (captive under the lid: 24 > 20 slot), 14 mm neck (rides the slot),
    34 mm square x 45 mm head (the grasp knob, always above the lid; 34 > 30 means
    the head can never enter any opening). Which socket holds the BLUE plug is
    randomized per episode.
  - tray: KINEMATIC open square tray, 110 mm interior, 30 mm walls.
Contact offsets are explicit and small (1 mm): the default ~2 cm offset would eat
the 3 mm working clearances in the slot, channel and opening.

Per-episode randomization (verified by readback in smoke): strip xy + free yaw (the
slide direction must be read from the scene — the wide opening marks it), tray xy +
free yaw with keep-out from the strip, and the blue/black socket assignment swaps.
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


def _spawn_strip(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC power strip at `prim_path`. Origin = base centre at floor
    level. Two keyhole sockets at local x = +/- `socket_dx`, each an internal channel
    (x half-width `cav_half_x`, y in [`cav_y0`, `cav_y1`], z in [`base_t`,
    `base_t + mid_t`]) under a lid whose opening is a slot (half-width `slot_half`,
    y in [`cav_y0`, `slot_y1`]) widening into the full-channel-width square opening
    (y in [`ap_y0`, `ap_y1`]). Everything is axis-aligned boxes."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass)

    co = cfg.contact_offset
    body_c, lid_c = cfg.body_color, cfg.lid_color
    hx, hy = cfg.strip_len / 2, cfg.strip_wid / 2
    bt, mt, lt = cfg.base_t, cfg.mid_t, cfg.lid_t
    mz = bt + mt / 2  # mid-layer centre z
    lz = bt + mt + lt / 2  # lid-layer centre z
    cx = cfg.cav_half_x
    y0, y1 = cfg.cav_y0, cfg.cav_y1
    sy1 = cfg.slot_y1
    ay1 = cfg.ap_y1
    dx = cfg.socket_dx

    # base slab (the channel floor)
    _box(stage, f"{prim_path}/base", (2 * hx, 2 * hy, bt), (0.0, 0.0, bt / 2), body_c, co)
    # mid layer: everything except the two channel interiors
    edge = hx - (dx + cx)  # solid width outboard of each channel
    mid = 2 * (dx - cx)  # solid width between the channels
    for tag, xc, w in (("mid_xn", -(dx + cx + edge / 2), edge),
                       ("mid_x0", 0.0, mid),
                       ("mid_xp", dx + cx + edge / 2, edge)):
        _box(stage, f"{prim_path}/{tag}", (w, 2 * hy, mt), (xc, 0.0, mz), body_c, co)
    for tag, sgn in (("n", -1.0), ("p", 1.0)):
        x0 = sgn * dx
        # channel end walls (lock end and beyond the opening)
        _box(stage, f"{prim_path}/mid_ylock_{tag}", (2 * cx, y0 + hy, mt),
             (x0, (y0 - hy) / 2, mz), body_c, co)
        _box(stage, f"{prim_path}/mid_yfree_{tag}", (2 * cx, hy - y1, mt),
             (x0, (y1 + hy) / 2, mz), body_c, co)
        # lid end slabs
        _box(stage, f"{prim_path}/lid_ylock_{tag}", (2 * cx, y0 + hy, lt),
             (x0, (y0 - hy) / 2, lz), lid_c, co)
        _box(stage, f"{prim_path}/lid_yfree_{tag}", (2 * cx, hy - ay1, lt),
             (x0, (ay1 + hy) / 2, lz), lid_c, co)
        # slot rails: the lid strips flanking the narrow slot (what makes the foot captive)
        rail_w = cx - cfg.slot_half
        rail_l = sy1 - y0
        for rtag, rsgn in (("a", -1.0), ("b", 1.0)):
            _box(stage, f"{prim_path}/rail_{tag}{rtag}",
                 (rail_w, rail_l, lt),
                 (x0 + rsgn * (cfg.slot_half + rail_w / 2), (y0 + sy1) / 2, lz), lid_c, co)
    # lid over the solid x-regions
    for tag, xc, w in (("lid_xn", -(dx + cx + edge / 2), edge),
                       ("lid_x0", 0.0, mid),
                       ("lid_xp", dx + cx + edge / 2, edge)):
        _box(stage, f"{prim_path}/{tag}", (w, 2 * hy, lt), (xc, 0.0, lz), lid_c, co)
    return root


def _spawn_plug(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a DYNAMIC keyhole plug at `prim_path`. Origin = FOOT CENTRE; local +z
    up: foot (captive), neck (rides the slot), head (the grasp knob)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass)

    co, color = cfg.contact_offset, cfg.color
    _box(stage, f"{prim_path}/foot", (cfg.foot_w, cfg.foot_w, cfg.foot_h),
         (0.0, 0.0, 0.0), cfg.foot_color, co)
    _box(stage, f"{prim_path}/neck", (cfg.neck_w, cfg.neck_w, cfg.neck_h),
         (0.0, 0.0, cfg.foot_h / 2 + cfg.neck_h / 2), color, co)
    _box(stage, f"{prim_path}/head", (cfg.head_w, cfg.head_w, cfg.head_h),
         (0.0, 0.0, cfg.foot_h / 2 + cfg.neck_h + cfg.head_h / 2), color, co)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC open tray at `prim_path`. Origin = base centre at floor
    level: floor slab + four walls around a square interior."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(2.0)

    co, color = cfg.contact_offset, cfg.color
    ih = cfg.inner_half
    wt, wh, ft = cfg.wall_t, cfg.wall_h, cfg.floor_t
    out = 2 * ih + 2 * wt
    _box(stage, f"{prim_path}/floor", (out, out, ft), (0.0, 0.0, ft / 2), color, co)
    for tag, cx, cy, sx, sy in (("wall_yp", 0.0, ih + wt / 2, out, wt),
                                ("wall_yn", 0.0, -(ih + wt / 2), out, wt),
                                ("wall_xp", ih + wt / 2, 0.0, wt, 2 * ih),
                                ("wall_xn", -(ih + wt / 2), 0.0, wt, 2 * ih)):
        _box(stage, f"{prim_path}/{tag}", (sx, sy, wh), (cx, cy, wh / 2), color, co)
    return root


def _strip_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "strip" not in _SPAWNER_CACHE:

        @configclass
        class KeyholeStripSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_strip)
            strip_len: float = 0.220
            strip_wid: float = 0.110
            base_t: float = 0.008
            mid_t: float = 0.016
            lid_t: float = 0.006
            socket_dx: float = 0.045
            cav_half_x: float = 0.015
            cav_y0: float = -0.030
            cav_y1: float = 0.032
            slot_half: float = 0.010
            slot_y1: float = 0.002
            ap_y1: float = 0.033
            mass: float = 1.5
            body_color: tuple = (0.52, 0.53, 0.56)
            lid_color: tuple = (0.38, 0.39, 0.42)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["strip"] = KeyholeStripSpawnerCfg

    return _SPAWNER_CACHE["strip"](
        mass_props=sim_utils.MassPropertiesCfg(mass=kw["mass"]),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        **kw,
    )


def _plug_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "plug" not in _SPAWNER_CACHE:

        @configclass
        class KeyholePlugSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plug)
            foot_w: float = 0.024
            foot_h: float = 0.010
            neck_w: float = 0.014
            neck_h: float = 0.014
            head_w: float = 0.034
            head_h: float = 0.045
            mass: float = 0.08
            color: tuple = (0.15, 0.35, 0.90)
            foot_color: tuple = (0.75, 0.72, 0.35)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["plug"] = KeyholePlugSpawnerCfg

    return _SPAWNER_CACHE["plug"](
        mass_props=sim_utils.MassPropertiesCfg(mass=kw["mass"]),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        **kw,
    )


def _tray_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tray" not in _SPAWNER_CACHE:

        @configclass
        class OpenTraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            inner_half: float = 0.055
            wall_t: float = 0.008
            wall_h: float = 0.030
            floor_t: float = 0.006
            color: tuple = (0.86, 0.83, 0.74)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["tray"] = OpenTraySpawnerCfg

    return _SPAWNER_CACHE["tray"](
        mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        **kw,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class KeyholeUnplugSceneCfg(BaseCfg):
    """Config for `KeyholeUnplugScene`. Honesty knobs asserted in `__post_init__`:
    the foot is captive under the slot (unlock is mandatory), slides and lifts with
    real clearance, the head can enter nothing (no burying loopholes), and the tray
    interior genuinely contains a lying plug within the judged band."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    in_tray_xy: float = tunable(0.050)  # blue plug origin within this of the tray axis, per axis (m)
    in_tray_z_lo: float = tunable(0.004)  # ... and origin height in the tray interior band (m)
    in_tray_z_hi: float = tunable(0.045)
    black_seat_tol: float = tunable(0.012)  # black plug within this of its lock seat, per axis (m)
    strip_drift_xy: float = tunable(0.050)  # strip position drift allowed from its reset pose (m)
    strip_drift_yaw_deg: float = tunable(20.0)  # strip yaw drift allowed (deg)
    strip_tilt_deg: float = tunable(10.0)  # strip +z within this of world-up
    settle_lin: float = tunable(0.05)  # max |lin vel| (strip AND both plugs) when judging (m/s)
    extract_z: float = tunable(0.038)  # blue foot centre above this (strip frame) = pulled out (m)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    strip_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the strip at reset (m)
    strip_yaw_deg: float = tunable(180.0)  # uniform +/- strip yaw (free — read it from the scene)
    tray_jitter: float = tunable(0.04)  # uniform +/- xy jitter of the tray at reset (m)
    tray_yaw_deg: float = tunable(180.0)  # uniform +/- tray yaw (free)
    swap_sockets: bool = tunable(True)  # randomize WHICH socket holds the blue plug

    # --- tunable: placement ------------------------------------------------------------------
    strip_pos: tuple = tunable((-0.04, 0.10))  # strip base centre, nominal
    tray_pos: tuple = tunable((0.10, -0.20))  # tray base centre, nominal

    # --- info: strip structure ---------------------------------------------------------------
    strip_len: float = info(0.220)
    strip_wid: float = info(0.110)
    base_t: float = info(0.008)  # channel floor slab
    mid_t: float = info(0.016)  # channel height
    lid_t: float = info(0.006)
    socket_dx: float = info(0.045)  # socket centres at local x = +/- this
    cav_half_x: float = info(0.015)  # channel x half-width (30 mm)
    cav_y0: float = info(-0.030)  # channel interior y range
    cav_y1: float = info(0.032)
    slot_half: float = info(0.010)  # lid slot x half-width (20 mm < 24 mm foot: captive)
    slot_y1: float = info(0.002)  # slot spans [cav_y0, slot_y1]
    ap_y1: float = info(0.033)  # wide opening spans [slot_y1, ap_y1], full channel width
    y_lock: float = info(-0.017)  # foot centre y at the lock seat
    y_free: float = info(0.017)  # foot centre y at the unlock point (under the opening)
    strip_mass: float = info(1.5)
    # --- info: plug structure ----------------------------------------------------------------
    foot_w: float = info(0.024)
    foot_h: float = info(0.010)
    neck_w: float = info(0.014)
    neck_h: float = info(0.014)
    head_w: float = info(0.034)  # > 30 mm opening: the head can enter nothing
    head_h: float = info(0.045)
    plug_mass: float = info(0.08)
    # --- info: tray structure ----------------------------------------------------------------
    tray_inner_half: float = info(0.055)
    tray_wall_t: float = info(0.008)
    tray_wall_h: float = info(0.030)
    tray_floor_t: float = info(0.006)
    # --- info: colors + misc -----------------------------------------------------------------
    strip_body_color: tuple = info((0.52, 0.53, 0.56))
    strip_lid_color: tuple = info((0.38, 0.39, 0.42))
    blue_color: tuple = info((0.15, 0.35, 0.90))
    black_color: tuple = info((0.07, 0.07, 0.09))
    foot_color: tuple = info((0.75, 0.72, 0.35))  # brass feet (visible through the openings)
    tray_color: tuple = info((0.86, 0.83, 0.74))
    contact_offset: float = info(0.001)  # explicit: default ~2 cm would eat the 3 mm clearances
    keepout_strip_tray: float = info(0.26)  # strip half-diagonal + tray half-diagonal + margin

    # Derived (filled in __post_init__).
    cav_z0: float = field(default=None, init=False)  # channel floor top
    cav_z1: float = field(default=None, init=False)  # lid underside
    lid_top: float = field(default=None, init=False)
    seat_z: float = field(default=None, init=False)  # foot centre z when seated
    plug_len: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.cav_z0 = self.base_t
        self.cav_z1 = self.base_t + self.mid_t
        self.lid_top = self.base_t + self.mid_t + self.lid_t
        self.seat_z = self.base_t + self.foot_h / 2
        self.plug_len = self.foot_h + self.neck_h + self.head_h

        assert self.foot_w >= 2 * self.slot_half + 0.003, (
            "foot must be captive under the slot rails (unlock is mandatory)")
        assert self.neck_w + 0.004 <= 2 * self.slot_half, (
            "neck must ride the slot with real clearance")
        assert self.foot_w + 0.004 <= 2 * self.cav_half_x, (
            "foot must slide the channel with real clearance")
        assert self.foot_w + 0.004 <= self.ap_y1 - self.slot_y1, (
            "foot must pass the wide opening with real clearance (y)")
        assert self.head_w >= 2 * self.cav_half_x + 0.003, (
            "head must be unable to enter any opening (no burying loopholes)")
        assert self.mid_t >= self.foot_h + 0.004, "foot needs channel headroom"
        assert self.neck_h >= self.lid_t + 0.006, (
            "neck must span the lid with margin (head rides clear above the lid)")
        assert self.y_lock - self.foot_w / 2 > self.cav_y0, "lock seat inside the channel"
        assert self.y_free - self.foot_w / 2 >= self.slot_y1 + 0.001, (
            "unlocked foot fully under the wide opening")
        assert self.y_free + self.foot_w / 2 <= self.ap_y1 + 0.001, (
            "unlocked foot fully under the wide opening (far edge)")
        assert self.y_free + self.foot_w / 2 <= self.cav_y1 + 0.001, "end wall beyond the opening"
        assert 2 * self.tray_inner_half >= self.plug_len + 0.030, (
            "a lying plug must fit the tray interior with margin")
        assert self.in_tray_xy <= self.tray_inner_half - 0.004, (
            "in-tray tolerance must stay inside the physical walls")
        assert self.in_tray_z_hi < self.tray_floor_t + self.tray_wall_h + 0.010, (
            "in-tray band must not accept a plug perched above the walls")
        assert self.strip_mass * 9.81 > 4.0, (
            "strip weight must dwarf the extraction forces (yank hoists it, but a "
            "few-newton pull cannot fling it)")
        assert self.extract_z > self.lid_top + 0.006, "extract latch strictly above the lid"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("keyhole_unplug")
class KeyholeUnplugScene(BaseScene):
    cfg: KeyholeUnplugSceneCfg

    def __init__(self, cfg: KeyholeUnplugSceneCfg | None = None) -> None:
        super().__init__(cfg or KeyholeUnplugSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        strip_kw = dict(
            strip_len=c.strip_len, strip_wid=c.strip_wid, base_t=c.base_t, mid_t=c.mid_t,
            lid_t=c.lid_t, socket_dx=c.socket_dx, cav_half_x=c.cav_half_x, cav_y0=c.cav_y0,
            cav_y1=c.cav_y1, slot_half=c.slot_half, slot_y1=c.slot_y1, ap_y1=c.ap_y1,
            mass=c.strip_mass, body_color=c.strip_body_color, lid_color=c.strip_lid_color,
            contact_offset=c.contact_offset,
        )
        plug_kw = dict(
            foot_w=c.foot_w, foot_h=c.foot_h, neck_w=c.neck_w, neck_h=c.neck_h,
            head_w=c.head_w, head_h=c.head_h, mass=c.plug_mass, foot_color=c.foot_color,
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
            "strip": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Strip",
                spawn=_strip_spawner_cfg(**strip_kw),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.strip_pos[0], c.strip_pos[1], 0.001)),
            ),
            "plug_blue": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PlugBlue",
                spawn=_plug_spawner_cfg(color=c.blue_color, **plug_kw),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.strip_pos[0] - c.socket_dx, c.strip_pos[1] + c.y_lock,
                         c.seat_z + 0.0005)),
            ),
            "plug_black": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PlugBlack",
                spawn=_plug_spawner_cfg(color=c.black_color, **plug_kw),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.strip_pos[0] + c.socket_dx, c.strip_pos[1] + c.y_lock,
                         c.seat_z + 0.0005)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=_tray_spawner_cfg(
                    inner_half=c.tray_inner_half, wall_t=c.tray_wall_t, wall_h=c.tray_wall_h,
                    floor_t=c.tray_floor_t, color=c.tray_color, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tray_pos[0], c.tray_pos[1], 0.0)),
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
        self.strip: RigidObject = env.iscene["strip"]
        self.plug_blue: RigidObject = env.iscene["plug_blue"]
        self.plug_black: RigidObject = env.iscene["plug_black"]
        self.tray: RigidObject = env.iscene["tray"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # per-episode state: socket assignment + the strip's reset pose (drift baseline)
        self.blue_x0 = torch.full((n,), -self.cfg.socket_dx, device=dev)
        self.strip0_xy = torch.zeros(n, 2, device=dev)
        self.strip0_yaw = torch.zeros(n, device=dev)
        # progress latches
        self.slide_latch = torch.zeros(n, device=dev)
        self.extract_latch = torch.zeros(n, device=dev)
        self.stow_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: strip with xy jitter + free yaw (the slide direction moves
        with it), BOTH plugs written at their lock seats inside the sockets (the
        seed's goal state is this task's start), blue/black socket assignment
        sampled, tray with xy + free yaw and keep-out from the strip; latches zeroed
        and the strip's drift baseline captured."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- strip: xy jitter + free yaw ---
        strip_xy = torch.tensor(c.strip_pos, device=dev).expand(m, 2).clone()
        strip_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.strip_jitter
        strip_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.strip_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = strip_xy
        st[:, 2] = 0.001
        st[:, 3] = torch.cos(strip_yaw / 2)
        st[:, 6] = torch.sin(strip_yaw / 2)
        st[:, 0:3] += origin
        self.strip.write_root_state_to_sim(st, env_ids)

        # --- socket assignment: which socket holds the BLUE plug ---
        if c.swap_sockets:
            sgn = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        else:
            sgn = torch.full((m,), -1.0, device=dev)
        self.blue_x0[env_ids] = sgn * c.socket_dx

        # --- both plugs at their lock seats (strip frame -> world) ---
        ca, sa = torch.cos(strip_yaw), torch.sin(strip_yaw)
        for body, x0 in ((self.plug_blue, sgn * c.socket_dx),
                         (self.plug_black, -sgn * c.socket_dx)):
            lx = x0
            ly = torch.full((m,), c.y_lock, device=dev)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = strip_xy[:, 0] + ca * lx - sa * ly
            st[:, 1] = strip_xy[:, 1] + sa * lx + ca * ly
            st[:, 2] = c.seat_z + 0.0015
            st[:, 3] = torch.cos(strip_yaw / 2)
            st[:, 6] = torch.sin(strip_yaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- tray: xy jitter + free yaw, keep-out from the strip ---
        base = torch.tensor(c.tray_pos, device=dev).expand(m, 2)
        tray_xy = base + (torch.rand(m, 2, device=dev) * 2 - 1) * c.tray_jitter
        for _ in range(12):
            bad = (tray_xy - strip_xy).norm(dim=-1) < c.keepout_strip_tray
            if not bad.any():
                break
            k = int(bad.sum())
            tray_xy[bad] = base[bad] + (torch.rand(k, 2, device=dev) * 2 - 1) * c.tray_jitter
        tray_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.tray_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = tray_xy
        st[:, 3] = torch.cos(tray_yaw / 2)
        st[:, 6] = torch.sin(tray_yaw / 2)
        st[:, 0:3] += origin
        self.tray.write_root_state_to_sim(st, env_ids)

        # --- baselines + latches ---
        self.strip0_xy[env_ids] = strip_xy
        self.strip0_yaw[env_ids] = strip_yaw
        self.slide_latch[env_ids] = 0.0
        self.extract_latch[env_ids] = 0.0
        self.stow_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "strip": self.strip.data.root_state_w[env_ids].clone(),
            "plug_blue": self.plug_blue.data.root_state_w[env_ids].clone(),
            "plug_black": self.plug_black.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "blue_x0": self.blue_x0[env_ids].clone(),
            "strip0_xy": self.strip0_xy[env_ids].clone(),
            "strip0_yaw": self.strip0_yaw[env_ids].clone(),
            "slide_latch": self.slide_latch[env_ids].clone(),
            "extract_latch": self.extract_latch[env_ids].clone(),
            "stow_latch": self.stow_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.strip.write_root_state_to_sim(state["strip"], env_ids)
        self.plug_blue.write_root_state_to_sim(state["plug_blue"], env_ids)
        self.plug_black.write_root_state_to_sim(state["plug_black"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.blue_x0[env_ids] = state["blue_x0"]
        self.strip0_xy[env_ids] = state["strip0_xy"]
        self.strip0_yaw[env_ids] = state["strip0_yaw"]
        self.slide_latch[env_ids] = state["slide_latch"]
        self.extract_latch[env_ids] = state["extract_latch"]
        self.stow_latch[env_ids] = state["stow_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A gray power strip ({c.strip_len * 1000:.0f} x {c.strip_wid * 1000:.0f} x "
            f"{c.lid_top * 1000:.0f} mm) rests free-standing on the floor — nothing anchors "
            f"it. Its darker top plate has two identical KEYHOLE sockets: each is a narrow "
            f"{2 * c.slot_half * 1000:.0f} mm slot that widens into one "
            f"{(c.ap_y1 - c.slot_y1) * 1000:.0f} mm SQUARE OPENING at one end (both keyholes "
            f"run parallel to the strip's short side, wide openings on the same side — the "
            f"wide opening tells you the slide direction). In each socket a plug is locked "
            f"at the slot's NARROW end: one with a BLUE head, one with a BLACK head "
            f"(which socket holds the blue plug varies). Each plug is a "
            f"{c.head_w * 1000:.0f} mm square knob riding above the plate on a thin neck; "
            f"under the plate its hidden {c.foot_w * 1000:.0f} mm foot is WIDER than the "
            f"slot, so a plug at the narrow end cannot be pulled straight up — yanking it "
            f"only hoists the whole strip. An open square tray "
            f"({2 * c.tray_inner_half * 1000:.0f} mm interior, "
            f"{c.tray_wall_h * 1000:.0f} mm walls, light beige) sits on the floor nearby.\n"
            f"Goal: take the BLUE plug out of the strip and leave it lying inside the tray. "
            f"To free it, slide it along its slot to the wide square opening (about "
            f"{(c.y_free - c.y_lock) * 1000:.0f} mm of travel), then lift it straight up "
            f"through the opening, carry it to the tray, and set it down inside; any "
            f"resting pose fully inside the tray counts.\n"
            f"Constraints, judged only when everything has settled: the BLACK plug must "
            f"still be locked at its own narrow end, and the strip must remain upright "
            f"within {c.strip_drift_xy * 1000:.0f} mm and "
            f"{c.strip_drift_yaw_deg:.0f} deg of where it started. A blue plug dropped on "
            f"the floor, left on the strip, balanced on the tray rim, or still in the "
            f"socket counts for nothing; so does a stowed plug with the black plug "
            f"unseated or the strip dragged or tipped over."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the blue plug along its slot to the wide opening in the power strip, "
            "lift it out, and lay it inside the tray. Leave the black plug locked in "
            "place and do not drag or tip the strip."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _strip_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> strip body frame (origin = base centre, floor level)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.strip.data.root_quat_w,
                                  p_w - self.strip.data.root_pos_w)

    def _tray_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> tray body frame (origin = base centre, floor level)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.tray.data.root_quat_w,
                                  p_w - self.tray.data.root_pos_w)

    def _yaw(self, body) -> torch.Tensor:
        """(N,) yaw of a (near-)pure-yaw body."""
        q = body.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    def _up_z(self, body) -> torch.Tensor:
        """(N,) world-z component of the body's local +z."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)[:, 2]

    # ----- predicates -------------------------------------------------------------------------
    def blue_in_tray(self) -> torch.Tensor:
        """(N,) bool: the BLUE plug's origin inside the tray interior (tray frame,
        per-axis `in_tray_xy`, height in the interior band — rim-perchers and
        floor-next-to-tray both fail)."""
        c = self.cfg
        loc = self._tray_local(self.plug_blue.data.root_pos_w)
        return ((loc[:, 0].abs() < c.in_tray_xy) & (loc[:, 1].abs() < c.in_tray_xy)
                & (loc[:, 2] > c.in_tray_z_lo) & (loc[:, 2] < c.in_tray_z_hi))

    def black_seated(self) -> torch.Tensor:
        """(N,) bool: the BLACK plug still at ITS lock seat (strip frame): foot at
        channel-floor height, at the lock-end of its own channel."""
        c = self.cfg
        loc = self._strip_local(self.plug_black.data.root_pos_w)
        x0 = -self.blue_x0  # the black plug's socket
        return ((loc[:, 0] - x0).abs() < c.black_seat_tol) \
            & ((loc[:, 1] - c.y_lock).abs() < c.black_seat_tol) \
            & (loc[:, 2] > c.cav_z0 - 0.004) & (loc[:, 2] < c.cav_z0 + c.foot_h)

    def strip_ok(self) -> torch.Tensor:
        """(N,) bool: the strip undisturbed — position within `strip_drift_xy` and
        yaw within `strip_drift_yaw_deg` of its reset pose, still upright."""
        c = self.cfg
        xy = (self.strip.data.root_pos_w - self.env_origins)[:, :2]
        near = (xy - self.strip0_xy).norm(dim=-1) < c.strip_drift_xy
        dyaw = self._yaw(self.strip) - self.strip0_yaw
        dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw)).abs()
        level = self._up_z(self.strip) >= math.cos(math.radians(c.strip_tilt_deg))
        return near & (dyaw < math.radians(c.strip_drift_yaw_deg)) & level

    def settled(self) -> torch.Tensor:
        """(N,) bool: strip AND both plugs |lin vel| below `settle_lin`."""
        c = self.cfg
        return ((self.strip.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.plug_blue.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.plug_black.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin))

    # ----- graded progress --------------------------------------------------------------------
    def slide_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: the blue foot's unlock travel from the lock seat toward the
        wide opening, gated on the foot actually being INSIDE its own channel (strip
        frame). Zero at reset; a plug waved around outside earns nothing here."""
        c = self.cfg
        loc = self._strip_local(self.plug_blue.data.root_pos_w)
        in_channel = ((loc[:, 0] - self.blue_x0).abs() < 0.012) \
            & (loc[:, 2] > c.cav_z0 - 0.004) & (loc[:, 2] < c.cav_z1 - 0.002) \
            & (loc[:, 1] > c.cav_y0) & (loc[:, 1] < c.cav_y1 + 0.004)
        frac = (loc[:, 1] - c.y_lock) / (c.y_free - c.y_lock)
        return frac.clamp(0.0, 1.0) * in_channel.float()

    def blue_extracted(self) -> torch.Tensor:
        """(N,) bool: the blue plug clear of the strip — foot above the lid top
        (strip frame, the moment of lift-out) OR entirely off the strip footprint."""
        c = self.cfg
        loc = self._strip_local(self.plug_blue.data.root_pos_w)
        above = loc[:, 2] > c.extract_z
        off = (loc[:, 0].abs() > c.strip_len / 2 + 0.015) \
            | (loc[:, 1].abs() > c.strip_wid / 2 + 0.015)
        return above | off

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch unlock-slide progress, extraction, and settled stowage each physics
        substep, so transient progress keeps its credit."""
        self.slide_latch = torch.maximum(self.slide_latch, self.slide_frac())
        self.extract_latch = torch.maximum(self.extract_latch, self.blue_extracted().float())
        stowed_now = self.blue_in_tray() \
            & (self.plug_blue.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin)
        self.stow_latch = torch.maximum(self.stow_latch, stowed_now.float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: blue plug settled inside the tray + black plug still at its
        lock seat + strip undisturbed, everything settled."""
        return self.blue_in_tray() & self.black_seated() & self.strip_ok() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25 * latched unlock slide + 0.30 * extracted +
        0.30 * stowed, capped at 0.85; exactly 1.0 iff success(). Doing nothing —
        which leaves the scene exactly in the SEED task's goal state — scores ~0."""
        base = (0.25 * self.slide_latch + 0.30 * self.extract_latch
                + 0.30 * self.stow_latch).clamp(0.0, 0.85)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="keyhole_unplug", robot="null"))
