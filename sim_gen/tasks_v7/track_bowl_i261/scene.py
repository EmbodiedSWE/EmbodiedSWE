"""CoveredDishScene — seat the bowl in the pedestal socket, drop the egg inside, then
cap it with the ONE lid whose skirt fits the mouth (sim_gen task `track_bowl_i261`).

Derived from pick_place/track_bowl, but STRATEGICALLY different: the seed's plan is
prescribed TRANSPORT of the bowl itself — grasp the bowl, lift it, and carry it through
free space along a dense waypoint trajectory (reward = per-step position+rotation
tracking of the current waypoint; the terminal state is just wherever the path ends).
Here NOTHING is prescribed and nothing is tracked: the goal is a settled three-piece
CONCENTRIC ASSEMBLY, and every stage is a contact-mated insertion with a real
tolerance, judged in body frames:

  SEAT — the bowl's bottom boss must drop into the octagonal socket on top of a
         pedestal stand (radial slack: socket inradius - boss radius);
  FILL — the egg must come to rest INSIDE the seated bowl (real containment on the
         bowl floor, judged in the bowl's body frame);
  CAP  — the LARGE lid must be seated on the bowl: its centering skirt drops into the
         bowl mouth and its cover disk lands on the rim ring, concentric and level.

A wrong-object trap makes identification part of the task: a DECOY lid of the same
color but visibly smaller diameter cannot cover the bowl — dropped on the mouth it
falls INSIDE, and any decoy resting in the bowl pokes above the rim plane by
construction, so the real lid then rides high on it and can never seat until the
decoy is removed (its knob is graspable — recoverable, but wasted work).

A solver therefore needs a different PLAN (multi-object assembly: seat a base part in
a fixture, load a payload into it, identify the right cover and mate it) and a
different code structure (body-frame mating predicates — boss-in-socket, egg-in-bowl,
skirt-in-mouth — instead of a waypoint follower). No execution order is imposed
beyond what physics forces (the egg cannot enter a capped bowl).

success() iff, settled: bowl seated in the socket (stand frame: axis offset, height
band, upright) AND egg inside the bowl (bowl frame: axis offset, height band) AND the
large lid seated (bowl frame: concentric, disk resting on the rim height band, level).
score() latches progress every physics substep: 0.20 * bowl ever seated + 0.25 * egg
ever inside the seated bowl + 0.25 * full assembly ever mated (capped at 0.70);
exactly 1.0 iff success(). Doing nothing scores ~0.

Assets are fully procedural (no external files):
  - stand: ONE kinematic pedestal — base plate, column (its flat top is the socket
    floor, 120 mm up) and an octagonal socket collar (inner inradius 45 mm, 22 mm
    tall) — the fixture that both locates the bowl and braces it during capping;
  - bowl: dynamic — octagonal wall (inner inradius 52 mm, 55 mm tall, 6 mm walls —
    a parallel-jaw rim grasp), interior floor disk, and a 76 mm bottom boss that
    mates with the socket (7 mm radial slack, boss taller than the collar so the
    floor disk never rests on the collar rim);
  - lid (the real one): dynamic — 132 mm cover disk (always bridges the 104 mm
    mouth), octagonal centering skirt (outer corners 86.6 mm across: 8.7 mm radial
    slack in the mouth), 22 mm knob + 36 mm cap flange on top;
  - decoy lid: same color, visibly smaller — 80 mm disk falls THROUGH the mouth;
    its knob is tall enough that any rest inside the bowl stands proud of the rim
    plane (blocks real capping by construction);
  - egg: a 34 mm orange sphere (parallel-jaw graspable), the payload to be covered.
Contact offsets are explicit and small (2 mm): the default would eat the 7-9 mm
mating slacks. Per-episode randomization (verified by readback in smoke): stand xy +
free yaw, xy jitter + free yaw for bowl / egg / both lids, and a coin flip for which
scatter slot holds the REAL lid vs the DECOY. Heavy imports (isaaclab, pxr) are
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


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float,
         yaw_deg: float = 0.0) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw_deg:
        sxf.AddRotateZOp().Set(float(yaw_deg))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _cyl(stage, path: str, radius: float, height: float, center, color,
         contact_offset: float) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    seg.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _octagon_ring(stage, prim_path: str, name: str, inner_r: float, wall_t: float,
                  height: float, z0: float, color, contact_offset: float) -> None:
    """8 yawed wall boxes forming an octagonal ring: inner flats at `inner_r`, from
    local z0 to z0 + height."""
    w_flat = 2.0 * (inner_r + wall_t) * math.tan(math.radians(22.5)) + 0.002
    for k in range(8):
        ang = k * 45.0
        rad = math.radians(ang)
        cx = (inner_r + wall_t / 2) * math.cos(rad)
        cy = (inner_r + wall_t / 2) * math.sin(rad)
        _box(stage, f"{prim_path}/{name}_{k}", (wall_t, w_flat, height),
             (cx, cy, z0 + height / 2), color, contact_offset, yaw_deg=ang)


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float,
                kinematic: bool = False):
    """Author an Xform root as a rigid body with explicit mass + physics armor."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    if not kinematic:
        px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        px.CreateLinearDampingAttr(0.10)
        px.CreateAngularDampingAttr(1.0)
        px.CreateMaxDepenetrationVelocityAttr(0.5)
        px.CreateSolverPositionIterationCountAttr(16)
        px.CreateSolverVelocityIterationCountAttr(4)
        px.CreateSleepThresholdAttr(0.0)
        px.CreateStabilizationThresholdAttr(0.0)
    return root


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC pedestal stand. Local origin = centre of the SOCKET FLOOR
    (the column's flat top): base plate and column below z = 0, socket collar above."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _rigid_root(stage, prim_path, translation, orientation, 25.0, kinematic=True)
    co = cfg.contact_offset
    _cyl(stage, f"{prim_path}/base", cfg.base_r, cfg.base_t,
         (0.0, 0.0, -cfg.stand_h + cfg.base_t / 2), cfg.base_color, co)
    _cyl(stage, f"{prim_path}/column", cfg.col_r, cfg.stand_h,
         (0.0, 0.0, -cfg.stand_h / 2), cfg.col_color, co)
    _octagon_ring(stage, prim_path, "collar", cfg.socket_r, cfg.collar_t,
                  cfg.collar_h, 0.0, cfg.collar_color, co)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC bowl. Local origin = centre of the INTERIOR FLOOR TOP
    (z = 0 local is the surface the egg rests on; CoM sits there — low and stable).
    Below: floor disk and the socket boss; above: the octagonal wall."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    co = cfg.contact_offset
    _cyl(stage, f"{prim_path}/floor", cfg.mouth_r + cfg.wall_t, cfg.floor_t,
         (0.0, 0.0, -cfg.floor_t / 2), cfg.color, co)
    _cyl(stage, f"{prim_path}/boss", cfg.boss_r, cfg.boss_h,
         (0.0, 0.0, -cfg.floor_t - cfg.boss_h / 2), cfg.color, co)
    _octagon_ring(stage, prim_path, "wall", cfg.mouth_r, cfg.wall_t,
                  cfg.wall_h, 0.0, cfg.color, co)
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a DYNAMIC lid (real or decoy — same recipe, different dimensions).
    Local origin = centre of the COVER DISK's BOTTOM face (z = 0 local rests on the
    bowl rim when seated; CoM there — low). Below: the octagonal centering skirt;
    above: disk, knob, cap flange."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    co = cfg.contact_offset
    _octagon_ring(stage, prim_path, "skirt", cfg.skirt_r - cfg.skirt_t, cfg.skirt_t,
                  cfg.skirt_h, -cfg.skirt_h, cfg.color, co)
    _cyl(stage, f"{prim_path}/disk", cfg.disk_r, cfg.disk_t,
         (0.0, 0.0, cfg.disk_t / 2), cfg.color, co)
    _cyl(stage, f"{prim_path}/knob", cfg.knob_r, cfg.knob_h,
         (0.0, 0.0, cfg.disk_t + cfg.knob_h / 2), cfg.knob_color, co)
    _cyl(stage, f"{prim_path}/cap", cfg.cap_r, cfg.cap_t,
         (0.0, 0.0, cfg.disk_t + cfg.knob_h + cfg.cap_t / 2), cfg.knob_color, co)
    return root


def _make_spawner(key: str, func: Callable, defaults: dict[str, Any],
                  values: dict[str, Any], mass: float, kinematic: bool) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if key not in _SPAWNER_CACHE:
        ns = {"__annotations__": {k: type(v).__name__ for k, v in defaults.items()}}
        ns["__annotations__"]["func"] = "Callable"
        ns["func"] = clone(func)
        ns.update(defaults)
        _SPAWNER_CACHE[key] = configclass(type(f"Spawner_{key}",
                                               (RigidObjectSpawnerCfg,), ns))
    rp = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True) if kinematic \
        else sim_utils.RigidBodyPropertiesCfg()
    return _SPAWNER_CACHE[key](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass), rigid_props=rp, **values)


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CoveredDishCfg(BaseCfg):
    """Config for `CoveredDishScene`. Honesty knobs asserted in `__post_init__`:
    every mating slack is generous enough for closed-loop arm precision, every
    tolerance band covers ALL physically-mated rests and rejects the nearest wrong
    rest with real margin, the decoy provably falls through the mouth AND provably
    blocks the real lid from seating, and the seated lid can never rest on the egg."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    seat_xy_tol: float = tunable(0.010)  # bowl axis within this of the socket axis (m)
    seat_z_tol: float = tunable(0.005)  # bowl origin height band around the seated height
    egg_xy_tol: float = tunable(0.042)  # egg centre within this of the bowl axis (bowl frame)
    egg_z_lo: float = tunable(0.004)  # egg centre above the bowl floor by at least this
    egg_z_hi: float = tunable(0.050)  # ... and below this (inside, not perched on the rim)
    lid_xy_tol: float = tunable(0.012)  # lid axis within this of the bowl axis (bowl frame)
    lid_z_tol: float = tunable(0.004)  # lid disk-bottom height band around the rim height
    upright_max_deg: float = tunable(10.0)  # bowl axis within this of world-up when seated
    lid_tilt_max_deg: float = tunable(8.0)  # lid axis within this of the BOWL axis when seated
    settle_lin: float = tunable(0.08)  # max |lin vel| of all dynamics when judging (m/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    stand_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the stand (m)
    stand_yaw_deg: float = tunable(180.0)  # uniform +/- stand yaw (socket corners rotate)
    scatter_jitter: float = tunable(0.04)  # uniform +/- xy jitter of each movable slot (m)
    slot_flip: bool = tunable(True)  # coin flip: which slot holds the REAL lid vs the DECOY

    # --- tunable: placement (stand-relative xy, WORLD axes) ----------------------------------
    stand_pos: tuple = tunable((0.50, 0.0))  # stand axis, world xy nominal
    bowl_slot: tuple = tunable((-0.28, 0.20))
    egg_slot: tuple = tunable((-0.30, -0.08))
    lid_slot_a: tuple = tunable((-0.10, -0.30))
    lid_slot_b: tuple = tunable((-0.05, 0.33))

    # --- info: stand (kinematic fixture) -----------------------------------------------------
    stand_h: float = info(0.120)  # socket floor (column top) above the ground
    base_r: float = info(0.105)
    base_t: float = info(0.016)
    col_r: float = info(0.070)
    socket_r: float = info(0.045)  # socket collar INNER inradius
    collar_t: float = info(0.008)
    collar_h: float = info(0.022)  # collar rim above the socket floor
    # --- info: bowl --------------------------------------------------------------------------
    mouth_r: float = info(0.052)  # bowl wall INNER inradius (mouth flats)
    wall_t: float = info(0.006)  # a parallel-jaw grasps this rim wall
    wall_h: float = info(0.055)  # interior depth (floor top -> rim plane)
    floor_t: float = info(0.008)
    boss_r: float = info(0.038)  # bottom boss that mates with the socket
    boss_h: float = info(0.028)  # > collar_h: the floor disk never rests on the collar
    bowl_mass: float = info(0.30)
    # --- info: real lid ----------------------------------------------------------------------
    lid_disk_r: float = info(0.066)  # cover disk: ALWAYS bridges the mouth
    lid_disk_t: float = info(0.008)
    lid_skirt_r: float = info(0.040)  # centering skirt OUTER inradius
    lid_skirt_t: float = info(0.005)
    lid_skirt_h: float = info(0.016)
    lid_mass: float = info(0.30)
    # --- info: decoy lid (same color, visibly smaller) ---------------------------------------
    dec_disk_r: float = info(0.040)  # 80 mm disk: falls THROUGH the 104 mm mouth
    dec_disk_t: float = info(0.008)
    dec_skirt_r: float = info(0.026)
    dec_skirt_t: float = info(0.005)
    dec_skirt_h: float = info(0.012)
    dec_knob_h: float = info(0.038)  # tall: any in-bowl rest pokes above the rim plane
    dec_mass: float = info(0.20)
    # --- info: shared knob + egg -------------------------------------------------------------
    knob_r: float = info(0.011)  # 22 mm knob shaft: a comfortable parallel-jaw grasp
    knob_h: float = info(0.030)
    cap_r: float = info(0.018)  # 36 mm cap flange above the shaft (hook-proof grasp)
    cap_t: float = info(0.008)
    egg_r: float = info(0.017)  # 34 mm sphere: parallel-jaw graspable payload
    egg_mass: float = info(0.04)
    jaw_span: float = info(0.080)  # the Franka jaw everything must fit
    # --- info: colors ------------------------------------------------------------------------
    base_color: tuple = info((0.35, 0.36, 0.40))
    col_color: tuple = info((0.48, 0.50, 0.55))
    collar_color: tuple = info((0.28, 0.30, 0.34))
    bowl_color: tuple = info((0.92, 0.91, 0.86))
    lid_color: tuple = info((0.72, 0.15, 0.14))
    knob_color: tuple = info((0.16, 0.16, 0.18))
    egg_color: tuple = info((0.95, 0.55, 0.10))
    # Explicit small offsets: the default ~2 cm would eat the 7-9 mm mating slacks.
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    seat_z: float = field(default=None, init=False)  # seated bowl origin, stand-local z
    lid_z: float = field(default=None, init=False)  # seated lid origin, bowl-local z
    bowl_org_h: float = field(default=None, init=False)  # bowl origin above its boss bottom
    mouth_corner_r: float = field(default=None, init=False)  # mouth circumradius (corners)
    skirt_corner_r: float = field(default=None, init=False)  # skirt circumradius (corners)

    def __post_init__(self) -> None:
        self.bowl_org_h = self.floor_t + self.boss_h  # origin height above boss bottom
        self.seat_z = self.bowl_org_h  # boss bottom on the socket floor
        self.lid_z = self.wall_h  # disk bottom on the rim plane
        self.mouth_corner_r = self.mouth_r / math.cos(math.radians(22.5))
        self.skirt_corner_r = self.lid_skirt_r / math.cos(math.radians(22.5))

        # SEAT: boss mates with the socket with generous slack, and every physically
        # seated bowl is inside the tolerance; a bowl perched on the collar rim or on
        # a fouled socket rides high out of the z band.
        seat_slack = self.socket_r - self.boss_r
        assert seat_slack >= 0.005, "boss-in-socket slack must be arm-friendly"
        assert self.seat_xy_tol >= seat_slack + 0.002, "any seated bowl must count"
        assert self.boss_h >= self.collar_h + 0.004, (
            "boss must out-reach the collar: floor disk never rests on the collar rim")
        assert self.collar_h >= self.seat_z_tol + 0.010, (
            "a bowl perched on the collar rim must ride out of the seat z band")
        # FILL: the egg fits the jaw and the mouth, any in-bowl rest counts (corner
        # reach), and out-of-bowl rests are far outside the tolerance.
        assert 2 * self.egg_r <= self.jaw_span - 0.010, "egg must fit the jaw"
        assert self.egg_xy_tol >= self.mouth_corner_r - self.egg_r + 0.002, (
            "an egg resting in the mouth corner must count")
        assert self.mouth_r + self.wall_t + self.egg_r >= self.egg_xy_tol + 0.020, (
            "an egg leaning outside the wall must be rejected with margin")
        assert self.egg_z_hi >= 2 * self.egg_r + 0.010, "an egg on the floor must count"
        # CAP: the skirt drops into the mouth with generous slack at ANY yaw (corner
        # reach vs mouth flats), every skirt-in-mouth rest is inside the concentric
        # tolerance, and the disk always bridges the mouth.
        cap_slack = self.mouth_r - self.skirt_corner_r
        assert cap_slack >= 0.006, "skirt-in-mouth slack must be arm-friendly"
        assert self.lid_xy_tol >= cap_slack + 0.002, "any capped lid must count"
        assert self.lid_disk_r >= self.mouth_corner_r + 0.005, (
            "the real lid must always bridge the mouth (it can never fall in)")
        # The seated lid rests on the RIM, never on the egg: egg top clears the skirt.
        assert 2 * self.egg_r <= self.wall_h - self.lid_skirt_h - 0.004, (
            "the seated lid's skirt must clear the egg everywhere")
        # DECOY: it falls THROUGH the mouth (wrong-object attempt visibly fails), and
        # any decoy resting inside pokes above the rim plane far enough that the real
        # lid rides out of its z band (capping is blocked until the decoy is removed).
        assert self.dec_disk_r <= self.mouth_r - 0.008, "decoy must fall through the mouth"
        dec_flat_h = (self.dec_skirt_h + self.dec_disk_t + self.dec_knob_h + self.cap_t)
        assert dec_flat_h >= self.wall_h + self.lid_z_tol + 0.004, (
            "a decoy flat in the bowl must block the real lid out of its z band")
        assert 2 * self.dec_disk_r >= self.wall_h, (
            "a decoy on its side inside the bowl also stands proud of the rim")
        # Grasp affordances fit the jaw.
        assert 2 * self.knob_r <= self.jaw_span - 0.030, "knob must fit the jaw"
        assert self.wall_t <= self.jaw_span - 0.030, "bowl rim wall must fit the jaw"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("covered_dish")
class CoveredDishScene(BaseScene):
    cfg: CoveredDishCfg

    def __init__(self, cfg: CoveredDishCfg | None = None) -> None:
        super().__init__(cfg or CoveredDishCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sx, sy = c.stand_pos

        stand_spawn = _make_spawner(
            "stand", _spawn_stand,
            dict(stand_h=0.120, base_r=0.105, base_t=0.016, col_r=0.070, socket_r=0.045,
                 collar_t=0.008, collar_h=0.022, base_color=(0.35, 0.36, 0.40),
                 col_color=(0.48, 0.50, 0.55), collar_color=(0.28, 0.30, 0.34),
                 contact_offset=0.002),
            dict(stand_h=c.stand_h, base_r=c.base_r, base_t=c.base_t, col_r=c.col_r,
                 socket_r=c.socket_r, collar_t=c.collar_t, collar_h=c.collar_h,
                 base_color=c.base_color, col_color=c.col_color,
                 collar_color=c.collar_color, contact_offset=c.contact_offset),
            mass=25.0, kinematic=True)
        bowl_spawn = _make_spawner(
            "bowl", _spawn_bowl,
            dict(mouth_r=0.052, wall_t=0.006, wall_h=0.055, floor_t=0.008, boss_r=0.038,
                 boss_h=0.028, mass=0.30, color=(0.92, 0.91, 0.86), contact_offset=0.002),
            dict(mouth_r=c.mouth_r, wall_t=c.wall_t, wall_h=c.wall_h, floor_t=c.floor_t,
                 boss_r=c.boss_r, boss_h=c.boss_h, mass=c.bowl_mass, color=c.bowl_color,
                 contact_offset=c.contact_offset),
            mass=c.bowl_mass, kinematic=False)

        def lid_spawn(key: str, disk_r: float, disk_t: float, skirt_r: float,
                      skirt_t: float, skirt_h: float, knob_h: float, mass: float) -> Any:
            return _make_spawner(
                key, _spawn_lid,
                dict(disk_r=0.066, disk_t=0.008, skirt_r=0.040, skirt_t=0.005,
                     skirt_h=0.016, knob_r=0.011, knob_h=0.030, cap_r=0.018,
                     cap_t=0.008, mass=0.30, color=(0.72, 0.15, 0.14),
                     knob_color=(0.16, 0.16, 0.18), contact_offset=0.002),
                dict(disk_r=disk_r, disk_t=disk_t, skirt_r=skirt_r, skirt_t=skirt_t,
                     skirt_h=skirt_h, knob_r=c.knob_r, knob_h=knob_h, cap_r=c.cap_r,
                     cap_t=c.cap_t, mass=mass, color=c.lid_color,
                     knob_color=c.knob_color, contact_offset=c.contact_offset),
                mass=mass, kinematic=False)

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
                spawn=stand_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, c.stand_h)),
            ),
            "bowl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl",
                spawn=bowl_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx + c.bowl_slot[0], sy + c.bowl_slot[1], c.bowl_org_h + 0.002)),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=lid_spawn("lid", c.lid_disk_r, c.lid_disk_t, c.lid_skirt_r,
                                c.lid_skirt_t, c.lid_skirt_h, c.knob_h, c.lid_mass),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx + c.lid_slot_a[0], sy + c.lid_slot_a[1],
                         c.lid_skirt_h + 0.002)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=lid_spawn("decoy", c.dec_disk_r, c.dec_disk_t, c.dec_skirt_r,
                                c.dec_skirt_t, c.dec_skirt_h, c.dec_knob_h, c.dec_mass),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx + c.lid_slot_b[0], sy + c.lid_slot_b[1],
                         c.dec_skirt_h + 0.002)),
            ),
            "egg": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Egg",
                spawn=sim_utils.SphereCfg(
                    radius=c.egg_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=0.05, angular_damping=0.30,
                        max_depenetration_velocity=0.5,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        sleep_threshold=0.0, stabilization_threshold=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.egg_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.5, dynamic_friction=0.4, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.egg_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx + c.egg_slot[0], sy + c.egg_slot[1], c.egg_r + 0.002)),
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
        self.bowl: RigidObject = env.iscene["bowl"]
        self.lid: RigidObject = env.iscene["lid"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.egg: RigidObject = env.iscene["egg"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.lid_slot = torch.ones(n, device=dev)  # +1: real lid in slot A; -1: slot B
        self.seat_latch = torch.zeros(n, device=dev)
        self.egg_latch = torch.zeros(n, device=dev)
        self.asm_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: stand with xy jitter + free yaw; bowl / egg / lids scattered
        around their stand-relative slots (world axes) with xy jitter + free yaw; a
        coin flip decides which slot holds the REAL lid vs the DECOY; latches zeroed.
        Slot layout guarantees separation by construction (worst-case jitter can
        never overlap two bodies or foul the stand), so no rejection loop is needed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- stand: xy jitter + free yaw ---
        stand_xy = torch.tensor(c.stand_pos, device=dev).expand(m, 2).clone()
        stand_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.stand_jitter
        syaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.stand_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = stand_xy
        st[:, 2] = c.stand_h
        st[:, 3], st[:, 6] = torch.cos(syaw / 2), torch.sin(syaw / 2)
        st[:, 0:3] += origin
        self.stand.write_root_state_to_sim(st, env_ids)

        def write(body, slot: tuple, z: float, yaw: torch.Tensor | None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = stand_xy + torch.tensor(slot, device=dev)
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.scatter_jitter
            st[:, 2] = z
            if yaw is not None:
                st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
            else:
                st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        def rand_yaw() -> torch.Tensor:
            return (torch.rand(m, device=dev) * 2 - 1) * math.pi

        # --- coin flip: which slot holds the REAL lid ---
        if c.slot_flip:
            flip = torch.where(torch.rand(m, device=dev) < 0.5,
                               -torch.ones(m, device=dev), torch.ones(m, device=dev))
        else:
            flip = torch.ones(m, device=dev)
        self.lid_slot[env_ids] = flip
        slot_a = torch.tensor(c.lid_slot_a, device=dev)
        slot_b = torch.tensor(c.lid_slot_b, device=dev)
        pick = (flip > 0).unsqueeze(1)
        lid_xy = torch.where(pick, slot_a, slot_b)
        dec_xy = torch.where(pick, slot_b, slot_a)

        write(self.bowl, c.bowl_slot, c.bowl_org_h + 0.002, rand_yaw())
        write(self.egg, c.egg_slot, c.egg_r + 0.002, None)
        for body, sl, z in ((self.lid, lid_xy, c.lid_skirt_h + 0.002),
                            (self.decoy, dec_xy, c.dec_skirt_h + 0.002)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = stand_xy + sl
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.scatter_jitter
            st[:, 2] = z
            yw = rand_yaw()
            st[:, 3], st[:, 6] = torch.cos(yw / 2), torch.sin(yw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        self.seat_latch[env_ids] = 0.0
        self.egg_latch[env_ids] = 0.0
        self.asm_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "bowl": self.bowl.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "egg": self.egg.data.root_state_w[env_ids].clone(),
            "lid_slot": self.lid_slot[env_ids].clone(),
            "seat_latch": self.seat_latch[env_ids].clone(),
            "egg_latch": self.egg_latch[env_ids].clone(),
            "asm_latch": self.asm_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.bowl.write_root_state_to_sim(state["bowl"], env_ids)
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.egg.write_root_state_to_sim(state["egg"], env_ids)
        self.lid_slot[env_ids] = state["lid_slot"]
        self.seat_latch[env_ids] = state["seat_latch"]
        self.egg_latch[env_ids] = state["egg_latch"]
        self.asm_latch[env_ids] = state["asm_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A gray pedestal STAND rises {c.stand_h * 1000:.0f} mm from the floor: its "
            f"flat column top is ringed by a dark octagonal SOCKET collar (inner width "
            f"{2 * c.socket_r * 1000:.0f} mm, {c.collar_h * 1000:.0f} mm tall) — the "
            f"only fixture in the scene. Scattered on the floor around it lie:\n"
            f"  - an off-white BOWL (octagonal cup, mouth inner width "
            f"{2 * c.mouth_r * 1000:.0f} mm, {c.wall_h * 1000:.0f} mm deep, "
            f"{c.wall_t * 1000:.0f} mm rim walls) with a round "
            f"{2 * c.boss_r * 1000:.0f} mm BOSS under its base — the boss mates with "
            f"the stand's socket;\n"
            f"  - an ORANGE egg (a {2 * c.egg_r * 1000:.0f} mm sphere);\n"
            f"  - TWO dark-red knobbed lids that differ ONLY in size: the LARGE lid "
            f"(cover disk {2 * c.lid_disk_r * 1000:.0f} mm across, wider than the bowl) "
            f"has a centering skirt that drops into the bowl mouth so its disk rests "
            f"on the rim; the SMALL decoy lid ({2 * c.dec_disk_r * 1000:.0f} mm) is "
            f"narrower than the mouth — dropped on the bowl it falls INSIDE and its "
            f"tall knob then stands proud of the rim, blocking the real lid until it "
            f"is lifted back out by its knob. Which lid lies where is shuffled every "
            f"episode: identify them by size.\n"
            f"Goal (a covered dish, judged only on the settled end state): the bowl "
            f"seated in the stand's socket (boss in the collar, upright, centered "
            f"within {c.seat_xy_tol * 1000:.0f} mm), the egg resting INSIDE the bowl, "
            f"and the LARGE lid seated on the bowl — skirt in the mouth, disk resting "
            f"level on the rim, concentric within {c.lid_xy_tol * 1000:.0f} mm. Grasp "
            f"the bowl by its rim wall, the egg directly, and either lid by its "
            f"{2 * c.knob_r * 1000:.0f} mm knob. No order is imposed, but physics "
            f"forces the egg in before the lid goes on, and the decoy must stay OUT "
            f"of the dish. Judged only when everything is settled."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Seat the off-white bowl in the pedestal's socket, place the orange egg "
            "inside the bowl, then cover the bowl with the LARGE dark-red lid so its "
            "skirt drops into the mouth and the disk rests level on the rim. The "
            "small lid is a decoy that falls into the bowl — do not use it."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _stand_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> stand body frame (origin = socket floor centre)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.stand.data.root_quat_w,
                                  p_w - self.stand.data.root_pos_w)

    def _bowl_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> bowl body frame (origin = interior floor top)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.bowl.data.root_quat_w,
                                  p_w - self.bowl.data.root_pos_w)

    def _up_axis(self, body) -> torch.Tensor:
        """(N,3) the body's local +z axis in world coordinates."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)

    # ----- predicates -------------------------------------------------------------------------
    def bowl_seated(self) -> torch.Tensor:
        """(N,) bool: the bowl's boss is mated in the stand's socket — stand-frame
        axis offset within `seat_xy_tol`, origin height in the seated band, upright.
        A bowl perched on the collar rim, standing on the floor, inverted, or riding
        on debris in the socket falls out of the height band or the upright cone."""
        c = self.cfg
        loc = self._stand_local(self.bowl.data.root_pos_w)
        near = loc[:, :2].norm(dim=-1) < c.seat_xy_tol
        z_ok = (loc[:, 2] - c.seat_z).abs() < c.seat_z_tol
        up = self._up_axis(self.bowl)[:, 2] >= math.cos(math.radians(c.upright_max_deg))
        return near & z_ok & up

    def egg_in_bowl(self) -> torch.Tensor:
        """(N,) bool: the egg rests INSIDE the bowl — bowl-frame axis offset within
        `egg_xy_tol` (covers every physical in-bowl rest incl. mouth corners), centre
        height between just above the floor and below the rim. An egg on the floor
        beside the bowl, in the bare socket, or perched on the lid is far outside."""
        c = self.cfg
        loc = self._bowl_local(self.egg.data.root_pos_w)
        near = loc[:, :2].norm(dim=-1) < c.egg_xy_tol
        return near & (loc[:, 2] > c.egg_z_lo) & (loc[:, 2] < c.egg_z_hi)

    def lid_seated(self) -> torch.Tensor:
        """(N,) bool: the REAL lid is capped on the bowl — bowl-frame concentric
        within `lid_xy_tol`, disk bottom in the rim height band, lid axis within
        `lid_tilt_max_deg` of the bowl axis. A lid propped on an in-bowl decoy rides
        high out of the band (asserted in cfg); a tilted rim-perch fails the band and
        the tilt cone."""
        c = self.cfg
        loc = self._bowl_local(self.lid.data.root_pos_w)
        near = loc[:, :2].norm(dim=-1) < c.lid_xy_tol
        z_ok = (loc[:, 2] - c.lid_z).abs() < c.lid_z_tol
        align = (self._up_axis(self.lid) * self._up_axis(self.bowl)).sum(dim=-1)
        level = align >= math.cos(math.radians(c.lid_tilt_max_deg))
        return near & z_ok & level

    def decoy_in_bowl(self) -> torch.Tensor:
        """(N,) bool: the decoy body centre sits inside the bowl's interior volume
        (diagnostic — the physical block it causes is what the rubric feels)."""
        c = self.cfg
        loc = self._bowl_local(self.decoy.data.root_pos_w)
        return (loc[:, :2].norm(dim=-1) < c.mouth_corner_r) \
            & (loc[:, 2] > -0.005) & (loc[:, 2] < c.wall_h + 0.04)

    def settled(self) -> torch.Tensor:
        """(N,) bool: every dynamic body below `settle_lin`."""
        c = self.cfg
        return ((self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.egg.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.lid.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.decoy.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin))

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch each assembly stage every physics substep, so transient progress
        keeps its credit: bowl ever seated; egg ever inside the SEATED bowl; the full
        stack ever mated (seat + egg + lid at once)."""
        seat = self.bowl_seated()
        egg = seat & self.egg_in_bowl()
        asm = egg & self.lid_seated()
        self.seat_latch = torch.maximum(self.seat_latch, seat.float())
        self.egg_latch = torch.maximum(self.egg_latch, egg.float())
        self.asm_latch = torch.maximum(self.asm_latch, asm.float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: bowl seated in the socket + egg inside it + the REAL lid seated
        on it, everything settled. (A decoy fouling the dish is physically excluded:
        it would hold the lid out of its height band.)"""
        return (self.bowl_seated() & self.egg_in_bowl() & self.lid_seated()
                & self.settled())

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.20 * bowl ever seated + 0.25 * egg ever inside the
        seated bowl + 0.25 * full stack ever mated, capped at 0.70; exactly 1.0 iff
        success(). Doing nothing scores ~0; the seed's strategy (carry the bowl
        somewhere and set it down) earns at most the seat credit."""
        base = (0.20 * self.seat_latch + 0.25 * self.egg_latch
                + 0.25 * self.asm_latch).clamp(0.0, 0.70)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="covered_dish", robot="null"))
