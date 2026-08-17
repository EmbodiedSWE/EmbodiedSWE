"""TeapotBayonetScene — seat the teapot's lid through its keyed (bayonet) mouth and
twist it to the internal stop so it locks captive (sim_gen task
`approach_grasp_ceramic_teapot_i264`).

Derived from pick_place/approach_grasp_ceramic_teapot, but STRATEGICALLY different:
the seed is a prehensile acquisition task — approach the known teapot in tabletop
clutter, close the parallel jaw on it, hold, small lift; success is a gripper-object
distance relation and the episode ends HOLDING. Here no gripper relation is ever read
and holding anything is worth at most a small carry latch: the judged outcome is a
MECHANISM STATE built between two rigid bodies — the free LID must end seated at
bearing depth inside the teapot's bayonet mouth AND rotated past full lug engagement,
so that it is geometrically CAPTIVE under the throat flange. The plan a solver needs
is align -> insert-through-key -> press-and-twist-to-stop, an ordered sequence that
geometry itself enforces (a rotated lid cannot enter; an unrotated lid is not locked
and lifts straight out).

The mouth is a bayonet collar: a cream throat RING (aperture r `ring_inner_r`) with
TWO diametric NOTCH gaps (`notch_deg` wide, angular position randomized with the
pot's yaw, marked by red stripes on the outside wall); below the ring an annular
CHAMBER, then a LEDGE the lid's plug rests on. The lid carries two RED LUGS that fit
only through the notches; once the plug is on the ledge, the lugs sit below the ring
and twisting the lid slides them under the solid ring — after `lock_min_deg` of
rotation the lugs are fully engaged and the lid cannot be pulled out; internal STOP
blocks end the travel at ~48 deg (either direction works).

success() (live, physical): lid origin within `xy_tol` of the pot axis, at seated
depth (`seat_z_lo`..`seat_z_hi`), upright, relative lid-vs-pot yaw (folded mod 180,
the lug/notch symmetry) in [`lock_min_deg`, `lock_max_deg`], and the lid settled.
score(): latched stages anchored in the demonstrated solution — 0.15 carried (lid
brought above the mouth), +0.25 entered (plug below the ring on the pot axis),
+0.35 * rotation progress toward `lock_min_deg` while seated (latched max), capped
at 0.75; exactly 1.0 iff success(). Null policy ~0.

Assets are fully procedural (no external files):
  - pot: KINEMATIC compound (base disc, 12-sector wall, 12-box ledge annulus,
    16-box throat ring forming the two notches, 4 chamber stop blocks, blocky spout
    + handle for identity, red notch stripes) — teleported at reset for xy + yaw
    randomization.
  - lid: DYNAMIC compound (pilot nose, plug, cap, dark-green knob, two red lugs;
    0.25 kg, origin/CoM at the nose bottom centre).

Per-episode randomization (verified by readback in smoke): pot xy jitter + FREE yaw
(the notch phase a solver must perceive), lid spawn polar angle + radius + free yaw.
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


def _box(stage, path: str, size, center, color, contact_offset: float | None,
         yaw: float = 0.0) -> None:
    """A colored box prim (optional yaw about local z); collides iff `contact_offset`
    is not None."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw != 0.0:
        sxf.AddRotateZOp().Set(math.degrees(yaw))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(seg.GetPrim(), contact_offset)


def _cylinder(stage, path: str, radius: float, height: float, center, color,
              contact_offset: float | None) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    seg.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(seg.GetPrim(), contact_offset)


def _phys_material(stage, path: str, static: float, dynamic: float) -> Any:
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


def _rigid_dynamic(root, mass: float, lin_damp: float, ang_damp: float) -> None:
    """Dynamic rigid-body armor on a compound root: explicit MassAPI mass (custom
    spawners apply NO cfg schemas — author everything here), damping, no sleeping
    while we judge velocities, and the depenetration cap."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _rigid_kinematic(root) -> None:
    from pxr import UsdPhysics

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(50.0)


def _spawn_pot(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC bayonet teapot at `prim_path`. Origin = ground-level
    centre of the base. Pot-local +x points at notch 0 (the other notch is at 180
    deg); the spout is at +90 deg, the handle at -90 deg, and a red stripe rides the
    outer wall under each notch."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_kinematic(root)

    co = cfg.contact_offset
    body_col = cfg.body_color
    interior: list[str] = []

    # base disc
    _cylinder(stage, f"{prim_path}/base", cfg.base_r, cfg.base_h,
              (0.0, 0.0, cfg.base_h / 2), body_col, co)

    # 12-sector body wall: inner r wall_inner_r, z base_h .. ring_z0
    n_wall = 12
    wall_mid = cfg.wall_inner_r + cfg.wall_t / 2
    wall_len = 2 * (cfg.wall_inner_r + cfg.wall_t) * math.tan(math.pi / n_wall) + 0.002
    wall_h = cfg.ring_z0 - cfg.base_h
    for k in range(n_wall):
        ang = 2 * math.pi * k / n_wall
        _box(stage, f"{prim_path}/wall_{k:02d}", (cfg.wall_t, wall_len, wall_h),
             (wall_mid * math.cos(ang), wall_mid * math.sin(ang),
              cfg.base_h + wall_h / 2), body_col, co, yaw=ang)
        interior.append(f"wall_{k:02d}")

    # ledge annulus (12 boxes): inner ledge_inner_r .. wall_inner_r, z ledge_z0..ledge_z1
    led_mid = (cfg.ledge_inner_r + cfg.wall_inner_r) / 2
    led_rad = cfg.wall_inner_r - cfg.ledge_inner_r
    led_len = 2 * cfg.wall_inner_r * math.tan(math.pi / n_wall) + 0.002
    led_h = cfg.ledge_z1 - cfg.ledge_z0
    for k in range(n_wall):
        ang = 2 * math.pi * (k + 0.5) / n_wall
        _box(stage, f"{prim_path}/ledge_{k:02d}", (led_rad, led_len, led_h),
             (led_mid * math.cos(ang), led_mid * math.sin(ang),
              (cfg.ledge_z0 + cfg.ledge_z1) / 2), cfg.ledge_color, co, yaw=ang)
        interior.append(f"ledge_{k:02d}")

    # throat ring: two solid arcs between the notches, 8 boxes each of 17 deg.
    # Notches centred at pot-local 0 and 180 deg, each notch_deg wide.
    ring_mid = (cfg.ring_inner_r + cfg.ring_outer_r) / 2
    ring_rad = cfg.ring_outer_r - cfg.ring_inner_r
    ring_h = cfg.ring_z1 - cfg.ring_z0
    n_seg = 8
    arc = math.pi - math.radians(cfg.notch_deg)  # solid arc angular length
    seg_w = arc / n_seg
    seg_len = 2 * cfg.ring_outer_r * math.tan(seg_w / 2) + 0.002
    idx = 0
    for arc_start in (math.radians(cfg.notch_deg / 2),
                      math.pi + math.radians(cfg.notch_deg / 2)):
        for j in range(n_seg):
            ang = arc_start + seg_w * (j + 0.5)
            _box(stage, f"{prim_path}/ring_{idx:02d}", (ring_rad, seg_len, ring_h),
                 (ring_mid * math.cos(ang), ring_mid * math.sin(ang),
                  (cfg.ring_z0 + cfg.ring_z1) / 2), cfg.ring_color, co, yaw=ang)
            interior.append(f"ring_{idx:02d}")
            idx += 1

    # chamber stop blocks at +-stop_center_deg around each notch
    stop_mid = (cfg.ring_inner_r + cfg.wall_inner_r) / 2
    stop_rad = cfg.wall_inner_r - cfg.ring_inner_r
    stop_len = 2 * stop_mid * math.tan(math.radians(cfg.stop_hw_deg)) + 0.001
    stop_h = cfg.ring_z0 - cfg.ledge_z1
    for i, base in enumerate((cfg.stop_center_deg, 180.0 - cfg.stop_center_deg,
                              180.0 + cfg.stop_center_deg, 360.0 - cfg.stop_center_deg)):
        ang = math.radians(base)
        _box(stage, f"{prim_path}/stop_{i}", (stop_rad, stop_len, stop_h),
             (stop_mid * math.cos(ang), stop_mid * math.sin(ang),
              (cfg.ledge_z1 + cfg.ring_z0) / 2), cfg.stop_color, co, yaw=ang)
        interior.append(f"stop_{i}")

    # blocky spout (+90 deg) and handle (-90 deg) — identity features, away from
    # the mouth (their tops stay below the rim so the approach cone is clean)
    _box(stage, f"{prim_path}/spout_a", (0.024, 0.055, 0.024),
         (0.0, 0.096, 0.150), body_col, co)
    _box(stage, f"{prim_path}/spout_b", (0.024, 0.024, 0.055),
         (0.0, 0.118, 0.180), body_col, co)
    _box(stage, f"{prim_path}/handle_v", (0.022, 0.022, 0.120),
         (0.0, -0.128, 0.125), body_col, co)
    _box(stage, f"{prim_path}/handle_t", (0.022, 0.045, 0.020),
         (0.0, -0.100, 0.185), body_col, co)
    _box(stage, f"{prim_path}/handle_b", (0.022, 0.045, 0.020),
         (0.0, -0.100, 0.065), body_col, co)

    # red notch stripes on the OUTER wall (visual only) under each notch
    for i, ang in enumerate((0.0, math.pi)):
        r = cfg.ring_outer_r + 0.003
        _box(stage, f"{prim_path}/notch_mark_{i}", (0.004, 0.018, 0.070),
             (r * math.cos(ang), r * math.sin(ang), 0.190), (0.90, 0.10, 0.08),
             None, yaw=ang)

    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.friction, cfg.friction - 0.05)
    for name in interior:
        _bind_material(stage.GetPrimAtPath(f"{prim_path}/{name}"), mat)
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC bayonet lid at `prim_path`. Origin = pilot-nose bottom
    centre (CoM sits there — low CoM keeps the insertion stable). Local +x points
    at lug 0 (the other lug is at 180 deg)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass, lin_damp=0.05, ang_damp=0.08)

    co = cfg.contact_offset
    _cylinder(stage, f"{prim_path}/nose", cfg.nose_r, cfg.nose_h,
              (0.0, 0.0, cfg.nose_h / 2), cfg.plug_color, co)
    _cylinder(stage, f"{prim_path}/plug", cfg.plug_r, cfg.plug_h,
              (0.0, 0.0, cfg.nose_h + cfg.plug_h / 2), cfg.plug_color, co)
    cap_z0 = cfg.nose_h + cfg.plug_h
    _cylinder(stage, f"{prim_path}/cap", cfg.cap_r, cfg.cap_h,
              (0.0, 0.0, cap_z0 + cfg.cap_h / 2), cfg.cap_color, co)
    _box(stage, f"{prim_path}/knob", (cfg.knob_a, cfg.knob_a, cfg.knob_h),
         (0.0, 0.0, cap_z0 + cfg.cap_h + cfg.knob_h / 2), cfg.knob_color, co)
    lug_mid = (cfg.lug_r0 + cfg.lug_r1) / 2
    for i, ang in enumerate((0.0, math.pi)):
        _box(stage, f"{prim_path}/lug_{i}",
             (cfg.lug_r1 - cfg.lug_r0, cfg.lug_w, cfg.lug_h),
             (lug_mid * math.cos(ang), lug_mid * math.sin(ang),
              cfg.lug_z0 + cfg.lug_h / 2), (0.90, 0.10, 0.08), co, yaw=ang)

    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.friction, cfg.friction - 0.05)
    for name in ("nose", "plug", "lug_0", "lug_1"):
        _bind_material(stage.GetPrimAtPath(f"{prim_path}/{name}"), mat)
    return root


def _pot_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pot" not in _SPAWNER_CACHE:

        @configclass
        class PotSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pot)
            base_r: float = 0.078
            base_h: float = 0.010
            wall_inner_r: float = 0.070
            wall_t: float = 0.008
            ledge_inner_r: float = 0.030
            ledge_z0: float = 0.190
            ledge_z1: float = 0.198
            ring_inner_r: float = 0.048
            ring_outer_r: float = 0.078
            ring_z0: float = 0.220
            ring_z1: float = 0.230
            notch_deg: float = 44.0
            stop_center_deg: float = 64.5
            stop_hw_deg: float = 5.0
            friction: float = 0.25
            body_color: tuple = (0.33, 0.42, 0.52)
            ring_color: tuple = (0.92, 0.88, 0.78)
            ledge_color: tuple = (0.22, 0.28, 0.35)
            stop_color: tuple = (0.15, 0.18, 0.22)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["pot"] = PotSpawnerCfg

    return _SPAWNER_CACHE["pot"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), **kw)


def _lid_spawner_cfg(**kw: Any) -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "lid" not in _SPAWNER_CACHE:

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            nose_r: float = 0.036
            nose_h: float = 0.008
            plug_r: float = 0.042
            plug_h: float = 0.032
            cap_r: float = 0.066
            cap_h: float = 0.008
            knob_a: float = 0.022
            knob_h: float = 0.034
            lug_r0: float = 0.040
            lug_r1: float = 0.058
            lug_w: float = 0.016
            lug_h: float = 0.008
            lug_z0: float = 0.008
            mass: float = 0.25
            friction: float = 0.30
            plug_color: tuple = (0.60, 0.66, 0.74)
            cap_color: tuple = (0.92, 0.88, 0.78)
            knob_color: tuple = (0.10, 0.45, 0.18)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["lid"] = LidSpawnerCfg

    return _SPAWNER_CACHE["lid"](**kw)


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TeapotBayonetSceneCfg(BaseCfg):
    """Config for `TeapotBayonetScene`. Honesty knobs asserted in `__post_init__`:
    the lock threshold sits above full lug engagement and below the stop travel; the
    seat band rejects a rim-resting lid with margin; the keyed aperture really keys
    (nose cannot pass the ledge hole, lugs cannot pass the solid ring); every
    clearance the arm needs (pilot capture, yaw margin, knob width) is real."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    xy_tol: float = tunable(0.010)  # lid origin within this of the pot axis (m)
    seat_z_lo: float = tunable(0.188)  # lid origin seated band (analytic seat 0.198;
    seat_z_hi: float = tunable(0.208)  # rim-resting lid reads 0.222 -> rejected)
    lock_min_deg: float = tunable(36.0)  # folded relative yaw >= this = LOCKED
    lock_max_deg: float = tunable(55.0)  # sanity ceiling (stops end travel at ~48)
    upright_max_deg: float = tunable(8.0)  # lid axis within this of world-up
    settle_lin: float = tunable(0.08)  # max lid |lin vel| at success (above the GPU
    settle_ang: float = tunable(0.50)  # phantom-velocity artifact band; geometry
    # windows + the solve persistence hold carry the honesty)

    # --- tunable: score latch gates ----------------------------------------------------------
    carried_xy: float = tunable(0.12)  # carried latch: lid above the mouth
    enter_z: float = tunable(0.210)  # entered latch: origin below this ...
    enter_xy: float = tunable(0.020)  # ... while on the pot axis
    rot_gate_xy: float = tunable(0.015)  # rotation progress only counts while seated

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    pot_nom: tuple = tunable((0.10, 0.0))  # nominal pot centre (env frame)
    pot_jitter: float = tunable(0.045)  # uniform +/- xy jitter
    lid_r_lo: float = tunable(0.28)  # lid spawn ring around the pot (m)
    lid_r_hi: float = tunable(0.38)

    # --- info: pot structure (mirrors the spawner defaults; single source here) --------------
    base_r: float = info(0.078)
    base_h: float = info(0.010)
    wall_inner_r: float = info(0.070)
    wall_t: float = info(0.008)
    ledge_inner_r: float = info(0.030)
    ledge_z0: float = info(0.190)
    ledge_z1: float = info(0.198)  # ledge TOP = the analytic seat plane
    ring_inner_r: float = info(0.048)  # throat aperture (12-ish-gon inradius)
    ring_outer_r: float = info(0.078)
    ring_z0: float = info(0.220)  # ring bottom (chamber top)
    ring_z1: float = info(0.230)  # rim top
    notch_deg: float = info(44.0)  # full angular width of each notch gap
    stop_center_deg: float = info(64.5)  # chamber stop blocks at +-this per notch
    stop_hw_deg: float = info(5.0)
    pot_friction: float = info(0.25)

    # --- info: lid structure ------------------------------------------------------------------
    nose_r: float = info(0.036)  # pilot nose (throat capture = ring_inner - nose_r)
    nose_h: float = info(0.008)
    plug_r: float = info(0.042)
    plug_h: float = info(0.032)
    cap_r: float = info(0.066)
    cap_h: float = info(0.008)
    knob_a: float = info(0.022)  # knob width — the Franka grasp feature
    knob_h: float = info(0.034)
    lug_r0: float = info(0.040)
    lug_r1: float = info(0.058)
    lug_w: float = info(0.016)
    lug_h: float = info(0.008)
    lug_z0: float = info(0.008)  # lug bottom above the lid origin
    lid_mass: float = info(0.25)
    lid_friction: float = info(0.30)
    jaw_max: float = info(0.080)  # Franka parallel-jaw max opening
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    lug_hw_deg: float = field(default=None, init=False)  # lug angular half-width
    engage_min_deg: float = field(default=None, init=False)  # lug fully under ring
    max_travel_deg: float = field(default=None, init=False)  # stop-limited travel
    seat_z: float = field(default=None, init=False)  # analytic seated origin z
    rim_rest_z: float = field(default=None, init=False)  # lug-on-rim origin z

    def __post_init__(self) -> None:
        self.lug_hw_deg = math.degrees(math.atan((self.lug_w / 2) / self.lug_r0))
        self.engage_min_deg = self.notch_deg / 2 + self.lug_hw_deg
        self.max_travel_deg = self.stop_center_deg - self.stop_hw_deg - self.lug_hw_deg
        self.seat_z = self.ledge_z1
        self.rim_rest_z = self.ring_z1 - self.lug_z0

        # The key really keys, both ways.
        assert self.notch_deg / 2 - self.lug_hw_deg >= 8.0, (
            "aligned lugs need >= 8 deg of yaw margin through the notch")
        assert self.engage_min_deg + 2.0 <= self.lock_min_deg <= self.max_travel_deg - 4.0, (
            "the lock threshold must sit above full engagement and below the stops")
        assert self.lock_max_deg > self.max_travel_deg + 4.0, (
            "the sanity ceiling must not clip physically reachable travel")
        # Captive play: lugs clear the ring bottom when seated, but only just.
        play = (self.ring_z0 - self.ledge_z1) - (self.lug_z0 + self.lug_h)
        assert 0.003 <= play <= 0.010, f"captive vertical play {play:.3f} out of band"
        # Aperture honesty.
        assert self.nose_r >= self.ledge_inner_r + 0.004, "nose must NOT pass the ledge hole"
        assert 0.005 <= self.ring_inner_r - self.plug_r <= 0.010, (
            "plug-throat radial clearance out of the honest band")
        assert self.ring_inner_r - self.nose_r >= 0.010, (
            "the pilot nose must give the arm >= 10 mm of xy capture")
        assert self.lug_r1 <= self.wall_inner_r - 0.008, "lug tips must clear the chamber wall"
        assert self.lug_r1 - self.ring_inner_r >= 0.008, (
            "lugs must overlap the ring by >= 8 mm radially (real captivity)")
        # Seat band rejects the rim rest with margin; the enter gate sits between them.
        assert self.rim_rest_z >= self.seat_z_hi + 0.010, "rim rest must fail the seat band"
        assert self.seat_z_lo < self.seat_z < self.seat_z_hi
        assert self.seat_z_hi < self.enter_z < self.ring_z0, (
            "the entered gate must sit between the seat band and the ring bottom")
        # Embodiment: knob fits the jaw; lid spawn ring clears the pot + its features.
        assert self.knob_a < self.jaw_max - 0.030
        assert self.lid_r_lo - self.cap_r > 0.145 + self.pot_jitter + 0.020, (
            "lid spawn ring must clear the pot's spout/handle envelope")
        # The cap never enters the mouth: seated cap bottom sits above the rim.
        assert self.seat_z + self.nose_h + self.plug_h > self.ring_z1 + 0.004


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("teapot_bayonet")
class TeapotBayonetScene(BaseScene):
    cfg: TeapotBayonetSceneCfg

    def __init__(self, cfg: TeapotBayonetSceneCfg | None = None) -> None:
        super().__init__(cfg or TeapotBayonetSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "pot": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pot",
                spawn=_pot_spawner_cfg(
                    base_r=c.base_r, base_h=c.base_h, wall_inner_r=c.wall_inner_r,
                    wall_t=c.wall_t, ledge_inner_r=c.ledge_inner_r,
                    ledge_z0=c.ledge_z0, ledge_z1=c.ledge_z1,
                    ring_inner_r=c.ring_inner_r, ring_outer_r=c.ring_outer_r,
                    ring_z0=c.ring_z0, ring_z1=c.ring_z1, notch_deg=c.notch_deg,
                    stop_center_deg=c.stop_center_deg, stop_hw_deg=c.stop_hw_deg,
                    friction=c.pot_friction, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.pot_nom[0], c.pot_nom[1], 0.0)),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=_lid_spawner_cfg(
                    nose_r=c.nose_r, nose_h=c.nose_h, plug_r=c.plug_r, plug_h=c.plug_h,
                    cap_r=c.cap_r, cap_h=c.cap_h, knob_a=c.knob_a, knob_h=c.knob_h,
                    lug_r0=c.lug_r0, lug_r1=c.lug_r1, lug_w=c.lug_w, lug_h=c.lug_h,
                    lug_z0=c.lug_z0, mass=c.lid_mass, friction=c.lid_friction,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.pot_nom[0] + 0.33, 0.0, 0.003)),
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
        self.pot: RigidObject = env.iscene["pot"]
        self.lid: RigidObject = env.iscene["lid"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.carried = torch.zeros(n, device=dev)  # lid brought above the mouth
        self.entered = torch.zeros(n, device=dev)  # plug below the ring, on axis
        self.rot_max = torch.zeros(n, device=dev)  # max folded rel yaw while seated (rad)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pot teleported to nominal + xy jitter with FREE yaw (the
        notch phase the solver must perceive); lid upright on the ground on a random
        polar slot around the pot with free yaw; latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        px = c.pot_nom[0] + (torch.rand(m, device=dev) * 2 - 1) * c.pot_jitter
        py = c.pot_nom[1] + (torch.rand(m, device=dev) * 2 - 1) * c.pot_jitter
        pot_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = px, py
        st[:, 3] = torch.cos(pot_yaw / 2)
        st[:, 6] = torch.sin(pot_yaw / 2)
        st[:, 0:3] += origin
        self.pot.write_root_state_to_sim(st, env_ids)

        ang = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        rad = c.lid_r_lo + torch.rand(m, device=dev) * (c.lid_r_hi - c.lid_r_lo)
        lid_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = px + rad * torch.cos(ang)
        st[:, 1] = py + rad * torch.sin(ang)
        st[:, 2] = 0.003
        st[:, 3] = torch.cos(lid_yaw / 2)
        st[:, 6] = torch.sin(lid_yaw / 2)
        st[:, 0:3] += origin
        self.lid.write_root_state_to_sim(st, env_ids)

        self.carried[env_ids] = 0.0
        self.entered[env_ids] = 0.0
        self.rot_max[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pot": self.pot.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "carried": self.carried[env_ids].clone(),
            "entered": self.entered[env_ids].clone(),
            "rot_max": self.rot_max[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.pot.write_root_state_to_sim(state["pot"], env_ids)
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        self.carried[env_ids] = state["carried"]
        self.entered[env_ids] = state["entered"]
        self.rot_max[env_ids] = state["rot_max"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A slate-blue ceramic TEAPOT (round, ~{2 * c.ring_outer_r * 100:.0f} cm wide, "
            f"{c.ring_z1 * 100:.0f} cm tall, blocky spout on one side, handle on the other) "
            f"stands on the ground, and its LID (cream cap with a dark-green knob on top, "
            f"{c.knob_a * 1000:.0f} mm square) lies upright on the ground nearby. The pot's "
            f"mouth is a BAYONET collar: a cream rim ring with a round "
            f"{2 * c.ring_inner_r * 1000:.0f} mm throat, interrupted by TWO diametrically "
            f"opposite NOTCH gaps ({c.notch_deg:.0f} deg wide); a RED STRIPE on the outside "
            f"wall marks each notch. The lid carries two RED LUGS on the sides of its plug "
            f"that fit only through those notches. The pot's position and the direction the "
            f"notches face vary by episode — read the red stripes.\n"
            f"Goal: LOCK the lid into the teapot. Turn the lid so its red lugs line up with "
            f"the red-marked notches, lower it into the throat until the plug seats on the "
            f"internal ledge (the cap then sits just above the rim), and TWIST it — either "
            f"direction — until it reaches the internal stop, about a quarter of a "
            f"quarter-turn (~{c.max_travel_deg:.0f} deg). A twist of at least "
            f"{c.lock_min_deg:.0f} deg from the notch alignment counts as locked; less than "
            f"that, or a lid merely resting on the rim, or a lid left anywhere else, does "
            f"not. Leave the lid seated, level and at rest; a locked lid cannot be pulled "
            f"back out."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lock the teapot's lid: align its two red lugs with the red-marked notch gaps "
            "in the rim, lower the lid into the mouth until it seats on the internal ledge, "
            "then twist it to the internal stop — at least 36 degrees past the notches. "
            "Leave it seated and at rest; an unrotated or rim-resting lid fails."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _lid_pos(self) -> torch.Tensor:
        return self.lid.data.root_pos_w - self.env_origins

    def _pot_pos(self) -> torch.Tensor:
        return self.pot.data.root_pos_w - self.env_origins

    @staticmethod
    def _yaw(quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def rel_yaw_fold(self) -> torch.Tensor:
        """(N,) folded relative lid-vs-pot yaw in [0, pi/2] — the lug/notch pattern
        is 2-fold symmetric, so the lock state lives mod 180 deg."""
        d = self._yaw(self.lid.data.root_quat_w) - self._yaw(self.pot.data.root_quat_w)
        t = torch.remainder(d, math.pi)
        return torch.minimum(t, math.pi - t)

    def lid_up_z(self) -> torch.Tensor:
        q = self.lid.data.root_quat_w
        return 1 - 2 * (q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2])

    def _on_axis(self, tol: float) -> torch.Tensor:
        d = (self._lid_pos()[:, :2] - self._pot_pos()[:, :2]).norm(dim=-1)
        return d < tol

    def seated(self) -> torch.Tensor:
        """(N,) bool: lid origin on the pot axis, in the seated depth band, upright."""
        c = self.cfg
        z = self._lid_pos()[:, 2]
        return (self._on_axis(c.xy_tol) & (z > c.seat_z_lo) & (z < c.seat_z_hi)
                & (self.lid_up_z() > math.cos(math.radians(c.upright_max_deg))))

    def locked(self) -> torch.Tensor:
        """(N,) bool: seated AND rotated into the lock window (lugs fully under the
        ring — the pull-out direction is geometrically blocked)."""
        c = self.cfg
        th = torch.rad2deg(self.rel_yaw_fold())
        return self.seated() & (th >= c.lock_min_deg) & (th <= c.lock_max_deg)

    def settled(self) -> torch.Tensor:
        c = self.cfg
        return ((self.lid.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.lid.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang))

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch carried (lid above the mouth), entered (plug below the ring on the
        pot axis — only reachable through the keyed throat), and the max folded
        rotation achieved while seated, each physics substep, so demonstrated
        progress keeps its credit."""
        c = self.cfg
        lp = self._lid_pos()
        z = lp[:, 2]
        self.carried = torch.maximum(
            self.carried, (self._on_axis(c.carried_xy) & (z > c.ring_z1)).float())
        self.entered = torch.maximum(
            self.entered, (self._on_axis(c.enter_xy) & (z < c.enter_z)).float())
        gate = (self._on_axis(c.rot_gate_xy) & (z < c.seat_z_hi)
                & (self.lid_up_z() > math.cos(math.radians(10.0))))
        self.rot_max = torch.maximum(self.rot_max,
                                     torch.where(gate, self.rel_yaw_fold(),
                                                 torch.zeros_like(self.rot_max)))

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the lid is LOCKED (seated at depth + rotated into the lock
        window) and settled — a live physical state, never bookkeeping."""
        return self.locked() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 carried + 0.25 entered + 0.35 * latched
        rotation progress toward `lock_min_deg`, capped at 0.75; exactly 1.0 iff
        success(). Null policy ~0; the seed's grasp-and-hold strategy tops out at
        the 0.15 carry latch."""
        c = self.cfg
        frac = (torch.rad2deg(self.rot_max) / c.lock_min_deg).clamp(0.0, 1.0)
        base = (0.15 * self.carried + 0.25 * self.entered + 0.35 * frac).clamp(max=0.75)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="teapot_bayonet", robot="null"))
