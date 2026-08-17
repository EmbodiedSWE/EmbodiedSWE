"""BallastStoveScene — light the deadman-ballast stove by LOADING it, then cook:
put the heavy iron ingot into the ballast tray so the rocker's burner trivet
swings level over the gas dish (the deadman valve opens only under ballast),
then seat the frying pan flat on the level trivet
(sim_gen task `libero_kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it_i242`).

Derived from libero_90/libero_kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it,
but STRATEGICALLY different. The seed is knob-then-place: grasp a stove knob,
rotate its revolute joint past a threshold (`joint_pos > 0.5`), then set the
frypan on a flat, always-available cook plate. Here NOTHING is grasped-and-
rotated and no fixture joint is commanded: the "turn on" act is executed by
PLACING A COUNTERWEIGHT — the stove is a ROCKER (a beam on a pedestal pivot):

  (1) the east arm carries the burner TRIVET (a fenced plate hovering over the
      gas dish); the west arm carries an open-top BALLAST TRAY. Unloaded, the
      beam's own bias tips the trivet DOWN against its tilt stop — gas OFF, and
      the cook surface is a slope. Loading the heavy IRON INGOT (1.4 kg) into
      the tray out-torques the bias and gravity rotates the beam to its LEVEL
      stop — the deadman gas valve opens, and the trivet becomes a level
      cook surface. The pale WOOD BLOCK (same size, 90 g) is a decoy: too
      light to tip the rocker — the discrimination is by MASS, readable only
      through the mechanism's response.
  (2) seat the frying pan flat on the level trivet, inside its rim fence.

The interlock is LIVE (a deadman, not a latch): success() requires the beam
level AND the ingot still in the tray AND the pan seated — lift the ballast
out and gravity re-tilts the trivet, un-lighting the stove (proven in smoke).
The naive seed-family plan (just put the pan "on the stove") leaves the pan on
a tilted, unlit trivet: partial delivery credit at most, never success.

Assets are fully procedural (native PhysX box colliders; the rocker rides a
spawn-authored per-env revolute joint whose LIMITS are the two hard stops):
  - bench (KINEMATIC compound): steel counter 800 x 500 x 160 mm, a pedestal
    column mid-counter carrying the pivot, and the glowing gas dish on the
    deck under the trivet arm (collider ON; flame flicker visual-only).
  - beam (DYNAMIC, on pivot): 440 mm bar; east arm = 150 mm square trivet
    plate with a low rim fence; west arm = 130 mm square tray with 35 mm
    walls. Authored mass 0.9 kg with CoM offset toward the trivet arm — the
    bias that makes the empty rocker rest trivet-down at +16 deg.
  - ingot (DYNAMIC, free): rust-red iron block 60 x 60 x 50 mm, 1.4 kg.
  - block (DYNAMIC, free): pale wood block 60 x 60 x 50 mm, 90 g (decoy).
  - pan (DYNAMIC, free): 90 mm square dish with a 140 mm stick handle, 300 g.

The mechanism arithmetic is asserted in cfg (honesty by construction): the
decoy can NEVER level the rocker (3.5x margin), the pan alone in the tray
cannot either, and the ingot levels it even with the pan's weight on the
trivet arm (1.5x margin) — so "which object is the ballast" is physics, not
decree.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.20 * tray  — ingot ever in the ballast tray (latched, calm)
  0.20 * level — beam ever at its level stop (latched, calm)
  0.15 * pan   — pan ever seated on the trivet, wherever it points (latched, calm)
  1.0 iff success() — beam level AND ingot in the tray AND pan seated on the
                 trivet, everything settled and finite. Non-success cap 0.55.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- procedural compound spawners -------------------------------------------------------------
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


def _box(stage, path: str, size, center, color, contact_offset: float | None) -> None:
    """Author one box child; `contact_offset=None` -> VISUAL ONLY (no collider)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(seg.GetPrim(), contact_offset)


def _rigid_root(root, mass: float, kinematic: bool, lin_damp: float = 0.0,
                ang_damp: float = 0.0, com=None, inertia=None) -> None:
    """RigidBody + Mass authoring. `com`/`inertia` (body-frame CoM offset and
    diagonal inertia) matter for the BEAM: MassAPI mass alone leaves the CoM at
    the body origin, which would erase the rocker's gravity bias."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:  # corpus-validated authoring: never author the attr False
        rb.CreateKinematicEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(mass))
    if com is not None:
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    if inertia is not None:
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    if not kinematic:
        px.CreateSleepThresholdAttr(0.0)
        px.CreateStabilizationThresholdAttr(0.0)


def _spawn_bench(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC stove bench. Origin = counter footprint centre at GROUND
    level: one solid counter slab, the pivot pedestal, and the glowing gas
    dish on the deck under the trivet arm (collider ON) with a flame flicker
    above it (visual only). The bench is the joint anchor body and is NEVER
    moved after spawn (kinematic joint anchors are world-fixed)."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg  # unwrap: the spawner cfg carries the scene cfg in its `cfg` field
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 40.0, kinematic=True)

    co = cfg.contact_offset
    hz = cfg.deck_h
    _box(stage, f"{prim_path}/deck",
         (2 * cfg.bench_half_x, cfg.bench_y_north - cfg.bench_y_south, hz),
         (0.0, (cfg.bench_y_north + cfg.bench_y_south) / 2, hz / 2),
         cfg.deck_color, co)
    # pivot pedestal — MUST clear the tilting bar's swept underside (forge
    # lesson: a flush pedestal props the bar level and masks the joint stops),
    # asserted in cfg (`ped_h` vs the bar sweep over the pedestal footprint)
    _box(stage, f"{prim_path}/pedestal", (0.05, 0.05, cfg.ped_h),
         (0.0, cfg.beam_y, hz + cfg.ped_h / 2), cfg.pedestal_color, co)
    # gas dish under the trivet arm (collider ON: things can rest on it)
    _box(stage, f"{prim_path}/dish", (cfg.dish_s, cfg.dish_s, cfg.dish_h),
         (cfg.arm, cfg.beam_y, hz + cfg.dish_h / 2), cfg.fire_color, co)
    # flame flicker (VISUAL ONLY, stays below the tilted trivet's lowest sweep)
    _box(stage, f"{prim_path}/flame", (0.07, 0.07, 0.014),
         (cfg.arm, cfg.beam_y, hz + cfg.dish_h + 0.007), cfg.flame_color, None)
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC rocker beam. Origin = the PIVOT point: a bar along local x,
    the fenced burner TRIVET plate on the east (+x) arm, the walled ballast
    TRAY on the west (-x) arm. Mass/CoM/inertia are authored explicitly: the
    CoM sits `bias_x` toward the trivet arm — the gravity bias that parks the
    empty rocker trivet-down."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.beam_mass, kinematic=False, lin_damp=cfg.beam_lin_damp,
                ang_damp=cfg.beam_ang_damp, com=(cfg.bias_x, 0.0, cfg.bias_z),
                inertia=cfg.beam_inertia)

    co = cfg.contact_offset
    a = cfg.arm
    _box(stage, f"{prim_path}/bar", (0.44, 0.03, 0.02), (0.0, 0.0, 0.0),
         cfg.beam_color, co)
    # --- east arm: trivet plate + rim fence ---
    tp = cfg.trivet_s
    _box(stage, f"{prim_path}/trivet", (tp, tp, cfg.plate_t),
         (a, 0.0, cfg.plate_z), cfg.trivet_color, co)
    lip_z = cfg.plate_z + cfg.plate_t / 2 + cfg.lip_h / 2
    for tag, sx in (("lip_e", 1.0), ("lip_w", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (cfg.lip_t, tp, cfg.lip_h),
             (a + sx * (tp / 2 - cfg.lip_t / 2), 0.0, lip_z), cfg.trim_color, co)
    for tag, sy in (("lip_n", 1.0), ("lip_s", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (tp - 2 * cfg.lip_t, cfg.lip_t, cfg.lip_h),
             (a, sy * (tp / 2 - cfg.lip_t / 2), lip_z), cfg.trim_color, co)
    # --- west arm: ballast tray (floor + four tall walls) ---
    ts = cfg.tray_s
    _box(stage, f"{prim_path}/tray", (ts, ts, cfg.plate_t),
         (-a, 0.0, cfg.plate_z), cfg.tray_color, co)
    wall_z = cfg.plate_z + cfg.plate_t / 2 + cfg.wall_h / 2
    for tag, sx in (("wall_e", 1.0), ("wall_w", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (cfg.wall_t, ts, cfg.wall_h),
             (-a + sx * (ts / 2 - cfg.wall_t / 2), 0.0, wall_z), cfg.trim_color, co)
    for tag, sy in (("wall_n", 1.0), ("wall_s", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (ts - 2 * cfg.wall_t, cfg.wall_t, cfg.wall_h),
             (-a, sy * (ts / 2 - cfg.wall_t / 2), wall_z), cfg.trim_color, co)
    return root


def _spawn_block(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC plain block — used for BOTH the iron ingot and the wood decoy
    (which one via the spawner-cfg `variant`); identical size, different mass
    and colour: the discrimination the task turns on is MASS."""
    import omni.usd
    from pxr import UsdGeom

    variant = cfg.variant
    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    mass = cfg.ingot_mass if variant == "ingot" else cfg.decoy_mass
    color = cfg.ingot_color if variant == "ingot" else cfg.decoy_color
    _rigid_root(root, mass, kinematic=False, lin_damp=0.2, ang_damp=0.3)
    _box(stage, f"{prim_path}/body", cfg.block_size, (0.0, 0.0, 0.0), color,
         cfg.contact_offset)
    return root


def _spawn_pan(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC frying pan. Origin = BASE plate centre: square dish (base +
    four walls) with a stick handle along local +x (the Franka pinch feature);
    the handle rides high enough to clear the trivet's rim fence."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.pan_mass, kinematic=False, lin_damp=0.2, ang_damp=0.3)

    co, col = cfg.contact_offset, cfg.pan_color
    w, bt, wt, wh = cfg.pan_w, cfg.pan_base_t, cfg.pan_wall_t, cfg.pan_wall_h
    _box(stage, f"{prim_path}/base", (w, w, bt), (0.0, 0.0, 0.0), col, co)
    zc = bt / 2 + wh / 2
    for tag, sx in (("w_xp", 1.0), ("w_xn", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (wt, w, wh),
             (sx * (w / 2 - wt / 2), 0.0, zc), col, co)
    for tag, sy in (("w_yp", 1.0), ("w_yn", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (w - 2 * wt, wt, wh),
             (0.0, sy * (w / 2 - wt / 2), zc), col, co)
    _box(stage, f"{prim_path}/handle",
         (cfg.handle_l, cfg.handle_s, cfg.handle_s),
         (w / 2 + cfg.handle_l / 2, 0.0, cfg.handle_z), cfg.handle_color, co)
    return root


def _compound_spawner_cfg(kind: str, spawn_fn, scene_cfg: Any, mass: float,
                          kinematic: bool, lin_damp: float = 0.0,
                          ang_damp: float = 0.0, variant: str = "") -> Any:
    """Build (once per kind) and instantiate a RigidObjectSpawnerCfg subclass
    wrapping `spawn_fn`, carrying the scene cfg (and a variant tag) through.
    The spawner-cfg rigid_props are applied by the isaaclab clone wrapper
    AFTER the spawn fn runs (sleep stays OFF: sleeping GPU bodies freeze
    mid-settle and ignore velocity writes — corpus lesson). NOTE: mass_props
    are deliberately NOT set here for the beam — the authored CoM/inertia in
    the spawn fn must stay authoritative."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if kind not in _SPAWNER_CACHE:

        @configclass
        class _Cfg(RigidObjectSpawnerCfg):
            func: Callable = clone(spawn_fn)
            cfg: Any = None
            variant: str = ""

        _Cfg.__name__ = f"{kind.title()}SpawnerCfg"
        _SPAWNER_CACHE[kind] = _Cfg

    if kinematic:
        rp = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
    else:
        rp = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False, sleep_threshold=0.0, stabilization_threshold=0.0,
            max_depenetration_velocity=0.5,
            linear_damping=lin_damp, angular_damping=ang_damp,
            solver_position_iteration_count=32, solver_velocity_iteration_count=1)
    return _SPAWNER_CACHE[kind](rigid_props=rp, cfg=scene_cfg, variant=variant)


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BallastStoveSceneCfg(BaseCfg):
    """Config for `BallastStoveScene`. The honesty arithmetic is asserted in
    __post_init__: the wood decoy can never level the rocker, the pan alone in
    the tray cannot either, the iron ingot levels it even against the pan's
    weight on the trivet arm, and the fence/tray windows box their contents
    inside the judged tolerances."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    level_tol_deg: float = tunable(3.0)   # |beam pitch| at most this counts as level
    pan_xy_tol: float = tunable(0.030)    # pan centre within this of the trivet centre (beam frame)
    pan_z_tol: float = tunable(0.010)     # pan base within this of its seat height (beam frame)
    pan_upright_max_deg: float = tunable(12.0)  # pan up-axis within this of the BEAM up-axis
    tray_xy_tol: float = tunable(0.045)   # ingot centre within this of the tray centre (beam frame)
    tray_z_lo: float = tunable(0.005)     # ingot centre above the tray floor by at least this
    tray_z_hi: float = tunable(0.050)     # ... and at most this (a wall-perched ingot reads higher)
    settle_speed: float = tunable(0.05)   # max |lin vel| of ingot and pan when judging (m/s)
    settle_ang: float = tunable(0.10)     # max |ang vel| of the beam when judging (rad/s)
    latch_speed: float = tunable(0.15)    # calm gate for the latches (m/s)
    latch_ang: float = tunable(0.30)      # calm gate for the beam latches (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    slot_x: float = tunable(0.26)         # block spawn slots at +/- this x (coin-swapped)
    slot_y: float = tunable(-0.16)        # block spawn slot y (south apron)
    slot_jitter: float = tunable(0.030)   # uniform +/- xy jitter for both blocks
    pan_x_jitter: float = tunable(0.070)  # pan spawn x jitter (+/- m)
    pan_y_nom: float = tunable(-0.165)    # pan spawn y (south apron)
    pan_y_jitter: float = tunable(0.030)  # pan spawn y jitter (+/- m)
    pan_yaw_center_deg: float = tunable(-90.0)  # handle nominally points south
    pan_yaw_half_deg: float = tunable(60.0)     # free yaw about that (+/- deg)

    # --- info: bench -----------------------------------------------------------------------------
    deck_h: float = info(0.160)
    bench_half_x: float = info(0.400)
    bench_y_north: float = info(0.250)
    bench_y_south: float = info(-0.250)
    beam_y: float = info(0.060)           # the rocker's y lane on the counter
    piv_h: float = info(0.110)            # pivot height above the deck
    ped_h: float = info(0.082)            # pedestal height (< piv_h: clears the bar sweep)
    dish_s: float = info(0.130)
    dish_h: float = info(0.025)
    # --- info: rocker beam -----------------------------------------------------------------------
    arm: float = info(0.185)              # pivot -> trivet/tray centre distance
    plate_t: float = info(0.010)
    plate_z: float = info(0.020)          # plate CENTRE above the beam origin (pivot)
    trivet_s: float = info(0.150)
    lip_t: float = info(0.008)
    lip_h: float = info(0.016)
    tray_s: float = info(0.130)
    wall_t: float = info(0.008)
    wall_h: float = info(0.035)
    tilt_deg: float = info(16.0)          # trivet-down rest angle (the OFF stop)
    beam_mass: float = info(0.90)
    bias_x: float = info(0.085)           # CoM offset toward the trivet arm (the bias)
    bias_z: float = info(0.005)
    beam_inertia: tuple = info((0.010, 0.100, 0.100))
    beam_lin_damp: float = info(2.0)
    beam_ang_damp: float = info(20.0)     # slows the swing so the stop-slam stays gentle
    # --- info: blocks ----------------------------------------------------------------------------
    block_size: tuple = info((0.060, 0.060, 0.050))
    ingot_mass: float = info(1.40)        # cast iron — the working ballast
    decoy_mass: float = info(0.09)        # pine — same size, 15x lighter
    # --- info: pan -------------------------------------------------------------------------------
    pan_w: float = info(0.090)
    pan_base_t: float = info(0.012)
    pan_wall_t: float = info(0.008)
    pan_wall_h: float = info(0.026)
    handle_l: float = info(0.140)
    handle_s: float = info(0.020)
    handle_z: float = info(0.026)         # handle centre above the pan ORIGIN (base centre)
    pan_mass: float = info(0.300)
    # --- info: misc ------------------------------------------------------------------------------
    contact_offset: float = info(0.0015)
    deck_color: tuple = info((0.38, 0.40, 0.44))
    pedestal_color: tuple = info((0.25, 0.26, 0.30))
    fire_color: tuple = info((1.0, 0.45, 0.05))
    flame_color: tuple = info((1.0, 0.65, 0.12))
    beam_color: tuple = info((0.30, 0.32, 0.38))
    trivet_color: tuple = info((0.08, 0.08, 0.09))
    tray_color: tuple = info((0.50, 0.50, 0.55))
    trim_color: tuple = info((0.17, 0.17, 0.20))
    ingot_color: tuple = info((0.42, 0.15, 0.09))
    decoy_color: tuple = info((0.82, 0.70, 0.48))
    pan_color: tuple = info((0.22, 0.22, 0.25))
    handle_color: tuple = info((0.05, 0.05, 0.05))
    # rubric weights (0.20 + 0.20 + 0.15 = 0.55 = the non-success cap)
    w_tray: float = info(0.20)
    w_level: float = info(0.20)
    w_pan: float = info(0.15)

    # ----- derived geometry ----------------------------------------------------------------------
    @property
    def pivot_z(self) -> float:
        return self.deck_h + self.piv_h

    @property
    def plate_top(self) -> float:
        """Plate TOP face height above the beam origin (both arms)."""
        return self.plate_z + self.plate_t / 2

    @property
    def pan_seat_z(self) -> float:
        """Pan ORIGIN height above the beam origin when seated on the trivet."""
        return self.plate_top + self.pan_base_t / 2

    @property
    def fence_win(self) -> float:
        """Clear inner span of the trivet's rim fence."""
        return self.trivet_s - 2 * self.lip_t

    @property
    def tray_win(self) -> float:
        """Clear inner span of the ballast tray."""
        return self.tray_s - 2 * self.wall_t

    @property
    def pan_bottom_dz(self) -> float:
        return self.pan_base_t / 2

    @property
    def block_spawn_z(self) -> float:
        return self.deck_h + self.block_size[2] / 2 + 0.002

    @property
    def pan_spawn_z(self) -> float:
        return self.deck_h + self.pan_bottom_dz + 0.002

    @property
    def bias_torque(self) -> float:
        """Gravity torque of the empty beam about the pivot (trivet-down)."""
        return self.beam_mass * 9.81 * self.bias_x

    def __post_init__(self) -> None:
        g = 9.81
        tau_decoy = self.decoy_mass * g * self.arm
        tau_pan = self.pan_mass * g * self.arm
        tau_ingot = self.ingot_mass * g * self.arm
        # Honesty-by-construction asserts (the claims the task rests on).
        assert tau_decoy <= 0.35 * self.bias_torque, \
            "the wood decoy must NEVER be able to level the rocker (3x margin)"
        assert tau_pan <= 0.85 * self.bias_torque, \
            "the pan alone in the tray must not level the rocker"
        assert tau_ingot >= 1.3 * (self.bias_torque + tau_pan), \
            "the ingot must level the rocker even against the pan on the trivet"
        assert (self.fence_win - self.pan_w) / 2 <= self.pan_xy_tol, \
            "the rim fence must box a seated pan inside the judged xy tolerance"
        assert self.pan_w < self.fence_win, \
            "the pan must drop BETWEEN the fence lips (no flat perch on both)"
        assert self.lip_h > self.pan_z_tol + 0.004, \
            "a pan resting ON a fence lip must read outside the z band"
        assert (self.tray_win - self.block_size[0]) / 2 <= self.tray_xy_tol, \
            "the tray walls must box the ingot inside the judged xy tolerance"
        assert self.wall_h + self.block_size[2] / 2 > self.tray_z_hi + 0.005, \
            "an ingot perched on a tray wall must read outside the z band"
        assert max(self.block_size[0], self.block_size[1]) / 2 + self.tray_z_hi \
            >= self.block_size[2] / 2, "any flat rest of the ingot fits the z band"
        # spawn layout: apron zones clear of the mechanism and of each other
        blk_hd = math.hypot(self.block_size[0], self.block_size[1]) / 2
        pan_hd = self.pan_w * math.sqrt(2.0) / 2
        mech_s = self.beam_y - self.trivet_s / 2
        assert self.slot_y + self.slot_jitter + blk_hd < mech_s - 0.004, \
            "block slots must stay clear of the rocker sweep"
        assert self.pan_y_nom + self.pan_y_jitter + pan_hd < mech_s - 0.004, \
            "the pan spawn zone must stay clear of the rocker sweep"
        assert (self.slot_x - self.slot_jitter - blk_hd) \
            - (self.pan_x_jitter + pan_hd) > 0.010, \
            "block slots must not overlap the pan spawn zone"
        assert self.slot_x + self.slot_jitter + blk_hd < self.bench_half_x - 0.01, \
            "block slots must sit fully on the counter"
        assert self.pan_y_nom - self.pan_y_jitter - self.pan_w / 2 \
            > self.bench_y_south + 0.004, "the pan dish must sit fully on the counter"
        assert math.sin(math.radians(self.pan_yaw_center_deg + self.pan_yaw_half_deg)) < 0.0, \
            "the pan handle must stay on the south (apron) side at any spawn yaw"
        assert self.handle_z - self.handle_s / 2 + self.pan_base_t / 2 \
            > self.lip_h + 0.004, "the seated pan's handle must clear the rim fence"
        # the pedestal must clear the BAR's swept underside at any pitch up to
        # the tilt stop (forge lesson: a flush pedestal props the beam level
        # through depenetration and masks the joint stops entirely)
        tilt = math.radians(self.tilt_deg)
        bar_low_over_ped = self.piv_h - 0.025 * math.sin(tilt) - 0.010 * math.cos(tilt)
        assert self.ped_h < bar_low_over_ped - 0.006, \
            "the pedestal must clear the tilting bar's swept underside"
        edge = self.arm + self.trivet_s / 2
        low = self.piv_h + (self.plate_z - self.plate_t / 2) * math.cos(tilt) \
            - edge * math.sin(tilt)
        assert low > self.dish_h + 0.014 + 0.004, \
            "the tilted trivet must sweep clear of the gas dish and flame"


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("ballast_stove")
class BallastStoveScene(BaseScene):
    cfg: BallastStoveSceneCfg

    def __init__(self, cfg: BallastStoveSceneCfg | None = None) -> None:
        super().__init__(cfg or BallastStoveSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
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
            # Bench authored at the env origin; the bind-time joint anchors here
            # and the bench is NEVER moved (kinematic joint anchors are world-fixed).
            "bench": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=_compound_spawner_cfg("bench", _spawn_bench, c, 40.0, kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            # Beam authored LEVEL at the pivot (its joint-frame pose); reset
            # writes it to the tilt stop (within the joint limits).
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=_compound_spawner_cfg("beam", _spawn_beam, c, c.beam_mass,
                                            kinematic=False, lin_damp=c.beam_lin_damp,
                                            ang_damp=c.beam_ang_damp),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, c.beam_y, c.pivot_z)),
            ),
            "ingot": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ingot",
                spawn=_compound_spawner_cfg("ingot", _spawn_block, c, c.ingot_mass,
                                            kinematic=False, lin_damp=0.2,
                                            ang_damp=0.3, variant="ingot"),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-c.slot_x, c.slot_y, c.block_spawn_z)),
            ),
            "block": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Block",
                spawn=_compound_spawner_cfg("decoy", _spawn_block, c, c.decoy_mass,
                                            kinematic=False, lin_damp=0.2,
                                            ang_damp=0.3, variant="decoy"),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_x, c.slot_y, c.block_spawn_z)),
            ),
            "pan": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pan",
                spawn=_compound_spawner_cfg("pan", _spawn_pan, c, c.pan_mass,
                                            kinematic=False, lin_damp=0.2, ang_damp=0.3),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, c.pan_y_nom, c.pan_spawn_z)),
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
        self.bench: RigidObject = env.iscene["bench"]
        self.beam: RigidObject = env.iscene["beam"]
        self.ingot: RigidObject = env.iscene["ingot"]
        self.block: RigidObject = env.iscene["block"]
        self.pan: RigidObject = env.iscene["pan"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.swap = torch.zeros(n, dtype=torch.bool, device=dev)  # True: ingot in +x slot
        # latches (partial credit survives transients; success is judged live)
        self._tray_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._level_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._pan_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._author_joints()

    def _author_joints(self) -> None:
        """Per-env pivot joint, authored ONCE at bind time against the AUTHORED
        (level) beam pose. Revolute about y, anchored at the pivot point on the
        KINEMATIC bench (its joint anchors are world-fixed, so the bench must
        never move). The joint LIMITS are the two hard stops. Forge-verified
        sign convention: the joint angle about "Y" EQUALS the `_qy` pitch
        (mirrored limits parked a +16 deg-written beam at exactly +0.5 deg =
        that authoring's upper limit), so the trivet-down OFF stop is the
        UPPER limit (+tilt_deg) and the LEVEL stop is the lower (-0.5 deg).
        Joint collision filtering only disables the bench<->beam pair, so the
        ingot, decoy and pan still collide with everything."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        c = self.cfg
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/rocker_pivot")
            j.CreateBody0Rel().SetTargets([f"{base}/Bench"])
            j.CreateBody1Rel().SetTargets([f"{base}/Beam"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, float(c.beam_y), float(c.pivot_z)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-0.5)
            j.CreateUpperLimitAttr(float(c.tilt_deg) + 0.5)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: bench re-asserted at its fixed pose, beam parked at
        its trivet-down tilt stop, ingot and decoy dealt to the two apron slots
        (coin-swapped per episode) with xy jitter + free yaw, pan on the apron
        (xy jitter + south-hemisphere yaw), latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, xyz, quat=None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = xyz + origin
            st[:, 3:7] = quat if quat is not None else torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=dev).expand(m, 4)
            body.write_root_state_to_sim(st, env_ids)

        write(self.bench, torch.zeros(m, 3, device=dev))

        # --- beam: at the pivot, rotated to the trivet-down tilt stop ---
        piv = torch.zeros(m, 3, device=dev)
        piv[:, 1] = c.beam_y
        piv[:, 2] = c.pivot_z
        write(self.beam, piv, _qy(torch.full((m,), math.radians(c.tilt_deg), device=dev)))

        # --- ingot / decoy: coin-swapped slots + jitter + free yaw ---
        self.swap[env_ids] = torch.rand(m, device=dev) < 0.5
        sgn = torch.where(self.swap[env_ids], 1.0, -1.0)
        for body, s in ((self.ingot, sgn), (self.block, -sgn)):
            p = torch.zeros(m, 3, device=dev)
            p[:, 0] = s * c.slot_x
            p[:, 1] = c.slot_y
            p[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            p[:, 2] = c.block_spawn_z
            yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            write(body, p, _qz(yaw))

        # --- pan: south apron, xy jitter, handle in the south hemisphere ---
        p = torch.zeros(m, 3, device=dev)
        p[:, 0] = (torch.rand(m, device=dev) * 2 - 1) * c.pan_x_jitter
        p[:, 1] = c.pan_y_nom + (torch.rand(m, device=dev) * 2 - 1) * c.pan_y_jitter
        p[:, 2] = c.pan_spawn_z
        pyaw = math.radians(c.pan_yaw_center_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pan_yaw_half_deg)
        write(self.pan, p, _qz(pyaw))

        # --- clear latches ---
        for latch in (self._tray_l, self._level_l, self._pan_l):
            latch[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bench": self.bench.data.root_state_w[env_ids].clone(),
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "ingot": self.ingot.data.root_state_w[env_ids].clone(),
            "block": self.block.data.root_state_w[env_ids].clone(),
            "pan": self.pan.data.root_state_w[env_ids].clone(),
            "swap": self.swap[env_ids].clone(),
            "tray_l": self._tray_l[env_ids].clone(),
            "level_l": self._level_l[env_ids].clone(),
            "pan_l": self._pan_l[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for name in ("bench", "beam", "ingot", "block", "pan"):
            getattr(self, name).write_root_state_to_sim(state[name], env_ids)
        self.swap[env_ids] = state["swap"]
        self._tray_l[env_ids] = state["tray_l"]
        self._level_l[env_ids] = state["level_l"]
        self._pan_l[env_ids] = state["pan_l"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A steel KITCHEN BENCH ({2 * c.bench_half_x * 100:.0f} x "
            f"{(c.bench_y_north - c.bench_y_south) * 100:.0f} cm counter, "
            f"{c.deck_h * 100:.0f} cm tall) carries a ROCKER STOVE: a grey beam "
            f"balanced on a pedestal pivot at mid-counter. Its right (east) arm is "
            f"the burner TRIVET — a black {c.trivet_s * 1000:.0f} mm square plate "
            f"with a low rim fence — hovering over a glowing gas dish on the "
            f"counter. Its left (west) arm is an open-top grey BALLAST TRAY "
            f"({c.tray_win * 1000:.0f} mm inner square, {c.wall_h * 1000:.0f} mm "
            f"walls). The rocker is a fail-safe DEADMAN VALVE: with the tray "
            f"empty, the beam rests tipped {c.tilt_deg:.0f} deg trivet-down and "
            f"the gas is OFF; the burner lights ONLY while enough ballast weight "
            f"sits in the tray to hold the beam level against its stop. On the "
            f"front apron sit three items, positions and yaws randomized per "
            f"episode: a rust-red IRON INGOT ({c.block_size[0] * 1000:.0f} x "
            f"{c.block_size[1] * 1000:.0f} x {c.block_size[2] * 1000:.0f} mm, "
            f"heavy — the working ballast), a pale WOOD BLOCK (identical size but "
            f"far too light to tip the rocker), and a square FRYING PAN "
            f"({c.pan_w * 1000:.0f} mm dish with a {c.handle_l * 1000:.0f} mm "
            f"stick handle).\n"
            f"Goal: turn the stove on and put the frying pan on it. Put the IRON "
            f"INGOT into the ballast tray — its weight rotates the beam to the "
            f"level stop (within {c.level_tol_deg:.0f} deg) and lights the burner "
            f"— then seat the pan flat on the trivet, inside its rim fence "
            f"(centre within about {c.pan_xy_tol * 1000:.0f} mm). The ingot must "
            f"STAY in the tray: the valve is a live deadman — removing the "
            f"ballast re-tilts the trivet and shuts the gas off. The wood block "
            f"cannot do the job, and a pan parked on the tilted, unlit trivet "
            f"cooks nothing. Success: beam level, ingot in the tray, pan seated "
            f"flat on the trivet, everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Turn the stove on and put the frying pan on it: place the rust-red "
            "iron ingot in the rocker's ballast tray so the burner trivet swings "
            "level and the gas lights, then set the pan flat on the trivet inside "
            "its rim. The ingot must stay in the tray — removing it shuts the "
            "stove off; the pale wood block is too light to work."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _axes_w(self, body) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,3) body local +x and +z in world."""
        from isaaclab.utils.math import quat_apply

        q = body.data.root_quat_w
        n = q.shape[0]
        ex = torch.tensor([1.0, 0.0, 0.0], device=q.device).expand(n, 3)
        ez = torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(n, 3)
        return quat_apply(q, ex), quat_apply(q, ez)

    def beam_pitch_deg(self) -> torch.Tensor:
        """(N,) beam pitch in deg; POSITIVE = trivet arm down (the OFF tilt)."""
        ax, _az = self._axes_w(self.beam)
        return torch.rad2deg(torch.asin((-ax[:, 2]).clamp(-1.0, 1.0)))

    def _in_beam(self, body) -> torch.Tensor:
        """(N,3) the body's origin in the BEAM body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        d = body.data.root_pos_w - self.beam.data.root_pos_w
        return quat_apply_inverse(self.beam.data.root_quat_w, d)

    def beam_level(self) -> torch.Tensor:
        """(N,) bool: beam within `level_tol_deg` of level — the valve-open
        attitude, held against the level stop only by sufficient ballast."""
        return self.beam_pitch_deg().abs() <= self.cfg.level_tol_deg

    def ingot_in_tray(self) -> torch.Tensor:
        """(N,) bool, geometric, BEAM-FRAME: the IRON INGOT's centre inside the
        tray window (xy) and resting in the tray's z band (a wall-perched or
        hovering ingot reads outside it)."""
        c = self.cfg
        p = self._in_beam(self.ingot)
        xy_ok = ((p[:, 0] + c.arm).abs() <= c.tray_xy_tol) \
            & (p[:, 1].abs() <= c.tray_xy_tol)
        dz = p[:, 2] - c.plate_top - c.block_size[2] / 2
        z_ok = (dz >= c.tray_z_lo - c.block_size[2] / 2) \
            & (p[:, 2] - c.plate_top <= c.tray_z_hi)
        return xy_ok & z_ok

    def pan_on_trivet(self) -> torch.Tensor:
        """(N,) bool, geometric, BEAM-FRAME: pan centre inside the trivet's rim
        fence (xy), pan base at its seat height (z band), pan up-axis aligned
        with the BEAM's up-axis — true wherever the beam points, so delivery
        credit survives tilt, while success additionally demands level."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        p = self._in_beam(self.pan)
        xy_ok = ((p[:, 0] - c.arm).abs() <= c.pan_xy_tol) \
            & (p[:, 1].abs() <= c.pan_xy_tol)
        z_ok = (p[:, 2] - c.pan_seat_z).abs() <= c.pan_z_tol
        n = self.pan.data.root_quat_w.shape[0]
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        pan_up = quat_apply(self.pan.data.root_quat_w, ez)
        _bx, beam_up = self._axes_w(self.beam)
        cosang = (pan_up * beam_up).sum(dim=-1).clamp(-1.0, 1.0)
        upright = cosang > math.cos(math.radians(c.pan_upright_max_deg))
        return xy_ok & z_ok & upright

    def settled(self) -> torch.Tensor:
        """(N,) bool: ingot and pan |lin vel| below `settle_speed`, beam
        |ang vel| below `settle_ang`."""
        c = self.cfg
        return (self.ingot.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.pan.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.beam.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def _update_latches(self) -> None:
        c = self.cfg
        calm_b = self.beam.data.root_ang_vel_w.norm(dim=-1) < c.latch_ang
        calm_i = self.ingot.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        calm_p = self.pan.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        self._tray_l |= self.ingot_in_tray() & calm_i & calm_b
        self._level_l |= self.beam_level() & calm_b
        self._pan_l |= self.pan_on_trivet() & calm_p & calm_b

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: beam at its level stop AND the iron ingot in the ballast
        tray AND the pan seated flat on the trivet, everything settled and
        finite. All clauses are LIVE physical outcomes — the deadman: lifting
        the ballast out re-tilts the beam and success collapses."""
        self._update_latches()
        finite = torch.isfinite(self.beam.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.ingot.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.pan.data.root_pos_w).all(dim=-1)
        return self.beam_level() & self.ingot_in_tray() & self.pan_on_trivet() \
            & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.20*ingot-in-tray + 0.20*beam-level +
        0.15*pan-on-trivet (all latched, calm-gated; ~0 for doing nothing),
        capped at 0.55 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_tray * self._tray_l.float()
                + c.w_level * self._level_l.float()
                + c.w_pan * self._pan_l.float()).clamp(max=0.55)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="ballast_stove", robot="null"))
