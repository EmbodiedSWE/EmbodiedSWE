"""DeadmanStoveScene — hold the stove's spring-loaded DEAD-MAN'S GAS PEDAL down
with the cast-iron ballast weight (the foam decoy is too light), then set the
frying pan on the burner plate (sim_gen task
`libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it_i278`).

Derived from libero_90/libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it,
but STRATEGICALLY different. The seed is knob-then-place: rotate a stove knob
past a JOINT-ANGLE threshold (`joint_pos > 0.5`) — a direct fixture actuation
whose state the rubric reads straight off the fixture joint — then set the
frypan on the always-available cook plate. Here NOTHING on the stove is
actuated directly and NOTHING latches:

  (1) the burner's gas feed runs through a DEAD-MAN'S PEDAL — a safety-yellow
      platform riding a vertical prismatic slide, sprung shut. The burner is
      lit ONLY WHILE the pedal is held depressed past the gas threshold
      (sustained, streak-gated); release the pedal and the spring closes the
      valve — the flame DIES. "Turning the stove on" is therefore not an
      actuation event at all: it is a placement that creates a lasting
      force-closure — PARK the cast-iron ballast weight on the pedal so its
      weight holds the gas open hands-free, forever.
  (2) the scene forces a SELECTION BY MASS: a look-alike foam block (same
      size and grip bar, 20x lighter) sits among the movables, and the spring
      provably beats it — the decoy cannot crack the pedal open, let alone
      hold it past the gas threshold.
  (3) then the frying pan — a pure payload here, never a tool — is seated on
      the burner plate.

Success is a live conjunction at judge time: pedal held past the threshold BY
WHATEVER REALLY RESTS ON IT + pan seated on the plate, all settled. A solver
needs a different plan than the seed (no fixture joint to turn) and a
different plan from any press-and-latch mechanism (holding, not clicking).

Assets are fully procedural (native PhysX box colliders; the pedal rides a
spawn-authored per-env prismatic Z joint whose LIMITS are the hard stops; a
scene-owned spring force in post_step gives it preload + return):
  - counter (KINEMATIC): 800 x 660 x 150 mm steel bench.
  - pedal housing (KINEMATIC, FIXED pose — it anchors the pedal joint and
    kinematic joint anchors are world-fixed): 140 mm base slab with four
    corner guide posts (visual).
  - pedal (DYNAMIC, on the Z joint): 110 mm safety-yellow platform, 60 g,
    spring-held at the top of a 22 mm travel; gas is OPEN while depth >=
    15 mm, sustained for 30 substeps (the dead-man streak).
  - burner (KINEMATIC, RANDOMIZED xy): 160 mm plinth + 140 mm cook plate,
    with a lamp on its south face (dark red <-> orange LIVE with the gas).
  - ballast (DYNAMIC): 70 mm cast-iron block with a pinch bar, 1.2 kg — the
    only thing in the scene heavy enough to hold the pedal down.
  - decoy (DYNAMIC): the SAME shape in pale foam, 60 g — the spring's preload
    alone beats it (it cannot even crack the pedal).
  - pan (DYNAMIC): 100 mm square dish + 140 mm stick handle, 500 g payload.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.15 * carry — ballast ever carried clearly off the bench (latched)
  0.25 * gas   — gas ever held open past the threshold (latched credit for
                 having found + operated the mechanism; the LIVE gas state is
                 what success needs)
  0.15 * seat  — pan ever seated on the burner plate, calm (latched)
  1.0 iff success() — gas held open LIVE (streak-gated) AND the pan seated
                 flat on the plate, settled and finite. Non-success cap 0.55.

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
                ang_damp: float = 0.0) -> None:
    from pxr import PhysxSchema, UsdPhysics

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:  # corpus-validated authoring: never author the attr False
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    if not kinematic:
        px.CreateSleepThresholdAttr(0.0)
        px.CreateStabilizationThresholdAttr(0.0)


def _spawn_counter(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC kitchen bench. Origin = deck centre at GROUND level; the solid
    counter block fills z in [0, deck_h]."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg  # unwrap: the spawner cfg carries the scene cfg in its `cfg` field
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 40.0, kinematic=True)
    _box(stage, f"{prim_path}/deck",
         (2 * cfg.counter_half_x, 2 * cfg.counter_half_y, cfg.deck_h),
         (0.0, 0.0, cfg.deck_h / 2), cfg.deck_color, cfg.contact_offset)
    return root


def _spawn_housing(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC pedal housing. Origin = slab base centre ON the deck top: a
    low base slab (collider) under the pedal's stroke, plus four corner guide
    posts (VISUAL ONLY — the prismatic joint is the real guide; visual-only
    posts can never prop a sloppily-placed weight). The housing anchors the
    pedal joint, so it is NEVER moved after spawn."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 5.0, kinematic=True)

    co = cfg.contact_offset
    _box(stage, f"{prim_path}/slab",
         (cfg.housing_s, cfg.housing_s, cfg.housing_h),
         (0.0, 0.0, cfg.housing_h / 2), cfg.housing_color, co)
    r = cfg.housing_s / 2 - cfg.post_s / 2
    for tag, sx, sy in (("post_ne", 1.0, 1.0), ("post_nw", -1.0, 1.0),
                        ("post_se", 1.0, -1.0), ("post_sw", -1.0, -1.0)):
        _box(stage, f"{prim_path}/{tag}", (cfg.post_s, cfg.post_s, cfg.post_h),
             (sx * r, sy * r, cfg.post_h / 2), cfg.post_color, None)
    return root


def _spawn_pedal(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC dead-man's pedal platform. Origin = platform centre. It rides a
    per-env prismatic Z joint (limits = the stops); a scene-owned spring in
    post_step holds it at the top of its travel (the gas valve shut)."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.pedal_mass, kinematic=False, lin_damp=cfg.pedal_damp)
    _box(stage, f"{prim_path}/plate", (cfg.pedal_s, cfg.pedal_s, cfg.pedal_t),
         (0.0, 0.0, 0.0), cfg.pedal_color, cfg.contact_offset)
    # hazard stripe (VISUAL ONLY) so the pedal reads as "the gas pedal"
    _box(stage, f"{prim_path}/stripe", (cfg.pedal_s * 0.9, cfg.pedal_s * 0.18, 0.002),
         (0.0, 0.0, cfg.pedal_t / 2 + 0.001), cfg.stripe_color, None)
    return root


def _spawn_burner(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC burner. Origin = plinth base centre ON the deck top: plinth +
    cook plate + an indicator lamp on the south face (VISUAL ONLY; post_step
    recolors it LIVE with the gas state: dark red shut, orange while open)."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 6.0, kinematic=True)

    co = cfg.contact_offset
    _box(stage, f"{prim_path}/plinth", (cfg.plinth_s, cfg.plinth_s, cfg.plinth_h),
         (0.0, 0.0, cfg.plinth_h / 2), cfg.plinth_color, co)
    _box(stage, f"{prim_path}/plate", (cfg.plate_s, cfg.plate_s, cfg.plate_h),
         (0.0, 0.0, cfg.plinth_h + cfg.plate_h / 2), cfg.plate_color, co)
    _box(stage, f"{prim_path}/lamp", (0.024, 0.006, 0.016),
         (0.0, -(cfg.plinth_s / 2 + 0.003), 0.020), cfg.lamp_off, None)
    return root


def _spawn_pan(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC frying pan (the payload). Origin = BASE plate centre: square
    dish (base + four walls) with a stick handle along local +x — the Franka
    pinch feature."""
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
         (cfg.handle_l, cfg.handle_wy, cfg.handle_wz),
         (w / 2 + cfg.handle_l / 2, 0.0, cfg.handle_z), cfg.handle_color, co)
    return root


def _block_geom(stage, prim_path: str, cfg: Any, body_color, bar_color) -> None:
    """Shared weight-block geometry: cubic body + a raised pinch bar (two feet
    + crossbar) on top — the canonical parallel-jaw grasp feature."""
    co = cfg.contact_offset
    _box(stage, f"{prim_path}/body", (cfg.block_s, cfg.block_s, cfg.block_h),
         (0.0, 0.0, 0.0), body_color, co)
    zf = cfg.block_h / 2 + cfg.bar_foot_h / 2
    for tag, sx in (("foot_e", 1.0), ("foot_w", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (cfg.bar_foot_s, cfg.bar_w, cfg.bar_foot_h),
             (sx * (cfg.bar_l / 2 - cfg.bar_foot_s / 2), 0.0, zf), bar_color, co)
    _box(stage, f"{prim_path}/bar", (cfg.bar_l, cfg.bar_w, cfg.bar_h),
         (0.0, 0.0, cfg.block_h / 2 + cfg.bar_foot_h + cfg.bar_h / 2), bar_color, co)


def _spawn_ballast(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC cast-iron ballast weight (1.2 kg). Origin = body centre. The
    ONLY object heavy enough to hold the dead-man's pedal down."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.ballast_mass, kinematic=False, lin_damp=0.2, ang_damp=0.3)
    _block_geom(stage, prim_path, cfg, cfg.ballast_color, cfg.ballast_bar_color)
    return root


def _spawn_decoy(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC foam decoy (60 g). Origin = body centre. Same shape and grip
    bar as the ballast — but the pedal spring's preload alone beats it."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.decoy_mass, kinematic=False, lin_damp=0.2, ang_damp=0.3)
    _block_geom(stage, prim_path, cfg, cfg.decoy_color, cfg.decoy_bar_color)
    return root


def _compound_spawner_cfg(kind: str, spawn_fn, scene_cfg: Any, mass: float,
                          kinematic: bool, lin_damp: float = 0.0,
                          ang_damp: float = 0.0) -> Any:
    """Build (once) and instantiate a RigidObjectSpawnerCfg subclass wrapping
    `spawn_fn`, carrying the scene cfg through a single `cfg` field. The
    spawner-cfg rigid_props are applied by the isaaclab clone wrapper AFTER the
    spawn fn runs, so they are the authoritative word (sleep stays OFF: sleeping
    GPU bodies freeze mid-settle and ignore velocity writes — corpus lesson)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if kind not in _SPAWNER_CACHE:

        @configclass
        class _Cfg(RigidObjectSpawnerCfg):
            func: Callable = clone(spawn_fn)
            cfg: Any = None

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
    return _SPAWNER_CACHE[kind](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=rp,
        cfg=scene_cfg,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DeadmanStoveSceneCfg(BaseCfg):
    """Config for `DeadmanStoveScene`. The honesty physics is asserted in
    __post_init__: the spring's preload alone beats the decoy (it can never
    crack the pedal), the ballast's weight beats the full spring stack with
    margin (it bottoms the pedal out), the pedal genuinely returns when
    unloaded, and the geometry (windows, slots, clearances) is load-bearing."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    pan_xy_tol: float = tunable(0.045)    # pan centre within this (radial) of the plate centre (m)
    pan_z_tol: float = tunable(0.012)     # pan bottom within this of the plate top (m)
    upright_max_deg: float = tunable(12.0)  # pan up-axis within this of world-up
    settle_speed: float = tunable(0.05)   # max pan |lin vel| when judging success (m/s)
    latch_speed: float = tunable(0.15)    # calm gate for the seat latch (m/s)
    gas_on_depth: float = tunable(0.015)  # pedal depth that opens the gas (m)
    gas_streak: int = tunable(30)         # consecutive substeps the depth must be held (0.25 s)
    lift_min: float = tunable(0.120)      # carry latch: ballast origin this far above the deck (m)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    burner_x_nom: float = tunable(-0.180)  # burner plinth centre, nominal (m)
    burner_y_nom: float = tunable(0.100)
    burner_jitter: float = tunable(0.035)  # burner xy jitter (+/- m)
    slot_dx: float = tunable(0.240)        # three spawn slots at x in {-dx, 0, +dx}
    slot_y: float = tunable(-0.200)
    slot_x_jitter: float = tunable(0.030)
    slot_y_jitter: float = tunable(0.025)
    pan_yaw_center_deg: float = tunable(-90.0)  # handle nominally points south
    pan_yaw_half_deg: float = tunable(35.0)     # free yaw about that (+/- deg)

    # --- info: counter ---------------------------------------------------------------------------
    deck_h: float = info(0.150)
    counter_half_x: float = info(0.400)
    counter_half_y: float = info(0.330)
    # --- info: pedal housing + pedal -------------------------------------------------------------
    pedal_x: float = info(0.240)          # FIXED (joint anchor body — never randomized)
    pedal_y: float = info(0.100)
    housing_s: float = info(0.140)        # base slab width
    housing_h: float = info(0.012)        # base slab height (under the stroke)
    post_s: float = info(0.012)           # corner guide posts (VISUAL ONLY)
    post_h: float = info(0.065)
    pedal_s: float = info(0.110)          # pedal platform width
    pedal_t: float = info(0.018)          # pedal platform thickness
    pedal_mass: float = info(0.060)
    pedal_damp: float = info(0.5)
    travel: float = info(0.022)           # pedal stroke (joint limit span)
    stop_gap: float = info(0.002)         # full-press pedal bottom above the slab (limit stop)
    spring_k: float = info(160.0)         # spring rate (N/m)
    spring_preload: float = info(2.0)     # upward preload at zero depth (N)
    spring_c: float = info(6.0)           # spring damping (N s/m)
    # --- info: burner ----------------------------------------------------------------------------
    plinth_s: float = info(0.160)
    plinth_h: float = info(0.030)
    plate_s: float = info(0.140)
    plate_h: float = info(0.008)
    # --- info: pan (payload) ---------------------------------------------------------------------
    pan_w: float = info(0.100)
    pan_base_t: float = info(0.012)
    pan_wall_t: float = info(0.008)
    pan_wall_h: float = info(0.024)
    handle_l: float = info(0.140)
    handle_wy: float = info(0.020)
    handle_wz: float = info(0.014)
    handle_z: float = info(0.012)         # handle centre above the pan ORIGIN (base centre)
    pan_mass: float = info(0.500)
    # --- info: weight blocks (ballast + decoy share the shape) -----------------------------------
    block_s: float = info(0.070)          # cubic body width
    block_h: float = info(0.050)          # cubic body height
    bar_l: float = info(0.060)            # pinch bar length
    bar_w: float = info(0.018)            # pinch bar width (the jaw span)
    bar_h: float = info(0.016)            # pinch bar height
    bar_foot_h: float = info(0.014)       # feet raising the bar (finger clearance)
    bar_foot_s: float = info(0.012)
    ballast_mass: float = info(1.200)
    decoy_mass: float = info(0.060)
    # --- info: misc ------------------------------------------------------------------------------
    contact_offset: float = info(0.0015)
    deck_color: tuple = info((0.42, 0.44, 0.48))
    housing_color: tuple = info((0.28, 0.29, 0.34))
    post_color: tuple = info((0.20, 0.21, 0.25))
    pedal_color: tuple = info((0.95, 0.75, 0.10))
    stripe_color: tuple = info((0.10, 0.10, 0.10))
    plinth_color: tuple = info((0.20, 0.21, 0.24))
    plate_color: tuple = info((0.10, 0.10, 0.12))
    lamp_off: tuple = info((0.16, 0.05, 0.04))
    lamp_on: tuple = info((1.00, 0.45, 0.05))
    pan_color: tuple = info((0.22, 0.22, 0.25))
    handle_color: tuple = info((0.05, 0.05, 0.05))
    ballast_color: tuple = info((0.15, 0.17, 0.21))
    ballast_bar_color: tuple = info((0.30, 0.33, 0.38))
    decoy_color: tuple = info((0.92, 0.85, 0.45))
    decoy_bar_color: tuple = info((0.80, 0.72, 0.35))
    # rubric weights (0.15 + 0.25 + 0.15 = 0.55 = the non-success cap)
    w_carry: float = info(0.15)
    w_gas: float = info(0.25)
    w_seat: float = info(0.15)

    # ----- derived geometry ----------------------------------------------------------------------
    @property
    def pedal_rest_z(self) -> float:
        """Pedal platform CENTRE above the deck at rest (top of travel)."""
        return self.housing_h + self.stop_gap + self.travel + self.pedal_t / 2

    @property
    def pedal_rest_w(self) -> float:
        """World z of the pedal platform centre at rest."""
        return self.deck_h + self.pedal_rest_z

    @property
    def pedal_top_rest_w(self) -> float:
        """World z of the pedal top face at rest."""
        return self.pedal_rest_w + self.pedal_t / 2

    @property
    def plate_top_dz(self) -> float:
        """Cook-plate top above the burner ORIGIN (the plinth base)."""
        return self.plinth_h + self.plate_h

    @property
    def pan_bottom_dz(self) -> float:
        """Pan origin (base centre) above the pan's bottom face."""
        return self.pan_base_t / 2

    @property
    def pan_spawn_z(self) -> float:
        return self.deck_h + self.pan_bottom_dz + 0.002

    @property
    def block_spawn_z(self) -> float:
        return self.deck_h + self.block_h / 2 + 0.002

    @property
    def handle_tip_reach(self) -> float:
        """Pan-origin -> handle-tip distance along local +x."""
        return self.pan_w / 2 + self.handle_l

    def __post_init__(self) -> None:
        # Honesty-by-construction asserts (the physical claims the task rests on).
        g = 9.81
        assert self.spring_preload > 1.3 * self.pedal_mass * g, \
            "the spring must hold the unloaded pedal at the TOP of its travel"
        assert self.spring_preload > 1.5 * (self.pedal_mass + self.decoy_mass) * g, \
            "the preload ALONE must beat pedal+decoy (the decoy can never crack the pedal)"
        assert 0.6 * (self.pedal_mass + self.ballast_mass) * g > \
            self.spring_preload + self.spring_k * self.travel, \
            "a fraction of the ballast's weight must bottom the pedal out (full stroke)"
        assert (self.pedal_mass + self.pan_mass) * g > \
            self.spring_preload + self.spring_k * (self.gas_on_depth + 0.004), \
            "the PAN parked on the pedal must also open the gas (the cross-arrangement " \
            "smoke probe is physically real, and mass — not identity — is the mechanism)"
        assert self.gas_on_depth <= self.travel - 0.004, \
            "the gas threshold must sit safely inside the stroke"
        assert self.pedal_rest_z - self.pedal_t / 2 - self.travel >= \
            self.housing_h + 0.0015 - 1e-9, \
            "full travel must stop on the JOINT LIMIT, above slab contact"
        assert self.pedal_s >= self.block_s + 0.030, \
            "the pedal platform must cover a sloppily-placed block with margin"
        assert self.housing_s / 2 - self.post_s > self.pedal_s / 2, \
            "the visual guide posts must sit outside the pedal platform"
        assert self.plate_s / 2 >= self.pan_xy_tol + 0.020, \
            "the plate must cover the judged xy window with margin"
        assert self.lift_min > self.pedal_rest_z + self.pedal_t / 2 \
            + self.block_h / 2 + 0.030, \
            "the carry latch must be unreachable by any on-pedal placement"
        assert self.lift_min > self.plate_top_dz + self.block_h / 2 + 0.030, \
            "the carry latch must be unreachable by an on-plate placement"
        # spawn slots: clear of the burner and the pedal housing at any jitter
        diag = self.block_s * math.sqrt(2.0) / 2
        pan_diag = self.pan_w * math.sqrt(2.0) / 2
        reach = max(diag, pan_diag)
        assert self.slot_y + self.slot_y_jitter + reach \
            < self.burner_y_nom - self.burner_jitter - self.plinth_s / 2 - 0.004, \
            "the spawn slots must be clear of the burner at any jitter"
        assert self.slot_y + self.slot_y_jitter + reach \
            < self.pedal_y - self.housing_s / 2 - 0.004, \
            "the spawn slots must be clear of the pedal housing"
        # neighbour slots: the pan handle's east-west reach at any cone yaw plus
        # both jitters stays short of the neighbour block's nearest extent
        assert self.handle_tip_reach * math.sin(math.radians(self.pan_yaw_half_deg)) \
            + self.slot_x_jitter \
            < self.slot_dx - diag - self.slot_x_jitter - 0.010, \
            "the pan handle must not reach a neighbour slot at any spawn yaw"
        assert math.sin(math.radians(self.pan_yaw_center_deg + self.pan_yaw_half_deg)) \
            < -0.5, \
            "the pan handle must stay pointed into the south half-plane at any spawn yaw"
        assert 2 * self.slot_x_jitter + diag + pan_diag + 0.010 < self.slot_dx, \
            "adjacent slots must never overlap at any jitter"
        # everything fits on the counter
        assert self.counter_half_x > self.slot_dx + self.slot_x_jitter + pan_diag + 0.02
        assert self.counter_half_x > self.pedal_x + self.housing_s / 2 + 0.02
        assert self.counter_half_x > abs(self.burner_x_nom) + self.burner_jitter \
            + self.plinth_s / 2 + 0.02
        assert self.counter_half_y > abs(self.slot_y) + self.slot_y_jitter + pan_diag + 0.02
        assert self.counter_half_y > self.burner_y_nom + self.burner_jitter \
            + self.plinth_s / 2 + 0.02
        # grasp feature: the pinch bar fits a parallel jaw with finger room
        assert self.bar_w <= 0.040 and self.bar_h + self.bar_foot_h >= 0.020, \
            "the pinch bar must be a real parallel-jaw grasp feature"


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("deadman_stove")
class DeadmanStoveScene(BaseScene):
    cfg: DeadmanStoveSceneCfg

    def __init__(self, cfg: DeadmanStoveSceneCfg | None = None) -> None:
        super().__init__(cfg or DeadmanStoveSceneCfg())

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
            "counter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Counter",
                spawn=_compound_spawner_cfg("counter", _spawn_counter, c, 40.0,
                                            kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            # Housing authored at its FIXED pose; the bind-time pedal joint
            # anchors here and the housing is NEVER moved (kinematic joint
            # anchors are world-fixed).
            "housing": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Housing",
                spawn=_compound_spawner_cfg("housing", _spawn_housing, c, 5.0,
                                            kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pedal_x, c.pedal_y, c.deck_h)),
            ),
            "pedal": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pedal",
                spawn=_compound_spawner_cfg("pedal", _spawn_pedal, c,
                                            c.pedal_mass, kinematic=False,
                                            lin_damp=c.pedal_damp),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pedal_x, c.pedal_y, c.pedal_rest_w)),
            ),
            "burner": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Burner",
                spawn=_compound_spawner_cfg("burner", _spawn_burner, c, 6.0,
                                            kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.burner_x_nom, c.burner_y_nom, c.deck_h)),
            ),
            "pan": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pan",
                spawn=_compound_spawner_cfg("pan", _spawn_pan, c, c.pan_mass,
                                            kinematic=False, lin_damp=0.2,
                                            ang_damp=0.3),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-c.slot_dx, c.slot_y, c.pan_spawn_z)),
            ),
            "ballast": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ballast",
                spawn=_compound_spawner_cfg("ballast", _spawn_ballast, c,
                                            c.ballast_mass, kinematic=False,
                                            lin_damp=0.2, ang_damp=0.3),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, c.slot_y, c.block_spawn_z)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=_compound_spawner_cfg("decoy", _spawn_decoy, c,
                                            c.decoy_mass, kinematic=False,
                                            lin_damp=0.2, ang_damp=0.3),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_dx, c.slot_y, c.block_spawn_z)),
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
        self.counter: RigidObject = env.iscene["counter"]
        self.housing: RigidObject = env.iscene["housing"]
        self.pedal: RigidObject = env.iscene["pedal"]
        self.burner: RigidObject = env.iscene["burner"]
        self.pan: RigidObject = env.iscene["pan"]
        self.ballast: RigidObject = env.iscene["ballast"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # obj_slot[e, j]: which of the three spawn slots object j sits on
        # (j: 0=pan, 1=ballast, 2=decoy) — the permutation randomization flag.
        self.obj_slot = torch.tensor([0, 1, 2], device=dev).expand(n, 3).clone()
        # latches (partial credit survives transients; success is judged live)
        self._carry = torch.zeros(n, dtype=torch.bool, device=dev)
        self._gas = torch.zeros(n, dtype=torch.bool, device=dev)
        self._seat = torch.zeros(n, dtype=torch.bool, device=dev)
        self._streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._lamp_shown: list[bool | None] = [None] * n
        self._lamp_prims = None  # lazily grabbed in _refresh_lamps
        self._author_joints()

    def _author_joints(self) -> None:
        """Per-env pedal joint, authored ONCE at bind time against the AUTHORED
        poses: a vertical prismatic joint anchored on the KINEMATIC housing
        (its joint anchors are world-fixed, so the housing must never move).
        The joint LIMITS are the hard stops — upper (0) is the spring-held
        rest (gas shut), lower (-travel) is the full-press stop. Joint
        collision filtering only disables the housing<->pedal pair, so the
        weights still contact both the pedal and the housing slab."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        c = self.cfg
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/pedal_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Housing"])
            j.CreateBody1Rel().SetTargets([f"{base}/Pedal"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.pedal_rest_z)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-float(c.travel))
            j.CreateUpperLimitAttr(0.0)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: counter and housing re-asserted at their FIXED poses
        (the housing anchors the pedal joint — kinematic joint anchors are
        world-fixed), pedal at the top of its travel (gas shut), burner
        teleported to a random xy, and the pan / ballast / decoy dealt onto
        the three front slots by a RANDOM PERMUTATION with per-object jitter
        and yaw; latches and lamp cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, xy, z, quat=None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3:7] = quat if quat is not None else torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=dev).expand(m, 4)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        zeros = torch.zeros(m, device=dev)
        write(self.counter, torch.zeros(m, 2, device=dev), zeros)
        pedal_xy = torch.tensor([c.pedal_x, c.pedal_y], device=dev).expand(m, 2)
        write(self.housing, pedal_xy, c.deck_h)
        write(self.pedal, pedal_xy, c.pedal_rest_w)

        # --- burner: random xy on the back-west area ---
        bx = c.burner_x_nom + (torch.rand(m, device=dev) * 2 - 1) * c.burner_jitter
        by = c.burner_y_nom + (torch.rand(m, device=dev) * 2 - 1) * c.burner_jitter
        write(self.burner, torch.stack([bx, by], dim=-1), c.deck_h)

        # --- pan / ballast / decoy: random permutation over the three slots
        # (torch.rand argsort — the FIRST torch.randint after manual_seed is
        # near-constant across seeds) + per-object jitter ---
        perm = torch.rand(m, 3, device=dev).argsort(dim=1)
        self.obj_slot[env_ids] = perm
        slot_x = torch.tensor([-c.slot_dx, 0.0, c.slot_dx], device=dev)

        def slot_xy(j: int) -> torch.Tensor:
            x = slot_x[perm[:, j]] \
                + (torch.rand(m, device=dev) * 2 - 1) * c.slot_x_jitter
            y = c.slot_y + (torch.rand(m, device=dev) * 2 - 1) * c.slot_y_jitter
            return torch.stack([x, y], dim=-1)

        pyaw = math.radians(c.pan_yaw_center_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pan_yaw_half_deg)
        write(self.pan, slot_xy(0), c.pan_spawn_z, _qz(pyaw))
        byaw = torch.rand(m, device=dev) * 2 * math.pi
        write(self.ballast, slot_xy(1), c.block_spawn_z, _qz(byaw))
        dyaw = torch.rand(m, device=dev) * 2 * math.pi
        write(self.decoy, slot_xy(2), c.block_spawn_z, _qz(dyaw))

        # --- clear latches + mechanism + lamp ---
        for latch in (self._carry, self._gas, self._seat):
            latch[env_ids] = False
        self._streak[env_ids] = 0
        for e in env_ids.tolist():
            self._lamp_shown[e] = None  # force a lamp refresh next post_step

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "counter": self.counter.data.root_state_w[env_ids].clone(),
            "housing": self.housing.data.root_state_w[env_ids].clone(),
            "pedal": self.pedal.data.root_state_w[env_ids].clone(),
            "burner": self.burner.data.root_state_w[env_ids].clone(),
            "pan": self.pan.data.root_state_w[env_ids].clone(),
            "ballast": self.ballast.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "obj_slot": self.obj_slot[env_ids].clone(),
            "carry": self._carry[env_ids].clone(),
            "gas": self._gas[env_ids].clone(),
            "seat": self._seat[env_ids].clone(),
            "streak": self._streak[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for name in ("counter", "housing", "pedal", "burner", "pan", "ballast",
                     "decoy"):
            getattr(self, name).write_root_state_to_sim(state[name], env_ids)
        self.obj_slot[env_ids] = state["obj_slot"]
        self._carry[env_ids] = state["carry"]
        self._gas[env_ids] = state["gas"]
        self._seat[env_ids] = state["seat"]
        self._streak[env_ids] = state["streak"]
        for e in env_ids.tolist():
            self._lamp_shown[e] = None

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A steel KITCHEN BENCH ({2 * c.counter_half_x * 100:.0f} x "
            f"{2 * c.counter_half_y * 100:.0f} cm, {c.deck_h * 100:.0f} cm tall). "
            f"On its back-west area stands the BURNER: a dark plinth topped by a "
            f"{c.plate_s * 100:.0f} cm square cook plate, with an indicator lamp on "
            f"its front face. On the back-east side sits the burner's gas feed: a "
            f"DEAD-MAN'S PEDAL — a safety-yellow {c.pedal_s * 1000:.0f} mm platform "
            f"with a black hazard stripe, riding a sprung vertical slide between "
            f"four grey guide posts. The gas is OPEN only while the pedal is held "
            f"at least {c.gas_on_depth * 1000:.0f} mm down its "
            f"{c.travel * 1000:.0f} mm stroke: while it is held down the lamp "
            f"glows orange and the burner is lit; the moment the load comes off, "
            f"the spring shuts the valve and the lamp goes dark red — NOTHING "
            f"LATCHES. On the three front slots (order randomized per episode, as "
            f"are the burner position and every spawn pose) lie: a square FRYING "
            f"PAN ({c.pan_w * 1000:.0f} mm dish, {c.handle_l * 1000:.0f} mm stick "
            f"handle), a dark CAST-IRON WEIGHT ({c.block_s * 1000:.0f} mm block "
            f"with a raised pinch bar, heavy), and a PALE-YELLOW FOAM BLOCK — the "
            f"exact same shape and pinch bar, but almost weightless.\n"
            f"Goal: turn the burner on and put the frying pan on it, hands-free. "
            f"Park the cast-iron weight squarely on the pedal so its weight holds "
            f"the gas open (the lamp stays orange), then set the pan flat on the "
            f"cook plate. The foam block is a decoy: the pedal spring beats it, so "
            f"it cannot hold the gas open. Holding the pedal down yourself and "
            f"letting go does not help — the flame dies as soon as the pedal "
            f"rises. Success: the pedal held past the gas threshold by what rests "
            f"on it AND the pan seated flat and centred on the cook plate, "
            f"everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Park the heavy cast-iron weight on the stove's spring-loaded gas "
            "pedal so the pedal stays pressed down, then set the frying pan on "
            "the burner plate. The foam block is too light to hold the pedal, "
            "and the burner is only lit while the pedal is held down."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _up_w(self, body) -> torch.Tensor:
        """(N,3) the body's local +z in world."""
        from isaaclab.utils.math import quat_apply

        q = body.data.root_quat_w
        ez = torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], 3)
        return quat_apply(q, ez)

    def _pan_bottom_z(self) -> torch.Tensor:
        """(N,) world z of the pan base's bottom face centre (exact when upright)."""
        return self.pan.data.root_pos_w[:, 2] - self._up_w(self.pan)[:, 2] \
            * self.cfg.pan_bottom_dz

    def pedal_depth(self) -> torch.Tensor:
        """(N,) pedal press depth below its rest stop (m, >= 0)."""
        rest = self.env_origins[:, 2] + self.cfg.pedal_rest_w
        return (rest - self.pedal.data.root_pos_w[:, 2]).clamp(min=0.0)

    def gas_open(self) -> torch.Tensor:
        """(N,) bool, instantaneous: pedal past the gas threshold right now."""
        return self.pedal_depth() >= self.cfg.gas_on_depth

    def lit(self) -> torch.Tensor:
        """(N,) bool, LIVE dead-man state: the pedal has been held past the gas
        threshold for `gas_streak` consecutive substeps (nothing latches — the
        streak counter zeroes the moment the pedal rises)."""
        return self._streak >= self.cfg.gas_streak

    def pan_on_burner(self) -> torch.Tensor:
        """(N,) bool, geometric, BURNER-RELATIVE: pan centre within the radial
        xy window of the plate centre, pan bottom at the plate top (z band),
        pan upright. The burner is kinematic and never rotates."""
        c = self.cfg
        d = self.pan.data.root_pos_w[:, :2] - self.burner.data.root_pos_w[:, :2]
        xy_ok = d.norm(dim=-1) <= c.pan_xy_tol
        top = self.burner.data.root_pos_w[:, 2] + c.plate_top_dz
        dz_ok = (self._pan_bottom_z() - top).abs() <= c.pan_z_tol
        upright = self._up_w(self.pan)[:, 2] > math.cos(math.radians(c.upright_max_deg))
        return xy_ok & dz_ok & upright

    def settled(self) -> torch.Tensor:
        """(N,) bool: pan |lin vel| below `settle_speed`."""
        return self.pan.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def _update_latches(self) -> None:
        """Carry + gas + seat credit latches (the streak itself advances only
        in post_step, once per physics substep)."""
        c = self.cfg
        bal_h = self.ballast.data.root_pos_w[:, 2] - self.env_origins[:, 2] - c.deck_h
        self._carry |= bal_h > c.lift_min
        self._gas |= self.lit()
        calm_p = self.pan.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        self._seat |= self.pan_on_burner() & calm_p

    def _refresh_lamps(self) -> None:
        from pxr import Gf

        if self._lamp_prims is None:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            self._lamp_prims = [
                UsdGeom.Cube(stage.GetPrimAtPath(f"/World/envs/env_{i}/Burner/lamp"))
                for i in range(self.env.num_envs)]
        c = self.cfg
        lit_now = self.lit()
        for e in range(self.env.num_envs):
            key = bool(lit_now[e])
            if key != self._lamp_shown[e]:
                self._lamp_shown[e] = key
                col = c.lamp_on if key else c.lamp_off
                self._lamp_prims[e].GetDisplayColorAttr().Set([Gf.Vec3f(*col)])

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Once per physics substep: advance the dead-man streak (LIVE gas
        state), update the credit latches, refresh the lamps, and apply the
        pedal's return spring (post_step OWNS the pedal force buffer)."""
        n, dev = self.env.num_envs, self.env.device
        c = self.cfg

        depth = self.pedal_depth()
        self._streak = torch.where(depth >= c.gas_on_depth, self._streak + 1,
                                   torch.zeros_like(self._streak))
        self._update_latches()
        self._refresh_lamps()

        # spring: preload + rate, damped; the pedal rides a vertical prismatic
        # joint and NEVER rotates, so the external-force frame quirk is moot.
        v_z = self.pedal.data.root_lin_vel_w[:, 2]
        f_z = c.spring_preload + c.spring_k * depth - c.spring_c * v_z
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        self.pedal.set_external_force_and_torque(
            f_z.view(n, 1, 1) * ez.view(n, 1, 3), torch.zeros(n, 1, 3, device=dev))

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: gas held open LIVE (the streak-gated dead-man state — a
        real sustained load rests on the pedal RIGHT NOW) AND the pan seated
        flat on the cook plate, settled and finite. Every clause is a live
        physical outcome; nothing here reads a latch."""
        self._update_latches()
        finite = torch.isfinite(self.pan.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.pedal.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.ballast.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.decoy.data.root_pos_w).all(dim=-1)
        return self.lit() & self.pan_on_burner() & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15*ballast-carried + 0.25*gas-ever-held-open +
        0.15*pan-seated-on-plate (latched; ~0 for doing nothing), capped at
        0.55 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_carry * self._carry.float()
                + c.w_gas * self._gas.float()
                + c.w_seat * self._seat.float()).clamp(max=0.55)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="deadman_stove", robot="null"))
