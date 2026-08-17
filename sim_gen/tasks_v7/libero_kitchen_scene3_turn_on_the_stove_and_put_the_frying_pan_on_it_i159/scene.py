"""PiezoStoveScene — light the bench burner with its recessed PIEZO IGNITER (a
spring-return striker plunger sunk at the bottom of a narrow guard shaft that
only the frying pan's stick handle can reach), then set the pan on the burner
plate (sim_gen task
`libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it_i159`).

Derived from libero_90/libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it,
but STRATEGICALLY different. The seed is knob-then-place: rotate a stove knob
past a JOINT-ANGLE threshold (`joint_pos > 0.5`) — a fixture actuation the
rubric reads directly from a joint and that must be HELD past the threshold —
then set the frypan on the always-available cook plate. Here there is NO KNOB
and no joint-angle clause anywhere in the rubric:

  (1) ignition is a MOMENTARY TOOL-MEDIATED PRESS. The igniter's striker cap
      sits 86 mm down a 48 x 48 mm guard shaft — too deep for fingertips, too
      narrow for the kettle (70 mm body; its 24 mm lid knob reaches only
      24 mm) — so the ONLY way to press it is to use the task's own payload as
      the tool: hold the frying pan handle-down and dip its 150 mm stick
      handle into the shaft until the pan's weight drives the spring-loaded
      striker >= 8 mm down. The click LATCHES the burner lit (the burner lamp
      turns orange and stays on); the spring returns the striker, and nothing
      needs to be held.
  (2) then the very same object flips role from tool back to payload: seat
      the pan flat on the burner plate.

So the pan is BOTH the actuation tool and the final payload — the actuation is
a transient contact event judged by latched mechanism state, not a fixture
joint angle held at success time.

Assets are fully procedural (native PhysX box colliders; the striker rides a
spawn-authored per-env prismatic Z joint whose LIMITS are the hard stops; a
scene-owned spring force in post_step gives it preload + return):
  - counter (KINEMATIC): 780 x 620 x 150 mm steel bench top.
  - igniter tower (KINEMATIC, FIXED pose — it anchors the striker joint and
    kinematic joint anchors are world-fixed): 90 mm square column, 150 mm
    tall, with a 48 x 48 mm guard shaft sunk 110 mm down its middle to the
    striker chamber.
  - striker (DYNAMIC, on the Z joint): 40 mm square cap, 30 g, spring-held at
    the top of a 12 mm travel; pressing >= 8 mm for 3 substeps = ignition.
  - burner (KINEMATIC, RANDOMIZED xy): 160 mm plinth + 140 mm cook plate with
    a lamp on its south face (dark red -> orange when lit; visual only).
  - pan (DYNAMIC, free): 110 mm square dish + 150 mm stick handle, 420 g.
  - kettle (DYNAMIC, free): 70 mm body + 24 mm lid knob, 500 g DISTRACTOR —
    provably useless: it cannot enter the shaft and its knob is too short.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.15 * lift — pan ever carried clearly off the bench (latched)
  0.25 * lit  — igniter ever pressed past the click depth (latched mechanism
                state; the lamp shows it)
  0.15 * seat — pan ever seated on the burner plate, calm (latched)
  1.0 iff success() — burner lit AND the pan seated flat on the plate, settled
                and finite. Non-success cap 0.55.

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


def _spawn_tower(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC igniter tower. Origin = column base centre ON the deck top:
    a base block (the striker chamber floor) and four guard walls forming the
    square guard shaft down the middle. The tower anchors the striker joint,
    so it is NEVER moved after spawn (kinematic joint anchors are world-fixed)."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 5.0, kinematic=True)

    co = cfg.contact_offset
    s, sh = cfg.tower_base_s, cfg.shaft_w
    t = (s - sh) / 2
    bh, wh = cfg.tower_base_h, cfg.tower_h - cfg.tower_base_h
    _box(stage, f"{prim_path}/base", (s, s, bh), (0.0, 0.0, bh / 2),
         cfg.tower_color, co)
    zc = bh + wh / 2
    for tag, sx in (("wall_e", 1.0), ("wall_w", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (t, s, wh),
             (sx * (sh + t) / 2, 0.0, zc), cfg.tower_color, co)
    for tag, sy in (("wall_n", 1.0), ("wall_s", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (sh, t, wh),
             (0.0, sy * (sh + t) / 2, zc), cfg.tower_color, co)
    # bright mouth trim (VISUAL ONLY) so the shaft reads as "the igniter port"
    _box(stage, f"{prim_path}/trim", (s + 0.004, s + 0.004, 0.004),
         (0.0, 0.0, cfg.tower_h - 0.002), cfg.trim_color, None)
    return root


def _spawn_striker(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC striker cap (the piezo button). Origin = cap centre. It rides a
    per-env prismatic Z joint (limits = the stops); a scene-owned spring in
    post_step holds it at the top of its travel."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.striker_mass, kinematic=False, lin_damp=cfg.striker_damp)
    _box(stage, f"{prim_path}/cap", (cfg.cap_s, cfg.cap_s, cfg.cap_t),
         (0.0, 0.0, 0.0), cfg.striker_color, cfg.contact_offset)
    return root


def _spawn_burner(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC burner. Origin = plinth base centre ON the deck top: plinth +
    cook plate + an indicator lamp on the south face (VISUAL ONLY; post_step
    recolors it dark-red -> orange when the burner is lit)."""
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
    """DYNAMIC frying pan. Origin = BASE plate centre: square dish (base + four
    walls) with a stick handle along local +x — the handle is BOTH the Franka
    pinch feature and the only tool that reaches the recessed striker."""
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


def _spawn_kettle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC kettle (distractor). Origin = body centre: cubic body + lid
    knob. Provably useless at the igniter: the body is wider than the guard
    shaft and the knob is far shorter than the striker recess."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.kettle_mass, kinematic=False, lin_damp=0.2, ang_damp=0.3)

    co = cfg.contact_offset
    _box(stage, f"{prim_path}/body", (cfg.kettle_s, cfg.kettle_s, cfg.kettle_h),
         (0.0, 0.0, 0.0), cfg.kettle_color, co)
    _box(stage, f"{prim_path}/knob", (cfg.knob_s, cfg.knob_s, cfg.knob_h),
         (0.0, 0.0, cfg.kettle_h / 2 + cfg.knob_h / 2), cfg.knob_color, co)
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
class PiezoStoveSceneCfg(BaseCfg):
    """Config for `PiezoStoveScene`. The honesty geometry is asserted in
    __post_init__: the kettle physically cannot press the striker (too wide
    for the shaft; knob too short), the pan handle reaches the click depth
    with margin while the dish stays above the shaft mouth, the pan's weight
    beats the spring, and the spring beats the striker's weight (true return)."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    pan_xy_tol: float = tunable(0.040)    # pan centre within this (radial) of the plate centre (m)
    pan_z_tol: float = tunable(0.010)     # pan bottom within this of the plate top (m)
    upright_max_deg: float = tunable(12.0)  # pan up-axis within this of world-up
    settle_speed: float = tunable(0.05)   # max pan |lin vel| when judging success (m/s)
    latch_speed: float = tunable(0.15)    # calm gate for the seat latch (m/s)
    press_depth: float = tunable(0.008)   # striker press depth that counts as the click (m)
    press_streak: int = tunable(3)        # consecutive substeps past click depth to ignite
    lift_min: float = tunable(0.120)      # lift latch: pan origin at least this above the deck (m)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    burner_x_nom: float = tunable(-0.130)  # burner plinth centre, nominal (m)
    burner_y_nom: float = tunable(0.080)
    burner_jitter: float = tunable(0.035)  # burner xy jitter (+/- m)
    slot_x: float = tunable(0.150)         # pan/kettle spawn slots at (+/- slot_x, slot_y)
    slot_y: float = tunable(-0.170)
    slot_x_jitter: float = tunable(0.035)
    slot_y_jitter: float = tunable(0.030)
    pan_yaw_center_deg: float = tunable(-90.0)  # handle nominally points south
    pan_yaw_half_deg: float = tunable(40.0)     # free yaw about that (+/- deg)

    # --- info: counter ---------------------------------------------------------------------------
    deck_h: float = info(0.150)
    counter_half_x: float = info(0.390)
    counter_half_y: float = info(0.310)
    # --- info: igniter tower + striker -----------------------------------------------------------
    tower_x: float = info(0.260)          # FIXED (joint anchor body — never randomized)
    tower_y: float = info(0.040)
    tower_base_s: float = info(0.090)     # column outer width
    tower_base_h: float = info(0.040)     # striker chamber floor (local z, above deck)
    tower_h: float = info(0.150)          # column top (local z; shaft mouth height)
    shaft_w: float = info(0.048)          # guard shaft clear width
    cap_s: float = info(0.040)            # striker cap width
    cap_t: float = info(0.012)            # striker cap thickness
    striker_rest_z: float = info(0.058)   # cap CENTRE local z at rest (top of travel)
    travel: float = info(0.012)           # striker stroke (joint limit span)
    striker_mass: float = info(0.030)
    striker_damp: float = info(2.0)
    spring_k: float = info(80.0)          # spring rate (N/m)
    spring_preload: float = info(0.8)     # upward preload at zero depth (N)
    spring_c: float = info(0.8)           # spring damping (N s/m)
    # --- info: burner ----------------------------------------------------------------------------
    plinth_s: float = info(0.160)
    plinth_h: float = info(0.030)
    plate_s: float = info(0.140)
    plate_h: float = info(0.008)
    # --- info: pan -------------------------------------------------------------------------------
    pan_w: float = info(0.110)
    pan_base_t: float = info(0.012)
    pan_wall_t: float = info(0.008)
    pan_wall_h: float = info(0.026)
    handle_l: float = info(0.150)
    handle_wy: float = info(0.020)
    handle_wz: float = info(0.014)
    handle_z: float = info(0.012)         # handle centre above the pan ORIGIN (base centre)
    pan_mass: float = info(0.420)
    # --- info: kettle (distractor) ---------------------------------------------------------------
    kettle_s: float = info(0.070)
    kettle_h: float = info(0.090)
    knob_s: float = info(0.024)
    knob_h: float = info(0.024)
    kettle_mass: float = info(0.500)
    # --- info: misc ------------------------------------------------------------------------------
    contact_offset: float = info(0.0015)
    deck_color: tuple = info((0.42, 0.44, 0.48))
    tower_color: tuple = info((0.30, 0.31, 0.36))
    trim_color: tuple = info((0.95, 0.60, 0.10))
    striker_color: tuple = info((0.90, 0.20, 0.15))
    plinth_color: tuple = info((0.20, 0.21, 0.24))
    plate_color: tuple = info((0.10, 0.10, 0.12))
    lamp_off: tuple = info((0.16, 0.05, 0.04))
    lamp_on: tuple = info((1.00, 0.45, 0.05))
    pan_color: tuple = info((0.22, 0.22, 0.25))
    handle_color: tuple = info((0.05, 0.05, 0.05))
    kettle_color: tuple = info((0.55, 0.57, 0.62))
    knob_color: tuple = info((0.12, 0.12, 0.14))
    # rubric weights (0.15 + 0.25 + 0.15 = 0.55 = the non-success cap)
    w_lift: float = info(0.15)
    w_lit: float = info(0.25)
    w_seat: float = info(0.15)

    # ----- derived geometry ----------------------------------------------------------------------
    @property
    def tower_top_w(self) -> float:
        """World z of the shaft mouth."""
        return self.deck_h + self.tower_h

    @property
    def striker_rest_w(self) -> float:
        """World z of the striker cap CENTRE at rest."""
        return self.deck_h + self.striker_rest_z

    @property
    def recess_rest(self) -> float:
        """Cap TOP below the shaft mouth at rest (the reach-denial depth)."""
        return self.tower_h - (self.striker_rest_z + self.cap_t / 2)

    @property
    def recess_pressed(self) -> float:
        """Cap TOP below the shaft mouth at FULL press (what a tool must reach)."""
        return self.recess_rest + self.travel

    @property
    def plate_top_w(self) -> float:
        """World z of the cook plate top face."""
        return self.deck_h + self.plinth_h + self.plate_h

    @property
    def handle_tip_reach(self) -> float:
        """Pan-origin -> handle-tip distance along local +x."""
        return self.pan_w / 2 + self.handle_l

    @property
    def dip_reach(self) -> float:
        """How far the handle tip reaches BELOW the shaft mouth when the pan is
        held handle-down (the wide dish stops at the mouth: its lowest point is
        pan_w/2 from the origin in that pose)."""
        return self.handle_tip_reach - self.pan_w / 2

    @property
    def pan_bottom_dz(self) -> float:
        """Pan origin (base centre) above the pan's bottom face."""
        return self.pan_base_t / 2

    @property
    def pan_spawn_z(self) -> float:
        return self.deck_h + self.pan_bottom_dz + 0.002

    @property
    def kettle_spawn_z(self) -> float:
        return self.deck_h + self.kettle_h / 2 + 0.002

    def __post_init__(self) -> None:
        # Honesty-by-construction asserts (the geometric claims the task rests on).
        g = 9.81
        assert self.kettle_s > self.shaft_w + 0.004, \
            "the kettle body must NOT fit into the guard shaft (distractor is useless)"
        assert self.knob_s < self.shaft_w - 0.008 and \
            self.knob_h < self.recess_rest - 0.010, \
            "the kettle knob fits the shaft but must fall FAR short of the striker"
        assert (self.shaft_w - max(self.handle_wy, self.handle_wz)) / 2 >= 0.010, \
            "the pan handle must enter the shaft with real clearance (no wedging)"
        assert self.pan_w > self.tower_base_s, \
            "the dish must be wider than the tower (it can never follow the handle in)"
        assert self.dip_reach >= self.recess_pressed + 0.030, \
            "handle-down, the handle tip must reach FULL press depth with margin"
        assert self.recess_rest >= 0.062, \
            "the striker must sit too deep for fingertips (~58 mm below the mouth)"
        assert 0.7 * self.pan_mass * g > \
            self.spring_preload + self.spring_k * self.travel + self.striker_mass * g, \
            "a fraction of the pan's weight must drive the striker to FULL travel"
        assert self.spring_preload > 1.5 * self.striker_mass * g, \
            "the spring must genuinely return the striker (preload beats its weight)"
        assert self.press_depth <= self.travel - 0.003, \
            "the click depth must sit safely inside the stroke"
        assert self.striker_rest_z - self.cap_t / 2 - self.travel >= \
            self.tower_base_h - 1e-9, \
            "full travel must not drive the cap through the chamber floor"
        assert self.cap_s < self.shaft_w - 0.004, \
            "the cap must slide freely in the shaft"
        assert self.plate_s / 2 >= self.pan_xy_tol + 0.020, \
            "the plate must cover the judged xy window with margin"
        assert self.lift_min > self.plinth_h + self.plate_h + self.pan_z_tol + 0.030, \
            "the lift latch must be unreachable by any on-fixture placement"
        assert self.slot_y + self.slot_y_jitter + self.pan_w * math.sqrt(2.0) / 2 \
            < self.tower_y - self.tower_base_s / 2 - 0.004, \
            "the spawn slots must be clear of the igniter tower"
        assert self.slot_y + self.slot_y_jitter + self.pan_w * math.sqrt(2.0) / 2 \
            < self.burner_y_nom - self.burner_jitter - self.plinth_s / 2 - 0.004, \
            "the spawn slots must be clear of the burner at any jitter"
        # the yaw cone pushes the handle tip at least tip*cos(half) SOUTH of the
        # dish centre — past the kettle's southmost extent at any slot jitter,
        # so the spawned handle can never rest against the kettle.
        assert self.handle_tip_reach * math.cos(math.radians(self.pan_yaw_half_deg)) \
            > 2 * self.slot_y_jitter + self.kettle_s * math.sqrt(2.0) / 2 + 0.020, \
            "the pan handle must not reach the kettle across the slots at any yaw"
        assert math.sin(math.radians(self.pan_yaw_center_deg + self.pan_yaw_half_deg)) \
            < -0.5, \
            "the pan handle must stay pointed into the south half-plane at any spawn yaw"
        assert self.counter_half_x > self.tower_x + self.tower_base_s / 2 + 0.02
        assert self.counter_half_x > self.slot_x + self.slot_x_jitter \
            + self.pan_w * math.sqrt(2.0) / 2 + 0.02
        assert self.counter_half_y > abs(self.slot_y) + self.slot_y_jitter \
            + self.pan_w * math.sqrt(2.0) / 2 + 0.02


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
@SCENES.register("piezo_stove")
class PiezoStoveScene(BaseScene):
    cfg: PiezoStoveSceneCfg

    def __init__(self, cfg: PiezoStoveSceneCfg | None = None) -> None:
        super().__init__(cfg or PiezoStoveSceneCfg())

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
            # Tower authored at its FIXED pose; the bind-time striker joint anchors
            # here and the tower is NEVER moved (kinematic joint anchors are
            # world-fixed).
            "tower": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tower",
                spawn=_compound_spawner_cfg("tower", _spawn_tower, c, 5.0,
                                            kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tower_x, c.tower_y, c.deck_h)),
            ),
            "striker": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Striker",
                spawn=_compound_spawner_cfg("striker", _spawn_striker, c,
                                            c.striker_mass, kinematic=False,
                                            lin_damp=c.striker_damp),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tower_x, c.tower_y, c.striker_rest_w)),
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
                    pos=(c.slot_x, c.slot_y, c.pan_spawn_z)),
            ),
            "kettle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Kettle",
                spawn=_compound_spawner_cfg("kettle", _spawn_kettle, c,
                                            c.kettle_mass, kinematic=False,
                                            lin_damp=0.2, ang_damp=0.3),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-c.slot_x, c.slot_y, c.kettle_spawn_z)),
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
        self.tower: RigidObject = env.iscene["tower"]
        self.striker: RigidObject = env.iscene["striker"]
        self.burner: RigidObject = env.iscene["burner"]
        self.pan: RigidObject = env.iscene["pan"]
        self.kettle: RigidObject = env.iscene["kettle"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.pan_east = torch.ones(n, dtype=torch.bool, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._lit = torch.zeros(n, dtype=torch.bool, device=dev)
        self._lift = torch.zeros(n, dtype=torch.bool, device=dev)
        self._seat = torch.zeros(n, dtype=torch.bool, device=dev)
        self._streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._lamp_shown: list[bool | None] = [None] * n
        self._lamp_prims = None  # lazily grabbed in _refresh_lamps
        self._author_joints()

    def _author_joints(self) -> None:
        """Per-env striker joint, authored ONCE at bind time against the
        AUTHORED poses: a vertical prismatic joint anchored on the KINEMATIC
        tower (its joint anchors are world-fixed, so the tower must never
        move). The joint LIMITS are the hard stops — upper (0) is the spring-
        held rest, lower (-travel) is the full-press stop. Joint collision
        filtering only disables the tower<->striker pair, so the pan handle
        still contacts both the shaft walls and the striker cap."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        c = self.cfg
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/striker_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Tower"])
            j.CreateBody1Rel().SetTargets([f"{base}/Striker"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.striker_rest_z)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-float(c.travel))
            j.CreateUpperLimitAttr(0.0)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: counter and tower re-asserted at their FIXED poses
        (the tower anchors the striker joint — kinematic joint anchors are
        world-fixed), striker at the top of its travel, burner teleported to a
        random xy, pan and kettle swapped between the two spawn slots by a
        coin, latches and lamp cleared."""
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
        tower_xy = torch.tensor([c.tower_x, c.tower_y], device=dev).expand(m, 2)
        write(self.tower, tower_xy, c.deck_h)
        write(self.striker, tower_xy, c.striker_rest_w)

        # --- burner: random xy on the back-left quarter ---
        bx = c.burner_x_nom + (torch.rand(m, device=dev) * 2 - 1) * c.burner_jitter
        by = c.burner_y_nom + (torch.rand(m, device=dev) * 2 - 1) * c.burner_jitter
        write(self.burner, torch.stack([bx, by], dim=-1), c.deck_h)

        # --- pan / kettle: slot-swap coin + jitter (torch.rand comparison —
        # the FIRST torch.randint after manual_seed is near-constant) ---
        self.pan_east[env_ids] = torch.rand(m, device=dev) < 0.5
        s = torch.where(self.pan_east[env_ids], torch.ones(m, device=dev),
                        -torch.ones(m, device=dev))
        px = s * c.slot_x + (torch.rand(m, device=dev) * 2 - 1) * c.slot_x_jitter
        py = c.slot_y + (torch.rand(m, device=dev) * 2 - 1) * c.slot_y_jitter
        pyaw = math.radians(c.pan_yaw_center_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pan_yaw_half_deg)
        write(self.pan, torch.stack([px, py], dim=-1), c.pan_spawn_z, _qz(pyaw))

        kx = -s * c.slot_x + (torch.rand(m, device=dev) * 2 - 1) * c.slot_x_jitter
        ky = c.slot_y + (torch.rand(m, device=dev) * 2 - 1) * c.slot_y_jitter
        kyaw = torch.rand(m, device=dev) * 2 * math.pi
        write(self.kettle, torch.stack([kx, ky], dim=-1), c.kettle_spawn_z, _qz(kyaw))

        # --- clear latches + mechanism + lamp ---
        for latch in (self._lit, self._lift, self._seat):
            latch[env_ids] = False
        self._streak[env_ids] = 0
        for e in env_ids.tolist():
            self._lamp_shown[e] = None  # force a lamp refresh next post_step

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "counter": self.counter.data.root_state_w[env_ids].clone(),
            "tower": self.tower.data.root_state_w[env_ids].clone(),
            "striker": self.striker.data.root_state_w[env_ids].clone(),
            "burner": self.burner.data.root_state_w[env_ids].clone(),
            "pan": self.pan.data.root_state_w[env_ids].clone(),
            "kettle": self.kettle.data.root_state_w[env_ids].clone(),
            "pan_east": self.pan_east[env_ids].clone(),
            "lit": self._lit[env_ids].clone(),
            "lift": self._lift[env_ids].clone(),
            "seat": self._seat[env_ids].clone(),
            "streak": self._streak[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for name in ("counter", "tower", "striker", "burner", "pan", "kettle"):
            getattr(self, name).write_root_state_to_sim(state[name], env_ids)
        self.pan_east[env_ids] = state["pan_east"]
        self._lit[env_ids] = state["lit"]
        self._lift[env_ids] = state["lift"]
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
            f"On its back-left area stands the BURNER: a dark plinth topped by a "
            f"{c.plate_s * 100:.0f} cm square cook plate, with an indicator lamp on "
            f"its front face — dark red now, because the burner is OFF. On the back-"
            f"right corner stands the burner's PIEZO IGNITER: a grey "
            f"{c.tower_base_s * 1000:.0f} mm column with an orange-rimmed "
            f"{c.shaft_w * 1000:.0f} mm square guard shaft down its middle; the red "
            f"spring-loaded STRIKER cap sits {c.recess_rest * 1000:.0f} mm below the "
            f"mouth, far beyond fingertip reach. Pressing the striker at least "
            f"{c.press_depth * 1000:.0f} mm down its {c.travel * 1000:.0f} mm stroke "
            f"clicks the igniter: the burner lights PERMANENTLY (the lamp turns "
            f"orange and stays on) and the spring pops the striker back — nothing "
            f"needs to be held. On the front apron sit a square FRYING PAN "
            f"({c.pan_w * 1000:.0f} mm dish, {c.handle_l * 1000:.0f} mm stick "
            f"handle) and a steel KETTLE ({c.kettle_s * 1000:.0f} mm body with a "
            f"small lid knob) — which one is left and which is right is randomized "
            f"per episode, as are the burner position and both spawn poses.\n"
            f"Goal: turn the burner on and put the frying pan on it. The only tool "
            f"that reaches the striker is the pan's own stick handle: hold the pan "
            f"handle-down over the shaft, dip the handle in, and let the pan's "
            f"weight press the striker past the click; then set the pan flat on the "
            f"cook plate. The kettle is a decoy: its body is wider than the shaft "
            f"and its knob is far too short to reach the striker — and a pan "
            f"parked on the plate with the burner off cooks nothing. Success: "
            f"burner lit AND the pan seated flat and centred on the cook plate, at "
            f"rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Turn the burner on and put the frying pan on it: dip the pan's stick "
            "handle down the igniter's guard shaft to click the recessed striker, "
            "then set the pan flat on the cook plate."
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

    def striker_depth(self) -> torch.Tensor:
        """(N,) striker press depth below its rest stop (m, >= 0)."""
        rest = self.env_origins[:, 2] + self.cfg.striker_rest_w
        return (rest - self.striker.data.root_pos_w[:, 2]).clamp(min=0.0)

    def pan_on_burner(self) -> torch.Tensor:
        """(N,) bool, geometric, BURNER-RELATIVE: pan centre within the radial
        xy window of the plate centre, pan bottom at the plate top (z band),
        pan upright. The burner is kinematic and never rotates."""
        c = self.cfg
        d = self.pan.data.root_pos_w[:, :2] - self.burner.data.root_pos_w[:, :2]
        xy_ok = d.norm(dim=-1) <= c.pan_xy_tol
        top = self.burner.data.root_pos_w[:, 2] + c.plinth_h + c.plate_h
        dz_ok = (self._pan_bottom_z() - top).abs() <= c.pan_z_tol
        upright = self._up_w(self.pan)[:, 2] > math.cos(math.radians(c.upright_max_deg))
        return xy_ok & dz_ok & upright

    def settled(self) -> torch.Tensor:
        """(N,) bool: pan |lin vel| below `settle_speed`."""
        return self.pan.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def _update_latches(self) -> None:
        """Lift + seat latches (the LIT latch advances only in post_step, where
        the press streak is counted once per physics substep)."""
        c = self.cfg
        pan_h = self.pan.data.root_pos_w[:, 2] - self.env_origins[:, 2] - c.deck_h
        self._lift |= pan_h > c.lift_min
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
        for e in range(self.env.num_envs):
            key = bool(self._lit[e])
            if key != self._lamp_shown[e]:
                self._lamp_shown[e] = key
                col = c.lamp_on if key else c.lamp_off
                self._lamp_prims[e].GetDisplayColorAttr().Set([Gf.Vec3f(*col)])

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Once per physics substep: count the press streak (ignition latch),
        update the credit latches, refresh the lamps, and apply the striker's
        return spring (post_step OWNS the striker force buffer)."""
        n, dev = self.env.num_envs, self.env.device
        c = self.cfg

        depth = self.striker_depth()
        self._streak = torch.where(depth > c.press_depth, self._streak + 1,
                                   torch.zeros_like(self._streak))
        self._lit |= self._streak >= c.press_streak
        self._update_latches()
        self._refresh_lamps()

        # spring: preload + rate, damped; the striker rides a vertical prismatic
        # joint and NEVER rotates, so the external-force frame quirk is moot.
        v_z = self.striker.data.root_lin_vel_w[:, 2]
        f_z = c.spring_preload + c.spring_k * depth - c.spring_c * v_z
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        self.striker.set_external_force_and_torque(
            f_z.view(n, 1, 1) * ez.view(n, 1, 3), torch.zeros(n, 1, 3, device=dev))

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: burner LIT (latched mechanism state — the click really
        happened) AND the pan seated flat on the cook plate, settled and
        finite. The placement clauses are live physical outcomes."""
        self._update_latches()
        finite = torch.isfinite(self.pan.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.striker.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.kettle.data.root_pos_w).all(dim=-1)
        return self._lit & self.pan_on_burner() & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15*pan-carried + 0.25*igniter-clicked +
        0.15*pan-seated-on-plate (latched; ~0 for doing nothing), capped at
        0.55 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_lift * self._lift.float()
                + c.w_lit * self._lit.float()
                + c.w_seat * self._seat.float()).clamp(max=0.55)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="piezo_stove", robot="null"))
