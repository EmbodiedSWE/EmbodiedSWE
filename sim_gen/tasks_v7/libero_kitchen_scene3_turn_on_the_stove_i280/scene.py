"""ValveBeamStoveScene — turn on the gas stove by LOADING its safety weigh-beam:
only when BOTH steel ingots sit in the weigh basket does the beam sink and hold
the gas valve open (sim_gen task `libero_kitchen_scene3_turn_on_the_stove_i280`).

Derived from libero_90/libero_kitchen_scene3_turn_on_the_stove, but
STRATEGICALLY different. The seed "turns on" the stove by rotating a knob past
a JOINT-ANGLE threshold (`flat_stove.joint_pos[:, 0] > 0.5`) — a single direct
fixture actuation judged from a fixture joint the robot grasps. Here the robot
NEVER touches the valve mechanism at all:

  (1) the stove's gas valve is held shut by a counterweighted WEIGH-BEAM
      (steelyard): a rigid beam on a revolute pivot atop the valve post. Its
      inner arm carries a cast counterweight; its outer arm carries an
      open-top WEIGH BASKET. Empty, the counterweight wins and the beam rests
      on the CLOSED stop (basket end up). "Turn on the stove" = make the
      BASKET end sink past the ON angle and stay there.
  (2) the only honest way to do that is to LOAD MASS: the counterweight is
      sized (and cfg-asserted, with cube-at-wall worst-case lever arms) so
      that BOTH dark steel ingots in the basket out-torque it by >= 15 % at
      any in-basket placement, while ONE steel plus ALL the pale wood decoy
      blocks falls >= 15 % short even at the most favourable placement. The
      wood blocks are identical in size — mass discrimination, not shape.
  (3) pressing the beam down by hand is NOT success: the rubric requires the
      known load (both steels) physically inside the basket volume (positions
      in the BEAM's body frame) at the same time as the angle — an empty
      pressed beam reads False, and released it swings back closed.
  (4) success is judged from the beam's FREE-BODY pose (root quaternion) and
      the ingots' beam-frame positions — no articulation joint readout, no
      grasp of any fixture, no single actuation event: it is a cumulative
      weighing outcome that only holds hands-off because statics holds it.

Assets are fully procedural (native PhysX box colliders; the beam rides a
spawn-authored per-env revolute Y joint whose LIMITS [-16 deg, +16 deg] are the
hard stops; the joint's collision filter disables ONLY the beam<->stove pair):
  - counter (KINEMATIC): 800 x 640 x 140 mm kitchen bench.
  - stove fixture (KINEMATIC, FIXED pose — it anchors the beam joint and
    kinematic joint anchors are world-fixed): gas burner box + brass supply
    pipe + valve post; a flame ring on the burner recolors when the valve is
    held open (visual only).
  - beam (DYNAMIC, 1.3 kg, authored CoM on the inner arm): spine +
    counterweight + red flag + open-top basket (interior 64 x 96 mm, 45 mm
    walls, floor sized so a resting cube's CoM sits AT hinge height — lever
    arms then scale with cos(angle) on BOTH sides and the torque margins are
    angle-invariant).
  - 2 STEEL ingots (DYNAMIC, 40 mm cubes, 400 g) and 2 WOOD decoys (DYNAMIC,
    40 mm cubes, 30 g) shuffled along the front apron.

Rubric (0..1; anchored in the demonstrated solve; monotone along it):
  0.25 per steel ingot deposited (latched in-basket-and-calm AND currently
       in-basket), up to 2 -> 0.50
  1.0  iff success(): beam angle >= theta_on AND both steels counted in the
       basket AND beam + counted steels calm AND all bodies finite.
  Doing nothing scores 0; pressing the empty beam scores 0.

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


_G = 9.81

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


def _spawn_stove(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC gas stove fixture. Origin = valve-post base centre ON the deck
    top. Children (one kinematic body): burner box (with dark top plate and a
    VISUAL flame ring that post_step recolors), brass supply pipe (visual),
    and the valve POST that carries the beam hinge. The stove anchors the beam
    joint, so it is NEVER moved after spawn (kinematic joint anchors are
    world-fixed)."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 15.0, kinematic=True)
    co = cfg.contact_offset

    # burner box + top plate + flame ring (flame is VISUAL ONLY, recolored)
    bx, bh = cfg.burner_x, cfg.burner_half
    _box(stage, f"{prim_path}/burner", (2 * bh, 2 * bh, cfg.burner_h),
         (bx, 0.0, cfg.burner_h / 2), cfg.burner_color, co)
    _box(stage, f"{prim_path}/burner_plate", (2 * bh - 0.010, 2 * bh - 0.010, 0.004),
         (bx, 0.0, cfg.burner_h + 0.002), cfg.plate_color, co)
    _box(stage, f"{prim_path}/flame", (0.060, 0.060, 0.010),
         (bx, 0.0, cfg.burner_h + 0.009), cfg.flame_off, None)

    # brass gas pipe from the burner to the valve post (visual only)
    _box(stage, f"{prim_path}/pipe", (abs(bx) - bh + 0.030, 0.018, 0.018),
         ((bx + bh) / 2 - 0.010, 0.0, 0.030), cfg.pipe_color, None)

    # valve post: carries the beam hinge at (0, 0, hinge_z)
    _box(stage, f"{prim_path}/post", (cfg.post_w, cfg.post_w, cfg.post_h),
         (0.0, 0.0, cfg.post_h / 2), cfg.post_color, co)
    _box(stage, f"{prim_path}/post_cap", (cfg.post_w + 0.008, cfg.post_w + 0.008, 0.006),
         (0.0, 0.0, cfg.post_h + 0.003), cfg.pipe_color, None)
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC weigh-beam. Origin = the HINGE point (the bind-time revolute Y
    joint anchors here). One compound body: spine along x, counterweight block
    + red flag on the inner (-x) arm, open-top basket on the outer (+x) arm.
    Mass properties are authored EXPLICITLY (MassAPI mass alone leaves the CoM
    at the body origin — corpus lesson): CoM on the inner arm gives the
    closing torque; diagonal inertia keeps the swing well-conditioned."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.beam_mass, kinematic=False, ang_damp=cfg.beam_damp)
    mapi = UsdPhysics.MassAPI(root)
    mapi.CreateCenterOfMassAttr(Gf.Vec3f(float(cfg.beam_com_x), 0.0, 0.0))
    mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in cfg.beam_inertia]))
    mapi.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    co, c = cfg.contact_offset, cfg

    # spine
    _box(stage, f"{prim_path}/spine",
         (c.arm_in + c.basket_x + c.basket_hx, 0.024, 0.014),
         ((-c.arm_in + c.basket_x + c.basket_hx) / 2, 0.0, 0.0), c.beam_color, co)
    # counterweight + red valve flag (flag visual only)
    _box(stage, f"{prim_path}/weight", (0.055, 0.050, 0.050),
         (-c.arm_in + 0.035, 0.0, 0.0), c.weight_color, co)
    _box(stage, f"{prim_path}/flag", (0.006, 0.030, 0.050),
         (-c.arm_in + 0.003, 0.0, 0.045), c.flag_color, None)

    # basket: floor + 4 walls; interior floor TOP at z = -cube_s/2 so a resting
    # cube's CoM sits AT hinge height (angle-invariant torque margins)
    fz = -c.cube_s / 2 - c.basket_floor_t / 2
    _box(stage, f"{prim_path}/basket_floor",
         (2 * c.basket_hx + 2 * c.basket_wall_t, 2 * c.basket_hy + 2 * c.basket_wall_t,
          c.basket_floor_t),
         (c.basket_x, 0.0, fz), c.basket_color, co)
    wz = -c.cube_s / 2 + c.basket_wall_h / 2
    for tag, sx in (("basket_e", 1.0), ("basket_w", -1.0)):
        _box(stage, f"{prim_path}/{tag}",
             (c.basket_wall_t, 2 * c.basket_hy + 2 * c.basket_wall_t, c.basket_wall_h),
             (c.basket_x + sx * (c.basket_hx + c.basket_wall_t / 2), 0.0, wz),
             c.basket_color, co)
    for tag, sy in (("basket_n", 1.0), ("basket_s", -1.0)):
        _box(stage, f"{prim_path}/{tag}",
             (2 * c.basket_hx, c.basket_wall_t, c.basket_wall_h),
             (c.basket_x, sy * (c.basket_hy + c.basket_wall_t / 2), wz),
             c.basket_color, co)
    return root


def _spawn_steel(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC steel ingot: one dark-metal cube (the real load)."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.steel_mass, kinematic=False, lin_damp=0.1, ang_damp=0.2)
    s = cfg.cube_s
    _box(stage, f"{prim_path}/cube", (s, s, s), (0.0, 0.0, 0.0),
         cfg.steel_color, cfg.contact_offset)
    return root


def _spawn_wood(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC wood decoy: same-size pale cube, ~13x lighter than steel."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.wood_mass, kinematic=False, lin_damp=0.1, ang_damp=0.2)
    s = cfg.cube_s
    _box(stage, f"{prim_path}/cube", (s, s, s), (0.0, 0.0, 0.0),
         cfg.wood_color, cfg.contact_offset)
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
class ValveBeamStoveSceneCfg(BaseCfg):
    """Config for `ValveBeamStoveScene`. The honesty statics are asserted in
    __post_init__: with cube-at-wall WORST-CASE lever arms, both steels
    out-torque the counterweight by >= 15 % while one steel plus all the wood
    falls >= 15 % short at the BEST-CASE arm; the basket admits/retains the
    cubes; the beam sweep clears the deck, burner, and post at both stops."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    need_steel: int = tunable(2)          # steels required in the basket for success
    theta_on_deg: float = tunable(12.0)   # beam angle (basket down) that opens the valve
    in_margin: float = tunable(0.004)     # xy margin inside the basket walls (m)
    in_z_lo: float = tunable(-0.012)      # in-basket window, beam-frame z (m)
    in_z_hi: float = tunable(0.070)       # ... covers a stacked second cube
    latch_speed: float = tunable(0.10)    # calm gate for the in-basket latch (m/s)
    settle_speed: float = tunable(0.05)   # counted steels must be this calm at success
    beam_calm: float = tunable(0.20)      # beam |ang vel| gate at success (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    slot_y: float = tunable(-0.200)       # item spawn row on the front apron (m)
    slot_jitter: float = tunable(0.012)   # +/- xy jitter per item (m)
    slot_xs: tuple = tunable((-0.27, -0.09, 0.09, 0.27))  # 4 slots, permuted

    # --- info: counter ---------------------------------------------------------------------------
    deck_h: float = info(0.140)
    counter_half_x: float = info(0.400)
    counter_half_y: float = info(0.320)
    # --- info: stove fixture (FIXED pose — beam joint anchor; never randomized) ------------------
    stove_x: float = info(-0.050)
    stove_y: float = info(0.100)
    burner_x: float = info(-0.200)        # burner centre (stove-local x)
    burner_half: float = info(0.080)
    burner_h: float = info(0.040)
    post_w: float = info(0.044)
    post_h: float = info(0.105)
    hinge_z: float = info(0.130)          # hinge height above the deck (stove-local z)
    # --- info: beam ------------------------------------------------------------------------------
    arm_in: float = info(0.150)           # inner arm reach (counterweight side)
    basket_x: float = info(0.160)         # basket centre (beam-frame x)
    basket_hx: float = info(0.032)        # basket interior half-extents
    basket_hy: float = info(0.048)
    basket_wall_t: float = info(0.006)
    basket_wall_h: float = info(0.045)
    basket_floor_t: float = info(0.008)
    beam_mass: float = info(1.3)
    beam_com_x: float = info(-0.0745)     # authored CoM -> closing torque 0.95 N*m
    beam_inertia: tuple = info((0.004, 0.033, 0.035))
    beam_damp: float = info(2.0)
    beam_lo_deg: float = info(-16.0)      # CLOSED stop (basket up; counterweight rest)
    beam_hi_deg: float = info(16.0)       # OPEN stop (basket down)
    # --- info: cubes -----------------------------------------------------------------------------
    n_steel: int = info(2)
    n_wood: int = info(2)
    cube_s: float = info(0.040)
    steel_mass: float = info(0.400)
    wood_mass: float = info(0.030)
    # --- info: misc ------------------------------------------------------------------------------
    contact_offset: float = info(0.0015)
    deck_color: tuple = info((0.42, 0.44, 0.48))
    burner_color: tuple = info((0.16, 0.17, 0.19))
    plate_color: tuple = info((0.10, 0.10, 0.11))
    post_color: tuple = info((0.34, 0.30, 0.20))
    pipe_color: tuple = info((0.55, 0.45, 0.20))
    beam_color: tuple = info((0.35, 0.36, 0.40))
    basket_color: tuple = info((0.28, 0.29, 0.33))
    weight_color: tuple = info((0.15, 0.15, 0.17))
    flag_color: tuple = info((0.85, 0.12, 0.10))
    steel_color: tuple = info((0.22, 0.24, 0.30))
    wood_color: tuple = info((0.78, 0.63, 0.38))
    flame_off: tuple = info((0.15, 0.06, 0.05))
    flame_on: tuple = info((1.00, 0.50, 0.08))

    # ----- derived -------------------------------------------------------------------------------
    @property
    def tau_close(self) -> float:
        """Counterweight closing torque at level (N*m)."""
        return self.beam_mass * _G * abs(self.beam_com_x)

    @property
    def cube_x_lo(self) -> float:
        """In-basket cube-CENTRE extreme nearest the hinge (beam-frame x)."""
        return self.basket_x - (self.basket_hx - self.cube_s / 2)

    @property
    def cube_x_hi(self) -> float:
        return self.basket_x + (self.basket_hx - self.cube_s / 2)

    @property
    def cube_spawn_z(self) -> float:
        return self.deck_h + self.cube_s / 2 + 0.002

    @property
    def basket_outer_hx(self) -> float:
        return self.basket_hx + self.basket_wall_t

    def __post_init__(self) -> None:
        # Honesty-by-construction asserts (the statics the task rests on).
        # Resting cube CoM sits AT hinge height (basket floor top = -cube_s/2),
        # so both sides' lever arms scale with cos(angle) and the level-arm
        # margins below hold at EVERY angle in the stop range; a stacked second
        # cube only ever helps the opening side at the open stop.
        tau_open_worst = self.n_steel * self.steel_mass * _G * self.cube_x_lo
        assert tau_open_worst >= 1.15 * self.tau_close, \
            f"both steels must out-torque the counterweight by >=15% even hugging " \
            f"the inner wall ({tau_open_worst:.3f} vs {self.tau_close:.3f})"
        tau_decoy_best = (self.steel_mass + self.n_wood * self.wood_mass) * _G \
            * self.cube_x_hi
        assert 1.15 * tau_decoy_best <= self.tau_close, \
            f"one steel + all wood must fall >=15% short even at the outer wall " \
            f"({tau_decoy_best:.3f} vs {self.tau_close:.3f})"
        # -- basket admits, seats, and retains the cubes --
        assert 2 * self.basket_hx - self.cube_s >= 0.020, \
            "the basket mouth must pass a cube with >= 20 mm x-clearance"
        assert 2 * self.basket_hy - 2 * self.cube_s >= 0.012, \
            "two cubes side by side must fit the basket with clearance"
        assert self.basket_wall_h >= self.cube_s + 0.004, \
            "the walls must stand above a seated cube (retention at full tilt)"
        # -- judged in-basket window: covers wall-hugging + stacked, stays inside --
        assert self.basket_hx - self.in_margin >= (self.basket_hx - self.cube_s / 2) + 0.004, \
            "a cube hugging the x wall must still be judged in-basket"
        assert self.basket_hy - self.in_margin >= (self.basket_hy - self.cube_s / 2) + 0.004, \
            "a cube hugging the y wall must still be judged in-basket"
        assert self.in_z_lo > -self.cube_s / 2 - self.basket_floor_t + 0.002, \
            "the window must exclude a cube trapped UNDER the basket floor"
        assert self.in_z_lo < -0.004 < 0.004 < self.in_z_hi, \
            "a cube seated on the basket floor (CoM z=0) must be well inside"
        assert self.in_z_hi >= self.cube_s + 0.010, \
            "a stacked second cube must still be judged in-basket"
        assert self.in_margin >= 0.002, "need a real wall margin"
        # -- stops and threshold --
        assert self.beam_lo_deg <= -10.0 < 0.0 < self.theta_on_deg, \
            "the closed rest angle must be clearly readable"
        assert self.theta_on_deg <= self.beam_hi_deg - 3.0, \
            "the ON threshold must sit clear of the open stop"
        # -- sweep clearances (worst corners over the stop range) --
        s_hi, c_hi = math.sin(math.radians(self.beam_hi_deg)), \
            math.cos(math.radians(self.beam_hi_deg))
        drop_open = (self.basket_x + self.basket_outer_hx) * s_hi \
            + (self.cube_s / 2 + self.basket_floor_t) * c_hi
        assert self.hinge_z - drop_open >= 0.030, \
            "the basket must clear the deck at the open stop"
        drop_closed = self.arm_in * s_hi + 0.025 * c_hi  # counterweight bottom
        assert self.hinge_z - drop_closed >= 0.030, \
            "the counterweight must clear the deck at the closed stop"
        cw_over_burner = self.hinge_z - (0.120 * s_hi + 0.025 * c_hi)
        assert self.burner_x + self.burner_half <= -0.120 and \
            cw_over_burner - self.burner_h >= 0.020, \
            "the inner arm must clear the burner box at the closed stop"
        assert self.hinge_z - 0.007 - self.post_h >= 0.010, \
            "the spine must swing clear above the post"
        # -- fixture + basket inside the counter --
        assert self.stove_x + self.basket_x + self.basket_outer_hx \
            < self.counter_half_x - 0.020
        assert self.stove_x + self.burner_x - self.burner_half \
            > -self.counter_half_x + 0.020
        assert abs(self.stove_y) + self.basket_hy + self.basket_wall_t \
            < self.counter_half_y - 0.020
        # -- layout: slots clear of the fixture, each other, and the counter edge --
        half_diag = self.cube_s * math.sqrt(2) / 2
        basket_south = self.stove_y - self.basket_hy - self.basket_wall_t
        assert self.slot_y + self.slot_jitter + half_diag < basket_south - 0.030, \
            "the spawn row must sit clear (south) of the beam sweep"
        xs = sorted(self.slot_xs)
        for a, b in zip(xs, xs[1:]):
            assert b - a >= 2 * (half_diag + self.slot_jitter) + 0.010, \
                "adjacent slots must never collide at any jitter/yaw"
        assert max(abs(x) for x in xs) + self.slot_jitter + half_diag \
            < self.counter_half_x - 0.020
        assert abs(self.slot_y) + self.slot_jitter + half_diag \
            < self.counter_half_y - 0.020
        assert len(self.slot_xs) == self.n_steel + self.n_wood, \
            "one slot per cube (2 steel + 2 wood)"
        assert 1 <= self.need_steel <= self.n_steel, "success must be reachable"


# ----- small helpers (wxyz quats, torch, batched) -----------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _quat_rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors `v` (N,3) into the body frame of quats `q` (N,4 wxyz)."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("valve_beam_stove")
class ValveBeamStoveScene(BaseScene):
    cfg: ValveBeamStoveSceneCfg

    def __init__(self, cfg: ValveBeamStoveSceneCfg | None = None) -> None:
        super().__init__(cfg or ValveBeamStoveSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        out: dict[str, Any] = {
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
            # Stove authored at its FIXED pose; the bind-time beam joint anchors
            # here and the stove is NEVER moved (kinematic joint anchors are
            # world-fixed).
            "stove": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stove",
                spawn=_compound_spawner_cfg("stove", _spawn_stove, c, 15.0,
                                            kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stove_x, c.stove_y, c.deck_h)),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=_compound_spawner_cfg("beam", _spawn_beam, c, c.beam_mass,
                                            kinematic=False, ang_damp=c.beam_damp),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stove_x, c.stove_y, c.deck_h + c.hinge_z)),
            ),
        }
        for k in range(c.n_steel):
            out[f"steel{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Steel" + str(k),
                spawn=_compound_spawner_cfg("steel", _spawn_steel, c,
                                            c.steel_mass, kinematic=False,
                                            lin_damp=0.1, ang_damp=0.2),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_xs[k], c.slot_y, c.cube_spawn_z)),
            )
        for k in range(c.n_wood):
            out[f"wood{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Wood" + str(k),
                spawn=_compound_spawner_cfg("wood", _spawn_wood, c,
                                            c.wood_mass, kinematic=False,
                                            lin_damp=0.1, ang_damp=0.2),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_xs[c.n_steel + k], c.slot_y, c.cube_spawn_z)),
            )
        return out

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
        self.stove: RigidObject = env.iscene["stove"]
        self.beam: RigidObject = env.iscene["beam"]
        self.steels: list[RigidObject] = [
            env.iscene[f"steel{k}"] for k in range(self.cfg.n_steel)]
        self.woods: list[RigidObject] = [
            env.iscene[f"wood{k}"] for k in range(self.cfg.n_wood)]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # latch: steel was ever in-basket-and-calm (counting also requires
        # CURRENTLY in-basket, so the latch can never over-count)
        self._latched = torch.zeros(n, self.cfg.n_steel, dtype=torch.bool, device=dev)
        self._flame_shown: list[bool | None] = [None] * n
        self._flame_prims = None  # lazily grabbed in _refresh_flames
        self._author_joints()

    def _author_joints(self) -> None:
        """Per-env beam pivot, authored ONCE at bind time against the AUTHORED
        poses: a revolute Y joint anchored on the KINEMATIC stove post (its
        joint anchors are world-fixed, so the stove must never move). The
        joint LIMITS are the stops — lower (-16 deg) is the CLOSED rest
        (counterweight wins), upper (+16 deg) the OPEN stop the loaded basket
        holds. Joint collision filtering only disables the stove<->beam pair,
        so the cubes still contact both."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        c = self.cfg
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/beam_pivot")
            j.CreateBody0Rel().SetTargets([f"{base}/Stove"])
            j.CreateBody1Rel().SetTargets([f"{base}/Beam"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.hinge_z)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(float(c.beam_lo_deg))
            j.CreateUpperLimitAttr(float(c.beam_hi_deg))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: counter and stove re-asserted at their FIXED poses
        (the stove anchors the beam joint — kinematic joint anchors are
        world-fixed), beam re-levelled at the hinge (it falls onto the closed
        stop during settle), and the 4 cubes (2 steel + 2 wood) PERMUTED over
        the 4 apron slots with xy jitter and free yaw (argsort of torch.rand —
        the FIRST torch.randint after manual_seed is near-constant), latches
        and flame cleared."""
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

        write(self.counter, torch.zeros(m, 2, device=dev),
              torch.zeros(m, device=dev))
        stove_xy = torch.tensor([c.stove_x, c.stove_y], device=dev).expand(m, 2)
        write(self.stove, stove_xy, c.deck_h)
        write(self.beam, stove_xy, c.deck_h + c.hinge_z)

        # --- cubes: permutation over the 4 slots + jitter + free yaw ---
        slot_xs = torch.tensor(c.slot_xs, device=dev)
        perm = torch.argsort(torch.rand(m, len(c.slot_xs), device=dev), dim=1)
        bodies = [*self.steels, *self.woods]
        for i, body in enumerate(bodies):
            sx = slot_xs[perm[:, i]] + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            sy = c.slot_y + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            yaw = torch.rand(m, device=dev) * 2 * math.pi
            write(body, torch.stack([sx, sy], dim=-1), c.cube_spawn_z, _qz(yaw))

        # --- clear latches + flame ---
        self._latched[env_ids] = False
        for e in env_ids.tolist():
            self._flame_shown[e] = None  # force a flame refresh next post_step

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "counter": self.counter.data.root_state_w[env_ids].clone(),
            "stove": self.stove.data.root_state_w[env_ids].clone(),
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "latched": self._latched[env_ids].clone(),
        }
        for k, b in enumerate(self.steels):
            out[f"steel{k}"] = b.data.root_state_w[env_ids].clone()
        for k, b in enumerate(self.woods):
            out[f"wood{k}"] = b.data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for name in ("counter", "stove", "beam"):
            getattr(self, name).write_root_state_to_sim(state[name], env_ids)
        for k, b in enumerate(self.steels):
            b.write_root_state_to_sim(state[f"steel{k}"], env_ids)
        for k, b in enumerate(self.woods):
            b.write_root_state_to_sim(state[f"wood{k}"], env_ids)
        self._latched[env_ids] = state["latched"]
        for e in env_ids.tolist():
            self._flame_shown[e] = None

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A steel KITCHEN BENCH ({2 * c.counter_half_x * 100:.0f} x "
            f"{2 * c.counter_half_y * 100:.0f} cm, {c.deck_h * 100:.0f} cm tall). "
            f"On its back area stands a GAS STOVE: a dark burner box on the left "
            f"(with an unlit flame ring on top) fed by a brass gas pipe from a "
            f"VALVE POST. Atop the post pivots a counterweighted WEIGH-BEAM — a "
            f"safety interlock: a black counterweight and a red flag ride the "
            f"inner arm, and an open-top WEIGH BASKET (interior "
            f"{2 * c.basket_hx * 1000:.0f} x {2 * c.basket_hy * 1000:.0f} mm, "
            f"{c.basket_wall_h * 1000:.0f} mm walls) hangs on the outer arm, to "
            f"the right of the post. With the basket empty the counterweight "
            f"holds the basket end UP and the gas valve SHUT; the valve opens "
            f"only while the basket end is sunk past "
            f"{c.theta_on_deg:.0f} degrees and stays there. Along the front "
            f"apron lie FOUR cubes in a shuffled row (order, exact position and "
            f"yaw are randomized per episode), all {c.cube_s * 1000:.0f} mm on a "
            f"side: two dark STEEL ingots ({c.steel_mass * 1000:.0f} g each) and "
            f"two pale WOOD blocks ({c.wood_mass * 1000:.0f} g each — decoys far "
            f"too light to matter). The counterweight is sized so that BOTH "
            f"steel ingots in the basket sink it decisively, while one steel "
            f"plus both wood blocks cannot. Pressing the beam down by hand does "
            f"NOT open the valve: the interlock also senses the load, so the "
            f"flame lights only with the required steel in the basket.\n"
            f"Goal: turn on the stove by loading the weigh basket. Place both "
            f"steel ingots into the basket; the beam sinks to its stop, the "
            f"valve opens, and the flame ring lights. Success: both steels "
            f"resting inside the basket, the beam holding past "
            f"{c.theta_on_deg:.0f} degrees hands-off, everything settled."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Turn on the gas stove: put both heavy steel ingots into the weigh "
            "basket so the beam sinks and holds the gas valve open. The pale "
            "wood blocks are too light to help."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def beam_angle(self) -> torch.Tensor:
        """(N,) beam pivot angle in radians (+ = basket down = OPEN). The beam
        only ever rotates about the hinge Y axis, so the quat is
        (cos a/2, 0, sin a/2, 0)."""
        q = self.beam.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 2], q[:, 0])

    def _beam_frame(self, body) -> torch.Tensor:
        """(N,3) body origin in the BEAM's body frame (origin = hinge)."""
        d = body.data.root_pos_w - self.beam.data.root_pos_w
        return _quat_rotate_inv(self.beam.data.root_quat_w, d)

    def _in_basket_pos(self, rel: torch.Tensor) -> torch.Tensor:
        """(N,) bool: a point (beam-frame) is inside the judged basket window —
        strictly within the walls, above the floor, below the stacked-load
        ceiling."""
        c = self.cfg
        x_ok = (rel[:, 0] - c.basket_x).abs() < c.basket_hx - c.in_margin
        y_ok = rel[:, 1].abs() < c.basket_hy - c.in_margin
        z_ok = (rel[:, 2] > c.in_z_lo) & (rel[:, 2] < c.in_z_hi)
        return x_ok & y_ok & z_ok

    def steels_in_basket(self) -> torch.Tensor:
        """(N, n_steel) bool: steel k currently inside the basket."""
        return torch.stack(
            [self._in_basket_pos(self._beam_frame(b)) for b in self.steels], dim=1)

    def steel_speeds(self) -> torch.Tensor:
        """(N, n_steel) steel |lin vel|."""
        return torch.stack(
            [b.data.root_lin_vel_w.norm(dim=-1) for b in self.steels], dim=1)

    def counted(self) -> torch.Tensor:
        """(N, n_steel) bool: latched in-basket-and-calm AND currently in."""
        return self._latched & self.steels_in_basket()

    def count(self) -> torch.Tensor:
        """(N,) long: steels counted as deposited."""
        return self.counted().sum(dim=1)

    def _update_latches(self) -> None:
        self._latched |= self.steels_in_basket() & \
            (self.steel_speeds() < self.cfg.latch_speed)

    def _refresh_flames(self) -> None:
        from pxr import Gf

        if self._flame_prims is None:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            self._flame_prims = [
                UsdGeom.Cube(stage.GetPrimAtPath(f"/World/envs/env_{i}/Stove/flame"))
                for i in range(self.env.num_envs)]
        c = self.cfg
        lit = (self.count() >= c.need_steel) & \
            (self.beam_angle() >= math.radians(c.theta_on_deg))
        for e in range(self.env.num_envs):
            key = bool(lit[e])
            if key != self._flame_shown[e]:
                self._flame_shown[e] = key
                col = c.flame_on if key else c.flame_off
                self._flame_prims[e].GetDisplayColorAttr().Set([Gf.Vec3f(*col)])

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Once per physics substep: advance the in-basket latches and refresh
        the flames. The scene owns NO forces — the beam is pure passive
        statics against its joint limits."""
        self._update_latches()
        self._refresh_flames()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the beam holds past theta_on (basket down) AND both
        steels are counted in the basket (latched in-and-calm AND currently
        in — pressing an empty or decoy-loaded beam reads False) AND the beam
        and every counted steel are calm AND all bodies finite. Held only by
        statics, so it persists hands-off."""
        self._update_latches()
        c = self.cfg
        cnt = self.counted()
        open_enough = self.beam_angle() >= math.radians(c.theta_on_deg)
        beam_calm = self.beam.data.root_ang_vel_w.norm(dim=-1) < c.beam_calm
        calm = ((~cnt) | (self.steel_speeds() < c.settle_speed)).all(dim=1)
        finite = torch.isfinite(self.beam.data.root_pos_w).all(dim=-1)
        for b in [*self.steels, *self.woods]:
            finite &= torch.isfinite(b.data.root_pos_w).all(dim=-1)
        return (cnt.sum(dim=1) >= c.need_steel) & open_enough & beam_calm \
            & calm & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25 per steel deposited in the basket (up to
        2 -> 0.50; ~0 for doing nothing; pressing the beam moves nothing
        here) — and exactly 1.0 iff success() holds live."""
        self._update_latches()
        base = 0.25 * self.count().clamp(max=self.cfg.n_steel).float()
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="valve_beam_stove", robot="null"))
