"""FlameKeeperScene — swap the pot onto a dead-man burner without losing the flame
(sim_gen task `libero_kitchen_scene3_put_the_moka_pot_on_the_stove_i375`).

Derived from libero_90/libero_kitchen_scene3_put_the_moka_pot_on_the_stove, but
STRATEGICALLY different. The seed is a single unconstrained pick-and-place: grasp the
moka pot and set it on a flat, passive stove — success is a pose predicate on ONE
object, the stove is inert, and the frypan is a mere distractor. Here the stove is a
live MECHANISM with a continuously-enforced occupancy constraint, and the seed's
distractor becomes the load-bearing obstacle:

  - The burner PLATE is spring-loaded (a dead-man pilot valve): a constant upward
    pilot-spring force would push the empty plate to its raised stop in a fraction of
    a second, and if the plate ever rests fully raised for more than a moment the
    pilot flame goes OUT — permanently for the episode (an irreversible foul latch on
    a purely physical event: the spring/payload force balance).
  - The FRYING PAN starts ON the plate, keeping it pressed. The goal is the moka pot
    ALONE on the pressed plate with the pan off on the counter and the flame still
    alive.
  - The seed's plan (clear the target spot, then place the pot) here FAILS
    IRREVERSIBLY: lifting the pan first unloads the plate, the spring pops it up, and
    the flame dies — after which the very same end state scores zero. The only
    winning order is to install the pot on the plate's free half FIRST (both objects
    share the plate transiently) and remove the pan SECOND, so the plate never
    unloads.

A solver therefore needs a different PLAN from the seed (an ordering constraint
enforced by a live force balance, with a transient two-objects-on-target state the
seed never needs) and a different code structure (the failure mode is a latched
physical event, not a missed pose: end-state-identical trajectories score 0 or 1
depending on the ORDER of the two moves).

Assets are fully procedural (native PhysX box colliders):
  - deck (KINEMATIC): wooden counter slab 900 x 750 x 24 mm on the ground.
  - housing (KINEMATIC): stove body — base slab plus a rectangular well ring whose
    top is flush with the pressed plate's top; an orange pilot-lamp visual.
  - plate (DYNAMIC, prismatic-Z joint to the housing, travel 0..20 mm): the burner
    plate, 260 x 150 x 16 mm, 0.32 kg. The scene applies a constant +Z pilot-spring
    force (4.7 N) to it every post_step. Empty plate: spring beats weight by ~50 %
    and it rises to the top stop in ~90 ms. Any one payload (pan 0.40 kg or pot
    0.45 kg): weight beats spring by >= 40 % and the plate stays pressed. Margins
    asserted in __post_init__.
  - pot (DYNAMIC): octagonal moka pot, 63 mm across, 85 mm tall, side handle,
    0.45 kg. Starts on the counter (jittered xy, free yaw).
  - pan (DYNAMIC): the seed's frying pan, 0.40 kg. Starts centered on a randomized
    HALF of the plate (left or right per episode), handle over a side rim.

Foul (flame-out) bookkeeping, all physical: the foul arms only after the plate has
first rested pressed for a streak of steps (so reset transients cannot foul), then
latches permanently once the plate stays above 70 % travel for a streak of steps.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve; the flame
gates EVERYTHING — a fouled episode scores 0):
  0.10  pot ever within 12 cm of the plate center (approach; latched)
  0.25  pot ever at rest upright ON the plate (latched)
  0.25  swap ever complete: pot on the pressed plate AND pan at rest on the counter
        clear of the stove (latched)
  1.0 iff success() live: pot alone upright at rest on the PRESSED plate, pan at
  rest on the counter clear of the stove, flame alive (no foul), everything
  settled and finite. Non-success capped at 0.60; any foul forces 0.

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


def _box(stage, path: str, size, center, color, contact_offset: float | None,
         orient=None) -> None:
    """Author one box child; `contact_offset=None` -> VISUAL ONLY (no collider)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(seg.GetPrim(), contact_offset)


def _rigid_root(root, mass: float, kinematic: bool, lin_damp: float = 0.0,
                ang_damp: float = 0.0, com=None, inertia=None) -> None:
    from pxr import Gf, PhysxSchema, UsdPhysics

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:  # corpus-validated authoring: never author the attr False
        rb.CreateKinematicEnabledAttr(True)
    ma = UsdPhysics.MassAPI.Apply(root)
    ma.CreateMassAttr(float(mass))
    if com is not None:  # MassAPI mass alone leaves the CoM at the body origin
        ma.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    if inertia is not None:
        ma.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    if not kinematic:
        px.CreateSleepThresholdAttr(0.0)
        px.CreateStabilizationThresholdAttr(0.0)


def _oct45():
    """wxyz quat: 45 degrees about z (the octagon's second square)."""
    return (math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8))


def _spawn_housing(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC stove housing. Origin = base BOTTOM centre (sits on the deck):
    base slab + four well-ring walls (top flush with the pressed plate's top) +
    an orange pilot-lamp window (VISUAL ONLY) on the -x face."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg  # unwrap: the spawner cfg carries the scene cfg in its `cfg` field
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 10.0, kinematic=True)

    co = cfg.contact_offset
    col = cfg.housing_color
    ox, oy = cfg.housing_outer
    bh, rh = cfg.housing_base_h, cfg.ring_h
    wx, wy = cfg.well_inner  # inner well opening
    tx = (ox - wx) / 2  # x-wall thickness
    ty = (oy - wy) / 2  # y-wall thickness
    _box(stage, f"{prim_path}/base", (ox, oy, bh), (0.0, 0.0, bh / 2), col, co)
    for sx in (1.0, -1.0):
        _box(stage, f"{prim_path}/wall_x{'p' if sx > 0 else 'n'}",
             (tx, oy, rh), (sx * (wx / 2 + tx / 2), 0.0, bh + rh / 2), col, co)
    for sy in (1.0, -1.0):
        _box(stage, f"{prim_path}/wall_y{'p' if sy > 0 else 'n'}",
             (wx, ty, rh), (0.0, sy * (wy / 2 + ty / 2), bh + rh / 2), col, co)
    # pilot lamp: VISUAL ONLY orange window on the -x face
    _box(stage, f"{prim_path}/pilot_lamp", (0.012, 0.030, 0.014),
         (-(ox / 2 + 0.004), 0.0, 0.014), (0.95, 0.45, 0.08), None)
    return root


def _spawn_plate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC burner plate. Origin = plate BOTTOM centre. Dark slab with a
    slightly-inset lighter coil pattern (VISUAL ONLY)."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    px, py, pz = cfg.plate_size
    m = cfg.plate_mass
    ix = m * (py * py + pz * pz) / 12.0
    iy = m * (px * px + pz * pz) / 12.0
    iz = m * (px * px + py * py) / 12.0
    _rigid_root(root, m, kinematic=False, lin_damp=cfg.plate_lin_damp, ang_damp=1.0,
                com=(0.0, 0.0, pz / 2), inertia=(ix, iy, iz))
    co = cfg.contact_offset
    _box(stage, f"{prim_path}/slab", (px, py, pz), (0.0, 0.0, pz / 2),
         cfg.plate_color, co)
    # coil rings: VISUAL ONLY, flush film on the top face
    _box(stage, f"{prim_path}/coil", (px - 0.06, py - 0.05, 0.0006),
         (0.0, 0.0, pz + 0.0003), (0.42, 0.15, 0.12), None)
    return root


def _spawn_pot(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC moka pot. Origin = base BOTTOM centre: octagonal base + waisted
    octagonal top + lid knob + side handle along local +x."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    m = cfg.pot_mass
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, m, kinematic=False, lin_damp=0.3, ang_damp=0.5,
                com=(0.0, 0.0, 0.038),
                inertia=tuple(m * u for u in cfg.pot_unit_inertia))

    co = cfg.contact_offset
    body, dark = cfg.pot_color, cfg.pot_handle_color
    b, bh = cfg.pot_base_s, 0.044
    t, th = 0.050, 0.032
    for tag, orient in (("a", None), ("b", _oct45())):
        _box(stage, f"{prim_path}/base_{tag}", (b, b, bh), (0.0, 0.0, bh / 2), body, co,
             orient=orient)
        _box(stage, f"{prim_path}/top_{tag}", (t, t, th), (0.0, 0.0, bh + th / 2), body, co,
             orient=orient)
    _box(stage, f"{prim_path}/knob", (0.014, 0.014, 0.009),
         (0.0, 0.0, bh + th + 0.0045), dark, co)
    _box(stage, f"{prim_path}/handle", (0.036, 0.012, 0.020),
         (0.043, 0.0, 0.055), dark, co)
    return root


def _spawn_pan(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC frying pan (the seed's distractor, here the plate's occupant).
    Origin = base BOTTOM centre; handle along local +x."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    m = cfg.pan_mass
    _rigid_root(root, m, kinematic=False, lin_damp=0.3, ang_damp=0.5,
                com=(0.0, 0.0, 0.008), inertia=(m * 0.0012, m * 0.0012, m * 0.002))
    co = cfg.contact_offset
    _box(stage, f"{prim_path}/base", (cfg.pan_base_s, cfg.pan_base_s, 0.012),
         (0.0, 0.0, 0.006), cfg.pan_color, co)
    _box(stage, f"{prim_path}/rim_visual", (cfg.pan_base_s - 0.014, cfg.pan_base_s - 0.014,
         0.001), (0.0, 0.0, 0.0125), (0.22, 0.22, 0.24), None)
    _box(stage, f"{prim_path}/handle", (0.050, 0.016, 0.010), (0.080, 0.0, 0.014),
         (0.05, 0.05, 0.06), co)
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
            solver_position_iteration_count=32, solver_velocity_iteration_count=4)
    return _SPAWNER_CACHE[kind](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=rp,
        cfg=scene_cfg,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class FlameKeeperSceneCfg(BaseCfg):
    """Config for `FlameKeeperScene`. Honesty physics asserted in __post_init__."""

    # --- tunable: dead-man flame bookkeeping -----------------------------------------------------
    spring_force: float = tunable(4.7)     # pilot-spring +Z force on the plate (N)
    pressed_q: float = tunable(0.004)      # plate travel below this = pressed (m)
    foul_q_frac: float = tunable(0.70)     # fraction of travel above which = raised
    foul_streak: int = tunable(8)          # consecutive raised steps -> flame out
    arm_streak: int = tunable(5)           # consecutive pressed steps -> foul armed
    # --- tunable: rubric thresholds --------------------------------------------------------------
    settle_speed: float = tunable(0.10)    # payload lin speed at rest (m/s)
    plate_calm: float = tunable(0.05)      # plate lin speed at rest (m/s)
    latch_speed: float = tunable(0.10)     # calm gate for credit latches (m/s)
    near_dist: float = tunable(0.12)       # pot approach radius (m)
    clear_dist: float = tunable(0.25)      # pan-to-plate horizontal clearance (m)
    pot_tilt_max_deg: float = tunable(10.0)  # pot upright cone
    pan_deck_dz: tuple = tunable((-0.004, 0.020))  # pan-origin band above the deck top
    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    pan_side_offset: float = tunable(0.065)  # |pan x| on the plate (side is random)
    pan_jitter: float = tunable(0.005)     # pan xy jitter at its plate spot (+/- m)
    pan_yaw_jit_deg: float = tunable(15.0)  # around the +/-90 deg handle-out yaw
    pot_start_jitter: float = tunable(0.025)  # pot counter-spawn xy jitter (+/- m)
    pot_yaw_deg: float = tunable(180.0)    # pot free yaw (+/- deg)

    # --- info: layout (world nominal, deck top = z 0.024) ----------------------------------------
    deck_size: tuple = info((0.90, 0.75, 0.024))
    deck_pos: tuple = info((0.45, 0.0))
    housing_pos: tuple = info((0.30, 0.10))
    pot_start: tuple = info((0.62, -0.20))
    # --- info: stove housing / plate -------------------------------------------------------------
    housing_outer: tuple = info((0.37, 0.27))
    housing_base_h: float = info(0.027)
    ring_h: float = info(0.019)            # ring top = base_h + ring_h = 0.046
    well_gap: float = info(0.004)          # lateral gap plate <-> ring walls, per side
    plate_size: tuple = info((0.260, 0.150, 0.016))
    plate_lift: float = info(0.030)        # plate bottom above the deck top at q=0
    plate_travel: float = info(0.020)      # prismatic joint range (m)
    plate_mass: float = info(0.32)
    plate_lin_damp: float = info(2.0)
    # --- info: pot / pan -------------------------------------------------------------------------
    pot_base_s: float = info(0.063)        # octagon across-flats (base)
    pot_h: float = info(0.085)
    pot_mass: float = info(0.45)
    pot_unit_inertia: tuple = info((0.00093, 0.00093, 0.00066))  # per kg
    pot_slot_x: float = info(0.062)        # |x| of the pot's landing slot on the plate
    pan_base_s: float = info(0.110)
    pan_mass: float = info(0.40)
    # --- info: judge volumes ---------------------------------------------------------------------
    on_x: float = info(0.100)              # plate-local |x| bound for "on the plate"
    on_y: float = info(0.055)              # plate-local |y| bound
    on_z: tuple = info((0.008, 0.026))     # origin z above the PLATE origin
    # --- info: misc ------------------------------------------------------------------------------
    contact_offset: float = info(0.0015)
    deck_color: tuple = info((0.55, 0.44, 0.30))
    housing_color: tuple = info((0.30, 0.31, 0.34))
    plate_color: tuple = info((0.13, 0.13, 0.15))
    pot_color: tuple = info((0.55, 0.56, 0.60))
    pot_handle_color: tuple = info((0.05, 0.05, 0.06))
    pan_color: tuple = info((0.15, 0.15, 0.17))
    # rubric weights (0.10 + 0.25 + 0.25 = 0.60 = the non-success cap)
    w_near: float = info(0.10)
    w_on: float = info(0.25)
    w_swap: float = info(0.25)

    # ----- derived geometry ----------------------------------------------------------------------
    @property
    def deck_top(self) -> float:
        return self.deck_size[2]

    @property
    def well_inner(self) -> tuple:
        return (self.plate_size[0] + 2 * self.well_gap,
                self.plate_size[1] + 2 * self.well_gap)

    @property
    def plate_z0(self) -> float:
        """World z of the plate ORIGIN (bottom) at the pressed stop."""
        return self.deck_top + self.plate_lift

    @property
    def plate_top0(self) -> float:
        """World z of the plate TOP face at the pressed stop."""
        return self.plate_z0 + self.plate_size[2]

    @property
    def foul_q(self) -> float:
        return self.foul_q_frac * self.plate_travel

    def __post_init__(self) -> None:
        # Honesty-by-construction asserts (the physical claims the task rests on).
        # (1) dead-man force balance, >= 40 % margins BOTH ways:
        w_plate = self.plate_mass * _G
        assert self.spring_force >= 1.4 * w_plate, \
            "the pilot spring must beat the empty plate's weight by >= 40 %"
        lightest = min(self.pan_mass, self.pot_mass)
        assert (self.plate_mass + lightest) * _G >= 1.4 * self.spring_force, \
            "any single payload must pin the plate down with >= 40 % margin"
        # (2) foul geometry: raised threshold inside the travel, well above pressed
        assert self.pressed_q < self.foul_q / 2 < self.plate_travel, \
            "pressed / raised bands must be well separated inside the travel"
        # (3) ring top flush with the pressed plate top (payloads placed from above)
        assert abs((self.housing_base_h + self.ring_h)
                   - (self.plate_lift + self.plate_size[2])) < 1e-6, \
            "well ring top must be flush with the pressed plate's top"
        # (4) coexistence: pan on its half + pot in its slot NEVER touch (worst case)
        pan_inner_edge = self.pan_side_offset - self.pan_base_s / 2 - self.pan_jitter
        pot_r_corner = self.pot_base_s * math.sqrt(2.0) / 2
        pot_outer_edge = -self.pot_slot_x + pot_r_corner
        assert pan_inner_edge - pot_outer_edge >= 0.012, \
            "pot slot and pan half must coexist on the plate with >= 12 mm gap"
        # ... and both stay on the plate:
        assert self.pan_side_offset + self.pan_base_s / 2 + self.pan_jitter \
            <= self.plate_size[0] / 2 - 0.004, "pan base must stay on the plate"
        assert self.pot_slot_x + pot_r_corner <= self.plate_size[0] / 2 - 0.004, \
            "pot footprint must stay on the plate"
        assert pot_r_corner <= self.plate_size[1] / 2 - 0.004, \
            "pot footprint must fit the plate's depth"
        # (5) the pan's handle (local z 0.009..0.019, reach 0.105) passes OVER the
        # y ring walls when the pan sits on the plate:
        assert self.plate_lift + self.plate_size[2] + 0.009 \
            >= self.housing_base_h + self.ring_h + 0.004, \
            "pan handle must clear the well ring by >= 4 mm"
        assert 0.105 > self.well_inner[1] / 2, "handle indeed overhangs the y ring"
        # (6) the well ring never binds the plate laterally (joint holds it centered)
        assert self.well_gap >= 0.004, "keep >= 4 mm plate-to-ring lateral gap"
        # (7) plate never touches the housing base (joint stops are the only stops)
        assert self.plate_lift - self.housing_base_h >= 0.002, \
            "pressed plate bottom must hover >= 2 mm above the housing base"
        # (8) approach radius can't be earned at spawn: pot spawns far from the plate
        d = math.hypot(self.pot_start[0] - self.housing_pos[0],
                       self.pot_start[1] - self.housing_pos[1])
        assert d > self.near_dist + self.pot_start_jitter * math.sqrt(2.0) + 0.06, \
            "the pot must spawn well outside the approach radius"
        # (9) pan-clear distance clears the whole housing footprint
        assert self.clear_dist > math.hypot(*self.housing_outer) / 2 + 0.01, \
            "clear_dist must put the pan beyond the housing footprint"
        # (10) everything sits on the deck (worst-case footprints inside the slab)
        hx, hy = self.housing_pos
        dx, dy = self.deck_pos
        assert abs(hx - dx) + self.housing_outer[0] / 2 < self.deck_size[0] / 2 and \
            abs(hy - dy) + self.housing_outer[1] / 2 < self.deck_size[1] / 2, \
            "housing must sit fully on the deck"
        assert abs(self.pot_start[0] - dx) + 0.06 + self.pot_start_jitter \
            < self.deck_size[0] / 2 and \
            abs(self.pot_start[1] - dy) + 0.06 + self.pot_start_jitter \
            < self.deck_size[1] / 2, "pot must spawn fully on the deck"
        # (11) upright cone is satisfiable at rest on the flat plate
        assert self.pot_tilt_max_deg > 2.0
        # (12) the on-plate z band brackets a pot standing on the plate top and
        # EXCLUDES a pot perched on the pan (pan base 12 mm raises it out of band)
        pz = self.plate_size[2]
        assert self.on_z[0] < pz < self.on_z[1] < pz + 0.012, \
            "on-plate z band must accept plate-top rest and reject pan-top perch"


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("flame_keeper")
class FlameKeeperScene(BaseScene):
    cfg: FlameKeeperSceneCfg

    def __init__(self, cfg: FlameKeeperSceneCfg | None = None) -> None:
        super().__init__(cfg or FlameKeeperSceneCfg())

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
            "deck": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Deck",
                spawn=sim_utils.CuboidCfg(
                    size=c.deck_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.deck_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.deck_pos[0], c.deck_pos[1], c.deck_size[2] / 2)),
            ),
            "housing": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Housing",
                spawn=_compound_spawner_cfg("housing", _spawn_housing, c, 10.0,
                                            kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.housing_pos[0], c.housing_pos[1], c.deck_top)),
            ),
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate",
                spawn=_compound_spawner_cfg("plate", _spawn_plate, c, c.plate_mass,
                                            kinematic=False,
                                            lin_damp=c.plate_lin_damp, ang_damp=1.0),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.housing_pos[0], c.housing_pos[1], c.plate_z0)),
            ),
            "pot": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pot",
                spawn=_compound_spawner_cfg("pot", _spawn_pot, c, c.pot_mass,
                                            kinematic=False, lin_damp=0.3, ang_damp=0.5),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pot_start[0], c.pot_start[1], c.deck_top + 0.001)),
            ),
            "pan": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pan",
                spawn=_compound_spawner_cfg("pan", _spawn_pan, c, c.pan_mass,
                                            kinematic=False, lin_damp=0.3, ang_damp=0.5),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.housing_pos[0] + c.pan_side_offset, c.housing_pos[1],
                         c.plate_top0 + 0.001)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # the pilot spring (and any smoke wrench probe) must act every
                # solver iteration — the sim's own recommendation
                "enable_external_forces_every_iteration": True,
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
        n, dev = env.num_envs, env.device
        self.deck: RigidObject = env.iscene["deck"]
        self.housing: RigidObject = env.iscene["housing"]
        self.plate: RigidObject = env.iscene["plate"]
        self.pot: RigidObject = env.iscene["pot"]
        self.pan: RigidObject = env.iscene["pan"]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        # the constant pilot-spring wrench buffers (re-applied every post_step;
        # NOTE for probes: writing an external force to the PLATE overwrites the
        # spring — a probe on the plate must ADD spring_force to its +Z term)
        self._spring_f = torch.zeros(n, 1, 3, device=dev)
        self._spring_f[:, 0, 2] = self.cfg.spring_force
        self._spring_t = torch.zeros(n, 1, 3, device=dev)
        # foul (flame) bookkeeping + credit latches
        self._bot_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._up_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._armed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._foul = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l_near = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l_on = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l_swap = torch.zeros(n, dtype=torch.bool, device=dev)

    def _author_joints(self) -> None:
        """Per env: a prismatic Z joint housing->plate, travel [0, plate_travel].
        The joint's limits ARE the plate's stops; the housing never touches it.
        The housing is written at reset to its authored spawn pose only —
        kinematic-body0 joint anchors stay world-fixed after teleports (corpus
        lesson), so the housing is deliberately NOT jittered."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/plate_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Housing"])
            j.CreateBody1Rel().SetTargets([f"{base}/Plate"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.plate_lift))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(0.0)
            j.CreateUpperLimitAttr(c.plate_travel)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: plate pressed, pan on a RANDOM half of the plate
        (handle out over a side rim), pot on the counter (jitter + free yaw);
        flame bookkeeping and latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, pos, quat=None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat if quat is not None else torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=dev).expand(m, 4)
            body.write_root_state_to_sim(st, env_ids)

        # burn one draw: the FIRST post-seed draw is near-constant across seeds
        # (corpus lesson) — never let it pick the discrete side
        _ = torch.rand(m, device=dev)

        # --- housing + plate: authored nominal pose, plate pressed ---
        hpos = torch.zeros(m, 3, device=dev)
        hpos[:, 0], hpos[:, 1], hpos[:, 2] = c.housing_pos[0], c.housing_pos[1], c.deck_top
        write(self.housing, hpos)
        ppos = hpos.clone()
        ppos[:, 2] = c.plate_z0
        write(self.plate, ppos)

        # --- pan: random plate half, xy jitter, handle out over a y rim ---
        side = torch.where(torch.rand(m, device=dev) > 0.5, 1.0, -1.0)
        qpos = ppos.clone()
        qpos[:, 0] += side * c.pan_side_offset \
            + (torch.rand(m, device=dev) * 2 - 1) * c.pan_jitter
        qpos[:, 1] += (torch.rand(m, device=dev) * 2 - 1) * c.pan_jitter
        qpos[:, 2] = c.plate_top0 + 0.001
        hsign = torch.where(torch.rand(m, device=dev) > 0.5, 1.0, -1.0)
        pyaw = hsign * (math.pi / 2) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pan_yaw_jit_deg)
        write(self.pan, qpos, _qz(pyaw))

        # --- pot: on the counter, jitter + free yaw ---
        tpos = torch.zeros(m, 3, device=dev)
        tpos[:, 0] = c.pot_start[0] \
            + (torch.rand(m, device=dev) * 2 - 1) * c.pot_start_jitter
        tpos[:, 1] = c.pot_start[1] \
            + (torch.rand(m, device=dev) * 2 - 1) * c.pot_start_jitter
        tpos[:, 2] = c.deck_top + 0.001
        tyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pot_yaw_deg)
        write(self.pot, tpos, _qz(tyaw))

        # --- clear flame bookkeeping + latches ---
        for buf in (self._bot_streak, self._up_streak):
            buf[env_ids] = 0
        for latch in (self._armed, self._foul, self._l_near, self._l_on, self._l_swap):
            latch[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "housing": self.housing.data.root_state_w[env_ids].clone(),
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "pot": self.pot.data.root_state_w[env_ids].clone(),
            "pan": self.pan.data.root_state_w[env_ids].clone(),
            "bot_streak": self._bot_streak[env_ids].clone(),
            "up_streak": self._up_streak[env_ids].clone(),
            "armed": self._armed[env_ids].clone(),
            "foul": self._foul[env_ids].clone(),
            "l_near": self._l_near[env_ids].clone(),
            "l_on": self._l_on[env_ids].clone(),
            "l_swap": self._l_swap[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for name in ("housing", "plate", "pot", "pan"):
            getattr(self, name).write_root_state_to_sim(state[name], env_ids)
        self._bot_streak[env_ids] = state["bot_streak"]
        self._up_streak[env_ids] = state["up_streak"]
        self._armed[env_ids] = state["armed"]
        self._foul[env_ids] = state["foul"]
        self._l_near[env_ids] = state["l_near"]
        self._l_on[env_ids] = state["l_on"]
        self._l_swap[env_ids] = state["l_swap"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden COUNTER ({c.deck_size[0] * 100:.0f} x {c.deck_size[1] * 100:.0f} cm "
            f"slab) lies on the ground. On it sits a gas STOVE with a DEAD-MAN pilot "
            f"valve: its dark burner PLATE ({c.plate_size[0] * 100:.0f} x "
            f"{c.plate_size[1] * 100:.0f} cm) rides a vertical spring inside a "
            f"well — an orange pilot lamp glows on the stove's front. The pilot "
            f"spring pushes the plate up; only weight resting on the plate keeps it "
            f"pressed down, and the pilot FLAME stays lit only while the plate is "
            f"pressed. If the plate ever pops fully up for more than an instant, the "
            f"flame goes out PERMANENTLY for the episode — nothing relights it, and "
            f"the task is failed no matter what is arranged afterwards. Any single "
            f"pot or pan is heavy enough to hold the plate down.\n"
            f"A dark FRYING PAN currently rests on one half of the plate (which half "
            f"varies per episode; its handle points out over a rim), keeping the "
            f"flame alive. An octagonal aluminium MOKA POT stands elsewhere on the "
            f"counter (its spot and heading vary).\n"
            f"Goal: SWAP the cookware without losing the flame — end with the moka "
            f"pot standing alone on the pressed plate and the frying pan resting on "
            f"the counter, clear of the stove. Because an empty plate pops up almost "
            f"immediately, the pan may only be removed while something else already "
            f"holds the plate: place the pot on the plate's FREE half first, and "
            f"only then move the pan to the counter. Success: pot upright at rest on "
            f"the pressed plate, pan at rest on the counter at least "
            f"{c.clear_dist * 100:.0f} cm from the stove's center, flame still lit, "
            f"everything settled, hands off. Removing the pan first — even if the "
            f"final arrangement then looks identical — scores nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Swap the frying pan for the moka pot on the spring-loaded burner "
            "without letting the plate pop up: set the pot on the plate's free half "
            "first, then move the pan onto the counter. Keep the pilot flame lit; "
            "hands off at the end."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def plate_q(self) -> torch.Tensor:
        """(N,) plate joint travel in meters (0 = pressed stop)."""
        return self.plate.data.root_pos_w[:, 2] - self.housing.data.root_pos_w[:, 2] \
            - self.cfg.plate_lift

    def plate_pressed(self) -> torch.Tensor:
        return self.plate_q() < self.cfg.pressed_q

    def flame_alive(self) -> torch.Tensor:
        return ~self._foul

    def pot_upright(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        q = self.pot.data.root_quat_w
        ez = torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], 3)
        return quat_apply(q, ez)[:, 2] >= math.cos(math.radians(self.cfg.pot_tilt_max_deg))

    def _on_plate(self, body) -> torch.Tensor:
        """(N,) bool: body origin inside the plate-local cargo box (rides with
        the plate at any travel)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        lp = quat_apply_inverse(self.plate.data.root_quat_w,
                                body.data.root_pos_w - self.plate.data.root_pos_w)
        return (lp[:, 0].abs() <= c.on_x) & (lp[:, 1].abs() <= c.on_y) \
            & (lp[:, 2] > c.on_z[0]) & (lp[:, 2] < c.on_z[1])

    def pot_on_plate(self) -> torch.Tensor:
        return self._on_plate(self.pot)

    def pan_on_plate(self) -> torch.Tensor:
        return self._on_plate(self.pan)

    def pot_near(self) -> torch.Tensor:
        d = (self.pot.data.root_pos_w[:, :2]
             - self.plate.data.root_pos_w[:, :2]).norm(dim=-1)
        return d < self.cfg.near_dist

    def pan_clear(self) -> torch.Tensor:
        """(N,) bool: pan at rest height ON THE COUNTER, horizontally clear of
        the stove (beyond the housing footprint — a pan parked on the well ring
        or stacked on the pot fails the z band)."""
        c = self.cfg
        d = (self.pan.data.root_pos_w[:, :2]
             - self.plate.data.root_pos_w[:, :2]).norm(dim=-1)
        dz = self.pan.data.root_pos_w[:, 2] - self.env_origins[:, 2] - c.deck_top
        dxy = self.pan.data.root_pos_w[:, :2] - self.env_origins[:, :2]
        on_deck = (dxy[:, 0] - c.deck_pos[0]).abs() < c.deck_size[0] / 2 - 0.02
        on_deck &= (dxy[:, 1] - c.deck_pos[1]).abs() < c.deck_size[1] / 2 - 0.02
        return (d > c.clear_dist) & (dz > c.pan_deck_dz[0]) \
            & (dz < c.pan_deck_dz[1]) & on_deck

    def settled(self) -> torch.Tensor:
        c = self.cfg
        return (self.pot.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.pan.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.plate.data.root_lin_vel_w.norm(dim=-1) < c.plate_calm)

    # ----- flame + latch bookkeeping -------------------------------------------------------------
    def _update_latches(self) -> None:
        c = self.cfg
        q = self.plate_q()
        # arm the foul only after the plate has genuinely rested pressed once
        # (reset-transient-proof: the pan needs a few steps to seat)
        self._bot_streak = torch.where(q < c.pressed_q, self._bot_streak + 1,
                                       torch.zeros_like(self._bot_streak))
        self._armed |= self._bot_streak >= c.arm_streak
        # flame-out: plate fully raised for a sustained streak while armed
        self._up_streak = torch.where(q > self.foul_q_m, self._up_streak + 1,
                                      torch.zeros_like(self._up_streak))
        self._foul |= self._armed & (self._up_streak >= c.foul_streak)
        # credit latches — ALL gated on the flame being alive at that moment
        alive = ~self._foul
        calm_pot = self.pot.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        self._l_near |= alive & self.pot_near()
        self._l_on |= alive & self.pot_on_plate() & self.pot_upright() & calm_pot
        self._l_swap |= alive & self.pot_on_plate() & self.pot_upright() & calm_pot \
            & self.pan_clear() & self.plate_pressed()

    @property
    def foul_q_m(self) -> float:
        return self.cfg.foul_q

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        # the pilot spring: a constant +Z external force on the plate, refreshed
        # every step (world axis == body axis: the plate cannot rotate)
        self.plate.set_external_force_and_torque(self._spring_f, self._spring_t)
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool, all live except the (one-way, physical) flame latch: pot
        upright at rest ON the pressed plate, pan at rest on the counter clear
        of the stove, flame alive, everything settled and finite."""
        self._update_latches()
        finite = torch.isfinite(self.pot.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.pan.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.plate.data.root_pos_w).all(dim=-1)
        return self.flame_alive() & self.pot_on_plate() & self.pot_upright() \
            & ~self.pan_on_plate() & self.pan_clear() & self.plate_pressed() \
            & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10*near + 0.25*pot_on + 0.25*swap (latched;
        ~0 for doing nothing), capped at 0.60, FORCED TO 0 by a flame-out —
        and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_near * self._l_near.float() + c.w_on * self._l_on.float()
                + c.w_swap * self._l_swap.float()).clamp(max=0.60)
        base = torch.where(self._foul, torch.zeros_like(base), base)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="flame_keeper", robot="null"))
