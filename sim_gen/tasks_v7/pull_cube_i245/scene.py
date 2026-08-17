"""PinLockDrawerScene — extract two interlocked cross-pins in their geometry-forced
order, then pull the freed captive drawer to its hard stop with the red cargo cube
riding in its open-top bay (sim_gen task `pull_cube_i245`).

Derived from maniskill/pull_cube, but STRATEGICALLY different: the seed is one planar
non-prehensile act — hook a free cube with an L-tool and drag it a few centimeters into
a painted floor region; success is an xy-distance readout on the moved object itself.
Here the cube is CAPTIVE from the start (riding in a lidded drawer bay under a roof it
cannot pass) and is never the manipulated object at all: the goal predicate reads the
CONFIGURATION OF A MECHANISM — drawer at its out-stop, both lock pins fully extracted
from the channel volume — with the cube merely required to still be aboard. There is no
floor goal region anywhere; dragging the cube to any floor spot is constructed in smoke
and scores ~0. The plan is a three-step ORDERED disassembly whose order is enforced by
geometry, not by rubric bookkeeping: the yellow GUARD pin carries an L-arm that
overhangs the green LOCK pin's exit hole (the lock pin physically jams into the arm
after ~11 mm), the guard itself is LATCHED — two catch posts on the housing ledge
dead-stop its arm after ~4 mm of sideways travel at rest height, so a shove on the lock
pin that daisy-chains +y force through the arm cannot expel the guard; the guard's hole
is twice the pin height, and the only way out is LIFT to the hole ceiling (arm clears
the posts), slide ~4 cm raised, drop, slide free. Both pins cross through notches in
the drawer's rear rails, so the drawer is pinned shut (~5 mm of rattle) until both are
out. Teleport-open cheats fail the rubric because success demands the pins OUTSIDE the
channel volume, and seated pins are inside it by construction.

Strategy vs the corpus: `pull_cube_i20` (same seed family) banks mass to tip an
untouched beam — a torque threshold; `pull_cube_tool_i186` is a one-shot stored-energy
release with passive gravity transport. Neither is an ordered multi-stage interlock:
here every stage is a guided sliding extraction under contact constraint (pin through
holes, pin past an overhanging arm, drawer along rails over a plinth), the stages only
compose in one order, and the goal is the mechanism's end configuration.

Mechanics: NO joints — every constraint is contact geometry. A kinematic housing
(plinth, two side rails with pin holes, roof strips, an out-stop lintel, outboard
support ledges) captures a dynamic drawer that slides along -x between the rails until
its top lug hits the lintel (travel stop 0.16 m). Two dynamic square pins lie across
the channel through rail holes and drawer-rail notches at stations x = 0.095 (guard)
and x = 0.035 (lock); extraction is +y, sliding on the flush hole-bottom/ledge surface.
A `post_step` plant consumes `self.drive` (world-frame forces at the CoM of pin1 /
pin2 / drawer — the push a fingertip exerts); the plant pre-encodes world -> body frame
per substep (validated recipe), so every caller is frame-correct by contract. Never
call set_external_force_and_torque on the bodies directly.

Rubric (stateless — no latches; monotone along the demonstrated solve trajectory):
  pin_clear(k): every sample point along pin k's shaft lies OUTSIDE the channel volume
    (housing frame: x in [-0.28, 0.13], |y| <= 0.0545, z in [0.06, 0.145]). A seated
    pin is inside by construction; a pin dumped back into the channel or the bay is
    inside too.
  success(): drawer at its out-stop (d >= d_thresh, |y| centered, upright, on the
    plinth), the cube riding inside the bay (drawer frame), BOTH pins clear, and
    drawer + pins + cube settled.
  score() = 0.15*clear1 + 0.15*clear2 + 0.45*clamp((d - 0.02)/0.125, 0, 1)  (max 0.75);
    1.0 iff success(). Null ~0: seated pins are not clear and the locked drawer's
    ~5 mm rattle sits inside the 0.02 progress dead-zone.

Per-episode randomization (verified by readback in smoke.py): housing xy jitter + yaw,
cube position + yaw in the bay, pin seat y-jitter, drawer closed-position jitter.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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


# ----- geometry constants (single source of truth: spawners + cfg asserts + rubric) ------------
# Housing frame: origin at housing center on the ground, drawer travels -x, pins extract +y.
_PIN_W = 0.018  # square pin shaft cross-section
_PIN_Z = 0.089  # pin shaft axis height (= hole bottom 0.080 + half section)
_ST1 = 0.095  # GUARD pin (yellow, pin1) station x — extracted FIRST
_ST2 = 0.035  # LOCK pin (green, pin2) station x — blocked by pin1's arm until pin1 is out
_HOLE_HALF = 0.012  # rail pin-hole half-width in x (3 mm play per side around the pin)
_RAIL_Y = 0.062  # rail center |y|; inner faces +-0.0545, outer faces +-0.0695
_HOLE_BOT = 0.080  # rail hole bottom z == support-ledge top z (flush sliding surface)
_PLINTH_TOP = 0.06
_DRAWER_Z = 0.066  # drawer origin height seated on the plinth (plate is 12 mm thick)
_TRAVEL_STOP = 0.16  # drawer travel at the lintel hard stop (lug front 0.112 -> -0.048)
_BAY_CX = -0.05  # bay cavity center, drawer local x (cavity 0.07 x 0.07)
_PIN1_SHAFT_Y = (-0.075, 0.155)  # guard shaft span, housing y (seated, zero jitter)
_PIN2_SHAFT_Y = (-0.075, 0.093)  # lock shaft span (25 mm stub past the outer rail face)
_ARM_Y = (0.104, 0.120)  # guard L-arm world y span (overhangs pin2's exit path)
_ARM_X = (0.010, 0.055)  # guard L-arm world x span when seated
_ARM_Z = (0.084, 0.120)  # guard L-arm world z span (curtains the lock hole top to bottom)
_HOLE_TOP = 0.116  # rail hole ceiling (hole is 36 mm tall -> the guard pin can be LIFTED)
_LEDGE_Y = 0.149  # support ledge center y (span 0.068..0.230, top z = 0.080)
# Catch posts (kinematic, on the front ledge): the guard's arm JAMS into these under any
# +y shove at rest height — including a daisy-chain shove transmitted through the lock
# pin — unless the guard is first lifted to its hole ceiling (the un-latch move). They
# straddle the lock pin's exit corridor with 3 mm clearance per side.
_CATCH_X = ((0.010, 0.020), (0.050, 0.060))
_CATCH_Y = (0.124, 0.134)
_CATCH_Z = (0.080, 0.092)
# Channel volume for the pins_clear predicate (housing frame).
_CH_X = (-0.28, 0.13)
_CH_Y = 0.0545
_CH_Z = (0.06, 0.145)


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


def _friction_material(stage, path: str, static: float, dynamic: float):
    """One USD physics material (PhysX pair-averages the two prims' mu)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float, material=None) -> None:
    """Author one colliding box child prim (translate -> scale, authored once —
    idempotent per prim, the duplicate-xformOp trap)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_housing(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC housing: plinth, rear wall, two side rails whose mid strips
    leave one square pin hole per station, roof strips with a central lug lane, the
    out-stop lintel in the lane, and one outboard support ledge per station (top flush
    with the hole bottoms, so pins slide out fully supported). Origin = housing center
    on the ground."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(20.0)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)

    slick = _friction_material(stage, f"{prim_path}/slick_mat",
                               cfg.mu_channel_s, cfg.mu_channel_d)
    gray = (0.45, 0.45, 0.50)
    dark = (0.25, 0.25, 0.30)
    # plinth (x -0.32..0.16, y +-0.24, top z 0.06)
    _box(stage, f"{prim_path}/plinth", (0.48, 0.48, 0.06), (-0.08, 0.0, 0.03), dark,
         0.0015, material=slick)
    # rear wall (x 0.132..0.152, z 0.06..0.15) — closes the back of the channel
    _box(stage, f"{prim_path}/rear", (0.02, 0.106, 0.09), (0.142, 0.0, 0.105), gray,
         0.001, material=slick)
    for tag, sy in (("n", 1.0), ("s", -1.0)):
        y = sy * _RAIL_Y
        # low strip (z 0.06..0.08): its top face is the hole bottom the pins ride on
        _box(stage, f"{prim_path}/lo_{tag}", (0.40, 0.015, 0.020), (-0.06, y, 0.070),
             gray, 0.001, material=slick)
        # top strip (z 0.116..0.15)
        _box(stage, f"{prim_path}/hi_{tag}", (0.40, 0.015, 0.034), (-0.06, y, 0.133),
             gray, 0.001, material=slick)
        # mid strips (z 0.08..0.116) leaving one 24 mm x 36 mm hole per pin station
        # (36 mm tall so the guard pin can be LIFTED ~12 mm to unlatch before sliding)
        _box(stage, f"{prim_path}/ma_{tag}", (0.283, 0.015, 0.036), (-0.1185, y, 0.098),
             gray, 0.001, material=slick)
        _box(stage, f"{prim_path}/mb_{tag}", (0.036, 0.015, 0.036), (0.065, y, 0.098),
             gray, 0.001, material=slick)
        _box(stage, f"{prim_path}/mc_{tag}", (0.033, 0.015, 0.036), (0.1235, y, 0.098),
             gray, 0.001, material=slick)
        # roof strip (z 0.118..0.138), central |y| <= 0.015 lug lane left open
        _box(stage, f"{prim_path}/roof_{tag}", (0.24, 0.038, 0.020), (0.02, sy * 0.034, 0.128),
             gray, 0.001, material=slick)
    # out-stop lintel in the lug lane (+x face at -0.048 -> travel stop 0.16)
    _box(stage, f"{prim_path}/lintel", (0.016, 0.03, 0.032), (-0.056, 0.0, 0.134), dark,
         0.001, material=slick)
    # outboard support ledges (top z 0.080, flush with the hole bottoms; y 0.068..0.230)
    for tag, sx in (("1", _ST1), ("2", _ST2)):
        _box(stage, f"{prim_path}/ledge_{tag}", (0.06, 0.162, 0.020), (sx, _LEDGE_Y, 0.070),
             gray, 0.001, material=slick)
    # catch posts on the front ledge: at rest height the guard's arm jams into these
    # after ~4 mm of +y travel (so shoving the LOCK pin daisy-chains to a dead stop);
    # lifting the guard to its hole ceiling raises the arm clear — the un-latch move.
    for tag, (x0, x1) in (("a", _CATCH_X[0]), ("b", _CATCH_X[1])):
        _box(stage, f"{prim_path}/catch_{tag}",
             (x1 - x0, _CATCH_Y[1] - _CATCH_Y[0], _CATCH_Z[1] - _CATCH_Z[0]),
             ((x0 + x1) / 2, (_CATCH_Y[0] + _CATCH_Y[1]) / 2, (_CATCH_Z[0] + _CATCH_Z[1]) / 2),
             dark, 0.001, material=slick)
    return root


def _dyn_root(stage, prim_path: str, translation, orientation, mass: float):
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.2)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    # ZERO sleep/stabilization thresholds: the post_step force plant must keep acting.
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return root


def _spawn_drawer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the dynamic drawer: plate, open-top cargo bay, rear guide rails with one
    pin notch per station, front handle, and the top lug that runs the roof lane and
    hits the lintel. Origin = plate center (seated world z 0.066)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _dyn_root(stage, prim_path, translation, orientation, cfg.mass_props.mass)
    body = _friction_material(stage, f"{prim_path}/body_mat", cfg.mu_body_s, cfg.mu_body_d)
    bay = _friction_material(stage, f"{prim_path}/bay_mat", cfg.mu_bay_s, cfg.mu_bay_d)
    blue = (0.10, 0.25, 0.85)
    lite = (0.30, 0.45, 0.95)
    _box(stage, f"{prim_path}/plate", (0.24, 0.10, 0.012), (0.0, 0.0, 0.0), blue,
         0.001, material=body)
    # bay: cavity 0.07 x 0.07 centered at local x -0.05, walls 15 mm thick, 30 mm tall
    _box(stage, f"{prim_path}/bay_f", (0.015, 0.10, 0.03), (-0.0925, 0.0, 0.021), lite,
         0.001, material=bay)
    _box(stage, f"{prim_path}/bay_r", (0.015, 0.10, 0.03), (-0.0075, 0.0, 0.021), lite,
         0.001, material=bay)
    for tag, sy in (("a", 1.0), ("b", -1.0)):
        _box(stage, f"{prim_path}/bay_{tag}", (0.07, 0.015, 0.03), (-0.05, sy * 0.0425, 0.021),
             lite, 0.001, material=bay)
        # rear guide rails (z local 0.006..0.036) with notches at the pin stations
        _box(stage, f"{prim_path}/r0_{tag}", (0.021, 0.015, 0.03), (0.0105, sy * 0.0425, 0.021),
             blue, 0.001, material=body)
        _box(stage, f"{prim_path}/r1_{tag}", (0.032, 0.015, 0.03), (0.065, sy * 0.0425, 0.021),
             blue, 0.001, material=body)
        _box(stage, f"{prim_path}/r2_{tag}", (0.011, 0.015, 0.03), (0.1145, sy * 0.0425, 0.021),
             blue, 0.001, material=body)
    # handle (world z 0.065..0.110; 60 mm y-width for a pinch grasp)
    _box(stage, f"{prim_path}/handle", (0.02, 0.06, 0.045), (-0.13, 0.0, 0.0215), blue,
         0.001, material=body)
    # lug (world z 0.072..0.135): runs the roof lane, front face 0.112 hits the lintel
    _box(stage, f"{prim_path}/lug", (0.016, 0.02, 0.063), (0.120, 0.0, 0.0375), blue,
         0.001, material=body)
    return root


def _spawn_pin1(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the yellow GUARD pin: 230 mm square shaft (origin = shaft center) plus an
    L-arm plate that, when seated, overhangs the lock pin's exit hole (arm local
    x [-0.085, -0.04], y [0.064, 0.080], z [-0.005, 0.031])."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _dyn_root(stage, prim_path, translation, orientation, cfg.mass_props.mass)
    mat = _friction_material(stage, f"{prim_path}/pin_mat", cfg.mu_pin_s, cfg.mu_pin_d)
    yellow = (0.90, 0.80, 0.05)
    _box(stage, f"{prim_path}/shaft", (_PIN_W, 0.23, _PIN_W), (0.0, 0.0, 0.0), yellow,
         0.001, material=mat)
    _box(stage, f"{prim_path}/arm", (0.045, 0.016, 0.036), (-0.0625, 0.072, 0.013), yellow,
         0.001, material=mat)
    return root


def _spawn_pin2(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the green LOCK pin: a plain 168 mm square shaft (origin = shaft center);
    its 25 mm stub protrudes past the outer rail face for a pinch grasp."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _dyn_root(stage, prim_path, translation, orientation, cfg.mass_props.mass)
    mat = _friction_material(stage, f"{prim_path}/pin_mat", cfg.mu_pin_s, cfg.mu_pin_d)
    green = (0.10, 0.70, 0.20)
    _box(stage, f"{prim_path}/shaft", (_PIN_W, 0.168, _PIN_W), (0.0, 0.0, 0.0), green,
         0.001, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (explicit @configclass subclasses
    of RigidObjectSpawnerCfg, defined lazily so the module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "housing" not in _SPAWNER_CACHE:

        @configclass
        class HousingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_housing)
            mu_channel_s: float = 0.10
            mu_channel_d: float = 0.08

        @configclass
        class DrawerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_drawer)
            mu_body_s: float = 0.25
            mu_body_d: float = 0.20
            mu_bay_s: float = 0.45
            mu_bay_d: float = 0.40

        @configclass
        class Pin1SpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pin1)
            mu_pin_s: float = 0.25
            mu_pin_d: float = 0.20

        @configclass
        class Pin2SpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pin2)
            mu_pin_s: float = 0.25
            mu_pin_d: float = 0.20

        _SPAWNER_CACHE.update(housing=HousingSpawnerCfg, drawer=DrawerSpawnerCfg,
                              pin1=Pin1SpawnerCfg, pin2=Pin2SpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PinLockDrawerSceneCfg(BaseCfg):
    """Config for `PinLockDrawerScene`. The interlock honesty is asserted in
    `__post_init__`: the locked drawer's rattle stays inside the score dead-zone, the
    lock pin genuinely jams into the guard's arm under the full jitter budget, and the
    drawer cannot be lifted far enough to bypass the pins."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    d_thresh: float = tunable(0.145)  # success: drawer travel at least this (stop at 0.16)
    y_tol: float = tunable(0.02)  # success: drawer |y| in the housing frame below this
    z_tol: float = tunable(0.012)  # success: drawer origin z within this of seated height
    upright_min: float = tunable(0.95)  # success: drawer +z axis dot world z at least this
    bay_xy_tol: float = tunable(0.037)  # cube-in-bay window half-width (drawer frame)
    bay_z: tuple = tunable((0.01, 0.06))  # cube-in-bay z window (drawer frame)
    settle_lin: float = tunable(0.06)  # settle gate, above the GPU phantom-velocity floor
    settle_ang: float = tunable(0.8)
    prog_zero: float = tunable(0.02)  # drawer travel below this earns nothing (locked rattle)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    housing_jitter: float = tunable(0.03)  # uniform +- xy jitter of the housing at reset (m)
    housing_yaw_max: float = tunable(20.0)  # uniform +- housing yaw at reset (deg)
    cube_jitter: float = tunable(0.008)  # cube xy jitter inside the bay (m)
    pin_seat_jitter: float = tunable(0.002)  # pin seat y jitter (m)
    drawer_out_jitter: float = tunable(0.0025)  # drawer closed-position out-travel jitter (m)

    # --- info: structure ---------------------------------------------------------------------
    drawer_mass: float = info(0.30)
    pin_mass: float = info(0.12)
    cube_mass: float = info(0.03)
    cube_size: float = info(0.035)  # rides the 0.07 x 0.07 bay (>= 15 mm slack per side)
    pin_w: float = info(_PIN_W)
    travel_stop: float = info(_TRAVEL_STOP)
    st1: float = info(_ST1)  # guard pin station
    st2: float = info(_ST2)  # lock pin station
    pin1_seat_y: float = info(0.040)  # guard shaft center y when seated
    pin2_seat_y: float = info(0.009)  # lock shaft center y when seated
    pin_z: float = info(_PIN_Z)
    drawer_z: float = info(_DRAWER_Z)
    pin_travel_out: float = info(0.15)  # extraction travel that leaves a pin clear + supported
    lock_jam_travel: float = info(0.011)  # lock pin free travel before it jams into the arm
    guard_latch_travel: float = info(0.004)  # guard free travel before its arm hits the catches
    guard_lift: float = info(_HOLE_TOP - (_PIN_Z + _PIN_W / 2))  # ceiling-press un-latch lift
    guard_drop_travel: float = info(0.040)  # lifted travel after which the lift can be dropped
    mu_channel_s: float = info(0.10)  # housing sliding faces
    mu_channel_d: float = info(0.08)
    mu_body_s: float = info(0.25)  # drawer body / pins
    mu_body_d: float = info(0.20)
    mu_bay_s: float = info(0.45)  # bay faces (keep the cube aboard during the pull)
    mu_bay_d: float = info(0.40)
    mu_cube_s: float = info(0.45)
    mu_cube_d: float = info(0.40)
    base_pos: tuple = info((-0.45, 0.35))  # documented Franka base xy (TASK.md)
    reach: float = info(0.75)  # documented comfortable arm envelope from base_pos

    # Derived (filled in __post_init__).
    cube_bay_z: float = field(default=None, init=False)  # cube center resting in the bay

    def __post_init__(self) -> None:
        # cube rests on the bay floor: plate top (local 0.006) + half cube, world frame
        self.cube_bay_z = _DRAWER_Z + 0.006 + self.cube_size / 2
        # -- locked-drawer rattle must stay inside the score dead-zone --
        # notch edge at local x 0.049 hits the seated lock pin's +x face (0.044) after
        # ~5 mm; same for the guard pin's station.
        rattle = 0.049 - (_ST2 + _PIN_W / 2) + self.drawer_out_jitter
        assert rattle + 0.004 < self.prog_zero, "locked rattle would earn progress credit"
        # -- the drawer-out jitter must not press the notch edge into the seated pin --
        assert 0.049 - self.drawer_out_jitter >= (_ST2 + _PIN_W / 2) + 0.002, \
            "drawer_out_jitter would spawn the notch edge into the lock pin"
        # -- order enforcement: the arm must catch the lock pin under the jitter budget --
        gap = (_ARM_Y[0] - self.pin_seat_jitter) - (_PIN2_SHAFT_Y[1] + self.pin_seat_jitter)
        assert 0.004 < gap < 0.02, "guard arm must jam the lock pin after a short free travel"
        # ... and the arm must NOT catch pin2's stub at spawn (no resting contact)
        assert gap > 2 * 0.0015, "arm and stub would spawn in contact"
        # -- the LATCH: the guard's arm jams into the catch posts under any +y shove --
        # (this is what stops the daisy-chain: shoving pin2 transfers +y force into the
        # arm, but the arm dead-stops on the housing after ~4 mm; only a LIFT frees it)
        assert _CATCH_Y[0] - (_ARM_Y[1] + self.pin_seat_jitter) >= 0.002, \
            "guard arm would spawn in contact with the catch posts"
        assert _CATCH_Z[1] - _ARM_Z[0] >= 0.006, "catch/arm z engagement too shallow"
        # un-latch is possible: pressed to its hole ceiling, the arm clears the catches
        assert _ARM_Z[0] + self.guard_lift >= _CATCH_Z[1] + 0.004, \
            "lifted arm must clear the catch posts"
        assert self.guard_lift >= 0.014, "hole must give the guard a clear lift stroke"
        # ... and the lift can be dropped only after the arm has passed the far catch face
        assert _ARM_Y[0] + self.guard_drop_travel - self.pin_seat_jitter >= _CATCH_Y[1] + 0.002, \
            "guard_drop_travel would drop the arm back onto the catches"
        # the arm curtains the lock pin's hole top-to-bottom at every achievable lift of
        # pin2 (no ceiling-ride bypass), and the slit under the arm is far below pin width
        assert _ARM_Z[0] <= _HOLE_BOT + 0.006 and _ARM_Z[1] >= _HOLE_TOP, \
            "arm must curtain the lock pin's exit hole"
        assert _ARM_Z[0] - _HOLE_BOT < 0.5 * _PIN_W, "slit under the arm admits the lock pin"
        # the freed lock pin's corridor passes BETWEEN the posts with margin at max drift
        assert _CATCH_X[0][1] + 0.002 <= _ST2 - _HOLE_HALF, "catch post a pinches the corridor"
        assert _CATCH_X[1][0] - 0.002 >= _ST2 + _HOLE_HALF, "catch post b pinches the corridor"
        # ... while the arm overlaps BOTH posts (the jam cannot slip past one post)
        assert _ARM_X[0] <= _CATCH_X[0][0] + 0.002 and _ARM_X[1] >= _CATCH_X[1][0] + 0.004, \
            "arm must bear on both catch posts"
        # -- pins cannot be bypassed by lifting the drawer (roof headroom < pin bypass) --
        lift_headroom = 0.118 - (_DRAWER_Z + 0.036)  # roof bottom - bay wall top = 16 mm
        # rear-rail bottom (world 0.072) must rise past the pin top (0.098) to disengage
        lift_needed = (_PIN_Z + _PIN_W / 2) - (_DRAWER_Z + 0.006)
        assert lift_headroom < 0.7 * lift_needed, "drawer lift could bypass the pins"
        # -- extraction leaves the shaft clear of the channel and supported by the ledge --
        assert _PIN1_SHAFT_Y[0] + self.pin_travel_out > _CH_Y + 0.015, \
            "guard pin not clear after pin_travel_out"
        assert _PIN2_SHAFT_Y[0] + self.pin_travel_out > _CH_Y + 0.015, \
            "lock pin not clear after pin_travel_out"
        assert self.pin1_seat_y + self.pin_travel_out < _LEDGE_Y + 0.081 - 0.02, \
            "guard pin CoM would leave the support ledge"
        # -- success window sits inside the physical travel --
        assert self.prog_zero < self.d_thresh < _TRAVEL_STOP, "d_thresh outside travel"
        # -- embodiment: everything pinched fits an ~80 mm parallel jaw --
        assert _PIN_W < 0.08 and self.cube_size < 0.08 and 0.06 < 0.08
        # -- bay gives the cube generous slack even at 45 deg yaw --
        diag = self.cube_size * math.sqrt(2.0) / 2.0
        assert 0.035 - diag > self.cube_jitter, "cube jitter + free yaw must fit the bay"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("pin_lock_drawer")
class PinLockDrawerScene(BaseScene):
    cfg: PinLockDrawerSceneCfg

    def __init__(self, cfg: PinLockDrawerSceneCfg | None = None) -> None:
        super().__init__(cfg or PinLockDrawerSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "housing": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Housing",
                spawn=sp["housing"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mu_channel_s=c.mu_channel_s, mu_channel_d=c.mu_channel_d),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "drawer": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Drawer",
                spawn=sp["drawer"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.drawer_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mu_body_s=c.mu_body_s, mu_body_d=c.mu_body_d,
                    mu_bay_s=c.mu_bay_s, mu_bay_d=c.mu_bay_d),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, _DRAWER_Z)),
            ),
            "pin1": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pin1",
                spawn=sp["pin1"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.pin_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mu_pin_s=c.mu_body_s, mu_pin_d=c.mu_body_d),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(_ST1, 0.040, _PIN_Z)),
            ),
            "pin2": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pin2",
                spawn=sp["pin2"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.pin_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mu_pin_s=c.mu_body_s, mu_pin_d=c.mu_body_d),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(_ST2, 0.009, _PIN_Z)),
            ),
            "cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube_size,) * 3,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.85, 0.10, 0.10)),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_cube_s, dynamic_friction=c.mu_cube_d,
                        restitution=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        sleep_threshold=0.0, stabilization_threshold=0.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.001, rest_offset=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(_BAY_CX, 0.0, _DRAWER_Z + 0.006 + c.cube_size / 2 + 0.001)),
            ),
        }
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "enable_external_forces_every_iteration": True,
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.housing: RigidObject = env.iscene["housing"]
        self.drawer: RigidObject = env.iscene["drawer"]
        self.pin1: RigidObject = env.iscene["pin1"]
        self.pin2: RigidObject = env.iscene["pin2"]
        self.cube: RigidObject = env.iscene["cube"]
        self.env_origins = env.iscene.env_origins
        # Drive input: WORLD-frame forces at the CoM of (pin1, pin2, drawer). The plant
        # pre-encodes world -> body per substep, so callers stay frame-correct even
        # under the housing yaw. Never call set_external_force_and_torque directly.
        self.drive = torch.zeros(n, 3, 3, device=dev)
        self.drive_encode = True  # solve may flip as a last-resort frame fallback

    # ----- reset ------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the housing pose (xy jitter + yaw), then pose the
        drawer (closed, small out-travel jitter), both pins (seated through rails +
        drawer notches, y seat jitter) and the cube (in the bay, xy jitter + free yaw)
        in the housing frame, with sub-mm spawn lifts so nothing spawns flush."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        hxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.housing_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.housing_yaw_max)
        half = yaw / 2
        zeros = torch.zeros(m, device=dev)
        q_yaw = torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        def write(body, local: torch.Tensor, quat: torch.Tensor) -> None:
            wx = hxy[:, 0] + local[:, 0] * cy - local[:, 1] * sy
            wy = hxy[:, 1] + local[:, 0] * sy + local[:, 1] * cy
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = wx
            st[:, 1] = wy
            st[:, 2] = local[:, 2]
            st[:, 0:3] += origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        write(self.housing, torch.zeros(m, 3, device=dev), q_yaw)

        # drawer: closed with a small out-travel jitter (never toward the rear wall)
        d0 = torch.rand(m, device=dev) * c.drawer_out_jitter
        write(self.drawer,
              torch.stack([-d0, zeros, torch.full((m,), _DRAWER_Z + 0.0008, device=dev)],
                          dim=-1), q_yaw)

        # pins: seated at their stations, y seat jitter, 0.5 mm lift
        for pin, st_x, seat_y in ((self.pin1, _ST1, c.pin1_seat_y),
                                  (self.pin2, _ST2, c.pin2_seat_y)):
            jy = (torch.rand(m, device=dev) * 2 - 1) * c.pin_seat_jitter
            write(pin, torch.stack([torch.full((m,), st_x, device=dev), seat_y + jy,
                                    torch.full((m,), _PIN_Z + 0.0005, device=dev)], dim=-1),
                  q_yaw)

        # cube: in the bay (rides the drawer, so offset by the drawer's d0), free yaw
        jxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.cube_jitter
        cyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        chalf = yaw / 2 + cyaw / 2
        cquat = torch.stack([torch.cos(chalf), zeros, zeros, torch.sin(chalf)], dim=-1)
        write(self.cube,
              torch.stack([_BAY_CX - d0 + jxy[:, 0], jxy[:, 1],
                           torch.full((m,), c.cube_bay_z + 0.001, device=dev)], dim=-1),
              cquat)

        self.drive[env_ids] = 0.0

    # ----- frames / readings ------------------------------------------------------------------
    def _housing_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N, 3) world points -> housing frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.housing.data.root_quat_w,
                                  p_w - self.housing.data.root_pos_w)

    def housing_axes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """((N,3), (N,3)) housing +x and +y axes in world (pull directions: drawer
        along -x_axis, pins along +y_axis)."""
        from isaaclab.utils.math import quat_apply

        q = self.housing.data.root_quat_w
        n = q.shape[0]
        ex = torch.tensor([1.0, 0.0, 0.0], device=q.device).expand(n, 3)
        ey = torch.tensor([0.0, 1.0, 0.0], device=q.device).expand(n, 3)
        return quat_apply(q, ex), quat_apply(q, ey)

    def drawer_local(self) -> torch.Tensor:
        """(N, 3) drawer origin in the housing frame."""
        return self._housing_local(self.drawer.data.root_pos_w)

    def drawer_d(self) -> torch.Tensor:
        """(N,) drawer out-travel (positive = pulled out along -x)."""
        return -self.drawer_local()[:, 0]

    def pin_travel(self, pin) -> torch.Tensor:
        """(N,) pin extraction travel: housing-frame y minus the nominal seat."""
        seat = self.cfg.pin1_seat_y if pin is self.pin1 else self.cfg.pin2_seat_y
        return self._housing_local(pin.data.root_pos_w)[:, 1] - seat

    def pin_clear(self, pin) -> torch.Tensor:
        """(N,) bool: every sample point along the pin's shaft lies OUTSIDE the channel
        volume (housing frame). Seated pins are inside by construction; so is a pin
        dumped back anywhere between the rails (including into the bay)."""
        from isaaclab.utils.math import quat_apply

        half = 0.115 if pin is self.pin1 else 0.084
        offs = torch.linspace(-half, half, 9, device=pin.data.root_pos_w.device)
        q = pin.data.root_quat_w
        n = q.shape[0]
        clear = torch.ones(n, dtype=torch.bool, device=q.device)
        for o in offs:
            loc = torch.tensor([0.0, float(o), 0.0], device=q.device).expand(n, 3)
            p = pin.data.root_pos_w + quat_apply(q, loc)
            h = self._housing_local(p)
            inside = ((h[:, 0] > _CH_X[0]) & (h[:, 0] < _CH_X[1])
                      & (h[:, 1].abs() < _CH_Y)
                      & (h[:, 2] > _CH_Z[0]) & (h[:, 2] < _CH_Z[1]))
            clear &= ~inside
        return clear

    def cube_in_bay(self) -> torch.Tensor:
        """(N,) bool: cube center inside the bay cavity, in the DRAWER's frame (the
        cube must ride the drawer wherever it is)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        loc = quat_apply_inverse(self.drawer.data.root_quat_w,
                                 self.cube.data.root_pos_w - self.drawer.data.root_pos_w)
        return ((loc[:, 0] - _BAY_CX).abs() <= c.bay_xy_tol) \
            & (loc[:, 1].abs() <= c.bay_xy_tol) \
            & (loc[:, 2] >= c.bay_z[0]) & (loc[:, 2] <= c.bay_z[1])

    def settled(self, body) -> torch.Tensor:
        return (body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    def drawer_upright(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        q = self.drawer.data.root_quat_w
        ez = torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], 3)
        return quat_apply(q, ez)[:, 2] >= self.cfg.upright_min

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: drawer at its out-stop (travel >= d_thresh, centered, upright,
        seated on the plinth), the cube riding inside the bay, BOTH pins fully outside
        the channel volume, and drawer + pins + cube settled."""
        c = self.cfg
        loc = self.drawer_local()
        drawer_ok = (self.drawer_d() >= c.d_thresh) & (loc[:, 1].abs() <= c.y_tol) \
            & ((loc[:, 2] - _DRAWER_Z).abs() <= c.z_tol) & self.drawer_upright()
        return drawer_ok & self.cube_in_bay() \
            & self.pin_clear(self.pin1) & self.pin_clear(self.pin2) \
            & self.settled(self.drawer) & self.settled(self.cube) \
            & self.settled(self.pin1) & self.settled(self.pin2)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15 per pin fully extracted from the channel + 0.45 *
        drawer-travel progress past the locked rattle (dead-zone 0.02, full credit at
        d_thresh); 1.0 iff success(). Null ~0: seated pins are inside the channel and
        the locked drawer cannot leave the dead-zone. Stateless — a drawer teleported
        open with the pins still seated earns only the travel term (0.45) and never
        success."""
        c = self.cfg
        prog = ((self.drawer_d() - c.prog_zero) / (c.d_thresh - c.prog_zero)).clamp(0.0, 1.0)
        base = 0.15 * self.pin_clear(self.pin1).float() \
            + 0.15 * self.pin_clear(self.pin2).float() + 0.45 * prog
        return torch.where(self.success(), base.new_tensor(1.0), base)

    # ----- step-coupled mechanics (every substep) ---------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Force plant: apply the WORLD-frame drive rows at each body's CoM. The
        default set_external_force_and_torque call applies wrenches in the BODY frame
        on this stack, so pre-encode with quat_apply_inverse(q_now, f_world) — exact
        for these guided (yaw-only) sliders and immune to the is_global R_ref drag."""
        from isaaclab.utils.math import quat_apply_inverse

        n = self.env.num_envs
        dev = self.env.device
        zero = torch.zeros(n, 1, 3, device=dev)
        for i, body in enumerate((self.pin1, self.pin2, self.drawer)):
            f_w = self.drive[:, i, :]
            f = quat_apply_inverse(body.data.root_quat_w, f_w) if self.drive_encode else f_w
            body.set_external_force_and_torque(f.reshape(n, 1, 3), zero)

    # ----- state (full, restorable) -----------------------------------------------------------
    def _bodies(self) -> dict[str, Any]:
        return {"housing": self.housing, "drawer": self.drawer, "pin1": self.pin1,
                "pin2": self.pin2, "cube": self.cube}

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self._bodies().items()},
            "drive": self.drive[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.drive[env_ids] = state["drive"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A dark plinth on the floor carries a low sliding-drawer mechanism: a blue "
            "drawer (24 x 10 cm plate with a raised open-top cargo bay, a front handle "
            "and a top lug) rides between two gray side rails under a pair of gray "
            "roof strips. A red cube (3.5 cm) sits inside the drawer's bay; the roof "
            "covers the bay while the drawer is shut, and the central roof slot (3 cm) "
            "is narrower than the cube, so the cube cannot be reached or removed until "
            "the drawer is out. Two square cross-pins (1.8 cm) lie sideways through "
            "holes in both rails AND through notches in the drawer's rear guide rails, "
            "pinning the drawer shut: a YELLOW guard pin (rear station) whose flat "
            "L-arm overhangs the exit hole of a GREEN lock pin (front station). "
            "Because of that arm, the green pin can slide only about a centimeter "
            "before it jams — the yellow pin must come out first. The yellow pin is "
            "itself LATCHED: two small dark catch blocks stand on the ledge just "
            "beyond its arm, so slid sideways at rest height the arm jams into them "
            "after a few millimeters. Its hole is about twice the pin's height, so the "
            "un-latch move is to LIFT the yellow pin to the top of its hole (about "
            "2 cm), slide it out a few centimeters while raised so the arm passes over "
            "the catch blocks, then lower it and slide it the rest of the way. Both "
            "pins slide out sideways (toward the ledge side of the plinth) and land on "
            "a flat support ledge; the drawer then pulls out by its handle along the "
            "rails until its lug hits the stop lintel (about 16 cm of travel). "
            "Positions vary per episode: the whole plinth shifts and turns, the pins "
            "seat with jitter, and the cube starts anywhere in the bay.\n"
            "Goal: open the mechanism in its forced order — lift the YELLOW guard pin "
            "to the top of its hole and slide it out sideways until it is fully free "
            "of the rails, then slide the GREEN lock pin out between the catch blocks "
            "the same way, then pull the drawer out by its handle to the hard stop — "
            "finishing with the drawer resting at its stop (at least "
            f"{c.d_thresh * 100:.1f} cm of travel), the red cube still riding inside "
            "the bay, and BOTH pins resting fully outside the mechanism (on the ledge "
            "or the floor, never left in the channel or dropped into the bay). "
            "Pulling the drawer while any pin is seated only rattles it a few "
            "millimeters; forcing the green pin first jams it against the yellow "
            "pin's arm (and shoving harder only drives that arm into the catch "
            "blocks); sliding the yellow pin without lifting it jams it after a few "
            "millimeters. The cube itself never needs to be touched."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Unlock and open the pinned drawer in order: lift the yellow guard pin to "
            "the top of its hole and slide it out sideways past the catch blocks, "
            "then slide out the green lock pin, then pull the blue drawer out by its "
            "handle to the hard stop with the red cube still in its bay. Leave both "
            "pins fully outside the mechanism."
        )


register_env("simgen", lambda: EnvCfg(scene="pin_lock_drawer", robot="null", env_spacing=3.0))
