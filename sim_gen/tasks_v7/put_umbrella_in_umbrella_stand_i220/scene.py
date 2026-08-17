"""UmbrellaUnrackCradleScene — pull the umbrella OUT of the stand and lay it to rest
across the two V-notches of a display cradle (sim_gen task
`put_umbrella_in_umbrella_stand_i220`).

Derived from rlbench/put_umbrella_in_umbrella_stand, but STRATEGICALLY different — the
seed's GOAL state is this task's INITIAL state. The seed is transport + VERTICAL
INSERTION: pick the umbrella up, point its tip down, and plunge it into the open tube
of a floor stand; success is vertical containment in a receptacle. Here the umbrella
STARTS seated tip-down in one deep socket of a two-socket floor stand (a decoy CANE
occupies the other socket, assignment randomized), and the job is the inverse plus a
different placement class:

  1. EXTRACT — slide the umbrella straight UP out of its socket, a constrained guided
     translation of ~0.30 m before the tip clears the mouth (the seed never extracts);
  2. REORIENT — vertical to HORIZONTAL (the seed keeps the umbrella tip-down);
  3. LAY TO REST — set it down across the two V-notches of the cradle so the bare
     shaft nests in BOTH notches with the center of mass between the pillars — an OPEN
     two-point support equilibrium, not containment in anything.

A solver needs a different PLAN (up-and-out first, then a sideways set-down; the seed's
plan, inserting into a receptacle, scores ZERO here — the reset state itself is the
seed's success state and is worth nothing) and a different code structure (a
two-seat-point-to-axis-segment rest predicate + a horizontality cone, not
point-inside-tube containment). No hook, nothing hangs, nothing is inserted.

Rest-pose honesty (asserted in cfg.__post_init__): a shaft (r 9 mm) nested in a 90-deg
V reads seat-point-to-axis distance r*sqrt(2) = 12.7 mm < `seat_tol` = 30 mm; every
wrong rest reads far outside on at least one seat: lying on the floor ~110 mm, perched
across the arm tips ~94 mm, balanced crosswise in ONE notch leaves the OTHER seat
~100 mm away; a diagonal lean (one end grounded) violates the 10-deg horizontality
cone by construction (min lean angle ~13 deg). The two supports land on the BARE SHAFT
(same radius at both seats -> a level rest) with the CoM strictly between the pillars.

success(): both seat points within `seat_tol` of the umbrella's tip->handle axis
SEGMENT, axis horizontal within `tilt_max_deg`, umbrella settled (stillness SUSTAINED
for `settle_steps` substeps — an instantaneous velocity threshold fires mid-flight).

score() is graded and latched (credit never evaporates): 0.30 once the umbrella has
ever been fully EXTRACTED (its lower end above the socket mouth plane + margin) + 0.30
once it has ever been seated horizontal in both notches, cap 0.60; 1.0 iff success().
The null policy scores ~0: seated in the socket, the tip sits ~0.30 m below the
extraction latch line, and no cradled pose can fire the extraction latch (a horizontal
umbrella's ends lie far below `clear_z` — asserted).

Per-episode randomization (readback-verified in smoke): stand position + yaw, cradle
position + yaw, WHICH socket holds the umbrella vs the cane, and both items' free yaw.
Assets are fully procedural compound spawners (boxes, capsules, spheres — one rigid
body each; decorations authored idempotently). Heavy imports (isaaclab, pxr) are
deferred so importing this module — and registering the scene — stays app-free.
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


# ----- shared geometry constants (spawners + cfg assertions + solve/smoke) ---------------------
# Umbrella body frame: origin = CoM on the shaft axis, +z runs tip -> handle.
TIP_LOCAL_Z = -0.347  # canopy tip (the end seated at the socket floor)
TOP_LOCAL_Z = +0.335  # handle-knob top
CANOPY_R = 0.027  # furled-canopy capsule radius
SHAFT_R = 0.009  # bare shaft radius
SHAFT_SPAN = (-0.138, 0.291)  # bare-shaft cylinder z-extent (usable rest section)
# Stand: two square sockets on one base slab.
SOCK_HALF = 0.040  # socket inner half-width (clears the canopy with 13 mm slack)
SOCK_WALL_T = 0.012
SOCK_H = 0.30  # wall height above the base top
BASE_T = 0.03  # stand base slab thickness
MOUTH_Z = BASE_T + SOCK_H  # socket mouth plane (0.33)
SOCK_Y = 0.09  # socket centers at stand-local y = +-SOCK_Y
# Cradle: two pillars, each topped by a 90-deg V-notch; groove axis = cradle local y.
SEAT_Y = 0.10  # pillars at cradle-local y = +-SEAT_Y
SEAT_Z = 0.140  # V vertex (seat point) height above the cradle origin
ARM_L = 0.12
ARM_T = 0.014
R_REST = SHAFT_R * math.sqrt(2.0)  # shaft-center height above the vertex at rest (12.7 mm)
Y_ORG = 0.0  # umbrella-origin cradle-local y when properly laid: supports hit body
#              z = -+SEAT_Y = -+0.10, SYMMETRIC about the CoM (50/50 load split — an
#              asymmetric split was observed live to pitch the canopy end down and
#              walk the shaft off the open groove during hands-off load transfer)


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


def _material(stage, path: str, static: float = 0.6, dynamic: float = 0.5):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, color, contact_offset: float, material) -> None:
    from pxr import Gf, PhysxSchema, UsdPhysics, UsdShade

    prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float, material=None,
         quat=None) -> None:
    """One box child prim (translate -> orient -> scale, authored once — the duplicate
    xformOp trap is avoided by never re-authoring an existing prim's ops)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if quat is not None:
        w, x, y, z = (float(v) for v in quat)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    _collide(seg, color, contact_offset, material)


def _capsule(stage, path: str, radius: float, height: float, center, quat, color,
             contact_offset: float, material=None) -> None:
    """One capsule child prim, axis local +z, oriented by `quat` (w,x,y,z)."""
    from pxr import UsdGeom

    seg = UsdGeom.Capsule.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    _apply_xform(seg, center, quat)
    _collide(seg, color, contact_offset, material)


def _sphere(stage, path: str, radius: float, center, color, contact_offset: float,
            material=None) -> None:
    from pxr import UsdGeom

    seg = UsdGeom.Sphere.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    _apply_xform(seg, center, None)
    _collide(seg, color, contact_offset, material)


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float,
                kinematic: bool = False):
    """One rigid-body root Xform with the standard physics armor (zero sleep /
    stabilization thresholds: a sleeping body silently ignores applied wrenches, which
    the solve/smoke force probes depend on). The center of mass is AUTHORED at the
    body origin — with mass alone, PhysX volume-weights the collision shapes and puts
    the umbrella's CoM ~15 cm into the canopy (observed live: every 'balanced' rest
    tipped canopy-end-down)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(mass))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)  # capsule-on-box creep armor
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return root, pxrb


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC two-socket umbrella stand: one base slab and two deep square
    sockets (4 walls each) at local y = +-SOCK_Y. Origin = footprint center at ground
    level. Kinematic so reset() can re-place it per episode."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 25.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    dark, slate = (0.24, 0.26, 0.30), (0.46, 0.48, 0.53)
    _box(stage, f"{prim_path}/base", (0.14, 0.32, BASE_T), (0.0, 0.0, BASE_T / 2),
         dark, co, material=mat)
    a, t = SOCK_HALF, SOCK_WALL_T
    zc = BASE_T + SOCK_H / 2
    for s, nm in ((+1.0, "p"), (-1.0, "n")):
        ys = s * SOCK_Y
        _box(stage, f"{prim_path}/sock_{nm}_xp", (t, 2 * a + 2 * t, SOCK_H),
             (a + t / 2, ys, zc), slate, co, material=mat)
        _box(stage, f"{prim_path}/sock_{nm}_xn", (t, 2 * a + 2 * t, SOCK_H),
             (-a - t / 2, ys, zc), slate, co, material=mat)
        _box(stage, f"{prim_path}/sock_{nm}_yp", (2 * a, t, SOCK_H),
             (0.0, ys + a + t / 2, zc), slate, co, material=mat)
        _box(stage, f"{prim_path}/sock_{nm}_yn", (2 * a, t, SOCK_H),
             (0.0, ys - a - t / 2, zc), slate, co, material=mat)
    return root


def _spawn_cradle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC display cradle: base slab + two pillars at local y = +-SEAT_Y,
    each topped by a 90-deg V-notch (two 45-deg arms whose INNER faces meet at the
    vertex line, groove axis = local y). Origin = footprint center at ground level."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 25.0,
                              kinematic=True)
    # grippy varnished wood: axial creep along the open groove must die in friction
    mat = _material(stage, f"{prim_path}/phys_mat", static=0.9, dynamic=0.8)
    co = cfg.contact_offset
    walnut, oak = (0.36, 0.24, 0.14), (0.55, 0.40, 0.24)
    _box(stage, f"{prim_path}/base", (0.16, 0.34, 0.024), (0.0, 0.0, 0.012), walnut,
         co, material=mat)
    s45, c45 = math.sin(math.pi / 4), math.cos(math.pi / 4)
    # arm center = vertex + zhat'*(L/2) + xhat'*(T/2), so the INNER face contains the
    # vertex line; quat = rot_y(+-45 deg)
    dx = ARM_L / 2 * s45 + ARM_T / 2 * c45  # 0.0474
    dz = ARM_L / 2 * c45 - ARM_T / 2 * s45  # 0.0375
    qy_p = (math.cos(math.pi / 8), 0.0, math.sin(math.pi / 8), 0.0)
    qy_n = (math.cos(math.pi / 8), 0.0, -math.sin(math.pi / 8), 0.0)
    for s, nm in ((+1.0, "p"), (-1.0, "n")):
        yp = s * SEAT_Y
        _box(stage, f"{prim_path}/post_{nm}", (0.05, 0.028, 0.108),
             (0.0, yp, 0.024 + 0.054), oak, co, material=mat)
        _box(stage, f"{prim_path}/arm_{nm}_r", (ARM_T, 0.028, ARM_L),
             (dx, yp, SEAT_Z + dz), oak, co, material=mat, quat=qy_p)
        _box(stage, f"{prim_path}/arm_{nm}_l", (ARM_T, 0.028, ARM_L),
             (-dx, yp, SEAT_Z + dz), oak, co, material=mat, quat=qy_n)
    return root


def _spawn_umbrella(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The umbrella: furled RED canopy capsule at the -z (tip) end, bare grey shaft,
    black handle knob at +z. Origin = CoM on the axis; +z runs tip -> handle."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.20)
    pxrb.CreateAngularDampingAttr(1.5)
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    red, grey, black = (0.80, 0.10, 0.12), (0.60, 0.60, 0.63), (0.10, 0.10, 0.12)
    qid = (1.0, 0.0, 0.0, 0.0)
    _capsule(stage, f"{prim_path}/canopy", CANOPY_R, 0.146, (0.0, 0.0, -0.247), qid,
             red, co, material=mat)
    _capsule(stage, f"{prim_path}/shaft", SHAFT_R, 0.429, (0.0, 0.0, 0.0765), qid,
             grey, co, material=mat)
    _sphere(stage, f"{prim_path}/handle", 0.020, (0.0, 0.0, 0.315), black, co,
            material=mat)
    return root


def _spawn_cane(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The decoy cane: one plain tan stick + a round knob — no canopy anywhere.
    Origin = CoM on the axis; +z runs tip -> knob."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.50)
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    tan, brown = (0.76, 0.60, 0.35), (0.45, 0.28, 0.12)
    _capsule(stage, f"{prim_path}/shaft", 0.009, 0.618, (0.0, 0.0, -0.02),
             (1.0, 0.0, 0.0, 0.0), tan, co, material=mat)
    _sphere(stage, f"{prim_path}/knob", 0.018, (0.0, 0.0, 0.310), brown, co,
            material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "stand" not in _SPAWNER_CACHE:

        @configclass
        class StandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            contact_offset: float = 0.002

        @configclass
        class CradleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cradle)
            contact_offset: float = 0.002

        @configclass
        class UmbrellaSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_umbrella)
            mass: float = 0.30
            contact_offset: float = 0.002

        @configclass
        class CaneSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cane)
            mass: float = 0.26
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(stand=StandSpawnerCfg, cradle=CradleSpawnerCfg,
                              umbrella=UmbrellaSpawnerCfg, cane=CaneSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class UmbrellaUnrackCradleSceneCfg(BaseCfg):
    """Config for `UmbrellaUnrackCradleScene`. Rest-pose honesty, latch honesty and
    layout feasibility are asserted in `__post_init__`."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    seat_tol: float = tunable(0.030)  # seat point -> umbrella axis SEGMENT distance (m)
    tilt_max_deg: float = tunable(10.0)  # axis within this of horizontal
    settle_lin: float = tunable(0.02)  # max |lin vel| at judging (m/s) — a regulated
    #                                    0.04 m/s set-down descent must NOT read still
    settle_ang: float = tunable(0.30)  # max |ang vel| at judging (rad/s)
    settle_steps: int = tunable(30)  # substeps of SUSTAINED stillness (0.25 s at 120 Hz)
    clear_margin: float = tunable(0.030)  # extraction latch: lower end above mouth + this

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    stand_jitter: float = tunable(0.04)  # +-xy jitter of the stand
    stand_yaw_deg: float = tunable(25.0)  # +-yaw of the stand
    cradle_jitter: float = tunable(0.04)  # +-xy jitter of the cradle
    cradle_yaw_deg: float = tunable(25.0)  # +-yaw of the cradle
    socket_swap: bool = tunable(True)  # randomize WHICH socket holds the umbrella
    stand_pos: tuple = tunable((0.50, -0.26))  # stand footprint center (env frame)
    cradle_pos: tuple = tunable((0.46, 0.30))  # cradle footprint center (env frame)

    # --- info: structure (the geometry the spawners author) ----------------------------------
    tip_local_z: float = info(TIP_LOCAL_Z)
    top_local_z: float = info(TOP_LOCAL_Z)
    shaft_r: float = info(SHAFT_R)
    canopy_r: float = info(CANOPY_R)
    sock_half: float = info(SOCK_HALF)
    mouth_z: float = info(MOUTH_Z)
    sock_y: float = info(SOCK_Y)
    seat_y: float = info(SEAT_Y)
    seat_z: float = info(SEAT_Z)
    r_rest: float = info(R_REST)
    y_org: float = info(Y_ORG)
    umb_mass: float = info(0.30)
    cane_mass: float = info(0.26)
    contact_offset: float = info(0.002)

    def __post_init__(self) -> None:
        c = self
        # -- seat_tol is honest by construction --
        assert R_REST + 0.004 <= c.seat_tol, \
            "every physically-nested rest must read inside seat_tol"
        perch = ARM_L * math.cos(math.pi / 4) + SHAFT_R  # bridging the arm tips
        assert perch > c.seat_tol + 0.02, \
            "an across-the-arm-tips perch must be rejected with margin"
        floor_d = SEAT_Z - CANOPY_R  # seat point above a floor-lying umbrella (origin at ground)
        assert floor_d > c.seat_tol + 0.05, \
            "an umbrella lying on the floor must be rejected with margin"
        # crosswise balance in ONE notch leaves the OTHER seat 2*SEAT_Y away:
        assert 2 * SEAT_Y > c.seat_tol + 0.05, \
            "a crosswise one-notch balance must be rejected by the second seat"
        # -- both supports land on the bare shaft, CoM strictly between the pillars --
        z_a, z_b = -SEAT_Y - Y_ORG, SEAT_Y - Y_ORG
        assert SHAFT_SPAN[0] + 0.02 < z_a < -0.02 and 0.02 < z_b < SHAFT_SPAN[1] - 0.02, \
            "the notches must hit the bare shaft with the CoM between them"
        # -- horizontality cone: the shallowest one-end-grounded lean violates it --
        # (axis end on the floor, axis nested at a seat: rise SEAT_Z + R_REST over at
        # most the full axis length)
        lean = math.degrees(math.atan2(SEAT_Z + R_REST, TOP_LOCAL_Z - TIP_LOCAL_Z))
        assert lean > c.tilt_max_deg + 2.0, \
            "a diagonal lean (one end grounded) must violate the horizontality cone"
        assert c.tilt_max_deg >= 6.0, "a level two-point rest must pass with margin"
        # -- extraction is feasible and the latch is honest --
        assert SOCK_HALF > CANOPY_R + 2 * c.contact_offset + 0.006, \
            "the socket must clear the canopy with real slack"
        clear_z = MOUTH_Z + c.clear_margin
        assert BASE_T + 0.003 + 0.02 < clear_z, \
            "a seated umbrella's tip must sit far below the extraction latch line"
        assert SEAT_Z + R_REST + CANOPY_R < clear_z - 0.03, \
            "no cradled pose may fire the extraction latch"
        # -- embodiment: the exposed grasp band above the mouth is generous --
        grasp = (BASE_T + 0.003 - TIP_LOCAL_Z + SHAFT_SPAN[1]) - MOUTH_Z
        assert grasp > 0.25, "the seated umbrella must expose >= 25 cm of bare shaft"
        # -- layout: fixtures never overlap under max jitter --
        gap = (c.cradle_pos[1] - c.cradle_jitter - 0.17) - \
            (c.stand_pos[1] + c.stand_jitter + 0.16)
        assert gap > 0.05, "stand and cradle footprints must stay separated"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("umbrella_unrack_cradle")
class UmbrellaUnrackCradleScene(BaseScene):
    cfg: UmbrellaUnrackCradleSceneCfg

    def __init__(self, cfg: UmbrellaUnrackCradleSceneCfg | None = None) -> None:
        super().__init__(cfg or UmbrellaUnrackCradleSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        seat_z0 = BASE_T + 0.003 - TIP_LOCAL_Z  # seated umbrella origin height
        return {
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
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=sp["stand"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_pos[0], c.stand_pos[1], 0.0)),
            ),
            "cradle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cradle",
                spawn=sp["cradle"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cradle_pos[0], c.cradle_pos[1], 0.0)),
            ),
            "umbrella": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Umbrella",
                spawn=sp["umbrella"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.umb_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.umb_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_pos[0], c.stand_pos[1] + SOCK_Y, seat_z0)),
            ),
            "cane": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cane",
                spawn=sp["cane"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cane_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.cane_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_pos[0], c.stand_pos[1] - SOCK_Y, BASE_T + 0.341)),
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
        n, dev = env.num_envs, env.device
        self.stand: RigidObject = env.iscene["stand"]
        self.cradle: RigidObject = env.iscene["cradle"]
        self.umbrella: RigidObject = env.iscene["umbrella"]
        self.cane: RigidObject = env.iscene["cane"]
        self.env_origins = env.iscene.env_origins
        # which stand socket holds the umbrella (+1 -> local +y, -1 -> local -y)
        self.umb_side = torch.ones(n, device=dev)
        # progress latches (post_step): fully extracted; seated horizontal in the cradle
        self.extract_latch = torch.zeros(n, device=dev)
        self.cradle_latch = torch.zeros(n, device=dev)
        # sustained-stillness counter (consecutive still substeps; reset by motion)
        self.still_count = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: stand and cradle re-placed (jitter + yaw), the umbrella
        seated tip-down in a sampled socket and the cane in the other (both with free
        yaw), latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def qz(yaw: torch.Tensor) -> torch.Tensor:
            half = yaw / 2
            z = torch.zeros_like(half)
            return torch.stack([torch.cos(half), z, z, torch.sin(half)], dim=-1)

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # stand + cradle: nominal pos + jitter, +-yaw (kinematic, re-placed per episode)
        sp = torch.zeros(m, 3, device=dev)
        sp[:, 0], sp[:, 1] = c.stand_pos[0], c.stand_pos[1]
        sp[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.stand_jitter
        syaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.stand_yaw_deg)
        write(self.stand, sp, qz(syaw))

        cp = torch.zeros(m, 3, device=dev)
        cp[:, 0], cp[:, 1] = c.cradle_pos[0], c.cradle_pos[1]
        cp[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.cradle_jitter
        cyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.cradle_yaw_deg)
        write(self.cradle, cp, qz(cyaw))

        # socket assignment (torch.rand comparison — first-randint degeneracy trap)
        if c.socket_swap:
            side = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        else:
            side = torch.ones(m, device=dev)
        self.umb_side[env_ids] = side

        # seat both items tip-down in their sockets (stand frame -> world), free yaw
        cs, sn = torch.cos(syaw), torch.sin(syaw)

        def sock_xy(sgn: torch.Tensor) -> torch.Tensor:
            loc_y = sgn * SOCK_Y
            return torch.stack([sp[:, 0] - sn * loc_y, sp[:, 1] + cs * loc_y], dim=-1)

        up = torch.zeros(m, 3, device=dev)
        up[:, 0:2] = sock_xy(side)
        up[:, 2] = BASE_T + 0.003 - TIP_LOCAL_Z
        write(self.umbrella, up, qz((torch.rand(m, device=dev) * 2 - 1) * math.pi))

        kp = torch.zeros(m, 3, device=dev)
        kp[:, 0:2] = sock_xy(-side)
        kp[:, 2] = BASE_T + 0.003 + 0.338
        write(self.cane, kp, qz((torch.rand(m, device=dev) * 2 - 1) * math.pi))

        self.extract_latch[env_ids] = 0.0
        self.cradle_latch[env_ids] = 0.0
        self.still_count[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"stand": self.stand, "cradle": self.cradle,
                  "umbrella": self.umbrella, "cane": self.cane}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "umb_side": self.umb_side[env_ids].clone(),
            "extract_latch": self.extract_latch[env_ids].clone(),
            "cradle_latch": self.cradle_latch[env_ids].clone(),
            "still_count": self.still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"stand": self.stand, "cradle": self.cradle,
                  "umbrella": self.umbrella, "cane": self.cane}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.umb_side[env_ids] = state["umb_side"]
        self.extract_latch[env_ids] = state["extract_latch"]
        self.cradle_latch[env_ids] = state["cradle_latch"]
        self.still_count[env_ids] = state["still_count"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A dark two-socket UMBRELLA STAND sits on the floor: a base slab with two "
            f"deep square sockets ({2 * SOCK_HALF * 100:.0f} cm wide inside, "
            f"{SOCK_H * 100:.0f} cm deep, mouths {MOUTH_Z * 100:.0f} cm above the "
            "floor), side by side. One socket holds an UMBRELLA standing tip-down: "
            "its furled RED canopy is down inside the socket and a bare grey shaft "
            f"(~{2 * SHAFT_R * 1000:.0f} mm thick) with a black knob handle sticks "
            "~35 cm up out of the mouth. The other socket holds a plain tan walking "
            "CANE with a round brown knob — a decoy that counts for nothing. Which "
            "socket holds which changes every episode: the umbrella is the one whose "
            "red canopy is visible down in the socket.\n"
            "A wooden DISPLAY CRADLE stands on the floor nearby: a low base with two "
            "upright pillars, each topped by an open 90-degree V-NOTCH, the two "
            f"notches {2 * SEAT_Y * 100:.0f} cm apart at {SEAT_Z * 100:.0f} "
            "cm height, their grooves aligned with each other. The stand's and the "
            "cradle's positions and headings change every episode: read them by "
            "looking.\n"
            "Goal: pull the umbrella straight UP out of its socket (it must rise "
            "~30 cm before the canopy clears the mouth), turn it sideways, and lay it "
            "HORIZONTALLY across the cradle so its bare shaft rests seated in BOTH "
            "V-notches — red canopy overhanging one end, handle the other — then let "
            "go and leave it at rest. The umbrella left in a socket (where it "
            "started), lying on the floor, leaning against anything, balanced across "
            "a single notch, or still moving does NOT count; the cane in the cradle "
            "counts for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pull the red-canopied umbrella straight up out of its stand socket, then "
            "lay it horizontally across the wooden cradle so its shaft rests seated "
            "in both V-notches, and leave it settled there. Do not use the tan cane."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _axis_ends(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,3),(N,3): world positions of the umbrella's tip (-z) and handle-top
        (+z) ends — the judged axis segment."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        dev = self.env.device
        p, q = self.umbrella.data.root_pos_w, self.umbrella.data.root_quat_w
        tip = p + quat_apply(q, torch.tensor([0.0, 0.0, TIP_LOCAL_Z], device=dev).expand(n, 3))
        top = p + quat_apply(q, torch.tensor([0.0, 0.0, TOP_LOCAL_Z], device=dev).expand(n, 3))
        return tip, top

    def seat_points_w(self) -> torch.Tensor:
        """(N,2,3): world positions of the two V-notch seat points (vertex lines at
        cradle-local (0, +-SEAT_Y, SEAT_Z))."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        dev = self.env.device
        p, q = self.cradle.data.root_pos_w, self.cradle.data.root_quat_w
        out = []
        for s in (-1.0, +1.0):
            loc = torch.tensor([0.0, s * SEAT_Y, SEAT_Z], device=dev).expand(n, 3)
            out.append(p + quat_apply(q, loc))
        return torch.stack(out, dim=1)

    def seat_dists(self) -> torch.Tensor:
        """(N,2): distance from each seat point to the umbrella's tip->handle axis
        SEGMENT. A nested rest reads ~R_REST = 12.7 mm at both seats; every wrong
        rest reads far outside on at least one seat."""
        tip, top = self._axis_ends()
        seats = self.seat_points_w()  # (N,2,3)
        a = tip.unsqueeze(1)  # (N,1,3)
        ab = (top - tip).unsqueeze(1)  # (N,1,3)
        t = ((seats - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-9)
        proj = a + ab * t.clamp(0.0, 1.0).unsqueeze(-1)
        return (seats - proj).norm(dim=-1)

    def seated(self) -> torch.Tensor:
        """(N,) bool: both seat points within `seat_tol` of the umbrella axis."""
        return (self.seat_dists() < self.cfg.seat_tol).all(dim=1)

    def horizontal(self) -> torch.Tensor:
        """(N,) bool: umbrella axis within `tilt_max_deg` of horizontal."""
        tip, top = self._axis_ends()
        ax = top - tip
        ax = ax / ax.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return ax[:, 2].abs() <= math.sin(math.radians(self.cfg.tilt_max_deg))

    def low_end_height(self) -> torch.Tensor:
        """(N,): the LOWER umbrella end above the env floor — the extraction gauge
        (only a fully-extracted umbrella lifts its lower end above the mouth plane)."""
        tip, top = self._axis_ends()
        return torch.minimum(tip[:, 2], top[:, 2]) - self.env_origins[:, 2]

    def _still_now(self) -> torch.Tensor:
        """(N,) bool: instantaneously below the stillness thresholds (never judge on
        this directly — it fires mid-flight and at swing turning points)."""
        c = self.cfg
        return (self.umbrella.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.umbrella.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def settled(self) -> torch.Tensor:
        """(N,) bool: stillness SUSTAINED for `settle_steps` consecutive substeps
        (counter kept in post_step)."""
        return self.still_count >= float(self.cfg.settle_steps)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch each demonstrated stage every physics substep: fully extracted from
        the socket; seated horizontal in both notches."""
        clear_z = MOUTH_Z + self.cfg.clear_margin
        self.extract_latch = torch.maximum(
            self.extract_latch, (self.low_end_height() > clear_z).float())
        self.cradle_latch = torch.maximum(
            self.cradle_latch, (self.seated() & self.horizontal()).float())
        self.still_count = torch.where(self._still_now(), self.still_count + 1.0,
                                       torch.zeros_like(self.still_count))

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the umbrella rests horizontal, seated in BOTH V-notches,
        settled. Judged on the umbrella by identity — the cane can never substitute."""
        return self.seated() & self.horizontal() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.30 once ever fully EXTRACTED (lower end above the
        socket mouth plane) + 0.30 once ever seated horizontal in both notches (cap
        0.60); 1.0 iff success(). Latched — credit never evaporates; the null policy
        scores ~0 (seated in the socket, the tip sits ~0.30 m below the latch line)."""
        base = (0.30 * self.extract_latch + 0.30 * self.cradle_latch).clamp(0.0, 0.60)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="umbrella_unrack_cradle", robot="null",
                                      env_spacing=3.0))
