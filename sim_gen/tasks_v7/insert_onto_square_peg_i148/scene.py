"""HookEscapeScene — free the captive BLUE ring from an inverted-L hook rail and lay it
in the dish (sim_gen task `insert_onto_square_peg_i148`).

Derived from rlbench/insert_onto_square_peg, but STRATEGICALLY INVERTED: the seed picks
up a FREE square ring and drops it DOWN over the correct one of three colored pegs —
one straight vertical insertion, judged by the ring encircling the chosen peg, choice
expressed in the DESTINATION. Here the rings START captive on a peg (the seed's goal
state is this task's RESET state — a ring encircling a vertical post scores ~0), the
choice is expressed in the OBJECT (blue target vs orange decoy ring, same size), and
the load-bearing interaction is the inverse and path-constrained: the target ring must
be slid UP the vertical post, pitched ~90 degrees through the ELBOW of the rail, run
OUT along the horizontal arm and off its open tip — the only topological exit (the
rail is captive: pulled straight up, a ring jams under the arm; smoke force-probes
this) — then laid FLAT inside a round dish on the floor. The orange decoy must NOT
end in the dish; on episodes where it spawns stacked ABOVE the blue ring, rail
topology forces the solver to shepherd the decoy off the hook first.

Strategy vs the corpus tasks read this session:
  - pen_holder (packing suite): repeated tip-up insertions INTO a container — here
    nothing is inserted; the core interaction is a constrained EXTRACTION with a
    mid-path 90-degree reorientation.
  - ramrod ball eject (peg_insertion_side_i1, TASK.md): a rod is inserted through a
    tube to push a third judged body out — here there is no tool and no third body;
    the judged body IS the manipulated ring, and the constraint is a rail threading
    its bore, not a bore swallowing a rod.
  - latch_vault (screw_nail_i59, TASK.md): unlock-uncover-retrieve on prismatic
    joints — here there are no joints and nothing is covered; the "lock" is pure
    topology (a bent rail through a bore), and the escape needs continuous guided
    motion along a 3-segment path, not discrete latch strokes.
  - hood_prop (lift_peg_upright_i116): a peg props a falling lid, judged body held
    up by contact statics — here nothing is propped and the goal state is a free
    settled containment.

success(): blue ring settled FLAT inside the dish (position in the dish frame,
z-band on the floor, plane near-horizontal), fully off the rail, with the orange
decoy clear of the dish. score(): latched stage credit anchored in the demonstrated
solution — 0.25 * best on-rail climb up the post + 0.25 * best on-rail travel along
the arm + 0.15 once fully off the rail + 0.15 once inside the dish region; capped at
0.80; exactly 1.0 iff success(). Latches update every physics substep (post_step) so
credit never evaporates; null policy scores ~0 (rings rest at the post base below the
climb deadband).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - rack (KINEMATIC): dark plinth block, slick square post (18 mm) rising to an
    elbow at 0.336 m, slick square arm (18 mm) running 0.15 m horizontally from the
    elbow to an OPEN tip. Origin = post axis at the ground; arm along rack +x.
  - two rings (dynamic, 80 g each): octagonal annuli (bore flat-to-flat 72 mm —
    4x the rail, so the elbow passes; radial band 16 mm; plate 14 mm thick — a
    parallel-jaw grasp anywhere on the rim). BLUE = target, ORANGE = decoy. Mass,
    CoM and diagonal inertia authored explicitly in the spawner (custom spawn funcs
    apply no cfg schemas).
  - dish (KINEMATIC): shallow round dish (floor disc + octagon wall, inner inradius
    100 mm, walls 35 mm) standing on the floor ~0.38 m from the rack, never under
    the arm sweep (bearing kept >= 70 degrees off the arm direction).

Per-episode randomization (readback-verified in smoke): rack xy jitter + free yaw
(the arm direction must be READ, not assumed), dish bearing/radius around the rack,
ring stacking ORDER (blue on top -> extract it directly; orange on top -> the decoy
must come off first), and free ring yaws. Heavy imports (isaaclab, pxr) are deferred
so importing this module stays app-free.
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
_PLINTH_W = 0.16  # plinth footprint (square)
_PLINTH_H = 0.08  # plinth top = the rings' rest shelf
_RAIL_W = 0.018  # square rail cross-section (post AND arm)
_Z_ARM = 0.336  # arm axis height (arm spans z 0.327..0.345)
_POST_TOP = 0.345  # post top face = arm top face (flush elbow)
_ARM_TIP = 0.150  # arm runs rack-local x in [-0.009, 0.150]; the tip is OPEN

_BORE_IN = 0.036  # ring bore inradius (flat-to-flat 72 mm — 4x the rail width)
_BAND = 0.016  # ring radial band width
_RING_T = 0.014  # ring plate thickness (the parallel-jaw grasp)
_RING_OUT = (_BORE_IN + _BAND) / math.cos(math.pi / 8)  # outer circumradius ~0.0563
_RING_M = 0.080  # authored ring mass (kg)

_REST_Z0 = _PLINTH_H + _RING_T / 2 + 0.002  # bottom ring rest center z ~0.089
_REST_Z1 = _REST_Z0 + _RING_T + 0.003  # top-stacked ring rest center z ~0.106
_CLIMB_Z0 = 0.115  # climb-credit deadband: above BOTH rest heights

_DISH_FLOOR_R = 0.115  # dish floor disc radius
_DISH_FLOOR_TOP = 0.010  # dish floor top (dish frame; the dish stands on the ground)
_DISH_WALL_IN = 0.100  # octagon wall inner inradius
_DISH_WALL_H = 0.035  # wall height above the floor top


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
    """One USD physics material (friction is load-bearing: the rail is SLICK so a
    guided ring slides instead of friction-locking; ring/dish faces are grippy so
    the placed ring stays put)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None, yaw_deg: float = 0.0) -> None:
    """Author one colliding box child prim (translate -> rotate -> scale, authored
    once — idempotent per prim, the duplicate-xformOp trap)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw_deg:
        sxf.AddRotateZOp().Set(float(yaw_deg))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC hook rack: plinth + slick vertical post + slick
    horizontal arm with an OPEN tip. Origin = the post axis at ground level; the arm
    runs along rack-local +x."""
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    from pxr import UsdGeom

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(10.0)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)

    plinth_m = _friction_material(stage, f"{prim_path}/plinth_mat", 0.50, 0.45)
    slick = _friction_material(stage, f"{prim_path}/rail_mat", cfg.mu_rail_s, cfg.mu_rail_d)
    dark = (0.22, 0.22, 0.26)
    steel = (0.55, 0.57, 0.62)

    # plinth: the rings' rest shelf (top at _PLINTH_H)
    _box(stage, f"{prim_path}/plinth", (_PLINTH_W, _PLINTH_W, _PLINTH_H),
         (0.0, 0.0, _PLINTH_H / 2), dark, 0.0015, material=plinth_m)
    # post: z from plinth top to the (flush) elbow top
    _box(stage, f"{prim_path}/post", (_RAIL_W, _RAIL_W, _POST_TOP - _PLINTH_H),
         (0.0, 0.0, (_POST_TOP + _PLINTH_H) / 2), steel, 0.001, material=slick)
    # arm: x from -0.009 (flush with the post's -x face) to the OPEN tip at _ARM_TIP
    _box(stage, f"{prim_path}/arm", (_ARM_TIP + _RAIL_W / 2, _RAIL_W, _RAIL_W),
         ((_ARM_TIP - _RAIL_W / 2) / 2, 0.0, _Z_ARM), steel, 0.001, material=slick)
    return root


def _spawn_ring(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one dynamic octagonal ring (8 overlapping box segments — children of
    one rigid body never self-collide). Mass, CoM and diagonal inertia authored
    EXPLICITLY (custom spawn funcs apply no cfg schemas — density-derived mass and a
    default CoM would break the servo tuning)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(_RING_M))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    # annulus about its axis: Izz ~ m(ri^2+ro^2)/2; Ixx=Iyy ~ Izz/2 + m t^2/12
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(8.4e-5, 8.4e-5, 1.6e-4))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.08)
    # angular damping quiets the ring-on-edge swing when a ring dangles on the arm
    pxrb.CreateAngularDampingAttr(0.15)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)  # kills GPU edge-contact creep
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    ring_m = _friction_material(stage, f"{prim_path}/ring_mat", 0.30, 0.25)
    rm = _BORE_IN + _BAND / 2  # segment mid inradius
    side = 2.0 * rm * math.tan(math.pi / 8) + 0.004  # +overlap: seals the corners
    for k in range(8):
        ang = k * 45.0
        a = math.radians(ang)
        _box(stage, f"{prim_path}/seg{k}", (_BAND, side, _RING_T),
             (rm * math.cos(a), rm * math.sin(a), 0.0), cfg.color, 0.0015,
             material=ring_m, yaw_deg=ang)
    return root


def _spawn_dish(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC dish: floor disc + octagon wall. Origin = floor center
    at ground level."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(2.0)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    dish_m = _friction_material(stage, f"{prim_path}/dish_mat", 0.50, 0.45)

    cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/floor")
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(_DISH_FLOOR_R)
    cyl.CreateHeightAttr(_DISH_FLOOR_TOP)
    cyl.CreateDisplayColorAttr([Gf.Vec3f(0.92, 0.92, 0.90)])
    cxf = UsdGeom.Xformable(cyl.GetPrim())
    cxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, _DISH_FLOOR_TOP / 2))
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(cyl.GetPrim())
    px.CreateContactOffsetAttr(0.0015)
    px.CreateRestOffsetAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(
        dish_m, UsdShade.Tokens.weakerThanDescendants, "physics")

    rw = _DISH_WALL_IN + 0.006  # wall segment mid inradius (12 mm thick walls)
    side = 2.0 * rw * math.tan(math.pi / 8) + 0.006
    for k in range(8):
        ang = k * 45.0
        a = math.radians(ang)
        _box(stage, f"{prim_path}/wall{k}", (0.012, side, _DISH_WALL_H),
             (rw * math.cos(a), rw * math.sin(a), _DISH_FLOOR_TOP + _DISH_WALL_H / 2),
             (0.80, 0.80, 0.78), 0.0015, material=dish_m, yaw_deg=ang)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily, app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rack" not in _SPAWNER_CACHE:

        @configclass
        class RackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            mu_rail_s: float = 0.06
            mu_rail_d: float = 0.06

        @configclass
        class RingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ring)
            color: tuple = (0.10, 0.25, 0.85)

        @configclass
        class DishSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_dish)

        _SPAWNER_CACHE.update(rack=RackSpawnerCfg, ring=RingSpawnerCfg, dish=DishSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class HookEscapeSceneCfg(BaseCfg):
    """Config for `HookEscapeScene`. Honesty knobs are asserted in `__post_init__`:
    the elbow is passable by construction, the climb deadband sits above both rest
    heights (null policy ~0), and the dish tolerance admits every ring that is
    PHYSICALLY inside the walls, flat on the floor."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    on_rail_tol: float = tunable(0.060)  # ring center within this of the rail path = ON the rail
    free_tol: float = tunable(0.120)  # ring center beyond this of the rail path = FREE (hysteresis)
    dish_xy_tol: float = tunable(0.053)  # ring center within this of the dish axis
    dish_z_lo: float = tunable(0.012)  # ring center z-band over the dish floor (dish frame):
    dish_z_hi: float = tunable(0.028)  # flat-on-floor only; a ring perched on the rim is out
    flat_max_deg: float = tunable(20.0)  # ring plane within this of horizontal when judged in-dish
    settle_lin: float = tunable(0.05)  # settle gates when judging success (m/s, rad/s)
    settle_ang: float = tunable(0.50)
    decoy_clear_r: float = tunable(0.130)  # decoy exclusion zone: within this of the dish axis
    decoy_clear_z: float = tunable(0.080)  # ... and below this height violates "decoy clear"

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    rack_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the rack at reset (m)
    rack_yaw_deg: float = tunable(180.0)  # uniform +/- rack yaw (the ARM DIRECTION varies)
    dish_r: float = tunable(0.38)  # dish center distance from the rack
    dish_r_jitter: float = tunable(0.04)
    dish_bear_min_deg: float = tunable(70.0)  # min |bearing - arm direction| (dish never
    # sits under the arm sweep: a ring dropping off the tip cannot land in the dish)
    p_blue_top: float = tunable(0.5)  # probability the BLUE ring spawns on top of the stack

    # --- info: structure ---------------------------------------------------------------------
    plinth_h: float = info(_PLINTH_H)
    rail_w: float = info(_RAIL_W)
    z_arm: float = info(_Z_ARM)
    arm_tip: float = info(_ARM_TIP)
    bore_in: float = info(_BORE_IN)
    ring_out: float = info(_RING_OUT)
    ring_t: float = info(_RING_T)
    ring_m: float = info(_RING_M)
    rest_z0: float = info(_REST_Z0)
    rest_z1: float = info(_REST_Z1)
    climb_z0: float = info(_CLIMB_Z0)
    dish_floor_top: float = info(_DISH_FLOOR_TOP)
    dish_wall_in: float = info(_DISH_WALL_IN)
    dish_wall_h: float = info(_DISH_WALL_H)

    def __post_init__(self) -> None:
        # -- the elbow is passable by construction: bore 4x the rail width --
        assert _BORE_IN >= 3.5 * (_RAIL_W / 2), "bore too tight for the elbow turn"
        # -- null policy ~0: the climb deadband sits above BOTH rest heights --
        assert _CLIMB_Z0 >= _REST_Z1 + 0.008, "climb deadband must clear the stacked rest"
        assert self.free_tol >= self.on_rail_tol + 0.05, "free/on-rail need hysteresis"
        # -- dish honesty: every ring PHYSICALLY inside the walls, flat, counts --
        corner_reach = _DISH_WALL_IN / math.cos(math.pi / 8)
        assert corner_reach - _RING_OUT <= self.dish_xy_tol, \
            "a ring wedged into a dish corner would be physically inside but rejected"
        # -- the dish can hold the ring with slack; the ring fits a parallel jaw --
        assert _DISH_WALL_IN - _RING_OUT >= 0.03, "dish too tight for a sloppy drop"
        assert _RING_T <= 0.05, "ring plate must fit the parallel jaw"
        # -- the dish never sits under the arm sweep (min-bearing law of cosines) --
        r = self.dish_r - self.dish_r_jitter
        b = math.radians(self.dish_bear_min_deg)
        d2 = r * r + _ARM_TIP * _ARM_TIP - 2 * r * _ARM_TIP * math.cos(b)
        assert math.sqrt(d2) > _DISH_FLOOR_R + 2 * _RING_OUT + 0.04, \
            "a ring dropped off the arm tip could reach the dish"
        # -- rings rest below the elbow with a real climb between them --
        assert _Z_ARM - _CLIMB_Z0 > 0.15, "climb must be a real stroke"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("hook_escape")
class HookEscapeScene(BaseScene):
    cfg: HookEscapeSceneCfg

    def __init__(self, cfg: HookEscapeSceneCfg | None = None) -> None:
        super().__init__(cfg or HookEscapeSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        spawners = _spawner_classes()
        rack_cls, ring_cls, dish_cls = spawners["rack"], spawners["ring"], spawners["dish"]
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.60, dynamic_friction=0.50, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "rack": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=rack_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "dish": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dish",
                spawn=dish_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.38, 0.0, 0.0)),
            ),
            "ring_blue": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RingBlue",
                spawn=ring_cls(
                    color=(0.10, 0.25, 0.85),
                    mass_props=sim_utils.MassPropertiesCfg(mass=_RING_M),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, _REST_Z0)),
            ),
            "ring_orange": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RingOrange",
                spawn=ring_cls(
                    color=(0.90, 0.45, 0.05),
                    mass_props=sim_utils.MassPropertiesCfg(mass=_RING_M),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, _REST_Z1)),
            ),
        }
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # the solve drives the rings with per-substep wrenches; without this
                # an applied wrench is integrated only on the first substep
                "enable_external_forces_every_iteration": True,
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
        self.rack: RigidObject = env.iscene["rack"]
        self.dish: RigidObject = env.iscene["dish"]
        self.ring_blue: RigidObject = env.iscene["ring_blue"]
        self.ring_orange: RigidObject = env.iscene["ring_orange"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.blue_top = torch.zeros(n, dtype=torch.bool, device=dev)
        # latched stage credit for the BLUE ring (running max, per substep)
        self.l_climb = torch.zeros(n, device=dev)
        self.l_arm = torch.zeros(n, device=dev)
        self.l_free = torch.zeros(n, device=dev)
        self.l_dish = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter/yaw the rack (kinematic teleport), place the dish on
        a randomized bearing kept off the arm sweep, thread BOTH rings around the
        post resting on the plinth in a sampled ORDER, clear all latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        zeros = torch.zeros(m, device=dev)

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        def q_yaw(yaw: torch.Tensor) -> torch.Tensor:
            half = yaw / 2
            return torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)

        # --- rack: xy jitter + free yaw (the arm direction varies) ---
        rxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.rack_jitter
        ryaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rack_yaw_deg)
        write(self.rack, torch.cat([rxy, zeros.unsqueeze(-1)], dim=-1), q_yaw(ryaw))

        # --- dish: bearing at least dish_bear_min off the arm direction ---
        sign = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        bmin = math.radians(c.dish_bear_min_deg)
        off = bmin + torch.rand(m, device=dev) * (math.pi - bmin)
        bear = ryaw + sign * off
        rad = c.dish_r + (torch.rand(m, device=dev) * 2 - 1) * c.dish_r_jitter
        dpos = torch.stack([rxy[:, 0] + rad * torch.cos(bear),
                            rxy[:, 1] + rad * torch.sin(bear), zeros], dim=-1)
        dyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        write(self.dish, dpos, q_yaw(dyaw))

        # --- rings: both threaded around the post, stacked in a sampled order ---
        # (torch.rand comparison, not randint: the first randint after manual_seed
        # is near-constant across seeds)
        top = torch.rand(m, device=dev) < c.p_blue_top
        self.blue_top[env_ids] = top
        z_blue = torch.where(top, torch.full((m,), _REST_Z1, device=dev),
                             torch.full((m,), _REST_Z0, device=dev))
        z_orng = torch.where(top, torch.full((m,), _REST_Z0, device=dev),
                             torch.full((m,), _REST_Z1, device=dev))
        for body, zz in ((self.ring_blue, z_blue), (self.ring_orange, z_orng)):
            yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            pos = torch.stack([rxy[:, 0], rxy[:, 1], zz], dim=-1)
            write(body, pos, q_yaw(yaw))

        # --- clear the latches ---
        for lat in (self.l_climb, self.l_arm, self.l_free, self.l_dish):
            lat[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "rack": self.rack.data.root_state_w[env_ids].clone(),
            "dish": self.dish.data.root_state_w[env_ids].clone(),
            "ring_blue": self.ring_blue.data.root_state_w[env_ids].clone(),
            "ring_orange": self.ring_orange.data.root_state_w[env_ids].clone(),
            "blue_top": self.blue_top[env_ids].clone(),
            "latches": torch.stack([self.l_climb[env_ids], self.l_arm[env_ids],
                                    self.l_free[env_ids], self.l_dish[env_ids]], dim=-1),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.rack.write_root_state_to_sim(state["rack"], env_ids)
        self.dish.write_root_state_to_sim(state["dish"], env_ids)
        self.ring_blue.write_root_state_to_sim(state["ring_blue"], env_ids)
        self.ring_orange.write_root_state_to_sim(state["ring_orange"], env_ids)
        self.blue_top[env_ids] = state["blue_top"]
        lat = state["latches"]
        self.l_climb[env_ids] = lat[:, 0]
        self.l_arm[env_ids] = lat[:, 1]
        self.l_free[env_ids] = lat[:, 2]
        self.l_dish[env_ids] = lat[:, 3]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A steel hook rack stands on the floor: a dark plinth block "
            f"({_PLINTH_W * 100:.0f} cm square, {_PLINTH_H * 100:.0f} cm tall) with a slick "
            f"square post ({_RAIL_W * 1000:.0f} mm across) rising from its center to an elbow "
            f"{_Z_ARM * 100:.0f} cm up, where the rail bends 90 degrees into a horizontal arm "
            f"{_ARM_TIP * 100:.0f} cm long whose far end is OPEN. Two flat ring washers "
            f"(octagonal, {2 * _RING_OUT * 100:.0f} cm across, {_RING_T * 1000:.0f} mm thick, "
            f"bore {2 * c.bore_in * 1000:.0f} mm) hang captive around the post, resting "
            f"stacked on the plinth: one BLUE, one ORANGE. Which one lies on top changes per "
            f"episode, as do the rack's position and heading (so the arm can point anywhere), "
            f"and the position of a shallow round gray DISH standing on the floor "
            f"~{c.dish_r * 100:.0f} cm away — look first. The rings cannot be lifted off "
            f"sideways or straight up: the ONLY way a ring comes off is sliding UP the post, "
            f"tilting ~90 degrees to turn the elbow, traveling OUT along the horizontal arm, "
            f"and off its open tip.\n"
            f"Goal: free the BLUE ring from the rack and lay it FLAT on the dish floor, inside "
            f"the dish walls, and leave it settled there. The ORANGE ring is a decoy: it must "
            f"NOT end up in (or on) the dish — anywhere else on the floor or still on the rack "
            f"is fine. If the orange ring lies ABOVE the blue one, it blocks the exit path and "
            f"must be shepherded off the hook first (drop it on the floor away from the dish). "
            f"A blue ring left hanging anywhere on the rack, dropped on the floor, or perched "
            f"on the dish rim does not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the blue ring up the post, tilt it through the elbow, run it off the open "
            "end of the horizontal arm — the only way it comes free — and lay it flat inside "
            "the round dish. If the orange ring sits above it, take that one off first and "
            "drop it on the floor; the orange ring must not end up in the dish."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _rack_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points (N,3) -> rack frame."""
        q = self.rack.data.root_quat_w
        return _qapply(_qinv(q), pos_w - self.rack.data.root_pos_w)

    def rail_dist(self, body) -> torch.Tensor:
        """(N,) distance from the ring CENTER to the rail path polyline (rack frame):
        post segment (0,0,rest..z_arm) + arm segment ((0..arm_tip),0,z_arm)."""
        p = self._rack_local(body.data.root_pos_w)
        # post segment: x=y=0, z in [_PLINTH_H, _Z_ARM]
        z_cl = p[:, 2].clamp(_PLINTH_H, _Z_ARM)
        d_post = torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2 + (p[:, 2] - z_cl) ** 2)
        # arm segment: y=0, z=_Z_ARM, x in [0, _ARM_TIP]
        x_cl = p[:, 0].clamp(0.0, _ARM_TIP)
        d_arm = torch.sqrt((p[:, 0] - x_cl) ** 2 + p[:, 1] ** 2 + (p[:, 2] - _Z_ARM) ** 2)
        return torch.minimum(d_post, d_arm)

    def on_rail(self, body) -> torch.Tensor:
        return self.rail_dist(body) < self.cfg.on_rail_tol

    def free_of_rail(self, body) -> torch.Tensor:
        return self.rail_dist(body) > self.cfg.free_tol

    def _dish_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        q = self.dish.data.root_quat_w
        return _qapply(_qinv(q), pos_w - self.dish.data.root_pos_w)

    def _ring_up(self, body) -> torch.Tensor:
        """(N,3) the ring's bore axis (local +z) in world."""
        q = body.data.root_quat_w
        n = q.shape[0]
        ez = torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(n, 3)
        return _qapply(q, ez)

    def in_dish(self, body) -> torch.Tensor:
        """(N,) bool, geometric: ring center over the dish floor within `dish_xy_tol`
        of the axis, center z in the flat-on-floor band, ring plane near-horizontal.
        Honest by construction: any ring physically inside the walls, flat on the
        floor, counts (asserted in cfg); a ring perched on the rim sits above the
        z-band and is out."""
        c = self.cfg
        p = self._dish_local(body.data.root_pos_w)
        near = p[:, :2].norm(dim=-1) < c.dish_xy_tol
        zed = (p[:, 2] > c.dish_z_lo) & (p[:, 2] < c.dish_z_hi)
        flat = self._ring_up(body)[:, 2].abs() >= math.cos(math.radians(c.flat_max_deg))
        return near & zed & flat

    def decoy_clear(self) -> torch.Tensor:
        """(N,) bool: the ORANGE ring is clear of the dish (not inside, not perched
        on the rim — outside the exclusion cylinder around the dish axis)."""
        c = self.cfg
        p = self._dish_local(self.ring_orange.data.root_pos_w)
        near = (p[:, :2].norm(dim=-1) < c.decoy_clear_r) & (p[:, 2] < c.decoy_clear_z)
        return ~near

    def settled(self, body) -> torch.Tensor:
        return (body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    # ----- latched progress (every physics substep) -------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._latch()

    def _latch(self) -> None:
        """Running-max stage credit for the BLUE ring: on-rail climb up the post,
        on-rail travel along the arm, fully-off-the-rail, inside-the-dish region.
        Latched so correct behavior never loses credit."""
        c = self.cfg
        p = self._rack_local(self.ring_blue.data.root_pos_w)
        d = self.rail_dist(self.ring_blue)
        on = (d < c.on_rail_tol).float()
        climb = ((p[:, 2] - _CLIMB_Z0) / (_Z_ARM - _CLIMB_Z0)).clamp(0.0, 1.0) * on
        near_arm = ((p[:, 2] - _Z_ARM).abs() < 0.05).float() * on
        arm = (p[:, 0] / _ARM_TIP).clamp(0.0, 1.0) * near_arm
        free = (d > c.free_tol).float()
        dish = self.in_dish(self.ring_blue).float()
        self.l_climb = torch.maximum(self.l_climb, climb)
        self.l_arm = torch.maximum(self.l_arm, arm)
        self.l_free = torch.maximum(self.l_free, free)
        self.l_dish = torch.maximum(self.l_dish, dish)

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: BLUE ring settled flat inside the dish, fully off the rail,
        with the ORANGE decoy clear of the dish."""
        self._latch()
        return self.in_dish(self.ring_blue) & self.settled(self.ring_blue) \
            & self.free_of_rail(self.ring_blue) & self.decoy_clear()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25*climb + 0.25*arm + 0.15*freed + 0.15*in-dish
        (all latched running-max for the blue ring), capped at 0.80; exactly 1.0 iff
        success(). Null policy ~0 (rings rest below the climb deadband); the decoy's
        motion earns nothing."""
        self._latch()
        base = (0.25 * self.l_climb + 0.25 * self.l_arm
                + 0.15 * self.l_free + 0.15 * self.l_dish).clamp(max=0.80)
        return torch.where(self.success(), base.new_tensor(1.0), base)


# ----- pure-torch quaternion helpers (shared with solve/smoke) ---------------------------------
def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4) wxyz, pure torch."""
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a*b, wxyz."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


register_env("simgen", lambda: EnvCfg(scene="hook_escape", robot="null", env_spacing=3.0))
