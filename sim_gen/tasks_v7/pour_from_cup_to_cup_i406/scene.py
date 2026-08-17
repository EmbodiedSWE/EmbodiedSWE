"""RationTicketScene — right two toppled cups onto their goal pads and load each with
EXACTLY the number of balls its pad's printed dot ticket shows; leave every surplus
ball in the fixed bin (sim_gen task `pour_from_cup_to_cup_i406`).

Derived from rlbench/pour_from_cup_to_cup ("pour the contents of the source cup into
the target cup among distractor cups"), but the TRANSFER MODEL is replaced wholesale.
The seed's plan is: grasp the free-standing source cup, carry it above the target,
tilt it with the wrist, dump the whole content in one gravity discharge — a bulk
transfer whose only decision is which look-alike cup is the target. Here bulk transfer
is IMPOSSIBLE BY CONSTRUCTION and would be WRONG if it were possible:
  - the source BIN is bolted to the floor (kinematic): there is nothing to pick up
    and pour — content leaves the bin only as individually carried balls;
  - each goal pad displays a per-episode DOT TICKET (1..3 printed dots): the cup
    standing on that pad must contain EXACTLY that many balls. The bin holds MORE
    balls than the two tickets sum to, so "move everything" over-fills: the surplus
    balls must be LEFT IN THE BIN;
  - both cups start MOUTH-DOWN off their pads: nothing can enter a mouth-down cup,
    so righting-and-placing each cup on a pad is a physically forced FIRST step
    (an ordering the geometry enforces, not the rubric);
  - an upside-down cup COVERING the right number of balls on its pad counts nothing:
    containment is judged in the cup's body frame and gated on the cup standing
    upright.
A solver needs a different PLAN (read two ticket counts, right and place two
containers, allocate counted per-item transfers, stop while balls remain) and a
different code structure (quota perception + per-item loop with a stopping rule),
not different numbers on the seed's grasp-carry-tilt-pour plan.

Assets are fully procedural (compound spawners; explicit MassAPI + authored CoM and
inertia; friction materials bound to every collider — custom-spawner colliders
otherwise get the ~0.5 default with no restitution control):
  - bin: KINEMATIC low open tray (interior 260 x 180 mm, walls 30 mm) at the origin —
    the fixed source. Balls spawn on a jittered, permuted slot grid inside it.
  - cups: two DYNAMIC compound bodies (octagonal wall ring + cylindrical floor puck,
    inner flat radius 31 mm, height 85 mm, 150 g, CoM authored low). They spawn
    MOUTH-DOWN, each a random 10-15 cm outboard of its pad, random yaw.
  - pads: two collider-less kinematic white discs (painted floor rings) at y = +-0.30,
    xy-jittered per episode.
  - dot tickets: 3 collider-less black pucks per pad on the bin side of the pad;
    reset() surfaces `quota` of them and parks the rest underground. The ticket is
    the ONLY statement of the required count — a solver must read it visually.
  - balls: 8 identical orange spheres (r 14 mm, 30 g, velocity iters 4 against the
    GPU sphere-creep artifact, restitution 0). present = quota_a + quota_b + surplus
    (1..2); absent balls park in a ground depot.

Per-episode randomization (readback-verifiable): quota per pad (1..3, independent),
surplus count (1..2), which ball bodies are present, ball slot permutation + jitter,
pad xy jitter, cup offset direction/distance + yaw.

Rubric (0..1, anchored in the demonstrated solve trajectory):
  0.15 per pad — placed latch: an upright, settled cup stood centered on that pad
        (latched; two pads -> 0.30);
  0.45 * fill — fraction of the ticket total that is correctly delivered:
        sum_j min(count_on_pad_j, quota_j) / (quota_a + quota_b), counting only balls
        inside an UPRIGHT cup ON a pad (live, physical; overfill earns nothing);
  1.0 iff success(): each pad carries exactly one upright settled cup holding EXACTLY
        its quota, every remaining present ball rests inside the bin, all settled.
        Non-success caps at 0.75.
Null policy scores ~0 (cups mouth-down off-pad: no latch, no countable containment).

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


# ----- compound spawner helpers ----------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _root_xform(prim_path: str, translation, orientation):
    """Define an Xform root and author its (idempotent, single) translate/orient ops."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    return stage, xform.GetPrim()


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable | None,
             orient=None):
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(box.GetPrim())
    return box.GetPrim()


def _add_cyl_z(stage, path: str, *, center, radius, height, color,
               collide: Callable | None):
    """Cylinder along local Z (cup floor puck / painted discs)."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(cyl.GetPrim())
    return cyl.GetPrim()


def _rigid_dynamic(root, mass: float, *, lin_damp: float, ang_damp: float,
                   com: tuple | None = None, inertia: tuple | None = None) -> None:
    """Dynamic rigid-body armor on a compound root: MassAPI mass + authored local CoM
    (mass-only authoring leaves the CoM at the body origin) + authored diagonal
    inertia; damping; no sleeping while velocities are judged."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    mapi = UsdPhysics.MassAPI.Apply(root)
    mapi.CreateMassAttr(float(mass))
    if com is not None:
        mapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    if inertia is not None:
        mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _rigid_kinematic(root) -> None:
    from pxr import UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)


def _bind_mat(stage_path: str, mat_path: str, static: float, dynamic: float) -> None:
    """Author (once) and bind a friction material (custom-spawner colliders otherwise
    get the ~0.5 default with no restitution control — a bouncing ball never settles)."""
    import isaaclab.sim as sim_utils
    import omni.usd
    from isaaclab.sim.utils import bind_physics_material

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(mat_path).IsValid():
        sim_utils.spawn_rigid_body_material(
            mat_path,
            sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic,
                                           restitution=0.0))
    bind_physics_material(stage_path, mat_path)


def _spawn_bin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC source tray, bolted to the floor: local origin at the footprint
    centre on the ground. Low walls — an open ball reservoir, not a pourable vessel."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_kinematic(root)
    collide = _make_collide(c.contact_offset)
    grey = (0.30, 0.30, 0.33)
    ix, iy = c.bin_in_x_half, c.bin_in_y_half
    t, ft, wh = c.bin_wall_t, c.bin_floor_t, c.bin_wall_h
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, ft / 2),
             size=(2 * ix + 2 * t, 2 * iy + 2 * t, ft), color=grey, collide=collide)
    for tag, sx in (("wx_n", -1.0), ("wx_p", 1.0)):
        _add_box(stage, f"{prim_path}/{tag}", center=(sx * (ix + t / 2), 0.0, ft + wh / 2),
                 size=(t, 2 * iy + 2 * t, wh), color=grey, collide=collide)
    for tag, sy in (("wy_n", -1.0), ("wy_p", 1.0)):
        _add_box(stage, f"{prim_path}/{tag}", center=(0.0, sy * (iy + t / 2), ft + wh / 2),
                 size=(2 * ix, t, wh), color=grey, collide=collide)
    # bolt-head studs on the outer flanges: a visual statement that the bin is FIXED
    for k, (bx, by) in enumerate(((-ix, -iy - t), (ix, -iy - t), (-ix, iy + t), (ix, iy + t))):
        _add_cyl_z(stage, f"{prim_path}/bolt_{k}", center=(bx, by, ft + 0.003),
                   radius=0.006, height=0.006, color=(0.75, 0.72, 0.10), collide=None)
    _bind_mat(prim_path, "/World/mats/bin", 0.55, 0.50)
    return root


def _spawn_cup(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC cup: octagonal wall ring + cylindrical floor puck. Local origin at the
    cup's geometric CENTRE (half height on the axis) so mouth-down and upright poses
    share the same centre height. CoM authored toward the floor."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    m, r, h = c.cup_mass, c.cup_inner_r, c.cup_h
    # thin shell approx: Izz ~ m r^2, Ixx ~ m (3 r^2 + h^2) / 12  (order-of-magnitude
    # correct is all the solver needs — nothing is spun; authored so the authored CoM
    # never pairs with a shape-derived tensor)
    _rigid_dynamic(root, m, lin_damp=0.05, ang_damp=0.2,
                   com=(0.0, 0.0, -0.020),
                   inertia=(m * (3 * r * r + h * h) / 12.0,
                            m * (3 * r * r + h * h) / 12.0, m * r * r))
    collide = _make_collide(c.contact_offset)
    blue = (0.22, 0.36, 0.68)
    t, ft = c.cup_wall_t, c.cup_floor_t
    _add_cyl_z(stage, f"{prim_path}/floor", center=(0.0, 0.0, -h / 2 + ft / 2),
               radius=r + t, height=ft, color=blue, collide=collide)
    rw = r + t / 2                     # wall centreline radius
    wall_len = 2 * rw * math.tan(math.pi / 8) + 0.002   # chord + overlap
    for k in range(8):
        a = k * math.pi / 4
        _add_box(stage, f"{prim_path}/wall_{k}",
                 center=(rw * math.cos(a), rw * math.sin(a), ft / 2),
                 size=(t, wall_len, h - ft), color=blue, collide=collide,
                 orient=(math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)))
    _bind_mat(prim_path, "/World/mats/cup", 0.45, 0.40)
    return root


def _spawn_pad(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Collider-less kinematic painted goal disc (a thin plate would edge-catch a
    settling cup; a painted ring cannot)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_kinematic(root)
    _add_cyl_z(stage, f"{prim_path}/disc", center=(0.0, 0.0, 0.0008),
               radius=cfg.pad_r, height=0.0016, color=(0.92, 0.92, 0.92), collide=None)
    return root


def _spawn_dot(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Collider-less kinematic printed ticket dot."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_kinematic(root)
    _add_cyl_z(stage, f"{prim_path}/dot", center=(0.0, 0.0, 0.002),
               radius=cfg.dot_r, height=0.004, color=(0.05, 0.05, 0.05), collide=None)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bin" not in _SPAWNER_CACHE:

        def _mk(fn):
            @configclass
            class _Cfg(RigidObjectSpawnerCfg):
                func: Callable = clone(fn)
                contact_offset: float = 0.002
                bin_in_x_half: float = 0.13
                bin_in_y_half: float = 0.09
                bin_wall_t: float = 0.008
                bin_floor_t: float = 0.006
                bin_wall_h: float = 0.030
                cup_inner_r: float = 0.031
                cup_wall_t: float = 0.005
                cup_h: float = 0.085
                cup_floor_t: float = 0.005
                cup_mass: float = 0.15
                pad_r: float = 0.050
                dot_r: float = 0.007

            return _Cfg

        _SPAWNER_CACHE.update(bin=_mk(_spawn_bin), cup=_mk(_spawn_cup),
                              pad=_mk(_spawn_pad), dot=_mk(_spawn_dot))
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RationTicketSceneCfg(BaseCfg):
    """Config for `RationTicketScene`. The geometric claims the task rests on are
    asserted in `__post_init__` (quota fits the cup, containment test conservative,
    graspability, pads/spawn rings never pre-satisfy the rubric)."""

    # --- tunable: rubric thresholds -----------------------------------------------------------
    pad_tol: float = tunable(0.030)          # cup centre within this of the pad centre (m)
    upright_max_deg: float = tunable(12.0)   # cup axis within this of world-up = "standing"
    bottom_z_tol: float = tunable(0.010)     # cup bottom within this of the floor (m)
    settle_speed: float = tunable(0.06)      # max |lin vel| when judging (m/s)

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    quota_min: int = tunable(1)              # per-pad ticket lower bound
    quota_max: int = tunable(3)              # per-pad ticket upper bound
    surplus_min: int = tunable(1)            # balls beyond the ticket sum (must stay in bin)
    surplus_max: int = tunable(2)
    pad_jitter: float = tunable(0.030)       # +- xy jitter of each pad
    cup_off_min: float = tunable(0.10)       # cup spawn ring around its pad (outboard)
    cup_off_max: float = tunable(0.15)
    ball_jitter: float = tunable(0.008)      # +- xy jitter of each ball on its bin slot

    # --- info: layout ---------------------------------------------------------------------------
    bin_in_x_half: float = info(0.130)       # bin interior half extents
    bin_in_y_half: float = info(0.090)
    bin_wall_t: float = info(0.008)
    bin_floor_t: float = info(0.006)
    bin_wall_h: float = info(0.030)
    pad_base: tuple = info((0.02, 0.30))     # pads at (pad_base_x, +-pad_base_y)
    pad_r: float = info(0.050)
    dot_r: float = info(0.007)
    dot_pitch: float = info(0.024)           # ticket dot spacing along x
    dot_off_y: float = info(0.110)           # ticket row offset from the pad toward the bin
    # --- info: cup ------------------------------------------------------------------------------
    cup_inner_r: float = info(0.031)         # octagon flat inradius (inner wall face)
    cup_wall_t: float = info(0.005)
    cup_h: float = info(0.085)
    cup_floor_t: float = info(0.005)
    cup_mass: float = info(0.15)
    # --- info: balls ----------------------------------------------------------------------------
    n_balls: int = info(8)                   # ball bodies (= max present)
    ball_r: float = info(0.014)
    ball_mass: float = info(0.030)
    depot: tuple = info((1.3, 1.3))          # ground depot for absent balls
    contact_offset: float = info(0.002)
    # rubric weights
    w_place: float = info(0.15)              # per pad, latched
    w_fill: float = info(0.45)               # ticket-fulfilment fraction, live

    # Derived (filled in __post_init__).
    cup_outer_r: float = field(default=None, init=False)   # outer corner reach
    slots: tuple = field(default=None, init=False)         # bin slot grid (8 xy pairs)

    def __post_init__(self) -> None:
        r, t = self.cup_inner_r, self.cup_wall_t
        self.cup_outer_r = (r + t) / math.cos(math.pi / 8)  # octagon outer corner reach
        # Ticket feasibility: 3 balls fit inside the cup — two on the floor plus one
        # nested on top stays far below the rim (worst stack the drops can build).
        top = self.cup_floor_t + self.ball_r * (1.0 + math.sqrt(3.0)) + self.ball_r
        assert top < self.cup_h - 0.015, f"3-ball stack {top:.3f} too close to the rim"
        assert 2 * 2 * self.ball_r < 2 * self.cup_inner_r - 0.004, \
            "two balls must fit side by side on the cup floor"
        # Containment test honesty: a ball anywhere inside the octagon has its centre
        # within (corner reach - ball_r) of the axis, safely inside the inradius test.
        corner_in = (r) / math.cos(math.pi / 8)
        assert corner_in - self.ball_r < r - 0.002, \
            "in-cup radial test must be conservative wrt the octagon corners"
        # Graspability: cup body fits the 80 mm Franka jaw across the flats.
        assert 2 * (r + t) < 0.078, "cup too wide for the parallel jaw"
        # Pads far enough apart that one cup can never be on both, and the two goal
        # zones never merge under jitter.
        assert 2 * (self.pad_base[1] - self.pad_jitter) > \
            2 * (self.pad_tol + self.cup_outer_r) + 0.10, "pads too close"
        # A spawned (mouth-down, off-pad) cup can never start inside the pad tolerance.
        assert self.cup_off_min - self.pad_tol > 0.05, "cup spawn ring overlaps the pad zone"
        # Ticket demand never exceeds the stock: 2*quota_max + surplus_max == n_balls.
        assert 2 * self.quota_max + self.surplus_max <= self.n_balls, "not enough ball bodies"
        # Bin slot grid: 4 x 2, pitch wide enough that jittered balls never touch.
        xs, ys = (-0.09, -0.03, 0.03, 0.09), (-0.04, 0.04)
        assert 0.06 - 2 * self.ball_r - 2 * self.ball_jitter > 0.010, "bin slots too tight"
        assert xs[-1] + self.ball_r + self.ball_jitter < self.bin_in_x_half - 0.005
        assert ys[-1] + self.ball_r + self.ball_jitter < self.bin_in_y_half - 0.005
        self.slots = tuple((x, y) for y in ys for x in xs)
        # Ticket row sits clear of the pad zone and of the bin walls.
        assert self.dot_off_y > self.pad_tol + self.cup_outer_r + 0.02, \
            "ticket dots inside the cup's goal footprint"
        assert self.pad_base[1] - self.pad_jitter - self.dot_off_y > \
            self.bin_in_y_half + self.bin_wall_t + 0.03, "ticket dots reach the bin"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("ration_ticket_station")
class RationTicketScene(BaseScene):
    cfg: RationTicketSceneCfg

    def __init__(self, cfg: RationTicketSceneCfg | None = None) -> None:
        super().__init__(cfg or RationTicketSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()

        def geo(cfg_cls):
            return cfg_cls(
                contact_offset=c.contact_offset,
                bin_in_x_half=c.bin_in_x_half, bin_in_y_half=c.bin_in_y_half,
                bin_wall_t=c.bin_wall_t, bin_floor_t=c.bin_floor_t, bin_wall_h=c.bin_wall_h,
                cup_inner_r=c.cup_inner_r, cup_wall_t=c.cup_wall_t,
                cup_h=c.cup_h, cup_floor_t=c.cup_floor_t, cup_mass=c.cup_mass,
                pad_r=c.pad_r, dot_r=c.dot_r,
            )

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
            "bin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bin",
                spawn=geo(cls["bin"]),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
        }
        for j, tag in enumerate(("a", "b")):
            sy = 1.0 if j == 0 else -1.0
            out[f"pad_{tag}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad_" + tag.upper(),
                spawn=geo(cls["pad"]),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pad_base[0], sy * c.pad_base[1], 0.0)),
            )
            out[f"cup_{tag}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cup_" + tag.upper(),
                spawn=geo(cls["cup"]),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pad_base[0], sy * (c.pad_base[1] + 0.12), c.cup_h / 2 + 0.004),
                    rot=(0.0, 1.0, 0.0, 0.0)),  # mouth-down
            )
            for i in range(c.quota_max):
                out[f"dot_{tag}_{i}"] = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/Dot_" + tag.upper() + f"_{i}",
                    spawn=geo(cls["dot"]),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(c.pad_base[0] + (i - 1) * c.dot_pitch,
                             sy * (c.pad_base[1] - c.dot_off_y), 0.0)),
                )
        for i in range(c.n_balls):
            out[f"ball_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball_" + str(i),
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.2,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,  # GPU sphere-creep artifact fix
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.40, dynamic_friction=0.35, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.95, 0.55, 0.08)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.depot[0], c.depot[1] + 0.07 * i, c.ball_r + 0.002)),
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        n = env.num_envs
        dev = env.device
        self.bin: RigidObject = env.iscene["bin"]
        self.pads: list[RigidObject] = [env.iscene["pad_a"], env.iscene["pad_b"]]
        self.cups: list[RigidObject] = [env.iscene["cup_a"], env.iscene["cup_b"]]
        self.dots: list[list[RigidObject]] = [
            [env.iscene[f"dot_{tag}_{i}"] for i in range(c.quota_max)]
            for tag in ("a", "b")]
        self.balls: list[RigidObject] = [env.iscene[f"ball_{i}"] for i in range(c.n_balls)]
        self.env_origins = env.iscene.env_origins
        # Episode state.
        self.quota = torch.ones(n, 2, dtype=torch.long, device=dev)
        self.surplus = torch.ones(n, dtype=torch.long, device=dev)
        self.present = torch.ones(n, c.n_balls, dtype=torch.bool, device=dev)
        self._placed_latch = torch.zeros(n, 2, dtype=torch.bool, device=dev)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample quotas + surplus, surface the ticket dots, jitter the
        pads, drop the cups MOUTH-DOWN a random 10-15 cm outboard of their pads with
        free yaw, lay the present balls on a permuted jittered bin slot grid, park the
        absent balls in the depot, clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- bin: fixed home pose ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.bin.write_root_state_to_sim(st, env_ids)

        # --- quotas + surplus ---
        q = torch.randint(c.quota_min, c.quota_max + 1, (m, 2), device=dev)
        e = torch.randint(c.surplus_min, c.surplus_max + 1, (m,), device=dev)
        self.quota[env_ids] = q
        self.surplus[env_ids] = e
        n_present = (q.sum(dim=1) + e).clamp(max=c.n_balls)

        # --- pads + ticket dots + cups per side ---
        for j in range(2):
            sy = 1.0 if j == 0 else -1.0
            pad = torch.zeros(m, 2, device=dev)
            pad[:, 0] = c.pad_base[0]
            pad[:, 1] = sy * c.pad_base[1]
            pad += (torch.rand(m, 2, device=dev) * 2 - 1) * c.pad_jitter
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = pad
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            self.pads[j].write_root_state_to_sim(st, env_ids)

            for i in range(c.quota_max):
                vis = (q[:, j] > i).unsqueeze(1)                       # dot i shown?
                shown = torch.stack([pad[:, 0] + (i - 1) * c.dot_pitch,
                                     pad[:, 1] - sy * c.dot_off_y,
                                     torch.zeros(m, device=dev)], dim=-1)
                hidden = shown.clone()
                hidden[:, 2] = -0.08                                   # parked underground
                st = torch.zeros(m, 13, device=dev)
                st[:, 0:3] = origin + torch.where(vis, shown, hidden)
                st[:, 3] = 1.0
                self.dots[j][i].write_root_state_to_sim(st, env_ids)

            # cup: mouth-down on the outboard ring around its pad, free yaw
            ang = sy * (math.pi / 2) + (torch.rand(m, device=dev) * 2 - 1) * 0.8
            rad = c.cup_off_min + torch.rand(m, device=dev) * (c.cup_off_max - c.cup_off_min)
            yaw_half = (torch.rand(m, device=dev) * 2 - 1) * math.pi / 2
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = pad[:, 0] + rad * torch.cos(ang)
            st[:, 1] = pad[:, 1] + rad * torch.sin(ang)
            st[:, 2] = c.cup_h / 2 + 0.004
            # q = qz(yaw) * qx(pi): (cy, 0, 0, sy) * (0, 1, 0, 0) = (0, cy, sy, 0)
            st[:, 4] = torch.cos(yaw_half)
            st[:, 5] = torch.sin(yaw_half)
            st[:, 0:3] += origin
            self.cups[j].write_root_state_to_sim(st, env_ids)

        # --- balls: permuted slot grid inside the bin; absent -> depot ---
        rank = torch.rand(m, c.n_balls, device=dev).argsort(dim=1).argsort(dim=1)
        self.present[env_ids] = rank < n_present.unsqueeze(1)
        slots = torch.tensor(c.slots, device=dev)                       # (8, 2)
        zball = c.bin_floor_t + c.ball_r + 0.004
        for i in range(c.n_balls):
            xy = slots[rank[:, i]] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.ball_jitter
            inb = torch.stack([xy[:, 0], xy[:, 1],
                               torch.full((m,), zball, device=dev)], dim=-1)
            park = torch.tensor([c.depot[0], c.depot[1], c.ball_r + 0.002],
                                device=dev).expand(m, 3).clone()
            park[:, 1] = park[:, 1] + 0.07 * i
            pres = self.present[env_ids, i].unsqueeze(1)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.where(pres, inb, park)
            st[:, 3] = 1.0
            self.balls[i].write_root_state_to_sim(st, env_ids)

        self._placed_latch[env_ids] = False

    # ----- readings -----------------------------------------------------------------------------
    def _cup_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos_w (N,2,3), quat (N,2,4), up_z (N,2), |lin vel| (N,2)) for both cups."""
        from isaaclab.utils.math import quat_apply

        pos = torch.stack([b.data.root_pos_w for b in self.cups], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.cups], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.cups], dim=1)
        n = pos.shape[0]
        ez = torch.tensor([0.0, 0.0, 1.0], device=pos.device).expand(n * 2, 3)
        up = quat_apply(quat.reshape(n * 2, 4), ez).reshape(n, 2, 3)
        return pos, quat, up[:, :, 2], vel

    def _ball_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos_w (N,B,3), |lin vel| (N,B)) for all ball bodies."""
        pos = torch.stack([b.data.root_pos_w for b in self.balls], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.balls], dim=1)
        return pos, vel

    def cup_upright(self) -> torch.Tensor:
        """(N,2) bool: cup axis within `upright_max_deg` of world-up."""
        _p, _q, upz, _v = self._cup_tensors()
        return upz.clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.upright_max_deg))

    def on_pad(self) -> torch.Tensor:
        """(N,2,2) bool [pad j, cup k]: cup k standing on pad j — upright, centre
        within `pad_tol` of the pad centre, bottom on the floor, still."""
        c = self.cfg
        pos, _q, upz, vel = self._cup_tensors()
        pad_xy = torch.stack([p.data.root_pos_w[:, :2] for p in self.pads], dim=1)  # (N,2,2)
        d = (pos[:, None, :, :2] - pad_xy[:, :, None, :]).norm(dim=-1)              # (N,2,2)
        bottom = pos[:, :, 2] - upz * c.cup_h / 2                                   # (N,2)
        ok_cup = (self.cup_upright() & (bottom.abs() < c.bottom_z_tol)
                  & (vel < c.settle_speed))                                         # (N,2)
        return (d < c.pad_tol) & ok_cup[:, None, :]

    def in_cup(self) -> torch.Tensor:
        """(N,2,B) bool [cup k, ball i]: ball centre inside cup k's interior, judged
        in the CUP'S BODY FRAME and gated on the cup standing upright — an upside-down
        cup COVERING a ball contains nothing."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        bpos, _bv = self._ball_tensors()                    # (N,B,3)
        cpos, cquat, _upz, _cv = self._cup_tensors()        # (N,2,3), (N,2,4)
        n, b = bpos.shape[0], bpos.shape[1]
        rel = bpos[:, None, :, :] - cpos[:, :, None, :]     # (N,2,B,3)
        cq = cquat[:, :, None, :].expand(n, 2, b, 4).reshape(-1, 4)
        loc = quat_apply_inverse(cq, rel.reshape(-1, 3)).reshape(n, 2, b, 3)
        inside = ((loc[:, :, :, :2].norm(dim=-1) < c.cup_inner_r - 0.002)
                  & (loc[:, :, :, 2] > -c.cup_h / 2 + c.cup_floor_t - 0.004)
                  & (loc[:, :, :, 2] < c.cup_h / 2))
        return inside & self.cup_upright()[:, :, None]

    def in_bin(self) -> torch.Tensor:
        """(N,B) bool: ball centre inside the bin interior, below the wall top."""
        c = self.cfg
        bpos, _bv = self._ball_tensors()
        bp = self.bin.data.root_pos_w[:, None, :]
        dx = (bpos[:, :, 0] - bp[:, :, 0]).abs()
        dy = (bpos[:, :, 1] - bp[:, :, 1]).abs()
        dz = bpos[:, :, 2] - bp[:, :, 2]
        return ((dx < c.bin_in_x_half) & (dy < c.bin_in_y_half)
                & (dz > c.bin_floor_t - 0.002) & (dz < c.bin_floor_t + c.bin_wall_h + 0.02))

    def pad_counts(self) -> torch.Tensor:
        """(N,2) long: PRESENT balls inside the upright cup standing on each pad
        (0 where no cup stands on the pad)."""
        onp = self.on_pad()                                  # (N,2pads,2cups)
        inc = (self.in_cup() & self.present[:, None, :])     # (N,2cups,B)
        per_cup = inc.sum(dim=-1)                            # (N,2cups)
        return (onp.float() @ per_cup.float().unsqueeze(-1)).squeeze(-1).long()

    def balls_settled(self) -> torch.Tensor:
        """(N,) bool: every present ball |lin vel| below `settle_speed`."""
        _p, vel = self._ball_tensors()
        return ((vel < self.cfg.settle_speed) | ~self.present).all(dim=1)

    # ----- step-coupled bookkeeping (every substep) ---------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch each pad once an upright settled cup has stood centered on it."""
        onp = self.on_pad().any(dim=-1)                      # (N,2pads)
        self._placed_latch |= torch.nan_to_num(onp.float(), nan=0.0).bool()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: each pad carries EXACTLY ONE upright settled cup holding EXACTLY
        its ticket quota; every remaining present ball rests inside the bin; all
        settled. All clauses are live physical outcomes."""
        onp = self.on_pad()                                  # (N,2,2)
        one_each = (onp.sum(dim=-1) == 1).all(dim=-1)        # each pad exactly one cup
        distinct = (onp.sum(dim=1) <= 1).all(dim=-1)         # no cup on two pads
        exact = (self.pad_counts() == self.quota).all(dim=-1)
        in_any_cup = self.in_cup().any(dim=1)                # (N,B)
        leftovers_ok = (~self.present | in_any_cup | self.in_bin()).all(dim=1)
        bpos, _bv = self._ball_tensors()
        cpos, _cq, _u, _cv = self._cup_tensors()
        finite = (torch.isfinite(bpos).all(dim=-1).all(dim=-1)
                  & torch.isfinite(cpos).all(dim=-1).all(dim=-1))
        return (one_each & distinct & exact & leftovers_ok
                & self.balls_settled() & finite)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 per placed-pad latch + 0.45 * ticket-fulfilment
        fraction (min(count, quota) per pad over the ticket total; overfill earns
        nothing) — max 0.75 without success; exactly 1.0 iff success() holds live."""
        c = self.cfg
        fulfilled = torch.minimum(self.pad_counts(), self.quota).sum(dim=-1).float()
        total = self.quota.sum(dim=-1).float().clamp(min=1.0)
        base = (c.w_place * self._placed_latch.float().sum(dim=-1)
                + c.w_fill * fulfilled / total)
        return torch.where(self.success(), torch.ones_like(base), base)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bin": self.bin.data.root_state_w[env_ids].clone(),
            "pads": [p.data.root_state_w[env_ids].clone() for p in self.pads],
            "cups": [b.data.root_state_w[env_ids].clone() for b in self.cups],
            "dots": [[d.data.root_state_w[env_ids].clone() for d in row] for row in self.dots],
            "balls": [b.data.root_state_w[env_ids].clone() for b in self.balls],
            "quota": self.quota[env_ids].clone(),
            "surplus": self.surplus[env_ids].clone(),
            "present": self.present[env_ids].clone(),
            "placed_latch": self._placed_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.bin.write_root_state_to_sim(state["bin"], env_ids)
        for p, s in zip(self.pads, state["pads"]):
            p.write_root_state_to_sim(s, env_ids)
        for b, s in zip(self.cups, state["cups"]):
            b.write_root_state_to_sim(s, env_ids)
        for row, srow in zip(self.dots, state["dots"]):
            for d, s in zip(row, srow):
                d.write_root_state_to_sim(s, env_ids)
        for b, s in zip(self.balls, state["balls"]):
            b.write_root_state_to_sim(s, env_ids)
        self.quota[env_ids] = state["quota"]
        self.surplus[env_ids] = state["surplus"]
        self.present[env_ids] = state["present"]
        self._placed_latch[env_ids] = state["placed_latch"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A grey RATION BIN (interior {2 * c.bin_in_x_half * 100:.0f} x "
            f"{2 * c.bin_in_y_half * 100:.0f} cm, walls {c.bin_wall_h * 1000:.0f} mm) is "
            f"BOLTED to the floor at the centre of the station — it cannot be lifted, "
            f"slid or tipped. Inside it lie identical loose orange BALLS "
            f"({2 * c.ball_r * 1000:.0f} mm across). On either side of the bin "
            f"(front and back) a round WHITE PAD ({2 * c.pad_r * 100:.0f} cm across) is "
            f"painted on the floor, and printed on the floor between each pad and the bin "
            f"is that pad's TICKET: a row of 1 to 3 BLACK DOTS. Near each pad lies a blue "
            f"octagonal CUP ({2 * (c.cup_inner_r + c.cup_wall_t) * 1000:.0f} mm wide, "
            f"{c.cup_h * 1000:.0f} mm tall, open at one end) resting UPSIDE-DOWN "
            f"(mouth on the floor).\n"
            f"Goal, for EACH of the two pads: stand a cup UPRIGHT (mouth up) centred on "
            f"the pad, and put into that cup EXACTLY as many balls from the bin as that "
            f"pad's ticket shows dots — count the dots; the two tickets usually differ. "
            f"The bin holds MORE balls than the two tickets need: every ball you do not "
            f"deliver must be LEFT INSIDE the bin. Nothing can be put into a cup while it "
            f"is upside-down, so right each cup first. Success is judged on the settled "
            f"end state: one upright, still cup centred on each pad (within "
            f"{c.pad_tol * 100:.0f} cm), each holding exactly its ticket count — one ball "
            f"too many or too few on either pad fails — every other ball at rest in the "
            f"bin, and no ball loose on the floor. A cup left upside-down over balls "
            f"contains nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Stand each upside-down blue cup upright on its white pad, then drop into "
            "each cup exactly as many orange balls from the fixed bin as the black dots "
            "printed beside that pad. Leave every remaining ball inside the bin; do not "
            "over- or under-fill either cup."
        )


# Scene-level task: no robot in the slot; solve/smoke drive the objects directly.
register_env("simgen", lambda: EnvCfg(scene="ration_ticket_station", robot="null"))
