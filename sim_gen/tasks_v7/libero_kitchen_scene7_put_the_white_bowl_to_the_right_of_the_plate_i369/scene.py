"""DieRollMatchScene — tip-roll the oversized six-color die until its top face matches
the displayed reference die, then slide it onto the crimson serving mat (sim_gen task
`libero_kitchen_scene7_put_the_white_bowl_to_the_right_of_the_plate_i369`).

Derived from libero_90/kitchen_scene7_put_the_white_bowl_to_the_right_of_the_plate,
but STRATEGICALLY different: the seed grasps a free bowl and translates it to a
position RELATIVE to a plate — a pure pick-and-place whose goal lives in position
space. Here the manipulated object is an oversized MENU DIE (a 140 mm cube — wider
than any parallel jaw, so it cannot be grasped or lifted), and the goal lives in
ORIENTATION space: per episode a small kinematic REFERENCE die on a pedestal is shown
with one face up, and the big die must end up flat on the crimson serving mat with
its matching color face UP. Sliding or spinning the die on the counter can NEVER
change which face is up — the only way to reach the goal orientation is to TIP the
die over its bottom edges, one quarter-roll at a time (rolling toward a direction u
brings the face that faced -u up; a target currently facing DOWN needs two rolls).
A solver therefore needs a different PLAN (read the reference face, plan a
quarter-roll sequence through the cube's rotation group, execute tip-rolls without
overshooting into chain-rolls, then push-slide the die onto the mat at a height low
enough not to tip it) and a different code STRUCTURE (SO(3) face bookkeeping +
non-prehensile edge-roll execution — not grasp-transport-place).

success(): die FLAT (top-face dominance within `flat_max_deg` of world-up), top face
== the sampled target face (the reference die's up face), die center within
`place_tol` (Chebyshev) of the crimson mat center, resting at counter height
(z band), and PERSISTENTLY still (stillness counter-latch). Judged purely on the
settled physical outcome, like the seed.

score(), latched (credit never evaporates): 0.15 once the die has ever RESTED FLAT
on a face different from its initial top (the first successful roll); +0.25 once the
die has ever rested flat with the TARGET face up (orientation solved); +0.20 x the
running-max normalized approach toward the mat center; 1.0 iff success(). Null
policy ~0 (the die spawns far from the mat with a non-target face up).

The physics margins are asserted in cfg.__post_init__ from first principles:
  - tip-before-slide: pushing at `roll_push_h` the tipping force ~0.58 mg is well
    below the sliding threshold mu_s*mg = 0.9 mg, so a high push rolls the die;
  - slide-without-tip: pushing at `slide_push_h` < w/(2 mu_s) the die slides without
    tipping, so a low push translates it with the face preserved;
  - ungraspable: the 140 mm width exceeds the ~80 mm parallel-jaw span.

Assets are fully procedural (compound spawner; explicit friction material;
mass/CoM/inertia AUTHORED — custom spawn funcs apply no cfg schemas):
  - die (dynamic): grey collider cube + six visual-only colored face plates
    (red +x / blue -x / yellow +y / green -y / white +z / orange -z; opposite faces
    are red-blue, yellow-green, white-orange);
  - ref (KINEMATIC): a 55 mm replica with the same face colors, re-oriented at reset
    so the TARGET face is up, standing on a static pedestal;
  - crimson mat + grey decoy mat: visual-only flush markers (no colliders — a proud
    collider edge would wall the sliding die);
  - ground plane with an explicit high-friction material.

Per-episode randomization (readback-verified in smoke): die initial orientation (one
of 24 rotations: random top face x free yaw), die start position jitter, and the
target face (uniform over the 5 faces that are NOT initially up — so the null policy
never starts solved).

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

# Face convention (body frame): index -> (outward normal axis, display color, name).
# Opposite pairs: (0,1) red-blue, (2,3) yellow-green, (4,5) white-orange.
FACE_AXES = ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
             (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))
FACE_COLORS = ((0.85, 0.10, 0.10), (0.12, 0.25, 0.85), (0.95, 0.85, 0.10),
               (0.10, 0.68, 0.20), (0.95, 0.95, 0.95), (0.95, 0.50, 0.10))
FACE_NAMES = ("red", "blue", "yellow", "green", "white", "orange")
# Quats (w,x,y,z) bringing face i UP (+z), zero yaw.
_S = math.sqrt(0.5)
FACE_UP_QUATS = ((_S, 0.0, -_S, 0.0), (_S, 0.0, _S, 0.0), (_S, _S, 0.0, 0.0),
                 (_S, -_S, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))


def opposite_face(i: int) -> int:
    return i ^ 1


# ----- custom compound spawner ------------------------------------------------------------------
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
    """One USD physics material (explicit binding — the default-material ~0.5 trap)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _visual_box(stage, path: str, size, center, color) -> None:
    """A visual-only box child prim (translate -> scale, authored once — idempotent
    per prim, the duplicate-xformOp trap). NO CollisionAPI."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _collider_box(stage, path: str, size, center, color, contact_offset: float,
                  material=None) -> None:
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


def _spawn_die(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A six-color die: ONE grey collider cube + six visual-only colored face plates
    (half proud of the surface so the colors read cleanly; they carry no collision so
    the contact geometry stays a perfect cube). Dynamic (mass/CoM/inertia AUTHORED)
    or kinematic (the reference replica) per cfg.kinematic."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    from pxr import UsdGeom

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    h = float(cfg.half)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(cfg.mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    ixx = float(cfg.mass) * (2.0 * h) ** 2 / 6.0
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(ixx, ixx, ixx))
    if cfg.kinematic:
        rb.CreateKinematicEnabledAttr(True)
    else:
        px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        px.CreateMaxDepenetrationVelocityAttr(0.5)
        px.CreateLinearDampingAttr(0.05)
        px.CreateAngularDampingAttr(0.05)
        px.CreateSolverPositionIterationCountAttr(16)
        px.CreateSolverVelocityIterationCountAttr(4)
        px.CreateSleepThresholdAttr(0.0)
        px.CreateStabilizationThresholdAttr(0.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    _collider_box(stage, f"{prim_path}/core", (2 * h, 2 * h, 2 * h), (0.0, 0.0, 0.0),
                  (0.30, 0.30, 0.32), cfg.contact_offset, material=mat)
    t = 0.002  # face plate thickness (1 mm proud, visual only)
    span = 2 * h - 0.010
    for i, (nx, ny, nz) in enumerate(FACE_AXES):
        size = (t if nx else span, t if ny else span, t if nz else span)
        center = (nx * h, ny * h, nz * h)
        _visual_box(stage, f"{prim_path}/face_{FACE_NAMES[i]}", size, center, FACE_COLORS[i])
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg class (lazy: module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "die" not in _SPAWNER_CACHE:

        @configclass
        class DieSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_die)
            half: float = 0.07
            mass: float = 1.0
            kinematic: bool = False
            mu_static: float = 0.9
            mu_dynamic: float = 0.8
            contact_offset: float = 0.002

        _SPAWNER_CACHE["die"] = DieSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DieRollMatchSceneCfg(BaseCfg):
    """Config for `DieRollMatchScene`. `__post_init__` asserts the strategic honesty
    invariants from pre-computed mechanics: the die is ungraspable; a high push tips
    it before it can slide while a low push slides it without tipping; the start
    region is disjoint from the mat tolerance zone (null progress ~0); the target
    face is never the initial top face (null orientation credit 0)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    flat_max_deg: float = tunable(10.0)  # top-face dominance cone for "flat"
    place_tol: float = tunable(0.08)  # Chebyshev |dx|,|dy| tolerance to the mat center (m)
    z_lo: float = tunable(0.055)  # resting-height band (die half = 0.07) ...
    z_hi: float = tunable(0.085)
    settle_lin: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.40)  # max |ang vel| when judging (rad/s)
    settle_steps_min: int = tunable(30)  # stillness must PERSIST this many steps

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    start_x: float = tunable(-0.30)  # die start center ...
    start_y: float = tunable(0.00)
    start_jit: float = tunable(0.07)  # uniform +/- xy jitter at reset
    yaw_deg: float = tunable(180.0)  # uniform +/- free yaw at reset

    # --- info: structure ---------------------------------------------------------------------
    die_half: float = info(0.07)  # 140 mm cube — wider than a ~80 mm parallel jaw
    die_mass: float = info(1.0)
    ref_half: float = info(0.0275)  # 55 mm kinematic reference replica
    pad_center: tuple = info((0.34, 0.0))  # crimson serving mat center
    pad_half: float = info(0.15)  # mat half-size (visual)
    decoy_center: tuple = info((0.02, 0.38))  # grey decoy mat (placement there earns nothing)
    ped_center: tuple = info((0.34, 0.36))  # reference-die pedestal center
    ped_half: float = info(0.045)
    ped_h: float = info(0.08)
    mu_static: float = info(0.9)
    mu_dynamic: float = info(0.8)
    contact_offset: float = info(0.002)
    roll_push_h: float = info(0.120)  # the solve pushes HERE to tip-roll (above CoM)
    slide_push_h: float = info(0.035)  # ... and HERE to slide without tipping
    jaw_span: float = info(0.080)  # parallel-jaw max opening (embodiment argument)

    # Derived (filled in __post_init__).
    d0_min: float = field(default=None, init=False)  # min possible start->mat distance

    def __post_init__(self) -> None:
        w = 2.0 * self.die_half
        # -- ungraspable: reorientation cannot be done by lift-and-turn --
        assert w >= self.jaw_span + 0.04, "die must be far wider than the jaw span"
        # -- tip-before-slide at the roll push height (F_tip = mg*w/(2 h) < mu_s*mg) --
        tip_ratio = w / (2.0 * self.roll_push_h)
        assert tip_ratio <= self.mu_static - 0.15, \
            f"high push must tip before it slides (ratio {tip_ratio:.3f} vs mu {self.mu_static})"
        assert self.roll_push_h <= w - 0.005, "roll push height must be on the die"
        # -- slide-without-tip at the slide push height (h < w/(2 mu_s)) --
        assert self.slide_push_h <= w / (2.0 * self.mu_static) - 0.02, \
            "low push must slide without tipping, with margin"
        # -- mat tolerance keeps the die essentially on the mat --
        assert self.place_tol + self.die_half <= self.pad_half + 0.01, \
            "place tolerance must keep the (axis-aligned) die footprint on the mat"
        assert self.place_tol >= 0.05, "place tolerance must stay coarse (arm-noise safe)"
        # -- start region disjoint from the mat tolerance zone; null progress ~0 --
        dx = abs(self.pad_center[0] - self.start_x) - self.start_jit
        self.d0_min = math.hypot(dx, 0.0)
        assert dx - self.die_half * math.sqrt(2.0) >= self.place_tol + 0.10, \
            "die start region must be well clear of the mat tolerance zone"
        assert self.d0_min >= 0.30, "start->mat distance must dominate the progress latch"
        # -- pedestal/decoy stay clear of the mat approach --
        assert self.ped_center[1] - self.ped_half >= self.pad_center[1] + self.pad_half + 0.05, \
            "pedestal must sit clear of the mat"
        assert abs(self.decoy_center[1]) - self.pad_half >= self.pad_half + 0.05, \
            "decoy mat must not overlap the real mat corridor"
        assert self.ref_half * 2.0 <= self.ped_half * 2.0 + 0.02, "reference die fits pedestal"
        # -- rubric bands --
        assert self.flat_max_deg <= 15.0, "flat cone must be far inside the 45 deg roll"
        assert self.z_lo < self.die_half < self.z_hi, "z band must contain the resting height"
        assert self.z_hi < self.ped_h + self.ref_half, "z band rejects perched states"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("die_roll_match")
class DieRollMatchScene(BaseScene):
    cfg: DieRollMatchSceneCfg

    def __init__(self, cfg: DieRollMatchSceneCfg | None = None) -> None:
        super().__init__(cfg or DieRollMatchSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_static, dynamic_friction=c.mu_dynamic,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            # crimson serving mat + grey decoy: visual-only FLUSH markers (no collider
            # — a proud collider edge would wall the sliding die).
            "pad_vis": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/PadVis",
                spawn=sim_utils.CuboidCfg(size=(2 * c.pad_half, 2 * c.pad_half, 0.002),
                                          visual_material=sim_utils.PreviewSurfaceCfg(
                                              diffuse_color=(0.55, 0.08, 0.10))),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(c.pad_center[0], c.pad_center[1], 0.001)),
            ),
            "decoy_vis": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/DecoyVis",
                spawn=sim_utils.CuboidCfg(size=(2 * c.pad_half, 2 * c.pad_half, 0.002),
                                          visual_material=sim_utils.PreviewSurfaceCfg(
                                              diffuse_color=(0.45, 0.45, 0.48))),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(c.decoy_center[0], c.decoy_center[1], 0.001)),
            ),
            # static pedestal carrying the kinematic reference die
            "pedestal": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Pedestal",
                spawn=sim_utils.CuboidCfg(
                    size=(2 * c.ped_half, 2 * c.ped_half, c.ped_h),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.20, 0.18, 0.22))),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(c.ped_center[0], c.ped_center[1], c.ped_h / 2)),
            ),
            "die": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Die",
                spawn=sp["die"](
                    mass_props=None, rigid_props=None,
                    half=c.die_half, mass=c.die_mass, kinematic=False,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.start_x, c.start_y, c.die_half + 0.004)),
            ),
            "ref": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ref",
                spawn=sp["die"](
                    mass_props=None, rigid_props=None,
                    half=c.ref_half, mass=0.1, kinematic=True,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.ped_center[0], c.ped_center[1], c.ped_h + c.ref_half + 0.001)),
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
        self.die: RigidObject = env.iscene["die"]
        self.ref: RigidObject = env.iscene["ref"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.face_up_quats = torch.tensor(FACE_UP_QUATS, device=dev)
        self.init_top = torch.zeros(n, dtype=torch.long, device=dev)
        self.target = torch.zeros(n, dtype=torch.long, device=dev)
        self.d0 = torch.full((n,), 1.0, device=dev)
        self.roll_latch = torch.zeros(n, device=dev)  # ever rested flat on a new face
        self.orient_latch = torch.zeros(n, device=dev)  # ever rested flat target-up
        self.prog_latch = torch.zeros(n, device=dev)  # running-max mat approach
        self.still_count = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample a top face (of 6) + a free yaw + xy jitter for the
        big die; sample the TARGET face uniformly among the 5 faces that are NOT the
        initial top; re-orient the kinematic reference die target-face-up on its
        pedestal; zero the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        _ = torch.rand(m, 2, device=dev)  # burn (the degenerate-first-draw trap)

        top = torch.randint(0, 6, (m,), device=dev)
        r = torch.randint(0, 5, (m,), device=dev)
        target = r + (r >= top).long()  # uniform over the 5 faces != top

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        half = yaw / 2
        qz = torch.stack([torch.cos(half), torch.zeros_like(half),
                          torch.zeros_like(half), torch.sin(half)], dim=-1)
        qf = self.face_up_quats[top]
        # q = qz (x) qf  with qz = (w1,0,0,z1)
        w1, z1 = qz[:, 0], qz[:, 3]
        w2, x2, y2, z2 = qf[:, 0], qf[:, 1], qf[:, 2], qf[:, 3]
        q = torch.stack([w1 * w2 - z1 * z2, w1 * x2 - z1 * y2,
                         w1 * y2 + z1 * x2, w1 * z2 + z1 * w2], dim=-1)

        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.start_x + (torch.rand(m, device=dev) * 2 - 1) * c.start_jit
        st[:, 1] = c.start_y + (torch.rand(m, device=dev) * 2 - 1) * c.start_jit
        st[:, 2] = c.die_half + 0.004
        pad = torch.tensor(c.pad_center, device=dev)
        self.d0[env_ids] = (st[:, 0:2] - pad).norm(dim=-1).clamp(min=0.15)
        st[:, 0:3] += origin
        st[:, 3:7] = q
        self.die.write_root_state_to_sim(st, env_ids)

        # kinematic reference die: target face up, zero yaw, on its pedestal
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.ped_center[0]
        st[:, 1] = c.ped_center[1]
        st[:, 2] = c.ped_h + c.ref_half + 0.001
        st[:, 0:3] += origin
        st[:, 3:7] = self.face_up_quats[target]
        self.ref.write_root_state_to_sim(st, env_ids)

        self.init_top[env_ids] = top
        self.target[env_ids] = target
        for t in (self.roll_latch, self.orient_latch, self.prog_latch, self.still_count):
            t[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "die": self.die.data.root_state_w[env_ids].clone(),
            "ref": self.ref.data.root_state_w[env_ids].clone(),
            "init_top": self.init_top[env_ids].clone(),
            "target": self.target[env_ids].clone(),
            "d0": self.d0[env_ids].clone(),
            "latches": torch.stack([
                self.roll_latch[env_ids], self.orient_latch[env_ids],
                self.prog_latch[env_ids], self.still_count[env_ids]], dim=-1).clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.die.write_root_state_to_sim(state["die"], env_ids)
        self.ref.write_root_state_to_sim(state["ref"], env_ids)
        self.init_top[env_ids] = state["init_top"]
        self.target[env_ids] = state["target"]
        self.d0[env_ids] = state["d0"]
        lat = state["latches"]
        (self.roll_latch[env_ids], self.orient_latch[env_ids],
         self.prog_latch[env_ids], self.still_count[env_ids]) = (
            lat[:, 0], lat[:, 1], lat[:, 2], lat[:, 3])

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A large cubic MENU DIE ({2 * c.die_half * 100:.0f} cm on a side, ~"
            f"{c.die_mass:.0f} kg) rests on the counter. Each face is a distinct solid "
            f"color: red, blue, yellow, green, white, orange — opposite faces pair "
            f"red-blue, yellow-green, white-orange. The die is far too wide for a "
            f"parallel-jaw gripper to grasp or lift; it can only be PUSHED. Pushing it "
            f"high (near its top edge) tips it over a bottom edge — one quarter-roll — "
            f"which changes the face that points up (rolling it toward some direction "
            f"brings the face that was facing the opposite way up). Pushing it low "
            f"slides it across the counter without tipping; sliding or spinning NEVER "
            f"changes the up face.\n"
            f"To one side lies a CRIMSON square serving mat ({2 * c.pad_half * 100:.0f} cm); "
            f"a grey mat elsewhere is a decoy. Behind the crimson mat a small REFERENCE "
            f"DIE with the same six colors stands on a dark pedestal, displaying one "
            f"face up — that displayed color is the episode's TARGET face (it is never "
            f"the face the big die starts with up, and it may be the face the big die "
            f"starts resting ON, which then takes two rolls to bring up).\n"
            f"Goal: leave the big die resting FLAT on the crimson mat with its TARGET "
            f"color face pointing UP. Judged only when everything is at rest: top face "
            f"within {c.flat_max_deg:.0f} deg of vertical and matching the reference, "
            f"die center within {c.place_tol * 100:.0f} cm of the mat center, die on "
            f"the counter surface. Any order of rolling and sliding is allowed."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Tip the big color die over its edges until its top face color matches the "
            "face shown up on the small reference die, then push it onto the crimson "
            "mat and leave it resting flat there. It is too wide to grasp — roll it "
            "with high pushes and slide it with low pushes; sliding never changes the "
            "up face."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    @staticmethod
    def _axes_z(q: torch.Tensor) -> torch.Tensor:
        """(N, 6) world-z components of the 6 face normals for quats q (N,4)."""
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        ex_z = 2.0 * (qx * qz - qw * qy)
        ey_z = 2.0 * (qy * qz + qw * qx)
        ez_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        return torch.stack([ex_z, -ex_z, ey_z, -ey_z, ez_z, -ez_z], dim=-1)

    def top_face(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,) long face index pointing most upward + (N,) its z-dominance."""
        dots = self._axes_z(self.die.data.root_quat_w)
        dom, idx = dots.max(dim=-1)
        return idx, dom

    def ref_top_face(self) -> torch.Tensor:
        """(N,) long: the reference die's up face (readback of the displayed target)."""
        return self._axes_z(self.ref.data.root_quat_w).max(dim=-1).indices

    def flat_now(self) -> torch.Tensor:
        _idx, dom = self.top_face()
        return dom >= math.cos(math.radians(self.cfg.flat_max_deg))

    def die_local(self) -> torch.Tensor:
        return self.die.data.root_pos_w - self.env_origins

    def pad_err(self) -> torch.Tensor:
        """(N,) Chebyshev distance of the die center to the mat center."""
        p = self.die_local()
        c = self.cfg
        return torch.maximum((p[:, 0] - c.pad_center[0]).abs(),
                             (p[:, 1] - c.pad_center[1]).abs())

    def in_pad(self) -> torch.Tensor:
        return self.pad_err() <= self.cfg.place_tol

    def z_ok(self) -> torch.Tensor:
        z = self.die_local()[:, 2]
        return (z >= self.cfg.z_lo) & (z <= self.cfg.z_hi)

    def _still_now(self) -> torch.Tensor:
        return ((self.die.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin)
                & (self.die.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang))

    def settled(self) -> torch.Tensor:
        """(N,) bool: stillness has PERSISTED `settle_steps_min` consecutive steps."""
        return self.still_count >= self.cfg.settle_steps_min

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Stillness counter + progress latches: ever rested flat on a NEW face (the
        first roll), ever rested flat TARGET-up, running-max approach to the mat."""
        self.still_count = (self.still_count + 1.0) * self._still_now().float()
        idx, _dom = self.top_face()
        flat = self.flat_now() & self.z_ok()
        self.roll_latch = torch.maximum(
            self.roll_latch, (flat & (idx != self.init_top)).float())
        self.orient_latch = torch.maximum(
            self.orient_latch, (flat & (idx == self.target)).float())
        p = self.die_local()
        pad = torch.tensor(self.cfg.pad_center, device=p.device)
        d = (p[:, 0:2] - pad).norm(dim=-1)
        self.prog_latch = torch.maximum(
            self.prog_latch, (1.0 - d / self.d0).clamp(0.0, 1.0))

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: die FLAT with the TARGET face up, centered on the crimson mat,
        at counter height, persistently still — the settled physical outcome."""
        idx, _dom = self.top_face()
        return (self.flat_now() & (idx == self.target) & self.in_pad()
                & self.z_ok() & self.settled())

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 first-roll latch + 0.25 orientation latch +
        0.20 x mat-approach running max — latched, credit never evaporates; capped
        0.60; 1.0 iff success(). Null policy ~0 (die spawns far out, non-target up)."""
        base = (0.15 * self.roll_latch + 0.25 * self.orient_latch
                + 0.20 * self.prog_latch).clamp(0.0, 0.60)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="die_roll_match", robot="null", env_spacing=3.0))
