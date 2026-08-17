"""DieTumbleScene — tumble an over-sized die edge-over-edge into a walled goal tray,
red face up.

Derived from maniskill/stack_pyramid ("pick up the red cube, place it next to the green
cube, stack the blue cube on top of both"), but the MANIPULATION MODE and the GOAL TYPE
are both replaced. The seed is repeated free-space pick-and-place of three graspable
40 mm cubes, judged purely on relative stacking POSITIONS — orientation never matters,
each placement is an independent grasp-transport-release, and the plan is the same three
moves regardless of the episode. Here there is ONE 90 mm die — WIDER THAN THE FRANKA'S
80 mm JAW SPAN, so it cannot be grasped at all — and the goal is a POSITION *AND*
ORIENTATION: the die must come to rest inside a shallow walled tray with its RED face
up. The only way to reorient it is to TUMBLE it: press near a top edge until it pivots
over a bottom edge and falls onto the next face. Every tumble simultaneously translates
the die one die-width and quarter-turns it (the classic rolling-cube coupling), and the
tray's raised rim — an 18 mm step seen from the ground — makes quasi-static slide-in
impossible (climbing the step by pushing needs > 5 N vs < 1.7 N to slide, and any rim
crossing IS a tumble), so the final entry necessarily applies one more quarter-turn. A solver must therefore PLAN a
tumble sequence in the rotation group of the cube — conditioned on the randomized start
orientation (red never starts up) — such that after the forced entry tumble the red
face lands on top. Nothing is stacked, nothing is grasped, and no two episodes share a
plan.

Bodies (all procedural, no external assets):
  - die: 90 mm dynamic cube, 0.30 kg, six distinctly colored faces (visual-only proud
    plates; collision is the bare cube): +z RED (the target face), -z WHITE, +x BLUE,
    -x GREEN, +y ORANGE, -y PURPLE.
  - tray: ONE kinematic compound — a bright YELLOW floor slab (12 mm) ringed by four
    amber curb walls (6 mm above the slab — an 18 mm step seen from the ground,
    8 mm thick), inner span 170 mm. The die fits inside with ~40 mm of slack per side.

Per-episode randomization (readback-verified in smoke): tray world xy jitter, die start
bearing/distance on the -x side of the tray, and the die's initial orientation sampled
from the 20 axis-aligned orientations whose red face is NOT up.

Rubric (0..1; latched, debounced partial credit anchored in the demonstrated solve):
  0.15 * tumbled  — the die has completed at least one quarter-turn (its settled
                    body-up axis differs from the reset readback)
  0.25 * oriented — red face up, settled, anywhere
  0.25 * arrived  — die settled fully inside the tray (any face up)
  1.0 iff success(): die at rest inside the tray with the red face up (live state,
                     finite). Non-success capped at 0.65.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- the 24 axis-aligned orientations (pure math, app-free) -----------------------------------
def _rot24() -> list[list[list[int]]]:
    """All 24 proper rotations mapping the cube onto itself (integer matrices)."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            mat = [[0] * 3 for _ in range(3)]
            for r, (p, s) in enumerate(zip(perm, signs)):
                mat[r][p] = s
            det = (mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1])
                   - mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0])
                   + mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]))
            if det == 1:
                out.append(mat)
    return out


def _mat_to_quat(mat: list[list[int]]) -> tuple[float, float, float, float]:
    """Rotation matrix -> quaternion (w, x, y, z), Shepperd's branches."""
    t = mat[0][0] + mat[1][1] + mat[2][2]
    if t > 0:
        w = math.sqrt(1.0 + t) / 2.0
        s = 1.0 / (4.0 * w)
        return (w, (mat[2][1] - mat[1][2]) * s, (mat[0][2] - mat[2][0]) * s,
                (mat[1][0] - mat[0][1]) * s)
    i = max(range(3), key=lambda k: mat[k][k])
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + mat[i][i] - mat[j][j] - mat[k][k]) * 2.0
    q = [0.0, 0.0, 0.0, 0.0]  # w, x, y, z
    q[0] = (mat[k][j] - mat[j][k]) / s
    q[1 + i] = s / 4.0
    q[1 + j] = (mat[j][i] + mat[i][j]) / s
    q[1 + k] = (mat[k][i] + mat[i][k]) / s
    return tuple(q)


def _up_idx_of(mat: list[list[int]]) -> int:
    """Which BODY axis points world-up for orientation `mat` (0..5 = +x,-x,+y,-y,+z,-z).

    world_up in body frame = mat^T @ ez = row 2 of mat.
    """
    row = mat[2]
    ax = max(range(3), key=lambda a: abs(row[a]))
    return ax * 2 + (0 if row[ax] > 0 else 1)


# start orientations: red (body +z) NOT up  ->  20 of the 24
_START_SET = [(mat, _mat_to_quat(mat), _up_idx_of(mat)) for mat in _rot24() if mat[2][2] != 1]
assert len(_START_SET) == 20


# ----- USD authoring helpers (custom compound spawners) -----------------------------------------
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable | None):
    """One box child: translate + scale, displayColor, optional collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(box.GetPrim())
    return box.GetPrim()


def _bind_phys_material(stage, mat_path: str, prims, *, static: float, dynamic: float,
                        restitution: float) -> None:
    """Author a UsdShade physics material and bind it to `prims`. Custom-spawner
    colliders otherwise fall back to a ~0.5-friction default; the tumble pivots and
    the curb-blocks-sliding certificate both depend on a KNOWN friction level."""
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, mat_path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(static))
    api.CreateDynamicFrictionAttr(float(dynamic))
    api.CreateRestitutionAttr(float(restitution))
    pxm = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    pxm.CreateFrictionCombineModeAttr("average")
    pxm.CreateRestitutionCombineModeAttr("min")
    for prim in prims:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics")


# face plates: (name, axis, sign, color) — the die's visual identity
_FACES = (
    ("red", 2, +1, (0.90, 0.06, 0.06)),
    ("white", 2, -1, (0.92, 0.92, 0.92)),
    ("blue", 0, +1, (0.10, 0.25, 0.92)),
    ("green", 0, -1, (0.08, 0.72, 0.12)),
    ("orange", 1, +1, (0.95, 0.52, 0.05)),
    ("purple", 1, -1, (0.55, 0.10, 0.75)),
)


def _spawn_die(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The die: DYNAMIC compound — a bare cube collider (clean edge pivots) plus six
    visual-only proud face plates. Root frame at the CUBE CENTER. Everything physical
    (mass, damping, solver iters, material) is authored here — the clone-wrapped
    custom func gets no schema help."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.02)
    px.CreateAngularDampingAttr(0.05)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))

    collide = _make_collide(cfg.contact_offset)
    s = cfg.die_s
    core = _add_box(stage, f"{prim_path}/core",
                    center=(0.0, 0.0, 0.0), size=(s, s, s),
                    color=(0.35, 0.35, 0.38), collide=collide)
    # visual-only plates, 0.5 mm proud of the collision cube
    plate_w = s - 0.006
    for name, axis, sign, color in _FACES:
        center = [0.0, 0.0, 0.0]
        center[axis] = sign * (s / 2 - 0.0001)
        size = [plate_w, plate_w, plate_w]
        size[axis] = 0.0012
        _add_box(stage, f"{prim_path}/face_{name}",
                 center=center, size=size, color=color, collide=None)
    _bind_phys_material(stage, f"{prim_path}/mat", (core,),
                        static=cfg.mu, dynamic=cfg.mu * 0.92, restitution=0.0)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The goal tray: ONE KINEMATIC compound. Local frame: origin at the plate bottom
    center. Yellow floor plate ringed by four amber curb walls."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateSolverPositionIterationCountAttr(4)
    px.CreateSolverVelocityIterationCountAttr(1)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(10.0)

    collide = _make_collide(cfg.contact_offset)
    c = cfg
    inner_h = c.inner / 2
    outer = c.inner + 2 * c.wall_t
    wall_top = c.plate_t + c.wall_h
    prims = [
        _add_box(stage, f"{prim_path}/plate",
                 center=(0.0, 0.0, c.plate_t / 2),
                 size=(outer, outer, c.plate_t),
                 color=(0.95, 0.80, 0.10), collide=collide),
        _add_box(stage, f"{prim_path}/wall_xp",
                 center=(inner_h + c.wall_t / 2, 0.0, c.plate_t + c.wall_h / 2),
                 size=(c.wall_t, outer, c.wall_h),
                 color=(0.80, 0.62, 0.05), collide=collide),
        _add_box(stage, f"{prim_path}/wall_xn",
                 center=(-inner_h - c.wall_t / 2, 0.0, c.plate_t + c.wall_h / 2),
                 size=(c.wall_t, outer, c.wall_h),
                 color=(0.80, 0.62, 0.05), collide=collide),
        _add_box(stage, f"{prim_path}/wall_yp",
                 center=(0.0, inner_h + c.wall_t / 2, c.plate_t + c.wall_h / 2),
                 size=(c.inner, c.wall_t, c.wall_h),
                 color=(0.80, 0.62, 0.05), collide=collide),
        _add_box(stage, f"{prim_path}/wall_yn",
                 center=(0.0, -inner_h - c.wall_t / 2, c.plate_t + c.wall_h / 2),
                 size=(c.inner, c.wall_t, c.wall_h),
                 color=(0.80, 0.62, 0.05), collide=collide),
    ]
    del wall_top
    _bind_phys_material(stage, f"{prim_path}/mat", prims,
                        static=cfg.mu, dynamic=cfg.mu * 0.92, restitution=0.0)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "die" not in _SPAWNER_CACHE:

        @configclass
        class DieSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_die)
            die_s: float = 0.090
            mass: float = 0.30
            mu: float = 0.60
            contact_offset: float = 0.002

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            inner: float = 0.170
            wall_t: float = 0.008
            wall_h: float = 0.006
            plate_t: float = 0.012
            mu: float = 0.60
            contact_offset: float = 0.002

        _SPAWNER_CACHE["die"] = DieSpawnerCfg
        _SPAWNER_CACHE["tray"] = TraySpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class DieTumbleSceneCfg(BaseCfg):
    """Config for `DieTumbleScene`. Honesty bounds: the die (90 mm) exceeds the
    Franka's 80 mm jaw span, so it can only be pushed/tumbled; the tray rim (6 mm
    curb on a 12 mm slab = an 18 mm step from the ground) cannot be crossed by a
    quasi-static slide (climbing the outside step by pushing needs > 5 N, leaving
    over the inside curb > 3.5 N, while sliding needs < 1.7 N — smoke's curb
    certificate), and ANY crossing pivots the die a quarter-turn, so entry always
    applies one more rotation. `in_tol` (55 mm) accepts every position physically
    inside the walls (max in-tray center offset = inner/2 - die/2 = 40 mm) and
    rejects the closest outside rest (center offset >= 138 mm); `z_tol` (10 mm about
    slab_top + die/2 = 57 mm) also excludes any ground-level rest (45 mm). The 10 deg
    `up_max_deg` cone accepts the post-entry rest even when the die settles in the
    shallow <= 4 deg lean the 6 mm inner curb allows."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    in_tol: float = tunable(0.055)       # die center |xy - tray center| window (m)
    z_tol: float = tunable(0.010)        # die center height window about plate_t + die_s/2 (m)
    up_max_deg: float = tunable(10.0)    # red face normal within this of world-up
    settle_lin: float = tunable(0.06)    # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.30)    # max |ang vel| when judging (rad/s)
    latch_steps: int = tunable(5)        # consecutive qualifying steps before a latch fires

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    tray_jitter: float = tunable(0.03)   # tray world xy jitter (+/- m)
    die_r_min: float = tunable(0.28)     # die start distance from tray center (m)
    die_r_max: float = tunable(0.36)
    die_arc_deg: float = tunable(50.0)   # die start bearing: 180 +/- this (deg, -x side)

    # --- info: geometry (must match the spawner classes) ----------------------------------------
    die_s: float = info(0.090)
    die_mass: float = info(0.30)
    mu: float = info(0.60)
    ground_mu: float = info(0.50)
    inner: float = info(0.170)
    wall_t: float = info(0.008)
    wall_h: float = info(0.006)          # curb height above the plate top
    plate_t: float = info(0.012)
    contact_offset: float = info(0.002)
    tray_pos: tuple = info((0.0, 0.0))   # nominal tray center, world xy
    # rubric weights (0.15 + 0.25 + 0.25 = 0.65 = the non-success cap)
    w_tumbled: float = info(0.15)
    w_oriented: float = info(0.25)
    w_arrived: float = info(0.25)


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("die_tumble")
class DieTumbleScene(BaseScene):
    cfg: DieTumbleSceneCfg

    def __init__(self, cfg: DieTumbleSceneCfg | None = None) -> None:
        super().__init__(cfg or DieTumbleSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
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
                        static_friction=c.ground_mu, dynamic_friction=c.ground_mu * 0.92,
                        restitution=0.0, friction_combine_mode="average",
                        restitution_combine_mode="min")),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=sp["tray"](contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tray_pos[0], c.tray_pos[1], 0.0)),
            ),
            "die": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Die",
                spawn=sp["die"](contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tray_pos[0] - 0.32, c.tray_pos[1], c.die_s / 2 + 0.002)),
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.die: RigidObject = env.iscene["die"]
        self.tray: RigidObject = env.iscene["tray"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.tray_xy = torch.zeros(n, 2, device=dev)  # tray center, world (incl env origin)
        self.init_up = torch.zeros(n, dtype=torch.long, device=dev)  # body-up idx at reset
        # start-orientation table (quats wxyz + their body-up indices)
        self._start_q = torch.tensor([q for _m, q, _u in _START_SET],
                                     dtype=torch.float, device=dev)
        self._start_up = torch.tensor([u for _m, _q, u in _START_SET],
                                      dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live) + debounce
        self._tumbled = torch.zeros(n, dtype=torch.bool, device=dev)
        self._oriented = torch.zeros(n, dtype=torch.bool, device=dev)
        self._arrived = torch.zeros(n, dtype=torch.bool, device=dev)
        self._tumbled_ct = torch.zeros(n, dtype=torch.long, device=dev)
        self._oriented_ct = torch.zeros(n, dtype=torch.long, device=dev)
        self._arrived_ct = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the tray (world xy jitter), then the die on the -x
        side of it (bearing 180 +/- arc, distance in [r_min, r_max]) in one of the
        20 axis-aligned orientations whose RED face is NOT up. Clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- tray: kinematic compound, xy jitter ---
        txy = torch.zeros(m, 2, device=dev)
        txy[:, 0] = c.tray_pos[0]
        txy[:, 1] = c.tray_pos[1]
        txy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.tray_jitter
        txy += origin[:, 0:2]
        self.tray_xy[env_ids] = txy
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = txy
        st[:, 2] = origin[:, 2]
        st[:, 3] = 1.0
        self.tray.write_root_state_to_sim(st, env_ids)

        # --- die: bearing/distance on the -x side + one of the 20 red-not-up orientations ---
        # (torch.rand-based draws: the first torch.randint after a manual seed is
        # near-degenerate across seeds)
        bearing = math.pi + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.die_arc_deg)
        dist = c.die_r_min + torch.rand(m, device=dev) * (c.die_r_max - c.die_r_min)
        idx = (torch.rand(m, device=dev) * len(_START_SET)).long().clamp(max=len(_START_SET) - 1)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = txy[:, 0] + dist * torch.cos(bearing)
        st[:, 1] = txy[:, 1] + dist * torch.sin(bearing)
        st[:, 2] = origin[:, 2] + c.die_s / 2 + 0.002
        st[:, 3:7] = self._start_q[idx]
        self.die.write_root_state_to_sim(st, env_ids)
        self.init_up[env_ids] = self._start_up[idx]

        # --- clear latches ---
        for t in (self._tumbled, self._oriented, self._arrived):
            t[env_ids] = False
        for t in (self._tumbled_ct, self._oriented_ct, self._arrived_ct):
            t[env_ids] = 0

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "die": self.die.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "tray_xy": self.tray_xy[env_ids].clone(),
            "init_up": self.init_up[env_ids].clone(),
            "tumbled": self._tumbled[env_ids].clone(),
            "oriented": self._oriented[env_ids].clone(),
            "arrived": self._arrived[env_ids].clone(),
            "tumbled_ct": self._tumbled_ct[env_ids].clone(),
            "oriented_ct": self._oriented_ct[env_ids].clone(),
            "arrived_ct": self._arrived_ct[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.die.write_root_state_to_sim(state["die"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.tray_xy[env_ids] = state["tray_xy"]
        self.init_up[env_ids] = state["init_up"]
        self._tumbled[env_ids] = state["tumbled"]
        self._oriented[env_ids] = state["oriented"]
        self._arrived[env_ids] = state["arrived"]
        self._tumbled_ct[env_ids] = state["tumbled_ct"]
        self._oriented_ct[env_ids] = state["oriented_ct"]
        self._arrived_ct[env_ids] = state["arrived_ct"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A large DIE — a {c.die_s * 1000:.0f} mm cube with a different color on "
            f"every face (RED, WHITE, BLUE, GREEN, ORANGE, PURPLE) — rests on the "
            f"floor. Nearby stands the GOAL TRAY: a bright YELLOW square plate "
            f"({c.inner * 1000:.0f} mm across inside) ringed by a low amber curb "
            f"({(c.plate_t + c.wall_h) * 1000:.0f} mm tall). The die's position, its "
            f"distance and direction from the tray, and which face starts on top all "
            f"vary per episode — the RED face never starts on top.\n"
            f"Goal: get the die to rest fully INSIDE the tray with its RED face "
            f"pointing UP. The die is wider than a parallel gripper can open, so it "
            f"cannot be grasped or lifted — move and reorient it by TUMBLING: press "
            f"near a top edge so it pivots over a bottom edge onto the next face "
            f"(each tumble moves it one die-width and quarter-turns it), or push it "
            f"at mid-height to slide it without rotation. The curb is too tall to "
            f"slide across, so the die must TUMBLE over the curb to enter the tray — "
            f"and that entry tumble rotates it one more quarter-turn, which your "
            f"plan must account for. Plan the tumble sequence so that after the "
            f"final tumble into the tray the red face lands on top. No fixed move "
            f"order is imposed; only the settled end state is judged: die at rest "
            f"inside the tray, red face up. A die left outside the tray (even red "
            f"face up), or inside with any other face up, fails."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Tumble the large die across the floor and over the curb into the "
            "yellow tray, leaving it at rest fully inside with its RED face up. "
            "The die is too wide to grasp — tip it edge over edge; each tumble, "
            "including the one over the curb, quarter-turns it."
        )

    # ----- frames / readbacks -------------------------------------------------------------------
    def die_tray_local(self) -> torch.Tensor:
        """(N,3): die center relative to the tray center (world-aligned axes; the
        tray never yaws), z relative to the env origin."""
        p = self.die.data.root_pos_w
        out = torch.zeros(p.shape[0], 3, device=p.device)
        out[:, 0:2] = p[:, 0:2] - self.tray_xy
        out[:, 2] = p[:, 2] - self.env_origins[:, 2]
        return out

    def red_world(self) -> torch.Tensor:
        """(N,3): world direction of the die's RED face normal (body +z)."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        return quat_apply(self.die.data.root_quat_w, ez)

    def body_up_idx(self) -> torch.Tensor:
        """(N,) long: which BODY axis currently points world-up
        (0..5 = +x,-x,+y,-y,+z,-z)."""
        from isaaclab.utils.math import quat_apply_inverse

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        b = quat_apply_inverse(self.die.data.root_quat_w, ez)
        six = torch.stack([b[:, 0], -b[:, 0], b[:, 1], -b[:, 1], b[:, 2], -b[:, 2]], dim=-1)
        return six.argmax(dim=-1)

    # ----- live predicates ----------------------------------------------------------------------
    def in_tray(self) -> torch.Tensor:
        """(N,) bool: die center inside the tray windows and at plate-rest height —
        physical containment (the walls bound the position by construction)."""
        c = self.cfg
        loc = self.die_tray_local()
        xy_ok = (loc[:, 0].abs() < c.in_tol) & (loc[:, 1].abs() < c.in_tol)
        z_ok = (loc[:, 2] - (c.plate_t + c.die_s / 2)).abs() < c.z_tol
        return xy_ok & z_ok

    def red_up(self) -> torch.Tensor:
        """(N,) bool: red face normal within `up_max_deg` of world-up."""
        return self.red_world()[:, 2].clamp(-1.0, 1.0) \
            >= math.cos(math.radians(self.cfg.up_max_deg))

    def still(self) -> torch.Tensor:
        """(N,) bool: die below the settle speeds."""
        c = self.cfg
        return (self.die.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.die.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def _finite(self) -> torch.Tensor:
        return torch.isfinite(self.die.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.die.data.root_quat_w).all(dim=-1)

    # ----- latches ------------------------------------------------------------------------------
    def _update_latches(self) -> None:
        """Debounced (counter) latching: a stage must hold `latch_steps` consecutive
        post-steps while STILL before it counts — a die sweeping through the tray or
        rolling past red-up mid-tumble carries velocity and never latches."""
        c = self.cfg
        still = self.still()
        tumbled_now = (self.body_up_idx() != self.init_up) & still
        oriented_now = self.red_up() & still
        arrived_now = self.in_tray() & still
        for now, ct, latch in ((tumbled_now, self._tumbled_ct, self._tumbled),
                               (oriented_now, self._oriented_ct, self._oriented),
                               (arrived_now, self._arrived_ct, self._arrived)):
            ct[:] = torch.where(now, ct + 1, torch.zeros_like(ct))
            latch |= ct >= c.latch_steps

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric -------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the die rests fully inside the tray with the red face up —
        live physical state (settled pose + real containment), finite. A die that
        leaves the tray or tips loses success on the spot, so success must persist
        on its own."""
        self._update_latches()
        return self.in_tray() & self.red_up() & self.still() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15*tumbled + 0.25*oriented + 0.25*arrived
        (latched, ~0 for the null policy), capped at 0.65 — and exactly 1.0 iff
        success()."""
        c = self.cfg
        self._update_latches()
        base = (c.w_tumbled * self._tumbled.float()
                + c.w_oriented * self._oriented.float()
                + c.w_arrived * self._arrived.float()).clamp(max=0.65)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; the die is driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="die_tumble", robot="null"))
