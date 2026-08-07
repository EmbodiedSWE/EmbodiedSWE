"""ClipFenceScene — squeeze two spring clothespin-clips open and clamp each astride
the top edge of a thin fence, inside its matching colored zone (sim_gen task
`native_liberoplus_i66`).

Derived from libero/native_liberoplus (LIBERO-plus: LIBERO pick-and-place / drawer /
stove scenes stressed with perturbations — lighting, distractors, textures, language;
the physical plan is always "grasp a rigid object, transport it, release it at a
target region"). The MANIPULATION MODEL here is replaced wholesale: the manipulated
object is itself a spring-loaded MECHANISM, and the goal state is not a resting pose
but a maintained FORCE-CLOSURE. Each clip is two crossed levers on a preloaded
torsion hinge. Its jaws are shut (their tips touch); the fence blade is 8 mm thick;
so a clip can NEVER arrive at the goal by being carried and released — the solver
must SQUEEZE the two flared tail levers toward each other (the parallel-jaw gripper's
own closing stroke is exactly this actuation), which cams the jaws open, hold the
squeeze while lowering the open jaws astride the fence's top edge, and then release
so the spring clamps the jaws flush onto the blade faces. Success is the clip
HANGING off the fence top by its own spring grip, inside the colored zone that
matches the clip's color. Nothing in the seed (or its whole suite) ever actuates the
carried object: transport-and-release is precisely the strategy this task rejects.

Assets are fully procedural (the compound-spawner pattern):
  - fence: KINEMATIC foot plate + a thin vertical blade (8 mm thick, 440 mm long,
    130 mm tall). Neutral grey.
  - zone bands: two KINEMATIC saddle sleeves (a colored plate on each blade face,
    70 mm wide, on the LOWER half of the blade, below the clamp region) teleported
    to per-episode positions along the fence; the clamp target is the top edge
    DIRECTLY ABOVE a band.
  - clips (one RED, one BLUE): each is TWO dynamic lever bodies (boss + jaw + flared
    tail) joined by a RevoluteJoint at the boss with an angular drive = the torsion
    spring (preloaded toward closed; joint limit = jaw tips touching). Jaw inner
    faces are authored so that at the clamped opening they sit flush and vertical
    against the blade faces (face-face contact, no edge chatter).

Joint coordinate psi (deg): 0 = jaws flush/parallel 8.6 mm apart (the clamped-on-
blade state), -10 = spring-closed free state (tips touching), +30 = wide open.
The drive targets -35 so a clamped clip presses the blade with ~0.075 N*m of
preloaded torque (~2.6 N per face); a free clip rests pinched shut at the limit.

Per-episode randomization (readback-verifiable): fence xy jitter + yaw; the two
zone-band centers sampled along the fence AND the color assignment shuffled; each
clip's ground pose (xy jitter + free yaw, lying flat on its side).

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  per clip (x2):
    0.08 near      — clip pivot ever within `near_dist` of its zone's top-edge point
    0.12 actuated  — jaws ever opened to >= `psi_engaged` while near the zone top
                     (the spring holds a free clip at -10 deg; only real squeezing,
                     or the blade itself between the jaws, can hold psi up here)
    0.20 astride   — clip ever clamped astride the zone top edge (full gate below)
  1.0 iff success() — BOTH clips simultaneously astride their matching zones,
  everything settled. Non-success is capped at 0.80.
"astride" is live-geometric, judged in the fence frame: pivot on the blade plane at
top-edge height inside the zone band's y-window, hinge axis parallel to the fence,
clip upright, the two jaws on OPPOSITE sides of the blade plane, and the jaw opening
psi >= `psi_engaged` (a free clip snaps to -10 deg: an open joint up there means the
blade is physically between the jaws).

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


# ----- geometry constants (meters; the lever frame's origin is the hinge pivot) -----------------
FOOT = (0.120, 0.460, 0.012)      # fence foot plate (x, y, z)
BLADE_T = 0.008                   # blade thickness
BLADE_L = 0.440                   # blade length (y)
BLADE_H = 0.130                   # blade height above the foot
Z_TOP = FOOT[2] + BLADE_H         # blade top edge height above the ground (0.142)

BAND_W = 0.070                    # zone band width along the fence
BAND_H = 0.060                    # zone band plate height (lower blade, below jaw reach)
BAND_T = 0.002                    # band plate thickness
BAND_Z0 = FOOT[2] + 0.005         # band plate bottom height

BOSS = (0.012, 0.016, 0.008)      # per-lever pivot boss (its bottom face rests on the blade top)
BOSS_C = 0.005                    # boss center x offset (signed by side)
JAW = (0.008, 0.016, 0.046)       # jaw plate
JAW_CX = 0.0083                   # jaw center |x| in the flush frame (inner face at 0.0043)
JAW_CZ = -0.037                   # jaw center z (span -0.060 .. -0.014: the jaw TOP is the
#                                   tightest gap point and must clear the blade at psi ~ +10,
#                                   the joint's usable opening on this backend)
TAIL = (0.006, 0.016, 0.044)      # flared tail lever
TAIL_C = (0.0176, 0.022)          # tail center (|x|, z)
TAIL_TILT = 35.0                  # tail outward lean (deg about y, signed by side)
LEVER_MASS = 0.015                # per lever (clip = 30 g)

PSI_CLOSED = -10.0                # joint limit: jaw tips touching (free clip rests here)
PSI_OPEN_MAX = 30.0               # joint upper limit
SPRING_TARGET = -35.0             # drive target (deg): preload against the closed limit
SPRING_K = 0.003                  # drive stiffness (N*m / deg)
SPRING_C = 0.0002                 # drive damping (N*m*s / deg)

CLIP_COLORS = {"red": (0.85, 0.12, 0.10), "blue": (0.12, 0.30, 0.88)}


# ----- custom compound spawners ------------------------------------------------------------------
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


def _friction_material(stage, path: str, mu_s: float, mu_d: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu_s))
    api.CreateDynamicFrictionAttr(float(mu_d))
    api.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, *, center, size, color, contact_offset=0.001,
         orient=None, material=None, collide=True):
    """One box child: translate (+ optional orient) + scale, displayColor, collider."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(box.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
        if material is not None:
            UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics")
    return box.GetPrim()


def _spawn_fence(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC fence: foot plate + thin vertical blade. Local frame: origin at the
    foot center on the ground; blade mid-plane is local x=0, length along local y."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    mat = _friction_material(stage, f"{prim_path}/mat", cfg.mu_s, cfg.mu_d)
    _box(stage, f"{prim_path}/foot", center=(0.0, 0.0, FOOT[2] / 2), size=FOOT,
         color=(0.30, 0.30, 0.33), material=mat)
    _box(stage, f"{prim_path}/blade",
         center=(0.0, 0.0, FOOT[2] + BLADE_H / 2), size=(BLADE_T, BLADE_L, BLADE_H),
         color=(0.55, 0.55, 0.58), material=mat)
    return root


def _spawn_band(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC zone band: one colored plate flush on each blade face, on the LOWER
    half of the blade (below jaw reach — it never touches a clamping clip). Local
    frame matches the fence's (origin at fence-foot level on the blade plane)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    for sgn in (1.0, -1.0):
        _box(stage, f"{prim_path}/plate_{'p' if sgn > 0 else 'n'}",
             center=(sgn * (BLADE_T / 2 + 0.0006 + BAND_T / 2), 0.0,
                     BAND_Z0 + BAND_H / 2),
             size=(BAND_T, BAND_W, BAND_H), color=cfg.color)
    return root


def _spawn_lever(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One clip lever (DYNAMIC): pivot boss + jaw plate + flared tail, authored in
    the FLUSH frame (origin = hinge pivot; jaw inner face vertical at |x| = 4.3 mm,
    which is where it sits when clamped on the 8 mm blade). `cfg.side` = +1 / -1
    mirrors the lever. `cfg.peer` (on the second lever only) names the sibling
    lever prim; then this spawner also authors the RevoluteJoint + torsion drive."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    s = float(cfg.side)
    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(LEVER_MASS))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.05)

    mat = _friction_material(stage, f"{prim_path}/mat", cfg.mu_s, cfg.mu_d)
    _box(stage, f"{prim_path}/boss", center=(s * BOSS_C, 0.0, 0.0), size=BOSS,
         color=cfg.color, material=mat)
    _box(stage, f"{prim_path}/jaw", center=(s * JAW_CX, 0.0, JAW_CZ), size=JAW,
         color=cfg.color, material=mat)
    half = math.radians(s * TAIL_TILT) / 2
    _box(stage, f"{prim_path}/tail", center=(s * TAIL_C[0], 0.0, TAIL_C[1]),
         size=TAIL, color=cfg.color, material=mat,
         orient=(math.cos(half), 0.0, math.sin(half), 0.0))

    if cfg.peer:
        base = prim_path.rsplit("/", 1)[0]
        j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
        j.CreateBody0Rel().SetTargets([f"{base}/{cfg.peer}"])
        j.CreateBody1Rel().SetTargets([prim_path])
        j.CreateAxisAttr("Y")
        j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        j.CreateLowerLimitAttr(float(PSI_CLOSED))
        j.CreateUpperLimitAttr(float(PSI_OPEN_MAX))
        drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        drv.CreateTypeAttr("force")
        drv.CreateStiffnessAttr(float(SPRING_K))
        drv.CreateDampingAttr(float(SPRING_C))
        drv.CreateTargetPositionAttr(float(SPRING_TARGET))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "fence" not in _SPAWNER_CACHE:

        @configclass
        class FenceSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_fence)
            mu_s: float = 0.70
            mu_d: float = 0.60

        @configclass
        class BandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_band)
            color: tuple = (1.0, 0.0, 0.0)

        @configclass
        class LeverSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lever)
            side: float = 1.0
            peer: str = ""
            color: tuple = (1.0, 0.0, 0.0)
            mu_s: float = 0.70
            mu_d: float = 0.60

        _SPAWNER_CACHE["fence"] = FenceSpawnerCfg
        _SPAWNER_CACHE["band"] = BandSpawnerCfg
        _SPAWNER_CACHE["lever"] = LeverSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ClipFenceSceneCfg(BaseCfg):
    """Config for `ClipFenceScene`. The astride gates are honest by construction:
    a free clip's spring pins psi at -10 deg (tips touching, no 8 mm blade can pass),
    so psi >= psi_engaged at the fence top means the blade is physically held
    between the jaws; the opposite-side clause rejects a clip leaning on one face."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    psi_engaged: float = tunable(-3.0)   # jaw opening (deg, 0 = flush-on-blade) counted as engaged
    x_tol: float = tunable(0.012)        # pivot distance from the blade plane (m)
    z_lo: float = tunable(-0.010)        # pivot height window about Z_TOP (m)
    z_hi: float = tunable(0.020)
    y_tol: float = tunable(0.025)        # pivot |y - zone center| along the fence (m)
    align_max_deg: float = tunable(30.0) # hinge-axis-vs-fence and clip-up-vs-world-up cones
    jaw_min_x: float = tunable(0.001)    # min jaw-point offset from the blade plane (straddle)
    settle_speed: float = tunable(0.05)  # max lever |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.6)   # max lever |ang vel| when judging (rad/s)
    near_dist: float = tunable(0.075)    # "near the zone top" latch radius (m)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    fence_yaw_deg: float = tunable(25.0)  # fence yaw (+/- deg)
    fence_jitter: float = tunable(0.05)   # fence xy jitter (+/- m)
    zone_lo: float = tunable(0.04)        # zone center |y| range along the fence (m):
    zone_hi: float = tunable(0.16)        #   one zone in [-hi,-lo], one in [lo,hi], colors shuffled
    clip_jitter: float = tunable(0.03)    # clip ground xy jitter (+/- m)
    clip_yaw_deg: float = tunable(180.0)  # clip ground yaw (+/- deg, FREE)

    # --- info: layout (fence-local nominal spots; clips lie on the near side) --------------------
    fence_pos: tuple = info((0.45, 0.0))
    clip_spots: tuple = info(((-0.22, -0.13), (-0.22, 0.13)))  # (red, blue) fence-local xy
    # rubric weights (2 x (0.08 + 0.12 + 0.20) = 0.80 = the non-success cap)
    w_near: float = info(0.08)
    w_act: float = info(0.12)
    w_astride: float = info(0.20)


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


def _qconj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qx(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 1] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def clip_root_states(pivot_w: torch.Tensor, q_asm: torch.Tensor,
                     psi_deg: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Root states (m, 13) for the two lever bodies of a clip whose assembly frame
    (origin at the hinge pivot) is at `pivot_w` / `q_asm`, with jaw opening
    `psi_deg` (0 = flush frame; -10 = closed). Lever A tilts by -psi/2 about the
    hinge (local y), lever B by +psi/2; both origins sit at the pivot."""
    m = pivot_w.shape[0]
    dev = pivot_w.device
    half = torch.full((m,), math.radians(psi_deg) / 2, device=dev)  # psi/2 (rad)
    st_a = torch.zeros(m, 13, device=dev)
    st_b = torch.zeros(m, 13, device=dev)
    st_a[:, 0:3] = pivot_w
    st_b[:, 0:3] = pivot_w
    st_a[:, 3:7] = _qmul(q_asm, _qy(-half))  # lever A: rotation by -psi/2 about y
    st_b[:, 3:7] = _qmul(q_asm, _qy(half))   # lever B: rotation by +psi/2 about y
    return st_a, st_b


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("clip_fence")
class ClipFenceScene(BaseScene):
    cfg: ClipFenceSceneCfg

    CLIPS = ("red", "blue")

    def __init__(self, cfg: ClipFenceSceneCfg | None = None) -> None:
        super().__init__(cfg or ClipFenceSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        cls = _spawner_classes()
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
            "fence": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Fence",
                spawn=cls["fence"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(self.cfg.fence_pos[0], self.cfg.fence_pos[1], 0.0)),
            ),
        }
        for name in self.CLIPS:
            out[f"band_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Band_" + name,
                spawn=cls["band"](
                    color=CLIP_COLORS[name],
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(self.cfg.fence_pos[0],
                                                               self.cfg.fence_pos[1], 0.0)),
            )
        for i, name in enumerate(self.CLIPS):
            for side, tag in ((1.0, "a"), (-1.0, "b")):
                out[f"{name}_{tag}"] = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/Clip_" + f"{name}_{tag}",
                    spawn=cls["lever"](
                        side=side, color=CLIP_COLORS[name],
                        peer=f"Clip_{name}_a" if tag == "b" else "",
                        mass_props=sim_utils.MassPropertiesCfg(mass=LEVER_MASS)),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(0.9 + 0.3 * i, -0.9, 0.05)),
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
        self.fence: RigidObject = env.iscene["fence"]
        self.bands: dict[str, RigidObject] = {
            n: env.iscene[f"band_{n}"] for n in self.CLIPS}
        self.levers: dict[str, tuple[RigidObject, RigidObject]] = {
            n: (env.iscene[f"{n}_a"], env.iscene[f"{n}_b"]) for n in self.CLIPS}
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self._zone_y = torch.zeros(n, 2, device=dev)  # fence-local y of [red, blue] zones
        # latches: [near, actuated, astride] per clip
        self._near = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self._act = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self._astr = torch.zeros(n, 2, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the fence (yaw + xy jitter); sample the two zone
        centers along the fence and SHUFFLE the color assignment; teleport the bands
        onto the blade; lay each clip flat on its side on the ground on the near
        side (jitter + free yaw, jaws spring-shut); clear the latches."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- fence ---
        psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.fence_yaw_deg)
        qf = _qz(psi)
        fp = torch.zeros(m, 3, device=dev)
        fp[:, 0] = c.fence_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.fence_jitter
        fp[:, 1] = c.fence_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.fence_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = fp + origin
        st[:, 3:7] = qf
        self.fence.write_root_state_to_sim(st, env_ids)

        # --- zones: one center on each half of the fence, colors shuffled ---
        y_neg = -(c.zone_lo + torch.rand(m, device=dev) * (c.zone_hi - c.zone_lo))
        y_pos = c.zone_lo + torch.rand(m, device=dev) * (c.zone_hi - c.zone_lo)
        swap = torch.rand(m, device=dev) < 0.5
        zy = torch.stack([torch.where(swap, y_pos, y_neg),
                          torch.where(swap, y_neg, y_pos)], dim=1)
        self._zone_y[env_ids] = zy
        for k, name in enumerate(self.CLIPS):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 1] = zy[:, k]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = fp + quat_apply(qf, loc) + origin
            st[:, 3:7] = qf
            self.bands[name].write_root_state_to_sim(st, env_ids)

        # --- clips: flat on their sides on the ground, jitter + free yaw, shut ---
        for k, name in enumerate(self.CLIPS):
            spot = torch.tensor(c.clip_spots[k], device=dev).expand(m, 2)
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0:2] = spot + (torch.rand(m, 2, device=dev) * 2 - 1) * c.clip_jitter
            pivot = fp + quat_apply(qf, loc) + origin
            pivot[:, 2] = 0.008 + 0.002  # side faces at local |y| = 8 mm
            yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.clip_yaw_deg)
            q_asm = _qmul(_qz(yaw), _qx(torch.full((m,), math.pi / 2, device=dev)))
            st_a, st_b = clip_root_states(pivot, q_asm, PSI_CLOSED)
            self.levers[name][0].write_root_state_to_sim(st_a, env_ids)
            self.levers[name][1].write_root_state_to_sim(st_b, env_ids)

        self._near[env_ids] = False
        self._act[env_ids] = False
        self._astr[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "fence": self.fence.data.root_state_w[env_ids].clone(),
            "bands": {n: b.data.root_state_w[env_ids].clone()
                      for n, b in self.bands.items()},
            "levers": {n: (a.data.root_state_w[env_ids].clone(),
                           b.data.root_state_w[env_ids].clone())
                       for n, (a, b) in self.levers.items()},
            "zone_y": self._zone_y[env_ids].clone(),
            "latch": (self._near[env_ids].clone(), self._act[env_ids].clone(),
                      self._astr[env_ids].clone()),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.fence.write_root_state_to_sim(state["fence"], env_ids)
        for n, b in self.bands.items():
            b.write_root_state_to_sim(state["bands"][n], env_ids)
        for n, (a, b) in self.levers.items():
            a.write_root_state_to_sim(state["levers"][n][0], env_ids)
            b.write_root_state_to_sim(state["levers"][n][1], env_ids)
        self._zone_y[env_ids] = state["zone_y"]
        self._near[env_ids], self._act[env_ids], self._astr[env_ids] = (
            state["latch"][0], state["latch"][1], state["latch"][2])

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A grey FENCE stands on the ground: a thin vertical blade "
            f"{BLADE_T * 1000:.0f} mm thick, {BLADE_L * 1000:.0f} mm long and "
            f"{BLADE_H * 1000:.0f} mm tall on a low foot plate (top edge "
            f"{Z_TOP * 1000:.0f} mm above the ground). Two colored ZONE BANDS — one "
            f"RED, one BLUE, each {BAND_W * 1000:.0f} mm wide — are wrapped around "
            f"the lower half of the blade; their positions along the fence and which "
            f"color is on which side change every episode. On the near side of the "
            f"fence lie two spring CLOTHESPIN CLIPS flat on the ground, one RED and "
            f"one BLUE (positions and headings vary): each clip is two crossed "
            f"levers on a spring-loaded hinge — a pair of JAWS on one end (spring "
            f"holds them pinched SHUT) and two flared TAIL levers on the other. "
            f"Squeezing the two tails toward each other (about "
            f"{2 * (TAIL_C[0] + 0.003 + 0.044 * 0.29):.3f} m apart at their tips, "
            f"a parallel-jaw gripper's natural closing stroke) cams the jaws open "
            f"to ~19 mm at their tips; releasing lets the spring snap them shut.\n"
            f"Goal: clamp EACH clip astride the TOP EDGE of the fence, directly "
            f"above the zone band of ITS OWN color — the red clip over the red "
            f"band, the blue clip over the blue band — so that each clip ends up "
            f"hanging on the blade by its own spring grip, jaws straddling the "
            f"blade with one jaw on each face, clip upright, hinge running along "
            f"the fence. The jaws are shut tighter ({-PSI_CLOSED:.0f} deg closed) "
            f"than the {BLADE_T * 1000:.0f} mm blade, so a clip simply set down or "
            f"pushed onto the edge does NOT go on: you must hold its tails squeezed "
            f"to keep the jaws open while lowering it over the edge, then release. "
            f"A clip resting against the fence, lying on top of the edge, clamped "
            f"outside its color's band, or on the wrong color does not count. "
            f"No particular order is required; both clips must be clamped on, at "
            f"rest, at the same time."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Squeeze each clothespin clip open by its tail levers and clamp it "
            "astride the fence's top edge directly above the zone band of its own "
            "color — red clip on the red band, blue clip on the blue band — so "
            "both hang by their spring grip."
        )

    # ----- frames / readbacks --------------------------------------------------------------------
    def _fence_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points (N, 3) -> fence frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.fence.data.root_quat_w,
                                  pos_w - self.fence.data.root_pos_w)

    def psi_deg(self) -> torch.Tensor:
        """(N, 2) jaw opening per clip (deg): relative lever rotation about the
        hinge axis; 0 = flush frame, -10 = spring-shut, positive = opened."""
        out = []
        for name in self.CLIPS:
            a, b = self.levers[name]
            qr = _qmul(_qconj(a.data.root_quat_w), b.data.root_quat_w)
            ang = 2.0 * torch.atan2(qr[:, 2], qr[:, 0])
            out.append(torch.rad2deg(ang))
        return torch.stack(out, dim=1)

    def _lever_tensors(self) -> tuple[torch.Tensor, ...]:
        """(pos (N,4,3), quat (N,4,4), |v| (N,4), |w| (N,4)) — levers in clip order
        red_a, red_b, blue_a, blue_b."""
        bodies = [lv for n in self.CLIPS for lv in self.levers[n]]
        pos = torch.stack([b.data.root_pos_w for b in bodies], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in bodies], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in bodies], dim=1)
        omg = torch.stack([b.data.root_ang_vel_w.norm(dim=-1) for b in bodies], dim=1)
        return pos, quat, vel, omg

    def zone_top_w(self) -> torch.Tensor:
        """(N, 2, 3) world position of each zone's top-edge target point."""
        from isaaclab.utils.math import quat_apply

        out = []
        for k in range(2):
            loc = torch.zeros(self.env.num_envs, 3, device=self.env.device)
            loc[:, 1] = self._zone_y[:, k]
            loc[:, 2] = Z_TOP
            out.append(self.fence.data.root_pos_w
                       + quat_apply(self.fence.data.root_quat_w, loc))
        return torch.stack(out, dim=1)

    # ----- live predicates ------------------------------------------------------------------------
    def near(self) -> torch.Tensor:
        """(N, 2) bool: clip pivot within `near_dist` of its zone-top point."""
        pos, _q, _v, _w = self._lever_tensors()
        pivot = 0.5 * (pos[:, 0::2] + pos[:, 1::2])  # (N, 2, 3)
        return (pivot - self.zone_top_w()).norm(dim=-1) < self.cfg.near_dist

    def astride(self) -> torch.Tensor:
        """(N, 2) bool, live-geometric in the fence frame: pivot on the blade plane
        at top-edge height inside the zone y-window; hinge axis parallel to the
        fence; clip upright; the two jaws on OPPOSITE sides of the blade plane;
        jaws engaged (psi >= psi_engaged — a free clip's spring pins psi at -10)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        pos, quat, _v, _w = self._lever_tensors()
        n = pos.shape[0]
        dev = pos.device
        cos_a = math.cos(math.radians(c.align_max_deg))

        # fence axes in world
        ey_f = quat_apply(self.fence.data.root_quat_w,
                          torch.tensor([0.0, 1.0, 0.0], device=dev).expand(n, 3))
        ey = torch.tensor([0.0, 1.0, 0.0], device=dev).expand(n * 4, 3)
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n * 4, 3)
        ly = quat_apply(quat.reshape(-1, 4), ey).reshape(n, 4, 3)  # lever hinge axes
        lz = quat_apply(quat.reshape(-1, 4), ez).reshape(n, 4, 3)  # lever up axes
        # jaw reference points (body-local (side*6, 0, -30) mm) -> fence-local x
        jaw_pt = torch.tensor([[0.006, 0.0, -0.030], [-0.006, 0.0, -0.030]],
                              device=dev).repeat(2, 1).expand(n, 4, 3)
        jw = quat_apply(quat.reshape(-1, 4), jaw_pt.reshape(-1, 3)).reshape(n, 4, 3) + pos
        jx = self._fence_local(jw.reshape(-1, 3)).reshape(n, 4, 3)[:, :, 0]

        psi = self.psi_deg()
        out = []
        for k in range(2):
            a, b = 2 * k, 2 * k + 1
            pivot = 0.5 * (pos[:, a] + pos[:, b])
            ploc = self._fence_local(pivot)
            in_x = ploc[:, 0].abs() <= c.x_tol
            in_z = ((ploc[:, 2] - Z_TOP) >= c.z_lo) & ((ploc[:, 2] - Z_TOP) <= c.z_hi)
            in_y = (ploc[:, 1] - self._zone_y[:, k]).abs() <= c.y_tol
            hinge_ok = (ly[:, a] * ey_f).sum(-1).abs() >= cos_a
            up_ok = (lz[:, a, 2] >= cos_a) & (lz[:, b, 2] >= cos_a)
            straddle = (jx[:, a] * jx[:, b] < 0) & \
                (jx[:, a].abs() >= c.jaw_min_x) & (jx[:, b].abs() >= c.jaw_min_x)
            engaged = psi[:, k] >= c.psi_engaged
            out.append(in_x & in_z & in_y & hinge_ok & up_ok & straddle & engaged)
        return torch.stack(out, dim=1)

    def settled(self) -> torch.Tensor:
        """(N,) bool: every lever below the settle gates."""
        _p, _q, vel, omg = self._lever_tensors()
        return (vel < self.cfg.settle_speed).all(dim=1) & \
            (omg < self.cfg.settle_omega).all(dim=1)

    def _update_latches(self) -> None:
        near = self.near()
        self._near |= near
        self._act |= near & (self.psi_deg() >= self.cfg.psi_engaged)
        self._astr |= self.astride()

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: BOTH clips astride their matching zones simultaneously, all
        levers settled, states finite. All clauses are live physical outcomes."""
        self._update_latches()
        pos, _q, _v, _w = self._lever_tensors()
        finite = torch.isfinite(pos).all(dim=-1).all(dim=-1)
        return self.astride().all(dim=1) & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: latched stage credit per clip (near 0.08, actuated
        0.12, astride 0.20 — anchored in the demonstrated solve; ~0 for the null
        policy), capped at 0.80 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_near * self._near.float() + c.w_act * self._act.float()
                + c.w_astride * self._astr.float()).sum(dim=1).clamp(max=0.80)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="clip_fence", robot="null"))
