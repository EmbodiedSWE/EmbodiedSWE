"""DieRollScene — tip an ungraspable die over its edges until the BLUE face points up,
then leave it resting centered on the green target mat (sim_gen task `pick_cube_i125`).

Derived from maniskill/pick_cube ("pick up the red cube with a Panda robot and lift it
by 0.1 m"), but the MANIPULATION MODEL is replaced wholesale. The seed's plan is ONE
grasp plus ONE guided free-space lift, judged by a position-shift checker on the held
object. Here the cube CANNOT be grasped and is NEVER lifted: it is a 90 mm die — wider
than a parallel jaw's ~80 mm span — and the goal is an ORIENTATION, not a height. The
solver must read the die's current pose, PLAN a sequence of quarter-rolls (discrete
edge-pivots, the classic rolling-cube puzzle), and execute each roll as a real tipping
contact: push high on a face to pivot the die over a ground edge, let gravity slam it
onto the next face. Sliding (pushing low) translates the die but can NEVER change which
face is up — reorientation is reachable only through rolls — and the blue face never
starts up, so at least one roll is always required. The episode ends with the die at
rest, blue up, centered on the GREEN target mat (a flat painted marker; a same-size
dark-gray DECOY mat on the other side pays nothing).

What the solver must bring, none of which exists in the seed:
  (1) orientation perception (which colored face is up / where blue currently points)
      and target identification (green mat vs the shuffled gray decoy);
  (2) a PLANNED roll sequence: each quarter-roll permutes the face cycle and displaces
      the die by one edge length, so the solver composes rotations, not waypoints;
  (3) two distinct contact skills on the SAME object: tipping (push high — the tip
      force ~2.5 N at 80 mm is BELOW the ~3.1 N sliding breakaway) and sliding
      (push low — tipping from CoM height needs ~4.4 N, above breakaway), selected by
      contact height;
  (4) restraint on the end state: at rest, centered, blue up — not merely displaced.

Assets are fully procedural (compound-spawner pattern):
  - die: DYNAMIC compound — one 90 mm collision cube (0.45 kg, mu 0.7/0.65 bound at
    the root) with six thin VISUAL-ONLY face plates in six distinct colors:
    +z BLUE, -z PURPLE, +x RED, -x ORANGE, +y YELLOW, -y WHITE (body frame).
  - target mat (GREEN) and decoy mat (GRAY): 190 mm square, 2 mm thin KINEMATIC
    plates with NO collider — flat painted markers the die rolls over freely.

Per-episode randomization (readback-verifiable): die start face sampled from the FIVE
non-blue faces (blue never starts up), free yaw, xy jitter; the two mats jitter in x/y
and RANDOMLY SWAP sides, so a memorized push direction fails.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.20 * rolled — the settled up-face ever differs from the reset up-face (latched)
  0.25 * blue   — the die ever settles blue-face-up anywhere (latched)
  0.15 * near   — the die center ever comes within `approach_r` of the target-mat
                  center (latched; spawn separation >= 0.24 m makes this unreachable
                  by the null policy)
  1.0 iff success() — blue up (within `up_max_deg`), center within `zone_tol` of the
                  GREEN mat center, resting on the ground, settled and finite.
  Non-success capped at 0.60 (float32 stores it as 0.60000002 — asserts elsewhere use
  a +0.001 epsilon).

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# The five START orientations (local face put world-up; blue = local +z is EXCLUDED):
# (w, x, y, z) rows for [+x up, -x up, +y up, -y up, -z up].
_R2 = math.sqrt(0.5)
START_QUATS = (
    (_R2, 0.0, -_R2, 0.0),   # +x (RED) up      : rot y -90
    (_R2, 0.0, +_R2, 0.0),   # -x (ORANGE) up   : rot y +90
    (_R2, +_R2, 0.0, 0.0),   # +y (YELLOW) up   : rot x +90
    (_R2, -_R2, 0.0, 0.0),   # -y (WHITE) up    : rot x -90
    (0.0, 1.0, 0.0, 0.0),    # -z (PURPLE) up   : rot x 180
)
# Direction-index enumeration for the up-face: 0:+x 1:-x 2:+y 3:-y 4:+z(BLUE) 5:-z.
START_FACE_IDX = (0, 1, 2, 3, 5)


# ----- external-wrench driver (solve/smoke instrumentation, scene-agnostic) ---------------------
class ExternalWrenchDriver:
    """Apply WORLD-frame wrenches to a RigidObject across the forge pods' frame quirks.

    On these pods `set_external_force_and_torque(..., is_global=True)` may rotate the
    given wrench by the body's rotation since reset (applied = R_now.R_ref^T.given),
    and older builds without `is_global` apply it in the FULL body frame
    (applied = R_now.given). Modes:
      "drag":  pre-encode with M = R_ref.R_now^T  (cancels the since-reset drag)
      "world": raw world vector
      "body":  pre-encode with R_now^T            (for is_global-unsupported builds)
    `calibrate()` picks the mode empirically by pushing and reading the velocity
    direction; callers may also `cycle()` when a measured response disagrees.
    """

    MODES = ("drag", "world", "body")

    def __init__(self, body: Any, num_envs: int, device: str) -> None:
        self.body = body
        self.n = num_envs
        self.device = device
        self.all_ids = torch.arange(num_envs, device=device)
        self.mode = "drag"
        self.q_ref = body.data.root_quat_w.clone()
        self.has_global = True
        try:  # probe the signature once with a zero wrench
            z = torch.zeros(self.n, 1, 3, device=device)
            body.set_external_force_and_torque(z, z, env_ids=self.all_ids, is_global=True)
        except TypeError:
            self.has_global = False
            self.mode = "body"

    def snapshot_ref(self) -> None:
        self.q_ref = self.body.data.root_quat_w.clone()

    def cycle(self) -> str:
        if not self.has_global:
            return self.mode  # only "body" makes sense without is_global
        i = self.MODES.index(self.mode)
        self.mode = self.MODES[(i + 1) % len(self.MODES)]
        return self.mode

    def _encode(self, vec_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        if self.mode == "world":
            return vec_w
        q_now = self.body.data.root_quat_w
        if self.mode == "drag":
            return quat_apply(_qmul(self.q_ref, _qconj(q_now)), vec_w)
        return quat_apply_inverse(q_now, vec_w)

    def apply(self, force_w: torch.Tensor, torque_w: torch.Tensor) -> None:
        f = self._encode(force_w).view(self.n, 1, 3)
        t = self._encode(torque_w).view(self.n, 1, 3)
        if self.has_global:
            self.body.set_external_force_and_torque(f, t, env_ids=self.all_ids, is_global=True)
        else:
            self.body.set_external_force_and_torque(f, t, env_ids=self.all_ids)

    def clear(self) -> None:
        z = torch.zeros(self.n, 3, device=self.device)
        self.apply(z, z)

    def calibrate(self, step: Callable[[], None], force: float = 3.6, tag: str = "") -> str:
        """Push along world +x (or -x when `force` < 0) for a few steps per candidate
        mode; keep the first mode whose measured horizontal DISPLACEMENT aligns with
        the push. Judged on position (not velocity) readback: velocity readback can
        be unreliable while an external wrench is active on some builds. At reset
        orientation the "drag" and "world" modes coincide (callers re-verify after
        rotations)."""
        sgn = 1.0 if force >= 0 else -1.0
        ex = torch.tensor([[force, 0.0, 0.0]], device=self.device).expand(self.n, 3)
        z = torch.zeros(self.n, 3, device=self.device)
        for _ in range(len(self.MODES)):
            p0 = self.body.data.root_pos_w[0, :2].clone()
            self.apply(ex, z)
            for _k in range(22):
                step()
            dp = self.body.data.root_pos_w[0, :2] - p0
            self.clear()
            for _k in range(40):
                step()
            dx, dn = float(dp[0]), float(dp.norm())
            print(f"[wrench] calibrate{tag}: mode={self.mode} dp=({dx * 1000:+.1f},"
                  f"{float(dp[1]) * 1000:+.1f}) mm", flush=True)
            if dn > 0.004 and sgn * dx > 0.7 * dn:
                return self.mode
            self.cycle()
        return self.mode


# ----- custom compound spawner (collision cube + six visual face plates) ------------------------
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


def _add_box(stage, path: str, *, center, size, color, collide=None):
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


def _spawn_die(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the die at `prim_path`: DYNAMIC compound. Local frame: origin at the cube
    center. Collision = the core cube ONLY (clean box contact); the six colored face
    plates are visual-only, 1 mm proud of the core. Friction material bound at the
    root (custom-spawner colliders otherwise default to ~0.5). MassAPI and the PhysX
    body props are authored HERE: a custom spawner func gets no automatic
    mass_props/rigid_props application, so without these the mass silently falls back
    to density x volume (0.729 kg, not 0.45 -> the slide window moves above the
    solve's forces) and a settled die could sleep through external wrenches."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.15)
    pxrb.CreateAngularDampingAttr(0.25)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    c = cfg
    e, t, s = c.die_e, c.face_t, c.die_e * 0.86
    _add_box(stage, f"{prim_path}/core", center=(0.0, 0.0, 0.0),
             size=(e, e, e), color=c.core_color, collide=collide)
    o = e / 2 - t / 2 + 0.001  # plate outer surface 1 mm proud of the core
    faces = (
        ("face_pz", (0.0, 0.0, +o), (s, s, t), c.blue),
        ("face_nz", (0.0, 0.0, -o), (s, s, t), c.purple),
        ("face_px", (+o, 0.0, 0.0), (t, s, s), c.red),
        ("face_nx", (-o, 0.0, 0.0), (t, s, s), c.orange),
        ("face_py", (0.0, +o, 0.0), (s, t, s), c.yellow),
        ("face_ny", (0.0, -o, 0.0), (s, t, s), c.white),
    )
    for name, center, size, color in faces:
        _add_box(stage, f"{prim_path}/{name}", center=center, size=size,
                 color=color, collide=None)  # visual only

    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    sim_utils.spawn_rigid_body_material(
        f"{prim_path}/physmat",
        sim_utils.RigidBodyMaterialCfg(
            static_friction=float(c.mu_static), dynamic_friction=float(c.mu_dynamic),
            restitution=0.0))
    bind_physics_material(prim_path, f"{prim_path}/physmat")
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclass (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "die" not in _SPAWNER_CACHE:

        @configclass
        class DieSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_die)
            die_e: float = 0.090
            face_t: float = 0.003
            contact_offset: float = 0.002
            mu_static: float = 0.7
            mu_dynamic: float = 0.65
            core_color: tuple = (0.45, 0.45, 0.48)
            blue: tuple = (0.10, 0.25, 0.95)
            purple: tuple = (0.55, 0.10, 0.80)
            red: tuple = (0.85, 0.08, 0.08)
            orange: tuple = (0.95, 0.55, 0.05)
            yellow: tuple = (0.92, 0.85, 0.05)
            white: tuple = (0.95, 0.95, 0.95)

        _SPAWNER_CACHE.update(die=DieSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DieRollSceneCfg(BaseCfg):
    """Config for `DieRollScene`. The physics claims are honest by construction
    (asserted in __post_init__): the die spawns at least `approach_r` + 0.06 m from
    the target mat (null policy can never earn the approach latch), the two mats stay
    farther apart than the success tolerance (a die on the decoy can never count),
    and the sliding force window is real (breakaway mu*m*g < CoM-height tip force
    m*g since mu < 1)."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    up_max_deg: float = tunable(12.0)    # blue axis within this of world-up
    zone_tol: float = tunable(0.055)     # die center within this of the GREEN mat center (m)
    z_tol: float = tunable(0.012)        # die center within this of e/2 (resting on the ground)
    settle_lin: float = tunable(0.05)    # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.40)    # max |ang vel| when judging (rad/s)
    latch_ang: float = tunable(0.50)     # max |ang vel| for the rolled/blue latches to arm
    up_snap: float = tunable(0.95)       # min face-up dot for the up-face index to be valid
    approach_r: float = tunable(0.15)    # latched approach credit radius around the target mat

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    die_jx: float = tunable(0.04)        # die xy jitter (+/- m) around die_center
    die_jy: float = tunable(0.12)
    yaw_deg: float = tunable(180.0)      # die free yaw (+/- deg)
    mat_x_jit: float = tunable(0.04)     # per-mat x jitter (+/- m)
    mat_y_jit: float = tunable(0.04)     # per-mat |y| jitter (+/- m)
    swap_sides: bool = tunable(True)     # target/decoy randomly swap y sides per episode

    # --- info: layout (env frame, ground z = 0) --------------------------------------------------
    die_center: tuple = info((0.28, 0.0))
    mat_x: float = info(0.60)            # both mats' nominal x
    mat_y: float = info(0.15)            # nominal |y| of each mat (opposite sides)
    # --- info: die -------------------------------------------------------------------------------
    die_e: float = info(0.090)           # edge length — wider than a parallel jaw's ~80 mm span
    die_m: float = info(0.45)
    face_t: float = info(0.003)
    contact_offset: float = info(0.002)
    mu_static: float = info(0.7)         # bound on die AND ground (avg combine -> 0.7)
    mu_dynamic: float = info(0.65)
    # --- info: mats ------------------------------------------------------------------------------
    mat_size: float = info(0.19)
    mat_t: float = info(0.002)
    goal_color: tuple = info((0.10, 0.72, 0.20))
    decoy_color: tuple = info((0.28, 0.28, 0.30))
    # --- info: rubric weights (0.20 + 0.25 + 0.15 = 0.60 = the non-success cap) ------------------
    w_roll: float = info(0.20)
    w_blue: float = info(0.25)
    w_near: float = info(0.15)

    def __post_init__(self) -> None:
        # Honesty-by-construction asserts (the claims the task rests on).
        min_sep = (self.mat_x - self.mat_x_jit) - (self.die_center[0] + self.die_jx)
        assert min_sep >= self.approach_r + 0.06, \
            "die spawn must stay outside the approach latch radius (null policy earns 0)"
        assert 2 * (self.mat_y - self.mat_y_jit) >= self.zone_tol + 0.08, \
            "the decoy mat must stay outside the success tolerance of the target mat"
        assert self.zone_tol <= self.mat_size / 2, \
            "the success tolerance must keep the die center on the painted mat"
        assert self.mu_static < 0.95, \
            "sliding must break away below the CoM-height tip force (mu < 1)"
        assert self.up_snap < math.cos(math.radians(self.up_max_deg)) + 0.05


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("die_roll")
class DieRollScene(BaseScene):
    cfg: DieRollSceneCfg

    def __init__(self, cfg: DieRollSceneCfg | None = None) -> None:
        super().__init__(cfg or DieRollSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()

        die_spawn = cls["die"](
            # consumed by _spawn_die, which authors MassAPI + PhysX body props itself
            # (a custom spawner func gets no automatic schema application)
            mass_props=sim_utils.MassPropertiesCfg(mass=c.die_m),
            die_e=c.die_e, face_t=c.face_t, contact_offset=c.contact_offset,
            mu_static=c.mu_static, mu_dynamic=c.mu_dynamic)

        def mat_cfg(color: tuple, x: float, y: float) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Mat_" + ("goal" if color == c.goal_color else "decoy"),
                spawn=sim_utils.CuboidCfg(
                    size=(c.mat_size, c.mat_size, c.mat_t),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
                    collision_props=None,  # flat painted marker: NO collider
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, c.mat_t / 2)),
            )

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
            "die": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Die",
                spawn=die_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.die_center[0], c.die_center[1], c.die_e / 2 + 0.003)),
            ),
            "goal_mat": mat_cfg(c.goal_color, c.mat_x, +c.mat_y),
            "decoy_mat": mat_cfg(c.decoy_color, c.mat_x, -c.mat_y),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # Without this, external wrenches are under-applied across TGS
                # iterations (IsaacLab warns at startup; treat that warning as fatal).
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
        self.die: RigidObject = env.iscene["die"]
        self.goal_mat: RigidObject = env.iscene["goal_mat"]
        self.decoy_mat: RigidObject = env.iscene["decoy_mat"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.init_face = torch.full((n,), 5, dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._rolled = torch.zeros(n, dtype=torch.bool, device=dev)
        self._blue = torch.zeros(n, dtype=torch.bool, device=dev)
        self._near = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: die at its spawn zone with xy jitter, free yaw, and a start
        face sampled from the FIVE non-blue faces (blue never starts up — at least
        one roll is always required); mats jittered and (optionally) side-swapped;
        latches cleared. Discrete draws use torch.rand comparisons (the first
        torch.randint after manual_seed is near-degenerate across seeds)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- die: start face (5 non-blue), free yaw, xy jitter ---
        pick = (torch.rand(m, device=dev) * 5).floor().long().clamp_(0, 4)
        qf = torch.tensor(START_QUATS, device=dev)[pick]
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.die_center[0] + (torch.rand(m, device=dev) * 2 - 1) * c.die_jx
        st[:, 1] = c.die_center[1] + (torch.rand(m, device=dev) * 2 - 1) * c.die_jy
        st[:, 2] = c.die_e / 2 + 0.003
        st[:, 3:7] = _qmul(_qz(yaw), qf)
        st[:, 0:3] += origin
        self.die.write_root_state_to_sim(st, env_ids)
        self.init_face[env_ids] = torch.tensor(START_FACE_IDX, device=dev)[pick]

        # --- mats: x/y jitter, random side swap ---
        side = torch.where(torch.rand(m, device=dev) > 0.5, 1.0, -1.0) \
            if c.swap_sides else torch.ones(m, device=dev)
        for body, sgn in ((self.goal_mat, side), (self.decoy_mat, -side)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.mat_x + (torch.rand(m, device=dev) * 2 - 1) * c.mat_x_jit
            st[:, 1] = sgn * (c.mat_y + (torch.rand(m, device=dev) * 2 - 1) * c.mat_y_jit)
            st[:, 2] = c.mat_t / 2
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._rolled[env_ids] = False
        self._blue[env_ids] = False
        self._near[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "die": self.die.data.root_state_w[env_ids].clone(),
            "goal_mat": self.goal_mat.data.root_state_w[env_ids].clone(),
            "decoy_mat": self.decoy_mat.data.root_state_w[env_ids].clone(),
            "init_face": self.init_face[env_ids].clone(),
            "rolled": self._rolled[env_ids].clone(),
            "blue": self._blue[env_ids].clone(),
            "near": self._near[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.die.write_root_state_to_sim(state["die"], env_ids)
        self.goal_mat.write_root_state_to_sim(state["goal_mat"], env_ids)
        self.decoy_mat.write_root_state_to_sim(state["decoy_mat"], env_ids)
        self.init_face[env_ids] = state["init_face"]
        self._rolled[env_ids] = state["rolled"]
        self._blue[env_ids] = state["blue"]
        self._near[env_ids] = state["near"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A large cubic DIE ({c.die_e * 1000:.0f} mm on a side — too wide for a "
            f"parallel-jaw gripper to grasp) rests on the floor; its position and heading "
            f"vary per episode. Its six faces are painted six distinct colors: BLUE, "
            f"PURPLE (the face opposite blue), RED, ORANGE, YELLOW and WHITE. The BLUE "
            f"face NEVER starts facing up. On the far side of the floor lie two flat "
            f"painted square mats ({c.mat_size * 100:.0f} cm across, paper-thin — the die "
            f"rolls and slides over them freely): a GREEN target mat and a dark-GRAY "
            f"decoy mat. Their positions jitter and they randomly swap sides every "
            f"episode, so identify the green one by color.\n"
            f"Goal: leave the die AT REST on the floor, centered on the GREEN mat (its "
            f"center within {c.zone_tol * 100:.1f} cm of the mat center), with the BLUE "
            f"face pointing UP (within {c.up_max_deg:.0f} degrees of vertical). Because "
            f"the die cannot be grasped, reorient it by TIPPING it over its bottom edges "
            f"— push high on a face and it pivots over onto the next face (one "
            f"quarter-roll changes which face is up and moves the die one edge-length); "
            f"push low and it slides without changing the up face. Plan the sequence of "
            f"quarter-rolls that brings blue up (one roll if blue faces sideways, two if "
            f"blue faces down), slide to position it, and leave it settled on the green "
            f"mat. The gray mat pays nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Tip the large die over its edges until its blue face points up, then leave "
            "it at rest centered on the green mat. The die is too wide to grasp — roll "
            "it by pushing high on a face, slide it by pushing low."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _die_axes(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """World directions of the die's local +x, +y, +z axes, each (N, 3)."""
        from isaaclab.utils.math import quat_apply

        q = self.die.data.root_quat_w
        n = q.shape[0]
        dev = q.device
        ex = quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3))
        ey = quat_apply(q, torch.tensor([0.0, 1.0, 0.0], device=dev).expand(n, 3))
        ez = quat_apply(q, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3))
        return ex, ey, ez

    def face_up(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(idx (N,) long, dot (N,)): which face direction points most nearly up
        (0:+x 1:-x 2:+y 3:-y 4:+z=BLUE 5:-z) and its dot with world-up."""
        ex, ey, ez = self._die_axes()
        dots = torch.stack([ex[:, 2], -ex[:, 2], ey[:, 2], -ey[:, 2],
                            ez[:, 2], -ez[:, 2]], dim=1)
        best, idx = dots.max(dim=1)
        return idx, best

    def blue_up(self) -> torch.Tensor:
        """(N,) bool: the local +z (BLUE) axis within `up_max_deg` of world-up."""
        _ex, _ey, ez = self._die_axes()
        return ez[:, 2] > math.cos(math.radians(self.cfg.up_max_deg))

    def goal_dist(self) -> torch.Tensor:
        """(N,) horizontal distance die center -> GREEN mat center."""
        return (self.die.data.root_pos_w[:, :2]
                - self.goal_mat.data.root_pos_w[:, :2]).norm(dim=-1)

    def on_goal(self) -> torch.Tensor:
        """(N,) bool, geometric: die center within `zone_tol` of the green mat center
        and resting on the ground (center height ~ e/2)."""
        c = self.cfg
        z = self.die.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return (self.goal_dist() < c.zone_tol) & ((z - c.die_e / 2).abs() < c.z_tol)

    def settled(self) -> torch.Tensor:
        """(N,) bool: |lin vel| and |ang vel| below the judging gates."""
        c = self.cfg
        return (self.die.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.die.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def _update_latches(self) -> None:
        c = self.cfg
        idx, best = self.face_up()
        calm = self.die.data.root_ang_vel_w.norm(dim=-1) < c.latch_ang
        finite = torch.isfinite(self.die.data.root_pos_w).all(dim=-1)
        snapped = (best > c.up_snap) & calm & finite
        self._rolled |= snapped & (idx != self.init_face)
        self._blue |= snapped & self.blue_up()
        self._near |= finite & (self.goal_dist() < c.approach_r)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: blue face up, die centered on the GREEN mat, resting on the
        ground, settled and finite. All clauses are live physical outcomes."""
        self._update_latches()
        finite = torch.isfinite(self.die.data.root_state_w).all(dim=-1)
        return self.blue_up() & self.on_goal() & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.20*rolled + 0.25*blue + 0.15*near (all latched;
        ~0 for doing nothing), capped at 0.60 — and exactly 1.0 iff success() holds
        live. (float32 stores the cap as 0.60000002; asserts use +0.001 epsilon.)"""
        c = self.cfg
        self._update_latches()
        base = (c.w_roll * self._rolled.float() + c.w_blue * self._blue.float()
                + c.w_near * self._near.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; the die is driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="die_roll", robot="null"))
