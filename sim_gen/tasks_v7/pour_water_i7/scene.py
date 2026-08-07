"""RampChockScene — park two balls on an inclined ramp's marked band, held by a chock
(sim_gen task `pour_water_i7`).

Derived from pick_place/pour_water, but STRATEGICALLY different: the seed is a
trajectory-tracking POUR — grasp a small block ("water"), carry it along five rotation-
tracked waypoints and tip it over a vase; its reward is dominated by how faithfully the
gripper follows a prescribed SE(3) path, and its end state is a held, tilted object above
a target. Here nothing is poured, tilted along a path, or held at the end: the task is to
CONSTRUCT A STATIC EQUILIBRIUM that does not exist naturally. Two green balls must end
up at rest on the marked band of a 12-degree ramp — a pose in which a free sphere
physically CANNOT rest (PhysX spheres are analytic and have no rolling resistance: on
any incline they roll) — so the solver must first wedge the blue CHOCK bar across the
band and then release each ball just uphill of it, letting gravity roll the ball back
into the chock face: a gravity-loaded force chain (deck -> chock -> ball). The judged
state is maintained entirely by contact; remove the chock and both balls roll off the
ramp (smoke proves this counterfactual). Execution order is enforced by GRAVITY, not by
rubric fiat: a ball released on the band before the chock simply rolls out of the
scene. The red BALL is a physically self-rejecting decoy chock: wedged into the band it
rolls off the ramp itself, taking the "chain" with it — only the flat-faced bar can
hold the slope.

A solver therefore needs a different PLAN each episode (read the ramp's heading — the
fall line rotates per episode — orient the chock across it, wedge the chock first, then
park each ball against it) and a different CODE STRUCTURE (a ramp-frame slope
coordinate system, a per-ball support-chain predicate over three bodies, latched
per-body credit) — not a waypoint tracker.

success(): both balls at rest ON the deck surface (center one radius above the deck
plane), centers inside the yellow band window, each DIRECTLY uphill-adjacent to the
chock (within `chock_gap_max` along-slope); the chock at rest flat on the deck,
cross-slope, in the band zone; all three settled (lin + ang gates) and finite. All
clauses are live physical outcomes — the equilibrium is held by contact alone.

score() is latched (credit never evaporates): 0.15 * chock ever parked in the band
zone + 0.05 * each ball ever on the deck + 0.20 * each ball ever parked against the
chock (in band, settled, supported) — cap 0.65; exactly 1.0 iff success() live.
Doing nothing scores ~0 (nothing spawns on the ramp).

Assets are fully procedural (no external files):
  - ramp: KINEMATIC compound — a 550 x 340 x 30 mm deck pitched 12 degrees with its
    downhill edge at ground level (balls roll off it onto the floor), a support block
    under the high end, and a VISUAL-ONLY yellow band stripe (no collider — a proud
    stripe would itself chock the balls) centered 300 mm up the slope. High-friction
    physics material (0.85/0.75) bound to the deck: friction is load-bearing (it is
    what keeps the loaded chock from sliding; the no-slide and no-tip inequalities
    are asserted in __post_init__).
  - ball (x2, dynamic, 400 g): green sphere, diameter 60 mm (< the 80 mm Franka jaw
    stroke).
  - chock (dynamic, 800 g): blue bar 50 x 160 x 34 mm — its flat face is taller than
    the ball contact height (asserted), so it arrests a rolling ball.
  - decoy (dynamic, 300 g): red sphere, diameter 40 mm — the round "chock" that
    cannot chock.
Contact offsets are explicit (1.5 mm) so the resting-height clause (ball center one
radius above the deck plane) stays real.

Per-episode randomization (readback-verified in smoke): ramp xy jitter + yaw (the fall
line swings +/-50 degrees — the cross-slope direction must be read, not memorized),
ball / chock / decoy ground spawns with free yaw and batched keep-out resampling
(outside the ramp footprint, pairwise separated, inside the reach annulus). Heavy
imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
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


# ----- custom compound spawner (ramp only; balls/chock/decoy use built-ins) ---------------------
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
    """One USD physics material (friction is load-bearing: it holds the loaded chock)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float,
         quat=None, material=None, collide: bool = True) -> None:
    """Author one box child prim (translate -> orient -> scale; authored once)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if quat is not None:
        w, x, y, z = (float(v) for v in quat)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
        if material is not None:
            UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_ramp(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC ramp at `prim_path`. Local frame: origin at the FOOT
    (downhill deck edge) on the ground, local +x pointing uphill (in plan), the deck
    top surface passing through the origin so balls roll off onto the floor."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(25.0)

    th = math.radians(cfg.slope_deg)
    ct, st = math.cos(th), math.sin(th)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    # deck: box rotated about +y by -th (local +x maps to the uphill direction
    # (ct, 0, st)); center = d*(L/2) - n*(t/2) with d=(ct,0,st), n=(-st,0,ct).
    qd = (math.cos(th / 2), 0.0, -math.sin(th / 2), 0.0)
    L, t = cfg.deck_len, cfg.deck_t
    _box(stage, f"{prim_path}/deck", (L, cfg.deck_w, t),
         (L / 2 * ct + t / 2 * st, 0.0, L / 2 * st - t / 2 * ct),
         cfg.deck_color, cfg.contact_offset, quat=qd, material=mat)
    # support block under the high end (visual grounding; unreachable by balls)
    ub = L - 0.06
    top = ub * st - t * ct
    if top > 0.02:
        _box(stage, f"{prim_path}/support", (0.10, cfg.deck_w - 0.06, top),
             (ub * ct, 0.0, top / 2), cfg.support_color, cfg.contact_offset, material=mat)
    # band stripe: VISUAL ONLY (no collider — a proud stripe would chock the balls)
    ub_c = (cfg.band_lo + cfg.band_hi) / 2
    _box(stage, f"{prim_path}/band", (cfg.band_hi - cfg.band_lo, cfg.deck_w - 0.04, 0.0012),
         (ub_c * ct + 0.0012 * st, 0.0, ub_c * st + 0.0012 * ct),
         cfg.band_color, cfg.contact_offset, quat=qd, collide=False)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclass (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "ramp" not in _SPAWNER_CACHE:

        @configclass
        class RampSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ramp)
            slope_deg: float = 12.0
            deck_len: float = 0.55
            deck_w: float = 0.34
            deck_t: float = 0.03
            band_lo: float = 0.22
            band_hi: float = 0.38
            mu_static: float = 0.85
            mu_dynamic: float = 0.75
            deck_color: tuple = (0.42, 0.40, 0.44)
            support_color: tuple = (0.30, 0.29, 0.33)
            band_color: tuple = (0.92, 0.80, 0.12)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["ramp"] = RampSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RampChockSceneCfg(BaseCfg):
    """Config for `RampChockScene`. Honesty is asserted in __post_init__: the loaded
    chock neither slides nor tips at these masses/frictions/slope, its flat face is
    tall enough to arrest a ball, and everything manipulated is jaw-sized. That a
    free sphere cannot rest on the deck needs no assert — PhysX spheres are analytic
    and roll on any incline (smoke proves it empirically)."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    rest_tol: float = tunable(0.010)      # |deck-normal height - nominal| gate for "on deck" (m)
    y_max: float = tunable(0.105)         # |ramp-frame y| gate (stay near deck center)
    chock_axis_max_deg: float = tunable(25.0)  # chock length axis within this of cross-slope
    chock_gap_max: float = tunable(0.085)  # max along-slope gap: ball center - chock center
    settle_lin: float = tunable(0.05)     # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(1.0)      # max |ang vel| when judging (rad/s; balls spin easily)
    zone_lo_slack: float = tunable(0.06)  # chock zone extends this far below the band

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    ramp_yaw_deg: float = tunable(50.0)   # ramp yaw about +x (uphill away from robot), +/- deg
    ramp_jitter: float = tunable(0.04)    # ramp foot xy jitter (+/- m)
    spawn_r_min: float = tunable(0.18)    # ground spawns: reach annulus (m)
    spawn_r_max: float = tunable(0.60)
    spawn_sep: float = tunable(0.13)      # pairwise ground-spawn separation (m)

    # --- info: layout ---------------------------------------------------------------------------
    ramp_pos: tuple = info((0.26, 0.0))   # ramp foot on the ground (nominal)
    # --- info: ramp geometry (local frame: origin at the foot, +x uphill in plan) ---------------
    slope_deg: float = info(12.0)
    deck_len: float = info(0.55)          # along-slope deck length
    deck_w: float = info(0.34)
    deck_t: float = info(0.03)
    band_lo: float = info(0.22)           # band window, along-slope coordinate u (m)
    band_hi: float = info(0.38)
    mu_static: float = info(0.85)
    mu_dynamic: float = info(0.75)
    # --- info: bodies ---------------------------------------------------------------------------
    ball_r: float = info(0.030)
    ball_mass: float = info(0.40)
    chock_w: float = info(0.050)          # along-slope footprint
    chock_len: float = info(0.160)        # cross-slope length (local y)
    chock_h: float = info(0.034)
    chock_mass: float = info(0.80)
    decoy_r: float = info(0.020)
    decoy_mass: float = info(0.30)
    ball_color: tuple = info((0.10, 0.62, 0.22))
    chock_color: tuple = info((0.12, 0.30, 0.85))
    decoy_color: tuple = info((0.85, 0.10, 0.10))
    contact_offset: float = info(0.0015)
    # rubric weights (0.15 + 2*0.05 + 2*0.20 = 0.65 = the non-success cap)
    w_chock: float = info(0.15)
    w_deck: float = info(0.05)
    w_park: float = info(0.20)

    def __post_init__(self) -> None:
        th = math.radians(self.slope_deg)
        mu = self.mu_dynamic  # deck & chock share the material (combine = average)
        # loaded chock does not SLIDE: friction on the chock exceeds its own downslope
        # pull plus the deck-parallel push from both balls
        drive = (self.chock_mass + 2 * self.ball_mass) * math.sin(th)
        hold = mu * self.chock_mass * math.cos(th)
        assert hold > 1.25 * drive, "chock would slide under the loaded chain"
        # loaded chock does not TIP about its downhill bottom edge
        tip = 2 * self.ball_mass * math.sin(th) * self.ball_r
        keep = self.chock_mass * math.cos(th) * (self.chock_w / 2)
        assert keep > 1.5 * tip, "chock would tip under the loaded chain"
        # the chock face is taller than the ball contact height: it arrests, the
        # ball cannot climb it quasi-statically
        assert self.chock_h >= self.ball_r, "chock face below ball equator"
        # two balls fit side by side against the chock face
        assert self.chock_len > 4 * self.ball_r + 0.02, "chock too short for two balls"
        # the parked chain fits inside the band window
        assert (self.band_hi - self.band_lo) > self.chock_w / 2 + 2 * self.ball_r + 0.02
        assert self.band_hi + 2 * self.ball_r < self.deck_len
        # graspable by the 80 mm Franka jaw
        assert 2 * self.ball_r <= 0.078 and self.chock_h <= 0.078 and 2 * self.decoy_r <= 0.078
        # the chock itself rests on the bare slope (its own no-slide condition)
        assert math.tan(th) < mu


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


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("ramp_chock")
class RampChockScene(BaseScene):
    cfg: RampChockSceneCfg

    BALL_NAMES = ("ball_a", "ball_b")

    def __init__(self, cfg: RampChockSceneCfg | None = None) -> None:
        super().__init__(cfg or RampChockSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        ramp_spawn = _spawner_classes()["ramp"](
            mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            slope_deg=c.slope_deg, deck_len=c.deck_len, deck_w=c.deck_w, deck_t=c.deck_t,
            band_lo=c.band_lo, band_hi=c.band_hi, mu_static=c.mu_static,
            mu_dynamic=c.mu_dynamic, contact_offset=c.contact_offset)

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.05,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=16, solver_velocity_iteration_count=1)
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        pmat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_static, dynamic_friction=c.mu_dynamic, restitution=0.0)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.85, dynamic_friction=0.75, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "ramp": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ramp",
                spawn=ramp_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.ramp_pos[0], c.ramp_pos[1], 0.0)),
            ),
            "chock": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Chock",
                spawn=sim_utils.CuboidCfg(
                    size=(c.chock_w, c.chock_len, c.chock_h),
                    rigid_props=rigid, collision_props=coll, physics_material=pmat,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.chock_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.chock_color)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, -0.6, c.chock_h / 2)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=sim_utils.SphereCfg(
                    radius=c.decoy_r,
                    rigid_props=rigid, collision_props=coll, physics_material=pmat,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.decoy_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.decoy_color)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.7, -0.6, c.decoy_r)),
            ),
        }
        for i, name in enumerate(self.BALL_NAMES):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball_" + name,
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    rigid_props=rigid, collision_props=coll, physics_material=pmat,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ball_color)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8 + 0.1 * i, -0.6, c.ball_r)),
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
        self.ramp: RigidObject = env.iscene["ramp"]
        self.chock: RigidObject = env.iscene["chock"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.balls: dict[str, RigidObject] = {n: env.iscene[n] for n in self.BALL_NAMES}
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # latches (partial credit survives transients; success is judged live)
        self._chock_zone = torch.zeros(n, dtype=torch.bool, device=dev)
        self._on_deck = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self._parked = torch.zeros(n, 2, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the ramp (yaw + xy jitter), scatter chock / decoy /
        balls on the flat ground (free yaw, keep-out resampled), clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- ramp: kinematic, yaw + xy jitter ---
        psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.ramp_yaw_deg)
        q_ramp = _qz(psi)
        rp = torch.zeros(m, 3, device=dev)
        rp[:, 0] = c.ramp_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.ramp_jitter
        rp[:, 1] = c.ramp_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.ramp_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = rp + origin
        st[:, 3:7] = q_ramp
        self.ramp.write_root_state_to_sim(st, env_ids)

        # --- ground spawns: 4 objects, keep-out resampled ---
        th = math.radians(c.slope_deg)
        foot_len = c.deck_len * math.cos(th)
        cpsi, spsi = torch.cos(psi), torch.sin(psi)
        xy = torch.zeros(m, 4, 2, device=dev)
        bad = torch.ones(m, 4, dtype=torch.bool, device=dev)
        for _ in range(40):
            if not bad.any():
                break
            k = int(bad.sum())
            cand = torch.rand(k, 2, device=dev) * (2 * (c.spawn_r_max + 0.02)) - (c.spawn_r_max + 0.02)
            xy[bad] = cand
            # inside the reach annulus
            r = xy.norm(dim=-1)
            ok = (r > c.spawn_r_min) & (r < c.spawn_r_max)
            # outside the ramp footprint (ramp frame, with margin)
            rel = xy - rp[:, None, 0:2]
            u = rel[..., 0] * cpsi[:, None] + rel[..., 1] * spsi[:, None]
            v = -rel[..., 0] * spsi[:, None] + rel[..., 1] * cpsi[:, None]
            ok &= ~((u > -0.10) & (u < foot_len + 0.12) & (v.abs() < c.deck_w / 2 + 0.10))
            # pairwise separation
            d = (xy[:, :, None, :] - xy[:, None, :, :]).norm(dim=-1)
            d += torch.eye(4, device=dev) * 10.0
            ok &= d.min(dim=-1).values > c.spawn_sep
            bad = ~ok
        yaw = torch.rand(m, 4, device=dev) * 2 * math.pi

        def _write(body, i: int, z: float) -> None:
            s = torch.zeros(m, 13, device=dev)
            s[:, 0] = xy[:, i, 0]
            s[:, 1] = xy[:, i, 1]
            s[:, 2] = z
            s[:, 3:7] = _qz(yaw[:, i])
            s[:, 0:3] += origin
            body.write_root_state_to_sim(s, env_ids)

        _write(self.chock, 0, c.chock_h / 2)
        _write(self.decoy, 1, c.decoy_r)
        _write(self.balls["ball_a"], 2, c.ball_r)
        _write(self.balls["ball_b"], 3, c.ball_r)

        # --- clear latches ---
        self._chock_zone[env_ids] = False
        self._on_deck[env_ids] = False
        self._parked[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "ramp": self.ramp.data.root_state_w[env_ids].clone(),
            "chock": self.chock.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "balls": {n: b.data.root_state_w[env_ids].clone()
                      for n, b in self.balls.items()},
            "chock_zone": self._chock_zone[env_ids].clone(),
            "on_deck": self._on_deck[env_ids].clone(),
            "parked": self._parked[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.ramp.write_root_state_to_sim(state["ramp"], env_ids)
        self.chock.write_root_state_to_sim(state["chock"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        for n, b in self.balls.items():
            b.write_root_state_to_sim(state["balls"][n], env_ids)
        self._chock_zone[env_ids] = state["chock_zone"]
        self._on_deck[env_ids] = state["on_deck"]
        self._parked[env_ids] = state["parked"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A gray RAMP rises at {c.slope_deg:.0f} degrees from the floor: its low edge "
            f"sits at ground level and its deck climbs {c.deck_len * 1000:.0f} mm up-slope. "
            f"Partway up, a YELLOW BAND is painted across the deck — the parking zone. "
            f"On the flat floor around the ramp lie two GREEN BALLS "
            f"({2 * c.ball_r * 1000:.0f} mm across), a BLUE CHOCK BAR (a rectangular "
            f"block, {c.chock_w * 1000:.0f} x {c.chock_len * 1000:.0f} x "
            f"{c.chock_h * 1000:.0f} mm) and a smaller RED BALL. The ramp's position and "
            f"heading change every episode, and every loose object starts at a random "
            f"spot — read the slope direction from the scene. A ball has nowhere to rest "
            f"on the bare deck: released anywhere on the ramp it rolls straight off.\n"
            f"Goal: park BOTH green balls at rest on the yellow band, held in place by "
            f"the blue chock. Lay the chock flat across the deck at the lower edge of the "
            f"band first; then release each green ball on the band just uphill of it, so "
            f"it rolls back and comes to rest against the chock face — the two balls "
            f"sitting side by side. The red ball is no chock — wedged on the band it "
            f"rolls away itself. Success: both green balls at rest inside the band, each "
            f"directly against the chock wedged below it, everything settled and held "
            f"only by gravity and contact."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lay the blue chock bar flat across the ramp at the lower edge of the "
            "yellow band, then release both green balls on the band just uphill of it "
            "so they roll back and rest side by side against the chock. Do not use the "
            "red ball."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _ramp_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points (N,3) -> ramp frame (kinematic, live-read)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.ramp.data.root_quat_w,
                                  pos_w - self.ramp.data.root_pos_w)

    def _slope_coords(self, pos_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(u, v, h): along-slope, cross-slope, height above the deck-top plane."""
        th = math.radians(self.cfg.slope_deg)
        loc = self._ramp_local(pos_w)
        u = loc[:, 0] * math.cos(th) + loc[:, 2] * math.sin(th)
        h = -loc[:, 0] * math.sin(th) + loc[:, 2] * math.cos(th)
        return u, loc[:, 1], h

    def _cross_axis_w(self) -> torch.Tensor:
        """(N,3) world unit vector of the ramp's cross-slope axis (local +y)."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ey = torch.tensor([0.0, 1.0, 0.0], device=self.env.device).expand(n, 3)
        return quat_apply(self.ramp.data.root_quat_w, ey)

    def _settled(self, body) -> torch.Tensor:
        c = self.cfg
        return (body.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def _status(self) -> dict[str, torch.Tensor]:
        """Live geometric predicates for the chain (all (N,) or (N,2))."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        cross = self._cross_axis_w()
        cos_c = math.cos(math.radians(c.chock_axis_max_deg))

        # chock: resting flat on the deck, cross-slope, in the zone
        cu, cv, chh = self._slope_coords(self.chock.data.root_pos_w)
        ey = torch.tensor([0.0, 1.0, 0.0], device=dev).expand(n, 3)
        ch_ax = quat_apply(self.chock.data.root_quat_w, ey)
        ch_on = (chh - c.chock_h / 2).abs() < c.rest_tol
        ch_cross = (ch_ax * cross).sum(dim=-1).abs() > cos_c
        ch_zone = ch_on & ch_cross & self._settled(self.chock) & (cv.abs() < c.y_max) \
            & (cu > c.band_lo - c.zone_lo_slack) & (cu < c.band_hi)

        # balls: resting ON the deck (center one radius above the plane), in band,
        # settled, and DIRECTLY uphill-adjacent to the chock (its support)
        on_l, park_l, u_l = [], [], []
        for name in self.BALL_NAMES:
            b = self.balls[name]
            u, v, h = self._slope_coords(b.data.root_pos_w)
            b_on = (h - c.ball_r).abs() < c.rest_tol
            supported = ch_zone & (u > cu) & ((u - cu) < c.chock_gap_max) \
                & ((v - cv).abs() < c.chock_len / 2)
            b_park = b_on & self._settled(b) & supported \
                & (u > c.band_lo) & (u < c.band_hi) & (v.abs() < c.y_max)
            on_l.append(b_on & (u > 0.03) & (u < c.deck_len) & (v.abs() < c.y_max + 0.03))
            park_l.append(b_park)
            u_l.append(u)
        return {"ch_zone": ch_zone, "on": torch.stack(on_l, dim=1),
                "park": torch.stack(park_l, dim=1), "u": torch.stack(u_l, dim=1),
                "cu": cu}

    def _update_latches(self) -> None:
        s = self._status()
        self._chock_zone |= s["ch_zone"]
        self._on_deck |= s["on"]
        self._parked |= s["park"]

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: BOTH balls at rest on the deck inside the band, each directly
        uphill-adjacent to the chock; the chock at rest flat on the deck, cross-slope,
        in the band zone; everything settled and finite. All clauses are live physical
        outcomes — the equilibrium is maintained by contact alone."""
        self._update_latches()
        s = self._status()
        pos = torch.stack([b.data.root_pos_w for b in self.balls.values()]
                          + [self.chock.data.root_pos_w], dim=1)
        finite = torch.isfinite(pos).all(dim=-1).all(dim=-1)
        return s["park"].all(dim=1) & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15*chock-in-zone + 0.05*each ball ever on deck
        + 0.20*each ball ever parked (all latched; ~0 for doing nothing), capped
        at 0.65 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_chock * self._chock_zone.float()
                + c.w_deck * self._on_deck.float().sum(dim=1)
                + c.w_park * self._parked.float().sum(dim=1)).clamp(max=0.65)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="ramp_chock", robot="null"))
