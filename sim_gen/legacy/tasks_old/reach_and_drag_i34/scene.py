"""CompassCrateScene — aim a marked crate at a beacon WITHOUT displacing it (sim_gen task
`reach_and_drag_i34`, derived from rlbench/reach_and_drag).

The seed grabs a stick and uses it to DRAG a cube across the table onto a distant target
zone: the whole task is transporting an object to an absolute goal region. This task is
the strategic inverse: the crate must NOT be transported. A tall beacon post stands
off to one side at a random bearing, and the crate carries a bright badge on ONE face;
the goal is to rotate the crate IN PLACE until the badge face points at the beacon.
Translation is the failure mode, not the goal: a permanent `strayed` latch trips the
moment the crate center leaves an 18 cm radius around its spawn, capping the score near
zero forever — so the seed's own plan ("drag the object to the landmark") is the tested
failing control. The required skill is controlled reorientation (pivoting / regrasping /
nudging about a vertical axis), not dragging to a zone.

Judged on PHYSICAL outcomes:
  - success() : the crate rests upright ON the ground at its home spot (center within
    `stay_radius` of spawn), settled, badge-face bearing within `yaw_tol_deg` of the
    beacon, and the stray latch never fired this episode.
  - score()   : 0 for doing nothing (initial aiming error is sampled >= 100 deg, credit
    starts below 90 deg); partial credit 0.8 * (90 deg - best_err) / (90 - 15) deg using
    the LATCHED best aiming error achieved while upright, in place and slow (a wild spin
    flying past the right heading does not latch); exactly 1.0 iff success; capped at
    0.05 forever once `strayed` latches — returning home afterwards does not restore it.

Per-episode randomization: crate spawn pose (xy + yaw), beacon bearing AND distance
(initial aiming error sampled in [100, 175] deg, random sign), distractor stick pose —
so a memorized rotation fails; the solver must read the beacon bearing from the scene.

Assets are fully procedural: the crate is a compound spawner (box collider + visual-only
badge plate and top stripe, the pen-tip pattern), the beacon a kinematic cylinder
re-posed per reset, plus the seed's stick as a free distractor / pushing tool. Heavy
imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- crate compound spawner ------------------------------------------------------------------
# One rigid body: a cube collider plus VISUAL-ONLY badge decorations (no CollisionAPI — the
# pen-holder cone-tip pattern), so "which face is marked" is visible but never touches physics.

_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_crate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the crate at `prim_path`: root Xform with RigidBodyAPI + explicit MassAPI, a
    cube collider, a badge plate on the +x face and a stripe on the +x half of the top face
    (both visual-only). Crate local frame: badge face normal = +x."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(1.0)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)

    s = float(cfg.size)
    t = float(cfg.badge_t)

    body = UsdGeom.Cube.Define(stage, f"{prim_path}/body")
    body.CreateSizeAttr(1.0)
    bxf = UsdGeom.Xformable(body.GetPrim())
    bxf.AddScaleOp().Set(Gf.Vec3f(s, s, s))
    body.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(body.GetPrim())
    px.CreateContactOffsetAttr(float(cfg.contact_offset))
    px.CreateRestOffsetAttr(0.0)

    badge = UsdGeom.Cube.Define(stage, f"{prim_path}/badge")  # visual only — NO CollisionAPI
    badge.CreateSizeAttr(1.0)
    gxf = UsdGeom.Xformable(badge.GetPrim())
    gxf.AddTranslateOp().Set(Gf.Vec3d(s / 2 + t / 2 + 0.0005, 0.0, 0.0))
    gxf.AddScaleOp().Set(Gf.Vec3f(t, 0.8 * s, 0.8 * s))
    badge.CreateDisplayColorAttr([Gf.Vec3f(*cfg.badge_color)])

    top = UsdGeom.Cube.Define(stage, f"{prim_path}/topmark")  # visual only — NO CollisionAPI
    top.CreateSizeAttr(1.0)
    txf = UsdGeom.Xformable(top.GetPrim())
    txf.AddTranslateOp().Set(Gf.Vec3d(s / 4, 0.0, s / 2 + t / 2 + 0.0005))
    txf.AddScaleOp().Set(Gf.Vec3f(0.45 * s, 0.25 * s, t))
    top.CreateDisplayColorAttr([Gf.Vec3f(*cfg.badge_color)])
    return root


def _crate_spawner_cfg(*, size: float, badge_t: float, mass: float, color: tuple,
                       badge_color: tuple, contact_offset: float) -> Any:
    """Build (lazily, app required) the crate spawner cfg — `clone` wraps `_spawn_crate`
    exactly like `spawn_cuboid` is wrapped."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "crate" not in _SPAWNER_CACHE:

        @configclass
        class CompassCrateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_crate)
            size: float = 0.12
            badge_t: float = 0.002
            color: tuple = (0.45, 0.48, 0.55)
            badge_color: tuple = (0.95, 0.15, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["crate"] = CompassCrateSpawnerCfg

    return _SPAWNER_CACHE["crate"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        size=size, badge_t=badge_t, color=color, badge_color=badge_color,
        contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CompassCrateSceneCfg(BaseCfg):
    """Config for `CompassCrateScene`. Env-local frame: crate spawns near the origin, the
    beacon stands `beacon_dist_range` away at a random bearing."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    yaw_tol_deg: float = tunable(15.0)      # badge bearing within this of the beacon = aligned
    upright_tol_deg: float = tunable(10.0)  # crate top face within this of world-up
    stay_radius: float = tunable(0.10)      # "in place": center within this of spawn (m)
    stray_radius: float = tunable(0.18)     # permanent stray latch beyond this (m)
    grounded_z_tol: float = tunable(0.02)   # center height tolerance for resting on the ground
    settle_lin: float = tunable(0.04)       # max |lin vel| when judging settled (m/s)
    settle_ang: float = tunable(0.40)       # max |ang vel| when judging settled (rad/s)
    latch_lin_max: float = tunable(0.50)    # best-err latch gate: crate not being carried fast
    latch_ang_max: float = tunable(1.50)    # best-err latch gate: a wild spin does not latch
    progress_anchor_deg: float = tunable(90.0)  # aiming error where partial credit starts

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    spawn_jitter: float = tunable(0.05)     # uniform +/- xy jitter of the crate spawn (m)
    spawn_yaw_deg: float = tunable(180.0)   # uniform +/- crate yaw at reset
    init_err_deg_range: tuple = tunable((100.0, 175.0))  # |initial aiming error| band; the
    # lower bound sits ABOVE the 90 deg credit anchor so doing nothing scores exactly 0.
    beacon_dist_range: tuple = tunable((0.42, 0.58))     # beacon distance from the crate (m)
    stick_jitter: float = tunable(0.05)     # distractor stick xy jitter
    stick_yaw_deg: float = tunable(30.0)    # distractor stick yaw jitter

    # --- info: structure ---------------------------------------------------------------------
    crate_size: float = info(0.12)
    crate_mass: float = info(0.40)
    crate_color: tuple = info((0.45, 0.48, 0.55))
    badge_color: tuple = info((0.95, 0.15, 0.10))
    badge_t: float = info(0.002)
    beacon_r: float = info(0.035)
    beacon_h: float = info(0.32)
    beacon_color: tuple = info((1.0, 0.65, 0.05))
    stick_size: tuple = info((0.36, 0.016, 0.016))  # the seed's stick, kept as a free tool
    stick_mass: float = info(0.06)
    stick_pos: tuple = info((0.0, -0.38))
    contact_offset: float = info(0.002)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("compass_crate")
class CompassCrateScene(BaseScene):
    cfg: CompassCrateSceneCfg

    def __init__(self, cfg: CompassCrateSceneCfg | None = None) -> None:
        super().__init__(cfg or CompassCrateSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the badge crate, the kinematic beacon post (re-posed by reset)
        and the distractor stick."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)

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
            "crate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Crate",
                spawn=_crate_spawner_cfg(
                    size=c.crate_size, badge_t=c.badge_t, mass=c.crate_mass,
                    color=c.crate_color, badge_color=c.badge_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, c.crate_size / 2 + 0.002)),
            ),
            "beacon": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beacon",
                spawn=sim_utils.CylinderCfg(
                    radius=c.beacon_r, height=c.beacon_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.beacon_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, c.beacon_h / 2)),
            ),
            "stick": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stick",
                spawn=sim_utils.CuboidCfg(
                    size=c.stick_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=1.0,
                        linear_damping=0.05, angular_damping=0.05),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.stick_mass),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.40, 0.22)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stick_pos[0], c.stick_pos[1], c.stick_size[2] / 2 + 0.002)),
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
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the per-episode buffers the rubric depends on."""
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.crate: RigidObject = env.iscene["crate"]
        self.beacon: RigidObject = env.iscene["beacon"]
        self.stick: RigidObject = env.iscene["stick"]
        self.env_origins = env.iscene.env_origins
        self.start_xy = torch.zeros(n, 2, device=dev)    # crate spawn center (env-local)
        self.beacon_xy = torch.zeros(n, 2, device=dev)   # beacon center (env-local)
        self.init_err = torch.zeros(n, device=dev)       # |aiming error| at reset (rad)
        self.best_err = torch.full((n,), math.pi, device=dev)  # latched best aiming error
        self.strayed = torch.zeros(n, dtype=torch.bool, device=dev)

    def _crate_local(self) -> torch.Tensor:
        return self.crate.data.root_pos_w - self.env_origins

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample crate pose, place the beacon at a bearing that puts the
        initial aiming error in `init_err_deg_range` (random sign), scatter the stick,
        clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def u(lo: float, hi: float) -> torch.Tensor:
            return lo + (hi - lo) * torch.rand(m, device=dev)

        xy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.spawn_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.spawn_yaw_deg)
        sign = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        err0 = sign * torch.deg2rad(u(*c.init_err_deg_range))
        heading = yaw + err0
        dist = u(*c.beacon_dist_range)
        bx = xy[:, 0] + dist * torch.cos(heading)
        by = xy[:, 1] + dist * torch.sin(heading)

        # crate: upright at spawn, badge face (+x) at `yaw`
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = xy
        st[:, 2] = c.crate_size / 2 + 0.002
        st[:, 3] = torch.cos(yaw / 2)
        st[:, 6] = torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.crate.write_root_state_to_sim(st, env_ids)

        # beacon: kinematic — POSE write only
        st = torch.zeros(m, 7, device=dev)
        st[:, 0] = bx
        st[:, 1] = by
        st[:, 2] = c.beacon_h / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.beacon.write_root_pose_to_sim(st, env_ids)

        # distractor stick: off to the side, lying flat
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.stick_pos[0]
        st[:, 1] = c.stick_pos[1]
        st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.stick_jitter
        st[:, 2] = c.stick_size[2] / 2 + 0.002
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.stick_yaw_deg) / 2
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.stick.write_root_state_to_sim(st, env_ids)

        self.start_xy[env_ids] = xy
        self.beacon_xy[env_ids, 0] = bx
        self.beacon_xy[env_ids, 1] = by
        self.init_err[env_ids] = err0.abs()
        self.best_err[env_ids] = math.pi
        self.strayed[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Every physics substep: trip the permanent stray latch, and latch the best aiming
        error — but only while the crate is upright, in place, slow (not carried, not in a
        wild spin) and not already strayed."""
        c = self.cfg
        p = self._crate_local()
        disp = (p[:, :2] - self.start_xy).norm(dim=-1)
        self.strayed |= disp > c.stray_radius
        lin = self.crate.data.root_lin_vel_w.norm(dim=-1)
        ang = self.crate.data.root_ang_vel_w.norm(dim=-1)
        gate = (self.upright() & (disp <= c.stay_radius)
                & (lin < c.latch_lin_max) & (ang < c.latch_ang_max) & ~self.strayed)
        self.best_err = torch.where(
            gate, torch.minimum(self.best_err, self.face_err()), self.best_err)

    # ----- state (full, restorable) ----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "crate": self.crate.data.root_state_w[env_ids].clone(),
            "beacon": self.beacon.data.root_state_w[env_ids].clone(),
            "stick": self.stick.data.root_state_w[env_ids].clone(),
            "start_xy": self.start_xy[env_ids].clone(),
            "beacon_xy": self.beacon_xy[env_ids].clone(),
            "init_err": self.init_err[env_ids].clone(),
            "best_err": self.best_err[env_ids].clone(),
            "strayed": self.strayed[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.crate.write_root_state_to_sim(state["crate"], env_ids)
        self.beacon.write_root_pose_to_sim(state["beacon"][:, 0:7], env_ids)
        self.stick.write_root_state_to_sim(state["stick"], env_ids)
        self.start_xy[env_ids] = state["start_xy"]
        self.beacon_xy[env_ids] = state["beacon_xy"]
        self.init_err[env_ids] = state["init_err"]
        self.best_err[env_ids] = state["best_err"]
        self.strayed[env_ids] = state["strayed"]

    # ----- description -----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A gray crate ({c.crate_size * 1000:.0f} mm) rests on the ground with a bright "
            f"red badge on ONE of its side faces. A tall orange beacon post stands about "
            f"half a meter away — its direction changes every episode — and a wooden stick "
            f"lies nearby (a free tool, not required).\n"
            f"Goal: rotate the crate IN PLACE, about the vertical axis, until the badge face "
            f"points at the beacon (within {c.yaw_tol_deg:.0f} deg), and leave it resting "
            f"upright where it started. Do NOT relocate the crate: if its center ever moves "
            f"more than {c.stray_radius * 100:.0f} cm from where it spawned, the episode is "
            f"spoiled for good — bringing it back does not help. Pivot it, don't ship it."
        )

    # ----- progress / rubric ------------------------------------------------------------------
    def face_err(self) -> torch.Tensor:
        """(N,) current aiming error (rad): angle between the badge-face normal (body +x,
        projected to the ground plane) and the crate->beacon bearing."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(n, 3)
        f = quat_apply(self.crate.data.root_quat_w, ex)[:, :2]
        f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        d = self.beacon_xy - self._crate_local()[:, :2]
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return torch.acos((f * d).sum(dim=-1).clamp(-1.0, 1.0))

    def upright(self) -> torch.Tensor:
        """(N,) bool: crate top face within `upright_tol_deg` of world-up."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(self.crate.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.upright_tol_deg))

    def in_place(self) -> torch.Tensor:
        """(N,) bool: crate center within `stay_radius` of its spawn point."""
        p = self._crate_local()
        return (p[:, :2] - self.start_xy).norm(dim=-1) <= self.cfg.stay_radius

    def grounded(self) -> torch.Tensor:
        """(N,) bool: crate center at resting height (on the ground, not held aloft)."""
        c = self.cfg
        z = self._crate_local()[:, 2]
        return (z - c.crate_size / 2).abs() < c.grounded_z_tol

    def settled(self) -> torch.Tensor:
        c = self.cfg
        lin = self.crate.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        ang = self.crate.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang
        return lin & ang

    def aligned(self) -> torch.Tensor:
        """(N,) bool: current aiming error inside the tolerance cone."""
        return self.face_err() < math.radians(self.cfg.yaw_tol_deg)

    def success(self) -> torch.Tensor:
        """(N,) bool: upright, at home, on the ground, settled, badge aimed at the beacon,
        and the crate never strayed this episode."""
        return (self.upright() & self.in_place() & self.grounded() & self.settled()
                & self.aligned() & ~self.strayed)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: partial credit from the LATCHED best aiming error, linear
        from 0 at `progress_anchor_deg` (90) down to 0.8 at `yaw_tol_deg` (15); exactly 1.0
        iff success; capped at 0.05 forever once the crate strayed. Doing nothing scores 0
        because the initial error is sampled at >= 100 deg, above the anchor."""
        c = self.cfg
        tol = math.radians(c.yaw_tol_deg)
        anchor = math.radians(c.progress_anchor_deg)
        base = 0.8 * ((anchor - self.best_err) / (anchor - tol)).clamp(0.0, 1.0)
        s = torch.where(self.success(), torch.ones_like(base), base)
        return torch.where(self.strayed, s.clamp(max=0.05), s)


# ----- env registration ------------------------------------------------------------------------
register_env("simgen", lambda: EnvCfg(scene="compass_crate", robot="null", env_spacing=4.0))
