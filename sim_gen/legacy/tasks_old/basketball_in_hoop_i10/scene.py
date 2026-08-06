"""TeeUpScene — un-basket a bouncy ball and PERCH it on a narrow tee (seed inversion).

Derived from rlbench/basketball_in_hoop but strategically inverted. The seed rewards
BALLISTIC CONTAINMENT: grab a free ball, hover anywhere above a forgiving elevated hoop,
release — gravity funnels it in and the bbox detector fires. Here the relationship is
flipped on both ends:

  - the ball STARTS contained (resting at the bottom of a low open basket) and must be
    EXTRACTED first — the seed's end state is this task's start state;
  - the goal support is CONVEX, not concave-forgiving: a shallow octagonal cradle
    (rim only ~10 mm tall, aperture inradius 20 mm < ball radius 30 mm) on top of a
    slender post. The ball PERCHES on the rim edge circle, mostly exposed.
  - the ball is BOUNCY (restitution ~0.6, combine-mode max): the seed's own strategy —
    dump from height above the target — physically bounces/deflects off the tee. Only a
    low-energy, laterally-aligned release seats it. Gentleness and precision are the
    skill, not containment.

Two ordered stages (order forced by geometry — the ball cannot be on the tee while
still in the basket): (1) extract the ball from the basket, (2) seat it on the tee.

Rubric (graded, transients latched in post_step so a momentary achievement is kept):
  0.00  nothing / ball still in the basket (null policy)
  0.20  latched: ball extracted (clear of the basket footprint or lifted clear of rim)
  0.45  latched: ball brought over the tee cradle (aligned above the rim)
  1.00  iff success(): ball CURRENTLY seated on the cradle — center within
        `seat_xy_tol` of the tee axis, center height within `seat_z_tol` of the
        geometric perch height rim_top + sqrt(ball_r^2 - cradle_inner_r^2) — and
        settled (|v| < settle_speed). Success is a physical, current-state predicate:
        a ball that bounced off, rolled away, or rests at the tee base scores the
        latched partial credit only.

Assets are fully procedural, one rigid body each (compound spawners, the pen_holder
pattern): a KINEMATIC basket (bottom disc + 8 wall boxes), a KINEMATIC tee (base disc +
post cylinder + cradle disc + 8 rim boxes), and a dynamic bouncy ball (plain sphere with
a restitution material). Basket and tee are fixed furniture (the seed's hoop was an
XFORM too) but are re-POSED with real per-episode randomization: basket xy + yaw, tee
xy + yaw, ball offset inside the basket — verified by readback in the smoke.

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


# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_root(stage_path_prim, translation, orientation, kinematic: bool, mass: float):
    """Shared root-body authoring: xform ops (authored fresh on the newly defined prim, so
    cloning never sees a duplicate op), RigidBodyAPI (+kinematic flag), explicit mass."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    xform, prim_path = stage_path_prim
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateMaxDepenetrationVelocityAttr(1.0)
    return root


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _ring_of_walls(stage, prim_path: str, *, inner_r: float, wall_t: float, height: float,
                   z0: float, color, contact_offset: float, n: int = 8) -> None:
    """8 box segments forming an octagonal shell of inner inradius `inner_r`, spanning
    z in [z0, z0 + height] in the body frame (root at the BASE of the object)."""
    from pxr import Gf, UsdGeom

    r_mid = inner_r + wall_t / 2
    seg_len = 2 * (inner_r + wall_t) * math.tan(math.pi / n) + 0.002
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang),
                                          z0 + height / 2))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(wall_t, seg_len, height))
        seg.CreateDisplayColorAttr([color])
        _collide(seg.GetPrim(), contact_offset)


def _disc(stage, prim_path: str, *, radius: float, thick: float, z_center: float, color,
          contact_offset: float):
    from pxr import Gf, UsdGeom

    d = UsdGeom.Cylinder.Define(stage, prim_path)
    d.CreateRadiusAttr(radius)
    d.CreateHeightAttr(thick)
    d.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -thick / 2),
                        Gf.Vec3f(radius, radius, thick / 2)])
    UsdGeom.Xformable(d.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z_center))
    d.CreateDisplayColorAttr([color])
    _collide(d.GetPrim(), contact_offset)
    return d


def _spawn_basket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Kinematic open basket, root frame at the BASE center: bottom disc (z in
    [0, bot_t]) + 8 wall boxes (z in [0, height])."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = _apply_root((xform, prim_path), translation, orientation, True, 0.5)
    color = Gf.Vec3f(*cfg.color)
    _disc(stage, f"{prim_path}/bottom", radius=cfg.inner_r + cfg.wall_t, thick=cfg.bot_t,
          z_center=cfg.bot_t / 2, color=color, contact_offset=cfg.contact_offset)
    _ring_of_walls(stage, prim_path, inner_r=cfg.inner_r, wall_t=cfg.wall_t,
                   height=cfg.height, z0=0.0, color=color,
                   contact_offset=cfg.contact_offset)
    return root


def _spawn_tee(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Kinematic tee, root frame at the BASE center: wide base disc, slender post
    cylinder, cradle disc on top, 8 tiny rim boxes forming the shallow perch."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = _apply_root((xform, prim_path), translation, orientation, True, 0.5)
    color = Gf.Vec3f(*cfg.color)
    rim_color = Gf.Vec3f(*cfg.rim_color)
    # base disc: [0, base_t]
    _disc(stage, f"{prim_path}/base", radius=cfg.base_r, thick=cfg.base_t,
          z_center=cfg.base_t / 2, color=color, contact_offset=cfg.contact_offset)
    # post: [base_t, post_top]
    post_h = cfg.post_top - cfg.base_t
    _disc(stage, f"{prim_path}/post", radius=cfg.post_r, thick=post_h,
          z_center=cfg.base_t + post_h / 2, color=color, contact_offset=cfg.contact_offset)
    # cradle disc: [post_top, post_top + disc_t]
    _disc(stage, f"{prim_path}/cradle", radius=cfg.cradle_inner_r + cfg.cradle_wall_t,
          thick=cfg.disc_t, z_center=cfg.post_top + cfg.disc_t / 2, color=rim_color,
          contact_offset=cfg.contact_offset)
    # rim ring: [post_top + disc_t, rim_top]
    _ring_of_walls(stage, prim_path, inner_r=cfg.cradle_inner_r, wall_t=cfg.cradle_wall_t,
                   height=cfg.rim_h, z0=cfg.post_top + cfg.disc_t, color=rim_color,
                   contact_offset=cfg.contact_offset)
    return root


def _basket_spawner_cfg(c: TeeUpSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "basket" not in _SPAWNER_CACHE:

        @configclass
        class BasketSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_basket)
            inner_r: float = 0.075
            wall_t: float = 0.008
            height: float = 0.070
            bot_t: float = 0.008
            color: tuple = (0.55, 0.35, 0.15)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["basket"] = BasketSpawnerCfg

    return _SPAWNER_CACHE["basket"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        inner_r=c.basket_inner_r, wall_t=c.basket_wall_t, height=c.basket_h,
        bot_t=c.basket_bot_t, color=c.basket_color, contact_offset=c.contact_offset,
    )


def _tee_spawner_cfg(c: TeeUpSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tee" not in _SPAWNER_CACHE:

        @configclass
        class TeeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tee)
            base_r: float = 0.05
            base_t: float = 0.008
            post_r: float = 0.014
            post_top: float = 0.14
            disc_t: float = 0.008
            cradle_inner_r: float = 0.020
            cradle_wall_t: float = 0.006
            rim_h: float = 0.010
            color: tuple = (0.30, 0.30, 0.65)
            rim_color: tuple = (0.85, 0.75, 0.20)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["tee"] = TeeSpawnerCfg

    return _SPAWNER_CACHE["tee"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        base_r=c.tee_base_r, base_t=c.tee_base_t, post_r=c.post_r, post_top=c.tee_h,
        disc_t=c.cradle_disc_t, cradle_inner_r=c.cradle_inner_r,
        cradle_wall_t=c.cradle_wall_t, rim_h=c.cradle_rim_h, color=c.tee_color,
        rim_color=c.rim_color, contact_offset=c.contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TeeUpSceneCfg(BaseCfg):
    """Config for `TeeUpScene`. The seat tolerances are geometric (derived from ball +
    cradle radii in __post_init__); the tunables are the difficulty dials."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    seat_xy_tol: float = tunable(0.012)  # ball center within this of the tee axis when seated
    seat_z_tol: float = tunable(0.010)  # |z - geometric perch height| below this when seated
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging seated (m/s)

    # --- tunable: ball (the gentleness dial) -------------------------------------------------
    ball_r: float = tunable(0.030)  # ball radius (m); must exceed cradle_inner_r (perch)
    ball_mass: float = tunable(0.057)
    restitution: float = tunable(0.60)  # bounciness — what makes the seed's dump fail

    # --- tunable: randomization (task-family knobs) ------------------------------------------
    basket_pos: tuple = tunable((-0.18, 0.0))  # basket base center (xy)
    tee_pos: tuple = tunable((0.20, 0.0))  # tee base center (xy)
    reset_pos_jitter: float = tunable(0.06)  # uniform +/- xy jitter (basket AND tee) at reset
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw (basket AND tee, cosmetic)
    tee_h: float = tunable(0.14)  # post top height — build-time dial (geometry is spawned)

    # --- info: structure ---------------------------------------------------------------------
    basket_inner_r: float = info(0.075)  # basket aperture inradius; ball fits with margin
    basket_wall_t: float = info(0.008)
    basket_h: float = info(0.070)  # ball (d=60 mm) sits fully below the rim -> contained
    basket_bot_t: float = info(0.008)
    basket_color: tuple = info((0.55, 0.35, 0.15))
    tee_base_r: float = info(0.05)
    tee_base_t: float = info(0.008)
    post_r: float = info(0.014)
    cradle_disc_t: float = info(0.008)
    cradle_inner_r: float = info(0.020)  # rim aperture inradius < ball_r -> a PERCH, not a cup
    cradle_wall_t: float = info(0.006)
    cradle_rim_h: float = info(0.010)  # shallow: ~7.6 mm escape barrier -> bounces eject
    tee_color: tuple = info((0.30, 0.30, 0.65))
    rim_color: tuple = info((0.85, 0.75, 0.20))
    contact_offset: float = info(0.002)
    ball_color: tuple = info((0.90, 0.45, 0.10))

    # Derived (filled in __post_init__).
    basket_outer_r: float = field(default=None, init=False)
    rim_top_local: float = field(default=None, init=False)  # cradle rim top, tee body frame
    seat_z_local: float = field(default=None, init=False)  # perched ball CENTER, tee body frame

    def __post_init__(self) -> None:
        assert self.cradle_inner_r < self.ball_r, "cradle must be a perch (aperture < ball)"
        self.basket_outer_r = round(self.basket_inner_r + self.basket_wall_t, 4)
        self.rim_top_local = round(self.tee_h + self.cradle_disc_t + self.cradle_rim_h, 4)
        # perched on the rim edge circle of radius cradle_inner_r:
        self.seat_z_local = round(
            self.rim_top_local + math.sqrt(self.ball_r**2 - self.cradle_inner_r**2), 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("tee_up")
class TeeUpScene(BaseScene):
    cfg: TeeUpSceneCfg

    def __init__(self, cfg: TeeUpSceneCfg | None = None) -> None:
        super().__init__(cfg or TeeUpSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
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
            "basket": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Basket",
                spawn=_basket_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.basket_pos[0], c.basket_pos[1], 0.0005)),
            ),
            "tee": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tee",
                spawn=_tee_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tee_pos[0], c.tee_pos[1], 0.0005)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=0.05, angular_damping=0.15,
                        max_depenetration_velocity=1.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5,
                        restitution=c.restitution,
                        restitution_combine_mode="max"),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ball_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.basket_pos[0], c.basket_pos[1],
                         c.basket_bot_t + c.ball_r + 0.003)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        # bounce_threshold_velocity low enough that a careless dump genuinely bounces,
        # high enough that a slow seated ball stops chattering and settles.
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": 2**21,
                "gpu_max_rigid_patch_count": 2**21,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.basket: RigidObject = env.iscene["basket"]
        self.tee: RigidObject = env.iscene["tee"]
        self.ball: RigidObject = env.iscene["ball"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        self._latch_extracted = torch.zeros(n, dtype=torch.bool, device=env.device)
        self._latch_over_tee = torch.zeros(n, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: basket and tee re-posed with xy jitter + yaw (kinematic teleport,
        verified by readback in the smoke), ball placed at a random offset on the basket
        floor, latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        yaw_amp = math.radians(c.reset_yaw_deg)

        def posed(base_xy: tuple, jitter: torch.Tensor) -> torch.Tensor:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = base_xy[0]
            st[:, 1] = base_xy[1]
            st[:, :2] += jitter
            st[:, 2] = 0.0005
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            return st

        bj = (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
        tj = (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
        basket_st = posed(c.basket_pos, bj)
        tee_st = posed(c.tee_pos, tj)
        self.basket.write_root_state_to_sim(basket_st, env_ids)
        self.tee.write_root_state_to_sim(tee_st, env_ids)

        # ball: random offset on the basket floor (kept clear of the walls)
        r_max = c.basket_inner_r - c.ball_r - 0.008
        r = torch.sqrt(torch.rand(m, device=dev)) * r_max
        ang = torch.rand(m, device=dev) * 2 * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = basket_st[:, 0] + r * torch.cos(ang)
        st[:, 1] = basket_st[:, 1] + r * torch.sin(ang)
        st[:, 2] = origin[:, 2] + c.basket_bot_t + c.ball_r + 0.003
        st[:, 3] = 1.0
        self.ball.write_root_state_to_sim(st, env_ids)

        self._latch_extracted[env_ids] = False
        self._latch_over_tee[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch transient achievements at sim rate (data is fresh — env.step updates the
        scene buffers before calling this)."""
        self._latch_extracted |= self._extracted_now()
        self._latch_over_tee |= self._over_tee_now()

    # ----- state (full, restorable) ---------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "basket": self.basket.data.root_state_w[env_ids].clone(),
            "tee": self.tee.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "latch_extracted": self._latch_extracted[env_ids].clone(),
            "latch_over_tee": self._latch_over_tee[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.basket.write_root_state_to_sim(state["basket"], env_ids)
        self.tee.write_root_state_to_sim(state["tee"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        self._latch_extracted[env_ids] = state["latch_extracted"]
        self._latch_over_tee[env_ids] = state["latch_over_tee"]

    # ----- description ----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A bouncy orange ball ({2 * c.ball_r * 1000:.0f} mm across) rests at the bottom "
            f"of a low open basket ({2 * c.basket_outer_r * 1000:.0f} mm wide, "
            f"{c.basket_h * 1000:.0f} mm tall walls) standing on the ground. Nearby stands a "
            f"tee: a slender post ({c.tee_h * 1000:.0f} mm tall) carrying a shallow gold "
            f"cradle whose rim opening ({2 * c.cradle_inner_r * 1000:.0f} mm) is narrower "
            f"than the ball, so the ball can only PERCH on the rim, mostly exposed.\n"
            f"Goal: take the ball out of the basket and set it down on the tee so it rests "
            f"seated in the cradle. The ball is bouncy and the cradle is shallow: dropping "
            f"or tossing it onto the tee from height makes it bounce off — lower it gently "
            f"and centered. The task is done when the ball sits still on the cradle."
        )

    # ----- predicates / rubric --------------------------------------------------------------
    def _ball(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos_w (N,3), |lin_vel| (N,))"""
        return self.ball.data.root_pos_w, self.ball.data.root_lin_vel_w.norm(dim=-1)

    def _extracted_now(self) -> torch.Tensor:
        """(N,) bool: ball clear of the basket — outside its footprint, or lifted so its
        bottom clears the rim."""
        c = self.cfg
        bp, _ = self._ball()
        kp = self.basket.data.root_pos_w
        d_xy = (bp[:, :2] - kp[:, :2]).norm(dim=-1)
        outside = d_xy > c.basket_outer_r + c.ball_r + 0.005
        above = (bp[:, 2] - kp[:, 2]) > c.basket_h + c.ball_r
        return outside | above

    def _over_tee_now(self) -> torch.Tensor:
        """(N,) bool: ball aligned above the cradle (center within 50 mm of the tee axis,
        center above the cradle base)."""
        c = self.cfg
        bp, _ = self._ball()
        tp = self.tee.data.root_pos_w
        d_xy = (bp[:, :2] - tp[:, :2]).norm(dim=-1)
        return (d_xy < 0.05) & ((bp[:, 2] - tp[:, 2]) > c.tee_h)

    def seated(self) -> torch.Tensor:
        """(N,) bool, geometric: ball center on the tee axis within `seat_xy_tol`, at the
        perch height within `seat_z_tol`."""
        c = self.cfg
        bp, _ = self._ball()
        tp = self.tee.data.root_pos_w
        d_xy = (bp[:, :2] - tp[:, :2]).norm(dim=-1)
        dz = (bp[:, 2] - tp[:, 2]) - c.seat_z_local
        return (d_xy < c.seat_xy_tol) & (dz.abs() < c.seat_z_tol)

    def settled(self) -> torch.Tensor:
        _, v = self._ball()
        return v < self.cfg.settle_speed

    def success(self) -> torch.Tensor:
        """(N,) bool: ball CURRENTLY seated on the cradle and settled — a physical,
        present-state outcome (bounced-off or rolled-away balls do not count)."""
        return self.seated() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0 nothing; 0.2 latched extraction; 0.45 latched
        aligned-over-tee; 1.0 iff success()."""
        ext = self._latch_extracted | self._extracted_now()
        over = self._latch_over_tee | self._over_tee_now()
        s = torch.zeros(self.env.num_envs, device=self.env.device)
        s = torch.where(ext, torch.full_like(s, 0.20), s)
        s = torch.where(over, torch.full_like(s, 0.45), s)
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("sim_gen", lambda: EnvCfg(scene="tee_up", robot="null"))
