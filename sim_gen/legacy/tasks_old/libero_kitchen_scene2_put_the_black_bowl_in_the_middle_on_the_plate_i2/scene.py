"""ServeIngredientScene — pour/pluck the ingredient ball out of its cup onto the serving dish.

Derived from libero_90 kitchen_scene2 "put the black bowl in the middle on the plate", but the
PLAN is inverted: the deliverable is the CONTENTS, not the container. Three identical dark cups
stand in a row; exactly ONE (sampled per episode) holds a bright ingredient ball. A shallow
serving dish with a low rim sits on the other side. Goal: the ball alone rests on the dish
floor inside the rim, with EVERY cup kept off/away from the dish. The seed's own strategy —
carry the loaded container onto the target — is an explicit failure mode here (the dish must
hold the ball alone, and a ball still sitting in its cup on the dish fails the on-floor gate).

Rubric (graded 0..1, latched partial progress):
  0.00  nothing happened (ball still in its source cup)
  0.25  the ball has LEFT its source cup at least once (latched in post_step — pour started /
        ball plucked; survives a ball that ends up on the table)
  0.60  ball currently on the dish floor inside the rim (dish upright), not yet settled
  0.85  ball on the dish and settled, but some cup is still over/on the dish (set-aside pending)
  1.00  success(): ball settled on the dish floor, dish upright, every cup clear of the dish

All rubric geometry is judged in the DISH BODY FRAME (a nudged/rotated dish judges the same),
and the on-floor gate is honest by construction: `on_xy_tol` (66 mm) is just above the deepest
corner-nestle a 15 mm ball can physically reach inside the octagonal rim (~65 mm), while a ball
balanced ON the rim (~79 mm, +z offset) or a ball riding inside a cup placed on the dish
(+8 mm z offset from the cup floor) both fail.

Assets are fully procedural, one rigid body each (the pen_holder compound-spawner pattern):
  - vessel spawner (shared by cups and dish): bottom disc + 8 box wall segments forming an
    open octagonal cup; cups are deep and narrow, the dish is wide and low-rimmed.
  - ball: a plain sphere (isaaclab SphereCfg), zero restitution + damping so it settles.

Per-episode randomization: dish pose (xy jitter + yaw), cup row poses (per-cup xy jitter +
yaw), and WHICH cup holds the ball (uniform over the three) — selection is by content, not by
position, so a memorized "grab the middle one" fails.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
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


# ----- custom compound spawner ------------------------------------------------------------------
# One rigid body per vessel: root Xform with RigidBodyAPI + explicit MassAPI (overlapping wall
# segments would double-count density), a bottom cylinder collider, and 8 box wall segments whose
# inner aperture is a regular octagon of inradius `inner_r`. Same recipe as the pen_holder
# exemplar (idempotent authoring; clone() handles per-env replication, so no duplicate xformOps).

_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_vessel(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one open octagonal vessel at `prim_path` (cup or dish, same geometry family)."""
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
    # Cap the contact-solver pop: a ball landing on the dish floor penetrates a little in one
    # 120 Hz step and the default 3 m/s depenetration would eject it (the pen_holder lesson).
    PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateMaxDepenetrationVelocityAttr(0.5)

    color = Gf.Vec3f(*cfg.color)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    outer_r = cfg.inner_r + cfg.wall_t
    bot = UsdGeom.Cylinder.Define(stage, f"{prim_path}/bottom")
    bot.CreateRadiusAttr(outer_r)
    bot.CreateHeightAttr(cfg.bot_t)
    bot.CreateExtentAttr([Gf.Vec3f(-outer_r, -outer_r, -cfg.bot_t / 2),
                          Gf.Vec3f(outer_r, outer_r, cfg.bot_t / 2)])
    UsdGeom.Xformable(bot.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, -cfg.height / 2 + cfg.bot_t / 2))
    bot.CreateDisplayColorAttr([color])
    collide(bot.GetPrim())

    n = cfg.n_segments
    r_mid = cfg.inner_r + cfg.wall_t / 2
    seg_len = 2 * (cfg.inner_r + cfg.wall_t) * math.tan(math.pi / n) + 0.002
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang), 0.0))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(cfg.wall_t, seg_len, cfg.height))
        seg.CreateDisplayColorAttr([color])
        collide(seg.GetPrim())
    return root


def _vessel_spawner_cfg(*, inner_r: float, wall_t: float, height: float, bot_t: float,
                        mass: float, color: tuple, n_segments: int,
                        contact_offset: float) -> Any:
    """Build (lazily, app required) the vessel spawner cfg — `clone` wraps `_spawn_vessel`
    exactly like `spawn_cuboid` is wrapped."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "vessel" not in _SPAWNER_CACHE:

        @configclass
        class VesselSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_vessel)
            inner_r: float = 0.03  # inner octagon INRADIUS (m)
            wall_t: float = 0.006
            height: float = 0.06
            bot_t: float = 0.008
            color: tuple = (0.2, 0.2, 0.2)
            n_segments: int = 8
            contact_offset: float = 0.002

        _SPAWNER_CACHE["vessel"] = VesselSpawnerCfg

    return _SPAWNER_CACHE["vessel"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=inner_r, wall_t=wall_t, height=height, bot_t=bot_t,
        color=color, n_segments=n_segments, contact_offset=contact_offset,
    )


# ----- scene cfg ----------------------------------------------------------------------------------
@dataclass
class ServeIngredientSceneCfg(BaseCfg):
    """Config for `ServeIngredientScene`. Tolerances stay SOFT (easy tier); harden for
    curriculum variants."""

    # --- tunable: rubric thresholds -----------------------------------------------------------
    # Honesty limit: deepest physical corner-nestle of the ball inside the octagonal rim is
    # circumradius - r/sin(67.5 deg) = inner_r/cos(pi/8) - 16.2 mm ~= 65 mm < 66 mm tol; a ball
    # balanced ON the 8 mm rim sits at ~79 mm and is rejected by BOTH the xy and the z gate.
    on_xy_tol: float = tunable(0.066)  # ball centre within this of the dish axis (dish frame)
    on_z_tol: float = tunable(0.006)  # |ball centre z - (dish floor + ball_r)| below this:
    # "resting ON the dish floor". A ball riding inside a cup placed on the dish sits cup_bot_t
    # (8 mm) higher and fails — the anti-seed-strategy gate.
    dish_tilt_max_deg: float = tunable(15.0)  # dish axis within this of world-up when judging
    clear_margin: float = tunable(0.005)  # cup centre must be > dish_outer_r + this from the
    # dish axis (horizontal) or the dish is "not cleared" — the ball must be served ALONE
    settle_speed: float = tunable(0.05)  # max |v| (ball AND dish) when judging (m/s)

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    reset_pos_jitter: float = tunable(0.03)  # uniform +/- xy jitter (dish AND cups) at reset
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per body at reset
    randomize_src: bool = tunable(True)  # sample WHICH cup holds the ball (False -> middle)
    ball_jitter: float = tunable(0.004)  # ball xy jitter inside its cup at reset

    # --- tunable: placement (robot embodiments raise the work onto a bench) --------------------
    surface_z: float = tunable(0.0)  # work-surface height; 0 = on the ground (null smoke)
    dish_pos: tuple = tunable((0.24, 0.0))  # dish centre on the surface
    cups_center: tuple = tunable((-0.14, 0.0))  # centre of the cup row
    cup_spacing: float = tunable(0.17)  # row pitch along y

    # --- info: structure ------------------------------------------------------------------------
    bench_size: tuple = info((1.1, 0.9))  # procedural bench top (x, y), used when surface_z > 0
    n_cups: int = info(3)
    cup_inner_r: float = info(0.030)  # ball (r 15) has 15 mm of pluck clearance
    cup_wall_t: float = info(0.006)
    cup_h: float = info(0.058)  # ball top sits ~20 mm below the rim: visible from above only
    cup_bot_t: float = info(0.008)
    cup_mass: float = info(0.08)
    cup_color: tuple = info((0.16, 0.16, 0.19))  # all three identical — identity is the CONTENT
    dish_inner_r: float = info(0.075)  # capture funnel = inner_r - ball_r = 60 mm
    dish_wall_t: float = info(0.008)
    dish_h: float = info(0.032)  # rim stands height - bot_t = 20 mm above the floor > ball_r:
    # a served ball cannot roll off a level dish
    # Floor sized against tunneling (no CCD on GPU): the oracle pours from ~55 mm above the rim
    # -> ~1.1 m/s = 9 mm/step at 120 Hz; 12 mm floor + 3 mm contact offset captures it.
    dish_bot_t: float = info(0.012)
    dish_mass: float = info(0.30)
    dish_color: tuple = info((0.93, 0.92, 0.86))
    ball_r: float = info(0.015)
    ball_mass: float = info(0.03)
    ball_color: tuple = info((0.95, 0.45, 0.10))
    n_segments: int = info(8)
    cup_contact_offset: float = info(0.002)
    dish_contact_offset: float = info(0.003)

    # Derived (filled in __post_init__).
    dish_outer_r: float = field(default=None, init=False)
    cup_outer_r: float = field(default=None, init=False)
    dish_floor_local_z: float = field(default=None, init=False)  # dish floor top, dish frame
    cup_floor_local_z: float = field(default=None, init=False)  # cup floor top, cup frame

    def __post_init__(self) -> None:
        self.dish_outer_r = round(self.dish_inner_r + self.dish_wall_t, 4)
        self.cup_outer_r = round(self.cup_inner_r + self.cup_wall_t, 4)
        self.dish_floor_local_z = round(-self.dish_h / 2 + self.dish_bot_t, 4)
        self.cup_floor_local_z = round(-self.cup_h / 2 + self.cup_bot_t, 4)


# ----- scene ---------------------------------------------------------------------------------------
@SCENES.register("serve_ingredient")
class ServeIngredientScene(BaseScene):
    cfg: ServeIngredientSceneCfg

    def __init__(self, cfg: ServeIngredientSceneCfg | None = None) -> None:
        super().__init__(cfg or ServeIngredientSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, optional bench, the dish, three cups in a row, and the ball (nominal
        poses only — reset() re-places everything and samples the loaded cup)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z

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
        }
        if z0 > 0:  # procedural workbench (crate pattern): kinematic slab, top at surface_z
            out["bench"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bench_size[0], c.bench_size[1], z0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z0 / 2)),
            )

        out["dish"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Dish",
            spawn=_vessel_spawner_cfg(
                inner_r=c.dish_inner_r, wall_t=c.dish_wall_t, height=c.dish_h,
                bot_t=c.dish_bot_t, mass=c.dish_mass, color=c.dish_color,
                n_segments=c.n_segments, contact_offset=c.dish_contact_offset,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.dish_pos[0], c.dish_pos[1], z0 + c.dish_h / 2 + 0.002)),
        )

        for i in range(c.n_cups):
            y = c.cups_center[1] + (i - (c.n_cups - 1) / 2) * c.cup_spacing
            out[f"cup_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cup_" + str(i),
                spawn=_vessel_spawner_cfg(
                    inner_r=c.cup_inner_r, wall_t=c.cup_wall_t, height=c.cup_h,
                    bot_t=c.cup_bot_t, mass=c.cup_mass, color=c.cup_color,
                    n_segments=c.n_segments, contact_offset=c.cup_contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cups_center[0], y, z0 + c.cup_h / 2 + 0.002)),
            )

        out["ball"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Ball",
            spawn=sim_utils.SphereCfg(
                radius=c.ball_r,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5,
                    linear_damping=0.05, angular_damping=0.2),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.003, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ball_color),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.cups_center[0], c.cups_center[1],
                     z0 + 0.002 + c.cup_bot_t + c.ball_r + 0.002)),
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

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the source-cup index and the escape latch."""
        super().bind(env)
        c = self.cfg
        self.dish: RigidObject = env.iscene["dish"]
        self.ball: RigidObject = env.iscene["ball"]
        self.cups: list[RigidObject] = [env.iscene[f"cup_{i}"] for i in range(c.n_cups)]
        self.env_origins = env.iscene.env_origins
        # src[e]: index of the cup that holds the ball in episode e (sampled at reset).
        self.src = torch.full((env.num_envs,), 1, dtype=torch.long, device=env.device)
        # escaped[e]: latched true once the ball has left its source cup (partial progress).
        self.escaped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the dish and the cup row with jitter + yaw, sample the source
        cup, seat the ball inside it, clear the escape latch."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        yaw_amp = math.radians(c.reset_yaw_deg)

        def yawed(st: torch.Tensor) -> None:
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)

        # --- dish ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.dish_pos[0]
        st[:, 1] = c.dish_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
        st[:, 2] = c.surface_z + c.dish_h / 2 + 0.002
        yawed(st)
        st[:, 0:3] += origin
        self.dish.write_root_state_to_sim(st, env_ids)

        # --- cups (row along y) ---
        cup_xy = torch.zeros(m, c.n_cups, 2, device=dev)
        for i, cup in enumerate(self.cups):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.cups_center[0]
            st[:, 1] = c.cups_center[1] + (i - (c.n_cups - 1) / 2) * c.cup_spacing
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            st[:, 2] = c.surface_z + c.cup_h / 2 + 0.002
            yawed(st)
            cup_xy[:, i] = st[:, :2]
            st[:, 0:3] += origin
            cup.write_root_state_to_sim(st, env_ids)

        # --- source cup + ball seated inside it ---
        if c.randomize_src:
            self.src[env_ids] = torch.randint(0, c.n_cups, (m,), device=dev)
        else:
            self.src[env_ids] = c.n_cups // 2
        sxy = torch.gather(cup_xy, 1,
                           self.src[env_ids].view(m, 1, 1).expand(m, 1, 2)).squeeze(1)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = sxy + (torch.rand(m, 2, device=dev) * 2 - 1) * c.ball_jitter
        st[:, 2] = c.surface_z + 0.002 + c.cup_bot_t + c.ball_r + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.ball.write_root_state_to_sim(st, env_ids)

        self.escaped[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch the escape achievement at sim rate: once the ball is not inside its source
        cup (judged in the cup's body frame, so a carried/tilted cup still contains it), the
        pour/pluck has begun and the 0.25 partial credit survives any later mishap."""
        self.escaped |= ~self._ball_in_source_cup()

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "dish": self.dish.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "cups": [cup.data.root_state_w[env_ids].clone() for cup in self.cups],
            "src": self.src[env_ids].clone(),
            "escaped": self.escaped[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.dish.write_root_state_to_sim(state["dish"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        for cup, st in zip(self.cups, state["cups"]):
            cup.write_root_state_to_sim(st, env_ids)
        self.src[env_ids] = state["src"]
        self.escaped[env_ids] = state["escaped"]

    # ----- description -----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        where = "on the ground" if c.surface_z <= 0 else "on a workbench"
        return (
            f"Three identical dark cups (~{2 * c.cup_outer_r * 1000:.0f} mm wide, "
            f"{c.cup_h * 1000:.0f} mm tall) stand in a row {where}; exactly ONE of them holds "
            f"a bright orange ingredient ball ({2 * c.ball_r * 1000:.0f} mm), visible only from "
            f"above — which cup is loaded changes every episode, so look before you act. On the "
            f"other side sits a shallow serving dish (~{2 * c.dish_outer_r * 1000:.0f} mm wide) "
            f"with a low rim.\n"
            f"Goal: serve the ingredient — the ball must end up resting directly on the dish "
            f"floor, inside the rim, and every cup must be kept off and away from the dish. "
            f"Pour the loaded cup out over the dish or pluck the ball from it, your choice, "
            f"then put the cup back down away from the dish. Carrying the loaded cup onto the "
            f"dish does NOT count: the dish must hold the ball alone."
        )

    # ----- progress / rubric ------------------------------------------------------------------------
    def _ball_in_source_cup(self) -> torch.Tensor:
        """(N,) bool: ball inside its source cup's cavity (cup body frame, generous margins —
        used only for the escape latch, so false-negatives are the dangerous direction)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        n = self.env.num_envs
        pos = torch.stack([cup.data.root_pos_w for cup in self.cups], dim=1)  # (N,C,3)
        quat = torch.stack([cup.data.root_quat_w for cup in self.cups], dim=1)  # (N,C,4)
        sp = torch.gather(pos, 1, self.src.view(n, 1, 1).expand(n, 1, 3)).squeeze(1)
        sq = torch.gather(quat, 1, self.src.view(n, 1, 1).expand(n, 1, 4)).squeeze(1)
        b_loc = quat_apply_inverse(sq, self.ball.data.root_pos_w - sp)
        in_xy = b_loc[:, :2].norm(dim=-1) < c.cup_inner_r + 0.005
        in_z = (b_loc[:, 2] > c.cup_floor_local_z - 0.005) & \
               (b_loc[:, 2] < c.cup_h / 2 + c.ball_r)
        return in_xy & in_z

    def dish_up(self) -> torch.Tensor:
        """(N,) bool: dish axis within `dish_tilt_max_deg` of world-up."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        up = quat_apply(self.dish.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.dish_tilt_max_deg))

    def ball_on_dish(self) -> torch.Tensor:
        """(N,) bool, geometric, in the DISH BODY FRAME: ball centre within `on_xy_tol` of the
        dish axis AND resting ON the dish floor (centre at floor + ball_r within `on_z_tol` —
        a ball riding inside a cup on the dish sits ~8 mm too high and fails), dish upright."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        b_loc = quat_apply_inverse(self.dish.data.root_quat_w,
                                   self.ball.data.root_pos_w - self.dish.data.root_pos_w)
        near_axis = b_loc[:, :2].norm(dim=-1) < c.on_xy_tol
        rest_z = c.dish_floor_local_z + c.ball_r
        on_floor = (b_loc[:, 2] - rest_z).abs() < c.on_z_tol
        return near_axis & on_floor & self.dish_up()

    def dish_clear(self) -> torch.Tensor:
        """(N,) bool: EVERY cup's centre is horizontally clear of the dish footprint — the dish
        holds the ball alone. A cup parked on (or hovering over) the dish blocks this."""
        c = self.cfg
        dxy = self.dish.data.root_pos_w[:, :2]
        clear = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for cup in self.cups:
            d = (cup.data.root_pos_w[:, :2] - dxy).norm(dim=-1)
            clear &= d > c.dish_outer_r + c.clear_margin
        return clear

    def settled(self) -> torch.Tensor:
        """(N,) bool: ball AND dish |lin vel| below `settle_speed`."""
        c = self.cfg
        return (self.ball.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) & \
               (self.dish.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)

    def success(self) -> torch.Tensor:
        """(N,) bool: ball settled on the dish floor inside the rim, dish upright, every cup
        clear of the dish (scene-level success; the NullRobot oracle's target)."""
        return self.ball_on_dish() & self.settled() & self.dish_clear()

    def score(self) -> torch.Tensor:
        """(N,) float 0..1 — see the module docstring rubric. Monotone along the intended
        solve: 0 -> 0.25 (pour/pluck started, latched) -> 0.60 (ball in the dish, moving) ->
        0.85 (ball served, cup still over the dish) -> 1.0 (dish cleared: success)."""
        s = torch.zeros(self.env.num_envs, device=self.env.device)
        s = torch.where(self.escaped, 0.25, s)
        on = self.ball_on_dish()
        s = torch.where(on, 0.60, s)
        s = torch.where(on & self.settled(), 0.85, s)
        s = torch.where(self.success(), 1.0, s)
        return s


# Scene-level task: robot bindings are a later stage.
register_env("simgen", lambda: EnvCfg(scene="serve_ingredient", robot="null"))
