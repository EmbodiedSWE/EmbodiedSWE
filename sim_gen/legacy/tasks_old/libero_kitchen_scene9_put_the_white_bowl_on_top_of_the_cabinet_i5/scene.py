"""EmptyBowlScene — pour the balls OUT of the white bowl into the sink basin, then set the
empty bowl back down upright on the counter.

Derived from libero_90 kitchen_scene9 "put the white bowl on top of the cabinet", but
STRATEGICALLY INVERTED: the seed carries an empty bowl level and sets it on an elevated
surface — the goal is the bowl's pose and the payload is irrelevant. Here the bowl starts
FULL (2-4 loose balls) and the goal is judged on the CONTENTS: every ball must end up
inside a separate walled basin, and the bowl must end up back on the work surface,
upright and empty, OUTSIDE the basin. The core skill is a controlled REORIENTATION
(tilt/pour over the basin) — the one motion the seed's plan forbids (tilting the carried
bowl is exactly how you fail the seed). The seed's own strategy is present as a trap: a
"cabinet ledge" fixture stands in the scene, and carrying the loaded bowl onto it scores
almost nothing (the balls are still in the bowl).

Judging is final-state and strategy-agnostic under honest physics: a ball counts as
delivered only when it is settled inside the basin's inner volume AND NOT inside the
bowl (so parking the loaded bowl inside the basin delivers nothing — the container-cheat
control), below the rim (a ball perched on the rim does not count). Success additionally
requires the bowl standing upright on the surface outside the basin, settled. Rubric:
0.1 latched once the bowl has ever been lifted off the surface (transient-achievement
latch, updated in post_step), + 0.6 * fraction of present balls delivered, 0.9 when ALL
present balls are delivered, 1.0 iff success (all delivered + bowl set down). Ball-by-ball
transfer (never lifting the bowl) is an admissible alternative plan and also reaches 1.0.

Per-episode randomization: bowl pose (xy jitter + yaw), basin and ledge poses (xy jitter
+ yaw), AND ball-count subset sampling (2-4 present; absent balls park in an off-camera
ground depot), so success is judged on the sampled subset and a memorized fixed pour
count fails. All assets are procedural: the bowl is a compound rigid body (bottom disc +
8 octagon wall boxes — the pen_holder spawner pattern, child colliders of one body never
self-collide), the basin is a kinematic compound tub (floor + 4 walls), the balls are
plain spheres, the ledge a kinematic box.

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
# One rigid body per object, several child colliders, authored with raw pxr APIs; only
# `isaaclab.sim.utils.clone` is borrowed (the regex-resolve + per-env replicate machinery every
# CuboidCfg spawn uses). Fresh Define per prim -> xformOps authored once (idempotent under clone).

_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the open white bowl at `prim_path`: root Xform with RigidBodyAPI + explicit
    MassAPI, a bottom cylinder collider and 8 box wall segments forming an octagonal shell of
    inner inradius `inner_r`. Depenetration capped at 0.5 m/s (balls rattling in a moving bowl
    must shed energy, not pop) and lightly damped so the set-down settles promptly."""
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
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)

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

    # 8 wall boxes; segment length closes the OUTER octagon (overlap inside one body is
    # harmless); the aperture is a regular octagon of inradius `inner_r`.
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


def _spawn_basin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the sink basin at `prim_path`: a KINEMATIC rigid body (a fixture that can still
    be re-placed per episode by root-state writes), floor slab + 4 wall boxes around a square
    inner well. Body frame: walls span local z in [-wall_h/2, +wall_h/2] (rim at +wall_h/2),
    the floor slab hangs below the interior."""
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
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)

    color = Gf.Vec3f(*cfg.color)

    def box(name: str, size, center) -> None:
        b = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        b.CreateSizeAttr(1.0)
        bxf = UsdGeom.Xformable(b.GetPrim())
        bxf.AddTranslateOp().Set(Gf.Vec3d(*center))
        bxf.AddScaleOp().Set(Gf.Vec3f(*size))
        b.CreateDisplayColorAttr([color])
        UsdPhysics.CollisionAPI.Apply(b.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(b.GetPrim())
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    inner, t, h, ft = cfg.inner, cfg.wall_t, cfg.wall_h, cfg.floor_t
    box("floor", (inner + 2 * t, inner + 2 * t, ft), (0.0, 0.0, -h / 2 - ft / 2))
    box("wall_n", (inner + 2 * t, t, h), (0.0, inner / 2 + t / 2, 0.0))
    box("wall_s", (inner + 2 * t, t, h), (0.0, -inner / 2 - t / 2, 0.0))
    box("wall_e", (t, inner, h), (inner / 2 + t / 2, 0.0, 0.0))
    box("wall_w", (t, inner, h), (-inner / 2 - t / 2, 0.0, 0.0))
    return root


def _bowl_spawner_cfg(*, inner_r: float, wall_t: float, height: float, bot_t: float,
                      mass: float, color: tuple, n_segments: int, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bowl" not in _SPAWNER_CACHE:

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            inner_r: float = 0.058
            wall_t: float = 0.008
            height: float = 0.05
            bot_t: float = 0.012
            color: tuple = (0.92, 0.92, 0.90)
            n_segments: int = 8
            contact_offset: float = 0.003

        _SPAWNER_CACHE["bowl"] = BowlSpawnerCfg

    return _SPAWNER_CACHE["bowl"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=inner_r, wall_t=wall_t, height=height, bot_t=bot_t,
        color=color, n_segments=n_segments, contact_offset=contact_offset,
    )


def _basin_spawner_cfg(*, inner: float, wall_t: float, wall_h: float, floor_t: float,
                       color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "basin" not in _SPAWNER_CACHE:

        @configclass
        class BasinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_basin)
            inner: float = 0.17
            wall_t: float = 0.010
            wall_h: float = 0.090
            floor_t: float = 0.010
            color: tuple = (0.45, 0.50, 0.58)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["basin"] = BasinSpawnerCfg

    return _SPAWNER_CACHE["basin"](
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        inner=inner, wall_t=wall_t, wall_h=wall_h, floor_t=floor_t,
        color=color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class EmptyBowlSceneCfg(BaseCfg):
    """Config for `EmptyBowlScene`. Tolerances are geometric-honest: a ball physically inside
    the basin well always satisfies the delivered clauses; a ball on the rim or beside the
    tub never does."""

    # --- tunable: rubric thresholds -----------------------------------------------------------
    lift_height: float = tunable(0.06)  # bowl centre this far above rest -> lift latch (m)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging a ball / the bowl (m/s)
    rim_margin: float = tunable(0.5)  # ball counts only if centre below rim by this * ball_r
    placed_tilt_deg: float = tunable(15.0)  # bowl "upright" gate for the final set-down
    placed_z_tol: float = tunable(0.012)  # bowl bottom within this of the surface (m)

    # --- tunable: randomization (the task-family knobs) ---------------------------------------
    reset_pos_jitter: float = tunable(0.035)  # uniform +/- xy jitter (bowl, basin, ledge)
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per body at reset
    subset_sample: bool = tunable(True)  # per-episode ball-count sampling
    min_present: int = tunable(2)  # lower bound of sampled ball count

    # --- tunable: placement --------------------------------------------------------------------
    surface_z: float = tunable(0.0)  # work-surface height; 0 = on the ground (null smoke)
    bowl_pos: tuple = tunable((-0.10, 0.0))  # bowl centre on the surface
    basin_pos: tuple = tunable((0.28, 0.0))  # basin centre
    ledge_pos: tuple = tunable((0.02, -0.30))  # "cabinet ledge" trap fixture centre

    # --- info: bowl (compound rigid body) -------------------------------------------------------
    bowl_inner_r: float = info(0.058)  # inner octagon inradius; 4 balls fit on the floor
    bowl_wall_t: float = info(0.008)
    bowl_h: float = info(0.050)  # shallow — a bowl, not a cup: pours at a modest tilt
    bowl_bot_t: float = info(0.012)  # thick floor: no CCD on GPU PhysX — anti-tunneling
    bowl_mass: float = info(0.15)
    bowl_color: tuple = info((0.92, 0.92, 0.90))  # the seed's WHITE bowl
    n_segments: int = info(8)
    contact_offset: float = info(0.003)

    # --- info: balls -----------------------------------------------------------------------------
    n_balls: int = info(4)
    ball_r: float = info(0.016)
    ball_mass: float = info(0.03)
    ball_colors: tuple = info(((0.85, 0.20, 0.20), (0.95, 0.55, 0.10),
                               (0.90, 0.85, 0.15), (0.45, 0.20, 0.70)))

    # --- info: basin (kinematic tub) + ledge (the seed-strategy trap) ---------------------------
    basin_inner: float = info(0.17)  # inner square side — generous target for an easy tier
    basin_wall_t: float = info(0.010)
    basin_wall_h: float = info(0.090)  # tall enough to keep poured balls from bouncing out
    basin_floor_t: float = info(0.010)
    basin_color: tuple = info((0.45, 0.50, 0.58))
    ledge_size: tuple = info((0.16, 0.16, 0.12))  # the elevated "cabinet top" of the seed
    ledge_color: tuple = info((0.45, 0.30, 0.18))
    parking_pos: tuple = info((1.0, 1.0))  # off-camera ground depot for absent balls

    # Derived (filled in __post_init__).
    bowl_outer_r: float = field(default=None, init=False)
    bowl_floor_local_z: float = field(default=None, init=False)  # bowl floor top, body frame
    basin_rim_local_z: float = field(default=None, init=False)  # rim, basin body frame

    def __post_init__(self) -> None:
        self.bowl_outer_r = round(self.bowl_inner_r + self.bowl_wall_t, 4)
        self.bowl_floor_local_z = round(-self.bowl_h / 2 + self.bowl_bot_t, 4)
        self.basin_rim_local_z = round(self.basin_wall_h / 2, 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("empty_the_bowl")
class EmptyBowlScene(BaseScene):
    cfg: EmptyBowlSceneCfg

    def __init__(self, cfg: EmptyBowlSceneCfg | None = None) -> None:
        super().__init__(cfg or EmptyBowlSceneCfg())

    # ----- assets ------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the free white bowl (nominal pose; reset() re-places everything),
        the kinematic basin and ledge fixtures, and the balls (nominal: inside the bowl)."""
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

        out["bowl"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bowl",
            spawn=_bowl_spawner_cfg(
                inner_r=c.bowl_inner_r, wall_t=c.bowl_wall_t, height=c.bowl_h,
                bot_t=c.bowl_bot_t, mass=c.bowl_mass, color=c.bowl_color,
                n_segments=c.n_segments, contact_offset=c.contact_offset,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.bowl_pos[0], c.bowl_pos[1], z0 + c.bowl_h / 2 + 0.002)),
        )

        out["basin"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Basin",
            spawn=_basin_spawner_cfg(
                inner=c.basin_inner, wall_t=c.basin_wall_t, wall_h=c.basin_wall_h,
                floor_t=c.basin_floor_t, color=c.basin_color, contact_offset=0.002,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.basin_pos[0], c.basin_pos[1],
                     z0 + c.basin_floor_t + c.basin_wall_h / 2)),
        )

        out["ledge"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Ledge",
            spawn=sim_utils.CuboidCfg(
                size=c.ledge_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002,
                                                                 rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ledge_color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.ledge_pos[0], c.ledge_pos[1], z0 + c.ledge_size[2] / 2)),
        )

        for i in range(c.n_balls):
            out[f"ball_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball_" + str(i),
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=0.10, angular_damping=0.40,
                        max_depenetration_velocity=0.5),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.ball_colors[i]),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bowl_pos[0], c.bowl_pos[1] + 0.02 * i,
                         z0 + c.bowl_bot_t + c.ball_r + 0.006)),
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
        c = self.cfg
        self.bowl: RigidObject = env.iscene["bowl"]
        self.basin: RigidObject = env.iscene["basin"]
        self.ledge: RigidObject = env.iscene["ledge"]
        self.balls: dict[str, RigidObject] = {
            f"ball_{i}": env.iscene[f"ball_{i}"] for i in range(c.n_balls)}
        self.env_origins = env.iscene.env_origins
        # present[e, i]: ball i participates in episode e (sampled at reset; judged subset).
        self.present = torch.ones(env.num_envs, c.n_balls, dtype=torch.bool, device=env.device)
        # lifted[e]: the bowl has been raised off the surface at least once this episode —
        # the transient-achievement latch, updated in post_step, paying 0.1 in score().
        self.lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the present ball subset, place bowl / basin / ledge with xy
        jitter + yaw, seat present balls INSIDE the bowl (they settle onto its floor), park
        absent balls in the ground depot, clear the lift latch."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        yaw_amp = math.radians(c.reset_yaw_deg)

        self.lifted[env_ids] = False

        # --- subset sampling: k ~ U{min_present..n_balls} present balls ---
        if c.subset_sample:
            k = torch.randint(c.min_present, c.n_balls + 1, (m,), device=dev)
        else:
            k = torch.full((m,), c.n_balls, dtype=torch.long, device=dev)
        rank = torch.rand(m, c.n_balls, device=dev).argsort(dim=1).argsort(dim=1)
        self.present[env_ids] = rank < k.unsqueeze(1)

        def yawed_state(base_xy: tuple, z: float, jitter: float) -> tuple[torch.Tensor, torch.Tensor]:
            """(state (m,13), yaw (m,)) — pos = base + jitter, quat = yaw about z."""
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = base_xy[0]
            st[:, 1] = base_xy[1]
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * jitter
            st[:, 2] = z
            yaw = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            return st, yaw

        # --- fixtures (kinematic, still re-placed each episode) ---
        basin_st, _ = yawed_state(c.basin_pos, c.surface_z + c.basin_floor_t + c.basin_wall_h / 2,
                                  c.reset_pos_jitter)
        self.basin.write_root_state_to_sim(basin_st, env_ids)
        ledge_st, _ = yawed_state(c.ledge_pos, c.surface_z + c.ledge_size[2] / 2,
                                  c.reset_pos_jitter)
        self.ledge.write_root_state_to_sim(ledge_st, env_ids)

        # --- bowl + its contents ---
        bowl_st, bowl_yaw = yawed_state(c.bowl_pos, c.surface_z + c.bowl_h / 2 + 0.002,
                                        c.reset_pos_jitter)
        self.bowl.write_root_state_to_sim(bowl_st, env_ids)
        cy, sy = torch.cos(bowl_yaw), torch.sin(bowl_yaw)
        seat_r = 0.024  # ball slot circle inside the bowl (4 x r16 balls just fit)
        ball_z = c.surface_z + 0.002 + c.bowl_bot_t + c.ball_r + 0.004
        for i, ball in enumerate(self.balls.values()):
            ang = 2 * math.pi * i / c.n_balls + math.pi / 4
            lx, ly = seat_r * math.cos(ang), seat_r * math.sin(ang)
            seat = torch.zeros(m, 3, device=dev)
            seat[:, 0] = bowl_st[:, 0] - origin[:, 0] + (cy * lx - sy * ly)
            seat[:, 1] = bowl_st[:, 1] - origin[:, 1] + (sy * lx + cy * ly)
            seat[:, 2] = ball_z
            park = torch.zeros(m, 3, device=dev)
            park[:, 0] = c.parking_pos[0] + (i % 2) * 0.12
            park[:, 1] = c.parking_pos[1] + (i // 2) * 0.12
            park[:, 2] = c.ball_r + 0.003
            pres = self.present[env_ids, i].unsqueeze(1)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.where(pres, seat, park)
            st[:, 3] = 1.0
            ball.write_root_state_to_sim(st, env_ids)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch the lift achievement at sim rate: the bowl centre ever rising `lift_height`
        above its resting height marks 'the bowl left the surface'."""
        c = self.cfg
        z = self.bowl.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        self.lifted |= z > c.surface_z + c.bowl_h / 2 + c.lift_height

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bowl": self.bowl.data.root_state_w[env_ids].clone(),
            "basin": self.basin.data.root_state_w[env_ids].clone(),
            "ledge": self.ledge.data.root_state_w[env_ids].clone(),
            "balls": {n: b.data.root_state_w[env_ids].clone() for n, b in self.balls.items()},
            "present": self.present[env_ids].clone(),
            "lifted": self.lifted[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.bowl.write_root_state_to_sim(state["bowl"], env_ids)
        self.basin.write_root_state_to_sim(state["basin"], env_ids)
        self.ledge.write_root_state_to_sim(state["ledge"], env_ids)
        for n, b in self.balls.items():
            b.write_root_state_to_sim(state["balls"][n], env_ids)
        self.present[env_ids] = state["present"]
        self.lifted[env_ids] = state["lifted"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A white bowl (~{2 * c.bowl_outer_r * 1000:.0f} mm wide, {c.bowl_h * 1000:.0f} mm "
            f"tall) stands on the work surface holding between 2 and {c.n_balls} loose colored "
            f"balls ({2 * c.ball_r * 1000:.0f} mm) — count what you see. To one side sits an "
            f"open sink basin (inner {c.basin_inner * 1000:.0f} mm square, walls "
            f"{c.basin_wall_h * 1000:.0f} mm); to another, a raised wooden ledge.\n"
            f"Goal: empty the bowl — get EVERY ball into the basin (e.g. by pouring), then set "
            f"the empty bowl back down upright on the work surface, outside the basin. A ball "
            f"still in the bowl, perched on the basin rim, or loose on the table does not "
            f"count; carrying the loaded bowl somewhere else (like onto the ledge) achieves "
            f"nothing. Only the final settled state is judged."
        )

    # ----- predicates / rubric --------------------------------------------------------------------
    def _ball_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos_w (N,B,3), |lin_vel| (N,B)) for all balls, index order."""
        pos = torch.stack([b.data.root_pos_w for b in self.balls.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.balls.values()],
                          dim=1)
        return pos, vel

    def _local_to(self, body, points: torch.Tensor) -> torch.Tensor:
        """Express world points (N,B,3) in `body`'s frame -> (N,B,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        n, b = points.shape[0], points.shape[1]
        q = body.data.root_quat_w[:, None, :].expand(n, b, 4).reshape(n * b, 4)
        p = body.data.root_pos_w[:, None, :]
        return quat_apply_inverse(q, (points - p).reshape(n * b, 3)).reshape(n, b, 3)

    def in_bowl(self) -> torch.Tensor:
        """(N,B) bool: ball centre inside the bowl's cup volume (bowl frame) — the clause that
        makes 'park the loaded bowl in the basin' deliver nothing."""
        c = self.cfg
        loc = self._local_to(self.bowl, self._ball_tensors()[0])
        r_ok = loc[:, :, :2].norm(dim=-1) < c.bowl_inner_r + 0.004
        z_ok = (loc[:, :, 2] > -c.bowl_h / 2 - 0.005) & (loc[:, :, 2] < c.bowl_h / 2 + c.ball_r)
        return r_ok & z_ok

    def in_basin(self) -> torch.Tensor:
        """(N,B) bool: ball centre inside the basin's inner well (basin frame), below the rim
        by `rim_margin * ball_r` — a ball perched ON the rim does not count."""
        c = self.cfg
        loc = self._local_to(self.basin, self._ball_tensors()[0])
        xy_ok = (loc[:, :, 0].abs() < c.basin_inner / 2) & (loc[:, :, 1].abs() < c.basin_inner / 2)
        z_ok = (loc[:, :, 2] > -c.basin_wall_h / 2 - 0.5 * c.ball_r) & \
               (loc[:, :, 2] < c.basin_rim_local_z - c.rim_margin * c.ball_r)
        return xy_ok & z_ok

    def ball_settled(self) -> torch.Tensor:
        """(N,B) bool: ball |lin vel| below `settle_speed`."""
        return self._ball_tensors()[1] < self.cfg.settle_speed

    def delivered(self) -> torch.Tensor:
        """(N,B) bool: present, settled inside the basin, and OUT of the bowl."""
        return self.present & self.in_basin() & ~self.in_bowl() & self.ball_settled()

    def all_delivered(self) -> torch.Tensor:
        """(N,) bool: every PRESENT ball delivered — judged on the sampled subset."""
        return (self.delivered() | ~self.present).all(dim=1)

    def bowl_placed(self) -> torch.Tensor:
        """(N,) bool: bowl standing upright ON the work surface, OUTSIDE the basin, settled.
        The final set-down clause — fails on the ledge (wrong height) and inside the basin
        (the outside-basin clause)."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(self.bowl.data.root_quat_w, ez)
        upright = up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(c.placed_tilt_deg))
        bottom_z = (self.bowl.data.root_pos_w - self.env_origins)[:, 2] - up[:, 2] * c.bowl_h / 2
        on_surface = (bottom_z - c.surface_z).abs() < c.placed_z_tol
        loc = quat_apply_inverse(self.basin.data.root_quat_w,
                                 self.bowl.data.root_pos_w - self.basin.data.root_pos_w)
        clear = c.basin_inner / 2 + c.basin_wall_t + c.bowl_outer_r
        outside = loc[:, :2].abs().max(dim=-1).values > clear
        still = self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return upright & on_surface & outside & still

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.1 lift latch + 0.6 * delivered fraction; 0.9 when ALL present
        balls are delivered; 1.0 iff success (all delivered + bowl set down upright outside
        the basin). ~0 for doing nothing; ball-by-ball transfer scores the same milestones."""
        k = self.delivered().sum(dim=1).float()
        tot = self.present.sum(dim=1).clamp(min=1).float()
        s = 0.1 * self.lifted.float() + 0.6 * k / tot
        all_in = self.all_delivered()
        s = torch.where(all_in, torch.full_like(s, 0.9), s)
        return torch.where(all_in & self.bowl_placed(), torch.ones_like(s), s)

    def success(self) -> torch.Tensor:
        """(N,) bool: every present ball settled in the basin AND the bowl standing upright,
        empty, on the surface outside the basin."""
        return self.all_delivered() & self.bowl_placed()


# Scene-level env binding (robot embodiments are a later stage).
register_env("simgen", lambda: EnvCfg(scene="empty_the_bowl", robot="null", env_spacing=3))
