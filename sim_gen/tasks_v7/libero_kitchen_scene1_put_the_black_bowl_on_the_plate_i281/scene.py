"""BerryPourScene — pour the berries out of the black bowl onto the plate, then set the
empty bowl back down clear of it
(libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i281).

Derived from libero_90 kitchen_scene1 "put the black bowl on the plate", where the whole
task is one blind pick-and-place of a known black bowl onto a passive, empty plate and
success is "bowl on plate". Here the same two protagonists return, but the bowl's ROLE
is inverted from payload to TOOL:

  - The BLACK BOWL is no longer the thing to deliver — it is a CONTAINER holding a
    sampled number of small red BERRIES (3-6 identical spheres, count randomized per
    episode).
  - The PLATE is still the destination, but for the bowl's CONTENTS, not the bowl.

  Goal: every berry resting on the plate's flat top inside its rim, the bowl set back
  down UPRIGHT on the floor CLEAR of the plate, everything at rest.

STRATEGIC DIFFERENCE from the seed: the seed's plan ("carry the bowl onto the plate")
is not just insufficient here — its exact end state is a REJECTED negative twice over:
berries still inside a bowl that rests on the plate are not "on the plate" (the in-bowl
exclusion), and a bowl anywhere on the plate violates the bowl-clear clause. The solver
must instead use the bowl as a pouring tool: lift it, carry it over the plate, TILT it
past horizontal so the berries roll out over the rim and land on the plate under
gravity, then return the emptied bowl to the floor. Success judges a content-transfer
relation (contents here, container there), not an object-on-surface relation.

Everything is procedural (custom compound spawners for the rimmed plate and the open
bowl; berries are plain spheres). No joints, no hidden state. Heavy imports (isaaclab,
pxr) are deferred so importing this module stays app-free.

Mechanism notes (proven corpus cribs):
  - Custom spawners silently ignore cfg mass_props (density mass) — MassAPI mass is
    authored inside the spawn funcs and smoke asserts get_masses() readback.
  - Berry on-plate tolerance covers the rim-junction corner rest of the inner rim n-gon
    ((apothem - berry_r)/cos(pi/n)) so every genuine against-the-rim rest counts, while
    rim-top perches (center xy >= apothem) do not — the containment-vs-corner rule.
  - Berries counted on the plate must be OUT of the bowl (in-bowl exclusion, computed
    in the bowl's body frame) — a bowl parked on the plate cannot donate its cargo.
  - Settle gates sit at 0.06 m/s, above the GPU phantom-creep artifact band; spheres
    run 4 velocity iterations + angular damping so rolling actually dies out.
  - Latched progress (bowl ever lifted; max fraction of berries ever delivered) is
    updated in post_step every step — latched credit never evaporates.
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


# ----- custom compound spawners -------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_body(root, mass: float, ang_damp: float) -> None:
    """Rigid-body + explicit MassAPI mass (custom spawners must author mass themselves —
    the cfg mass_props path is silently ignored for custom funcs) + damping."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _spawn_plate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: a white dinner plate — flat disc + a low raised rim ring of box
    segments around the edge. Body frame: origin at the disc center, +z up; the flat
    top the berries land on is at local z = disc_t/2."""
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
    _apply_body(root, cfg.mass, 0.05)
    color = Gf.Vec3f(*cfg.color)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    disc = UsdGeom.Cylinder.Define(stage, f"{prim_path}/disc")
    disc.CreateRadiusAttr(float(cfg.disc_r))
    disc.CreateHeightAttr(float(cfg.disc_t))
    disc.CreateExtentAttr([Gf.Vec3f(-cfg.disc_r, -cfg.disc_r, -cfg.disc_t / 2),
                           Gf.Vec3f(cfg.disc_r, cfg.disc_r, cfg.disc_t / 2)])
    disc.CreateDisplayColorAttr([color])
    collide(disc.GetPrim())

    n = int(cfg.rim_n)
    rim_mid = cfg.disc_r - cfg.rim_t / 2 - 0.002
    seg_len = 2 * math.pi * rim_mid / n + 0.004
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/rim_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(rim_mid * math.cos(ang), rim_mid * math.sin(ang),
                                          cfg.disc_t / 2 + cfg.rim_h / 2))
        sxf.AddRotateZOp().Set(math.degrees(ang) + 90.0)
        sxf.AddScaleOp().Set(Gf.Vec3f(seg_len, cfg.rim_t, cfg.rim_h))
        seg.CreateDisplayColorAttr([color])
        collide(seg.GetPrim())
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: the black open bowl (a cup). Body frame: +z up (open rim up),
    origin at the wall cylinder's mid-height. Ten wall box segments around a closed
    bottom disc; the interior floor is at local z = -wall_h/2, the open rim at
    +wall_h/2, the outer bottom face at -wall_h/2 - bot_t."""
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
    _apply_body(root, cfg.mass, 0.20)
    color = Gf.Vec3f(*cfg.color)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    n = 10
    outer_r = cfg.inner_r + cfg.wall_t
    r_mid = cfg.inner_r + cfg.wall_t / 2
    seg_len = 2 * outer_r * math.tan(math.pi / n) + 0.002
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang), 0.0))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(cfg.wall_t, seg_len, cfg.wall_h))
        seg.CreateDisplayColorAttr([color])
        collide(seg.GetPrim())

    bot = UsdGeom.Cylinder.Define(stage, f"{prim_path}/bottom")
    bot.CreateRadiusAttr(float(outer_r))
    bot.CreateHeightAttr(float(cfg.bot_t))
    bot.CreateExtentAttr([Gf.Vec3f(-outer_r, -outer_r, -cfg.bot_t / 2),
                          Gf.Vec3f(outer_r, outer_r, cfg.bot_t / 2)])
    UsdGeom.Xformable(bot.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, -cfg.wall_h / 2 - cfg.bot_t / 2))
    bot.CreateDisplayColorAttr([color])
    collide(bot.GetPrim())
    return root


def _plate_spawner_cfg(c: BerryPourSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "plate" not in _SPAWNER_CACHE:

        @configclass
        class PlateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plate)
            disc_r: float = 0.115
            disc_t: float = 0.014
            rim_h: float = 0.014
            rim_t: float = 0.008
            rim_n: int = 12
            mass: float = 1.2
            color: tuple = (0.93, 0.93, 0.90)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["plate"] = PlateSpawnerCfg

    return _SPAWNER_CACHE["plate"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.plate_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        disc_r=c.plate_r, disc_t=c.plate_t, rim_h=c.plate_rim_h, rim_t=c.plate_rim_t,
        rim_n=c.plate_rim_n, mass=c.plate_mass, color=c.plate_color,
        contact_offset=c.contact_offset,
    )


def _bowl_spawner_cfg(c: BerryPourSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bowl" not in _SPAWNER_CACHE:

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            inner_r: float = 0.055
            wall_t: float = 0.008
            wall_h: float = 0.060
            bot_t: float = 0.008
            mass: float = 0.25
            color: tuple = (0.08, 0.08, 0.09)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["bowl"] = BowlSpawnerCfg

    return _SPAWNER_CACHE["bowl"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=c.bowl_inner_r, wall_t=c.bowl_wall_t, wall_h=c.bowl_wall_h,
        bot_t=c.bowl_bot_t, mass=c.bowl_mass, color=c.bowl_color,
        contact_offset=c.contact_offset,
    )


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class BerryPourSceneCfg(BaseCfg):
    """Config for `BerryPourScene`. World frame (per env origin): the plate rests on the
    floor around `plate_base`; the bowl rests on the floor around `bowl_base` holding
    the sampled berries; berries not present this episode park in an off-scene depot."""

    # --- tunable: rubric thresholds ----------------------------------------------------------------
    on_tol: float = tunable(0.100)  # berry center within this of the plate center (xy).
    # Corner-rest honesty (asserted in __post_init__): the rim inner faces form a
    # `plate_rim_n`-gon with apothem 0.105; a berry wedged into a face JUNCTION rests at
    # (0.105 - berry_r)/cos(pi/n) ~ 0.0958 <= this — every genuine against-the-rim rest
    # counts. A berry perched ON the rim top has its contact on the top face, so its
    # center xy >= the apothem 0.105 > this — perches do not count.
    on_z_lo: float = tunable(-0.005)  # berry-center z band about (plate_top + berry_r)
    on_z_hi: float = tunable(0.055)  # ... generous above: berries may pile 2-3 deep
    bowl_up_max_deg: float = tunable(20.0)  # bowl +z within this of world-up at the end
    clear_dist: float = tunable(0.20)  # bowl center at least this far (xy) from the plate
    bowl_rest_z_tol: float = tunable(0.012)  # bowl root z within this of its floor rest height
    settle_lin: float = tunable(0.06)  # max |lin vel| when judging (above GPU creep band)
    settle_ang: float = tunable(0.50)  # max bowl |ang vel| when judging (rad/s)
    lift_h: float = tunable(0.05)  # bowl root this far above rest height latches "lifted"
    deliver_slow: float = tunable(0.15)  # berry |v| below this to latch delivered credit

    # --- tunable: randomization --------------------------------------------------------------------
    plate_base: tuple = tunable((0.38, 0.02))  # nominal plate center (env frame)
    plate_jitter: float = tunable(0.030)  # uniform +/- xy jitter of the plate at reset
    bowl_base: tuple = tunable((0.04, -0.04))  # nominal bowl center (env frame)
    bowl_jitter: float = tunable(0.040)  # uniform +/- xy jitter of the bowl at reset
    min_present: int = tunable(3)  # sampled berry count lower bound
    berry_max: int = tunable(6)  # total berry bodies (= count upper bound)

    # --- info: plate -------------------------------------------------------------------------------
    plate_r: float = info(0.115)
    plate_t: float = info(0.014)
    plate_rim_h: float = info(0.014)
    plate_rim_t: float = info(0.008)
    plate_rim_n: int = info(16)
    plate_mass: float = info(1.2)
    plate_color: tuple = info((0.93, 0.93, 0.90))

    # --- info: bowl (the seed's black bowl, now a pouring tool) ------------------------------------
    bowl_inner_r: float = info(0.055)
    bowl_wall_t: float = info(0.008)
    bowl_wall_h: float = info(0.060)
    bowl_bot_t: float = info(0.008)
    bowl_mass: float = info(0.25)
    bowl_color: tuple = info((0.08, 0.08, 0.09))

    # --- info: berries -----------------------------------------------------------------------------
    berry_r: float = info(0.011)
    berry_mass: float = info(0.008)
    berry_color: tuple = info((0.78, 0.07, 0.10))
    depot_pos: tuple = info((1.2, 0.9))  # off-scene ground depot for absent berries

    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    bowl_outer_r: float = field(default=None, init=False)
    bowl_rest_z: float = field(default=None, init=False)  # root z when resting on the floor
    plate_rim_inner_r: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.bowl_outer_r = round(self.bowl_inner_r + self.bowl_wall_t, 4)
        self.bowl_rest_z = round(self.bowl_wall_h / 2 + self.bowl_bot_t, 4)
        self.plate_rim_inner_r = round(self.plate_r - self.plate_rim_t - 0.002, 4)
        # corner-rest honesty: a berry wedged into a rim face JUNCTION (the farthest
        # genuine against-the-rim rest of the inner n-gon) must count ...
        corner_rest = (self.plate_rim_inner_r - self.berry_r) / math.cos(math.pi / self.plate_rim_n)
        assert self.on_tol >= corner_rest + 0.003, (
            f"on_tol {self.on_tol} rejects a berry wedged into a rim junction ({corner_rest:.4f})")
        # ... but a berry perched on the rim TOP must not: a sphere resting on the flat
        # rim top has its contact directly beneath the center, so center xy >= the inner
        # face apothem.
        assert self.on_tol <= self.plate_rim_inner_r - 0.004, (
            f"on_tol {self.on_tol} would accept a berry perched on the rim top "
            f"(min perch xy {self.plate_rim_inner_r:.4f})")
        # bowl-clear geometry: clear_dist really separates the footprints
        assert self.clear_dist >= self.plate_r + self.bowl_outer_r + 0.015, (
            "clear_dist must put the bowl fully off the plate with margin")
        # reset never starts in violation of clear_dist even at worst-case jitter
        base_d = math.hypot(self.plate_base[0] - self.bowl_base[0],
                            self.plate_base[1] - self.bowl_base[1])
        worst = base_d - math.sqrt(2) * (self.plate_jitter + self.bowl_jitter)
        assert worst > self.clear_dist + 0.02, (
            f"bowl/plate bases too close: worst-case start distance {worst:.3f}")
        # the sampled berries fit inside the bowl in <= 2 layers of 3
        assert self.bowl_inner_r >= 0.024 + self.berry_r + 0.004, "berry seat ring too tight"
        assert self.berry_max <= 6 and self.min_present >= 1
        assert (self.bowl_bot_t + 2 * (2 * self.berry_r + 0.004)
                < self.bowl_bot_t + self.bowl_wall_h), "berry stack taller than the bowl wall"
        # the bowl rim is pinch-graspable (8 mm wall in a parallel jaw)
        assert self.bowl_wall_t <= 0.020, "bowl wall too thick for a rim pinch-grasp"


# ----- scene ----------------------------------------------------------------------------------------
@SCENES.register("berry_pour")
class BerryPourScene(BaseScene):
    cfg: BerryPourSceneCfg

    def __init__(self, cfg: BerryPourSceneCfg | None = None) -> None:
        super().__init__(cfg or BerryPourSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
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
        out["plate"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plate",
            spawn=_plate_spawner_cfg(c),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.plate_base[0], c.plate_base[1], c.plate_t / 2 + 0.001)),
        )
        out["bowl"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bowl",
            spawn=_bowl_spawner_cfg(c),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.bowl_base[0], c.bowl_base[1], c.bowl_rest_z + 0.001)),
        )
        for i in range(c.berry_max):
            out[f"berry_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Berry_" + str(i),
                spawn=sim_utils.SphereCfg(
                    radius=c.berry_r,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.10, angular_damping=0.30),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.berry_mass),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.berry_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bowl_base[0], c.bowl_base[1] + 0.02 * i,
                         c.bowl_bot_t + c.berry_r + 0.02 + 0.03 * i)),
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
        super().bind(env)
        n = env.num_envs
        dev = env.device
        c = self.cfg
        self.plate: RigidObject = env.iscene["plate"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.berries: list[RigidObject] = [env.iscene[f"berry_{i}"] for i in range(c.berry_max)]
        self.env_origins = env.iscene.env_origins
        # present[e, i]: berry i participates in episode e (sampled at reset)
        self.present = torch.ones(n, c.berry_max, dtype=torch.bool, device=dev)
        # latched progress (updated in post_step every step)
        self._lift_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._deliver_frac_max = torch.zeros(n, device=dev)

    # ----- helper readouts ------------------------------------------------------------------------
    def plate_top_z(self) -> torch.Tensor:
        """(N,) world z of the plate's flat top (live plate pose)."""
        return self.plate.data.root_pos_w[:, 2] + self.cfg.plate_t / 2

    def _up_of(self, body: RigidObject) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)

    def _berry_pos(self) -> torch.Tensor:
        """(N, B, 3) world berry positions."""
        return torch.stack([b.data.root_pos_w for b in self.berries], dim=1)

    def _berry_vel(self) -> torch.Tensor:
        """(N, B) berry |lin vel|."""
        return torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.berries], dim=1)

    def n_present(self) -> torch.Tensor:
        """(N,) sampled berry count."""
        return self.present.sum(dim=1).clamp(min=1)

    # ----- predicates -----------------------------------------------------------------------------
    def in_bowl(self) -> torch.Tensor:
        """(N, B) bool: berry center inside the bowl's interior volume, computed in the
        BOWL'S BODY FRAME (tilt-aware). Load-bearing exclusion: a berry that is still in
        the bowl is never 'on the plate', no matter where the bowl is."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        pos = self._berry_pos()
        n, b = pos.shape[0], pos.shape[1]
        bq = self.bowl.data.root_quat_w[:, None, :].expand(n, b, 4).reshape(-1, 4)
        bp = self.bowl.data.root_pos_w[:, None, :]
        loc = quat_apply_inverse(bq, (pos - bp).reshape(-1, 3)).reshape(n, b, 3)
        inside_r = loc[:, :, :2].norm(dim=-1) < c.bowl_inner_r
        z_ok = ((loc[:, :, 2] > -c.bowl_wall_h / 2 - c.bowl_bot_t - 0.005)
                & (loc[:, :, 2] < c.bowl_wall_h / 2 + c.berry_r))
        return inside_r & z_ok

    def on_plate(self) -> torch.Tensor:
        """(N, B) bool, geometric: berry OUT of the bowl, resting on the plate's flat
        top inside the rim — xy within `on_tol` of the plate center, center in the
        resting z band (single berries and small piles both count; a berry on the rim
        top, on the floor beside the plate, or hovering does not)."""
        c = self.cfg
        pos = self._berry_pos()
        dxy = (pos[:, :, :2] - self.plate.data.root_pos_w[:, None, :2]).norm(dim=-1)
        dz = pos[:, :, 2] - (self.plate_top_z()[:, None] + c.berry_r)
        z_ok = (dz > c.on_z_lo) & (dz < c.on_z_hi)
        return (dxy < c.on_tol) & z_ok & ~self.in_bowl()

    def counted(self) -> torch.Tensor:
        """(N, B) bool: on the plate AND present — what the rubric counts."""
        return self.on_plate() & self.present

    def all_delivered(self) -> torch.Tensor:
        """(N,) bool: every present berry is on the plate."""
        return (self.counted() | ~self.present).all(dim=1)

    def bowl_upright(self) -> torch.Tensor:
        """(N,) bool: bowl +z within `bowl_up_max_deg` of world-up."""
        cos_max = math.cos(math.radians(self.cfg.bowl_up_max_deg))
        return self._up_of(self.bowl)[:, 2].clamp(-1.0, 1.0) >= cos_max

    def bowl_clear(self) -> torch.Tensor:
        """(N,) bool: the bowl set back down on the floor — upright, its center at
        least `clear_dist` (xy) from the plate center, root z at floor rest height."""
        c = self.cfg
        dxy = (self.bowl.data.root_pos_w[:, :2] - self.plate.data.root_pos_w[:, :2]).norm(dim=-1)
        z = (self.bowl.data.root_pos_w - self.env_origins)[:, 2]
        z_ok = (z - c.bowl_rest_z).abs() < c.bowl_rest_z_tol
        return (dxy > c.clear_dist) & z_ok & self.bowl_upright()

    def settled(self) -> torch.Tensor:
        """(N,) bool: present berries, bowl and plate still (lin), bowl not spinning."""
        c = self.cfg
        bv = self._berry_vel()
        berries_still = ((bv < c.settle_lin) | ~self.present).all(dim=1)
        bowl_still = self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        bowl_slow = self.bowl.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang
        plate_still = self.plate.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        return berries_still & bowl_still & bowl_slow & plate_still

    # ----- mechanism (every step) -----------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        c = self.cfg
        z = (self.bowl.data.root_pos_w - self.env_origins)[:, 2]
        self._lift_ever |= z > c.bowl_rest_z + c.lift_h
        slow = self._berry_vel() < c.deliver_slow
        frac = (self.counted() & slow).sum(dim=1).float() / self.n_present().float()
        self._deliver_frac_max = torch.maximum(self._deliver_frac_max, frac)

    # ----- reset ----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the berry count (3-6), place the plate and the bowl on
        the floor with xy jitter + free yaw, seat the present berries INSIDE the bowl
        (two layers of a 3-slot ring, tiny drop), park absent berries in the depot."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def yaw_quat_cols(st: torch.Tensor, yaw: torch.Tensor) -> None:
            half = yaw / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)

        # berry count: torch.rand comparison (first-randint-after-seed is degenerate
        # on this stack) -> k in [min_present, berry_max]
        span = c.berry_max - c.min_present + 1
        k = c.min_present + (torch.rand(m, device=dev) * span).long().clamp(max=span - 1)
        rank = torch.rand(m, c.berry_max, device=dev).argsort(dim=1).argsort(dim=1)
        self.present[env_ids] = rank < k.unsqueeze(1)

        # plate
        pxy = torch.tensor(c.plate_base, device=dev).expand(m, 2).clone()
        pxy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.plate_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = pxy
        st[:, 2] = c.plate_t / 2 + 0.001
        yaw_quat_cols(st, (torch.rand(m, device=dev) * 2 - 1) * math.pi)
        st[:, 0:3] += origin
        self.plate.write_root_state_to_sim(st, env_ids)

        # bowl
        bxy = torch.tensor(c.bowl_base, device=dev).expand(m, 2).clone()
        bxy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.bowl_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = bxy
        st[:, 2] = c.bowl_rest_z + 0.001
        yaw_quat_cols(st, (torch.rand(m, device=dev) * 2 - 1) * math.pi)
        st[:, 0:3] += origin
        self.bowl.write_root_state_to_sim(st, env_ids)

        # berries: layered 3-slot ring inside the bowl (per-episode ring phase)
        ring_r = 0.024
        phase = torch.rand(m, device=dev) * 2 * math.pi
        floor_z = c.bowl_bot_t  # interior floor world z at rest
        for i, body in enumerate(self.berries):
            layer, slot = divmod(i, 3)
            ang = phase + 2 * math.pi * slot / 3 + layer * (math.pi / 3)
            seat = torch.zeros(m, 3, device=dev)
            seat[:, 0] = bxy[:, 0] + ring_r * torch.cos(ang)
            seat[:, 1] = bxy[:, 1] + ring_r * torch.sin(ang)
            seat[:, 2] = floor_z + c.berry_r + 0.003 + layer * (2 * c.berry_r + 0.004)
            park = torch.zeros(m, 3, device=dev)
            park[:, 0] = c.depot_pos[0]
            park[:, 1] = c.depot_pos[1] + 0.07 * i
            park[:, 2] = c.berry_r + 0.002
            pres = self.present[env_ids, i].unsqueeze(1)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.where(pres, seat, park)
            st[:, 3] = 1.0
            body.write_root_state_to_sim(st, env_ids)

        self._lift_ever[env_ids] = False
        self._deliver_frac_max[env_ids] = 0.0

    # ----- state (full, restorable) ---------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "bowl": self.bowl.data.root_state_w[env_ids].clone(),
            "berries": [b.data.root_state_w[env_ids].clone() for b in self.berries],
            "present": self.present[env_ids].clone(),
            "lift": self._lift_ever[env_ids].clone(),
            "frac": self._deliver_frac_max[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.plate.write_root_state_to_sim(state["plate"], env_ids)
        self.bowl.write_root_state_to_sim(state["bowl"], env_ids)
        for b, st in zip(self.berries, state["berries"]):
            b.write_root_state_to_sim(st, env_ids)
        self.present[env_ids] = state["present"]
        self._lift_ever[env_ids] = state["lift"]
        self._deliver_frac_max[env_ids] = state["frac"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "On the floor stands a round WHITE PLATE (about "
            f"{2 * c.plate_r * 100:.0f} cm across, with a low raised rim around its "
            "edge). Some distance beside it, on the floor, stands an open BLACK BOWL "
            f"(a cup about {2 * c.bowl_outer_r * 100:.1f} cm wide and "
            f"{(c.bowl_wall_h + c.bowl_bot_t) * 100:.1f} cm tall, open at the top, "
            "with a thin rim you can grip). Inside the bowl lie several small RED "
            f"BERRIES (spheres {2 * c.berry_r * 1000:.0f} mm across). Between "
            f"{c.min_present} and {c.berry_max} berries are present in any episode: "
            "count what you see in the bowl.\n"
            "Goal: every berry must end up resting on the plate's flat top inside its "
            "rim, and the black bowl must end standing upright on the FLOOR, well "
            "clear of the plate (its center at least "
            f"{c.clear_dist * 100:.0f} cm from the plate center), with everything at "
            "rest. The intended way is to pour: lift the bowl, carry it over the "
            "plate, and tip it past horizontal so the berries roll out over its rim "
            "and drop onto the plate, then set the emptied bowl back down where there "
            "is free floor.\n"
            "Berries still inside the bowl do NOT count as on the plate even if the "
            "bowl itself is placed on the plate — and a bowl left anywhere on the "
            "plate, tipped over on its side, or upside-down at the end fails. A berry "
            "that lands on the floor beside the plate or sits on top of the plate's "
            "rim does not count."
        )

    def instruction(self) -> str:
        return (
            "Pour all the red berries out of the black bowl onto the white plate so "
            "every berry rests on the plate inside its rim, then set the empty bowl "
            "back down upright on the floor clear of the plate; leaving any berry in "
            "the bowl or off the plate, or leaving the bowl on the plate or tipped "
            "over, fails."
        )

    # ----- rubric ---------------------------------------------------------------------------------
    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched stages of the demonstrated solution: 0.10 the
        bowl ever lifted off the floor + 0.55 * (max fraction of present berries ever
        delivered onto the plate, out of the bowl and slow); exactly 1.0 iff
        success(). The null policy (bowl at rest, berries in it) scores 0; latched
        credit never evaporates."""
        s = 0.10 * self._lift_ever.float() + 0.55 * self._deliver_frac_max
        return torch.where(self.success(),
                           torch.ones(self.env.num_envs, device=self.env.device),
                           s.clamp(0.0, 0.90))

    def success(self) -> torch.Tensor:
        """(N,) bool, all physical: every present berry resting on the plate top (out
        of the bowl) AND the bowl standing upright on the floor clear of the plate AND
        everything settled."""
        return self.all_delivered() & self.bowl_clear() & self.settled()


register_env("simgen", lambda: EnvCfg(scene="berry_pour", robot="null"))
