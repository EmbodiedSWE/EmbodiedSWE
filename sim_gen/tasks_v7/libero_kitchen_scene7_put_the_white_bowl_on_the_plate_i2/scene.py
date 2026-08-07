"""BowlDecantScene — empty the loaded white bowl into the rimmed dish, then park the
bowl UPSIDE-DOWN in the drying tray.

Derived from libero_90/kitchen_scene7 "put the white bowl on the plate", but the
MANIPULATION MODEL is replaced wholesale. The seed's plan is: grasp the (empty) white
bowl, carry it, set it down upright so its center lands within 6 cm of the plate
center — one rigid pick-and-place judged purely by the bowl's own resting pose. Here
the bowl is never the payload: it arrives LOADED with 2-4 blue balls, and the task is
to TRANSFER THE CONTENTS — tip the bowl over the teal dish so the balls roll out over
its lip and are caught by the dish's raised rim (a pour: the last centimeters of every
transfer are performed by gravity, not the hand) — and then to stow the bowl
rim-down (INVERTED) inside the square wooden drying tray on the opposite side. The
seed's own end state (bowl set upright on the dish, contents still inside) is
expressible in this scene and scores ~0: contents inside the bowl are explicitly NOT
"in the dish" (containment-exclusion clause), and the bowl is neither inverted nor in
the tray. Execution order is physically encoded, not declared by fiat: inverting the
bowl before pouring dumps the balls wherever the bowl happens to be, and a ball
trapped under the parked bowl is still "in the bowl", so pour-then-park is forced.

Assets are fully procedural (the compound-spawner pattern — child colliders of one
body never self-collide):
  - bowl: DYNAMIC white octagonal cup (inner r 48 mm, wall 6 mm x 50 mm tall, floor
    10 mm; outer ~118 mm across, 60 mm tall, 150 g). Local origin at the bottom
    center of the floor, +z up.
  - dish: KINEMATIC teal basin — base disc r 95 mm + 12-sided rim, inner r 80 mm,
    rim 30 mm tall. The rim strictly overtops a resting ball (30 > 2*12 mm): a
    settled ball physically inside cannot be pushed out over it (smoke-verified).
  - tray: KINEMATIC brown square drying tray 175 x 175 mm, 14 mm pad, 6 mm lip.
  - balls: four DYNAMIC blue spheres r 12 mm, 10 g; 2..4 are PRESENT per episode
    (subset sampling — count what you see), absentees parked in a ground depot.

Per-episode randomization (readback-verifiable): WHICH SIDE the dish and the tray
occupy (they swap stations at random — perceive, don't memorize), xy jitter on dish /
tray / bowl, tray yaw, bowl yaw, ball count 2..4 and in-bowl placement.

Rubric (0..1; latched partial credit anchored in the demonstrated solve trajectory):
  0.10 * lift    — the bowl was ever raised above `lift_z` (latched; null policy 0)
  0.20 * first   — some present ball ever settled INSIDE THE DISH (latched)
  0.25 * all_in  — ALL present balls settled inside the dish simultaneously (latched)
  0.20 * parked  — the bowl at rest inverted inside the tray while `all_in` was
                   already latched (latched; parking before pouring earns nothing)
  1.0 iff success() — live: every present ball inside the dish (and not inside the
                   bowl), the bowl at rest upside-down inside the tray, all settled.
  Non-success capped at 0.75.

"In the dish" is honest by construction: the rim bounds a resting ball center to
r <= 70.8 mm (12-gon corner reach 82.8 mm minus the ball radius) < the 74 mm gate,
while a ball outside the rim sits at r >= 100 mm; the
containment-exclusion clause (ball not inside the bowl volume) is what rejects the
seed strategy of putting the loaded bowl on the dish. "In the tray" likewise: the
lips bound a resting inverted bowl center to |xy| <= ~21 mm < the 45 mm gate.

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


# ----- custom compound spawners ---------------------------------------------------------------
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable, orient=None):
    """One box child: translate [+ orient] + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable):
    """One z-axis cylinder child: translate, displayColor, collider."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr("Z")
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    return cyl.GetPrim()


def _qz_tuple(yaw: float) -> tuple:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the bowl at `prim_path`: DYNAMIC compound. Local frame: origin at the
    bottom center of the floor disc, +z up, rim plane at z = floor_t + wall_h.

    Children: floor disc + eight wall boxes forming an octagonal cup."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateMaxDepenetrationVelocityAttr(1.0)
    px.CreateLinearDampingAttr(0.2)
    px.CreateAngularDampingAttr(0.2)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_cyl(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.floor_t / 2),
             radius=c.inner_r + c.wall_t, height=c.floor_t, color=c.color,
             collide=collide)
    n_side = 8
    rmid = c.inner_r + c.wall_t / 2
    side_l = 2 * rmid * math.tan(math.pi / n_side) + 0.004
    zc = c.floor_t + c.wall_h / 2
    for i in range(n_side):
        a = i * 2 * math.pi / n_side
        _add_box(stage, f"{prim_path}/wall_{i}",
                 center=(rmid * math.cos(a), rmid * math.sin(a), zc),
                 size=(c.wall_t, side_l, c.wall_h), color=c.color,
                 collide=collide, orient=_qz_tuple(a))
    return root


def _spawn_dish(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the dish: KINEMATIC compound. Local frame: origin at the bottom center
    on the ground. Base disc + a 12-sided raised rim (inner face at `inner_r`)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_cyl(stage, f"{prim_path}/base", center=(0.0, 0.0, c.base_t / 2),
             radius=c.base_r, height=c.base_t, color=c.color, collide=collide)
    n_side = 12
    rmid = c.inner_r + c.wall_t / 2
    side_l = 2 * rmid * math.tan(math.pi / n_side) + 0.004
    zc = c.base_t + c.wall_h / 2
    for i in range(n_side):
        a = i * 2 * math.pi / n_side
        _add_box(stage, f"{prim_path}/rim_{i}",
                 center=(rmid * math.cos(a), rmid * math.sin(a), zc),
                 size=(c.wall_t, side_l, c.wall_h), color=c.rim_color,
                 collide=collide, orient=_qz_tuple(a))
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the drying tray: KINEMATIC compound. Local frame: origin at the bottom
    center on the ground. Square pad + four low lip walls."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/pad", center=(0.0, 0.0, c.base_t / 2),
             size=(c.side, c.side, c.base_t), color=c.color, collide=collide)
    off = c.side / 2 - c.lip_t / 2
    zc = c.base_t + c.lip_h / 2
    for i, (dx, dy, sx, sy) in enumerate((
            (off, 0.0, c.lip_t, c.side), (-off, 0.0, c.lip_t, c.side),
            (0.0, off, c.side, c.lip_t), (0.0, -off, c.side, c.lip_t))):
        _add_box(stage, f"{prim_path}/lip_{i}", center=(dx, dy, zc),
                 size=(sx, sy, c.lip_h), color=c.lip_color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bowl" not in _SPAWNER_CACHE:

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            inner_r: float = 0.048
            wall_t: float = 0.006
            wall_h: float = 0.050
            floor_t: float = 0.010
            mass: float = 0.15
            color: tuple = (0.95, 0.95, 0.92)
            contact_offset: float = 0.002

        @configclass
        class DishSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_dish)
            base_r: float = 0.095
            base_t: float = 0.012
            inner_r: float = 0.080
            wall_t: float = 0.008
            wall_h: float = 0.030
            color: tuple = (0.10, 0.50, 0.47)
            rim_color: tuple = (0.13, 0.62, 0.58)
            contact_offset: float = 0.002

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            side: float = 0.175
            base_t: float = 0.014
            lip_t: float = 0.008
            lip_h: float = 0.006
            color: tuple = (0.46, 0.31, 0.16)
            lip_color: tuple = (0.55, 0.38, 0.20)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(bowl=BowlSpawnerCfg, dish=DishSpawnerCfg,
                              tray=TraySpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BowlDecantSceneCfg(BaseCfg):
    """Config for `BowlDecantScene`. The in-dish and in-tray tolerances are honest by
    construction: the dish rim bounds a resting ball center to r <= 70.8 mm (gate
    74 mm), the tray lips bound a resting inverted bowl center to |xy| <= ~21 mm
    (gate 45 mm), and a ball/bowl OUTSIDE those walls cannot pass the gates."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    dish_xy_tol: float = tunable(0.074)   # ball center within this of the dish axis: the
    # 12-gon rim's inner CORNER reach is 80/cos(15) = 82.8 mm, so a ball nestled into a
    # corner rests at up to 70.8 mm (measured 69.6 on the forge); outside the rim >= 100 mm
    dish_z_lo: float = tunable(0.015)     # dish-frame z band for a ball in the basin
    dish_z_hi: float = tunable(0.055)     # (resting center 24 mm; rim-top rest excluded by xy)
    bowl_excl_r: float = tunable(0.052)   # containment exclusion: ball within this of the bowl
    bowl_excl_z_lo: float = tunable(0.004)   # axis and this bowl-frame z band is IN THE BOWL
    bowl_excl_z_hi: float = tunable(0.095)   # (and therefore never "in the dish")
    tray_xy_tol: float = tunable(0.045)   # bowl origin within this of the tray center (lips: 21)
    tray_z_lo: float = tunable(0.058)     # tray-frame bowl-origin z band when parked inverted
    tray_z_hi: float = tunable(0.090)     # (resting: pad 14 + bowl height 60 = 74 mm)
    invert_max_deg: float = tunable(12.0)  # bowl -z axis within this of world up when parked
    settle_speed: float = tunable(0.05)   # max |lin vel| (balls AND bowl) when judging (m/s)
    lift_z: float = tunable(0.10)         # latched lift credit: bowl origin ever above this

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    side_swap: bool = tunable(True)       # dish/tray swap stations at random
    fixture_jitter: float = tunable(0.03)  # dish and tray xy jitter (+/- m)
    tray_yaw_deg: float = tunable(20.0)   # tray yaw (+/- deg)
    bowl_jitter: float = tunable(0.04)    # bowl xy jitter (+/- m)
    bowl_yaw_deg: float = tunable(180.0)  # bowl yaw (+/- deg)
    subset_sample: bool = tunable(True)   # per-episode ball-count sampling (2..4)
    min_present: int = tunable(2)
    ball_ring_r: float = tunable(0.018)   # in-bowl spawn ring radius
    ball_jitter: float = tunable(0.004)   # in-bowl spawn jitter (+/- m)

    # --- info: layout (env-local, ground plane z = 0) ------------------------------------------
    station_x: float = info(0.36)         # dish/tray stations: (station_x, +/-station_y)
    station_y: float = info(0.16)
    bowl_pos: tuple = info((0.16, 0.0))   # bowl start (between the robot base and stations)
    parking_pos: tuple = info((1.1, 1.1))  # ground depot for absent balls
    # --- info: bowl structure (local origin at the bottom center of the floor) -----------------
    bowl_inner_r: float = info(0.048)
    bowl_wall_t: float = info(0.006)
    bowl_wall_h: float = info(0.050)
    bowl_floor_t: float = info(0.010)
    bowl_h: float = info(0.060)           # floor_t + wall_h: rim plane height above origin
    bowl_outer_r: float = info(0.059)     # octagon corner reach (wall box far corner)
    bowl_mass: float = info(0.15)
    # --- info: dish -----------------------------------------------------------------------------
    dish_base_r: float = info(0.095)
    dish_base_t: float = info(0.012)      # basin floor top height
    dish_inner_r: float = info(0.080)
    dish_wall_h: float = info(0.030)      # rim top = base_t + wall_h = 42 mm
    # --- info: tray -----------------------------------------------------------------------------
    tray_side: float = info(0.175)
    tray_base_t: float = info(0.014)      # pad top height
    tray_lip_h: float = info(0.006)
    # --- info: balls ----------------------------------------------------------------------------
    ball_r: float = info(0.012)
    ball_mass: float = info(0.010)
    n_balls: int = info(4)
    ball_color: tuple = info((0.15, 0.35, 0.90))
    contact_offset: float = info(0.002)
    # --- info: solve geometry (release point used by the certified solution) -------------------
    pour_hold_z: float = info(0.155)      # bowl-origin hold height over the dish while tipping
    # rubric weights (0.10 + 0.20 + 0.25 + 0.20 = 0.75 = the non-success cap)
    w_lift: float = info(0.10)
    w_first: float = info(0.20)
    w_all: float = info(0.25)
    w_park: float = info(0.20)


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
@SCENES.register("bowl_decant")
class BowlDecantScene(BaseScene):
    cfg: BowlDecantSceneCfg

    BALL_NAMES = ("ball_0", "ball_1", "ball_2", "ball_3")

    def __init__(self, cfg: BowlDecantSceneCfg | None = None) -> None:
        super().__init__(cfg or BowlDecantSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        bowl_spawn = cls["bowl"](
            inner_r=c.bowl_inner_r, wall_t=c.bowl_wall_t, wall_h=c.bowl_wall_h,
            floor_t=c.bowl_floor_t, mass=c.bowl_mass, contact_offset=c.contact_offset)
        dish_spawn = cls["dish"](
            base_r=c.dish_base_r, base_t=c.dish_base_t, inner_r=c.dish_inner_r,
            wall_h=c.dish_wall_h, contact_offset=c.contact_offset,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True))
        tray_spawn = cls["tray"](
            side=c.tray_side, base_t=c.tray_base_t, lip_h=c.tray_lip_h,
            contact_offset=c.contact_offset,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True))

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
            "bowl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl",
                spawn=bowl_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bowl_pos[0], c.bowl_pos[1], 0.001)),
            ),
            "dish": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dish",
                spawn=dish_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.station_x, c.station_y, 0.0)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=tray_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.station_x, -c.station_y, 0.0)),
            ),
        }
        for i, name in enumerate(self.BALL_NAMES):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball_" + name,
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.3, angular_damping=0.8,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=1),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.003, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.7, dynamic_friction=0.6,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.ball_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.7 + 0.05 * i, -0.7, 0.02)),
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
        self.bowl: RigidObject = env.iscene["bowl"]
        self.dish: RigidObject = env.iscene["dish"]
        self.tray: RigidObject = env.iscene["tray"]
        self.balls: dict[str, RigidObject] = {
            n: env.iscene[n] for n in self.BALL_NAMES}
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # present[e, i]: ball i participates in episode e (sampled at reset)
        self.present = torch.ones(n, len(self.BALL_NAMES), dtype=torch.bool, device=dev)
        self.dish_side = torch.ones(n, dtype=torch.long, device=dev)  # +1 / -1 (y sign)
        # latches (partial credit survives transients; success is judged live)
        self._lift = torch.zeros(n, dtype=torch.bool, device=dev)
        self._m1 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._mall = torch.zeros(n, dtype=torch.bool, device=dev)
        self._park = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pick the dish/tray side split, place both fixtures (jitter,
        tray yaw), place the bowl (jitter + yaw), sample the present ball subset and
        seat present balls INSIDE the bowl on a small ring; park absentees in the
        ground depot; clear the latches."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- which side is the dish on ---
        if c.side_swap:
            side = torch.where(torch.rand(m, device=dev) < 0.5,
                               torch.ones(m, dtype=torch.long, device=dev),
                               -torch.ones(m, dtype=torch.long, device=dev))
        else:
            side = torch.ones(m, dtype=torch.long, device=dev)
        self.dish_side[env_ids] = side

        def jit(k: float) -> torch.Tensor:
            return (torch.rand(m, 2, device=dev) * 2 - 1) * k

        # --- dish (kinematic): station + jitter ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.station_x
        st[:, 1] = side.float() * c.station_y
        st[:, :2] += jit(c.fixture_jitter)
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.dish.write_root_state_to_sim(st, env_ids)

        # --- tray (kinematic): opposite station + jitter + yaw ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.station_x
        st[:, 1] = -side.float() * c.station_y
        st[:, :2] += jit(c.fixture_jitter)
        st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1)
                         * math.radians(c.tray_yaw_deg))
        st[:, 0:3] += origin
        self.tray.write_root_state_to_sim(st, env_ids)

        # --- bowl (dynamic): start pose + jitter + yaw ---
        b_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.bowl_yaw_deg)
        q_bowl = _qz(b_yaw)
        bp = torch.zeros(m, 3, device=dev)
        bp[:, 0] = c.bowl_pos[0]
        bp[:, 1] = c.bowl_pos[1]
        bp[:, :2] += jit(c.bowl_jitter)
        bp[:, 2] = 0.001
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + origin
        st[:, 3:7] = q_bowl
        self.bowl.write_root_state_to_sim(st, env_ids)

        # --- balls: sample subset, seat present balls inside the bowl on a ring ---
        nb = len(self.BALL_NAMES)
        if c.subset_sample:
            k = torch.randint(c.min_present, nb + 1, (m,), device=dev)
        else:
            k = torch.full((m,), nb, dtype=torch.long, device=dev)
        rank = torch.rand(m, nb, device=dev).argsort(dim=1).argsort(dim=1)
        pres = rank < k.unsqueeze(1)
        self.present[env_ids] = pres
        phase = torch.rand(m, device=dev) * 2 * math.pi
        for i, name in enumerate(self.BALL_NAMES):
            ang = phase + i * (2 * math.pi / nb)
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = c.ball_ring_r * torch.cos(ang)
            loc[:, 1] = c.ball_ring_r * torch.sin(ang)
            loc[:, :2] += jit(c.ball_jitter)
            loc[:, 2] = c.bowl_floor_t + c.ball_r + 0.003
            in_bowl_pos = bp + quat_apply(q_bowl, loc)
            park = torch.zeros(m, 3, device=dev)
            park[:, 0] = c.parking_pos[0] + 0.06 * i
            park[:, 1] = c.parking_pos[1]
            park[:, 2] = c.ball_r + 0.002
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.where(pres[:, i].unsqueeze(1),
                                              in_bowl_pos, park)
            st[:, 3] = 1.0
            self.balls[name].write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._lift[env_ids] = False
        self._m1[env_ids] = False
        self._mall[env_ids] = False
        self._park[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bowl": self.bowl.data.root_state_w[env_ids].clone(),
            "dish": self.dish.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "balls": {n: b.data.root_state_w[env_ids].clone()
                      for n, b in self.balls.items()},
            "present": self.present[env_ids].clone(),
            "dish_side": self.dish_side[env_ids].clone(),
            "lift": self._lift[env_ids].clone(),
            "m1": self._m1[env_ids].clone(),
            "mall": self._mall[env_ids].clone(),
            "park": self._park[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.bowl.write_root_state_to_sim(state["bowl"], env_ids)
        self.dish.write_root_state_to_sim(state["dish"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        for n, b in self.balls.items():
            b.write_root_state_to_sim(state["balls"][n], env_ids)
        self.present[env_ids] = state["present"]
        self.dish_side[env_ids] = state["dish_side"]
        self._lift[env_ids] = state["lift"]
        self._m1[env_ids] = state["m1"]
        self._mall[env_ids] = state["mall"]
        self._park[env_ids] = state["park"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On the floor stands a WHITE octagonal BOWL (~{2 * c.bowl_outer_r * 1000:.0f} mm "
            f"across, {c.bowl_h * 1000:.0f} mm tall, thin {c.bowl_wall_t * 1000:.0f} mm walls) "
            f"loaded with BLUE BALLS ({2 * c.ball_r * 1000:.0f} mm diameter). Between two and "
            f"four balls are present in any episode — look into the bowl and count them. "
            f"Farther away sit two fixtures side by side: a TEAL round DISH (a shallow basin "
            f"~{2 * c.dish_base_r * 1000:.0f} mm across whose raised rim, "
            f"{c.dish_wall_h * 1000:.0f} mm high, keeps balls from rolling out) and a BROWN "
            f"square wooden DRYING TRAY (~{c.tray_side * 1000:.0f} mm, a flat pad with a low "
            f"raised edge). Which fixture is on which side is randomized every episode, along "
            f"with all positions, headings and the ball count — identify them by color and "
            f"shape.\n"
            f"Goal: first TRANSFER EVERY BALL from the bowl into the teal dish — tip the bowl "
            f"over the dish and let the balls roll out over its lip and drop in; a ball that "
            f"ends up anywhere outside the dish basin fails the task. A ball still sitting "
            f"inside the bowl NEVER counts as in the dish, even if the bowl itself is standing "
            f"on the dish. Then park the EMPTIED bowl UPSIDE-DOWN (rim facing down) inside the "
            f"wooden tray, resting within its raised edge. Do it in that order — tipping the "
            f"bowl upside down before it is empty spills the balls where they do not belong. "
            f"Success: all present balls at rest inside the dish basin and the bowl at rest "
            f"inverted inside the tray, nothing still moving."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pour all the blue balls from the white bowl into the teal dish, then place "
            "the empty bowl upside down inside the brown drying tray. Every ball must "
            "end up inside the dish and the bowl must rest inverted in the tray."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _local(self, body, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> `body`'s frame. Accepts (N,3) or (N,P,3); same shape out."""
        from isaaclab.utils.math import quat_apply_inverse

        bp = body.data.root_pos_w
        bq = body.data.root_quat_w
        if pos_w.dim() == 3:
            n, p = pos_w.shape[0], pos_w.shape[1]
            rel = (pos_w - bp[:, None, :]).reshape(n * p, 3)
            q = bq[:, None, :].expand(n, p, 4).reshape(n * p, 4)
            return quat_apply_inverse(q, rel).reshape(n, p, 3)
        return quat_apply_inverse(bq, pos_w - bp)

    def _ball_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos_w (N,4,3), |lin_vel| (N,4)) for all balls, name order."""
        pos = torch.stack([b.data.root_pos_w for b in self.balls.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.balls.values()], dim=1)
        return pos, vel

    def bowl_up(self) -> torch.Tensor:
        """(N, 3): the bowl's local +z axis in world coordinates."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0],
                          device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(self.bowl.data.root_quat_w, ez)

    def in_bowl(self) -> torch.Tensor:
        """(N, 4) bool: ball center inside the bowl's interior volume (bowl frame).
        The CONTAINMENT-EXCLUSION clause: such a ball is never 'in the dish'."""
        c = self.cfg
        pos, _v = self._ball_tensors()
        loc = self._local(self.bowl, pos)
        return (loc[:, :, :2].norm(dim=-1) < c.bowl_excl_r) \
            & (loc[:, :, 2] > c.bowl_excl_z_lo) & (loc[:, :, 2] < c.bowl_excl_z_hi)

    def in_dish(self) -> torch.Tensor:
        """(N, 4) bool, geometric: ball center inside the dish basin (dish frame,
        within `dish_xy_tol` of the axis, z in the basin band) AND NOT inside the
        bowl. Honest by construction: the rim bounds resting in-basin centers to
        r <= 70.8 mm (corner reach); outside the rim r >= 100 mm."""
        c = self.cfg
        pos, _v = self._ball_tensors()
        loc = self._local(self.dish, pos)
        geo = (loc[:, :, :2].norm(dim=-1) < c.dish_xy_tol) \
            & (loc[:, :, 2] > c.dish_z_lo) & (loc[:, :, 2] < c.dish_z_hi)
        return geo & ~self.in_bowl()

    def bowl_parked(self) -> torch.Tensor:
        """(N,) bool: the bowl at rest UPSIDE-DOWN inside the tray — its -z axis
        within `invert_max_deg` of world up, origin within `tray_xy_tol` of the tray
        center and in the parked z band (tray frame), and the bowl settled."""
        c = self.cfg
        up = self.bowl_up()
        inverted = -up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(c.invert_max_deg))
        loc = self._local(self.tray, self.bowl.data.root_pos_w)
        in_tray = (loc[:, :2].norm(dim=-1) < c.tray_xy_tol) \
            & (loc[:, 2] > c.tray_z_lo) & (loc[:, 2] < c.tray_z_hi)
        still = self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return inverted & in_tray & still

    def balls_settled(self) -> torch.Tensor:
        """(N, 4) bool: per-ball |lin vel| below `settle_speed`."""
        _p, vel = self._ball_tensors()
        return vel < self.cfg.settle_speed

    def _update_latches(self) -> None:
        c = self.cfg
        bowl_z = (self.bowl.data.root_pos_w - self.env_origins)[:, 2]
        self._lift |= bowl_z > c.lift_z
        counted = self.in_dish() & self.balls_settled() & self.present
        self._m1 |= counted.any(dim=1)
        self._mall |= (counted | ~self.present).all(dim=1)
        # parking only earns credit AFTER the pour is complete (order in the credit)
        self._park |= self._mall & self.bowl_parked()

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool, live: every PRESENT ball inside the dish basin (and not inside
        the bowl), all present balls settled, the bowl at rest inverted inside the
        tray, everything finite. All clauses are physical outcomes."""
        self._update_latches()
        counted = self.in_dish() & self.balls_settled() & self.present
        all_in = (counted | ~self.present).all(dim=1)
        pos, _v = self._ball_tensors()
        finite = torch.isfinite(pos).all(dim=-1).all(dim=-1) \
            & torch.isfinite(self.bowl.data.root_pos_w).all(dim=-1)
        return all_in & self.bowl_parked() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10*lift + 0.20*first-ball + 0.25*all-balls +
        0.20*parked-after-pour (all latched; ~0 for the null policy), capped at
        0.75 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_lift * self._lift.float() + c.w_first * self._m1.float()
                + c.w_all * self._mall.float()
                + c.w_park * self._park.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="bowl_decant", robot="null"))
