"""MugTipoutScene — tip the fruit OUT of the mug into the green dish, park the mug on its pad.

Derived from embodiedgen/put_banana ("pick the banana off the cluttered table and drop it
into the mug"), but the task relation is INVERTED and the manipulandum is SWAPPED: here the
fruit (an orange, a 32 mm ball) STARTS inside the mug — the seed's success state is this
task's initial condition — and the mug bore (46 mm across flats, 95 mm deep) is far too
narrow for a parallel-jaw gripper to reach in around the ball, so the fruit itself is
UNGRASPABLE while inside. The only feasible plan manipulates the CONTAINER: grasp the mug
body, carry it over the green target dish, tip it until the orange rolls out over the rim
and drops into the dish, then set the empty mug down standing upright on its blue home pad.
An identical-shaped WHITE dish (the decoy) swaps sides with the green one per episode, so
the pour target must be identified by color, not by layout. A solver replaying the seed's
plan — reach in, pick the fruit, drop it somewhere — cannot even start (no grasp exists),
and the seed's terminal relation (fruit inside mug) is worth exactly nothing here.

Assets are fully procedural, authored by custom compound spawners (the pen_holder /
pan_cubby pattern — child colliders of one body never self-collide):
  - mug: DYNAMIC compound — floor disc + 8 wall boxes forming an octagonal bore
    (inner inradius 23 mm, outer ~27 mm across flats, 95 mm tall, 6 mm floor). Origin at
    the mid-height centre of the bore axis. Sleep/stabilization thresholds zeroed at
    spawn (the solve pose-holds it during the pour and it must respond instantly).
  - orange: a 16 mm-radius dynamic sphere, restitution 0 + angular damping 0.4 (the
    proven pour-and-settle ball recipe) — spawned resting inside the mug bore.
  - dishes (green target + white decoy): KINEMATIC compounds — floor disc + 8 wall
    boxes, inner inradius 70 mm, walls 28 mm tall. Origin at the centre of the interior
    floor TOP plane.
  - home pad: kinematic flat blue cylinder (r = 60 mm, 8 mm tall) — the mug's parking
    target.

Per-episode randomization (readback-verifiable): mug spawn xy + yaw, green/white dish
SLOT ASSIGNMENT (Bernoulli side swap) + per-dish xy jitter, home pad xy jitter. Success
cannot be memorized as one pose or one side.

Rubric (0..1 floats; partial progress latched so transient achievements keep credit):
  0.20 * poured_out          — orange ever outside the mug bore (latched bool)
  0.30 * approach progress   — 1 - dist(orange, green dish)/approach_d0, gated on
                               poured_out (latched running max; ~0 for doing nothing)
  0.20 * in_target           — orange ever inside the green dish and not inside the mug
                               (latched bool; the in-mug clause kills the container cheat
                               of parking the loaded mug in the dish)
  0.15 * mug-home progress   — 1 - dist(mug, home pad)/approach_d0, gated on poured_out
                               (latched running max; the mug starting near the pad with
                               the fruit still inside earns nothing)
  1.0 iff success()          — orange settled ON the green dish floor (inside the wall
                               ring, resting at floor height, not inside the mug, still)
                               AND the mug standing upright ON the home pad (centre
                               within mug_home_xy_tol, base at pad-top height, tilt
                               within mug_tilt_max_deg, still). Non-success capped 0.85.

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


# ----- custom compound spawners ---------------------------------------------------------------
# One rigid body per object, several child colliders, authored with raw pxr APIs; only
# `isaaclab.sim.utils.clone` is borrowed (regex-resolve + per-env replication).

_SPAWNER_CACHE: dict[str, Any] = {}


def _octagon_walls(stage, prim_path: str, *, inner_r: float, wall_t: float, wall_h: float,
                   z0: float, color, collide: Callable) -> None:
    """Author 8 wall boxes forming an octagonal ring: inner flat faces at `inner_r` from
    the axis, box bottoms at local z0."""
    from pxr import Gf, UsdGeom

    r_mid = inner_r + wall_t / 2
    seg_len = 2 * (inner_r + wall_t) * math.tan(math.pi / 8) + 0.002
    for k in range(8):
        ang = 2 * math.pi * k / 8
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang),
                                          z0 + wall_h / 2))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(wall_t, seg_len, wall_h))
        seg.CreateDisplayColorAttr([color])
        collide(seg.GetPrim())


def _spawn_mug(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the mug at `prim_path`: DYNAMIC root Xform with RigidBodyAPI + explicit
    MassAPI, a floor disc and 8 wall boxes forming a deep octagonal bore. Origin at the
    mid-height centre of the bore axis; up = +z."""
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
    # Gentle depenetration + a whiff of damping; sleep/stabilization thresholds zeroed —
    # the solve pose-holds the mug through the pour and drops it onto the pad, and a
    # sleeping body would freeze mid-episode.
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    color = Gf.Vec3f(*cfg.color)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    floor = UsdGeom.Cylinder.Define(stage, f"{prim_path}/floor")
    r_out = cfg.inner_r + cfg.wall_t
    floor.CreateRadiusAttr(r_out)
    floor.CreateHeightAttr(cfg.floor_t)
    floor.CreateExtentAttr([Gf.Vec3f(-r_out, -r_out, -cfg.floor_t / 2),
                            Gf.Vec3f(r_out, r_out, cfg.floor_t / 2)])
    fxf = UsdGeom.Xformable(floor.GetPrim())
    fxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -cfg.h / 2 + cfg.floor_t / 2))
    floor.CreateDisplayColorAttr([color])
    collide(floor.GetPrim())

    _octagon_walls(stage, prim_path, inner_r=cfg.inner_r, wall_t=cfg.wall_t,
                   wall_h=cfg.h - cfg.floor_t, z0=-cfg.h / 2 + cfg.floor_t,
                   color=color, collide=collide)
    return root


def _spawn_dish(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a shallow dish at `prim_path`: KINEMATIC rigid body (repositionable at
    reset, immovable to contacts) — floor disc + 8 low wall boxes. Origin at the centre
    of the interior floor TOP plane."""
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

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    floor = UsdGeom.Cylinder.Define(stage, f"{prim_path}/floor")
    r_out = cfg.inner_r + cfg.wall_t
    floor.CreateRadiusAttr(r_out)
    floor.CreateHeightAttr(cfg.floor_t)
    floor.CreateExtentAttr([Gf.Vec3f(-r_out, -r_out, -cfg.floor_t / 2),
                            Gf.Vec3f(r_out, r_out, cfg.floor_t / 2)])
    fxf = UsdGeom.Xformable(floor.GetPrim())
    fxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -cfg.floor_t / 2))
    floor.CreateDisplayColorAttr([color])
    collide(floor.GetPrim())

    _octagon_walls(stage, prim_path, inner_r=cfg.inner_r, wall_t=cfg.wall_t,
                   wall_h=cfg.wall_h, z0=0.0, color=color, collide=collide)
    return root


def _mug_spawner_cfg(*, inner_r: float, wall_t: float, h: float, floor_t: float,
                     mass: float, color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "mug" not in _SPAWNER_CACHE:

        @configclass
        class MugSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_mug)
            inner_r: float = 0.023
            wall_t: float = 0.004
            h: float = 0.095
            floor_t: float = 0.006
            color: tuple = (0.70, 0.12, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["mug"] = MugSpawnerCfg

    return _SPAWNER_CACHE["mug"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=inner_r, wall_t=wall_t, h=h, floor_t=floor_t,
        color=color, contact_offset=contact_offset,
    )


def _dish_spawner_cfg(*, inner_r: float, wall_t: float, wall_h: float, floor_t: float,
                      color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "dish" not in _SPAWNER_CACHE:

        @configclass
        class DishSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_dish)
            inner_r: float = 0.070
            wall_t: float = 0.005
            wall_h: float = 0.028
            floor_t: float = 0.008
            color: tuple = (0.5, 0.5, 0.5)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["dish"] = DishSpawnerCfg

    return _SPAWNER_CACHE["dish"](
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        inner_r=inner_r, wall_t=wall_t, wall_h=wall_h, floor_t=floor_t,
        color=color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class MugTipoutSceneCfg(BaseCfg):
    """Config for `MugTipoutScene`. Bore clearance = inner flat-to-flat (46 mm) minus ball
    diameter (32 mm) = 7 mm per side: the ball pours freely but no parallel-jaw finger pair
    fits around it inside the bore — the geometry, not the rubric, forbids the seed's
    pick-the-fruit plan."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    ball_z_tol: float = tunable(0.012)  # orange centre at dish-floor + r within this (m)
    mug_home_xy_tol: float = tunable(0.045)  # mug centre within this of the pad axis (m)
    mug_home_z_tol: float = tunable(0.010)  # mug base at pad-top height within this (m)
    mug_tilt_max_deg: float = tunable(10.0)  # mug up-axis within this of world-up
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    approach_d0: float = tunable(0.45)  # progress ramps: p = 1 - d/approach_d0

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    mug_jitter: float = tunable(0.04)  # mug spawn xy jitter (+/- m)
    mug_yaw_deg: float = tunable(180.0)  # mug spawn yaw (+/- deg; cosmetic for the octagon)
    dish_jitter: float = tunable(0.025)  # per-dish xy jitter (+/- m)
    home_jitter: float = tunable(0.03)  # home pad xy jitter (+/- m)
    swap_dishes: bool = tunable(True)  # Bernoulli green/white slot swap (demo sets False)

    # --- info: layout (single Franka base at the origin; all radii 0.40-0.62 m) -----------------
    mug_pos: tuple = info((0.55, 0.12))  # mug spawn centre (fruit inside)
    dish_slot_a: tuple = info((0.50, -0.10))  # dish slot A
    dish_slot_b: tuple = info((0.44, -0.32))  # dish slot B
    home_pos: tuple = info((0.40, 0.30))  # home pad centre

    # --- info: structure -------------------------------------------------------------------------
    mug_inner_r: float = info(0.023)  # bore inradius (flats); ball clearance 7 mm/side
    mug_wall_t: float = info(0.004)
    mug_h: float = info(0.095)  # deep bore: ball top sits ~57 mm below the rim
    mug_floor_t: float = info(0.006)
    mug_mass: float = info(0.15)
    mug_color: tuple = info((0.70, 0.12, 0.10))  # red
    ball_r: float = info(0.016)  # the orange; dia 32 mm
    ball_mass: float = info(0.03)
    ball_color: tuple = info((0.90, 0.55, 0.10))  # orange
    dish_inner_r: float = info(0.070)
    dish_wall_t: float = info(0.005)
    dish_wall_h: float = info(0.028)
    dish_floor_t: float = info(0.008)
    target_color: tuple = info((0.10, 0.55, 0.15))  # green — the pour target
    decoy_color: tuple = info((0.85, 0.85, 0.85))  # white — identical shape, wrong target
    pad_r: float = info(0.060)
    pad_h: float = info(0.008)
    pad_color: tuple = info((0.10, 0.25, 0.75))  # blue
    contact_offset: float = info(0.002)
    # rubric weights (0.20 + 0.30 + 0.20 + 0.15 = 0.85 = the non-success cap)
    w_out: float = info(0.20)
    w_approach: float = info(0.30)
    w_in: float = info(0.20)
    w_home: float = info(0.15)

    # Derived (filled in __post_init__).
    ball_seat_z: float = field(default=None, init=False)  # ball centre when seated in the mug,
    # relative to the mug BASE (bore floor top + ball_r)
    dish_capture_r: float = field(default=None, init=False)  # max in-dish xy offset judged

    def __post_init__(self) -> None:
        self.ball_seat_z = round(self.mug_floor_t + self.ball_r, 4)
        self.dish_capture_r = round(self.dish_inner_r - 0.005, 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("mug_tipout_rehome")
class MugTipoutScene(BaseScene):
    cfg: MugTipoutSceneCfg

    def __init__(self, cfg: MugTipoutSceneCfg | None = None) -> None:
        super().__init__(cfg or MugTipoutSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
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
            "mug": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Mug",
                spawn=_mug_spawner_cfg(
                    inner_r=c.mug_inner_r, wall_t=c.mug_wall_t, h=c.mug_h,
                    floor_t=c.mug_floor_t, mass=c.mug_mass, color=c.mug_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.mug_pos[0], c.mug_pos[1], c.mug_h / 2 + 0.002)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05,
                        angular_damping=0.4,  # poured-ball recipe: settles, never rings
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ball_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.mug_pos[0], c.mug_pos[1], 0.002 + c.ball_seat_z)),
            ),
            "dish_target": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/DishTarget",
                spawn=_dish_spawner_cfg(
                    inner_r=c.dish_inner_r, wall_t=c.dish_wall_t, wall_h=c.dish_wall_h,
                    floor_t=c.dish_floor_t, color=c.target_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.dish_slot_a[0], c.dish_slot_a[1], c.dish_floor_t)),
            ),
            "dish_decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/DishDecoy",
                spawn=_dish_spawner_cfg(
                    inner_r=c.dish_inner_r, wall_t=c.dish_wall_t, wall_h=c.dish_wall_h,
                    floor_t=c.dish_floor_t, color=c.decoy_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.dish_slot_b[0], c.dish_slot_b[1], c.dish_floor_t)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/HomePad",
                spawn=sim_utils.CylinderCfg(
                    radius=c.pad_r, height=c.pad_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.home_pos[0], c.home_pos[1], c.pad_h / 2)),
            ),
        }
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
        self.mug: RigidObject = env.iscene["mug"]
        self.ball: RigidObject = env.iscene["ball"]
        self.dish_t: RigidObject = env.iscene["dish_target"]
        self.dish_d: RigidObject = env.iscene["dish_decoy"]
        self.pad: RigidObject = env.iscene["pad"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        # latches: partial progress survives transient achievements (rubric requirement)
        self._out = torch.zeros(n, dtype=torch.bool, device=env.device)  # ever out of the bore
        self._app_max = torch.zeros(n, device=env.device)  # approach to green dish, running max
        self._in = torch.zeros(n, dtype=torch.bool, device=env.device)  # ever in green dish
        self._home_max = torch.zeros(n, device=env.device)  # mug approach to pad, running max

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: mug upright with the orange seated inside (xy + yaw jitter), the
        green/white dishes randomly ASSIGNED to the two slots (+ per-dish jitter), the home
        pad jittered; clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- mug + seated orange ---
        mx = c.mug_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.mug_jitter
        my = c.mug_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.mug_jitter
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.mug_yaw_deg) / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = mx, my, c.mug_h / 2 + 0.002
        st[:, 3], st[:, 6] = torch.cos(half), torch.sin(half)
        st[:, 0:3] += origin
        self.mug.write_root_state_to_sim(st, env_ids)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = mx, my, 0.002 + c.ball_seat_z
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.ball.write_root_state_to_sim(st, env_ids)

        # --- dish slot assignment (the color-identification knob) + jitter ---
        if c.swap_dishes:
            swap = torch.rand(m, device=dev) < 0.5
        else:
            swap = torch.zeros(m, dtype=torch.bool, device=dev)
        slot_a = torch.tensor(c.dish_slot_a, device=dev).expand(m, 2)
        slot_b = torch.tensor(c.dish_slot_b, device=dev).expand(m, 2)
        t_xy = torch.where(swap.unsqueeze(1), slot_b, slot_a) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.dish_jitter
        d_xy = torch.where(swap.unsqueeze(1), slot_a, slot_b) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.dish_jitter
        for dish, xy in ((self.dish_t, t_xy), (self.dish_d, d_xy)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = c.dish_floor_t
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            dish.write_root_state_to_sim(st, env_ids)

        # --- home pad ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.home_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.home_jitter
        st[:, 1] = c.home_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.home_jitter
        st[:, 2] = c.pad_h / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.pad.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._out[env_ids] = False
        self._app_max[env_ids] = 0.0
        self._in[env_ids] = False
        self._home_max[env_ids] = 0.0

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "mug": self.mug.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "dish_target": self.dish_t.data.root_state_w[env_ids].clone(),
            "dish_decoy": self.dish_d.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "out": self._out[env_ids].clone(),
            "app_max": self._app_max[env_ids].clone(),
            "in": self._in[env_ids].clone(),
            "home_max": self._home_max[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.mug.write_root_state_to_sim(state["mug"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        self.dish_t.write_root_state_to_sim(state["dish_target"], env_ids)
        self.dish_d.write_root_state_to_sim(state["dish_decoy"], env_ids)
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        self._out[env_ids] = state["out"]
        self._app_max[env_ids] = state["app_max"]
        self._in[env_ids] = state["in"]
        self._home_max[env_ids] = state["home_max"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On the ground stand: a tall RED mug (octagonal, "
            f"~{2 * (c.mug_inner_r + c.mug_wall_t) * 100:.1f} cm across, "
            f"{c.mug_h * 100:.1f} cm tall) with a small ORANGE (a "
            f"{2 * c.ball_r * 100:.1f} cm fruit ball) resting at the bottom of its bore; "
            f"two identical shallow dishes (~{2 * c.dish_inner_r * 100:.0f} cm across, "
            f"{c.dish_wall_h * 100:.1f} cm walls) — one GREEN, one WHITE — whose left/right "
            f"positions swap between episodes; and a flat BLUE circular pad "
            f"(~{2 * c.pad_r * 100:.0f} cm across). The mug's bore is only "
            f"{2 * c.mug_inner_r * 100:.1f} cm wide — far too narrow to reach in and grip "
            f"the {2 * c.ball_r * 100:.1f} cm orange directly, so the fruit can only leave "
            f"the mug by tipping the mug and pouring it out.\n"
            f"Goal: the orange must end up resting INSIDE the GREEN dish (the white dish is "
            f"a decoy — fruit poured there fails), and the RED mug must end up standing "
            f"UPRIGHT and EMPTY on the BLUE pad (centred within "
            f"{c.mug_home_xy_tol * 100:.1f} cm, tilted less than "
            f"{c.mug_tilt_max_deg:.0f} deg, at rest). Identify the green dish by its color, "
            f"not its position. Either goal may be completed first; both must hold at the "
            f"end. Leaving the orange inside the mug — even with the mug parked on the pad "
            f"or set into the green dish — scores nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick up the red mug and tip it to pour the orange out into the green dish — "
            "not the white one — then set the mug down upright on the blue pad and leave "
            "it there empty and at rest."
        )

    # ----- progress / rubric ------------------------------------------------------------------------
    def _in_mug(self) -> torch.Tensor:
        """(N,) bool: orange centre inside the mug bore (mug body frame — a tilted or lying
        mug still contains its ball)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        rel = self.ball.data.root_pos_w - self.mug.data.root_pos_w
        loc = quat_apply_inverse(self.mug.data.root_quat_w, rel)
        return (loc[:, :2].norm(dim=-1) < c.mug_inner_r) \
            & (loc[:, 2] > -c.mug_h / 2) & (loc[:, 2] < c.mug_h / 2 + c.ball_r)

    def _in_dish_now(self, dish: RigidObject) -> torch.Tensor:
        """(N,) bool: orange resting inside `dish` — centre within the wall ring, at
        floor-contact height, and NOT inside the mug (the container-cheat clause: a loaded
        mug parked in the dish does not count)."""
        c = self.cfg
        d_xy = (self.ball.data.root_pos_w[:, :2] - dish.data.root_pos_w[:, :2]).norm(dim=-1)
        z_ok = (self.ball.data.root_pos_w[:, 2]
                - (dish.data.root_pos_w[:, 2] + c.ball_r)).abs() < c.ball_z_tol
        return (d_xy < c.dish_capture_r) & z_ok & ~self._in_mug()

    def _mug_home_now(self) -> torch.Tensor:
        """(N,) bool: mug standing upright ON the home pad — centre within
        `mug_home_xy_tol` of the pad axis, base at pad-top height, tilt within
        `mug_tilt_max_deg`, still."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(self.mug.data.root_quat_w, ez)
        upright = up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(c.mug_tilt_max_deg))
        d_xy = (self.mug.data.root_pos_w[:, :2] - self.pad.data.root_pos_w[:, :2]).norm(dim=-1)
        pad_top = self.pad.data.root_pos_w[:, 2] + c.pad_h / 2
        base_z = self.mug.data.root_pos_w[:, 2] - up[:, 2] * c.mug_h / 2
        on_z = (base_z - pad_top).abs() < c.mug_home_z_tol
        still = self.mug.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return upright & (d_xy < c.mug_home_xy_tol) & on_z & still

    def _update_latches(self) -> None:
        """Refresh the latched progress terms from the current physics state."""
        c = self.cfg
        out_now = ~self._in_mug()
        self._out |= out_now
        d_ball = (self.ball.data.root_pos_w[:, :2]
                  - self.dish_t.data.root_pos_w[:, :2]).norm(dim=-1)
        app = (1.0 - d_ball / c.approach_d0).clamp(0.0, 1.0) * self._out.float()
        self._app_max = torch.maximum(self._app_max, app)
        self._in |= self._in_dish_now(self.dish_t)
        d_mug = (self.mug.data.root_pos_w[:, :2]
                 - self.pad.data.root_pos_w[:, :2]).norm(dim=-1)
        home = (1.0 - d_mug / c.approach_d0).clamp(0.0, 1.0) * self._out.float()
        self._home_max = torch.maximum(self._home_max, home)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: orange settled inside the GREEN dish (floor height, inside the wall
        ring, not inside the mug, still) AND the mug standing upright, empty, ON the blue
        home pad (still). Physical outcomes only."""
        self._update_latches()
        ball_still = self.ball.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed
        return self._in_dish_now(self.dish_t) & ball_still & self._mug_home_now()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20*poured-out + 0.30*ball-approach (gated on out) +
        0.20*in-green-dish + 0.15*mug-home approach (gated on out) — all latched, ~0 for
        doing nothing, capped 0.85 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_out * self._out.float() + c.w_approach * self._app_max
                + c.w_in * self._in.float() + c.w_home * self._home_max).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="mug_tipout_rehome", robot="null"))
