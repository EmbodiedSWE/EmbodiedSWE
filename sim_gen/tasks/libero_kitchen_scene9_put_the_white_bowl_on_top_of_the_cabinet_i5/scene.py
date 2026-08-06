"""CounterweightShelfScene — level the tipping shelf with a counterweight, THEN put the
white bowl on top (libero_kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet_i5).

Derived from libero_90 kitchen_scene9 "put the white bowl on top of the cabinet", where
the whole task is one pick-and-place of the bowl onto a static elevated surface. Here
that surface is no longer static: the pedestal's top is a TIPPING SHELF — a tray hinged
along the pedestal's front top edge, front-heavy by construction, resting tipped ~38
degrees forward like a ramp against its lower joint stop. Executing the seed's plan
verbatim (grasp the bowl, set it down on top) works mechanically and fails the task: the
bowl slides straight off the ramp onto the floor (the tested seed-strategy control).

The solver must instead work on the MECHANISM first:

  1. seat the heavy dark-steel counterweight cube into the walled socket that the shelf
     carries BEHIND its hinge — the rear torque swings the shelf up against its level
     stop and pins it there (a binary gravity latch, no fine balancing);
  2. only then pick the white bowl and set it upright on the now-level front platform.

The order is enforced by physics, not by code: with the socket empty the shelf cannot
hold the bowl, and nothing else in the scene can level it (an arm pressing the shelf
level must let go eventually, and success requires a settled, self-supporting state).

Success is judged on the PHYSICAL terminal state: shelf level against its stop and
still, bowl upright, resting on the front platform (judged in the SHELF'S BODY FRAME),
everything settled. Partial credit latches the stages the real solution passes through:
counterweight seated -> shelf leveled -> bowl lifted -> bowl placed.

Mechanism notes (proven robobench cribs):
  - The pedestal is kinematic and NEVER teleported; the tray is one compound rigid body
    on a per-env authored USD revolute joint (bind-time authoring, the microwave/fridge
    door pattern): axis Y, body0 = pedestal, body1 = tray, authored LEVEL = joint zero,
    tipping is NEGATIVE, limits [-tip_deg, 0] so the upper limit IS the level stop. Pair
    collision tray<->pedestal disabled (an authored clearance gap is real clearance).
  - Reset re-poses only the FOLLOWER about the unchanged hinge (pure joint-coordinate
    teleport + 2-substep grace re-pin — the pin-drag-safe move); the randomization that
    matters lives in the bowl/counterweight spawn poses (flank swap + jitter + yaw).
  - The bowl is an octagonal open cup (pen-holder compound-spawner pattern): the thick
    12 mm rim is the pinch-grasp affordance; the cube fits the jaw across its faces.

Everything is procedural. Heavy imports (isaaclab, pxr) are deferred so importing this
module stays app-free.
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


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: the tipping shelf. Body frame: ORIGIN ON THE HINGE LINE (axis =
    local y), front platform extending toward -x with its top face at local z = 0, rear
    socket (floor + 4 walls) behind the hinge toward +x. Authored LEVEL."""
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
    # Damp the counterweight-driven swing so the tray arrives at its level stop gently
    # instead of slamming (the block must stay seated through the swing).
    pxrb.CreateAngularDampingAttr(float(cfg.ang_damping))
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    def _box(path, size, center, color):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        bxf = UsdGeom.Xformable(cube.GetPrim())
        bxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        bxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    pl, pw, pt = cfg.plat_l, cfg.plat_w, cfg.plat_t
    _box(f"{prim_path}/platform", (pl, pw, pt), (-pl / 2, 0.0, -pt / 2), cfg.color)
    # rear socket: floor + 4 walls, inner opening (sock_in_x, sock_in_y), wall top at
    # local z = sock_wall_h
    sx, sy = cfg.sock_in_x, cfg.sock_in_y
    wt, wh = cfg.sock_wall_t, cfg.sock_wall_h
    scx = cfg.sock_cx
    _box(f"{prim_path}/sock_floor", (sx + 2 * wt, sy + 2 * wt, cfg.sock_floor_t),
         (scx, 0.0, -cfg.sock_floor_t / 2), cfg.color)
    for sgn in (-1.0, 1.0):
        _box(f"{prim_path}/sock_wx_{'p' if sgn > 0 else 'n'}",
             (wt, sy + 2 * wt, wh), (scx + sgn * (sx / 2 + wt / 2), 0.0, wh / 2),
             cfg.sock_color)
        _box(f"{prim_path}/sock_wy_{'p' if sgn > 0 else 'n'}",
             (sx, wt, wh), (scx, sgn * (sy / 2 + wt / 2), wh / 2), cfg.sock_color)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: the white bowl — an open octagonal cup (bottom disc + 8 wall
    segments, the pen-holder cup pattern). Body frame: axis = +z (up when upright),
    origin at mid-height."""
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

    n = 8
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


def _tray_spawner_cfg(c: CounterweightShelfSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tray" not in _SPAWNER_CACHE:

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            plat_l: float = 0.175
            plat_w: float = 0.22
            plat_t: float = 0.012
            sock_cx: float = 0.105
            sock_in_x: float = 0.075
            sock_in_y: float = 0.070
            sock_wall_t: float = 0.014
            sock_wall_h: float = 0.028
            sock_floor_t: float = 0.010
            ang_damping: float = 8.0
            color: tuple = (0.55, 0.38, 0.22)
            sock_color: tuple = (0.42, 0.28, 0.16)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["tray"] = TraySpawnerCfg

    return _SPAWNER_CACHE["tray"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.tray_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        plat_l=c.plat_l, plat_w=c.plat_w, plat_t=c.plat_t,
        sock_cx=c.sock_cx, sock_in_x=c.sock_in_x, sock_in_y=c.sock_in_y,
        sock_wall_t=c.sock_wall_t, sock_wall_h=c.sock_wall_h,
        sock_floor_t=c.sock_floor_t, ang_damping=c.tray_ang_damping,
        color=c.tray_color, sock_color=c.sock_color, contact_offset=c.contact_offset,
    )


def _bowl_spawner_cfg(c: CounterweightShelfSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bowl" not in _SPAWNER_CACHE:

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            inner_r: float = 0.045
            wall_t: float = 0.008
            height: float = 0.050
            bot_t: float = 0.010
            color: tuple = (0.95, 0.95, 0.92)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["bowl"] = BowlSpawnerCfg

    return _SPAWNER_CACHE["bowl"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=c.bowl_inner_r, wall_t=c.bowl_wall_t, height=c.bowl_h,
        bot_t=c.bowl_bot_t, color=c.bowl_color, contact_offset=0.002,
    )


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class CounterweightShelfSceneCfg(BaseCfg):
    """Config for `CounterweightShelfScene`. World frame (per env origin): the pedestal
    stands at +x, its front face toward the robot side (-x); the hinge axis runs along y
    at the front top edge; the tray platform extends toward -x (over open air), the
    counterweight socket toward +x (over the pedestal top)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------------
    level_tol_deg: float = tunable(3.0)  # |tray angle| below this counts as LEVEL
    settle_speed: float = tunable(0.05)  # max |lin vel| (bowl, block, tray) when judging (m/s)
    tray_settle_w: float = tunable(0.10)  # max |ang vel| of the tray when judging (rad/s)
    bowl_up_max_deg: float = tunable(20.0)  # bowl axis within this of world-up to count placed
    # bowl root (mid-height) in the tray frame: x band ON the front platform, inset so the
    # bowl footprint stays supported; z band = resting on the platform top (rejects hovering
    # and stacking); |y| bound keeps the footprint on the platform.
    place_x_lo: float = tunable(-0.110)
    place_x_hi: float = tunable(-0.030)
    place_y_abs: float = tunable(0.050)
    place_z_lo: float = tunable(0.012)
    place_z_hi: float = tunable(0.060)
    # block center in the tray frame to count "seated in the socket"
    seat_x_lo: float = tunable(0.060)
    seat_x_hi: float = tunable(0.150)
    seat_y_abs: float = tunable(0.040)
    seat_z_lo: float = tunable(0.005)
    seat_z_hi: float = tunable(0.055)
    lift_z: float = tunable(0.12)  # bowl root height (env z) that latches "lifted"

    # --- tunable: mechanism --------------------------------------------------------------------
    tip_deg: float = tunable(38.0)  # revolute lower limit (tipping is negative)
    reset_tip_lo_deg: float = tunable(31.0)  # sampled initial tray angle |range| (falls to stop)
    reset_tip_hi_deg: float = tunable(38.0)

    # --- tunable: randomization ------------------------------------------------------------------
    spawn_jitter: float = tunable(0.035)  # uniform +/- xy jitter of bowl and block spawns
    bowl_slot: tuple = tunable((0.12, 0.24))  # nominal bowl spawn (x, |y|); flank sign sampled
    block_slot: tuple = tunable((0.14, 0.22))  # nominal block spawn (x, |y|); opposite flank

    # --- info: pedestal / hinge (kinematic, FIXED — never teleported) ----------------------------
    ped_center: tuple = info((0.42, 0.0))
    ped_size: tuple = info((0.24, 0.26, 0.22))  # top at 0.22
    hinge_x: float = info(0.30)  # pedestal front face plane
    hinge_z: float = info(0.240)  # pedestal top + 0.020 clearance
    tray_mass: float = info(0.35)
    tray_ang_damping: float = info(8.0)
    plat_l: float = info(0.175)
    plat_w: float = info(0.22)
    plat_t: float = info(0.012)
    sock_cx: float = info(0.105)  # socket center behind the hinge (tray local +x)
    sock_in_x: float = info(0.075)
    sock_in_y: float = info(0.070)
    sock_wall_t: float = info(0.014)
    sock_wall_h: float = info(0.028)
    sock_floor_t: float = info(0.010)
    tray_color: tuple = info((0.55, 0.38, 0.22))
    sock_color: tuple = info((0.42, 0.28, 0.16))
    ped_color: tuple = info((0.50, 0.50, 0.54))
    contact_offset: float = info(0.003)

    # --- info: movable objects -------------------------------------------------------------------
    block_size: float = info(0.045)  # counterweight cube edge
    block_mass: float = info(0.62)
    block_mu: float = info(0.8)
    block_color: tuple = info((0.16, 0.17, 0.22))  # dark steel
    bowl_inner_r: float = info(0.045)
    bowl_wall_t: float = info(0.012)
    bowl_h: float = info(0.050)
    bowl_bot_t: float = info(0.010)
    bowl_mass: float = info(0.26)
    bowl_color: tuple = info((0.95, 0.95, 0.92))

    # Derived (filled in __post_init__).
    bowl_outer_r: float = field(default=None, init=False)
    ped_top: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.bowl_outer_r = round(self.bowl_inner_r + self.bowl_wall_t, 4)
        self.ped_top = round(self.ped_size[2], 4)


# ----- scene ----------------------------------------------------------------------------------------
@SCENES.register("counterweight_shelf")
class CounterweightShelfScene(BaseScene):
    cfg: CounterweightShelfSceneCfg

    def __init__(self, cfg: CounterweightShelfSceneCfg | None = None) -> None:
        super().__init__(cfg or CounterweightShelfSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        tight = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset,
                                                 rest_offset=0.0)

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
        out["pedestal"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Pedestal",
            spawn=sim_utils.CuboidCfg(
                size=c.ped_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=tight,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ped_color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.ped_center[0], c.ped_center[1], c.ped_size[2] / 2)),
        )
        # tray authored LEVEL (joint zero); reset swings it to the tipped stop
        out["tray"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Tray",
            spawn=_tray_spawner_cfg(c),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(c.hinge_x, 0.0, c.hinge_z)),
        )
        out["block"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Block",
            spawn=sim_utils.CuboidCfg(
                size=(c.block_size, c.block_size, c.block_size),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.block_mass),
                collision_props=tight,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=c.block_mu, dynamic_friction=c.block_mu - 0.1,
                    restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.block_color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.block_slot[0], -c.block_slot[1], c.block_size / 2 + 0.002)),
        )
        out["bowl"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bowl",
            spawn=_bowl_spawner_cfg(c),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.bowl_slot[0], c.bowl_slot[1], c.bowl_h / 2 + 0.002)),
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

    # ----- lifecycle --------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.tray: RigidObject = env.iscene["tray"]
        self.block: RigidObject = env.iscene["block"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.pedestal: RigidObject = env.iscene["pedestal"]
        self.env_origins = env.iscene.env_origins
        self._author_joint()
        # latched progress (post_step)
        self._seated_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._leveled_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._lifted_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._placed_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._theta0 = torch.full((n,), -math.radians(self.cfg.tip_deg), device=dev)
        self._grace = torch.zeros(n, dtype=torch.long, device=dev)

    def _author_joint(self) -> None:
        """Per env: one revolute hinge (axis Y) between the kinematic pedestal and the
        tray, on the pedestal's front top edge. Tray authored level = joint zero;
        tipping (front platform down) rotates NEGATIVE; limits [-tip_deg, 0], so the
        upper limit is the level stop. Pair collision disabled (authored clearance)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        pc = (c.ped_center[0], c.ped_center[1], c.ped_size[2] / 2)
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/tray_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Pedestal"])
            j.CreateBody1Rel().SetTargets([f"{base}/Tray"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(c.hinge_x - pc[0], 0.0 - pc[1],
                                           c.hinge_z - pc[2]))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-c.tip_deg)
            j.CreateUpperLimitAttr(0.0)

    # ----- tray geometry ----------------------------------------------------------------------------
    def tray_angle(self) -> torch.Tensor:
        """(N,) tray hinge angle (rad): 0 = level, tipping NEGATIVE. The hinge admits
        only y-rotation, so the root quat is qy(theta)."""
        q = self.tray.data.root_quat_w
        theta = 2.0 * torch.atan2(q[:, 2], q[:, 0])
        return torch.remainder(theta + math.pi, 2 * math.pi) - math.pi

    def tray_pose_at(self, theta: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """(M,13) tray root state at hinge angle `theta` (zero velocity), world frame.
        The tray root sits ON the hinge line, so only the orientation changes."""
        c = self.cfg
        m = theta.shape[0]
        st = torch.zeros(m, 13, device=theta.device)
        st[:, 0] = c.hinge_x
        st[:, 2] = c.hinge_z
        st[:, 3] = torch.cos(theta / 2)
        st[:, 5] = torch.sin(theta / 2)
        st[:, 0:3] += self.env_origins[env_ids]
        return st

    def tray_level(self) -> torch.Tensor:
        return self.tray_angle().abs() < math.radians(self.cfg.level_tol_deg)

    def tray_still(self) -> torch.Tensor:
        v = self.tray.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed
        w = self.tray.data.root_ang_vel_w.norm(dim=-1) < self.cfg.tray_settle_w
        return v & w

    def _to_tray_frame(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> tray body frame (origin = hinge line)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.tray.data.root_quat_w,
                                  pos_w - self.tray.data.root_pos_w)

    # ----- predicates -------------------------------------------------------------------------------
    def block_seated(self) -> torch.Tensor:
        """(N,) bool, geometric: block center inside the socket volume, tray frame."""
        c = self.cfg
        loc = self._to_tray_frame(self.block.data.root_pos_w)
        return ((loc[:, 0] > c.seat_x_lo) & (loc[:, 0] < c.seat_x_hi)
                & (loc[:, 1].abs() < c.seat_y_abs)
                & (loc[:, 2] > c.seat_z_lo) & (loc[:, 2] < c.seat_z_hi))

    def bowl_upright(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        up = quat_apply(self.bowl.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.bowl_up_max_deg))

    def bowl_on_platform(self) -> torch.Tensor:
        """(N,) bool: bowl root resting on the front platform (tray frame), upright,
        with the tray LEVEL — the physical goal geometry."""
        c = self.cfg
        loc = self._to_tray_frame(self.bowl.data.root_pos_w)
        in_x = (loc[:, 0] > c.place_x_lo) & (loc[:, 0] < c.place_x_hi)
        in_y = loc[:, 1].abs() < c.place_y_abs
        on_z = (loc[:, 2] > c.place_z_lo) & (loc[:, 2] < c.place_z_hi)
        return in_x & in_y & on_z & self.bowl_upright() & self.tray_level()

    def settled(self) -> torch.Tensor:
        """(N,) bool: bowl, block and tray all still."""
        c = self.cfg
        sb = self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        sk = self.block.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return sb & sk & self.tray_still()

    def bowl_lifted(self) -> torch.Tensor:
        z = (self.bowl.data.root_pos_w - self.env_origins)[:, 2]
        return z > self.cfg.lift_z

    # ----- mechanism (every substep) ------------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        # reset grace: re-pin the freshly tipped tray while write timing settles
        gids = (self._grace > 0).nonzero(as_tuple=False).squeeze(-1)
        if len(gids):
            st = self.tray_pose_at(self._theta0[gids], gids)
            self.tray.write_root_state_to_sim(st, gids)
            self._grace[gids] -= 1

        # latch progress (all physical)
        self._seated_ever |= self.block_seated()
        self._leveled_ever |= self.tray_level() & self.block_seated()
        self._lifted_ever |= self.bowl_lifted()
        self._placed_ever |= self.bowl_on_platform()

    # ----- reset --------------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Pedestal stays put (jointed pair, never teleported). Sample: tray tip angle
        (falls to the stop), bowl/block flank sides, xy jitter and yaw."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # tray: pure joint-coordinate teleport of the follower about the unchanged hinge
        th0 = -(c.reset_tip_lo_deg + (c.reset_tip_hi_deg - c.reset_tip_lo_deg)
                * torch.rand(m, device=dev))
        th0 = torch.deg2rad(th0)
        self._theta0[env_ids] = th0
        self.tray.write_root_state_to_sim(self.tray_pose_at(th0, env_ids), env_ids)

        # bowl and block: opposite flanks (sampled sign), xy jitter, free yaw
        flank = torch.where(torch.rand(m, device=dev) < 0.5,
                            torch.ones(m, device=dev), -torch.ones(m, device=dev))
        for body, slot, sgn, rest_z in (
                (self.bowl, c.bowl_slot, flank, c.bowl_h / 2 + 0.002),
                (self.block, c.block_slot, -flank, c.block_size / 2 + 0.002)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = slot[0]
            st[:, 1] = sgn * slot[1]
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.spawn_jitter
            st[:, 2] = rest_z
            half = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        self._seated_ever[env_ids] = False
        self._leveled_ever[env_ids] = False
        self._lifted_ever[env_ids] = False
        self._placed_ever[env_ids] = False
        self._grace[env_ids] = 2

    # ----- state (full, restorable) ---------------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"tray": self.tray, "block": self.block, "bowl": self.bowl}
        return {
            "bodies": {k: b.data.root_state_w[env_ids].clone() for k, b in bodies.items()},
            "theta0": self._theta0[env_ids].clone(),
            "latches": torch.stack([self._seated_ever[env_ids], self._leveled_ever[env_ids],
                                    self._lifted_ever[env_ids], self._placed_ever[env_ids]],
                                   dim=1).clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"tray": self.tray, "block": self.block, "bowl": self.bowl}
        for k, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][k], env_ids)
        self._theta0[env_ids] = state["theta0"]
        lat = state["latches"]
        self._seated_ever[env_ids] = lat[:, 0]
        self._leveled_ever[env_ids] = lat[:, 1]
        self._lifted_ever[env_ids] = lat[:, 2]
        self._placed_ever[env_ids] = lat[:, 3]

    # ----- description ----------------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A gray pedestal (about 22 cm tall) stands on the floor. Its top is a wooden "
            "TIPPING SHELF: a brown tray hinged along the pedestal's front top edge. The "
            "tray's flat front platform extends forward from the hinge; behind the hinge "
            "the tray carries a small dark-brown walled socket. With nothing in the "
            "socket the shelf is front-heavy and hangs tipped forward like a ramp — "
            "anything set on the platform slides straight off. On the floor in front of "
            "the pedestal, one on each side, lie a WHITE BOWL (open cup, about "
            f"{2 * c.bowl_outer_r * 100:.0f} cm wide with a thick 12 mm rim) and a small dark "
            f"steel CUBE (the counterweight, {c.block_size * 100:.1f} cm, heavy).\n"
            "Goal: the white bowl standing upright on the front platform of the shelf "
            "with the shelf LEVEL and everything at rest. The only way to level the "
            "shelf is to put the counterweight cube into the socket behind the hinge — "
            "its weight swings the shelf up against its level stop and holds it there. "
            "Order matters physically: placing the bowl while the shelf is still tipped "
            "just dumps the bowl on the floor. The cube in the socket does NOT count as "
            "the goal; the bowl must rest on the front platform, not in the socket."
        )

    # ----- rubric ---------------------------------------------------------------------------------------
    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched stages of the demonstrated solution:
        0.20 counterweight ever seated in the socket + 0.10 shelf ever leveled (with the
        counterweight seated) + 0.15 bowl ever lifted + 0.45 bowl ever resting on the
        level platform; exactly 1.0 iff success(). Null policy ~0; the seed's plan
        (place the bowl on the tipped shelf) earns only the 0.15 lift credit."""
        s = (0.20 * self._seated_ever.float()
             + 0.10 * self._leveled_ever.float()
             + 0.15 * self._lifted_ever.float()
             + 0.45 * self._placed_ever.float())
        return torch.where(self.success(),
                           torch.ones(self.env.num_envs, device=self.env.device),
                           s.clamp(0.0, 0.95))

    def success(self) -> torch.Tensor:
        """(N,) bool: bowl upright, resting on the front platform of the LEVEL shelf,
        everything settled — the physical terminal state (self-supporting by
        construction: only the seated counterweight can hold the shelf level)."""
        return self.bowl_on_platform() & self.settled()


register_env("simgen", lambda: EnvCfg(scene="counterweight_shelf", robot="null"))
