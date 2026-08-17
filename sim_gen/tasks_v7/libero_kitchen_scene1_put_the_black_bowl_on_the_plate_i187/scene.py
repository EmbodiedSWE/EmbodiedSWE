"""BeamBalanceScene — find the HEAVIEST of three identical black bowls by weighing, and
leave it on the balance plate (libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i187).

Derived from libero_90 kitchen_scene1 "put the black bowl on the plate", where the whole
task is one pick-and-place of a known bowl onto a passive plate. Here the plate is no
longer passive and the bowl is no longer known:

  - The PLATE is the pan of a BEAM BALANCE: a rimmed white plate mounted on one end of a
    pivoting beam whose other end carries a built-in steel counterweight bias. Empty (or
    lightly loaded) the beam rests tilted plate-side-UP against its upper stop.
  - THREE visually IDENTICAL black bowls stand on the ground. Exactly one of them is
    secretly HEAVY (its mass, and which bowl it is, are randomized per episode); the two
    light bowls cannot overcome the counterweight bias, the heavy one always can.

  Goal: the beam fully tipped DOWN against its lower stop, held there by exactly ONE bowl
  standing upright on the plate — which the physics permits only for the heavy bowl —
  with the other two bowls at rest on the ground, off the balance.

STRATEGIC DIFFERENCE from the seed (and from the corpus): the core of the task is not
placement but MEASUREMENT — an information-gathering plan. The solver cannot see which
bowl is correct; it must use the balance as an instrument (put a bowl on the plate, watch
whether the beam drops, remove it and try the next if not). The rubric never reads the
hidden masses: the beam reaching its down-stop IS the mass check, physically. A blind
seed-style pick-and-place succeeds only by 1-in-3 luck; stacking extra bowls to fake the
weight is rejected by the exactly-one-bowl clause.

Mechanism notes (proven corpus cribs):
  - The pivot column is kinematic and NEVER teleported (safe joint anchor); the beam is
    one compound rigid body on a per-env authored USD revolute joint (axis X, limits
    +/- tilt_limit_deg — the two hard stops). Pair collision disabled AND geometric
    clearance kept between beam and column at all angles.
  - The counterweight bias is authored EXPLICITLY: MassAPI mass + CenterOfMassAttr +
    diagonal inertia on the beam root (compound roots leave the CoM at the origin
    otherwise, and the balance would have no bias at all).
  - Bowl masses are written per episode through root_physx_view.set_masses and verified
    by READBACK (custom spawners silently ignore cfg mass_props).
  - Reset re-poses only the FOLLOWER beam about the unchanged pivot (joint-coordinate
    teleport + 2-substep grace re-pin).

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


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: the balance beam. Body frame: ORIGIN ON THE PIVOT AXIS (axis =
    local x), bar running along y; the rimmed white plate (the pan) on the +y end, the
    visible steel counterweight block on the -y end. Authored LEVEL. The counterweight
    BIAS is authored explicitly (mass + CoM + inertia on the root — compound roots leave
    the CoM at the origin otherwise and the balance would have no bias)."""
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
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(cfg.beam_mass))
    # THE bias: CoM offset toward the counterweight end. Explicit, or the compound root
    # CoM sits at the pivot and the balance never restores.
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, float(cfg.com_y), 0.0))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in cfg.inertia]))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    # jointFriction is inert on plain USD joints — angular damping settles the swing.
    pxrb.CreateAngularDampingAttr(float(cfg.ang_damping))
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    def _box(path, size, center, color):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        bxf = UsdGeom.Xformable(cube.GetPrim())
        bxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        bxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        collide(cube.GetPrim())

    # bar along y
    _box(f"{prim_path}/bar", (cfg.bar_w, 2 * cfg.half_len, cfg.bar_t),
         (0.0, 0.0, 0.0), cfg.bar_color)
    # counterweight block riding the -y end (visual identity of the bias)
    _box(f"{prim_path}/counterweight", (cfg.cw_size, cfg.cw_size, cfg.cw_size),
         (0.0, -cfg.arm_len, cfg.bar_t / 2 + cfg.cw_size / 2), cfg.cw_color)
    # the plate: disc on the +y end, top at local z = bar_t/2 + plate_t
    plate = UsdGeom.Cylinder.Define(stage, f"{prim_path}/plate")
    plate.CreateRadiusAttr(float(cfg.plate_r))
    plate.CreateHeightAttr(float(cfg.plate_t))
    plate.CreateExtentAttr([Gf.Vec3f(-cfg.plate_r, -cfg.plate_r, -cfg.plate_t / 2),
                            Gf.Vec3f(cfg.plate_r, cfg.plate_r, cfg.plate_t / 2)])
    UsdGeom.Xformable(plate.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, float(cfg.arm_len), cfg.bar_t / 2 + cfg.plate_t / 2))
    plate.CreateDisplayColorAttr([Gf.Vec3f(*cfg.plate_color)])
    collide(plate.GetPrim())
    # raised rim around the plate edge (keeps a bowl aboard while the beam tilts)
    n = int(cfg.rim_n)
    rim_r = cfg.plate_r - cfg.rim_t / 2 - 0.001
    seg_len = 2 * math.pi * rim_r / n + 0.004
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/rim_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(rim_r * math.cos(ang),
                                          cfg.arm_len + rim_r * math.sin(ang),
                                          cfg.bar_t / 2 + cfg.plate_t + cfg.rim_h / 2))
        sxf.AddRotateZOp().Set(math.degrees(ang) + 90.0)
        sxf.AddScaleOp().Set(Gf.Vec3f(seg_len, cfg.rim_t, cfg.rim_h))
        seg.CreateDisplayColorAttr([Gf.Vec3f(*cfg.plate_color)])
        collide(seg.GetPrim())
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: a black bowl — an open ten-sided cup (bottom disc + wall
    segments). Body frame: axis = +z (up when upright), origin at mid-height. The rim
    wall is the pinch-grasp affordance for a parallel jaw."""
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
    # spawn-time mass is a placeholder; reset() writes the episode masses through the
    # physx view (and smoke asserts the readback)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass0))
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

    n = 10
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


def _beam_spawner_cfg(c: BeamBalanceSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "beam" not in _SPAWNER_CACHE:

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            half_len: float = 0.25
            arm_len: float = 0.20
            bar_w: float = 0.045
            bar_t: float = 0.014
            plate_r: float = 0.078
            plate_t: float = 0.010
            rim_h: float = 0.018
            rim_t: float = 0.007
            rim_n: int = 10
            cw_size: float = 0.055
            beam_mass: float = 1.0
            com_y: float = -0.086
            inertia: tuple = (0.028, 0.004, 0.028)
            ang_damping: float = 3.0
            bar_color: tuple = (0.55, 0.40, 0.24)
            cw_color: tuple = (0.20, 0.21, 0.26)
            plate_color: tuple = (0.93, 0.93, 0.90)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["beam"] = BeamSpawnerCfg

    return _SPAWNER_CACHE["beam"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.beam_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        half_len=c.beam_half_len, arm_len=c.arm_len, bar_w=c.bar_w, bar_t=c.bar_t,
        plate_r=c.plate_r, plate_t=c.plate_t, rim_h=c.rim_h, rim_t=c.rim_t,
        rim_n=c.rim_n, cw_size=c.cw_size, beam_mass=c.beam_mass, com_y=c.beam_com_y,
        inertia=c.beam_inertia, ang_damping=c.beam_ang_damping,
        bar_color=c.bar_color, cw_color=c.cw_color, plate_color=c.plate_color,
        contact_offset=c.contact_offset,
    )


def _bowl_spawner_cfg(c: BeamBalanceSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bowl" not in _SPAWNER_CACHE:

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            inner_r: float = 0.042
            wall_t: float = 0.010
            height: float = 0.048
            bot_t: float = 0.010
            mass0: float = 0.26
            color: tuple = (0.10, 0.10, 0.12)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["bowl"] = BowlSpawnerCfg

    return _SPAWNER_CACHE["bowl"](
        mass_props=sim_utils.MassPropertiesCfg(mass=0.26),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=c.bowl_inner_r, wall_t=c.bowl_wall_t, height=c.bowl_h,
        bot_t=c.bowl_bot_t, mass0=0.26, color=c.bowl_color, contact_offset=0.002,
    )


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class BeamBalanceSceneCfg(BaseCfg):
    """Config for `BeamBalanceScene`. World frame (per env origin): the balance column
    stands at +x, the beam runs along y — plate pan on the +y arm, counterweight block on
    the -y arm; the pivot axis is x. Beam angle: 0 = level, PAN UP is POSITIVE; the
    joint's hard stops sit at +/- tilt_limit_deg.

    Bias calibration (authored CoM): tau_bias = beam_mass*g*|com_y| = 1.0*9.81*0.086
    = 0.84 N*m = an equivalent pan load of 0.43 kg at arm_len 0.20. Light bowls
    (<= 0.30 kg) leave >= 0.26 N*m of net PAN-UP torque; the heavy bowl (>= 0.60 kg)
    wins with >= 0.33 N*m of net PAN-DOWN torque. The margins hold at every angle
    (all lever arms scale together with cos(theta))."""

    # --- tunable: rubric thresholds ----------------------------------------------------------------
    down_tol_deg: float = tunable(6.0)  # beam angle <= -this counts as fully tipped DOWN
    up_ref_deg: float = tunable(5.5)  # beam angle >= +this = resting pan-up (reference)
    pan_xy_tol: float = tunable(0.050)  # bowl center within this of the pan center (beam frame)
    pan_z_lo: float = tunable(0.028)  # bowl center z band above the pivot plane (beam frame;
    pan_z_hi: float = tunable(0.075)  # rest is ~0.041) — rejects hovering and stacking
    bowl_up_max_deg: float = tunable(30.0)  # pan bowl axis within this of world-up
    ground_z_max: float = tunable(0.10)  # a bowl below this is "on the ground, off the balance"
    settle_speed: float = tunable(0.05)  # max |lin vel| of every bowl when judging (m/s)
    beam_settle_w: float = tunable(0.10)  # max |ang vel| of the beam when judging (rad/s)
    lift_z: float = tunable(0.14)  # bowl root height that latches "lifted"

    # --- tunable: mechanism ------------------------------------------------------------------------
    tilt_limit_deg: float = tunable(8.0)  # revolute limits: +/- this (the two hard stops)
    reset_up_lo_deg: float = tunable(6.5)  # sampled initial beam angle (falls onto the up stop)
    reset_up_hi_deg: float = tunable(8.0)

    # --- tunable: randomization --------------------------------------------------------------------
    spawn_jitter: float = tunable(0.030)  # uniform +/- xy jitter of each bowl spawn
    heavy_mass_lo: float = tunable(0.60)  # the one heavy bowl (kg)
    heavy_mass_hi: float = tunable(0.75)
    light_mass_lo: float = tunable(0.22)  # the two light bowls (kg)
    light_mass_hi: float = tunable(0.30)

    # --- info: balance (column kinematic, FIXED — never teleported) --------------------------------
    col_center: tuple = info((0.38, 0.0))
    col_size: tuple = info((0.10, 0.10, 0.18))
    pivot_z: float = info(0.22)  # column top + 0.04 swing clearance
    beam_half_len: float = info(0.25)
    arm_len: float = info(0.20)  # pivot -> pan center / counterweight center
    bar_w: float = info(0.045)
    bar_t: float = info(0.014)
    plate_r: float = info(0.078)
    plate_t: float = info(0.010)
    rim_h: float = info(0.018)
    rim_t: float = info(0.007)
    rim_n: int = info(10)
    cw_size: float = info(0.055)
    beam_mass: float = info(1.0)
    beam_com_y: float = info(-0.086)  # the authored bias (see class docstring)
    beam_inertia: tuple = info((0.028, 0.004, 0.028))
    beam_ang_damping: float = info(3.0)
    bar_color: tuple = info((0.55, 0.40, 0.24))
    cw_color: tuple = info((0.20, 0.21, 0.26))
    plate_color: tuple = info((0.93, 0.93, 0.90))
    col_color: tuple = info((0.48, 0.48, 0.52))
    contact_offset: float = info(0.003)

    # --- info: bowls -------------------------------------------------------------------------------
    bowl_inner_r: float = info(0.042)
    bowl_wall_t: float = info(0.010)
    bowl_h: float = info(0.048)
    bowl_bot_t: float = info(0.010)
    bowl_color: tuple = info((0.10, 0.10, 0.12))
    bowl_slots: tuple = info(((0.08, -0.18), (0.11, 0.01), (0.08, 0.20)))

    # Derived (filled in __post_init__).
    bowl_outer_r: float = field(default=None, init=False)
    plate_top_z: float = field(default=None, init=False)  # beam-local z of the plate top

    def __post_init__(self) -> None:
        self.bowl_outer_r = round(self.bowl_inner_r + self.bowl_wall_t, 4)
        self.plate_top_z = round(self.bar_t / 2 + self.plate_t, 4)
        # equivalent bias load must separate the sampled mass bands with margin
        eq = self.beam_mass * abs(self.beam_com_y) / self.arm_len
        assert self.light_mass_hi + 0.10 <= eq <= self.heavy_mass_lo - 0.10, (
            f"bias calibration broken: equivalent pan load {eq:.3f} kg must sit between "
            f"light_hi {self.light_mass_hi} and heavy_lo {self.heavy_mass_lo} with margin")
        # beam bar must clear the column top at full tilt (joint pair is collision-
        # filtered, but the filter is unreliable for overlapping compound children)
        drop = (self.col_size[0] / 2 + 0.01) * math.sin(math.radians(self.tilt_limit_deg))
        assert self.pivot_z - self.bar_t / 2 - drop > self.col_size[2] + 0.010, (
            "beam bar would graze the column top at full tilt")


# ----- scene ----------------------------------------------------------------------------------------
@SCENES.register("beam_balance")
class BeamBalanceScene(BaseScene):
    cfg: BeamBalanceSceneCfg

    N_BOWLS = 3

    def __init__(self, cfg: BeamBalanceSceneCfg | None = None) -> None:
        super().__init__(cfg or BeamBalanceSceneCfg())

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
        out["column"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Column",
            spawn=sim_utils.CuboidCfg(
                size=c.col_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=tight,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.col_color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.col_center[0], c.col_center[1], c.col_size[2] / 2)),
        )
        # beam authored LEVEL (joint zero); reset swings it onto the pan-up stop
        out["beam"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Beam",
            spawn=_beam_spawner_cfg(c),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.col_center[0], c.col_center[1], c.pivot_z)),
        )
        for k in range(self.N_BOWLS):
            sx, sy = c.bowl_slots[k]
            out[f"bowl_{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl_" + str(k),
                spawn=_bowl_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx, sy, c.bowl_h / 2 + 0.002)),
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
        self.beam: RigidObject = env.iscene["beam"]
        self.column: RigidObject = env.iscene["column"]
        self.bowls: list[RigidObject] = [env.iscene[f"bowl_{k}"] for k in range(self.N_BOWLS)]
        self.env_origins = env.iscene.env_origins
        self._author_joint()
        # episode mass table (written to physx at reset; the RUBRIC never reads it —
        # it exists for randomization bookkeeping and the solve/smoke oracles)
        self._masses = torch.zeros(n, self.N_BOWLS, device=dev)
        # latched progress (post_step)
        self._lift_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._weigh_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._tip_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._theta0 = torch.full((n,), math.radians(self.cfg.tilt_limit_deg), device=dev)
        self._grace = torch.zeros(n, dtype=torch.long, device=dev)

    def _author_joint(self) -> None:
        """Per env: one revolute pivot (axis X) between the kinematic column (never
        teleported — the anchor stays valid) and the beam. Beam authored level = joint
        zero; pan-up is POSITIVE; limits [-tilt, +tilt] are the two hard stops. Pair
        collision disabled (geometric clearance is also kept at all angles)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        cc = (c.col_center[0], c.col_center[1], c.col_size[2] / 2)
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/beam_pivot")
            j.CreateBody0Rel().SetTargets([f"{base}/Column"])
            j.CreateBody1Rel().SetTargets([f"{base}/Beam"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.pivot_z - cc[2]))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-c.tilt_limit_deg)
            j.CreateUpperLimitAttr(c.tilt_limit_deg)

    # ----- beam geometry --------------------------------------------------------------------------
    def beam_angle(self) -> torch.Tensor:
        """(N,) beam pivot angle (rad): 0 = level, PAN UP positive. The pivot admits
        only x-rotation, so the root quat is qx(theta)."""
        q = self.beam.data.root_quat_w
        theta = 2.0 * torch.atan2(q[:, 1], q[:, 0])
        return torch.remainder(theta + math.pi, 2 * math.pi) - math.pi

    def beam_pose_at(self, theta: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """(M,13) beam root state at pivot angle `theta` (zero velocity), world frame.
        The beam root sits ON the pivot axis, so only the orientation changes."""
        c = self.cfg
        m = theta.shape[0]
        st = torch.zeros(m, 13, device=theta.device)
        st[:, 0] = c.col_center[0]
        st[:, 1] = c.col_center[1]
        st[:, 2] = c.pivot_z
        st[:, 3] = torch.cos(theta / 2)
        st[:, 4] = torch.sin(theta / 2)
        st[:, 0:3] += self.env_origins[env_ids]
        return st

    def beam_down(self) -> torch.Tensor:
        """(N,) bool: beam tipped fully DOWN (pan side at/near its lower stop)."""
        return self.beam_angle() <= -math.radians(self.cfg.down_tol_deg)

    def beam_up(self) -> torch.Tensor:
        """(N,) bool: beam resting pan-side-UP (its unloaded/underloaded rest)."""
        return self.beam_angle() >= math.radians(self.cfg.up_ref_deg)

    def beam_still(self) -> torch.Tensor:
        v = self.beam.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed
        w = self.beam.data.root_ang_vel_w.norm(dim=-1) < self.cfg.beam_settle_w
        return v & w

    def _to_beam_frame(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> beam body frame (origin = pivot axis)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.beam.data.root_quat_w,
                                  pos_w - self.beam.data.root_pos_w)

    def _bowl_pos_w(self) -> torch.Tensor:
        """(N,3,3) bowl root positions, world frame."""
        return torch.stack([b.data.root_pos_w for b in self.bowls], dim=1)

    # ----- predicates -----------------------------------------------------------------------------
    def on_pan(self) -> torch.Tensor:
        """(N,3) bool, geometric: bowl center inside the pan region, BEAM frame — within
        `pan_xy_tol` of the pan center and inside the resting z band (rejects hovering
        above and stacking a second bowl into the first)."""
        c = self.cfg
        n = self.env.num_envs
        out = torch.zeros(n, self.N_BOWLS, dtype=torch.bool, device=self.env.device)
        for k, b in enumerate(self.bowls):
            loc = self._to_beam_frame(b.data.root_pos_w)
            dx = loc[:, 0]
            dy = loc[:, 1] - c.arm_len
            near = (dx * dx + dy * dy).sqrt() < c.pan_xy_tol
            z_ok = (loc[:, 2] > c.pan_z_lo) & (loc[:, 2] < c.pan_z_hi)
            out[:, k] = near & z_ok
        return out

    def bowls_upright(self) -> torch.Tensor:
        """(N,3) bool: bowl axis within `bowl_up_max_deg` of world-up."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        cos_max = math.cos(math.radians(self.cfg.bowl_up_max_deg))
        cols = []
        for b in self.bowls:
            up = quat_apply(b.data.root_quat_w, ez)
            cols.append(up[:, 2].clamp(-1.0, 1.0) >= cos_max)
        return torch.stack(cols, dim=1)

    def grounded(self) -> torch.Tensor:
        """(N,3) bool: bowl down on the ground (NOT anywhere on the balance)."""
        z = self._bowl_pos_w()[:, :, 2] - self.env_origins[:, None, 2]
        return z < self.cfg.ground_z_max

    def bowls_still(self) -> torch.Tensor:
        """(N,3) bool: bowl |lin vel| below `settle_speed`."""
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.bowls], dim=1)
        return v < self.cfg.settle_speed

    def settled(self) -> torch.Tensor:
        """(N,) bool: every bowl and the beam still."""
        return self.bowls_still().all(dim=1) & self.beam_still()

    def heavy_index(self) -> torch.Tensor:
        """(N,) episode bookkeeping: which bowl carries the heavy mass. NOT used by the
        rubric (the beam is the mass check); exists for the solve/smoke oracles."""
        return self._masses.argmax(dim=1)

    # ----- mechanism (every substep) --------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        # reset grace: re-pin the freshly posed beam while write timing settles
        gids = (self._grace > 0).nonzero(as_tuple=False).squeeze(-1)
        if len(gids):
            st = self.beam_pose_at(self._theta0[gids], gids)
            self.beam.write_root_state_to_sim(st, gids)
            self._grace[gids] -= 1

        # latch progress (all physical)
        onp = self.on_pan()
        z = self._bowl_pos_w()[:, :, 2] - self.env_origins[:, None, 2]
        self._lift_ever |= (z > self.cfg.lift_z).any(dim=1)
        self._weigh_ever |= onp.any(dim=1)
        self._tip_ever |= self.beam_down() & (onp.sum(dim=1) == 1)

    # ----- reset ----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Column stays put (jointed pair, never teleported). Sample: which bowl is
        heavy + both mass values (written through the physx view), a random permutation
        of the three ground slots, xy jitter, free yaw, and the initial beam angle
        (falls onto the pan-up stop)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # beam: pure joint-coordinate teleport of the follower about the unchanged pivot
        th0 = c.reset_up_lo_deg + (c.reset_up_hi_deg - c.reset_up_lo_deg) * torch.rand(m, device=dev)
        th0 = torch.deg2rad(th0)
        self._theta0[env_ids] = th0
        self.beam.write_root_state_to_sim(self.beam_pose_at(th0, env_ids), env_ids)

        # masses: one heavy bowl (torch.rand argmax — first-randint-after-seed is
        # degenerate on this stack), two light, values sampled per episode
        heavy = torch.rand(m, self.N_BOWLS, device=dev).argmax(dim=1)
        masses = c.light_mass_lo + (c.light_mass_hi - c.light_mass_lo) * torch.rand(
            m, self.N_BOWLS, device=dev)
        hv = c.heavy_mass_lo + (c.heavy_mass_hi - c.heavy_mass_lo) * torch.rand(m, device=dev)
        masses[torch.arange(m, device=dev), heavy] = hv
        self._masses[env_ids] = masses
        ids_cpu = env_ids.detach().cpu()
        for k, b in enumerate(self.bowls):
            full = b.root_physx_view.get_masses().clone()
            full.reshape(-1)[ids_cpu] = masses[:, k].detach().cpu()
            b.root_physx_view.set_masses(full, ids_cpu)

        # bowls: random slot permutation + jitter + free yaw, resting on the ground
        perm = torch.rand(m, self.N_BOWLS, device=dev).argsort(dim=1)
        slots = torch.tensor(c.bowl_slots, device=dev)  # (3,2)
        for k, b in enumerate(self.bowls):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = slots[perm[:, k]]
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.spawn_jitter
            st[:, 2] = c.bowl_h / 2 + 0.002
            half = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            b.write_root_state_to_sim(st, env_ids)

        self._lift_ever[env_ids] = False
        self._weigh_ever[env_ids] = False
        self._tip_ever[env_ids] = False
        self._grace[env_ids] = 2

    # ----- state (full, restorable) ---------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "bowls": [b.data.root_state_w[env_ids].clone() for b in self.bowls],
            "masses": self._masses[env_ids].clone(),
            "theta0": self._theta0[env_ids].clone(),
            "latches": torch.stack([self._lift_ever[env_ids], self._weigh_ever[env_ids],
                                    self._tip_ever[env_ids]], dim=1).clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.beam.write_root_state_to_sim(state["beam"], env_ids)
        for k, b in enumerate(self.bowls):
            b.write_root_state_to_sim(state["bowls"][k], env_ids)
        self._masses[env_ids] = state["masses"]
        ids_cpu = env_ids.detach().cpu()
        for k, b in enumerate(self.bowls):
            full = b.root_physx_view.get_masses().clone()
            full.reshape(-1)[ids_cpu] = state["masses"][:, k].detach().cpu()
            b.root_physx_view.set_masses(full, ids_cpu)
        self._theta0[env_ids] = state["theta0"]
        lat = state["latches"]
        self._lift_ever[env_ids] = lat[:, 0]
        self._weigh_ever[env_ids] = lat[:, 1]
        self._tip_ever[env_ids] = lat[:, 2]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A BEAM BALANCE stands on the floor: a gray column (about 18 cm tall) carrying "
            "a pivoting brown beam. One end of the beam holds a round WHITE PLATE with a "
            "small raised rim (the pan); the other end carries a fixed dark steel "
            "counterweight block. With nothing on the plate the counterweight wins and the "
            "beam rests tilted PLATE-SIDE-UP against its stop; the beam can swing about "
            f"{c.tilt_limit_deg:.0f} degrees each way between its two hard stops.\n"
            f"On the ground in front of the balance stand THREE IDENTICAL BLACK BOWLS "
            f"(open cups, about {2 * c.bowl_outer_r * 100:.0f} cm wide, {c.bowl_h * 100:.0f} cm "
            "tall, with a 10 mm thick rim wall). They look exactly alike, but exactly ONE "
            "of them is much heavier than the other two — which one varies and cannot be "
            "seen. Only the heavy bowl is heavy enough to overcome the counterweight; a "
            "light bowl on the plate leaves the beam resting plate-side-up.\n"
            "Goal: find the heavy bowl and leave it standing upright on the balance plate "
            "with the beam tipped fully DOWN against its lower stop, the other two bowls "
            "at rest on the ground clear of the balance, and everything still. Use the "
            "balance itself to tell the bowls apart: set a bowl on the plate and watch the "
            "beam — if it stays up, that bowl is light; take it off, set it back on the "
            "ground and try another. Exactly one bowl may be on the balance at the end "
            "(piling extra bowls onto the plate does not count), and a bowl left lying "
            "anywhere else on the balance also does not count."
        )

    def instruction(self) -> str:
        return (
            "Exactly one of the three identical black bowls is heavy. Find it by weighing "
            "bowls on the balance plate, and leave only the heavy bowl upright on the "
            "plate so the beam tips fully down; the other two bowls must end up back on "
            "the ground off the balance."
        )

    # ----- rubric ---------------------------------------------------------------------------------
    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched stages of the demonstrated solution:
        0.10 any bowl ever lifted + 0.15 any bowl ever weighed on the pan + 0.35 the
        beam ever driven to its down stop by a single panned bowl; exactly 1.0 iff
        success(). Null policy ~0; weighing a light bowl (the seed's blind plan, 2 of 3
        times) tops out at 0.25 until the heavy bowl is found."""
        s = (0.10 * self._lift_ever.float()
             + 0.15 * self._weigh_ever.float()
             + 0.35 * self._tip_ever.float())
        return torch.where(self.success(),
                           torch.ones(self.env.num_envs, device=self.env.device),
                           s.clamp(0.0, 0.90))

    def success(self) -> torch.Tensor:
        """(N,) bool, all physical: beam at its DOWN stop and still; exactly ONE bowl on
        the pan, upright; both other bowls at rest on the ground (not anywhere on the
        balance); every body settled. The rubric never reads the hidden masses — only
        the heavy bowl can physically hold the beam down through the settle window."""
        onp = self.on_pan()
        one = onp.sum(dim=1) == 1
        pan_upright = (onp & self.bowls_upright()).sum(dim=1) == 1
        placed_ok = (onp | self.grounded()).all(dim=1)
        return self.beam_down() & one & pan_upright & placed_ok & self.settled()


register_env("simgen", lambda: EnvCfg(scene="beam_balance", robot="null"))
