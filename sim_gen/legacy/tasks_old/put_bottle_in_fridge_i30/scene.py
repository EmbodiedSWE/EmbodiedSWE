"""DecantAndChillScene — the bottle does NOT fit in the mini-fridge: decant its contents
into the jar that does, stow the loaded jar inside, recycle the bottle.

Derived from rlbench "put_bottle_in_fridge", but STRATEGICALLY DIFFERENT: the seed is a
prehensile transport task — grasp the bottle, carry it, release it inside the fridge; the
bottle itself is the goal object and the fridge is just a destination volume. Here that
plan is MEASURED-IMPOSSIBLE: the bottle (a 260 mm tall open tube holding 3-5 loose
"vitamin" balls) is LONGER than the fridge chamber's interior space diagonal (~246 mm),
so no orientation of the bottle can ever be contained — sliding it in through the front
opening leaves ~110 mm protruding (the tested negative control). The correct plan is a
REPACKAGING plan the seed never needs: pour/transfer the bottle's contents into the squat
jar that DOES pass the front opening, carry the loaded jar THROUGH the opening (a real
physical transit — see the latch below), and dispose of the now-useless bottle on the
recycling pad. The graspable seed object becomes a means (a source container to empty),
not the end.

Judged on physical outcome only, strategy-agnostic:
  - a ball is DELIVERED when it rests settled inside the jar's cup volume and NOT inside
    the bottle (so nesting the whole bottle in the jar delivers nothing — the
    container-cheat clause);
  - the jar is STOWED when it stands upright, settled, fully inside the fridge interior,
    AND its centre once crossed the opening plane inward through the aperture with a
    small per-step displacement (the anti-teleport transit latch: a kinematic teleport
    through the wall never latches);
  - success() = every present ball delivered + jar stowed + bottle resting on the
    recycling pad (both ends over the pad, low, settled).
Rubric: 0.05 latched once the bottle is ever lifted, + 0.45 * delivered fraction,
+ 0.30 * delivered fraction once the jar is stowed (an EMPTY stowed jar pays zero),
0.95 when all present balls are delivered AND the jar is stowed, 1.0 iff success.
Doing nothing scores exactly 0. Required order: decanting must precede stowing — the
roofed chamber blocks pouring into a jar already inside (the opening is lower than the
interior ceiling and the bottle cannot enter tilted); bottle-to-pad may interleave.

Per-episode randomization: fridge pose (xy jitter + FULL yaw — the opening heading must
be read per episode), bottle / jar / pad poses (xy jitter + yaw), and ball-count subset
sampling (3-5 present; absent balls park in an off-camera depot). All assets procedural:
bottle and jar are compound rigid bodies (bottom disc + 8 octagon wall boxes — the
pen_holder spawner pattern), the fridge a kinematic compound chamber (floor, roof, back,
sides, front jambs + header framing the opening), the pad a kinematic slab.

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
# `isaaclab.sim.utils.clone` is borrowed. Fresh Define per prim -> xformOps authored once
# (idempotent under clone).

_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_cup(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author an open octagonal cup/tube at `prim_path`: root Xform with RigidBodyAPI +
    explicit MassAPI, a bottom cylinder collider and 8 box wall segments of inner inradius
    `inner_r`, open at +z. Used for BOTH the tall bottle and the squat jar. Depenetration
    capped at 0.5 m/s (balls rattling in a carried vessel must shed energy, not pop) and
    lightly damped so set-downs settle promptly."""
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


def _spawn_fridge(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the mini-fridge at `prim_path`: a KINEMATIC rigid body (re-placed per episode
    by root-state writes). Body frame: interior centre; interior floor top at -ih/2, ceiling
    at +ih/2; the front face opens at local +x with an `ow` x `oh` doorway rising from the
    interior floor (two jambs + a header frame it). Explicit small contact offsets — the
    jar clears the doorway by ~12 mm per side."""
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

    def box(name: str, size, center, color, offset=None) -> None:
        b = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        b.CreateSizeAttr(1.0)
        bxf = UsdGeom.Xformable(b.GetPrim())
        bxf.AddTranslateOp().Set(Gf.Vec3d(*center))
        bxf.AddScaleOp().Set(Gf.Vec3f(*size))
        b.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(b.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(b.GetPrim())
        px.CreateContactOffsetAttr(float(cfg.contact_offset if offset is None else offset))
        px.CreateRestOffsetAttr(0.0)

    ix, iy, ih = cfg.inner_x, cfg.inner_y, cfg.inner_h
    t, rt, ft = cfg.wall_t, cfg.roof_t, cfg.floor_t
    ow, oh = cfg.open_w, cfg.open_h
    ox, oy = ix + 2 * t, iy + 2 * t
    body_c, roof_c = cfg.color, cfg.accent_color
    jw = (oy - ow) / 2  # jamb width

    # Interior floor takes drop-in impacts: slightly generous speculative offset there.
    box("floor", (ox, oy, ft), (0.0, 0.0, -ih / 2 - ft / 2), body_c,
        offset=cfg.floor_contact_offset)
    box("roof", (ox, oy, rt), (0.0, 0.0, ih / 2 + rt / 2), roof_c)
    box("back", (t, oy, ih), (-(ix / 2 + t / 2), 0.0, 0.0), body_c)
    box("side_l", (ox, t, ih), (0.0, iy / 2 + t / 2, 0.0), body_c)
    box("side_r", (ox, t, ih), (0.0, -(iy / 2 + t / 2), 0.0), body_c)
    box("jamb_l", (t, jw, ih), (ix / 2 + t / 2, ow / 2 + jw / 2, 0.0), roof_c)
    box("jamb_r", (t, jw, ih), (ix / 2 + t / 2, -(ow / 2 + jw / 2), 0.0), roof_c)
    box("header", (t, ow, ih - oh), (ix / 2 + t / 2, 0.0, oh - ih / 2 + (ih - oh) / 2), roof_c)
    return root


def _cup_spawner_cfg(key: str, *, inner_r: float, wall_t: float, height: float, bot_t: float,
                     mass: float, color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cup" not in _SPAWNER_CACHE:

        @configclass
        class CupSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cup)
            inner_r: float = 0.037
            wall_t: float = 0.007
            height: float = 0.085
            bot_t: float = 0.012
            color: tuple = (0.9, 0.9, 0.9)
            n_segments: int = 8
            contact_offset: float = 0.002

        _SPAWNER_CACHE["cup"] = CupSpawnerCfg

    return _SPAWNER_CACHE["cup"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=inner_r, wall_t=wall_t, height=height, bot_t=bot_t,
        color=color, n_segments=8, contact_offset=contact_offset,
    )


def _fridge_spawner_cfg(*, inner_x: float, inner_y: float, inner_h: float, wall_t: float,
                        roof_t: float, floor_t: float, open_w: float, open_h: float,
                        color: tuple, accent_color: tuple, contact_offset: float,
                        floor_contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "fridge" not in _SPAWNER_CACHE:

        @configclass
        class FridgeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_fridge)
            inner_x: float = 0.15
            inner_y: float = 0.15
            inner_h: float = 0.125
            wall_t: float = 0.012
            roof_t: float = 0.012
            floor_t: float = 0.010
            open_w: float = 0.120
            open_h: float = 0.110
            color: tuple = (0.88, 0.90, 0.92)
            accent_color: tuple = (0.35, 0.45, 0.60)
            contact_offset: float = 0.002
            floor_contact_offset: float = 0.003

        _SPAWNER_CACHE["fridge"] = FridgeSpawnerCfg

    return _SPAWNER_CACHE["fridge"](
        mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        inner_x=inner_x, inner_y=inner_y, inner_h=inner_h, wall_t=wall_t, roof_t=roof_t,
        floor_t=floor_t, open_w=open_w, open_h=open_h, color=color,
        accent_color=accent_color, contact_offset=contact_offset,
        floor_contact_offset=floor_contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DecantAndChillSceneCfg(BaseCfg):
    """Config for `DecantAndChillScene`. The impossibility of the seed plan is asserted BY
    CONSTRUCTION in __post_init__: the bottle is longer than the fridge interior's space
    diagonal, so no orientation of the bottle fits inside."""

    # --- tunable: rubric thresholds -----------------------------------------------------------
    lift_height: float = tunable(0.05)  # bottle centre this far above rest -> lift latch (m)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging a body (m/s)
    stow_tilt_deg: float = tunable(15.0)  # jar "upright" gate inside the fridge
    stow_margin: float = tunable(0.003)  # extra slack on full-containment of the jar (m)
    transit_max_step: float = tunable(0.04)  # per-step displacement cap for the transit latch
    pad_z_max: float = tunable(0.20)  # bottle ends must rest below this over the pad (m)

    # --- tunable: randomization (the task-family knobs) ---------------------------------------
    reset_pos_jitter: float = tunable(0.03)  # uniform +/- xy jitter (all bodies)
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per body at reset (fridge: FULL)
    subset_sample: bool = tunable(True)  # per-episode ball-count sampling
    min_present: int = tunable(3)  # lower bound of sampled ball count

    # --- tunable: placement --------------------------------------------------------------------
    fridge_pos: tuple = tunable((0.32, 0.08))  # fridge centre on the ground
    bottle_pos: tuple = tunable((-0.14, 0.14))  # bottle spawn (upright, full)
    jar_pos: tuple = tunable((-0.06, -0.16))  # jar spawn (upright, empty)
    pad_pos: tuple = tunable((0.14, -0.52))  # recycling pad centre (clear of any stow path)

    # --- info: bottle (tall open tube — the seed object, repurposed as a source) ----------------
    bottle_inner_r: float = info(0.024)  # inner octagon inradius (balls r14 slide freely)
    bottle_wall_t: float = info(0.006)
    bottle_h: float = info(0.260)  # LONGER than the fridge interior diagonal — the crux
    bottle_bot_t: float = info(0.010)
    bottle_mass: float = info(0.10)
    bottle_color: tuple = info((0.20, 0.45, 0.20))  # green glass

    # --- info: jar (squat cup — passes the doorway) ---------------------------------------------
    jar_inner_r: float = info(0.037)
    jar_wall_t: float = info(0.007)
    jar_h: float = info(0.085)
    jar_bot_t: float = info(0.012)  # thick floor: no CCD on GPU PhysX — anti-tunneling
    jar_mass: float = info(0.12)
    jar_color: tuple = info((0.85, 0.75, 0.35))  # amber

    # --- info: balls -----------------------------------------------------------------------------
    n_balls: int = info(5)
    ball_r: float = info(0.014)
    ball_mass: float = info(0.025)
    ball_colors: tuple = info(((0.85, 0.20, 0.20), (0.95, 0.55, 0.10), (0.90, 0.85, 0.15),
                               (0.45, 0.20, 0.70), (0.20, 0.55, 0.85)))

    # --- info: fridge chamber (kinematic) --------------------------------------------------------
    fridge_inner_x: float = info(0.150)  # interior depth (front face at local +x)
    fridge_inner_y: float = info(0.150)  # interior width
    fridge_inner_h: float = info(0.125)  # interior height
    fridge_wall_t: float = info(0.012)
    fridge_roof_t: float = info(0.012)
    fridge_floor_t: float = info(0.010)
    open_w: float = info(0.120)  # doorway width (local y)
    open_h: float = info(0.110)  # doorway height from the interior floor
    fridge_color: tuple = info((0.88, 0.90, 0.92))
    fridge_accent: tuple = info((0.35, 0.45, 0.60))

    # --- info: recycling pad (kinematic slab) ----------------------------------------------------
    pad_size: tuple = info((0.36, 0.18, 0.008))
    pad_color: tuple = info((0.20, 0.55, 0.30))
    parking_pos: tuple = info((1.0, 1.0))  # off-camera ground depot for absent balls

    # Derived (filled in __post_init__).
    bottle_outer_r: float = field(default=None, init=False)  # circumradius (widest half-width)
    jar_outer_r: float = field(default=None, init=False)  # circumradius
    fridge_root_z: float = field(default=None, init=False)  # root height when on the ground
    open_plane_x: float = field(default=None, init=False)  # doorway plane, fridge frame

    def __post_init__(self) -> None:
        cos8 = math.cos(math.pi / 8)
        self.bottle_outer_r = round((self.bottle_inner_r + self.bottle_wall_t) / cos8, 4)
        self.jar_outer_r = round((self.jar_inner_r + self.jar_wall_t) / cos8, 4)
        self.fridge_root_z = round(self.fridge_inner_h / 2 + self.fridge_floor_t, 4)
        self.open_plane_x = round(self.fridge_inner_x / 2 + self.fridge_wall_t / 2, 4)
        # The strategic crux, honest BY CONSTRUCTION:
        diag = math.sqrt(self.fridge_inner_x ** 2 + self.fridge_inner_y ** 2
                         + self.fridge_inner_h ** 2)
        assert self.bottle_h > diag + 0.010, "bottle must NOT fit the fridge in any orientation"
        # ... while the jar passes the doorway with real margin, and the bottle can at least
        # ENTER the doorway lying down (so the seed-strategy control is physical, not a wall).
        assert 2 * self.jar_outer_r <= self.open_w - 0.020, "jar must pass the doorway width"
        assert self.jar_h <= self.open_h - 0.020, "jar must pass the doorway height"
        assert 2 * self.jar_outer_r <= min(self.fridge_inner_x, self.fridge_inner_y) - 0.040, \
            "jar must fit fully inside with containment margin"
        assert 2 * self.bottle_outer_r <= self.open_h - 0.020, \
            "the bottle must enter the doorway lying down (seed control is a jam, not a wall)"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("decant_and_chill")
class DecantAndChillScene(BaseScene):
    cfg: DecantAndChillSceneCfg

    def __init__(self, cfg: DecantAndChillSceneCfg | None = None) -> None:
        super().__init__(cfg or DecantAndChillSceneCfg())

    # ----- assets ------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic fridge + pad fixtures, the free bottle (full of
        balls) and jar (empty) — nominal poses; reset() re-places everything."""
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

        out["fridge"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Fridge",
            spawn=_fridge_spawner_cfg(
                inner_x=c.fridge_inner_x, inner_y=c.fridge_inner_y, inner_h=c.fridge_inner_h,
                wall_t=c.fridge_wall_t, roof_t=c.fridge_roof_t, floor_t=c.fridge_floor_t,
                open_w=c.open_w, open_h=c.open_h, color=c.fridge_color,
                accent_color=c.fridge_accent, contact_offset=0.002, floor_contact_offset=0.003,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.fridge_pos[0], c.fridge_pos[1], c.fridge_root_z)),
        )

        out["pad"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Pad",
            spawn=sim_utils.CuboidCfg(
                size=c.pad_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002,
                                                                 rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.pad_pos[0], c.pad_pos[1], c.pad_size[2] / 2)),
        )

        out["bottle"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bottle",
            spawn=_cup_spawner_cfg(
                "bottle", inner_r=c.bottle_inner_r, wall_t=c.bottle_wall_t, height=c.bottle_h,
                bot_t=c.bottle_bot_t, mass=c.bottle_mass, color=c.bottle_color,
                contact_offset=0.002,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.bottle_pos[0], c.bottle_pos[1], c.bottle_h / 2 + 0.002)),
        )

        out["jar"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Jar",
            spawn=_cup_spawner_cfg(
                "jar", inner_r=c.jar_inner_r, wall_t=c.jar_wall_t, height=c.jar_h,
                bot_t=c.jar_bot_t, mass=c.jar_mass, color=c.jar_color, contact_offset=0.002,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.jar_pos[0], c.jar_pos[1], c.jar_h / 2 + 0.002)),
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
                    pos=(c.bottle_pos[0], c.bottle_pos[1],
                         c.bottle_bot_t + c.ball_r + 0.006 + i * (2 * c.ball_r + 0.003))),
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
        self.fridge: RigidObject = env.iscene["fridge"]
        self.pad: RigidObject = env.iscene["pad"]
        self.bottle: RigidObject = env.iscene["bottle"]
        self.jar: RigidObject = env.iscene["jar"]
        self.balls: dict[str, RigidObject] = {
            f"ball_{i}": env.iscene[f"ball_{i}"] for i in range(c.n_balls)}
        self.env_origins = env.iscene.env_origins
        # present[e, i]: ball i participates in episode e (sampled at reset; judged subset).
        self.present = torch.ones(env.num_envs, c.n_balls, dtype=torch.bool, device=env.device)
        # lifted[e]: the bottle ever left the ground (transient latch, pays 0.05).
        self.lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # transited[e]: the jar centre crossed the doorway plane inward, inside the aperture,
        # with a small per-step displacement — the anti-teleport latch.
        self.transited = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._prev_jar_loc = torch.full((env.num_envs, 3), float("nan"), device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the present ball subset, place fridge / pad / bottle / jar
        with xy jitter + yaw, seat present balls INSIDE the standing bottle (they settle onto
        its floor), park absent balls in the depot, clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        yaw_amp = math.radians(c.reset_yaw_deg)

        self.lifted[env_ids] = False
        self.transited[env_ids] = False
        self._prev_jar_loc[env_ids] = float("nan")

        # --- subset sampling: k ~ U{min_present..n_balls} present balls ---
        if c.subset_sample:
            k = torch.randint(c.min_present, c.n_balls + 1, (m,), device=dev)
        else:
            k = torch.full((m,), c.n_balls, dtype=torch.long, device=dev)
        rank = torch.rand(m, c.n_balls, device=dev).argsort(dim=1).argsort(dim=1)
        self.present[env_ids] = rank < k.unsqueeze(1)

        def yawed_state(base_xy: tuple, z: float, jitter: float) -> torch.Tensor:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = base_xy[0]
            st[:, 1] = base_xy[1]
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * jitter
            st[:, 2] = z
            yaw = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            return st

        # --- fixtures (kinematic, re-placed each episode) ---
        self.fridge.write_root_state_to_sim(
            yawed_state(c.fridge_pos, c.fridge_root_z, c.reset_pos_jitter), env_ids)
        self.pad.write_root_state_to_sim(
            yawed_state(c.pad_pos, c.pad_size[2] / 2, c.reset_pos_jitter), env_ids)

        # --- jar (empty, upright) ---
        self.jar.write_root_state_to_sim(
            yawed_state(c.jar_pos, c.jar_h / 2 + 0.002, c.reset_pos_jitter), env_ids)

        # --- bottle (upright, full) + its ball column ---
        bottle_st = yawed_state(c.bottle_pos, c.bottle_h / 2 + 0.002, c.reset_pos_jitter)
        self.bottle.write_root_state_to_sim(bottle_st, env_ids)
        for i, ball in enumerate(self.balls.values()):
            seat = torch.zeros(m, 3, device=dev)
            ang = 2.4 * i  # small alternating offset so the column stacks stably
            seat[:, 0] = bottle_st[:, 0] - origin[:, 0] + 0.004 * math.cos(ang)
            seat[:, 1] = bottle_st[:, 1] - origin[:, 1] + 0.004 * math.sin(ang)
            seat[:, 2] = 0.002 + c.bottle_bot_t + c.ball_r + 0.004 + i * (2 * c.ball_r + 0.003)
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
        """Latch achievements at sim rate: the bottle lift, and the jar's doorway TRANSIT —
        the anti-teleport clause. The transit latches only when the jar centre crosses the
        doorway plane inward (fridge frame +x -> interior) while inside the aperture band,
        moving less than `transit_max_step` in one step. A kinematic teleport into the
        chamber never satisfies this; a physical carry/slide through the doorway does."""
        c = self.cfg
        z = self.bottle.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        self.lifted |= z > c.bottle_h / 2 + c.lift_height

        loc = self._local_to(self.fridge, self.jar.data.root_pos_w.unsqueeze(1))[:, 0]
        prev = self._prev_jar_loc
        valid = torch.isfinite(prev).all(dim=-1)
        in_ap = (loc[:, 1].abs() < c.open_w / 2) & \
                (loc[:, 2] < -c.fridge_inner_h / 2 + c.open_h)
        prev_ap = (prev[:, 1].abs() < c.open_w / 2) & \
                  (prev[:, 2] < -c.fridge_inner_h / 2 + c.open_h)
        crossed = (prev[:, 0] > c.fridge_inner_x / 2) & (loc[:, 0] <= c.fridge_inner_x / 2)
        small = (loc - prev).norm(dim=-1) < c.transit_max_step
        self.transited |= valid & crossed & prev_ap & in_ap & small
        self._prev_jar_loc = loc.clone()

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "fridge": self.fridge.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "bottle": self.bottle.data.root_state_w[env_ids].clone(),
            "jar": self.jar.data.root_state_w[env_ids].clone(),
            "balls": {n: b.data.root_state_w[env_ids].clone() for n, b in self.balls.items()},
            "present": self.present[env_ids].clone(),
            "lifted": self.lifted[env_ids].clone(),
            "transited": self.transited[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.fridge.write_root_state_to_sim(state["fridge"], env_ids)
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        self.bottle.write_root_state_to_sim(state["bottle"], env_ids)
        self.jar.write_root_state_to_sim(state["jar"], env_ids)
        for n, b in self.balls.items():
            b.write_root_state_to_sim(state["balls"][n], env_ids)
        self.present[env_ids] = state["present"]
        self.lifted[env_ids] = state["lifted"]
        self.transited[env_ids] = state["transited"]
        self._prev_jar_loc[env_ids] = float("nan")

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A mini-fridge chamber (interior {c.fridge_inner_x * 1000:.0f} x "
            f"{c.fridge_inner_y * 1000:.0f} x {c.fridge_inner_h * 1000:.0f} mm, one open "
            f"doorway {c.open_w * 1000:.0f} x {c.open_h * 1000:.0f} mm in its front) stands on "
            f"the floor at a random heading. A tall green bottle ({c.bottle_h * 1000:.0f} mm, "
            f"open top) holds 3-{c.n_balls} loose colored balls ({2 * c.ball_r * 1000:.0f} mm) "
            f"— count what you see. Nearby: an empty amber jar "
            f"({2 * c.jar_outer_r * 1000:.0f} mm wide, {c.jar_h * 1000:.0f} mm tall) and a "
            f"green recycling pad.\n"
            f"Goal: the balls must end up chilled INSIDE the fridge — but the bottle is longer "
            f"than the chamber's interior diagonal and can never fit. Transfer every ball into "
            f"the jar (pour, or move them one by one), carry the loaded jar in through the "
            f"doorway and leave it standing upright inside, and lay the empty bottle on the "
            f"recycling pad. Balls loose on the floor, a jar left at the threshold, or the "
            f"bottle jammed half-in the doorway do not count. Only settled final states are "
            f"judged; the jar must really pass through the doorway (no shortcuts through the "
            f"walls). Transfer the balls BEFORE stowing the jar — the chamber roof blocks "
            f"pouring into a jar that is already inside."
        )

    # ----- predicates / rubric --------------------------------------------------------------------
    def _ball_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _body_up(self, body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)

    def in_jar(self) -> torch.Tensor:
        """(N,B) bool: ball centre inside the jar's cup volume (jar frame)."""
        c = self.cfg
        loc = self._local_to(self.jar, self._ball_tensors()[0])
        r_ok = loc[:, :, :2].norm(dim=-1) < c.jar_inner_r + 0.005
        z_ok = (loc[:, :, 2] > -c.jar_h / 2 - 0.005) & (loc[:, :, 2] < c.jar_h / 2 + c.ball_r)
        return r_ok & z_ok

    def in_bottle(self) -> torch.Tensor:
        """(N,B) bool: ball centre inside the bottle's tube volume (bottle frame) — the
        clause that makes 'stand the whole full bottle in the jar' deliver nothing."""
        c = self.cfg
        loc = self._local_to(self.bottle, self._ball_tensors()[0])
        r_ok = loc[:, :, :2].norm(dim=-1) < c.bottle_inner_r + 0.005
        z_ok = (loc[:, :, 2] > -c.bottle_h / 2 - 0.005) & \
               (loc[:, :, 2] < c.bottle_h / 2 + c.ball_r)
        return r_ok & z_ok

    def ball_settled(self) -> torch.Tensor:
        return self._ball_tensors()[1] < self.cfg.settle_speed

    def delivered(self) -> torch.Tensor:
        """(N,B) bool: present, settled inside the jar, and OUT of the bottle."""
        return self.present & self.in_jar() & ~self.in_bottle() & self.ball_settled()

    def all_delivered(self) -> torch.Tensor:
        return (self.delivered() | ~self.present).all(dim=1)

    def jar_inside(self) -> torch.Tensor:
        """(N,) bool geometric leg: jar centre such that the jar is FULLY inside the
        interior (fridge frame), resting near the interior floor, upright."""
        c = self.cfg
        loc = self._local_to(self.fridge, self.jar.data.root_pos_w.unsqueeze(1))[:, 0]
        m_x = c.fridge_inner_x / 2 - c.jar_outer_r + c.stow_margin
        m_y = c.fridge_inner_y / 2 - c.jar_outer_r + c.stow_margin
        xy_ok = (loc[:, 0].abs() < m_x) & (loc[:, 1].abs() < m_y)
        z_lo = -c.fridge_inner_h / 2
        z_ok = (loc[:, 2] > z_lo - 0.005) & (loc[:, 2] < z_lo + c.jar_h / 2 + 0.025)
        upright = self._body_up(self.jar)[:, 2].clamp(-1.0, 1.0) >= \
            math.cos(math.radians(c.stow_tilt_deg))
        return xy_ok & z_ok & upright

    def jar_stowed(self) -> torch.Tensor:
        """(N,) bool: jar fully inside + settled + it PHYSICALLY transited the doorway."""
        still = self.jar.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed
        return self.jar_inside() & still & self.transited

    def _bottle_ends(self) -> torch.Tensor:
        """(N,2,3) world positions of the bottle's two axis endpoints."""
        axis = self._body_up(self.bottle)
        p = self.bottle.data.root_pos_w
        h = self.cfg.bottle_h / 2
        return torch.stack([p - axis * h, p + axis * h], dim=1)

    def bottle_in_fridge(self) -> torch.Tensor:
        """(N,) bool: BOTH bottle endpoints inside the interior — by construction this can
        never be true (the interior diagonal is shorter than the bottle)."""
        c = self.cfg
        loc = self._local_to(self.fridge, self._bottle_ends())
        inside = (loc[:, :, 0].abs() < c.fridge_inner_x / 2) & \
                 (loc[:, :, 1].abs() < c.fridge_inner_y / 2) & \
                 (loc[:, :, 2].abs() < c.fridge_inner_h / 2 + 0.01)
        return inside.all(dim=1)

    def bottle_on_pad(self) -> torch.Tensor:
        """(N,) bool: both bottle endpoints over the pad footprint, resting low, settled,
        and the bottle centre not inside the fridge interior region."""
        c = self.cfg
        ends = self._bottle_ends()
        loc = self._local_to(self.pad, ends)
        xy_ok = ((loc[:, :, 0].abs() < c.pad_size[0] / 2) &
                 (loc[:, :, 1].abs() < c.pad_size[1] / 2)).all(dim=1)
        low = ((ends[:, :, 2] - self.env_origins[:, None, 2]) < c.pad_z_max).all(dim=1)
        still = self.bottle.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        ctr = self._local_to(self.fridge, self.bottle.data.root_pos_w.unsqueeze(1))[:, 0]
        in_int = (ctr[:, 0].abs() < c.fridge_inner_x / 2 + c.fridge_wall_t) & \
                 (ctr[:, 1].abs() < c.fridge_inner_y / 2 + c.fridge_wall_t) & \
                 (ctr[:, 2].abs() < c.fridge_inner_h / 2 + c.fridge_roof_t)
        return xy_ok & low & still & ~in_int

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.05 bottle-lift latch + 0.45 * delivered fraction
        + 0.30 * delivered fraction once the jar is stowed (an empty stowed jar pays 0);
        0.95 when ALL present balls are delivered and the jar is stowed; 1.0 iff success
        (+ bottle on the recycling pad). ~0 for doing nothing."""
        k = self.delivered().sum(dim=1).float()
        tot = self.present.sum(dim=1).clamp(min=1).float()
        frac = k / tot
        stowed = self.jar_stowed()
        s = 0.05 * self.lifted.float() + 0.45 * frac + 0.30 * stowed.float() * frac
        done = self.all_delivered() & stowed
        s = torch.where(done, torch.full_like(s, 0.95), s)
        return torch.where(done & self.bottle_on_pad(), torch.ones_like(s), s)

    def success(self) -> torch.Tensor:
        """(N,) bool: every present ball settled in the jar, the jar stowed inside the
        fridge (having physically passed the doorway), the bottle resting on the pad."""
        return self.all_delivered() & self.jar_stowed() & self.bottle_on_pad()


# Scene-level env binding (robot embodiments are a later stage).
register_env("simgen", lambda: EnvCfg(scene="decant_and_chill", robot="null", env_spacing=3))
