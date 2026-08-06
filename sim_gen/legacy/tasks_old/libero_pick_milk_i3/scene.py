"""MilkUprightScene — stand the fallen milk carton upright on the tabletop.

Derived from the LIBERO seed `libero_pick_milk` ("pick the milk and place it in the
basket") but STRATEGICALLY INVERTED: nothing is picked up and carried anywhere, and no
container is the goal. The milk carton starts FALLEN — lying on one of its four side
faces — and the task is to REORIENT it: pivot it back up so it stands on its base,
directly on the work surface, and stays standing. A raised-floor open crate sits nearby
as seed-bait: executing the seed's plan (put the carton in the container) leaves the
carton's base ~35 mm above the surface and FAILS the base-height gate, so a solver that
replays the seed strategy scores partial credit at best. The carton's top is a gabled
roof (two slanted panels meeting at a ridge), so standing it upside-down is physically
impossible — it balances on a knife edge and topples; the "which end is the base" clause
is honest by construction, not by rubric fiat.

Judged on PHYSICAL outcome:
  - upright: carton body +z (roof direction) within `tilt_tol_deg` of world-up;
  - on the table: the carton's bottom face within `base_z_tol` of the work surface
    (rejects: inside the crate — raised floor; on the crate rim; on the butter box);
  - settled: carton linear AND angular velocity below the settle gates;
  - scene intact: the standing juice bottle must still be upright (`bottle_up`) —
    knocking the scene apart while flailing at the carton is not a solve.

Graded score (latched transient achievements, 1.0 iff success):
  0.25 once the carton's long axis has been raised past 30 deg from horizontal,
  0.50 past 60 deg, 0.75 once it has been within the upright tolerance anywhere
  (e.g. held upright in the air), 1.0 iff success(). If the juice bottle is currently
  knocked over the score is capped at `bottle_down_cap` until the bottle stands again.

Assets are fully procedural, one rigid body each, authored by custom compound spawners
(the pen_holder pattern — child colliders of one body never self-collide):
  - carton: dark base plinth + white body box + two blue roof panels forming the gable
    (panels pulled inward by half their thickness so the outer surfaces lie exactly in
    the ideal roof planes — lying on a side face rests flat, no eave-corner prop);
  - crate: bottom slab (THICK: 35 mm — the raised floor that defeats the seed strategy)
    + 4 walls;
  - juice bottle: plain cylinder; butter box: plain cuboid (inert clutter).

Per-episode randomization: which of the 4 side faces the carton lies on (roll about its
long axis), free yaw, xy jitter — plus xy/yaw jitter on the crate, bottle and butter, so
a memorized fixed pivot direction fails.

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


def _spawn_carton(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the milk carton at `prim_path`: root Xform with RigidBodyAPI + explicit
    MassAPI, a dark base plinth box, a white body box and two slanted blue roof panels
    forming a gable ridge along local +x. Local frame: +z points from base to roof;
    origin at the centre of the FULL height (plinth bottom at -H/2, ridge at +H/2)."""
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
    # Contact-solver pop cap + a whiff of damping so the released carton crosses the
    # settle gates promptly instead of ringing (the pen_holder precedent).
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    # NEVER let the carton sleep: a body authored at zero velocity in an unstable pose
    # (leaning, or balanced on the roof ridge) would be put to sleep by PhysX and freeze
    # there forever — run 1 measured exactly that (an upside-down carton "balancing"
    # indefinitely on its knife edge). The rubric's physics must stay live.
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    def box(name: str, size, center, color, rot_x_deg: float = 0.0) -> None:
        b = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        b.CreateSizeAttr(1.0)
        bxf = UsdGeom.Xformable(b.GetPrim())
        bxf.AddTranslateOp().Set(Gf.Vec3d(*center))
        if rot_x_deg:
            bxf.AddRotateXOp().Set(rot_x_deg)
        bxf.AddScaleOp().Set(Gf.Vec3f(*size))
        b.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(b.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(b.GetPrim())
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    w = cfg.body_w
    pitch = math.radians(cfg.roof_pitch_deg)
    roof_h = (w / 2) * math.tan(pitch)
    slant = (w / 2) / math.cos(pitch)
    H = cfg.plinth_t + cfg.body_h + roof_h
    z_bot = -H / 2
    z_top = z_bot + cfg.plinth_t + cfg.body_h  # body top = roof eave level

    # base plinth (dark — the visible "this end down" marker) + body (white)
    box("plinth", (w, w, cfg.plinth_t), (0.0, 0.0, z_bot + cfg.plinth_t / 2), cfg.plinth_color)
    box("body", (w, w, cfg.body_h), (0.0, 0.0, z_bot + cfg.plinth_t + cfg.body_h / 2),
        cfg.body_color)
    # two roof panels, ridge along local +x. Ideal midplane runs eave (y=+/-w/2, z_top)
    # -> ridge (y=0, z_top+roof_h); centre the PANEL half its thickness INSIDE that
    # plane so the outer surface is exactly the ideal roof plane (no eave-corner
    # protruding past the body side face — a lying carton must rest flat).
    t = cfg.roof_t
    sp, cp = math.sin(pitch), math.cos(pitch)
    for sgn in (1.0, -1.0):
        cy = sgn * (w / 4 - (t / 2) * sp)
        cz = z_top + roof_h / 2 - (t / 2) * cp
        # ridge overlap kept to 1 mm: a wider overlap makes the ridge a blunt flat strip
        # an inverted carton can balance on (run 2 measured it) — it must be a knife edge
        box(f"roof_{'p' if sgn > 0 else 'n'}", (w, slant + 0.001, t), (0.0, cy, cz),
            cfg.roof_color, rot_x_deg=-sgn * cfg.roof_pitch_deg)
    return root


def _spawn_crate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the open crate at `prim_path`: THICK bottom slab (raised inner floor — the
    honesty device that defeats put-it-in-the-container) + 4 full-height walls. Origin
    at the centre of the total height."""
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
    PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateMaxDepenetrationVelocityAttr(0.5)

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

    inner, wt, h, bt = cfg.inner_w, cfg.wall_t, cfg.height, cfg.bottom_t
    outer = inner + 2 * wt
    box("bottom", (inner, inner, bt), (0.0, 0.0, -h / 2 + bt / 2))
    for k, (dx, dy, sx, sy) in enumerate((
            (1, 0, wt, outer), (-1, 0, wt, outer), (0, 1, outer, wt), (0, -1, outer, wt))):
        box(f"wall_{k}", (sx, sy, h),
            (dx * (inner / 2 + wt / 2), dy * (inner / 2 + wt / 2), 0.0))
    return root


def _carton_spawner_cfg(c: MilkUprightSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "carton" not in _SPAWNER_CACHE:

        @configclass
        class CartonSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_carton)
            body_w: float = 0.066
            body_h: float = 0.148
            plinth_t: float = 0.012
            roof_pitch_deg: float = 40.0
            roof_t: float = 0.007
            body_color: tuple = (0.93, 0.93, 0.95)
            plinth_color: tuple = (0.15, 0.15, 0.18)
            roof_color: tuple = (0.20, 0.35, 0.85)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["carton"] = CartonSpawnerCfg

    return _SPAWNER_CACHE["carton"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.carton_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        body_w=c.body_w, body_h=c.body_h, plinth_t=c.plinth_t,
        roof_pitch_deg=c.roof_pitch_deg, roof_t=c.roof_t,
        body_color=c.body_color, plinth_color=c.plinth_color, roof_color=c.roof_color,
        contact_offset=c.contact_offset,
    )


def _crate_spawner_cfg(c: MilkUprightSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "crate" not in _SPAWNER_CACHE:

        @configclass
        class CrateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_crate)
            inner_w: float = 0.110
            wall_t: float = 0.010
            height: float = 0.110
            bottom_t: float = 0.035
            color: tuple = (0.55, 0.42, 0.28)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["crate"] = CrateSpawnerCfg

    return _SPAWNER_CACHE["crate"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.crate_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_w=c.crate_inner_w, wall_t=c.crate_wall_t, height=c.crate_h,
        bottom_t=c.crate_bottom_t, color=c.crate_color, contact_offset=c.contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class MilkUprightSceneCfg(BaseCfg):
    """Config for `MilkUprightScene`."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    tilt_tol_deg: float = tunable(12.0)  # carton roof-axis within this of world-up = "upright"
    # Bottom face within this of the work surface = "on the table". The crate's inner
    # floor sits `crate_bottom_t` (35 mm) above the surface, > 2x this tolerance, so the
    # seed strategy (carton in the container) is rejected by construction.
    base_z_tol: float = tunable(0.015)
    settle_lin: float = tunable(0.05)  # max carton |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.5)  # max carton |ang vel| when judging (rad/s)
    bottle_tilt_max_deg: float = tunable(30.0)  # juice bottle "still standing" gate
    bottle_down_cap: float = tunable(0.4)  # score cap while the bottle is knocked over

    # --- tunable: randomization ----------------------------------------------------------------
    reset_pos_jitter: float = tunable(0.05)  # uniform +/- xy jitter per body at reset
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw at reset
    randomize_face: bool = tunable(True)  # sample WHICH side face the carton lies on

    # --- tunable: placement --------------------------------------------------------------------
    surface_z: float = tunable(0.0)  # work-surface height; 0 = on the ground (null smoke)
    carton_pos: tuple = tunable((0.0, 0.0))
    bottle_pos: tuple = tunable((0.22, 0.18))
    crate_pos: tuple = tunable((0.26, -0.20))
    butter_pos: tuple = tunable((-0.20, 0.20))

    # --- info: structure -----------------------------------------------------------------------
    body_w: float = info(0.066)  # square cross-section side
    body_h: float = info(0.148)
    plinth_t: float = info(0.012)  # dark base segment (the "this end down" marker)
    roof_pitch_deg: float = info(40.0)  # gable pitch; upside-down = knife-edge ridge
    roof_t: float = info(0.007)
    carton_mass: float = info(0.15)
    body_color: tuple = info((0.93, 0.93, 0.95))
    plinth_color: tuple = info((0.15, 0.15, 0.18))
    roof_color: tuple = info((0.20, 0.35, 0.85))
    crate_inner_w: float = info(0.110)  # carton diagonal 93 mm < 110 mm: it FITS (bait works)
    crate_wall_t: float = info(0.010)
    crate_h: float = info(0.110)
    crate_bottom_t: float = info(0.035)  # raised floor = 2.3x base_z_tol (seed-strategy honesty)
    crate_mass: float = info(0.50)
    crate_color: tuple = info((0.55, 0.42, 0.28))
    bottle_r: float = info(0.030)
    bottle_h: float = info(0.140)
    bottle_mass: float = info(0.10)
    bottle_color: tuple = info((0.95, 0.55, 0.15))
    butter_size: tuple = info((0.080, 0.050, 0.030))
    butter_mass: float = info(0.05)
    butter_color: tuple = info((0.92, 0.85, 0.35))
    contact_offset: float = info(0.002)
    milestones: tuple = info((0.25, 0.50, 0.75))  # latched partial credit (see module doc)
    bench_size: tuple = info((1.1, 0.9))  # procedural bench top, used when surface_z > 0

    # Derived (filled in __post_init__).
    roof_h: float = field(default=None, init=False)
    carton_H: float = field(default=None, init=False)  # full height, plinth bottom -> ridge

    def __post_init__(self) -> None:
        self.roof_h = round((self.body_w / 2) * math.tan(math.radians(self.roof_pitch_deg)), 5)
        self.carton_H = round(self.plinth_t + self.body_h + self.roof_h, 5)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("milk_upright")
class MilkUprightScene(BaseScene):
    cfg: MilkUprightSceneCfg

    def __init__(self, cfg: MilkUprightSceneCfg | None = None) -> None:
        super().__init__(cfg or MilkUprightSceneCfg())

    # ----- assets -------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
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
        if z0 > 0:  # procedural workbench: kinematic slab, top at surface_z
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

        # fallen carton (reset() re-places it lying on a random face)
        out["carton"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Carton",
            spawn=_carton_spawner_cfg(c),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.carton_pos[0], c.carton_pos[1], z0 + c.body_w / 2 + 0.004),
                rot=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0),  # lying
            ),
        )
        out["crate"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Crate",
            spawn=_crate_spawner_cfg(c),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.crate_pos[0], c.crate_pos[1], z0 + c.crate_h / 2 + 0.002)),
        )
        out["bottle"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bottle",
            spawn=sim_utils.CylinderCfg(
                radius=c.bottle_r, height=c.bottle_h,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.bottle_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.bottle_color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.bottle_pos[0], c.bottle_pos[1], z0 + c.bottle_h / 2 + 0.002)),
        )
        out["butter"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Butter",
            spawn=sim_utils.CuboidCfg(
                size=c.butter_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.butter_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.butter_color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.butter_pos[0], c.butter_pos[1], z0 + c.butter_size[2] / 2 + 0.002)),
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

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.carton: RigidObject = env.iscene["carton"]
        self.crate: RigidObject = env.iscene["crate"]
        self.bottle: RigidObject = env.iscene["bottle"]
        self.butter: RigidObject = env.iscene["butter"]
        self.env_origins = env.iscene.env_origins
        # latched milestone flags per env: [past 30 deg, past 60 deg, was-upright]
        self.m = torch.zeros(env.num_envs, 3, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: carton lying on a RANDOM side face (roll about its long axis)
        with free yaw + xy jitter; crate / bottle / butter re-jittered; latches cleared."""
        from isaaclab.utils.math import quat_mul

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        yaw_amp = math.radians(c.reset_yaw_deg)

        def qz(half: torch.Tensor) -> torch.Tensor:
            q = torch.zeros(m, 4, device=dev)
            q[:, 0], q[:, 3] = torch.cos(half), torch.sin(half)
            return q

        # carton: q = qz(yaw) * qy(90 deg) * qz(roll) — qy lays it down (+z -> +x), the
        # pre-roll about the long axis picks WHICH side face ends up underneath.
        yaw = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
        if c.randomize_face:
            roll = torch.randint(0, 4, (m,), device=dev).float() * (math.pi / 2)
        else:
            roll = torch.zeros(m, device=dev)
        qy90 = torch.tensor([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0],
                            device=dev).expand(m, 4)
        q = quat_mul(quat_mul(qz(yaw / 2), qy90), qz(roll / 2))
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.carton_pos[0]
        st[:, 1] = c.carton_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
        st[:, 2] = c.surface_z + c.body_w / 2 + 0.004
        st[:, 3:7] = q
        st[:, 0:3] += origin
        self.carton.write_root_state_to_sim(st, env_ids)

        def place(body, pos_xy, z, yawed: bool, jitter_scale: float = 1.0) -> None:
            s = torch.zeros(m, 13, device=dev)
            s[:, 0], s[:, 1] = pos_xy[0], pos_xy[1]
            s[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter * jitter_scale
            s[:, 2] = z
            if yawed:
                s[:, 3:7] = qz((torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2)
            else:
                s[:, 3] = 1.0
            s[:, 0:3] += origin
            body.write_root_state_to_sim(s, env_ids)

        place(self.crate, c.crate_pos, c.surface_z + c.crate_h / 2 + 0.002, True, 0.6)
        place(self.bottle, c.bottle_pos, c.surface_z + c.bottle_h / 2 + 0.002, False)
        place(self.butter, c.butter_pos, c.surface_z + c.butter_size[2] / 2 + 0.002, True)

        self.m[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch transient reorientation milestones every physics substep."""
        up_z = self.carton_up_z()
        self.m[:, 0] |= up_z > 0.5  # long axis raised past 30 deg
        self.m[:, 1] |= up_z > math.cos(math.radians(30.0))  # past 60 deg
        self.m[:, 2] |= up_z >= math.cos(math.radians(self.cfg.tilt_tol_deg))  # was upright

    # ----- state (full, restorable) ---------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "carton": self.carton.data.root_state_w[env_ids].clone(),
            "crate": self.crate.data.root_state_w[env_ids].clone(),
            "bottle": self.bottle.data.root_state_w[env_ids].clone(),
            "butter": self.butter.data.root_state_w[env_ids].clone(),
            "m": self.m[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.carton.write_root_state_to_sim(state["carton"], env_ids)
        self.crate.write_root_state_to_sim(state["crate"], env_ids)
        self.bottle.write_root_state_to_sim(state["bottle"], env_ids)
        self.butter.write_root_state_to_sim(state["butter"], env_ids)
        self.m[env_ids] = state["m"]

    # ----- description ----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        where = "on the ground" if c.surface_z <= 0 else "on a workbench"
        return (
            f"A milk carton (white box, {c.body_w * 1000:.0f} mm square, "
            f"{c.carton_H * 1000:.0f} mm tall, with a dark base and a blue gabled roof) "
            f"lies FALLEN on its side {where}. Nearby stand an open wooden crate with a "
            f"raised floor, an orange juice bottle (standing upright) and a small butter "
            f"box.\n"
            f"Goal: stand the fallen carton back UPRIGHT — dark base down, roof up — "
            f"resting directly on the surface, and leave it standing still. The carton "
            f"does not count inside or on top of the crate (or any other object): its "
            f"base must be on the surface itself. It cannot balance upside-down (the "
            f"roof is a ridge). Do not knock over the standing juice bottle — the task "
            f"is not complete while it is down."
        )

    # ----- progress / rubric ------------------------------------------------------------------
    def carton_up_z(self) -> torch.Tensor:
        """(N,) world-z component of the carton's roof axis (local +z)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(self.carton.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)

    def upright_geo(self) -> torch.Tensor:
        """(N,) bool: roof axis within `tilt_tol_deg` of world-up."""
        return self.carton_up_z() >= math.cos(math.radians(self.cfg.tilt_tol_deg))

    def base_on_table(self) -> torch.Tensor:
        """(N,) bool: the carton's bottom face within `base_z_tol` of the work surface.
        Rejects the crate's raised inner floor (35 mm up), the crate rim, the butter box
        — the base must be on the TABLE, not in or on anything."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        up = quat_apply(self.carton.data.root_quat_w, ez)
        base_z = (self.carton.data.root_pos_w - self.env_origins)[:, 2] - up[:, 2] * c.carton_H / 2
        return (base_z - c.surface_z).abs() < c.base_z_tol

    def carton_settled(self) -> torch.Tensor:
        """(N,) bool: carton linear AND angular velocity below the settle gates."""
        lin = self.carton.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin
        ang = self.carton.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang
        return lin & ang

    def bottle_up(self) -> torch.Tensor:
        """(N,) bool: the juice bottle still stands within `bottle_tilt_max_deg` of up."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        up = quat_apply(self.bottle.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.bottle_tilt_max_deg))

    def success(self) -> torch.Tensor:
        """(N,) bool: carton upright on its base, directly on the surface, settled, with
        the juice bottle still standing (physical outcome; the teleport-oracle's target)."""
        return self.upright_geo() & self.base_on_table() & self.carton_settled() & self.bottle_up()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: latched milestones [0.25 / 0.50 / 0.75] for raising the
        carton past 30 / 60 deg / having had it upright; capped at `bottle_down_cap`
        while the bottle is knocked over; 1.0 iff success(). Doing nothing scores 0."""
        c = self.cfg
        v = torch.tensor(c.milestones, device=self.env.device)
        s = torch.where(self.m[:, 2], v[2], torch.where(
            self.m[:, 1], v[1], torch.where(
                self.m[:, 0], v[0], torch.zeros(self.env.num_envs, device=self.env.device))))
        s = torch.where(self.bottle_up(), s, s.clamp(max=c.bottle_down_cap))
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("simgen", lambda: EnvCfg(scene="milk_upright", robot="null", env_spacing=2.5))
