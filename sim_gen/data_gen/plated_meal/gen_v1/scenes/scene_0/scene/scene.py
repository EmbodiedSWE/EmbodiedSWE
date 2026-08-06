"""PlatedMealScene — gather every food cube into ONE bowl, then present that bowl on the plate.

Derived from libero_90 kitchen_scene2 "put the black bowl in the middle on the plate", but the
GOAL TYPE changes: the seed judges a single spatial relation (a chosen container sitting on a
flat target) reached by one grasp-carry-place; here the deliverable is a CONSTRUCTED ASSEMBLY —
a served meal.  Three identical dark bowls stand on a kitchen counter, a white low-rim serving
plate sits apart from them, and 2-3 bright "food" cubes (count sampled per episode) lie
scattered on the counter.  Success requires, simultaneously and settled:

  * every present food cube rests INSIDE one single bowl (the serving bowl — any of the three
    may be chosen; identity is free, the ASSEMBLY is what is judged);
  * that bowl stands upright with its bottom ON the plate floor, inside the plate rim;
  * BOTH other bowls are kept well clear of the plate (exclusive presentation);
  * nothing may be left over: a food cube on the counter, on the plate floor beside the bowl,
    or in a second bowl fails the all-in-one clause.

The seed's complete goal state — a bare bowl set perfectly in the middle of the plate — is an
explicitly tested, insufficient outcome here (latched 0.15 credit, never success).

Rubric (graded 0..1, latched partial progress in post_step; 1.0 iff success()):
  0.00  nothing happened
  0.15  a bowl has been ON the plate at least once (the seed's whole goal = first rung)
  0.25  a food cube has been INSIDE a bowl at least once
  0.50  all present food has been gathered in ONE bowl at the same time
  0.75  full assembly reached at least once (all food in a bowl that is on the plate)
  1.00  success(): assembly settled + exclusivity (distractor bowls clear of the plate)

All containment geometry is judged in body frames (a yawed/nudged plate or bowl judges the
same). A cube physically inside a bowl's cavity is at most ~27 mm off-axis (< food_xy_tol
30 mm — any cube genuinely inside counts); a cube perched on a bowl rim sits at ~44 mm and
above the depth gate, and a bowl resting on the plate rim fails the bottom-height band. The
bowl-centering tolerance (36 mm) is a real precision requirement: the plate aperture leaves
~41 mm of physical play, so an uncentered set-down near the rim is a genuine near-miss.

Assets are fully procedural, one rigid body each (the pen_holder compound-spawner pattern):
  - vessel spawner (shared by the bowls and the plate): bottom disc + 8 box wall segments
    forming an open octagonal cup; bowls are deep (75 mm) with a 12 mm pinch-friendly wall,
    the plate is wide (210 mm aperture) and low (30 mm).
  - food: plain colored cubes (30 mm), zero restitution + light damping so they settle.

Per-episode randomization: whole-layout mirror (the plate side flips), per-body xy jitter +
free yaw, food count 2-3 with a random slot permutation; absent cubes park in an off-counter
ground depot.  Verified by readback in the smoke.

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
    """Author one open octagonal vessel at `prim_path` (bowl or plate, same geometry family)."""
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
    # Cap the contact-solver pop: a cube dropped into the bowl penetrates a little in one
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

    if cfg.ear_h > 0:  # bail handle (two rim posts + a crossbar over the mouth centre):
        # the crossbar sits directly ABOVE the bowl's centre of mass, so a parallel jaw
        # pinching its middle carries the bowl torque-free (no pivot-escape failure mode).
        ear_color = Gf.Vec3f(*cfg.ear_color)
        post_r = cfg.inner_r + cfg.wall_t / 2  # posts stand on two opposite wall tops
        post_h = cfg.ear_h + 0.006  # exposed height + 6 mm wall-top embed
        for sgn in (1.0, -1.0):
            post = UsdGeom.Cube.Define(stage, f"{prim_path}/handle_post_{'p' if sgn > 0 else 'n'}")
            post.CreateSizeAttr(1.0)
            pxf = UsdGeom.Xformable(post.GetPrim())
            pxf.AddTranslateOp().Set(Gf.Vec3d(
                sgn * post_r, 0.0, cfg.height / 2 - 0.006 + post_h / 2))
            pxf.AddScaleOp().Set(Gf.Vec3f(cfg.ear_t, cfg.ear_t, post_h))
            post.CreateDisplayColorAttr([ear_color])
            collide(post.GetPrim())
        beam = UsdGeom.Cube.Define(stage, f"{prim_path}/handle_beam")
        beam.CreateSizeAttr(1.0)
        bxf = UsdGeom.Xformable(beam.GetPrim())
        bxf.AddTranslateOp().Set(Gf.Vec3d(
            0.0, 0.0, cfg.height / 2 + cfg.ear_h + cfg.ear_t / 2))
        bxf.AddScaleOp().Set(Gf.Vec3f(2 * post_r + cfg.ear_t, cfg.ear_t, cfg.ear_t))
        beam.CreateDisplayColorAttr([ear_color])
        collide(beam.GetPrim())

    return root


def _vessel_spawner_cfg(*, inner_r: float, wall_t: float, height: float, bot_t: float,
                        mass: float, color: tuple, n_segments: int,
                        contact_offset: float, ear_t: float = 0.0, ear_h: float = 0.0,
                        ear_color: tuple = (0.45, 0.45, 0.50)) -> Any:
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
            inner_r: float = 0.045  # inner octagon INRADIUS (m)
            wall_t: float = 0.012
            height: float = 0.075
            bot_t: float = 0.010
            color: tuple = (0.13, 0.13, 0.15)
            n_segments: int = 8
            contact_offset: float = 0.002
            ear_t: float = 0.0  # handle bar/post cross-section (square)
            ear_h: float = 0.0  # crossbar clearance above the rim plane
            ear_color: tuple = (0.45, 0.45, 0.50)

        _SPAWNER_CACHE["vessel"] = VesselSpawnerCfg

    return _SPAWNER_CACHE["vessel"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=inner_r, wall_t=wall_t, height=height, bot_t=bot_t,
        color=color, n_segments=n_segments, contact_offset=contact_offset,
        ear_t=ear_t, ear_h=ear_h, ear_color=ear_color,
    )


# ----- scene cfg ----------------------------------------------------------------------------------
@dataclass
class PlatedMealSceneCfg(BaseCfg):
    """Config for `PlatedMealScene`. Tolerances are soft where geometry already enforces the
    real limit (honesty by construction — see the module docstring)."""

    # --- tunable: rubric thresholds -----------------------------------------------------------
    bowl_xy_tol: float = tunable(0.036)  # bowl centre within this of the plate axis (plate
    # frame). Physical max offset of a bowl inside the rim = 95 - 61.7 = 33.3 mm < tol, so any
    # bowl genuinely inside the plate counts; a bowl on the rim fails the bottom-height band.
    bowl_z_tol: float = tunable(0.012)  # |bowl bottom - plate floor top| below this (m)
    bowl_tilt_max_deg: float = tunable(10.0)  # serving bowl axis within this of world-up
    plate_tilt_max_deg: float = tunable(10.0)  # plate axis within this of world-up
    food_xy_tol: float = tunable(0.030)  # cube centre within this of the bowl axis (bowl
    # frame). Physical max for a cube in the cavity ~ 38 - 15 + tilt ~ 27 mm < tol; a cube on
    # the 12 mm rim sits at ~44 mm and is rejected by both gates.
    food_depth_min: float = tunable(0.008)  # cube centre this far below the rim plane (m)
    clear_dist: float = tunable(0.135)  # NON-serving bowl centres must be farther than this
    # from the plate axis (horizontal). A bowl standing beside the plate, touching its rim,
    # sits at >= 175 mm and passes; any bowl on/over the plate (< ~114 mm) fails.
    settle_speed: float = tunable(0.05)  # max |v| of every judged body when judging (m/s)

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    plate_jitter: float = tunable(0.025)  # uniform +/- xy jitter of the plate at reset
    bowl_jitter: float = tunable(0.020)  # uniform +/- xy jitter per bowl at reset
    food_jitter: float = tunable(0.015)  # uniform +/- xy jitter per cube at reset
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per body at reset
    mirror_layout: bool = tunable(True)  # per-episode whole-layout y-mirror (plate side flips)
    subset_sample: bool = tunable(True)  # sample the food count per episode (else all 3)
    min_present: int = tunable(2)  # lower bound of the sampled food count

    # --- tunable: placement --------------------------------------------------------------------
    surface_z: float = tunable(0.20)  # counter height; the arm base is mounted on the counter
    plate_pos: tuple = tunable((0.03, -0.31))  # plate centre slot (counter frame)
    bowl_slots: tuple = tunable(((0.05, 0.16), (0.22, 0.16), (0.22, 0.36)))  # bowl slots
    food_slots: tuple = tunable(((-0.02, 0.02), (0.08, -0.05), (0.02, -0.12)))  # cube slots

    # --- info: structure ------------------------------------------------------------------------
    bench_size: tuple = info((1.5, 1.3))  # kinematic counter slab top (x, y)
    n_bowls: int = info(3)
    bowl_inner_r: float = info(0.038)  # cavity inradius: flat-to-flat 76 mm < the 80 mm jaw
    # span, so a parallel jaw can grip the bowl from INSIDE (expansion grip on two opposite
    # inner walls); 3 cubes still fit (2 on the floor + 1 stacked); drop funnel 38 - 21 = 17 mm
    bowl_wall_t: float = info(0.012)  # wall thickness (also an outside pinch affordance)
    bowl_h: float = info(0.075)  # interior depth 65 mm: a 2-cube pile stays below the rim
    bowl_bot_t: float = info(0.010)
    # Bail handle over each bowl's mouth (two rim posts + a crossbar): the crossbar's
    # middle sits directly above the CoM, so a parallel jaw carries the bowl torque-free.
    # Food is dropped through the two ~31 mm-wide openings beside the crossbar.
    ear_t: float = info(0.012)  # handle bar/post cross-section (square)
    ear_h: float = info(0.035)  # crossbar clearance above the rim plane
    ear_color: tuple = info((0.45, 0.45, 0.50))
    bowl_mass: float = info(0.15)
    bowl_color: tuple = info((0.13, 0.13, 0.15))  # all three identical — pick any
    plate_inner_r: float = info(0.095)  # bowl (circum 61.7) drops in with 33 mm of funnel
    plate_wall_t: float = info(0.010)
    plate_h: float = info(0.030)  # rim stands 20 mm above the floor: cubes cannot roll off
    plate_bot_t: float = info(0.010)
    plate_mass: float = info(0.35)
    plate_color: tuple = info((0.93, 0.92, 0.88))
    food_size: float = info(0.030)  # cube edge
    food_mass: float = info(0.04)
    food_colors: tuple = info(((0.85, 0.15, 0.12), (0.95, 0.78, 0.10), (0.15, 0.65, 0.20)))
    food_names: tuple = info(("red", "yellow", "green"))
    n_segments: int = info(8)
    bowl_contact_offset: float = info(0.002)
    plate_contact_offset: float = info(0.003)
    parking_pos: tuple = info((1.15, 1.15))  # ground depot for absent cubes (off the counter)

    # Derived (filled in __post_init__).
    bowl_outer_r: float = field(default=None, init=False)  # outer octagon inradius
    bowl_circum_r: float = field(default=None, init=False)  # outer octagon circumradius
    plate_outer_r: float = field(default=None, init=False)
    plate_floor_local_z: float = field(default=None, init=False)  # plate floor top, plate frame
    bowl_floor_local_z: float = field(default=None, init=False)  # bowl floor top, bowl frame

    def __post_init__(self) -> None:
        self.bowl_outer_r = round(self.bowl_inner_r + self.bowl_wall_t, 4)
        self.bowl_circum_r = round(self.bowl_outer_r / math.cos(math.pi / self.n_segments), 4)
        self.plate_outer_r = round(self.plate_inner_r + self.plate_wall_t, 4)
        self.plate_floor_local_z = round(-self.plate_h / 2 + self.plate_bot_t, 4)
        self.bowl_floor_local_z = round(-self.bowl_h / 2 + self.bowl_bot_t, 4)


# ----- scene ---------------------------------------------------------------------------------------
@SCENES.register("plated_meal")
class PlatedMealScene(BaseScene):
    cfg: PlatedMealSceneCfg

    def __init__(self, cfg: PlatedMealSceneCfg | None = None) -> None:
        super().__init__(cfg or PlatedMealSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the counter slab, the plate, three bowls, and three food cubes at
        nominal slots (reset() re-places everything and samples the present subset)."""
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
        if z0 > 0:  # kinematic counter slab, top at surface_z
            out["bench"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bench_size[0], c.bench_size[1], z0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.08, 0.0, z0 / 2)),
            )

        out["plate"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plate",
            spawn=_vessel_spawner_cfg(
                inner_r=c.plate_inner_r, wall_t=c.plate_wall_t, height=c.plate_h,
                bot_t=c.plate_bot_t, mass=c.plate_mass, color=c.plate_color,
                n_segments=c.n_segments, contact_offset=c.plate_contact_offset,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.plate_pos[0], c.plate_pos[1], z0 + c.plate_h / 2 + 0.002)),
        )

        for i in range(c.n_bowls):
            sx, sy = c.bowl_slots[i]
            out[f"bowl_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl_" + str(i),
                spawn=_vessel_spawner_cfg(
                    inner_r=c.bowl_inner_r, wall_t=c.bowl_wall_t, height=c.bowl_h,
                    bot_t=c.bowl_bot_t, mass=c.bowl_mass, color=c.bowl_color,
                    n_segments=c.n_segments, contact_offset=c.bowl_contact_offset,
                    ear_t=c.ear_t, ear_h=c.ear_h, ear_color=c.ear_color,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx, sy, z0 + c.bowl_h / 2 + 0.002)),
            )

        for i, name in enumerate(c.food_names):
            sx, sy = c.food_slots[i]
            out[f"food_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Food_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(c.food_size, c.food_size, c.food_size),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.10),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.food_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.food_colors[i]),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7, restitution=0.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx, sy, z0 + c.food_size / 2 + 0.002)),
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
        """Grab handles + allocate the presence mask and the progress latches."""
        super().bind(env)
        c = self.cfg
        self.plate: RigidObject = env.iscene["plate"]
        self.bowls: list[RigidObject] = [env.iscene[f"bowl_{i}"] for i in range(c.n_bowls)]
        self.food: list[RigidObject] = [env.iscene[f"food_{nm}"] for nm in c.food_names]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # present[e, f]: cube f participates in episode e (sampled at reset; judged subset).
        self.present = torch.ones(n, len(c.food_names), dtype=torch.bool, device=dev)
        self.side = torch.ones(n, device=dev)  # layout mirror sign (readback knob)
        # progress latches (post_step; cleared per reset)
        self.ever_plated = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_loaded = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_gathered = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_assembled = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the layout mirror, place plate/bowls with jitter + yaw, sample
        the present food subset + a slot permutation, park absent cubes, clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        yaw_amp = math.radians(c.reset_yaw_deg)

        side = (torch.randint(0, 2, (m,), device=dev, dtype=torch.float32) * 2 - 1) \
            if c.mirror_layout else torch.ones(m, device=dev)
        self.side[env_ids] = side

        def yawed(st: torch.Tensor) -> None:
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)

        # --- plate ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.plate_pos[0]
        st[:, 1] = c.plate_pos[1] * side
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.plate_jitter
        st[:, 2] = c.surface_z + c.plate_h / 2 + 0.002
        yawed(st)
        st[:, 0:3] += origin
        self.plate.write_root_state_to_sim(st, env_ids)

        # --- bowls ---
        for i, bowl in enumerate(self.bowls):
            sx, sy = c.bowl_slots[i]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = sx
            st[:, 1] = sy * side
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.bowl_jitter
            st[:, 2] = c.surface_z + c.bowl_h / 2 + 0.002
            yawed(st)
            st[:, 0:3] += origin
            bowl.write_root_state_to_sim(st, env_ids)

        # --- food: subset sampling + slot permutation; absent -> ground depot ---
        n_food = len(c.food_names)
        if c.subset_sample:
            k = torch.randint(c.min_present, n_food + 1, (m,), device=dev)
        else:
            k = torch.full((m,), n_food, dtype=torch.long, device=dev)
        rank = torch.rand(m, n_food, device=dev).argsort(dim=1).argsort(dim=1)
        pres = rank < k.unsqueeze(1)  # (m, F)
        self.present[env_ids] = pres
        slot_of = torch.rand(m, n_food, device=dev).argsort(dim=1)  # cube f -> slot index
        slots = torch.tensor(c.food_slots, device=dev)  # (F, 2)
        for f, cube in enumerate(self.food):
            sxy = slots[slot_of[:, f]]  # (m, 2)
            scat = torch.zeros(m, 3, device=dev)
            scat[:, 0] = sxy[:, 0]
            scat[:, 1] = sxy[:, 1] * side
            scat[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.food_jitter
            scat[:, 2] = c.surface_z + c.food_size / 2 + 0.002
            park = torch.zeros(m, 3, device=dev)
            park[:, 0] = c.parking_pos[0] + f * 0.12
            park[:, 1] = c.parking_pos[1]
            park[:, 2] = c.food_size / 2 + 0.003
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.where(pres[:, f].unsqueeze(1), scat, park)
            yawed(st)
            cube.write_root_state_to_sim(st, env_ids)

        for latch in (self.ever_plated, self.ever_loaded, self.ever_gathered,
                      self.ever_assembled):
            latch[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch the progress milestones at sim rate so partial credit survives later mishaps
        (and along a correct trajectory the printed score never decreases)."""
        onp = self.bowls_on_plate()  # (N, B)
        fib = self.food_in_bowl()  # (N, F, B)
        gath = self.gathered()  # (N, B)
        self.ever_plated |= onp.any(dim=1)
        self.ever_loaded |= (fib.any(dim=2) & self.present).any(dim=1)
        self.ever_gathered |= gath.any(dim=1)
        self.ever_assembled |= (gath & onp).any(dim=1)

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "bowls": [b.data.root_state_w[env_ids].clone() for b in self.bowls],
            "food": [f.data.root_state_w[env_ids].clone() for f in self.food],
            "present": self.present[env_ids].clone(),
            "side": self.side[env_ids].clone(),
            "latches": torch.stack([self.ever_plated[env_ids], self.ever_loaded[env_ids],
                                    self.ever_gathered[env_ids],
                                    self.ever_assembled[env_ids]], dim=1),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.plate.write_root_state_to_sim(state["plate"], env_ids)
        for b, st in zip(self.bowls, state["bowls"]):
            b.write_root_state_to_sim(st, env_ids)
        for f, st in zip(self.food, state["food"]):
            f.write_root_state_to_sim(st, env_ids)
        self.present[env_ids] = state["present"]
        self.side[env_ids] = state["side"]
        lat = state["latches"]
        self.ever_plated[env_ids] = lat[:, 0]
        self.ever_loaded[env_ids] = lat[:, 1]
        self.ever_gathered[env_ids] = lat[:, 2]
        self.ever_assembled[env_ids] = lat[:, 3]

    # ----- description -----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A kitchen counter. On one side sits a round WHITE serving plate "
            f"(~{2 * c.plate_outer_r * 1000:.0f} mm across, with a low {1000 * (c.plate_h - c.plate_bot_t):.0f} mm rim). "
            f"On the other side stand three identical DARK bowls "
            f"(~{2 * c.bowl_circum_r * 1000:.0f} mm across, {c.bowl_h * 1000:.0f} mm tall, open on top; each "
            f"carries a light-gray BAIL HANDLE: an arch over its mouth whose "
            f"{c.ear_t * 1000:.0f} mm crossbar runs {c.ear_h * 1000:.0f} mm above the rim — pinch the "
            f"crossbar's middle to carry the bowl; food fits through the two openings beside "
            f"the crossbar). Scattered between them lie "
            f"bright food cubes ({c.food_size * 1000:.0f} mm): red, yellow and/or green — between "
            f"{c.min_present} and {len(c.food_names)} are present in any episode, so count what "
            f"you see. Which side the plate is on, all positions and headings, and the food "
            f"count change every episode.\n"
            f"Goal: serve the meal. Gather EVERY food cube on the counter into ONE bowl (any of "
            f"the three — they are interchangeable), and end with that bowl standing upright, "
            f"bottom resting on the plate floor near the plate centre, all cubes inside it, and "
            f"BOTH other bowls kept well clear of the plate (more than ~{c.clear_dist * 100:.0f} cm "
            f"from its centre — where they start is fine). Any order works: fill the bowl first "
            f"and carry it loaded, or set the empty bowl on the plate and drop the food in. An "
            f"empty bowl on the plate, food lying on the plate outside the bowl, food split "
            f"between bowls, or a second bowl on the plate do NOT count. Everything must come "
            f"to rest."
        )

    # ----- geometric predicates ---------------------------------------------------------------------
    def _up_z(self, quat: torch.Tensor) -> torch.Tensor:
        """z-component of a body's local +z in world (…,) for tilt gates."""
        from isaaclab.utils.math import quat_apply

        shape = quat.shape[:-1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(*shape, 3)
        return quat_apply(quat.reshape(-1, 4), ez.reshape(-1, 3)).reshape(*shape, 3)[..., 2]

    def plate_up(self) -> torch.Tensor:
        """(N,) bool: plate axis within `plate_tilt_max_deg` of world-up."""
        return self._up_z(self.plate.data.root_quat_w).clamp(-1, 1) >= \
            math.cos(math.radians(self.cfg.plate_tilt_max_deg))

    def bowls_upright(self) -> torch.Tensor:
        """(N, B) bool: bowl axis within `bowl_tilt_max_deg` of world-up."""
        quat = torch.stack([b.data.root_quat_w for b in self.bowls], dim=1)
        return self._up_z(quat).clamp(-1, 1) >= \
            math.cos(math.radians(self.cfg.bowl_tilt_max_deg))

    def bowls_on_plate(self) -> torch.Tensor:
        """(N, B) bool: bowl centre within `bowl_xy_tol` of the plate axis (PLATE frame), bowl
        bottom within `bowl_z_tol` of the plate floor top, bowl upright, plate upright."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        n, nb = self.env.num_envs, c.n_bowls
        pos = torch.stack([b.data.root_pos_w for b in self.bowls], dim=1)  # (N,B,3)
        pq = self.plate.data.root_quat_w[:, None, :].expand(n, nb, 4).reshape(-1, 4)
        pp = self.plate.data.root_pos_w[:, None, :]
        loc = quat_apply_inverse(pq, (pos - pp).reshape(-1, 3)).reshape(n, nb, 3)
        near = loc[:, :, :2].norm(dim=-1) < c.bowl_xy_tol
        # bottom height in world z (both bodies are gated upright, so world-z is honest)
        quat = torch.stack([b.data.root_quat_w for b in self.bowls], dim=1)
        bowl_bottom = pos[:, :, 2] - self._up_z(quat) * c.bowl_h / 2
        plate_floor = self.plate.data.root_pos_w[:, 2] + c.plate_floor_local_z
        on_floor = (bowl_bottom - plate_floor.unsqueeze(1)).abs() < c.bowl_z_tol
        return near & on_floor & self.bowls_upright() & self.plate_up().unsqueeze(1)

    def food_in_bowl(self) -> torch.Tensor:
        """(N, F, B) bool: cube centre inside bowl b's cavity, judged in the BOWL body frame
        (a carried/tilted bowl still contains its food)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        n = self.env.num_envs
        nf, nb = len(self.food), c.n_bowls
        fpos = torch.stack([f.data.root_pos_w for f in self.food], dim=1)  # (N,F,3)
        bpos = torch.stack([b.data.root_pos_w for b in self.bowls], dim=1)  # (N,B,3)
        bquat = torch.stack([b.data.root_quat_w for b in self.bowls], dim=1)  # (N,B,4)
        rel = fpos[:, :, None, :] - bpos[:, None, :, :]  # (N,F,B,3)
        bq = bquat[:, None, :, :].expand(n, nf, nb, 4).reshape(-1, 4)
        loc = quat_apply_inverse(bq, rel.reshape(-1, 3)).reshape(n, nf, nb, 3)
        near = loc[..., :2].norm(dim=-1) < c.food_xy_tol
        z_lo = c.bowl_floor_local_z - 0.006
        z_hi = c.bowl_h / 2 - c.food_depth_min
        in_z = (loc[..., 2] > z_lo) & (loc[..., 2] < z_hi)
        return near & in_z

    def gathered(self) -> torch.Tensor:
        """(N, B) bool: EVERY present cube is inside bowl b (judged on the sampled subset)."""
        fib = self.food_in_bowl()  # (N,F,B)
        ok = fib | ~self.present[:, :, None]
        return ok.all(dim=1)

    def bowls_clear(self) -> torch.Tensor:
        """(N, B) bool: bowl centre horizontally farther than `clear_dist` from the plate axis."""
        pos = torch.stack([b.data.root_pos_w for b in self.bowls], dim=1)
        d = (pos[:, :, :2] - self.plate.data.root_pos_w[:, None, :2]).norm(dim=-1)
        return d > self.cfg.clear_dist

    def settled(self) -> torch.Tensor:
        """(N,) bool: plate, every bowl, and every PRESENT cube |lin vel| below settle_speed."""
        c = self.cfg
        still = self.plate.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        for b in self.bowls:
            still &= b.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        for f, cube in enumerate(self.food):
            slow = cube.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
            still &= slow | ~self.present[:, f]
        return still

    def success(self) -> torch.Tensor:
        """(N,) bool: some bowl is the served meal — on the plate, holding ALL present food —
        while both other bowls are clear of the plate; everything settled."""
        onp = self.bowls_on_plate()  # (N,B)
        gath = self.gathered()  # (N,B)
        clear = self.bowls_clear()  # (N,B)
        nb = self.cfg.n_bowls
        served = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for b in range(nb):
            others = [j for j in range(nb) if j != b]
            served |= onp[:, b] & gath[:, b] & clear[:, others].all(dim=1)
        return served & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float 0..1 — the latched ladder from the module docstring. Monotone along the
        intended solve: 0 -> 0.15 (bowl plated) -> 0.25 (first food in a bowl) -> 0.50 (all
        food gathered in one bowl) -> 0.75 (full assembly reached) -> 1.0 (settled + exclusive:
        success)."""
        s = torch.zeros(self.env.num_envs, device=self.env.device)
        s = torch.where(self.ever_plated, torch.full_like(s, 0.15), s)
        s = torch.where(self.ever_loaded, torch.full_like(s, 0.25), s)
        s = torch.where(self.ever_gathered, torch.full_like(s, 0.50), s)
        s = torch.where(self.ever_assembled, torch.full_like(s, 0.75), s)
        s = torch.where(self.success(), torch.full_like(s, 1.0), s)
        return s


# Scene-level task: solve.py builds its own Franka binding.
register_env("simgen", lambda: EnvCfg(scene="plated_meal", robot="null"))
