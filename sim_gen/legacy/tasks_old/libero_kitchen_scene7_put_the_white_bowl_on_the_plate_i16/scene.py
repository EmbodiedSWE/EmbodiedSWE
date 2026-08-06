"""DishRackScene — load the drying rack: plate VERTICAL into the rail slot, bowl and cup
FLIPPED upside-down over their drying pegs.

Derived from libero_90 kitchen_scene7 "put the white bowl on the plate", but STRATEGICALLY
DIFFERENT: the seed is a single upright pick-and-place — grasp the white bowl, keep it level,
set it down ON the flat plate (an on-top relation; the plate is a passive target that never
moves). Here every one of the seed's roles is overturned:

  1. ROLE SWAP — the plate is no longer the target; it is a manipulated object. Its goal
     pose is VERTICAL: it must be rotated 90 deg (flat -> edge-on) and lowered into a
     narrow rail slot on a drying rack, like a real dish rack.
  2. REORIENTATION IS THE SKILL — the white bowl (and a cup) must be flipped a full
     180 deg and set OPENING-DOWN, capped over a drying peg. The seed's plan is
     level-carry; any solver that keeps dishes upright scores nothing.
  3. RELATION CHANGE — on-top stacking is replaced by slotted insertion (plate between
     rails, held by them) and capping (peg inside the inverted vessel's cavity).
  4. The seed's own goal state — the upright bowl standing ON the flat plate on the
     counter — is expressible in-scene and is an explicit ~0-score negative control.

Judging is final-state, physical, and strategy-agnostic: each item counts only when its
settled pose really is held by the rack geometry — the plate leaning inside the slot
(verticality is honest by construction: rail height/gap bound the possible lean), each
vessel inverted with the peg inside its cavity and its rim on the rack base. Rubric:
0.1 latched once any dish has ever been lifted off the counter (transient-achievement
latch in post_step) + 0.26 per racked item; 1.0 iff success (all three racked, settled).
Null policy scores exactly 0.

Per-episode randomization: rack pose (xy jitter + free yaw — all judging is done in the
rack's body frame), per-dish xy jitter + free yaw, and a Bernoulli mirror flip of the
whole dish spawn layout across the counter midline, so a memorized fixed trajectory fails.

All assets are procedural: the rack is one kinematic compound body (base slab + 2 slot
rails + 2 end stops + 2 cylindrical pegs), the plate is a plain rigid disc, bowl and cup
are open octagonal vessels (the pen_holder compound-spawner pattern — child colliders of
one body never self-collide).

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
# `isaaclab.sim.utils.clone` is borrowed (the regex-resolve + per-env replicate machinery every
# CuboidCfg spawn uses). Fresh Define per prim -> xformOps authored once (idempotent under clone).

_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_vessel(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author an open octagonal vessel (bowl / cup) at `prim_path`: root Xform with
    RigidBodyAPI + explicit MassAPI, a bottom cylinder collider and 8 box wall segments of
    inner inradius `inner_r`. Body frame: geometric centre, opening toward +z. Depenetration
    capped at 0.5 m/s and lightly damped so flips and set-downs settle promptly."""
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
    pxrb.CreateAngularDampingAttr(0.10)

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


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the drying rack at `prim_path`: a KINEMATIC rigid body (a fixture, still
    re-posed per episode by root-state writes). Body frame origin = base-slab centre.
    Children: base slab, 2 slot rails along local x at y = slot_y, 2 end stops closing the
    groove (a disc cannot roll out), 2 drying pegs on the +y half."""
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

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    def box(name: str, size, center, color) -> None:
        b = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        b.CreateSizeAttr(1.0)
        bxf = UsdGeom.Xformable(b.GetPrim())
        bxf.AddTranslateOp().Set(Gf.Vec3d(*center))
        bxf.AddScaleOp().Set(Gf.Vec3f(*size))
        b.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        collide(b.GetPrim())

    rail_c, peg_c = cfg.rail_color, cfg.peg_color
    st = cfg.slab_t
    box("base", (cfg.base_x, cfg.base_y, st), (0.0, 0.0, 0.0), cfg.base_color)

    # slot: rails along local x at y = slot_y +/- (gap/2 + rail_t/2), sitting ON the slab top
    zr = st / 2 + cfg.rail_h / 2
    for s, name in ((-1.0, "rail_n"), (1.0, "rail_p")):
        box(name, (cfg.slot_len, cfg.rail_t, cfg.rail_h),
            (0.0, cfg.slot_y + s * (cfg.slot_gap / 2 + cfg.rail_t / 2), zr), rail_c)
    stop_w = cfg.slot_gap + 2 * cfg.rail_t
    for s, name in ((-1.0, "stop_n"), (1.0, "stop_p")):
        box(name, (cfg.rail_t, stop_w, cfg.rail_h),
            (s * (cfg.slot_len / 2 + cfg.rail_t / 2), cfg.slot_y, zr), rail_c)

    # drying pegs (cylinders standing on the slab top)
    for (px_, py_, pr, ph), name in (
        ((cfg.peg1_xy[0], cfg.peg1_xy[1], cfg.peg1_r, cfg.peg_h), "peg1"),
        ((cfg.peg2_xy[0], cfg.peg2_xy[1], cfg.peg2_r, cfg.peg_h), "peg2"),
    ):
        p = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{name}")
        p.CreateRadiusAttr(pr)
        p.CreateHeightAttr(ph)
        p.CreateExtentAttr([Gf.Vec3f(-pr, -pr, -ph / 2), Gf.Vec3f(pr, pr, ph / 2)])
        UsdGeom.Xformable(p.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(px_, py_, st / 2 + ph / 2))
        p.CreateDisplayColorAttr([Gf.Vec3f(*peg_c)])
        collide(p.GetPrim())
    return root


def _vessel_spawner_cfg(*, inner_r: float, wall_t: float, height: float, bot_t: float,
                        mass: float, color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "vessel" not in _SPAWNER_CACHE:

        @configclass
        class VesselSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_vessel)
            inner_r: float = 0.052
            wall_t: float = 0.008
            height: float = 0.048
            bot_t: float = 0.010
            color: tuple = (0.92, 0.92, 0.90)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["vessel"] = VesselSpawnerCfg

    return _SPAWNER_CACHE["vessel"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=inner_r, wall_t=wall_t, height=height, bot_t=bot_t,
        color=color, contact_offset=contact_offset,
    )


def _rack_spawner_cfg(cfg: "DishRackSceneCfg") -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rack" not in _SPAWNER_CACHE:

        @configclass
        class RackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            base_x: float = 0.36
            base_y: float = 0.26
            slab_t: float = 0.024
            slot_len: float = 0.20
            slot_gap: float = 0.026
            slot_y: float = -0.065
            rail_t: float = 0.012
            rail_h: float = 0.055
            peg1_xy: tuple = (-0.07, 0.065)
            peg1_r: float = 0.020
            peg2_xy: tuple = (0.09, 0.065)
            peg2_r: float = 0.014
            peg_h: float = 0.030
            base_color: tuple = (0.55, 0.57, 0.60)
            rail_color: tuple = (0.75, 0.77, 0.80)
            peg_color: tuple = (0.80, 0.55, 0.20)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["rack"] = RackSpawnerCfg

    return _SPAWNER_CACHE["rack"](
        mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        base_x=cfg.rack_base[0], base_y=cfg.rack_base[1], slab_t=cfg.rack_slab_t,
        slot_len=cfg.slot_len, slot_gap=cfg.slot_gap, slot_y=cfg.slot_y,
        rail_t=cfg.rail_t, rail_h=cfg.rail_h,
        peg1_xy=cfg.peg1_xy, peg1_r=cfg.peg1_r,
        peg2_xy=cfg.peg2_xy, peg2_r=cfg.peg2_r, peg_h=cfg.peg_h,
        contact_offset=0.002,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DishRackSceneCfg(BaseCfg):
    """Config for `DishRackScene`. Tolerances are geometric-honest: the slot's rail
    height/gap bound the physically possible lean of a racked plate below `plate_lean_max`,
    and a vessel's capping tolerance is its inner inradius minus the peg radius (minus
    margin), so any pose satisfying the clauses really is held by the rack."""

    # --- tunable: rubric thresholds -----------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max |lin vel| of an item when judging (m/s)
    lift_height: float = tunable(0.15)  # any dish centre this high off the counter -> latch (m)
    plate_lean_max_deg: float = tunable(25.0)  # racked plate axis within this of horizontal
    plate_z_tol: float = tunable(0.015)  # plate centre-height band half-width in the slot (m)
    vessel_tilt_max_deg: float = tunable(20.0)  # inverted vessel axis within this of straight-down
    vessel_z_tol: float = tunable(0.012)  # capped vessel centre-height band half-width (m)
    cap_margin: float = tunable(0.004)  # xy capping tolerance = inner_r - peg_r - this (m)

    # --- tunable: randomization (the task-family knobs) ---------------------------------------
    reset_pos_jitter: float = tunable(0.03)  # uniform +/- xy jitter (rack AND dishes) at reset
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per body at reset
    mirror_layout: bool = tunable(True)  # Bernoulli mirror of the dish spawn side per episode

    # --- tunable: placement --------------------------------------------------------------------
    surface_z: float = tunable(0.0)  # counter height; 0 = on the ground (null smoke)
    rack_pos: tuple = tunable((0.26, 0.0))  # rack base centre on the counter
    plate_spawn: tuple = tunable((-0.20, -0.10))  # plate flat on the counter
    bowl_spawn: tuple = tunable((-0.16, 0.14))  # bowl upright on the counter
    cup_spawn: tuple = tunable((0.00, -0.26))  # cup upright on the counter

    # --- info: rack (kinematic compound fixture) ------------------------------------------------
    rack_base: tuple = info((0.36, 0.26))  # base slab (x, y)
    rack_slab_t: float = info(0.024)
    slot_len: float = info(0.20)  # groove interior length (end stops close it)
    slot_gap: float = info(0.026)  # rail gap; plate_t 0.014 -> 6 mm clearance per side
    slot_y: float = info(-0.065)  # slot centreline, rack local y
    rail_t: float = info(0.012)
    rail_h: float = info(0.055)  # with the gap this bounds racked-plate lean to ~13 deg
    peg1_xy: tuple = info((-0.07, 0.065))  # bowl peg, rack local
    peg1_r: float = info(0.020)
    peg2_xy: tuple = info((0.09, 0.065))  # cup peg, rack local
    peg2_r: float = info(0.014)
    peg_h: float = info(0.030)  # < vessel interior depths: rims land on the slab, not the peg

    # --- info: dishes ----------------------------------------------------------------------------
    plate_r: float = info(0.085)
    plate_t: float = info(0.014)
    plate_mass: float = info(0.25)
    plate_color: tuple = info((0.93, 0.93, 0.96))
    bowl_inner_r: float = info(0.052)  # capping tol = 0.052 - 0.020 - margin = 28 mm
    bowl_wall_t: float = info(0.008)
    bowl_h: float = info(0.048)  # interior depth 0.038 > peg_h: rim rests on the slab
    bowl_bot_t: float = info(0.010)
    bowl_mass: float = info(0.15)
    bowl_color: tuple = info((0.92, 0.92, 0.90))  # the seed's WHITE bowl
    cup_inner_r: float = info(0.034)  # capping tol = 0.034 - 0.014 - margin = 16 mm
    cup_wall_t: float = info(0.007)
    cup_h: float = info(0.060)
    cup_bot_t: float = info(0.010)
    cup_mass: float = info(0.08)
    cup_color: tuple = info((0.25, 0.45, 0.75))
    contact_offset: float = info(0.003)

    # Derived (filled in __post_init__).
    slab_top_local: float = field(default=None, init=False)  # slab top, rack body frame
    bowl_outer_r: float = field(default=None, init=False)
    cup_outer_r: float = field(default=None, init=False)
    bowl_cap_tol: float = field(default=None, init=False)
    cup_cap_tol: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.slab_top_local = round(self.rack_slab_t / 2, 4)
        self.bowl_outer_r = round(self.bowl_inner_r + self.bowl_wall_t, 4)
        self.cup_outer_r = round(self.cup_inner_r + self.cup_wall_t, 4)
        self.bowl_cap_tol = round(self.bowl_inner_r - self.peg1_r - self.cap_margin, 4)
        self.cup_cap_tol = round(self.cup_inner_r - self.peg2_r - self.cap_margin, 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("dish_rack")
class DishRackScene(BaseScene):
    cfg: DishRackSceneCfg

    def __init__(self, cfg: DishRackSceneCfg | None = None) -> None:
        super().__init__(cfg or DishRackSceneCfg())

    # ----- assets ------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic drying rack, and the three free dishes at their
        nominal counter spots (reset() re-places everything)."""
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

        out["rack"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Rack",
            spawn=_rack_spawner_cfg(c),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.rack_pos[0], c.rack_pos[1], z0 + c.rack_slab_t / 2)),
        )

        out["plate"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plate",
            spawn=sim_utils.CylinderCfg(
                radius=c.plate_r,
                height=c.plate_t,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.05, angular_damping=0.20,
                    max_depenetration_velocity=0.5),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.plate_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.plate_color),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.plate_spawn[0], c.plate_spawn[1], z0 + c.plate_t / 2 + 0.002)),
        )

        out["bowl"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bowl",
            spawn=_vessel_spawner_cfg(
                inner_r=c.bowl_inner_r, wall_t=c.bowl_wall_t, height=c.bowl_h,
                bot_t=c.bowl_bot_t, mass=c.bowl_mass, color=c.bowl_color,
                contact_offset=c.contact_offset,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.bowl_spawn[0], c.bowl_spawn[1], z0 + c.bowl_h / 2 + 0.002)),
        )

        out["cup"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cup",
            spawn=_vessel_spawner_cfg(
                inner_r=c.cup_inner_r, wall_t=c.cup_wall_t, height=c.cup_h,
                bot_t=c.cup_bot_t, mass=c.cup_mass, color=c.cup_color,
                contact_offset=c.contact_offset,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.cup_spawn[0], c.cup_spawn[1], z0 + c.cup_h / 2 + 0.002)),
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
        self.rack: RigidObject = env.iscene["rack"]
        self.plate: RigidObject = env.iscene["plate"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.cup: RigidObject = env.iscene["cup"]
        self.dishes: dict[str, RigidObject] = {
            "plate": self.plate, "bowl": self.bowl, "cup": self.cup}
        self.env_origins = env.iscene.env_origins
        # lifted[e]: any dish has been raised off the counter at least once this episode —
        # the transient-achievement latch, updated in post_step, paying 0.1 in score().
        self.lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the rack with xy jitter + free yaw, mirror the dish layout
        across y with p=0.5, scatter the dishes (plate FLAT, bowl and cup UPRIGHT) with xy
        jitter + free yaw, clear the lift latch."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        yaw_amp = math.radians(c.reset_yaw_deg)

        self.lifted[env_ids] = False

        if c.mirror_layout:
            mirror = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        else:
            mirror = torch.ones(m, device=dev)

        def yawed_state(base_xy: tuple, z: float, mirror_y: bool = True) -> torch.Tensor:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = base_xy[0]
            st[:, 1] = base_xy[1] * (mirror if mirror_y else 1.0)
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            st[:, 2] = z
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            return st

        # rack (kinematic, still re-posed each episode); its own yaw is free — all judging
        # happens in the rack body frame
        self.rack.write_root_state_to_sim(
            yawed_state(c.rack_pos, c.surface_z + c.rack_slab_t / 2, mirror_y=False), env_ids)

        # dishes: plate flat, vessels upright, all with jitter + yaw, layout mirrored
        self.plate.write_root_state_to_sim(
            yawed_state(c.plate_spawn, c.surface_z + c.plate_t / 2 + 0.002), env_ids)
        self.bowl.write_root_state_to_sim(
            yawed_state(c.bowl_spawn, c.surface_z + c.bowl_h / 2 + 0.002), env_ids)
        self.cup.write_root_state_to_sim(
            yawed_state(c.cup_spawn, c.surface_z + c.cup_h / 2 + 0.002), env_ids)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch the lift achievement at sim rate: any dish centre ever rising above
        `lift_height` over the counter marks 'a dish left the surface'."""
        c = self.cfg
        for b in self.dishes.values():
            z = b.data.root_pos_w[:, 2] - self.env_origins[:, 2]
            self.lifted |= z > c.surface_z + c.lift_height

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "rack": self.rack.data.root_state_w[env_ids].clone(),
            "dishes": {n: b.data.root_state_w[env_ids].clone()
                       for n, b in self.dishes.items()},
            "lifted": self.lifted[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.rack.write_root_state_to_sim(state["rack"], env_ids)
        for n, b in self.dishes.items():
            b.write_root_state_to_sim(state["dishes"][n], env_ids)
        self.lifted[env_ids] = state["lifted"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A dish-drying rack stands on the counter: a metal base "
            f"({c.rack_base[0] * 1000:.0f}x{c.rack_base[1] * 1000:.0f} mm) carrying a plate "
            f"slot (two rails {c.slot_gap * 1000:.0f} mm apart, {c.rail_h * 1000:.0f} mm "
            f"tall) and two orange drying pegs. Nearby on the counter lie a flat white plate "
            f"(disc, {2 * c.plate_r * 1000:.0f} mm across), an upright white bowl and an "
            f"upright blue cup.\n"
            f"Goal: load the rack — stand the plate VERTICALLY (on edge) inside the rail "
            f"slot, and place the bowl and the cup UPSIDE-DOWN so each caps over its drying "
            f"peg (bowl on the thick peg, cup on the thin peg), rims resting on the rack "
            f"base. A plate laid flat, a vessel left upright, or anything left on the "
            f"counter does not count. Only the final settled state is judged."
        )

    # ----- predicates / rubric --------------------------------------------------------------------
    def _rack_local(self, points: torch.Tensor) -> torch.Tensor:
        """Express world points (N,3) in the rack body frame -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.rack.data.root_quat_w,
                                  points - self.rack.data.root_pos_w)

    def _axis_w(self, body) -> torch.Tensor:
        """(N,3): the body's local +z axis in world frame."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)

    def _settled(self, body) -> torch.Tensor:
        return body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def plate_racked(self) -> torch.Tensor:
        """(N,) bool: plate standing on edge INSIDE the rail slot — centre within the groove
        span (rack frame), centre height one plate-radius above the slab (a plate lying flat
        across the rail tops sits lower and fails), axis within `plate_lean_max_deg` of
        horizontal (honest by construction: the gap/rail geometry bounds a slotted plate's
        lean to ~13 deg), settled."""
        c = self.cfg
        loc = self._rack_local(self.plate.data.root_pos_w)
        in_span = loc[:, 0].abs() < c.slot_len / 2
        in_gap = (loc[:, 1] - c.slot_y).abs() < c.slot_gap
        z_ok = (loc[:, 2] - (c.slab_top_local + c.plate_r)).abs() < c.plate_z_tol
        lean_ok = self._axis_w(self.plate)[:, 2].abs() < math.sin(
            math.radians(c.plate_lean_max_deg))
        return in_span & in_gap & z_ok & lean_ok & self._settled(self.plate)

    def _vessel_capped(self, body, height: float, peg_xy: tuple,
                       cap_tol: float) -> torch.Tensor:
        """(N,) bool: vessel INVERTED (opening within `vessel_tilt_max_deg` of straight
        down) with its axis within `cap_tol` of the peg (rack frame — the peg is inside the
        cavity by construction) and its rim resting on the slab (centre-height band),
        settled."""
        c = self.cfg
        upside_down = self._axis_w(body)[:, 2] < -math.cos(
            math.radians(c.vessel_tilt_max_deg))
        loc = self._rack_local(body.data.root_pos_w)
        peg = torch.tensor(peg_xy, device=self.env.device)
        near_peg = (loc[:, :2] - peg).norm(dim=-1) < cap_tol
        z_ok = (loc[:, 2] - (c.slab_top_local + height / 2)).abs() < c.vessel_z_tol
        return upside_down & near_peg & z_ok & self._settled(body)

    def bowl_capped(self) -> torch.Tensor:
        c = self.cfg
        return self._vessel_capped(self.bowl, c.bowl_h, c.peg1_xy, c.bowl_cap_tol)

    def cup_capped(self) -> torch.Tensor:
        c = self.cfg
        return self._vessel_capped(self.cup, c.cup_h, c.peg2_xy, c.cup_cap_tol)

    def n_racked(self) -> torch.Tensor:
        """(N,) int: how many of the three items are correctly racked right now."""
        return (self.plate_racked().int() + self.bowl_capped().int()
                + self.cup_capped().int())

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.1 lift latch + 0.26 per racked item (0.36 / 0.62 partial);
        1.0 iff success (all three racked, settled — each clause includes settling). ~0 for
        doing nothing; items may be racked in any order."""
        s = 0.1 * self.lifted.float() + 0.26 * self.n_racked().float()
        return torch.where(self.success(), torch.ones_like(s), s)

    def success(self) -> torch.Tensor:
        """(N,) bool: plate vertical in the slot AND bowl AND cup inverted over their pegs,
        all settled."""
        return self.plate_racked() & self.bowl_capped() & self.cup_capped()


# Scene-level env binding (robot embodiments are a later stage).
register_env("simgen", lambda: EnvCfg(scene="dish_rack", robot="null", env_spacing=3))
