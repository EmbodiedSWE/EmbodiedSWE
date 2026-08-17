"""RampHutchScene — slide the MIDDLE black bowl up the service ramp into the roofed
hutch on top of the cabinet
(libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i140).

Derived from libero_90 kitchen_scene2 "put the middle black bowl on top of the
cabinet", where the whole task is one vertical pick-and-place: grasp the middle of
three identical bowls and set it down inside a bbox above a cabinet's always-open flat
top. Here the DESTINATION'S ACCESS DIRECTION is inverted: the cabinet top is a walled
tray sealed under a fixed ROOF — there is no way to lower anything onto it from above.
The only opening is a DOORWAY in the tray's front wall, at the top of an inclined,
curb-fenced service RAMP that runs from the floor up to the door sill.

The seed's plan — carry the bowl above the cabinet and release it — executed here
drops the bowl onto the ROOF, a plain slab that is NOT the goal surface (smoke
negative: bowl-on-roof scores nothing). The intended strategy:

  PHASE T   TRANSPORT — pick the MIDDLE bowl of the row of three (identity matters:
            the other two bowls are decoys that must stay out) and set it down on the
            lower ramp, inside the curb channel.
  PHASE P   PUSH — slide the bowl UP the ramp: a sustained, friction-loaded uphill
            push (the 20 deg slope holds a parked bowl but resists motion), through
            the doorway, over the crest and sill; the bowl drops 3 cm into the tray
            and cannot come back out (the sill is a one-way step).

Success is judged on the PHYSICAL terminal state: the middle bowl upright and at rest
inside the tray band under the roof, having physically PASSED THROUGH THE DOORWAY
(a post_step passage latch — the only physical way in, so a state teleported into the
tray without the passage does not count), with neither decoy bowl inside the tray and
everything settled.

Mechanism notes:
  - The whole static structure (cabinet core, tray walls, door stubs, sill, roof,
    ramp, curbs) is ONE kinematic compound body, re-posed per reset (xy jitter +
    yaw); every predicate is evaluated in the fixture's body frame, so randomization
    is real. No joints anywhere.
  - The three bowls are IDENTICAL octagonal open cups (black, like the seed's akita
    bowls); which physical body lands in which row slot is a per-episode random
    permutation, so "the middle bowl" is a fresh identity every episode (stored
    per-env at reset; readback-verified in smoke).
  - Explicit physics materials everywhere (custom-spawner colliders otherwise default
    to mu~0.5): the bowl/ramp pair is calibrated so a parked bowl HOLDS on the slope
    (mu_s 0.55 > tan 20.4 deg = 0.37) while a ~2.5 N push moves it.

Everything is procedural. Heavy imports (isaaclab, pxr) are deferred so importing
this module stays app-free.
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


# ----- custom compound spawners -------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _bind_phys_material(stage, prim_path: str, root, mu_s: float, mu_d: float) -> None:
    """Author a physics material under the body and bind it to the whole subtree.
    Custom-spawner colliders otherwise get an uncontrolled default (~0.5)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, f"{prim_path}/physmat")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu_s))
    api.CreateDynamicFrictionAttr(float(mu_d))
    api.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(root).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_fixture(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One KINEMATIC rigid body: cabinet + roofed tray + door + ramp. Body frame:
    origin on the floor under the cabinet, door face toward -x, ramp descending
    toward -x. Landmarks: tray floor z=0.20, sill top z=0.23, ramp crest z=0.235
    at x=-0.19, roof underside z=0.315 / top z=0.33, doorway |y|<0.075."""
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
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(60.0)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)

    def _box(path, size, center, color, rot_y_deg: float | None = None):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        bxf = UsdGeom.Xformable(cube.GetPrim())
        bxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        if rot_y_deg is not None:
            bxf.AddRotateYOp().Set(float(rot_y_deg))
        bxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    body, wall, roof, ramp, curb, sill = (cfg.body_color, cfg.wall_color,
                                          cfg.roof_color, cfg.ramp_color,
                                          cfg.curb_color, cfg.sill_color)
    # cabinet core; its top face (z=0.20) IS the tray floor
    _box(f"{prim_path}/core", (0.28, 0.28, 0.20), (-0.05, 0.0, 0.10), body)
    # tray walls (rise from the tray floor to the roof underside — no gaps)
    _box(f"{prim_path}/wall_px", (0.02, 0.24, 0.115), (0.08, 0.0, 0.2575), wall)
    for sgn in (-1.0, 1.0):
        _box(f"{prim_path}/wall_y{'p' if sgn > 0 else 'n'}",
             (0.28, 0.02, 0.115), (-0.05, sgn * 0.11, 0.2575), wall)
        # door stubs: the -x face is closed except the doorway |y| < 0.075
        _box(f"{prim_path}/stub_{'p' if sgn > 0 else 'n'}",
             (0.02, 0.025, 0.115), (-0.18, sgn * 0.0875, 0.2575), wall)
    # door sill: fills the doorway from the tray floor up to z=0.23 (one-way step)
    _box(f"{prim_path}/sill", (0.02, 0.15, 0.03), (-0.18, 0.0, 0.215), sill)
    # roof: seals the tray from above (overhangs every wall)
    _box(f"{prim_path}/roof", (0.34, 0.32, 0.015), (-0.04, 0.0, 0.3225), roof)
    # service ramp: floor (x=-0.822) up to the crest (x=-0.19, z=0.235), 20.39 deg
    theta = math.degrees(math.atan2(0.235, 0.632))
    _box(f"{prim_path}/ramp", (0.6743, 0.15, 0.02), (-0.50252, 0.0, 0.10813),
         ramp, rot_y_deg=-theta)
    for sgn in (-1.0, 1.0):
        _box(f"{prim_path}/curb_{'p' if sgn > 0 else 'n'}",
             (0.6743, 0.02, 0.06), (-0.51297, sgn * 0.085, 0.13625),
             curb, rot_y_deg=-theta)
    _bind_phys_material(stage, prim_path, root, cfg.mu, cfg.mu - 0.05)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: a BLACK bowl — an open octagonal cup (bottom disc + 8 wall
    segments). Body frame: axis = +z (up when upright), origin at mid-height."""
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
    pxrb.CreateAngularDampingAttr(0.1)

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
    _bind_phys_material(stage, prim_path, root, cfg.mu, cfg.mu - 0.05)
    return root


def _spawn_plate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: the distractor plate (a squat cylinder, near-white)."""
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
    pxrb.CreateAngularDampingAttr(0.1)
    cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/disc")
    cyl.CreateRadiusAttr(cfg.radius)
    cyl.CreateHeightAttr(cfg.height)
    cyl.CreateExtentAttr([Gf.Vec3f(-cfg.radius, -cfg.radius, -cfg.height / 2),
                          Gf.Vec3f(cfg.radius, cfg.radius, cfg.height / 2)])
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(cyl.GetPrim())
    px.CreateContactOffsetAttr(float(cfg.contact_offset))
    px.CreateRestOffsetAttr(0.0)
    _bind_phys_material(stage, prim_path, root, 0.4, 0.35)
    return root


def _fixture_spawner_cfg(c: RampHutchSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "fixture" not in _SPAWNER_CACHE:

        @configclass
        class FixtureSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_fixture)
            mu: float = 0.55
            body_color: tuple = (0.50, 0.35, 0.22)
            wall_color: tuple = (0.58, 0.42, 0.27)
            roof_color: tuple = (0.25, 0.20, 0.16)
            ramp_color: tuple = (0.45, 0.45, 0.48)
            curb_color: tuple = (0.75, 0.60, 0.15)
            sill_color: tuple = (0.70, 0.55, 0.20)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["fixture"] = FixtureSpawnerCfg

    return _SPAWNER_CACHE["fixture"](
        mass_props=sim_utils.MassPropertiesCfg(mass=60.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mu=c.fixture_mu, body_color=c.body_color, wall_color=c.wall_color,
        roof_color=c.roof_color, ramp_color=c.ramp_color, curb_color=c.curb_color,
        sill_color=c.sill_color, contact_offset=c.contact_offset,
    )


def _bowl_spawner_cfg(c: RampHutchSceneCfg) -> Any:
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
            height: float = 0.062
            bot_t: float = 0.010
            mu: float = 0.55
            color: tuple = (0.08, 0.08, 0.09)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["bowl"] = BowlSpawnerCfg

    return _SPAWNER_CACHE["bowl"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=c.bowl_inner_r, wall_t=c.bowl_wall_t, height=c.bowl_h,
        bot_t=c.bowl_bot_t, mu=c.bowl_mu, color=c.bowl_color, contact_offset=0.002,
    )


def _plate_spawner_cfg(c: RampHutchSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "plate" not in _SPAWNER_CACHE:

        @configclass
        class PlateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plate)
            radius: float = 0.09
            height: float = 0.014
            color: tuple = (0.88, 0.88, 0.86)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["plate"] = PlateSpawnerCfg

    return _SPAWNER_CACHE["plate"](
        mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        radius=c.plate_r, height=c.plate_h, color=c.plate_color, contact_offset=0.002,
    )


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class RampHutchSceneCfg(BaseCfg):
    """Config for `RampHutchScene`. All geometry is FIXTURE-FRAME (origin on the floor
    under the cabinet; doorway/ramp toward -x). The fixture root is re-posed per reset
    (xy jitter + yaw), so nothing is world-anchored.

    Landmarks: tray floor z=0.20 (bowl rest z ~ 0.233); sill top z=0.23; ramp crest
    (x=-0.19, z=0.235); ramp foot x=-0.822 on the floor; ramp surface
    z(x) = (x + 0.822) * 0.37184; doorway |y| < 0.075, z 0.23..0.315; roof underside
    z=0.315, top z=0.33."""

    # --- tunable: rubric thresholds ---------------------------------------------------------------
    settle_speed: float = tunable(0.05)     # max |lin vel| (all bowls, plate) when judging (m/s)
    bowl_up_max_deg: float = tunable(25.0)  # bowl axis within this of world-up (a bowl
    # resting flat on the 20.4 deg ramp is tilted 20.4 deg — must still count upright)
    # target bowl center, fixture frame — "high on the ramp" credit zone:
    hi_x_lo: float = tunable(-0.33)
    hi_x_hi: float = tunable(-0.15)
    hi_z_lo: float = tunable(0.17)
    hi_z_hi: float = tunable(0.31)
    hi_y_abs: float = tunable(0.09)
    # doorway passage slab (inside the door cut only — the physical way in):
    door_x_lo: float = tunable(-0.20)
    door_x_hi: float = tunable(-0.14)
    door_y_abs: float = tunable(0.085)
    door_z_lo: float = tunable(0.215)
    door_z_hi: float = tunable(0.315)
    # seated band inside the tray (bowl center, fixture frame):
    seat_x_lo: float = tunable(-0.15)
    seat_x_hi: float = tunable(0.04)
    seat_y_abs: float = tunable(0.06)
    seat_z_lo: float = tunable(0.212)
    seat_z_hi: float = tunable(0.28)
    # tray OCCUPANCY volume (any bowl; blocks success if a decoy is inside):
    tray_x_lo: float = tunable(-0.165)
    tray_x_hi: float = tunable(0.06)
    tray_y_abs: float = tunable(0.095)
    tray_z_lo: float = tunable(0.205)
    tray_z_hi: float = tunable(0.30)

    # --- tunable: randomization -------------------------------------------------------------------
    fix_jitter: float = tunable(0.04)       # fixture root xy jitter (+/- m)
    fix_yaw_deg: float = tunable(10.0)      # fixture root yaw (+/- deg)
    row_cx: float = tunable(-0.52)          # bowl-row center (fixture x)
    row_cx_jitter: float = tunable(0.04)
    row_y: float = tunable(-0.38)           # bowl-row line (fixture y)
    row_y_jitter: float = tunable(0.03)
    row_spacing: tuple = tunable((0.16, 0.20))  # slot spacing band
    bowl_jitter: float = tunable(0.015)     # per-bowl xy jitter inside its slot
    plate_pos: tuple = tunable((-0.20, -0.55))  # distractor plate center (fixture xy)
    plate_jitter: float = tunable(0.05)

    # --- info: fixture ----------------------------------------------------------------------------
    fixture_mu: float = info(0.55)
    tray_floor_z: float = info(0.20)
    sill_top_z: float = info(0.23)
    crest_x: float = info(-0.19)
    crest_z: float = info(0.235)
    ramp_foot_x: float = info(-0.822)
    ramp_slope: float = info(0.37184)       # tan of the ramp angle (20.39 deg)
    roof_top_z: float = info(0.33)
    body_color: tuple = info((0.50, 0.35, 0.22))
    wall_color: tuple = info((0.58, 0.42, 0.27))
    roof_color: tuple = info((0.25, 0.20, 0.16))
    ramp_color: tuple = info((0.45, 0.45, 0.48))
    curb_color: tuple = info((0.75, 0.60, 0.15))
    sill_color: tuple = info((0.70, 0.55, 0.20))
    contact_offset: float = info(0.003)

    # --- info: bowls / plate ----------------------------------------------------------------------
    bowl_inner_r: float = info(0.042)
    bowl_wall_t: float = info(0.010)
    bowl_h: float = info(0.062)
    bowl_bot_t: float = info(0.010)
    bowl_mass: float = info(0.15)
    bowl_mu: float = info(0.55)
    bowl_color: tuple = info((0.08, 0.08, 0.09))
    plate_r: float = info(0.09)
    plate_h: float = info(0.014)
    plate_color: tuple = info((0.88, 0.88, 0.86))


# ----- scene ----------------------------------------------------------------------------------------
@SCENES.register("ramp_hutch")
class RampHutchScene(BaseScene):
    cfg: RampHutchSceneCfg

    def __init__(self, cfg: RampHutchSceneCfg | None = None) -> None:
        super().__init__(cfg or RampHutchSceneCfg())

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
            "fixture": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Fixture",
                spawn=_fixture_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate",
                spawn=_plate_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.plate_pos[0], c.plate_pos[1], c.plate_h / 2 + 0.002)),
            ),
        }
        for i in range(3):
            out[f"bowl_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl_" + str(i),
                spawn=_bowl_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.row_cx + (i - 1) * 0.18, c.row_y, c.bowl_h / 2 + 0.002)),
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
        self.fixture: RigidObject = env.iscene["fixture"]
        self.plate: RigidObject = env.iscene["plate"]
        self.bowls: list[RigidObject] = [env.iscene[f"bowl_{i}"] for i in range(3)]
        self.env_origins = env.iscene.env_origins
        # which body index is "the middle bowl" this episode (sampled at reset)
        self.target_idx = torch.zeros(n, dtype=torch.long, device=dev)
        # latched progress (post_step)
        self._hi_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._doored_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._seated_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        # reset grace re-pin buffers
        self._grace = torch.zeros(n, dtype=torch.long, device=dev)
        self._pin_states = {k: torch.zeros(n, 13, device=dev)
                            for k in ("fixture", "plate", "bowl_0", "bowl_1", "bowl_2")}

    # ----- frames ---------------------------------------------------------------------------------
    def _to_fix(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(...,3) world points -> fixture body frame (broadcast over leading dims)."""
        from isaaclab.utils.math import quat_apply_inverse

        q = self.fixture.data.root_quat_w
        p = self.fixture.data.root_pos_w
        if pos_w.dim() == 3:  # (N, B, 3)
            nb = pos_w.shape[1]
            q = q[:, None, :].expand(-1, nb, -1).reshape(-1, 4)
            return quat_apply_inverse(
                q, (pos_w - p[:, None, :]).reshape(-1, 3)).reshape(pos_w.shape)
        return quat_apply_inverse(q, pos_w - p)

    def _bowl_pos_fix(self) -> torch.Tensor:
        """(N, 3bowls, 3) all bowl centers in the fixture frame."""
        pos = torch.stack([b.data.root_pos_w for b in self.bowls], dim=1)
        return self._to_fix(pos)

    def _bowl_up(self) -> torch.Tensor:
        """(N, 3bowls) cos of each bowl axis vs world-up."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        return torch.stack(
            [quat_apply(b.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)
             for b in self.bowls], dim=1)

    def _gather_target(self, per_bowl: torch.Tensor) -> torch.Tensor:
        """(N, 3bowls, ...) -> (N, ...) rows for each env's target bowl."""
        idx = self.target_idx.view(-1, *([1] * (per_bowl.dim() - 1)))
        idx = idx.expand(-1, 1, *per_bowl.shape[2:])
        return per_bowl.gather(1, idx).squeeze(1)

    # ----- predicates -----------------------------------------------------------------------------
    def bowls_upright(self) -> torch.Tensor:
        return self._bowl_up() >= math.cos(math.radians(self.cfg.bowl_up_max_deg))

    def bowls_in_tray(self) -> torch.Tensor:
        """(N, 3bowls) bool: bowl center inside the tray occupancy volume."""
        c = self.cfg
        loc = self._bowl_pos_fix()
        return ((loc[:, :, 0] > c.tray_x_lo) & (loc[:, :, 0] < c.tray_x_hi)
                & (loc[:, :, 1].abs() < c.tray_y_abs)
                & (loc[:, :, 2] > c.tray_z_lo) & (loc[:, :, 2] < c.tray_z_hi))

    def target_fix(self) -> torch.Tensor:
        """(N, 3) target bowl center in the fixture frame."""
        return self._gather_target(self._bowl_pos_fix())

    def target_hi_ramp(self) -> torch.Tensor:
        """(N,) bool: target bowl upright, high on the ramp / at the doorway."""
        c = self.cfg
        loc = self.target_fix()
        return ((loc[:, 0] > c.hi_x_lo) & (loc[:, 0] < c.hi_x_hi)
                & (loc[:, 1].abs() < c.hi_y_abs)
                & (loc[:, 2] > c.hi_z_lo) & (loc[:, 2] < c.hi_z_hi)
                & self._gather_target(self.bowls_upright()))

    def target_in_door(self) -> torch.Tensor:
        """(N,) bool: target bowl center inside the doorway cut (the only way in)."""
        c = self.cfg
        loc = self.target_fix()
        return ((loc[:, 0] > c.door_x_lo) & (loc[:, 0] < c.door_x_hi)
                & (loc[:, 1].abs() < c.door_y_abs)
                & (loc[:, 2] > c.door_z_lo) & (loc[:, 2] < c.door_z_hi))

    def target_seated(self) -> torch.Tensor:
        """(N,) bool: target bowl upright inside the tray's seated band (live)."""
        c = self.cfg
        loc = self.target_fix()
        return ((loc[:, 0] > c.seat_x_lo) & (loc[:, 0] < c.seat_x_hi)
                & (loc[:, 1].abs() < c.seat_y_abs)
                & (loc[:, 2] > c.seat_z_lo) & (loc[:, 2] < c.seat_z_hi)
                & self._gather_target(self.bowls_upright()))

    def decoy_in_tray(self) -> torch.Tensor:
        """(N,) bool: any NON-target bowl inside the tray occupancy volume."""
        n = self.env.num_envs
        in_tray = self.bowls_in_tray()
        mask = torch.ones(n, 3, dtype=torch.bool, device=self.env.device)
        mask.scatter_(1, self.target_idx.view(-1, 1), False)
        return (in_tray & mask).any(dim=1)

    def settled(self) -> torch.Tensor:
        c = self.cfg
        ok = self.plate.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        for b in self.bowls:
            ok &= b.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return ok

    # ----- mechanism (every substep) --------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        # reset grace: re-pin freshly reset bodies while write timing settles
        gids = (self._grace > 0).nonzero(as_tuple=False).squeeze(-1)
        if len(gids):
            for name, body in (("fixture", self.fixture), ("plate", self.plate),
                               ("bowl_0", self.bowls[0]), ("bowl_1", self.bowls[1]),
                               ("bowl_2", self.bowls[2])):
                body.write_root_state_to_sim(self._pin_states[name][gids], gids)
            self._grace[gids] -= 1
            return

        self._hi_ever |= self.target_hi_ramp()
        self._doored_ever |= self.target_in_door()
        self._seated_ever |= self.target_seated() & self._doored_ever

    # ----- reset ----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Sample the fixture root pose (xy jitter + yaw), a random permutation of the
        three bowl bodies into the three row slots (target = the body in the MIDDLE
        slot), row placement, per-bowl jitter + free yaw, plate placement. Everything
        written with zero velocity; a 2-substep grace re-pin absorbs write races."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # fixture pose
        yaw = torch.deg2rad((torch.rand(m, device=dev) * 2 - 1) * c.fix_yaw_deg)
        half = yaw / 2
        fq = torch.zeros(m, 4, device=dev)
        fq[:, 0] = torch.cos(half)
        fq[:, 3] = torch.sin(half)
        fp = torch.zeros(m, 3, device=dev)
        fp[:, :2] = (torch.rand(m, 2, device=dev) * 2 - 1) * c.fix_jitter
        fp += origin
        fst = torch.zeros(m, 13, device=dev)
        fst[:, 0:3] = fp
        fst[:, 3:7] = fq
        self.fixture.write_root_state_to_sim(fst, env_ids)
        self._pin_states["fixture"][env_ids] = fst

        cy, sy = torch.cos(yaw), torch.sin(yaw)

        def to_world(loc: torch.Tensor) -> torch.Tensor:
            w = torch.zeros(m, 3, device=dev)
            w[:, 0] = cy * loc[:, 0] - sy * loc[:, 1]
            w[:, 1] = sy * loc[:, 0] + cy * loc[:, 1]
            w[:, 2] = loc[:, 2]
            return w + fp

        # body -> slot permutation (torch.rand argsort: healthy across seeds)
        perm = torch.rand(m, 3, device=dev).argsort(dim=1)  # perm[e, slot] = body idx
        self.target_idx[env_ids] = perm[:, 1]

        # row geometry
        row_x = c.row_cx + (torch.rand(m, device=dev) * 2 - 1) * c.row_cx_jitter
        row_y = c.row_y + (torch.rand(m, device=dev) * 2 - 1) * c.row_y_jitter
        spacing = (c.row_spacing[0]
                   + (c.row_spacing[1] - c.row_spacing[0]) * torch.rand(m, device=dev))

        slot_of_body = perm.argsort(dim=1)  # slot_of_body[e, body] = slot idx
        for i, bowl in enumerate(self.bowls):
            slot = slot_of_body[:, i].float() - 1.0  # -1, 0, +1 along the row
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = row_x + slot * spacing
            loc[:, 1] = row_y
            loc[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.bowl_jitter
            loc[:, 2] = c.bowl_h / 2 + 0.003
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = to_world(loc)
            bhalf = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            st[:, 3] = torch.cos(bhalf)
            st[:, 6] = torch.sin(bhalf)
            self.bowls[i].write_root_state_to_sim(st, env_ids)
            self._pin_states[f"bowl_{i}"][env_ids] = st

        # plate distractor
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.plate_pos[0]
        loc[:, 1] = c.plate_pos[1]
        loc[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.plate_jitter
        loc[:, 2] = c.plate_h / 2 + 0.003
        pst = torch.zeros(m, 13, device=dev)
        pst[:, 0:3] = to_world(loc)
        pst[:, 3] = 1.0
        self.plate.write_root_state_to_sim(pst, env_ids)
        self._pin_states["plate"][env_ids] = pst

        for lat in (self._hi_ever, self._doored_ever, self._seated_ever):
            lat[env_ids] = False
        self._grace[env_ids] = 2

    # ----- state (full, restorable) ---------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"fixture": self.fixture, "plate": self.plate,
                  "bowl_0": self.bowls[0], "bowl_1": self.bowls[1],
                  "bowl_2": self.bowls[2]}
        return {
            "bodies": {k: b.data.root_state_w[env_ids].clone() for k, b in bodies.items()},
            "target_idx": self.target_idx[env_ids].clone(),
            "latches": torch.stack(
                [self._hi_ever[env_ids], self._doored_ever[env_ids],
                 self._seated_ever[env_ids]], dim=1).clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"fixture": self.fixture, "plate": self.plate,
                  "bowl_0": self.bowls[0], "bowl_1": self.bowls[1],
                  "bowl_2": self.bowls[2]}
        for k, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][k], env_ids)
        self.target_idx[env_ids] = state["target_idx"]
        lat = state["latches"]
        self._hi_ever[env_ids] = lat[:, 0]
        self._doored_ever[env_ids] = lat[:, 1]
        self._seated_ever[env_ids] = lat[:, 2]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "A wooden cabinet (28 x 28 cm, 20 cm tall) stands on the floor. Its top is "
            "a walled TRAY sealed under a fixed dark ROOF slab — there is NO opening "
            "above: anything released over the cabinet just lands on the roof, which "
            "is not the goal surface. The tray's only entrance is a DOORWAY (15 cm "
            "wide, 8.5 cm tall) cut into the front wall, reached by an inclined gray "
            "service RAMP with yellow side curbs that runs from the floor up to the "
            "door sill (23 cm high, ~20 degree slope). Just inside the doorway the "
            "tray floor lies 3 cm below the sill, so a bowl that crosses the crest "
            "drops in and cannot slide back out. On the floor beside the ramp, three "
            "IDENTICAL BLACK BOWLS (open cups, 10.4 cm across, 6.2 cm tall) stand in "
            "a line; a white plate lies nearby as a distractor. The task concerns "
            "ONLY the bowl in the MIDDLE of the line (identify it by its position "
            "between the other two at the start). Goal: the MIDDLE bowl upright and "
            "at rest inside the roofed tray on top of the cabinet. It can only get "
            "there through the doorway: bring it to the ramp and slide it UP the "
            "curb-fenced channel, through the doorway and over the sill, until it "
            "drops into the tray. The two outer bowls and the plate must stay OUT of "
            "the tray — a wrong bowl inside the tray forfeits the task. The bowl must "
            "end upright (open side up); everything must be at rest at the end."
        )

    def instruction(self) -> str:
        return (
            "Put the middle bowl of the three black bowls on top of the cabinet: the "
            "top compartment is roofed, so slide the bowl up the yellow-curbed ramp "
            "and through the doorway until it drops inside. Leave the other two bowls "
            "and the plate outside, and make sure the bowl ends upright."
        )

    # ----- rubric ---------------------------------------------------------------------------------
    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched stages of the demonstrated solution:
        0.15 target bowl ever high on the ramp (upper third, upright) + 0.25 target
        ever inside the doorway cut (the passage) + 0.35 target ever seated in the
        tray after the passage; exactly 1.0 iff success(). Null policy ~0; the seed's
        plan (release the bowl over the cabinet) parks it on the roof and scores 0."""
        s = (0.15 * self._hi_ever.float()
             + 0.25 * self._doored_ever.float()
             + 0.35 * self._seated_ever.float())
        return torch.where(self.success(),
                           torch.ones(self.env.num_envs, device=self.env.device),
                           s.clamp(0.0, 0.95))

    def success(self) -> torch.Tensor:
        """(N,) bool: the MIDDLE bowl upright at rest inside the tray band under the
        roof, having physically passed through the doorway (the only way in), with
        neither decoy bowl inside the tray, and everything settled."""
        return (self.target_seated() & self._doored_ever & ~self.decoy_in_tray()
                & self.settled())


register_env("simgen", lambda: EnvCfg(scene="ramp_hutch", robot="null"))
