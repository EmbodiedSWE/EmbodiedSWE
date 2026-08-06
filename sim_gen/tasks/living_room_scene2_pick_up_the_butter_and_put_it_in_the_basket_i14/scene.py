"""ButterHopperScene — stage the basket under an elevated butter dispenser, then pull the
dispenser's sliding tray so the butter drops into the basket (seed:
libero_90/living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket, strategically
inverted).

The seed task is a plain pick-and-place: grasp the butter off the table, carry it, drop
it into an open basket. Here that plan is PHYSICALLY IMPOSSIBLE and the roles are
inverted:

- the butter is a 96 x 96 x 36 mm slab — wider than the Franka's 80 mm jaw aperture in
  BOTH horizontal dimensions, so it cannot be pinched in any resting pose (and lying
  flat its 36 mm face cannot be reached from the side without a finger below the
  support surface). The seed's core skill — grasping the butter — is off the table;
- the butter starts CONFINED inside an elevated, capped hopper shaft whose only floor
  is a sliding tray on a horizontal prismatic slide. The only way to move the butter is
  to pull the tray: the shaft wall scrapes the slab off the receding tray and it
  free-falls out of the shaft's open bottom;
- the MANIPULATED object is therefore the CONTAINER: the basket must be carried/pushed
  onto the drop zone under the shaft BEFORE the tray is pulled. The drop is
  irreversible — butter dispensed onto the bare floor is a flat ungraspable slab that
  cannot be lifted over the basket's wall, so the execution order (stage basket, then
  pull) is enforced by physics, not by fiat.

Mechanism (all procedural primitives): a FIXED kinematic housing (one compound rigid:
four shaft walls, a full lid, a rear support column, plus a visual-only green drop-zone
pad painted on the floor) and a dynamic tray (plate + bright-yellow handle fin, one
compound rigid) joined to the housing by a Y prismatic joint (pair collision disabled;
SYMMETRIC limits +/- travel — the GPU sign-convention hedge; linear damping so the tray
stays where it is left; gravity borne by the joint). The butter is a free rigid slab
resting on the tray inside the shaft. The housing/tray pair never randomizes or
teleports per episode (jointed pairs must not teleport on this PhysX stack — measured);
the free bodies (basket, butter jitter) carry the randomization.

Judged on PHYSICAL outcome only: success() = the butter is resting INSIDE the basket
(basket-frame containment, below the rim), butter and basket settled, basket upright,
the tray open past `open_min` — sustained for `hold_steps` consecutive substeps — and
the butter has never lain on the bare floor (`_floored` latch). The tray-open clause is
what physically must be true of any honest delivery (the butter cannot leave the shaft
otherwise) and rejects the seed-strategy end state (butter magically inside the basket
with the dispenser still sealed); the floor latch makes the execution order REAL: a
slab dispensed onto the floor cannot be scored later, not even by lowering the basket
over it to trap it (measured hole, closed). score() in
[0, 1]: 0.30 latched once the basket has been staged on the drop zone; +0.30 scaled by
the latched maximum tray opening; 0.85 latched once the butter has settled inside the
basket; 1.0 iff success. Null policy scores 0.

Layout is designed for a single Franka at base (-0.50, 0, 0), chosen in solve.py: the
shaft axis stands 0.60 m ahead of the base, the handle fin sweeps y -0.105 -> -0.275
at hand height ~0.45, and the basket spawns on a lateral arc 0.44-0.54 m from the base.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- custom compound spawners -----------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_root(prim_path: str, translation, orientation, *, kinematic: bool, mass: float,
                max_depen: float = 1.0):
    """Root Xform + rigid-body APIs, xform ops authored fresh (no duplicate-op clones)."""
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
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateMaxDepenetrationVelocityAttr(max_depen)
    return stage, root


def _box_part(stage, prim_path: str, name: str, size, center, color,
              contact_offset: float | None) -> None:
    """One box child; collider iff contact_offset is not None."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    seg = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*center))
    sxf.AddScaleOp().Set(Gf.Vec3f(*size))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)


def _spawn_housing(prim_path: str, cfg: Any, translation=None, orientation=None):
    """FIXED kinematic dispenser housing, root at the SHAFT AXIS on the floor: four
    shaft walls (open bottom), a full lid capping the shaft top, a rear support column,
    a beam from column to lid, and a visual-only green drop-zone pad on the floor."""
    stage, root = _apply_root(prim_path, translation, orientation, kinematic=True, mass=5.0)
    ih, wt = cfg.shaft_inner_half, cfg.shaft_wall_t
    oh = ih + wt
    z0, z1 = cfg.shaft_z_bot, cfg.shaft_z_top  # wall band
    wall_h = z1 - z0
    co, col = cfg.contact_offset, cfg.housing_color
    # shaft walls (open bottom, open top capped by the lid)
    for sy in (-1.0, 1.0):
        _box_part(stage, prim_path, f"wall_y{'p' if sy > 0 else 'n'}",
                  (2 * oh, wt, wall_h), (0.0, sy * (ih + wt / 2), z0 + wall_h / 2), col, co)
    for sx in (-1.0, 1.0):
        _box_part(stage, prim_path, f"wall_x{'p' if sx > 0 else 'n'}",
                  (wt, 2 * ih, wall_h), (sx * (ih + wt / 2), 0.0, z0 + wall_h / 2), col, co)
    # lid: covers the shaft top and runs back to the column (no reach-in from above)
    lid_x0, lid_x1 = -oh, cfg.column_x + cfg.column_size[0] / 2
    _box_part(stage, prim_path, "lid",
              (lid_x1 - lid_x0, 2 * oh, cfg.lid_t),
              ((lid_x0 + lid_x1) / 2, 0.0, z1 + cfg.lid_t / 2), col, co)
    # rear support column, ground -> lid
    _box_part(stage, prim_path, "column",
              (cfg.column_size[0], cfg.column_size[1], z1 + cfg.lid_t),
              (cfg.column_x, 0.0, (z1 + cfg.lid_t) / 2), col, co)
    # visual-only drop-zone pad painted on the floor under the shaft
    _box_part(stage, prim_path, "drop_pad",
              (cfg.pad_side, cfg.pad_side, 0.002), (0.0, 0.0, 0.001), cfg.pad_color, None)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Dispenser tray, root at the PLATE center: flat plate collider + bright-yellow
    handle fin standing on the plate's -y end (the graspable feature — a 12 mm pinch
    with positive engagement for a -y pull)."""
    stage, root = _apply_root(prim_path, translation, orientation, kinematic=False,
                              mass=cfg.mass_props.mass, max_depen=0.5)
    from pxr import PhysxSchema

    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateLinearDampingAttr(3.0)   # the tray stays where it is left
    pxrb.CreateAngularDampingAttr(2.0)
    _box_part(stage, prim_path, "plate", (cfg.plate_x, cfg.plate_y, cfg.plate_t),
              (0.0, 0.0, 0.0), cfg.plate_color, cfg.contact_offset)
    _box_part(stage, prim_path, "fin", (cfg.fin_x, cfg.fin_t, cfg.fin_h),
              (0.0, cfg.fin_y_local, cfg.plate_t / 2 + cfg.fin_h / 2),
              cfg.fin_color, cfg.contact_offset)
    return root


def _spawn_basket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Free open-top basket, root at the BASE center: floor slab + 4 walls (12 mm — the
    proven rim-pinch wall thickness)."""
    stage, root = _apply_root(prim_path, translation, orientation, kinematic=False,
                              mass=cfg.mass_props.mass, max_depen=0.5)
    from pxr import PhysxSchema

    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.20)
    ih, wt, oh = cfg.inner_half, cfg.wall_t, cfg.inner_half + cfg.wall_t
    co, col = cfg.contact_offset, cfg.color
    _box_part(stage, prim_path, "floor", (2 * oh, 2 * oh, cfg.floor_t),
              (0.0, 0.0, cfg.floor_t / 2), col, co)
    for sy in (-1.0, 1.0):
        _box_part(stage, prim_path, f"wall_y{'p' if sy > 0 else 'n'}",
                  (2 * oh, wt, cfg.wall_h), (0.0, sy * (ih + wt / 2),
                                             cfg.floor_t + cfg.wall_h / 2), col, co)
    for sx in (-1.0, 1.0):
        _box_part(stage, prim_path, f"wall_x{'p' if sx > 0 else 'n'}",
                  (wt, 2 * ih, cfg.wall_h), (sx * (ih + wt / 2), 0.0,
                                             cfg.floor_t + cfg.wall_h / 2), col, co)
    return root


def _housing_spawner_cfg(c: ButterHopperSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "housing" not in _SPAWNER_CACHE:

        @configclass
        class HousingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_housing)
            shaft_inner_half: float = 0.058
            shaft_wall_t: float = 0.012
            shaft_z_bot: float = 0.352
            shaft_z_top: float = 0.482
            lid_t: float = 0.030
            column_x: float = 0.175
            column_size: tuple = (0.07, 0.09)
            pad_side: float = 0.22
            pad_color: tuple = (0.15, 0.60, 0.20)
            housing_color: tuple = (0.25, 0.30, 0.38)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["housing"] = HousingSpawnerCfg

    return _SPAWNER_CACHE["housing"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        shaft_inner_half=c.shaft_inner_half, shaft_wall_t=c.shaft_wall_t,
        shaft_z_bot=c.shaft_z_bot, shaft_z_top=c.shaft_z_top, lid_t=c.lid_t,
        column_x=c.column_dx, column_size=c.column_size,
        pad_side=c.pad_side, pad_color=c.pad_color,
        housing_color=c.housing_color, contact_offset=c.contact_offset,
    )


def _tray_spawner_cfg(c: ButterHopperSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tray" not in _SPAWNER_CACHE:

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            plate_x: float = 0.132
            plate_y: float = 0.250
            plate_t: float = 0.014
            fin_x: float = 0.048
            fin_t: float = 0.012
            fin_h: float = 0.075
            fin_y_local: float = -0.117
            plate_color: tuple = (0.55, 0.55, 0.58)
            fin_color: tuple = (0.95, 0.80, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["tray"] = TraySpawnerCfg

    return _SPAWNER_CACHE["tray"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.tray_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        plate_x=c.tray_plate[0], plate_y=c.tray_plate[1], plate_t=c.tray_plate[2],
        fin_x=c.fin_size[0], fin_t=c.fin_size[1], fin_h=c.fin_size[2],
        fin_y_local=c.fin_y_local, plate_color=c.tray_color, fin_color=c.fin_color,
        contact_offset=c.contact_offset,
    )


def _basket_spawner_cfg(c: ButterHopperSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "basket" not in _SPAWNER_CACHE:

        @configclass
        class BasketSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_basket)
            inner_half: float = 0.085
            wall_t: float = 0.012
            wall_h: float = 0.095
            floor_t: float = 0.010
            color: tuple = (0.72, 0.52, 0.28)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["basket"] = BasketSpawnerCfg

    return _SPAWNER_CACHE["basket"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.basket_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_half=c.basket_inner_half, wall_t=c.basket_wall_t, wall_h=c.basket_wall_h,
        floor_t=c.basket_floor_t, color=c.basket_color, contact_offset=c.contact_offset,
    )


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class ButterHopperSceneCfg(BaseCfg):
    """Config for `ButterHopperScene`. All geometry laid out for a Franka base at
    (-0.50, 0, 0); the dispenser mechanism is FIXED (jointed pairs must not teleport
    on this PhysX stack), free bodies carry the per-episode randomization."""

    # --- tunable: rubric thresholds -----------------------------------------------------------
    open_min: float = tunable(0.14)        # tray opening (m) that counts as "open" (travel 0.19)
    stage_r: float = tunable(0.030)        # basket center within this of the shaft axis = staged
    contain_xy_margin: float = tunable(0.010)  # butter center inside inner_half - margin
    contain_z_lo: float = tunable(0.004)   # butter center above this (basket frame)
    contain_z_hi: float = tunable(0.080)   # butter center below this (basket frame; rim is 0.105)
    upright_max_deg: float = tunable(20.0)  # basket tilt beyond this voids containment
    hold_steps: int = tunable(60)          # consecutive substeps success must be sustained
    settle_speed: float = tunable(0.05)    # max |v| when judging butter/basket (m/s)
    floored_z: float = tunable(0.022)      # butter center below this while NOT contained
    # latches the irreversible-failure flag (bare floor = 0.018; basket floor = 0.028+)

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    basket_bearing_deg: tuple = tunable((25.0, 55.0))  # |bearing| from base heading, side random
    basket_radius: tuple = tunable((0.44, 0.54))       # distance from the planned base (m)
    basket_yaw_deg: float = tunable(180.0)             # uniform +/- yaw at reset
    butter_jitter: float = tunable(0.004)              # +/- xy jitter of the slab in the shaft
    butter_yaw_deg: float = tunable(3.0)               # +/- yaw jitter of the slab

    # --- tunable: placement (laid out for a Franka base at (-0.50, 0, 0)) ----------------------
    stage_anchor: tuple = tunable((-0.50, 0.0))  # planned robot base (solve.py uses this)
    hopper_xy: tuple = tunable((0.10, 0.0))      # shaft axis == drop zone center

    # --- info: dispenser structure --------------------------------------------------------------
    shaft_inner_half: float = info(0.058)   # shaft interior 116 mm square
    shaft_wall_t: float = info(0.012)
    shaft_z_bot: float = info(0.352)        # wall band bottom (2 mm above the tray top)
    shaft_z_top: float = info(0.482)
    lid_t: float = info(0.030)
    column_dx: float = info(0.175)          # rear column x offset from the shaft axis
    column_size: tuple = info((0.07, 0.09))
    pad_side: float = info(0.22)
    tray_plate: tuple = info((0.132, 0.250, 0.014))  # plate x, y, t; top at z 0.350 closed
    tray_z: float = info(0.343)             # plate center height (home)
    stop_size: tuple = info((0.06, 0.014, 0.06))  # +y end stop (its own kinematic body:
    stop_y: float = info(0.135)             # tray-housing pair collision is joint-disabled)
    tray_travel: float = info(0.19)         # prismatic limit (symmetric hedge: +/- travel)
    tray_mass: float = info(0.20)
    fin_size: tuple = info((0.048, 0.012, 0.075))  # handle fin w(x), t(y), h — the 48 mm
    fin_y_local: float = info(-0.117)       # width is the pinchable feature (jaw along x)
    # --- info: butter + basket ------------------------------------------------------------------
    butter_size: tuple = info((0.096, 0.096, 0.036))  # > 80 mm jaw aperture in x AND y
    butter_mass: float = info(0.22)
    basket_inner_half: float = info(0.085)
    basket_wall_t: float = info(0.012)      # the proven rim-pinch wall thickness
    basket_wall_h: float = info(0.095)
    basket_floor_t: float = info(0.010)
    basket_mass: float = info(0.30)
    # --- info: colors / misc ----------------------------------------------------------------------
    housing_color: tuple = info((0.25, 0.30, 0.38))
    tray_color: tuple = info((0.55, 0.55, 0.58))
    fin_color: tuple = info((0.95, 0.80, 0.10))
    butter_color: tuple = info((0.96, 0.88, 0.55))
    basket_color: tuple = info((0.72, 0.52, 0.28))
    pad_color: tuple = info((0.15, 0.60, 0.20))
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    butter_home_z: float = field(default=None, init=False)   # slab center z at reset
    basket_rim_local: float = field(default=None, init=False)  # rim top in basket frame

    def __post_init__(self) -> None:
        self.butter_home_z = round(self.tray_z + self.tray_plate[2] / 2
                                   + self.butter_size[2] / 2 + 0.002, 4)
        self.basket_rim_local = round(self.basket_floor_t + self.basket_wall_h, 4)


# ----- scene -------------------------------------------------------------------------------------
@SCENES.register("butter_hopper")
class ButterHopperScene(BaseScene):
    cfg: ButterHopperSceneCfg

    def __init__(self, cfg: ButterHopperSceneCfg | None = None) -> None:
        super().__init__(cfg or ButterHopperSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        hx, hy = c.hopper_xy
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "housing": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Housing",
                spawn=_housing_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, 0.0)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=_tray_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, c.tray_z)),
            ),
            "butter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Butter",
                spawn=sim_utils.CuboidCfg(
                    size=c.butter_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.butter_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.butter_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, c.butter_home_z)),
            ),
            # +y end stop for the tray: a SEPARATE kinematic body — the joint disables
            # all tray-housing collision, so a housing part could not block overtravel
            "stop": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/TrayStop",
                spawn=sim_utils.CuboidCfg(
                    size=c.stop_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.housing_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy + c.stop_y, c.tray_z)),
            ),
            "basket": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Basket",
                spawn=_basket_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stage_anchor[0] + 0.48, c.stage_anchor[1] + 0.30, 0.002)),
            ),
        }

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
        n = env.num_envs
        dev = env.device
        self.housing: RigidObject = env.iscene["housing"]
        self.tray: RigidObject = env.iscene["tray"]
        self.stop: RigidObject = env.iscene["stop"]
        self.butter: RigidObject = env.iscene["butter"]
        self.basket: RigidObject = env.iscene["basket"]
        self.env_origins = env.iscene.env_origins
        # Tray home y is a constant of the fixed mechanism.
        self._tray_home_y = self.env_origins[:, 1] + c.hopper_xy[1]
        # Latches + the sustained-success counter.
        self._staged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._open_max = torch.zeros(n, device=dev)
        self._delivered = torch.zeros(n, dtype=torch.bool, device=dev)
        self._floored = torch.zeros(n, dtype=torch.bool, device=dev)
        self._hold = torch.zeros(n, dtype=torch.long, device=dev)
        self._author_joints()

    def _author_joints(self) -> None:
        """Per env: the tray's Y prismatic slide on the housing — pair collision
        disabled (the tray runs 2 mm under the shaft walls), SYMMETRIC limits
        +/- travel (the microwave GPU lesson: a one-sided range can pin the body at
        zero under either joint-coordinate sign convention)."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/tray_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Housing"])
            j.CreateBody1Rel().SetTargets([f"{base}/Tray"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.tray_z))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-c.tray_travel)
            j.CreateUpperLimitAttr(c.tray_travel)
            lim = PhysxSchema.PhysxLimitAPI.Apply(j.GetPrim(), "linear")
            if hasattr(lim, "CreateContactDistanceAttr"):  # removed in Isaac Sim 5.1 schema
                lim.CreateContactDistanceAttr(0.001)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: housing + tray re-pinned at the FIXED authored pose (the
        housing never moves; the tray is the joint FOLLOWER — follower-only re-posing
        is safe on this stack), butter re-seated in the shaft with xy/yaw jitter,
        basket re-spawned on a random polar slot from the planned base; latches
        cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        hx, hy = c.hopper_xy

        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = hx, hy, 0.0
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.housing.write_root_state_to_sim(st, env_ids)

        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = hx, hy, c.tray_z
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.tray.write_root_state_to_sim(st, env_ids)

        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = hx, hy + c.stop_y, c.tray_z
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.stop.write_root_state_to_sim(st, env_ids)

        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = hx + (torch.rand(m, device=dev) * 2 - 1) * c.butter_jitter
        st[:, 1] = hy + (torch.rand(m, device=dev) * 2 - 1) * c.butter_jitter
        st[:, 2] = c.butter_home_z
        bhalf = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.butter_yaw_deg) / 2
        st[:, 3] = torch.cos(bhalf)
        st[:, 6] = torch.sin(bhalf)
        st[:, 0:3] += origin
        self.butter.write_root_state_to_sim(st, env_ids)

        # basket: polar slot from the planned robot base (reach by construction)
        b0, b1 = (math.radians(v) for v in c.basket_bearing_deg)
        r0, r1 = c.basket_radius
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.full((m,), -1.0, device=dev),
                           torch.full((m,), 1.0, device=dev))
        bear = (b0 + torch.rand(m, device=dev) * (b1 - b0)) * side
        rad = r0 + torch.rand(m, device=dev) * (r1 - r0)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.stage_anchor[0] + rad * torch.cos(bear)
        st[:, 1] = c.stage_anchor[1] + rad * torch.sin(bear)
        st[:, 2] = 0.003
        yhalf = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.basket_yaw_deg) / 2
        st[:, 3] = torch.cos(yhalf)
        st[:, 6] = torch.sin(yhalf)
        st[:, 0:3] += origin
        self.basket.write_root_state_to_sim(st, env_ids)

        self._staged[env_ids] = False
        self._open_max[env_ids] = 0.0
        self._delivered[env_ids] = False
        self._floored[env_ids] = False
        self._hold[env_ids] = 0

    # ----- geometry queries -----------------------------------------------------------------------
    def opening(self) -> torch.Tensor:
        """(N,) tray opening (m): how far the tray has been pulled out along -y."""
        return (self._tray_home_y - self.tray.data.root_pos_w[:, 1]).clamp(min=0.0)

    def shaft_axis_w(self) -> torch.Tensor:
        """(N, 2) the shaft axis / drop zone center in world xy."""
        c = self.cfg
        return self.env_origins[:, :2] + torch.tensor(
            [c.hopper_xy[0], c.hopper_xy[1]], device=self.env.device)

    def _basket_frame(self, p_w: torch.Tensor) -> torch.Tensor:
        """World points (N, 3) -> basket body frame (root at base center)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.basket.data.root_quat_w,
                                  p_w - self.basket.data.root_pos_w)

    def _up_z(self, body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)

    def basket_upright(self) -> torch.Tensor:
        return self._up_z(self.basket) >= math.cos(math.radians(self.cfg.upright_max_deg))

    def staged_now(self) -> torch.Tensor:
        """(N,) bool: basket upright, settled, on the ground, centered on the drop zone."""
        c = self.cfg
        bp = self.basket.data.root_pos_w
        near = (bp[:, :2] - self.shaft_axis_w()).norm(dim=-1) <= c.stage_r
        on_ground = (bp[:, 2] - self.env_origins[:, 2]) < 0.030
        still = self.basket.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return near & on_ground & still & self.basket_upright()

    def contained(self) -> torch.Tensor:
        """(N,) bool: butter center inside the basket interior (basket frame), between
        the floor and well below the rim, basket upright. Pure geometry — settledness
        is judged separately."""
        c = self.cfg
        loc = self._basket_frame(self.butter.data.root_pos_w)
        inside_xy = loc[:, :2].abs().amax(dim=-1) <= (c.basket_inner_half
                                                      - c.contain_xy_margin)
        inside_z = (loc[:, 2] > c.contain_z_lo) & (loc[:, 2] <= c.contain_z_hi)
        return inside_xy & inside_z & self.basket_upright()

    def butter_settled(self) -> torch.Tensor:
        return self.butter.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def basket_settled(self) -> torch.Tensor:
        return self.basket.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def success_now(self) -> torch.Tensor:
        """(N,) bool, instantaneous: butter resting inside the basket, tray open past
        `open_min` (any honest delivery leaves the dispenser open — the butter cannot
        leave the shaft otherwise), and the butter has NEVER lain on the bare floor
        (the `_floored` latch makes the wrong-order drop irreversible in the rubric,
        not just in narration: capturing a floored slab by lowering the basket over it
        does not count)."""
        return (self.contained() & self.butter_settled() & self.basket_settled()
                & (self.opening() >= self.cfg.open_min) & ~self._floored)

    # ----- mechanics (every substep) ----------------------------------------------------------------
    def post_step(self) -> None:
        self._staged |= self.staged_now()
        self._open_max = torch.maximum(self._open_max, self.opening())
        # floored: butter center at bare-floor height while NOT inside the basket
        # (an honest drop is `contained` before its center ever gets this low)
        low = (self.butter.data.root_pos_w[:, 2] - self.env_origins[:, 2]
               ) < self.cfg.floored_z
        self._floored |= low & ~self.contained()
        self._delivered |= (self.contained() & self.butter_settled() & ~self._floored)
        now = self.success_now()
        self._hold = torch.where(now, self._hold + 1, torch.zeros_like(self._hold))

    # ----- state (full, restorable) -----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "housing": self.housing.data.root_state_w[env_ids].clone(),
            "stop": self.stop.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "butter": self.butter.data.root_state_w[env_ids].clone(),
            "basket": self.basket.data.root_state_w[env_ids].clone(),
            "latches": {k: getattr(self, k)[env_ids].clone()
                        for k in ("_staged", "_open_max", "_delivered", "_floored",
                                  "_hold")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.housing.write_root_state_to_sim(state["housing"], env_ids)
        self.stop.write_root_state_to_sim(state["stop"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.butter.write_root_state_to_sim(state["butter"], env_ids)
        self.basket.write_root_state_to_sim(state["basket"], env_ids)
        for k, v in state["latches"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ------------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A dark slate-blue BUTTER DISPENSER stands on the floor: a capped square "
            f"hopper shaft (interior {2 * c.shaft_inner_half * 100:.1f} cm) held about "
            f"{c.shaft_z_bot * 100:.0f} cm above the ground by a rear support column, with "
            f"a bright GREEN square painted on the floor directly under it (the drop "
            f"zone). A pale-yellow butter slab "
            f"({c.butter_size[0] * 100:.1f} x {c.butter_size[1] * 100:.1f} x "
            f"{c.butter_size[2] * 100:.1f} cm) is loaded inside the shaft, resting on the "
            f"dispenser's grey SLIDING TRAY — the shaft's only floor. The tray sticks out "
            f"of one side of the dispenser and carries an upright bright-YELLOW handle "
            f"fin ({c.fin_size[0] * 1000:.0f} mm wide, {c.fin_size[1] * 1000:.0f} mm "
            f"thick; pinch it across its width) on its outer end. Elsewhere on "
            f"the floor stands an empty tan open-top BASKET (interior "
            f"{2 * c.basket_inner_half * 100:.0f} cm square, walls "
            f"{c.basket_wall_h * 100:.0f} cm high).\n"
            f"Goal: dispense the butter INTO the basket — end with the butter slab "
            f"resting inside the basket and the tray left pulled open. Move the basket "
            f"first: set it upright on the green drop-zone square, centered under the "
            f"shaft (within ~{c.stage_r * 100:.0f} cm). Then grip the yellow handle fin "
            f"and pull the tray straight out along its slide (away from the dispenser, "
            f"at least {c.open_min * 100:.0f} cm): the shaft wall scrapes the slab off "
            f"the receding tray and it falls out of the shaft's open bottom into "
            f"whatever is below. ORDER MATTERS and the drop is IRREVERSIBLE: the slab is "
            f"wider than a parallel-jaw gripper's opening in both directions, so it can "
            f"never be grasped — if it is dispensed onto the bare floor the episode can no "
            f"longer be completed (trapping a floor-dropped slab by lowering the basket "
            f"over it does NOT count: success requires butter that never lay on the "
            f"bare floor). The butter itself must never be carried by the robot; only "
            f"the basket and the tray handle are manipulated."
        )

    # ----- progress / rubric ----------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: success_now sustained for `hold_steps` consecutive substeps."""
        return self._hold >= self.cfg.hold_steps

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30 latched once the basket has been staged on the
        drop zone; +0.30 scaled by the latched max tray opening (capped at open_min);
        0.85 latched once the butter has settled inside the basket; 1.0 iff success."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        s = torch.zeros(n, device=dev)
        s = s + 0.30 * self._staged.float()
        s = s + 0.30 * (self._open_max / c.open_min).clamp(0.0, 1.0)
        s = torch.where(self._delivered, torch.maximum(s, torch.full_like(s, 0.85)), s)
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("simgen", lambda: EnvCfg(scene="butter_hopper", robot="null", env_spacing=3))
