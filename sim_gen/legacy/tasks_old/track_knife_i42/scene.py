"""ButterStationScene — load the knife with butter and frost the tart, cell by cell
(sim_gen task `track_knife_i42`, derived from pick_place/track_knife).

The seed is stage-3 trajectory TRACKING: the knife starts already grasped (states from a
pkl), five XFORM markers prescribe the exact free-space path, randomization is zeroed,
the gripper is forced closed and the episode dies if the knife is released. The whole
skill is carrying a held object along a given aerial curve, never touching anything.

This task inverts every load-bearing pillar. The knife is not a payload but a
CONSUMABLE-TRANSFER TOOL with a finite charge:

  1. nothing starts in hand — the knife lies in a cradle (its rest);
  2. there is no prescribed path and no markers — the stations (butter dish, tart jig,
     serving plate, knife rest) are re-sampled around the workspace every episode;
  3. the work is contact-band COVERAGE, not free-space transit: the tart's top face is a
     4x3 grid of cells; a cell is frosted only while the blade tip sweeps through a thin
     height band over that cell (judged in the TART'S BODY FRAME), blade flat, in a
     continuous stroke (see the entry latch below);
  4. the knife's charge is finite: one dip in the butter dish loads `capacity` (4) cells'
     worth; frosting decrements it, at zero nothing accrues — the solver must interleave
     dip trips with strokes (12 cells => at least 3 dips), a recharge/apply alternation
     the seed has no concept of;
  5. the payload transport that IS here (tart -> plate) is a final stage gated on full
     coverage, and the knife must end where it began (net tool transport = zero).

Anti-teleport (transit-latch family, entry-latched): frosting and dipping only accrue
while a "stroke" is ACTIVE, and a stroke activates only by ENTERING the working zone
with a small per-substep tip displacement (< `tip_step_max`). A knife teleported into a
perfect frosting pose arrives with a huge step displacement, never activates, and can
only activate by leaving the zone and re-entering continuously — i.e. by honestly
stroking. The same latch guards the dip dwell.

Rubric (score in [0, 1], ~0 for doing nothing, 1.0 iff success):
  +0.10  loaded_ever latch (the knife has been charged at least once)
  +0.50 * frosted fraction (12 latched cells)
  +0.10  all cells frosted
  +0.15  tart served: resting settled + upright on the plate — GATED on full frosting
         (serving an unfrosted tart scores nothing: the seed's transport plan fails)
  +0.15  knife parked back in its rest — GATED on served + frosted
success() = all frosted & tart settled on plate & knife settled in rest.

Assets are fully procedural compound spawners (pen_holder pattern): dynamic knife
(handle + blade, flat underside) and tart (box + 12 visual-only cell tiles that recolor
cream as they frost); kinematic fixtures (tart jig with lip walls, octagonal butter
dish, serving plate, knife cradle) re-posed per reset via pose writes. Heavy imports
(isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- compound spawners -----------------------------------------------------------------------
# One rigid body per object, several child colliders + visual-only decoration, authored with raw
# pxr APIs; `isaaclab.sim.utils.clone` provides the per-env replication (the pen_holder pattern).

_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_root(stage, prim_path: str, translation, orientation, *, kinematic: bool,
                mass: float, max_depen: float = 0.5):
    """Author the common root Xform: RigidBodyAPI (+kinematic flag), MassAPI, depenetration
    cap + light damping (dynamic bodies)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

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
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(max_depen)
    if not kinematic:
        pxrb.CreateLinearDampingAttr(0.05)
        pxrb.CreateAngularDampingAttr(0.05)
    return root


def _box(stage, path: str, center, size, color, *, contact_offset: float | None):
    """A unit cube scaled to `size` at `center`; collider iff contact_offset is not None."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    b = UsdGeom.Cube.Define(stage, path)
    b.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(b.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    b.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        UsdPhysics.CollisionAPI.Apply(b.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(b.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
    return b


def _spawn_knife(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Butter knife: blade (thin box) along +x from the root, handle along -x, BOTTOM FACES
    FLUSH at z = -blade_t/2 (a flat underside, so the handle never fouls the tart while the
    blade sweeps its top). Tip point local = (blade_l, 0, -blade_t/2)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _apply_root(stage, prim_path, translation, orientation,
                       kinematic=False, mass=cfg.mass_props.mass)
    co = float(cfg.contact_offset)
    zb = -cfg.blade_t / 2  # common bottom plane
    _box(stage, f"{prim_path}/blade",
         (cfg.blade_l / 2, 0.0, 0.0),
         (cfg.blade_l, cfg.blade_w, cfg.blade_t), cfg.blade_color, contact_offset=co)
    _box(stage, f"{prim_path}/handle",
         (-cfg.handle_l / 2, 0.0, zb + cfg.handle_h / 2),
         (cfg.handle_l, cfg.handle_w, cfg.handle_h), cfg.handle_color, contact_offset=co)
    return root


def _spawn_tart(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Tart: one box collider + nx*ny VISUAL-ONLY cell tiles floating just above the top
    face (no CollisionAPI — recolored cream as cells frost)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _apply_root(stage, prim_path, translation, orientation,
                       kinematic=False, mass=cfg.mass_props.mass)
    lx, ly, lz = cfg.size
    _box(stage, f"{prim_path}/body", (0.0, 0.0, 0.0), (lx, ly, lz), cfg.color,
         contact_offset=float(cfg.contact_offset))
    nx, ny = cfg.grid
    dx, dy = lx / nx, ly / ny
    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            _box(stage, f"{prim_path}/cell_{k}",
                 (-lx / 2 + (i + 0.5) * dx, -ly / 2 + (j + 0.5) * dy, lz / 2 + 0.0012),
                 (dx * 0.90, dy * 0.90, 0.001), cfg.cell_color, contact_offset=None)
    return root


def _spawn_jig(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Prep jig (kinematic): base slab (top at z=base_t, root at BOTTOM plane) + 4 lip
    walls enclosing a pocket that holds the tart still while it is frosted."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _apply_root(stage, prim_path, translation, orientation, kinematic=True, mass=1.0)
    co = float(cfg.contact_offset)
    bx, by, bt = cfg.base_size
    px, py = cfg.pocket
    wt, wh = cfg.lip_t, cfg.lip_h
    _box(stage, f"{prim_path}/base", (0.0, 0.0, bt / 2), (bx, by, bt), cfg.color,
         contact_offset=co)
    for k, (cx, cy, sx, sy) in enumerate((
            (0.0, py / 2 + wt / 2, px + 2 * wt, wt),
            (0.0, -py / 2 - wt / 2, px + 2 * wt, wt),
            (px / 2 + wt / 2, 0.0, wt, py),
            (-px / 2 - wt / 2, 0.0, wt, py))):
        _box(stage, f"{prim_path}/lip_{k}", (cx, cy, bt + wh / 2), (sx, sy, wh),
             cfg.color, contact_offset=co)
    return root


def _spawn_dish(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Butter dish (kinematic): octagonal open cup, root at the BOTTOM plane — bottom disc
    + 8 wall boxes (pen_holder shell geometry) + a VISUAL-ONLY butter block inside."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _apply_root(stage, prim_path, translation, orientation, kinematic=True, mass=1.0)
    co = float(cfg.contact_offset)
    n = 8
    outer_r = cfg.inner_r + cfg.wall_t
    bot = UsdGeom.Cylinder.Define(stage, f"{prim_path}/bottom")
    bot.CreateRadiusAttr(outer_r)
    bot.CreateHeightAttr(cfg.bot_t)
    bot.CreateExtentAttr([Gf.Vec3f(-outer_r, -outer_r, -cfg.bot_t / 2),
                          Gf.Vec3f(outer_r, outer_r, cfg.bot_t / 2)])
    UsdGeom.Xformable(bot.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, cfg.bot_t / 2))
    bot.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    UsdPhysics.CollisionAPI.Apply(bot.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(bot.GetPrim())
    px.CreateContactOffsetAttr(co)
    px.CreateRestOffsetAttr(0.0)
    r_mid = cfg.inner_r + cfg.wall_t / 2
    seg_len = 2 * outer_r * math.tan(math.pi / n) + 0.002
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang),
                                          cfg.height / 2))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(cfg.wall_t, seg_len, cfg.height))
        seg.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        px.CreateContactOffsetAttr(co)
        px.CreateRestOffsetAttr(0.0)
    _box(stage, f"{prim_path}/butter",
         (0.0, 0.0, cfg.bot_t + cfg.butter_h / 2),
         (cfg.butter_w, cfg.butter_w, cfg.butter_h), cfg.butter_color, contact_offset=None)
    return root


def _spawn_rest(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Knife cradle (kinematic): slab (root at BOTTOM plane) + two rails along +/-y forming
    a channel the knife lies in along local x."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _apply_root(stage, prim_path, translation, orientation, kinematic=True, mass=1.0)
    co = float(cfg.contact_offset)
    sx, sy, st = cfg.slab_size
    _box(stage, f"{prim_path}/slab", (0.0, 0.0, st / 2), (sx, sy, st), cfg.color,
         contact_offset=co)
    for k, sgn in enumerate((1.0, -1.0)):
        _box(stage, f"{prim_path}/rail_{k}",
             (0.0, sgn * (cfg.channel_w / 2 + cfg.rail_t / 2), st + cfg.rail_h / 2),
             (sx, cfg.rail_t, cfg.rail_h), cfg.color, contact_offset=co)
    return root


def _compound_cfg(key: str, fields: dict, mass: float) -> Any:
    """Build (lazily, app required) a compound spawner cfg wrapping the matching spawn
    function with `clone` — the pen_holder pattern, one explicit configclass per body."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if not _SPAWNER_CACHE:

        @configclass
        class KnifeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_knife)
            blade_l: float = 0.110
            blade_w: float = 0.024
            blade_t: float = 0.004
            handle_l: float = 0.095
            handle_w: float = 0.022
            handle_h: float = 0.016
            blade_color: tuple = (0.62, 0.64, 0.68)
            handle_color: tuple = (0.25, 0.16, 0.10)
            contact_offset: float = 0.002

        @configclass
        class TartSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tart)
            size: tuple = (0.140, 0.100, 0.024)
            grid: tuple = (4, 3)
            color: tuple = (0.78, 0.58, 0.30)
            cell_color: tuple = (0.42, 0.26, 0.13)
            contact_offset: float = 0.002

        @configclass
        class JigSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_jig)
            base_size: tuple = (0.20, 0.16, 0.010)
            pocket: tuple = (0.146, 0.106)
            lip_t: float = 0.008
            lip_h: float = 0.018
            color: tuple = (0.45, 0.32, 0.20)
            contact_offset: float = 0.002

        @configclass
        class DishSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_dish)
            inner_r: float = 0.045
            wall_t: float = 0.006
            height: float = 0.036
            bot_t: float = 0.006
            butter_w: float = 0.050
            butter_h: float = 0.014
            color: tuple = (0.20, 0.45, 0.50)
            butter_color: tuple = (0.98, 0.86, 0.25)
            contact_offset: float = 0.002

        @configclass
        class RestSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rest)
            slab_size: tuple = (0.24, 0.07, 0.006)
            channel_w: float = 0.044
            rail_t: float = 0.008
            rail_h: float = 0.014
            color: tuple = (0.30, 0.30, 0.34)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(knife=KnifeSpawnerCfg, tart=TartSpawnerCfg, jig=JigSpawnerCfg,
                              dish=DishSpawnerCfg, rest=RestSpawnerCfg)

    return _SPAWNER_CACHE[key](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        **fields,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ButterStationSceneCfg(BaseCfg):
    """Config for `ButterStationScene`. Env-local frame: tart jig near the origin; dish,
    plate and knife rest on sampled bearings around it."""

    # --- tunable: frosting mechanics ---------------------------------------------------------
    capacity: int = tunable(4)            # cells one butter dip is good for
    band_lo: float = tunable(0.003)       # blade tip may reach this far BELOW the tart top (m)
    band_hi: float = tunable(0.005)       # ... and this far above it (the working band)
    flat_max_deg: float = tunable(30.0)   # blade plane within this of the tart top plane
    tip_step_max: float = tunable(0.006)  # per-substep tip travel above this = teleport (m)
    dip_dwell_substeps: int = tunable(8)  # continuous in-dish substeps to load the knife
    dip_below_rim: float = tunable(0.004) # tip must be at least this below the dish rim (m)
    dip_margin_r: float = tunable(0.007)  # tip within (inner_r - this) of the dish axis
    paint_tart_vmax: float = tunable(0.10)  # tart |lin vel| gate while frosting (m/s)

    # --- tunable: serve / park thresholds ----------------------------------------------------
    serve_xy_tol: float = tunable(0.050)  # tart center within this of the plate axis (m)
    serve_z_tol: float = tunable(0.010)   # tart bottom within this of the plate top (m)
    upright_tol_deg: float = tunable(15.0)
    park_x_tol: float = tunable(0.075)    # knife root inside the cradle channel (rest frame)
    park_y_tol: float = tunable(0.017)
    park_z_max: float = tunable(0.045)
    park_flat_deg: float = tunable(35.0)
    settle_lin: float = tunable(0.05)     # max |lin vel| when judging settled (m/s)

    # --- tunable: rubric weights -------------------------------------------------------------
    w_load: float = tunable(0.10)
    w_cov: float = tunable(0.50)
    w_full: float = tunable(0.10)
    w_serve: float = tunable(0.15)
    w_park: float = tunable(0.15)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    jig_jitter: float = tunable(0.025)          # tart jig xy jitter (m)
    jig_yaw_deg: float = tunable(180.0)         # tart jig (and tart) yaw, uniform +/-
    dish_arc: tuple = tunable((70.0, 140.0))    # dish bearing (deg) around the jig
    dish_rr: tuple = tunable((0.30, 0.36))
    plate_arc: tuple = tunable((200.0, 260.0))  # plate bearing (deg)
    plate_rr: tuple = tunable((0.32, 0.38))
    rest_arc: tuple = tunable((-40.0, 20.0))    # knife-rest bearing (deg)
    rest_rr: tuple = tunable((0.30, 0.36))
    rest_yaw_deg: float = tunable(180.0)        # cradle channel yaw, uniform +/-

    # --- info: structure ---------------------------------------------------------------------
    grid: tuple = info((4, 3))                  # frosting cells (nx along x, ny along y)
    tart_size: tuple = info((0.140, 0.100, 0.024))
    tart_mass: float = info(0.15)
    tart_color: tuple = info((0.78, 0.58, 0.30))
    cell_color: tuple = info((0.42, 0.26, 0.13))     # unfrosted cell tile (dark chocolate)
    frosted_color: tuple = info((0.96, 0.93, 0.80))  # frosted cell tile (cream)
    blade_l: float = info(0.110)
    blade_w: float = info(0.024)
    blade_t: float = info(0.004)
    handle_l: float = info(0.095)
    handle_w: float = info(0.022)
    handle_h: float = info(0.016)
    knife_mass: float = info(0.06)
    blade_color: tuple = info((0.62, 0.64, 0.68))    # steel; tinted yellow while charged
    blade_loaded_color: tuple = info((0.93, 0.83, 0.35))
    handle_color: tuple = info((0.25, 0.16, 0.10))
    jig_base: tuple = info((0.20, 0.16, 0.010))
    jig_pocket: tuple = info((0.146, 0.106))    # 3 mm clearance per side around the tart
    jig_lip_t: float = info(0.008)
    jig_lip_h: float = info(0.018)              # above the base top; tart stands 6 mm proud
    jig_color: tuple = info((0.45, 0.32, 0.20))
    dish_inner_r: float = info(0.045)           # octagon inradius
    dish_wall_t: float = info(0.006)
    dish_h: float = info(0.036)
    dish_bot_t: float = info(0.006)
    dish_color: tuple = info((0.20, 0.45, 0.50))
    butter_w: float = info(0.050)
    butter_h: float = info(0.014)
    butter_color: tuple = info((0.98, 0.86, 0.25))
    plate_r: float = info(0.085)
    plate_h: float = info(0.008)
    plate_color: tuple = info((0.92, 0.92, 0.95))
    rest_slab: tuple = info((0.24, 0.07, 0.006))
    rest_channel_w: float = info(0.044)
    rest_rail_t: float = info(0.008)
    rest_rail_h: float = info(0.014)
    rest_color: tuple = info((0.30, 0.30, 0.34))
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    n_cells: int = field(default=None, init=False)
    tip_local: tuple = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.n_cells = self.grid[0] * self.grid[1]
        self.tip_local = (self.blade_l, 0.0, -self.blade_t / 2)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("butter_station")
class ButterStationScene(BaseScene):
    cfg: ButterStationSceneCfg

    def __init__(self, cfg: ButterStationSceneCfg | None = None) -> None:
        super().__init__(cfg or ButterStationSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, dynamic knife + tart, kinematic jig/dish/plate/rest (all re-posed
        by reset)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
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
            "knife": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Knife",
                spawn=_compound_cfg("knife", dict(
                    blade_l=c.blade_l, blade_w=c.blade_w, blade_t=c.blade_t,
                    handle_l=c.handle_l, handle_w=c.handle_w, handle_h=c.handle_h,
                    blade_color=c.blade_color, handle_color=c.handle_color,
                    contact_offset=c.contact_offset), c.knife_mass),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.33, 0.0, 0.012)),
            ),
            "tart": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tart",
                spawn=_compound_cfg("tart", dict(
                    size=c.tart_size, grid=c.grid, color=c.tart_color,
                    cell_color=c.cell_color, contact_offset=c.contact_offset), c.tart_mass),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, c.jig_base[2] + c.tart_size[2] / 2 + 0.002)),
            ),
            "jig": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Jig",
                spawn=_compound_cfg("jig", dict(
                    base_size=c.jig_base, pocket=c.jig_pocket, lip_t=c.jig_lip_t,
                    lip_h=c.jig_lip_h, color=c.jig_color,
                    contact_offset=c.contact_offset), 1.0),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "dish": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dish",
                spawn=_compound_cfg("dish", dict(
                    inner_r=c.dish_inner_r, wall_t=c.dish_wall_t, height=c.dish_h,
                    bot_t=c.dish_bot_t, butter_w=c.butter_w, butter_h=c.butter_h,
                    color=c.dish_color, butter_color=c.butter_color,
                    contact_offset=c.contact_offset), 1.0),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.33, 0.0)),
            ),
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate",
                spawn=sim_utils.CylinderCfg(
                    radius=c.plate_r, height=c.plate_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.plate_color)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.35, c.plate_h / 2)),
            ),
            "rest": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rest",
                spawn=_compound_cfg("rest", dict(
                    slab_size=c.rest_slab, channel_w=c.rest_channel_w,
                    rail_t=c.rest_rail_t, rail_h=c.rest_rail_h, color=c.rest_color,
                    contact_offset=c.contact_offset), 1.0),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.33, 0.0, 0.0)),
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
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the frosting state machine buffers + cache the per-env
        displayColor attrs used to paint frosted cells / the charged blade."""
        super().bind(env)
        c = self.cfg
        n, dev = env.num_envs, env.device
        self.knife: RigidObject = env.iscene["knife"]
        self.tart: RigidObject = env.iscene["tart"]
        self.jig: RigidObject = env.iscene["jig"]
        self.dish: RigidObject = env.iscene["dish"]
        self.plate: RigidObject = env.iscene["plate"]
        self.rest: RigidObject = env.iscene["rest"]
        self.env_origins = env.iscene.env_origins
        self._tip_local = torch.tensor(c.tip_local, device=dev)
        self._ez = torch.tensor([0.0, 0.0, 1.0], device=dev)

        self.charge = torch.zeros(n, dtype=torch.long, device=dev)
        self.frosted = torch.zeros(n, c.n_cells, dtype=torch.bool, device=dev)
        self.loaded_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self.dips = torch.zeros(n, dtype=torch.long, device=dev)
        self.dip_dwell = torch.zeros(n, dtype=torch.long, device=dev)
        self.dish_active = torch.zeros(n, dtype=torch.bool, device=dev)
        self.stroke_active = torch.zeros(n, dtype=torch.bool, device=dev)
        self.prev_in_dish = torch.zeros(n, dtype=torch.bool, device=dev)
        self.prev_in_band = torch.zeros(n, dtype=torch.bool, device=dev)
        self.prev_tip = torch.full((n, 3), 1.0e6, device=dev)

        # visual state hooks (best-effort; physics/rubric never depend on them)
        self._cell_attrs, self._blade_attrs = [], []
        self._blade_shown_loaded = [False] * n
        try:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            for e in range(n):
                cells = []
                for k in range(c.n_cells):
                    prim = stage.GetPrimAtPath(f"/World/envs/env_{e}/Tart/cell_{k}")
                    cells.append(UsdGeom.Gprim(prim).GetDisplayColorAttr()
                                 if prim and prim.IsValid() else None)
                self._cell_attrs.append(cells)
                bp = stage.GetPrimAtPath(f"/World/envs/env_{e}/Knife/blade")
                self._blade_attrs.append(UsdGeom.Gprim(bp).GetDisplayColorAttr()
                                         if bp and bp.IsValid() else None)
        except Exception:  # noqa: BLE001
            self._cell_attrs = [[None] * c.n_cells for _ in range(n)]
            self._blade_attrs = [None] * n

    def _set_cell_color(self, e: int, k: int, rgb: tuple) -> None:
        try:
            from pxr import Gf

            a = self._cell_attrs[e][k]
            if a is not None:
                a.Set([Gf.Vec3f(*rgb)])
        except Exception:  # noqa: BLE001
            pass

    def _set_blade_color(self, e: int, loaded: bool) -> None:
        if self._blade_shown_loaded[e] == loaded:
            return
        self._blade_shown_loaded[e] = loaded
        try:
            from pxr import Gf

            a = self._blade_attrs[e]
            if a is not None:
                c = self.cfg.blade_loaded_color if loaded else self.cfg.blade_color
                a.Set([Gf.Vec3f(*c)])
        except Exception:  # noqa: BLE001
            pass

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the jig pose (xy jitter + yaw) and the dish / plate / rest
        bearings, re-pose the kinematic fixtures (POSE writes), seat the tart in the jig
        pocket and lay the knife in its cradle, clear the frosting state machine."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def u(lo: float, hi: float) -> torch.Tensor:
            return lo + (hi - lo) * torch.rand(m, device=dev)

        def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
            q = torch.zeros(m, 4, device=dev)
            q[:, 0] = torch.cos(yaw / 2)
            q[:, 3] = torch.sin(yaw / 2)
            return q

        def pose(xy: torch.Tensor, z: float, q: torch.Tensor) -> torch.Tensor:
            p = torch.zeros(m, 7, device=dev)
            p[:, 0:2] = xy
            p[:, 2] = z
            p[:, 3:7] = q
            p[:, 0:3] += origin
            return p

        def polar(arc: tuple, rr: tuple) -> torch.Tensor:
            ang = torch.deg2rad(u(*arc))
            r = u(*rr)
            return torch.stack([r * torch.cos(ang), r * torch.sin(ang)], dim=1)

        jig_xy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.jig_jitter
        jig_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.jig_yaw_deg)
        jq = yaw_quat(jig_yaw)
        self.jig.write_root_pose_to_sim(pose(jig_xy, 0.0, jq), env_ids)
        self.dish.write_root_pose_to_sim(
            pose(jig_xy + polar(c.dish_arc, c.dish_rr), 0.0, yaw_quat(jig_yaw * 0)), env_ids)
        self.plate.write_root_pose_to_sim(
            pose(jig_xy + polar(c.plate_arc, c.plate_rr), c.plate_h / 2,
                 yaw_quat(jig_yaw * 0)), env_ids)
        rest_xy = jig_xy + polar(c.rest_arc, c.rest_rr)
        rest_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rest_yaw_deg)
        rq = yaw_quat(rest_yaw)
        self.rest.write_root_pose_to_sim(pose(rest_xy, 0.0, rq), env_ids)

        # tart: seated in the jig pocket, matching yaw
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = jig_xy
        st[:, 2] = c.jig_base[2] + c.tart_size[2] / 2 + 0.001
        st[:, 3:7] = jq
        st[:, 0:3] += origin
        self.tart.write_root_state_to_sim(st, env_ids)

        # knife: lying in the cradle channel (blade +x along the channel), root slightly
        # off-center so the -95/+110 mm extents sit inside the 240 mm slab
        st = torch.zeros(m, 13, device=dev)
        off = torch.stack([torch.cos(rest_yaw), torch.sin(rest_yaw)], dim=1) * (-0.008)
        st[:, 0:2] = rest_xy + off
        st[:, 2] = c.rest_slab[2] + c.blade_t / 2 + 0.003
        st[:, 3:7] = rq
        st[:, 0:3] += origin
        self.knife.write_root_state_to_sim(st, env_ids)

        self.charge[env_ids] = 0
        self.frosted[env_ids] = False
        self.loaded_ever[env_ids] = False
        self.dips[env_ids] = 0
        self.dip_dwell[env_ids] = 0
        self.dish_active[env_ids] = False
        self.stroke_active[env_ids] = False
        self.prev_in_dish[env_ids] = False
        self.prev_in_band[env_ids] = False
        self.prev_tip[env_ids] = 1.0e6
        for e in env_ids.tolist():
            for k in range(c.n_cells):
                self._set_cell_color(e, k, c.cell_color)
            self._set_blade_color(e, False)

    # ----- frosting state machine (every physics substep) ------------------------------------
    def _tip_w(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        return self.knife.data.root_pos_w + quat_apply(
            self.knife.data.root_quat_w, self._tip_local.expand(n, 3))

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Advance the dip / stroke machine. Both the dish dwell and the frosting band use
        an ENTRY LATCH: they activate only on a small-displacement crossing INTO the zone
        (per-substep tip travel < `tip_step_max`), so a knife teleported into a working
        pose never accrues — it must leave and come back in a continuous stroke."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        n = self.env.num_envs
        tip = self._tip_w()
        moved_ok = (tip - self.prev_tip).norm(dim=-1) < c.tip_step_max

        # --- dip: tip inside the dish aperture, below the rim ---
        d_loc = tip - self.dish.data.root_pos_w
        in_dish = ((d_loc[:, :2].norm(dim=-1) < c.dish_inner_r - c.dip_margin_r)
                   & (d_loc[:, 2] > c.dish_bot_t)
                   & (d_loc[:, 2] < c.dish_h - c.dip_below_rim))
        self.dish_active = in_dish & moved_ok & (self.dish_active | ~self.prev_in_dish)
        self.dip_dwell = torch.where(self.dish_active, self.dip_dwell + 1,
                                     torch.zeros_like(self.dip_dwell))
        fire = self.dip_dwell == c.dip_dwell_substeps
        if fire.any():
            self.charge = torch.where(fire, torch.full_like(self.charge, c.capacity),
                                      self.charge)
            self.loaded_ever |= fire
            self.dips += fire.long()

        # --- frosting: tip in the working band over the tart, blade flat, tart still ---
        tq = self.tart.data.root_quat_w
        t_loc = quat_apply_inverse(tq, tip - self.tart.data.root_pos_w)
        lx, ly, lz = c.tart_size
        over = (t_loc[:, 0].abs() < lx / 2) & (t_loc[:, 1].abs() < ly / 2)
        dz = t_loc[:, 2] - lz / 2
        in_z = (dz > -c.band_lo) & (dz < c.band_hi)
        kz = quat_apply(self.knife.data.root_quat_w, self._ez.expand(n, 3))
        tz = quat_apply(tq, self._ez.expand(n, 3))
        flat = (kz * tz).sum(dim=-1) > math.cos(math.radians(c.flat_max_deg))
        in_band = over & in_z & flat
        self.stroke_active = in_band & moved_ok & (self.stroke_active | ~self.prev_in_band)
        tart_still = self.tart.data.root_lin_vel_w.norm(dim=-1) < c.paint_tart_vmax
        can = self.stroke_active & tart_still & (self.charge > 0)
        if can.any():
            nx, ny = c.grid
            ci = ((t_loc[:, 0] + lx / 2) / (lx / nx)).long().clamp(0, nx - 1)
            cj = ((t_loc[:, 1] + ly / 2) / (ly / ny)).long().clamp(0, ny - 1)
            cell = cj * nx + ci
            fresh = can & ~self.frosted.gather(1, cell.unsqueeze(1)).squeeze(1)
            if fresh.any():
                idx = fresh.nonzero(as_tuple=False).squeeze(-1)
                self.frosted[idx, cell[idx]] = True
                self.charge[idx] -= 1
                for e in idx.tolist():
                    self._set_cell_color(e, int(cell[e]), c.frosted_color)

        for e in range(n):
            self._set_blade_color(e, bool(self.charge[e] > 0))
        self.prev_in_dish = in_dish
        self.prev_in_band = in_band
        self.prev_tip = tip.clone()

    # ----- state (full, restorable) ----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "knife": self.knife.data.root_state_w[env_ids].clone(),
            "tart": self.tart.data.root_state_w[env_ids].clone(),
            "jig": self.jig.data.root_state_w[env_ids, 0:7].clone(),
            "dish": self.dish.data.root_state_w[env_ids, 0:7].clone(),
            "plate": self.plate.data.root_state_w[env_ids, 0:7].clone(),
            "rest": self.rest.data.root_state_w[env_ids, 0:7].clone(),
            "charge": self.charge[env_ids].clone(),
            "frosted": self.frosted[env_ids].clone(),
            "loaded_ever": self.loaded_ever[env_ids].clone(),
            "dips": self.dips[env_ids].clone(),
            "dip_dwell": self.dip_dwell[env_ids].clone(),
            "dish_active": self.dish_active[env_ids].clone(),
            "stroke_active": self.stroke_active[env_ids].clone(),
            "prev_in_dish": self.prev_in_dish[env_ids].clone(),
            "prev_in_band": self.prev_in_band[env_ids].clone(),
            "prev_tip": self.prev_tip[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.knife.write_root_state_to_sim(state["knife"], env_ids)
        self.tart.write_root_state_to_sim(state["tart"], env_ids)
        for name in ("jig", "dish", "plate", "rest"):
            getattr(self, name).write_root_pose_to_sim(state[name], env_ids)
        for name in ("charge", "frosted", "loaded_ever", "dips", "dip_dwell", "dish_active",
                     "stroke_active", "prev_in_dish", "prev_in_band", "prev_tip"):
            getattr(self, name)[env_ids] = state[name]

    # ----- description -----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A rectangular tart ({c.tart_size[0] * 1000:.0f} x {c.tart_size[1] * 1000:.0f} mm) "
            f"sits in a shallow prep jig on the ground, its dark top divided into "
            f"{c.grid[0]}x{c.grid[1]} cells. Around it stand a teal butter dish (open cup "
            f"with yellow butter inside), a white serving plate, and a gray cradle holding "
            f"a butter knife — their bearings change every episode.\n"
            f"Goal: frost EVERY cell of the tart top, then serve and tidy up. The knife "
            f"only spreads while its blade tip sweeps a slow, continuous, flat stroke "
            f"within a few millimetres of the tart's top face — and only while it carries "
            f"butter: one dip of the blade into the butter dish loads enough for "
            f"{c.capacity} cells (the blade turns yellow while charged), so you must "
            f"return to the dish repeatedly. When all {c.n_cells} cells are frosted "
            f"(they turn cream), lift the tart out of the jig onto the plate, and lay the "
            f"knife back in its cradle. Serving an unfrosted tart counts for nothing; "
            f"hovering high over the tart, stabbing at it, or waving an unloaded knife "
            f"spreads nothing."
        )

    # ----- predicates / rubric ----------------------------------------------------------------
    def _upright(self, quat: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        n = quat.shape[0]
        up = quat_apply(quat, self._ez.expand(n, 3))
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.upright_tol_deg))

    def all_frosted(self) -> torch.Tensor:
        return self.frosted.all(dim=1)

    def tart_on_plate(self) -> torch.Tensor:
        """(N,) bool: tart resting settled + upright on the plate (physical outcome)."""
        c = self.cfg
        tp = self.tart.data.root_pos_w
        pp = self.plate.data.root_pos_w
        near = (tp[:, :2] - pp[:, :2]).norm(dim=-1) < c.serve_xy_tol
        bottom = tp[:, 2] - c.tart_size[2] / 2
        plate_top = pp[:, 2] + c.plate_h / 2
        on_top = (bottom - plate_top).abs() < c.serve_z_tol
        still = self.tart.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        return near & on_top & self._upright(self.tart.data.root_quat_w) & still

    def knife_parked(self) -> torch.Tensor:
        """(N,) bool: knife lying settled, roughly flat, inside the cradle channel
        (judged in the REST'S frame so the sampled cradle yaw is honored)."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        n = self.env.num_envs
        k_loc = quat_apply_inverse(self.rest.data.root_quat_w,
                                   self.knife.data.root_pos_w - self.rest.data.root_pos_w)
        in_ch = ((k_loc[:, 0].abs() < c.park_x_tol) & (k_loc[:, 1].abs() < c.park_y_tol)
                 & (k_loc[:, 2] > 0.0) & (k_loc[:, 2] < c.park_z_max))
        kz = quat_apply(self.knife.data.root_quat_w, self._ez.expand(n, 3))
        flat = kz[:, 2].abs() >= math.cos(math.radians(c.park_flat_deg))
        still = self.knife.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        return in_ch & flat & still

    def success(self) -> torch.Tensor:
        """(N,) bool: every cell frosted, tart settled on the plate, knife back in its
        cradle — the full station reset to 'served'."""
        return self.all_frosted() & self.tart_on_plate() & self.knife_parked()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]. Doing nothing scores 0 (the knife STARTS parked, but the
        park credit is gated on served+frosted; the tart starts in the jig). Serving an
        unfrosted tart scores 0 for the serve — the seed's own transport plan. 1.0 iff
        success()."""
        c = self.cfg
        full = self.all_frosted()
        served = full & self.tart_on_plate()
        s = (c.w_load * self.loaded_ever.float()
             + c.w_cov * self.frosted.float().mean(dim=1)
             + c.w_full * full.float()
             + c.w_serve * served.float()
             + c.w_park * (served & self.knife_parked()).float())
        return torch.where(self.success(), torch.ones_like(s), s.clamp(max=0.97))


# ----- env registration ------------------------------------------------------------------------
register_env("simgen", lambda: EnvCfg(scene="butter_station", robot="null", env_spacing=4.0))
