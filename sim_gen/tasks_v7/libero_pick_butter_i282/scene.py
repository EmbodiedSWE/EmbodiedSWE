"""ButterDispenserScene — stage the basket under the outlet, then dispense ONE butter.

Derived from libero/libero_pick_butter but strategically different. The seed is one
prehensile transport INTO a container: the butter stands free on the table among
passive distractors, and the plan is grasp -> carry -> release over the basket.

Here the butter is never grasped and never carried. All butter blocks live stacked in
a gravity-fed MAGAZINE TOWER: an elevated chamber whose front outlet slot admits only
the bottom block, fed by a prismatic PUSHER BLADE that enters through a back slot.
One full blade stroke shoves the bottom block over the chamber lip; it free-falls off
the elevated floor. The plan is inverted end to end:

  1. STAGE   — carry the BASKET (the container moves, not the butter) onto the green
               catch mat under the outlet lip and set it down upright;
  2. DISPENSE— push the blade through one full stroke: the bottom butter is extruded
               through the outlet slot under real contact (stack riding on it, the
               retained block dragged against the front wall) and falls into the
               staged basket;
  3. STOP    — exactly one block: the block above is retained by the front wall
               during the stroke (physics, tested in smoke), and success requires
               every other present block still inside the tower.

Required order (physically enforced, declared in TASK.md): stage BEFORE dispensing.
A block dispensed with no basket lands on the ground; blocks are 88 mm in every
horizontal footprint (> the 80 mm Franka jaw) and can never be put back into the
elevated magazine, so an early dispense permanently forfeits that block.

Rubric (graded [0, 1]; latched stages + a current-state terminal conjunction):
  0.00   null policy (basket parked away, blade home, stack full)
  0.20   latched: basket ever settled upright on the catch mat (staged)
  +0.15  latched: blade ever driven past 60% of its stroke (actuated)
  +0.35  latched: a block's outlet-transit credential fired WHILE the basket was
         staged (delivered) — the credential is the block's CoM crossing the outlet
         slot volume with outward velocity, which a drop from above can never
         produce (the magazine front wall fills that xy above the slot)
  1.00   iff success(): exactly ONE present block settled inside the upright staged
         basket, that block carries the outlet credential, every other present block
         still inside the tower, everything still.

Honesty by construction (asserted in __post_init__):
  - the outlet slot admits only the bottom block (slot_h < 2*block_h by a margin);
  - the back slot is shorter than a block (blocks cannot leave backward) and taller
    than the blade;
  - the basket fits under the chamber overhang and can never collide with the
    pedestal column anywhere inside the staging tolerance;
  - the ejected block's landing window lies inside the basket interior across the
    whole staging tolerance;
  - the basket spawn zone is far outside the staging gate.

Assets are fully procedural: a kinematic tower (pedestal column + elevated chamber
floor with lip + magazine walls with front outlet slot and back blade slot), a
dynamic pusher blade (blade + rod + push paddle, one compound body) on a bind-time
prismatic joint, 2-3 dynamic butter blocks (pale-yellow boxes with a wax-paper wrap
band), a dynamic open basket, and a kinematic green catch mat. Per-episode
randomization: block count (2 or 3), in-magazine jitter + yaw, basket spawn side,
xy jitter + free yaw.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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


def _apply_root(prim_path: str, translation, orientation, *, kinematic: bool, mass: float,
                max_depen: float = 0.5, live: bool = False):
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
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(max_depen)
    if live:
        # external-wrench drives do not wake a sleeping body
        px.CreateSleepThresholdAttr(0.0)
        px.CreateStabilizationThresholdAttr(0.0)
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


def _spawn_tower(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC dispenser tower, root at GROUND level on the chamber center axis.
    Outlet faces local -x. Parts: pedestal column, elevated chamber floor (its front
    edge is the drop lip), two full-height side walls, a front wall that stops
    slot_h above the floor (the outlet slot), a back wall that stops back_slot_h
    above the floor (the blade slot)."""
    stage, root = _apply_root(prim_path, translation, orientation, kinematic=True, mass=5.0)
    co = cfg.contact_offset
    ix, iy = cfg.inner_x_half, cfg.inner_y_half
    wt = cfg.wall_t
    zf, zt = cfg.floor_z, cfg.top_z
    # pedestal column (front face recessed so the basket wall can sit under the overhang)
    _box_part(stage, prim_path, "column",
              (cfg.col_back - cfg.col_front, 2 * (iy + wt), zf - cfg.floor_t),
              ((cfg.col_front + cfg.col_back) / 2, 0.0, (zf - cfg.floor_t) / 2),
              cfg.column_color, co)
    # chamber floor: spans the full outer footprint; front edge = the lip
    _box_part(stage, prim_path, "floor",
              (2 * (ix + wt), 2 * (iy + wt), cfg.floor_t),
              (0.0, 0.0, zf - cfg.floor_t / 2), cfg.tower_color, co)
    # side walls, full height
    for s, tag in ((-1.0, "l"), (1.0, "r")):
        _box_part(stage, prim_path, f"side_{tag}",
                  (2 * (ix + wt), wt, zt - zf),
                  (0.0, s * (iy + wt / 2), (zf + zt) / 2), cfg.tower_color, co)
    # front wall: from the top of the outlet slot up (only the bottom block passes)
    _box_part(stage, prim_path, "front",
              (wt, 2 * iy, zt - (zf + cfg.slot_h)),
              (-(ix + wt / 2), 0.0, (zf + cfg.slot_h + zt) / 2), cfg.tower_color, co)
    # back wall: from the top of the blade slot up
    _box_part(stage, prim_path, "back",
              (wt, 2 * iy, zt - (zf + cfg.back_slot_h)),
              (ix + wt / 2, 0.0, (zf + cfg.back_slot_h + zt) / 2), cfg.tower_color, co)
    return root


def _spawn_blade(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Dynamic pusher blade, root at the BLADE BOX CENTER: blade box + rod back
    through the blade slot + a tall push paddle for the arm. One compound body on a
    bind-time prismatic joint (x axis)."""
    stage, root = _apply_root(prim_path, translation, orientation, kinematic=False,
                              mass=cfg.mass, live=True)
    from pxr import PhysxSchema

    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(cfg.damping)
    co = cfg.contact_offset
    _box_part(stage, prim_path, "blade", (cfg.blade_len, cfg.blade_w, cfg.blade_h),
              (0.0, 0.0, 0.0), cfg.color, co)
    _box_part(stage, prim_path, "rod", (cfg.rod_len, cfg.rod_w, cfg.rod_h),
              (cfg.blade_len / 2 + cfg.rod_len / 2, 0.0, -cfg.blade_h / 2 + cfg.rod_h / 2 + 0.004),
              cfg.color, co)
    _box_part(stage, prim_path, "paddle", (cfg.paddle_t, cfg.paddle_w, cfg.paddle_h),
              (cfg.blade_len / 2 + cfg.rod_len + cfg.paddle_t / 2, 0.0,
               cfg.paddle_zc), cfg.paddle_color, co)
    return root


def _spawn_basket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Dynamic open basket, root at the BASE BOTTOM CENTER: floor + 4 walls."""
    stage, root = _apply_root(prim_path, translation, orientation, kinematic=False,
                              mass=cfg.mass, live=True)
    from pxr import PhysxSchema

    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.10)
    px.CreateAngularDampingAttr(0.10)
    ih, wt, wh = cfg.inner_half, cfg.wall_t, cfg.wall_h
    oh = ih + wt
    co, color = cfg.contact_offset, cfg.color
    _box_part(stage, prim_path, "floor", (2 * oh, 2 * oh, cfg.bot_t),
              (0.0, 0.0, cfg.bot_t / 2), color, co)
    for s, tag in ((-1.0, "yn"), (1.0, "yp")):
        _box_part(stage, prim_path, f"wall_{tag}", (2 * oh, wt, wh),
                  (0.0, s * (ih + wt / 2), cfg.bot_t + wh / 2), color, co)
    for s, tag in ((-1.0, "xn"), (1.0, "xp")):
        _box_part(stage, prim_path, f"wall_{tag}", (wt, 2 * ih, wh),
                  (s * (ih + wt / 2), 0.0, cfg.bot_t + wh / 2), color, co)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Explicit @configclass spawner cfgs (defined once, lazily — heavy imports)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tower" not in _SPAWNER_CACHE:

        @configclass
        class TowerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tower)
            inner_x_half: float = 0.052
            inner_y_half: float = 0.050
            wall_t: float = 0.010
            floor_z: float = 0.220
            floor_t: float = 0.012
            top_z: float = 0.430
            slot_h: float = 0.058
            back_slot_h: float = 0.048
            col_front: float = 0.010
            col_back: float = 0.065
            contact_offset: float = 0.002
            tower_color: tuple = (0.45, 0.32, 0.18)
            column_color: tuple = (0.35, 0.36, 0.40)

        @configclass
        class BladeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_blade)
            mass: float = 0.60
            damping: float = 3.0
            blade_len: float = 0.030
            blade_w: float = 0.090
            blade_h: float = 0.040
            rod_len: float = 0.100
            rod_w: float = 0.020
            rod_h: float = 0.016
            paddle_t: float = 0.012
            paddle_w: float = 0.080
            paddle_h: float = 0.110
            paddle_zc: float = 0.050
            contact_offset: float = 0.002
            color: tuple = (0.30, 0.30, 0.34)
            paddle_color: tuple = (0.85, 0.15, 0.15)

        @configclass
        class BasketSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_basket)
            mass: float = 0.50
            inner_half: float = 0.100
            wall_t: float = 0.009
            wall_h: float = 0.100
            bot_t: float = 0.008
            contact_offset: float = 0.002
            color: tuple = (0.62, 0.45, 0.22)

        _SPAWNER_CACHE.update(tower=TowerSpawnerCfg, blade=BladeSpawnerCfg,
                              basket=BasketSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class ButterDispenserSceneCfg(BaseCfg):
    """Config for `ButterDispenserScene`. Interlock margins asserted in __post_init__."""

    # --- tunable: rubric thresholds -----------------------------------------------------------
    catch_tol: float = tunable(0.030)  # basket center within this (xy) of the mat center
    basket_tilt_max_deg: float = tunable(15.0)  # "upright" gate on the staged basket
    basket_z_tol: float = tunable(0.012)  # basket bottom within this of the mat top
    in_basket_margin: float = tunable(0.010)  # xy margin inside the basket interior
    in_basket_zmax: float = tunable(0.085)  # block CoM below this (basket frame) = contained
    actuated_frac: float = tunable(0.60)  # blade past this fraction of the stroke = actuated
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)

    # --- tunable: randomization (task-family knobs) --------------------------------------------
    min_blocks: int = tunable(2)  # per-episode block count sampled in {min..max}
    max_blocks: int = tunable(3)
    block_jitter: float = tunable(0.0015)  # in-magazine xy jitter per block
    block_yaw_deg: float = tunable(2.0)  # in-magazine yaw jitter per block
    basket_spawn: tuple = tunable((0.02, 0.30))  # |y| side sampled 50/50 per episode
    basket_jitter: float = tunable(0.040)  # uniform +/- xy jitter of the basket spawn
    basket_yaw_deg: float = tunable(180.0)  # uniform +/- basket yaw

    # --- tunable: plant -------------------------------------------------------------------------
    blade_damping: float = tunable(3.0)  # blade body linear damping

    # --- info: structure ------------------------------------------------------------------------
    tower_pos: tuple = info((0.30, 0.0))  # tower axis (xy) — FIXED (prismatic anchor)
    inner_x_half: float = info(0.052)  # chamber interior half-depth (x)
    inner_y_half: float = info(0.050)  # chamber interior half-width (y)
    wall_t: float = info(0.010)
    floor_z: float = info(0.220)  # chamber floor TOP (the elevated deck)
    floor_t: float = info(0.012)
    top_z: float = info(0.430)  # magazine wall top (open)
    slot_h: float = info(0.058)  # front outlet slot height above the floor
    back_slot_h: float = info(0.048)  # back blade slot height above the floor
    col_front: float = info(0.010)  # pedestal column front face (tower local x)
    col_back: float = info(0.065)
    block_s: float = info(0.088)  # butter footprint (square)
    block_h: float = info(0.050)
    block_mass: float = info(0.15)
    blade_len: float = info(0.030)
    blade_w: float = info(0.090)
    blade_h: float = info(0.040)
    blade_mass: float = info(0.60)
    rod_len: float = info(0.100)
    paddle_t: float = info(0.012)
    paddle_h: float = info(0.110)
    stroke: float = info(0.100)  # blade travel, home -> fully extended (toward -x)
    basket_inner_half: float = info(0.100)
    basket_wall_t: float = info(0.009)
    basket_wall_h: float = info(0.100)
    basket_bot_t: float = info(0.008)
    basket_mass: float = info(0.50)
    mat_size: float = info(0.26)
    mat_t: float = info(0.004)
    catch_back: float = info(0.085)  # mat center this far outboard (-x) of the lip
    parking_pos: tuple = info((1.25, 1.25))  # ground depot for absent blocks
    contact_offset: float = info(0.002)
    tower_color: tuple = info((0.45, 0.32, 0.18))
    column_color: tuple = info((0.35, 0.36, 0.40))
    blade_color: tuple = info((0.30, 0.30, 0.34))
    paddle_color: tuple = info((0.85, 0.15, 0.15))
    butter_color: tuple = info((0.94, 0.88, 0.55))
    wrap_color: tuple = info((0.92, 0.96, 0.98))
    basket_color: tuple = info((0.62, 0.45, 0.22))
    mat_color: tuple = info((0.10, 0.60, 0.20))

    # Derived (filled in __post_init__).
    lip_x: float = field(default=None, init=False)  # tower-local x of the drop lip
    blade_home_x: float = field(default=None, init=False)  # tower-local blade root x, home
    blade_zc: float = field(default=None, init=False)  # blade root z
    catch_center: tuple = field(default=None, init=False)  # world xy of the mat center

    def __post_init__(self) -> None:
        ix, wt = self.inner_x_half, self.wall_t
        self.lip_x = -(ix + wt)  # -0.062
        self.blade_home_x = ix + self.blade_len / 2  # blade front face flush at +ix
        self.blade_zc = self.floor_z + 0.001 + self.blade_h / 2
        tx, ty = self.tower_pos
        self.catch_center = (tx + self.lip_x - self.catch_back, ty)

        bh, bs = self.block_h, self.block_s
        # outlet slot admits ONLY the bottom block
        assert self.slot_h > bh + 0.006, "outlet slot must pass the bottom block"
        assert 2 * bh > self.slot_h + 0.030, \
            "the riding block must be retained by the front wall (exactly-one physics)"
        # blocks cannot leave backward; the blade fits its slot
        assert self.back_slot_h < bh, "back slot must be shorter than a block"
        assert self.blade_h + 0.006 < self.back_slot_h, "blade must fit the back slot"
        # magazine clearances: worst-case yawed block never wedges (>= 2.5 mm per side)
        half_diag = bs / 2 * (math.cos(math.radians(self.block_yaw_deg))
                              + math.sin(math.radians(self.block_yaw_deg)))
        assert half_diag + self.block_jitter + 0.0025 < ix, "chamber depth clearance"
        assert half_diag + self.block_jitter + 0.0025 < self.inner_y_half, \
            "chamber width clearance"
        # blade home leaves a gap behind the resting bottom block
        assert half_diag + self.block_jitter + 0.003 < ix, "blade-home gap"
        # basket fits under the chamber overhang and never hits the pedestal column
        assert self.basket_wall_h + self.basket_bot_t + 0.02 < self.floor_z - self.floor_t, \
            "basket wall must clear the chamber floor overhang"
        worst_outer = (self.catch_center[0] - self.tower_pos[0]) + self.catch_tol \
            + self.basket_inner_half + self.basket_wall_t
        assert worst_outer + 0.005 < self.col_front, \
            "staged basket must never touch the pedestal column"
        # ejected-block landing window [lip-0.12, lip-0.03] inside the basket interior
        cx_rel = self.catch_center[0] - self.tower_pos[0]  # catch center, tower local
        assert cx_rel - self.catch_tol + self.basket_inner_half >= self.lip_x - 0.030, \
            "near landing edge covered at worst staging tolerance"
        assert cx_rel + self.catch_tol - self.basket_inner_half <= self.lip_x - 0.120, \
            "far landing edge covered at worst staging tolerance"
        # basket spawn is far outside the staging gate
        d = math.hypot(self.basket_spawn[0] - self.catch_center[0],
                       abs(self.basket_spawn[1]) - self.catch_center[1])
        assert d - self.basket_jitter > self.catch_tol + 0.10, "spawn far from the mat"
        # embodiment: butter blocks are NOT graspable (every horizontal footprint pair
        # exceeds the 80 mm jaw in the magazine); the basket rim IS graspable
        assert bs > 0.082, "blocks must exceed the jaw span"
        assert self.basket_wall_t < 0.078, "basket rim must fit the jaw"


# ----- scene ---------------------------------------------------------------------------------------
@SCENES.register("butter_dispenser")
class ButterDispenserScene(BaseScene):
    cfg: ButterDispenserSceneCfg

    N_BLOCKS = 3  # bodies spawned; per-episode presence sampled in {min..max}

    def __init__(self, cfg: ButterDispenserSceneCfg | None = None) -> None:
        super().__init__(cfg or ButterDispenserSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        spawners = _spawner_classes()
        tower_cls, blade_cls, basket_cls = (spawners["tower"], spawners["blade"],
                                            spawners["basket"])
        tx, ty = c.tower_pos

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
            "tower": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tower",
                spawn=tower_cls(
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    inner_x_half=c.inner_x_half, inner_y_half=c.inner_y_half,
                    wall_t=c.wall_t, floor_z=c.floor_z, floor_t=c.floor_t,
                    top_z=c.top_z, slot_h=c.slot_h, back_slot_h=c.back_slot_h,
                    col_front=c.col_front, col_back=c.col_back,
                    contact_offset=c.contact_offset, tower_color=c.tower_color,
                    column_color=c.column_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(tx, ty, 0.0)),
            ),
            "blade": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Blade",
                spawn=blade_cls(
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.blade_mass, damping=c.blade_damping,
                    blade_len=c.blade_len, blade_w=c.blade_w, blade_h=c.blade_h,
                    rod_len=c.rod_len, paddle_t=c.paddle_t, paddle_h=c.paddle_h,
                    contact_offset=c.contact_offset, color=c.blade_color,
                    paddle_color=c.paddle_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(tx + c.blade_home_x, ty, c.blade_zc)),
            ),
            "basket": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Basket",
                spawn=basket_cls(
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.basket_mass, inner_half=c.basket_inner_half,
                    wall_t=c.basket_wall_t, wall_h=c.basket_wall_h,
                    bot_t=c.basket_bot_t, contact_offset=c.contact_offset,
                    color=c.basket_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.basket_spawn[0], c.basket_spawn[1], 0.002)),
            ),
            "mat": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Mat",
                spawn=sim_utils.CuboidCfg(
                    size=(c.mat_size, c.mat_size, c.mat_t),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.mat_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.catch_center[0], c.catch_center[1], c.mat_t / 2)),
            ),
        }
        for i in range(self.N_BLOCKS):
            out[f"butter_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Butter_" + str(i),
                spawn=sim_utils.CuboidCfg(
                    size=(c.block_s, c.block_s, c.block_h),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.08,
                        sleep_threshold=0.0, stabilization_threshold=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.block_mass),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.35, dynamic_friction=0.30, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.butter_color if i != 1 else
                        tuple(v * 0.98 for v in c.butter_color)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(tx, ty, c.floor_z + c.block_h / 2 + 0.001 + i * (c.block_h + 0.0008))),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.tower: RigidObject = env.iscene["tower"]
        self.blade: RigidObject = env.iscene["blade"]
        self.basket: RigidObject = env.iscene["basket"]
        self.mat: RigidObject = env.iscene["mat"]
        self.blocks: list[RigidObject] = [env.iscene[f"butter_{i}"]
                                          for i in range(self.N_BLOCKS)]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        # Episode state.
        self.present = torch.ones(n, self.N_BLOCKS, dtype=torch.bool, device=dev)
        self._staged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._actuated = torch.zeros(n, dtype=torch.bool, device=dev)
        self._dispensed = torch.zeros(n, self.N_BLOCKS, dtype=torch.bool, device=dev)
        self._delivered = torch.zeros(n, dtype=torch.bool, device=dev)
        # External drive inputs (solve/smoke write; post_step consumes + owns the
        # blade's and blocks' external-wrench slots).
        self.blade_force = torch.zeros(n, device=dev)  # x-force on the blade (N)
        self.block_force = torch.zeros(n, self.N_BLOCKS, 3, device=dev)

    def _author_joints(self) -> None:
        """Per env: an X-axis prismatic joint tower -> blade with the stroke as its
        limits. Body0 is the kinematic tower, which NEVER moves after spawn (the
        anchor is world-fixed). The joint pair never collides."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/pusher_rail")
            j.CreateBody0Rel().SetTargets([f"{base}/Tower"])
            j.CreateBody1Rel().SetTargets([f"{base}/Blade"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(c.blade_home_x, 0.0, c.blade_zc))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-float(c.stroke + 0.001))
            j.CreateUpperLimitAttr(0.002)

    # ----- reset ----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: block count sampled in {min..max} (torch.rand comparison —
        first-randint degeneracy), present blocks stacked into the magazine with xy +
        yaw jitter, absent blocks parked in the ground depot, blade re-posed to home
        (follower-only write along the unchanged rail), basket dealt to a random side
        with jitter + free yaw, latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        tx, ty = c.tower_pos

        def write(body, px, py, pz, yaw) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1], st[:, 2] = px, py, pz
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- block count: min + sum of uniform comparisons (avoids torch.randint).
        # Burn a few draws first: the FIRST post-seed draw is biased on this stack. ---
        torch.rand(8, device=dev)
        n_extra = c.max_blocks - c.min_blocks
        count = torch.full((m,), c.min_blocks, dtype=torch.long, device=dev)
        for _ in range(n_extra):
            count = count + (torch.rand(m, device=dev) < 0.5).long()
        idx = torch.arange(self.N_BLOCKS, device=dev)
        pres = idx.unsqueeze(0) < count.unsqueeze(1)  # bottom-up fill
        self.present[env_ids] = pres

        # --- blocks: stacked in the magazine (present) or parked in the depot ---
        yaw_amp = math.radians(c.block_yaw_deg)
        for i, body in enumerate(self.blocks):
            jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.block_jitter
            in_x = torch.full((m,), tx, device=dev) + jit[:, 0]
            in_y = torch.full((m,), ty, device=dev) + jit[:, 1]
            in_z = torch.full((m,), c.floor_z + c.block_h / 2 + 0.001
                              + i * (c.block_h + 0.0008), device=dev)
            park_x = torch.full((m,), c.parking_pos[0] + 0.12 * i, device=dev)
            park_y = torch.full((m,), c.parking_pos[1], device=dev)
            park_z = torch.full((m,), c.block_h / 2 + 0.002, device=dev)
            p = pres[:, i]
            write(body, torch.where(p, in_x, park_x), torch.where(p, in_y, park_y),
                  torch.where(p, in_z, park_z),
                  (torch.rand(m, device=dev) * 2 - 1) * yaw_amp)

        # --- blade: home pose (follower-only write along the fixed rail) ---
        write(self.blade, torch.full((m,), tx + c.blade_home_x, device=dev),
              torch.full((m,), ty, device=dev),
              torch.full((m,), c.blade_zc, device=dev), torch.zeros(m, device=dev))

        # --- basket: random side, jitter, free yaw ---
        side = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        bj = (torch.rand(m, 2, device=dev) * 2 - 1) * c.basket_jitter
        write(self.basket,
              torch.full((m,), c.basket_spawn[0], device=dev) + bj[:, 0],
              side * c.basket_spawn[1] + bj[:, 1],
              torch.full((m,), 0.002, device=dev),
              (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.basket_yaw_deg))

        # --- latches + drive buffers ---
        self._staged[env_ids] = False
        self._actuated[env_ids] = False
        self._dispensed[env_ids] = False
        self._delivered[env_ids] = False
        self.blade_force[env_ids] = 0.0
        self.block_force[env_ids] = 0.0

    # ----- step-coupled mechanics (every substep) ---------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Apply the blade drive + block probe forces (owned wrench slots), then latch
        the rubric stages (NaN-guarded — a diverged frame earns no progress)."""
        n = self.env.num_envs
        dev = self.env.device

        f = torch.zeros(n, 1, 3, device=dev)
        f[:, 0, 0] = torch.nan_to_num(self.blade_force, nan=0.0, posinf=0.0, neginf=0.0)
        self.blade.set_external_force_and_torque(f, torch.zeros(n, 1, 3, device=dev))
        for i, body in enumerate(self.blocks):
            body.set_external_force_and_torque(
                torch.nan_to_num(self.block_force[:, i]).reshape(n, 1, 3),
                torch.zeros(n, 1, 3, device=dev))

        good = torch.isfinite(self.basket.data.root_pos_w).all(dim=-1)
        staged_now = self.staged_now() & good
        self._staged |= staged_now
        self._actuated |= self.actuated_now() & good
        disp = self.dispensing_now() & good.unsqueeze(-1)
        self._dispensed |= disp
        # delivered: an outlet transit fired while the basket sat staged (current)
        self._delivered |= disp.any(dim=-1) & staged_now

    # ----- state (full, restorable) -----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"tower": self.tower, "blade": self.blade, "basket": self.basket,
                  "mat": self.mat,
                  **{f"butter_{i}": b for i, b in enumerate(self.blocks)}}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("present", "_staged", "_actuated", "_dispensed",
                               "_delivered", "blade_force", "block_force")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        names = {"tower": self.tower, "blade": self.blade, "basket": self.basket,
                 "mat": self.mat,
                 **{f"butter_{i}": b for i, b in enumerate(self.blocks)}}
        for nm, st in state["bodies"].items():
            names[nm].write_root_state_to_sim(st, env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ------------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A butter dispenser tower stands on the ground: a gray pedestal column "
            f"carrying an elevated brown chamber (floor {c.floor_z * 100:.0f} cm up) whose "
            f"open-topped magazine holds a stack of pale-yellow butter blocks "
            f"({c.block_s * 1000:.0f} mm square, {c.block_h * 1000:.0f} mm tall — wider than "
            f"a parallel gripper can span; two or three are present, count what you see). "
            f"The chamber's FRONT face has a low outlet slot: one full push of the dispenser "
            f"ejects exactly the BOTTOM block through the slot and over the front lip, and it "
            f"falls off the elevated floor. The dispenser is driven by the RED PADDLE at the "
            f"BACK of the tower: push the paddle horizontally toward the tower to drive the "
            f"internal blade through its {c.stroke * 1000:.0f} mm stroke. On the ground in "
            f"front of the outlet lies a flat GREEN CATCH MAT, and off to one side sits an "
            f"open brown BASKET (walls {c.basket_wall_h * 1000:.0f} mm tall, thin graspable "
            f"rim).\n"
            f"Goal: place the basket upright on the green mat (centered within "
            f"{c.catch_tol * 1000:.0f} mm, resting on the mat — not held in the air), then "
            f"push the paddle through ONE full stroke so exactly one butter block is "
            f"dispensed through the outlet and lands inside the basket. Every other butter "
            f"block must remain inside the tower. Required order: stage the basket FIRST — "
            f"a block dispensed onto the bare ground cannot be recovered (the blocks are too "
            f"wide to grasp and the magazine is elevated), and a block placed into the "
            f"basket by any route other than falling through the outlet does not count. "
            f"Do not dispense a second block."
        )

    def instruction(self) -> str:
        return (
            "Place the basket upright on the green catch mat under the dispenser outlet, "
            "then push the red paddle through one full stroke so exactly one butter block "
            "is dispensed into the basket. Dispensing before the basket is staged, putting "
            "butter in the basket by hand, or dispensing more than one block fails the task."
        )

    # ----- predicates / rubric ----------------------------------------------------------------------
    def _catch_xy(self) -> torch.Tensor:
        c = self.cfg
        return self.env_origins[:, :2] + torch.tensor(
            [c.catch_center[0], c.catch_center[1]], device=self.env.device)

    def _tower_xy(self) -> torch.Tensor:
        c = self.cfg
        return self.env_origins[:, :2] + torch.tensor(
            [c.tower_pos[0], c.tower_pos[1]], device=self.env.device)

    def _up_z(self, body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)

    def blade_disp(self) -> torch.Tensor:
        """(N,) blade travel from home along -x (m, >= 0 when pushed in)."""
        c = self.cfg
        home = self._tower_xy()[:, 0] + c.blade_home_x
        return home - self.blade.data.root_pos_w[:, 0]

    def staged_now(self) -> torch.Tensor:
        """(N,) bool: basket upright ON the mat inside the catch tolerance, settled."""
        c = self.cfg
        bp = self.basket.data.root_pos_w
        near = (bp[:, :2] - self._catch_xy()).norm(dim=-1) <= c.catch_tol
        upright = self._up_z(self.basket) >= math.cos(math.radians(c.basket_tilt_max_deg))
        on_mat = (bp[:, 2] - self.env_origins[:, 2] - c.mat_t).abs() <= c.basket_z_tol
        still = self.basket.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return near & upright & on_mat & still

    def actuated_now(self) -> torch.Tensor:
        """(N,) bool: blade past `actuated_frac` of its stroke."""
        return self.blade_disp() > self.cfg.actuated_frac * self.cfg.stroke

    def dispensing_now(self) -> torch.Tensor:
        """(N, B) bool: block CoM inside the outlet slot volume MOVING OUTWARD — the
        transit credential. The magazine front wall fills this xy above the slot, so
        a block can only be here by coming through the chamber; the outward-velocity
        clause rejects anything lowered from outside."""
        c = self.cfg
        twr = self._tower_xy()
        out = []
        for body in self.blocks:
            p = body.data.root_pos_w
            dx = p[:, 0] - twr[:, 0]
            in_x = (dx > c.lip_x - 0.012) & (dx < c.lip_x + 0.012)
            in_y = (p[:, 1] - twr[:, 1]).abs() < c.inner_y_half + 0.008
            in_z = (p[:, 2] > c.floor_z - 0.008) & (p[:, 2] < c.floor_z + c.slot_h + 0.012)
            moving_out = body.data.root_lin_vel_w[:, 0] < -0.03
            out.append(in_x & in_y & in_z & moving_out)
        return torch.stack(out, dim=1)

    def in_basket(self) -> torch.Tensor:
        """(N, B) bool: block CoM inside the basket interior (basket frame), below the
        rim by a margin — geometric containment."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        bq = self.basket.data.root_quat_w
        bp = self.basket.data.root_pos_w
        lim = c.basket_inner_half - c.in_basket_margin
        out = []
        for body in self.blocks:
            loc = quat_apply_inverse(bq, body.data.root_pos_w - bp)
            out.append((loc[:, 0].abs() < lim) & (loc[:, 1].abs() < lim)
                       & (loc[:, 2] > c.basket_bot_t - 0.005)
                       & (loc[:, 2] < c.in_basket_zmax))
        return torch.stack(out, dim=1)

    def in_tower(self) -> torch.Tensor:
        """(N, B) bool: block CoM inside the tower interior (magazine or chamber)."""
        c = self.cfg
        twr = self._tower_xy()
        out = []
        for body in self.blocks:
            p = body.data.root_pos_w
            out.append(((p[:, 0] - twr[:, 0]).abs() < c.inner_x_half + 0.012)
                       & ((p[:, 1] - twr[:, 1]).abs() < c.inner_y_half + 0.012)
                       & (p[:, 2] > c.floor_z - 0.010) & (p[:, 2] < c.top_z + 0.05))
        return torch.stack(out, dim=1)

    def blocks_settled(self) -> torch.Tensor:
        """(N, B) bool."""
        return torch.stack([b.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed
                            for b in self.blocks], dim=1)

    def outcome_ok(self) -> torch.Tensor:
        """(N,) bool, PURE current-state geometry (latch-free; smoke tests this):
        basket staged, exactly one present block contained in it, every other present
        block still inside the tower, blocks settled."""
        ib = self.in_basket() & self.present
        one_in = ib.sum(dim=1) == 1
        others_home = (self.in_tower() | ib | ~self.present).all(dim=1)
        settled = (self.blocks_settled() | ~self.present).all(dim=1)
        return self.staged_now() & one_in & others_home & settled

    def success(self) -> torch.Tensor:
        """(N,) bool: outcome_ok AND the contained block carries the outlet-transit
        credential AND the staged-dispense latch fired (the block got there by falling
        through the outlet with the basket in place — not by hand)."""
        ib = self.in_basket() & self.present
        credential = (ib & self._dispensed).sum(dim=1) == 1
        return self.outcome_ok() & credential & self._delivered

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: latched 0.20 staged + 0.15 actuated + 0.35 delivered
        (cap 0.70); exactly 1.0 iff success(); ~0 for the null policy (all latches
        start False; the basket spawns far off the mat and nothing moves by itself)."""
        s = 0.20 * (self._staged | self.staged_now()).float() \
            + 0.15 * (self._actuated | self.actuated_now()).float() \
            + 0.35 * self._delivered.float()
        return torch.where(self.success(), torch.ones_like(s), s.clamp(max=0.70))


register_env("simgen", lambda: EnvCfg(scene="butter_dispenser", robot="null"))
