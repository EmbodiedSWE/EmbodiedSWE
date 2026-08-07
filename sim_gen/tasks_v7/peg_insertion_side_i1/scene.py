"""RamEjectScene — free the trapped ball by RAMMING it through the tube into the basin
(sim_gen task `peg_insertion_side_i1`).

Derived from maniskill/peg_insertion_side, but STRATEGICALLY different: the seed is a
goal-insertion — grasp a peg lying on the table, align it horizontally, and translate
it into a tight hole in a fixed block; the checker is pure relative-bbox containment of
the PEG in the block's frame, so the manipulated object IS the judged object and the
whole task ends at insertion depth. Here insertion is INSTRUMENTAL TOOL USE and the
judged object is one the gripper can never touch: a yellow ball sits trapped deep
inside an enclosed horizontal tube (32 mm square bore, ball at least 60 mm from the
entry — no gripper reaches it), and success requires using the long green ramrod as a
push-rod: insert it into the tube's ENTRY end and drive a ~170 mm stroke so the BALL is
expelled from the far muzzle and falls into the blue catch basin mounted under it. The
rod's own final pose is irrelevant; success() judges the BALL's settled containment in
the basin. A solver needs a different plan (identify the entry end as the open end away
from the basin, fetch the rod from its rack, align with the bore, then execute a long
through-stroke whose purpose is another object's motion) and a different code structure
(a tool-transmission loop terminated by the ball's ejection + a containment predicate
on the non-manipulated object, not an insertion servo + a peg-pose bbox check). The
seed's own end state — rod inserted partway into the hole, nothing else moved — is
expressible here and is smoke control #6: it is NOT success and its credit caps at the
small instrumental-insertion share.

Judged in the ASSEMBLY body frame (the tube+basin unit is one kinematic body whose xy
and yaw are randomized, so the bore axis must be read from the scene). success() iff
the YELLOW ball rests inside the basin interior (x/y within the walls, centre below the
wall-top height, i.e. real containment on the basin floor) AND settled. Identity
matters: a red decoy ball of the same size lies loose on the table, and placing IT in
the basin counts for nothing. score() is graded and latched every physics substep:
0.10 * best rod approach (normalized by the episode's own spawn distance) + 0.25 * best
instrumental insertion depth (rod tip inside the bore) + 0.45 * best ball progress
along the bore toward the muzzle (the load-bearing transmission), capped at 0.80; 0.90
once the ball is in the basin; 1.0 iff success(). Doing nothing scores ~0.

Assets are fully procedural (no external files):
  - assembly: one KINEMATIC compound body — a pedestal carrying an enclosed square tube
    (bore 32 x 32 mm, 180 mm long, floor 60 mm above the table; both ends open; roof
    and side walls make the bore reachable only through the ends), whose muzzle half
    overhangs a BLUE walled catch basin (interior 144 x 144 mm, walls 45 mm — the wall
    top clears the overhanging tube bottom by 7 mm).
  - ball: YELLOW sphere, 28 mm dia (2 mm clearance per side in the bore), spawned ON
    the bore floor 60-105 mm deep from the entry.
  - rod: GREEN cylinder, 22 mm dia x 280 mm (5 mm radial clearance in the bore; long
    enough that the hand stays >= 90 mm outside the entry at full stroke), resting in a
    U-notched RACK (two dark supports, top at 45 mm) so the jaw grasps it well off the
    ground.
  - decoy: RED sphere, same 28 mm dia, loose on the table (identity control).
Contact offsets are explicit and small (1 mm): the default ~2 cm offset would eat the
2-5 mm working clearances inside the bore.

Per-episode randomization: assembly xy jitter + yaw (moves the bore axis and the basin
with it), rack xy jitter + free yaw (the rod spawns on the rack, so its pose and axis
move with it, plus axial jitter), ball depth in the bore, decoy xy jitter. Heavy
imports (isaaclab, pxr) are deferred so importing this module — and registering the
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
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _spawn_assembly(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC tube + pedestal + basin assembly at `prim_path`. Origin =
    the ENTRY face centre at TABLE level, local +x = bore axis toward the muzzle:
      - pedestal under the entry half of the tube;
      - enclosed square tube (bottom/top slabs + two side walls), both ends open;
      - walled catch basin on the table under the overhanging muzzle half."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(10.0)

    bh, wt, bf, L = cfg.bore_half, cfg.wall_t, cfg.bore_floor_z, cfg.tube_len
    co = cfg.contact_offset
    tube_c, basin_c, ped_c = cfg.tube_color, cfg.basin_color, cfg.pedestal_color
    outer_w = 2 * (bh + wt)  # tube outer width (y)
    roof_z0 = bf + 2 * bh  # bore roof (top-slab underside)

    # pedestal: under the entry half of the tube, table -> tube-bottom-slab underside
    ped_h = bf - wt
    _box(stage, f"{prim_path}/pedestal", (cfg.ped_len, outer_w, ped_h),
         (cfg.ped_len / 2, 0.0, ped_h / 2), ped_c, co)
    # tube: bottom slab, top slab, two side walls (x spans 0..L, both ends OPEN)
    _box(stage, f"{prim_path}/tube_bottom", (L, outer_w, wt),
         (L / 2, 0.0, bf - wt / 2), tube_c, co)
    _box(stage, f"{prim_path}/tube_top", (L, outer_w, wt),
         (L / 2, 0.0, roof_z0 + wt / 2), tube_c, co)
    for tag, sgn in (("tube_yp", 1.0), ("tube_yn", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (L, wt, 2 * bh),
             (L / 2, sgn * (bh + wt / 2), bf + bh), tube_c, co)
    # basin: floor + four walls, on the table under the overhanging muzzle
    bx0, bx1, byh, bwh = cfg.basin_x0, cfg.basin_x1, cfg.basin_y_half, cfg.basin_wall_h
    blen = bx1 - bx0
    _box(stage, f"{prim_path}/basin_floor", (blen, 2 * byh, cfg.basin_floor_t),
         ((bx0 + bx1) / 2, 0.0, cfg.basin_floor_t / 2), basin_c, co)
    _box(stage, f"{prim_path}/basin_xn", (wt, 2 * byh, bwh),
         (bx0 + wt / 2, 0.0, bwh / 2), basin_c, co)
    _box(stage, f"{prim_path}/basin_xp", (wt, 2 * byh, bwh),
         (bx1 - wt / 2, 0.0, bwh / 2), basin_c, co)
    for tag, sgn in (("basin_yp", 1.0), ("basin_yn", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (blen - 2 * wt, wt, bwh),
             ((bx0 + bx1) / 2, sgn * (byh - wt / 2), bwh / 2), basin_c, co)
    return root


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC rod rack at `prim_path`: two U-notched supports (base box +
    two ridge boxes each) holding the rod horizontal at grasp height, axis = local x."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(2.0)

    co, color = cfg.contact_offset, cfg.color
    st, sw, sh = cfg.sup_len, cfg.sup_w, cfg.sup_h
    for tag, sx in (("sup_xp", cfg.sup_x), ("sup_xn", -cfg.sup_x)):
        _box(stage, f"{prim_path}/{tag}", (st, sw, sh), (sx, 0.0, sh / 2), color, co)
        for rtag, sgn in (("rp", 1.0), ("rn", -1.0)):
            _box(stage, f"{prim_path}/{tag}_{rtag}", (st, cfg.ridge_w, cfg.ridge_h),
                 (sx, sgn * (cfg.ridge_gap_half + cfg.ridge_w / 2), sh + cfg.ridge_h / 2),
                 color, co)
    return root


def _assembly_spawner_cfg(*, bore_half: float, wall_t: float, bore_floor_z: float,
                          tube_len: float, ped_len: float, basin_x0: float,
                          basin_x1: float, basin_y_half: float, basin_wall_h: float,
                          basin_floor_t: float, tube_color: tuple, basin_color: tuple,
                          pedestal_color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "assembly" not in _SPAWNER_CACHE:

        @configclass
        class RamAssemblySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_assembly)
            bore_half: float = 0.016
            wall_t: float = 0.008
            bore_floor_z: float = 0.060
            tube_len: float = 0.18
            ped_len: float = 0.09
            basin_x0: float = 0.16
            basin_x1: float = 0.32
            basin_y_half: float = 0.08
            basin_wall_h: float = 0.045
            basin_floor_t: float = 0.008
            tube_color: tuple = (0.40, 0.40, 0.44)
            basin_color: tuple = (0.15, 0.35, 0.85)
            pedestal_color: tuple = (0.30, 0.30, 0.33)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["assembly"] = RamAssemblySpawnerCfg

    return _SPAWNER_CACHE["assembly"](
        mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        bore_half=bore_half, wall_t=wall_t, bore_floor_z=bore_floor_z,
        tube_len=tube_len, ped_len=ped_len, basin_x0=basin_x0, basin_x1=basin_x1,
        basin_y_half=basin_y_half, basin_wall_h=basin_wall_h,
        basin_floor_t=basin_floor_t, tube_color=tube_color, basin_color=basin_color,
        pedestal_color=pedestal_color, contact_offset=contact_offset,
    )


def _rack_spawner_cfg(*, sup_x: float, sup_len: float, sup_w: float, sup_h: float,
                      ridge_gap_half: float, ridge_w: float, ridge_h: float,
                      color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rack" not in _SPAWNER_CACHE:

        @configclass
        class RodRackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            sup_x: float = 0.08
            sup_len: float = 0.03
            sup_w: float = 0.05
            sup_h: float = 0.045
            ridge_gap_half: float = 0.013
            ridge_w: float = 0.008
            ridge_h: float = 0.012
            color: tuple = (0.20, 0.20, 0.22)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["rack"] = RodRackSpawnerCfg

    return _SPAWNER_CACHE["rack"](
        mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        sup_x=sup_x, sup_len=sup_len, sup_w=sup_w, sup_h=sup_h,
        ridge_gap_half=ridge_gap_half, ridge_w=ridge_w, ridge_h=ridge_h,
        color=color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RamEjectSceneCfg(BaseCfg):
    """Config for `RamEjectScene`. Honesty knobs asserted in `__post_init__`: the ball
    spawns deep enough that no gripper reaches it through the 32 mm bore, the rod is
    long enough that a hand driving the full eject stroke stays well outside the entry,
    and the basin interior brackets the muzzle so the free-falling ball lands inside."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    settle_lin: float = tunable(0.05)  # max ball |lin vel| when judging success (m/s)
    basin_margin: float = tunable(0.002)  # xy slack inside the basin interior when judging
    in_basin_z_max: float = tunable(0.042)  # ball centre below this = under the wall top (m)
    in_basin_z_min: float = tunable(0.010)  # ball centre above this = on the basin floor (m)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    asm_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the assembly at reset (m)
    asm_yaw_deg: float = tunable(25.0)  # uniform +/- assembly yaw at reset (deg)
    rack_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the rod rack at reset (m)
    rack_yaw_deg: float = tunable(180.0)  # uniform +/- rack yaw at reset (free)
    rod_axial_jitter: float = tunable(0.02)  # uniform +/- rod slide along the rack notches (m)
    ball_depth_range: tuple = tunable((0.060, 0.105))  # ball centre from the entry face (m)
    decoy_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the red decoy (m)

    # --- tunable: placement ------------------------------------------------------------------
    asm_pos: tuple = tunable((-0.10, 0.06))  # entry-face centre, nominal (bore axis -> +x)
    rack_pos: tuple = tunable((-0.08, -0.18))  # rod rack centre, nominal
    decoy_pos: tuple = tunable((0.30, -0.08))  # red decoy ball, nominal

    # --- info: structure ---------------------------------------------------------------------
    bore_half: float = info(0.016)  # square bore half-width (32 x 32 mm inside)
    wall_t: float = info(0.008)
    bore_floor_z: float = info(0.060)  # bore floor above the table
    tube_len: float = info(0.18)  # entry face at local x=0, muzzle at x=tube_len
    ped_len: float = info(0.09)  # pedestal under the entry half; the rest overhangs
    basin_x0: float = info(0.16)  # basin outer footprint along the bore axis
    basin_x1: float = info(0.32)
    basin_y_half: float = info(0.08)
    basin_wall_h: float = info(0.045)
    basin_floor_t: float = info(0.008)
    ball_r: float = info(0.014)  # 28 mm ball in the 32 mm bore (2 mm/side clearance)
    ball_mass: float = info(0.03)
    rod_r: float = info(0.011)  # 22 mm rod (5 mm radial clearance in the bore)
    rod_len: float = info(0.28)
    rod_mass: float = info(0.15)
    rack_sup_x: float = info(0.08)  # rack support half-spacing along the rod axis
    rack_sup_len: float = info(0.03)
    rack_sup_w: float = info(0.05)
    rack_sup_h: float = info(0.045)  # rod rests at sup_h + rod_r (grasp height)
    rack_ridge_gap_half: float = info(0.013)  # U-notch inner half-gap (26 mm for the 22 mm rod)
    rack_ridge_w: float = info(0.008)
    rack_ridge_h: float = info(0.012)
    tube_color: tuple = info((0.40, 0.40, 0.44))
    basin_color: tuple = info((0.15, 0.35, 0.85))
    pedestal_color: tuple = info((0.30, 0.30, 0.33))
    rack_color: tuple = info((0.20, 0.20, 0.22))
    rod_color: tuple = info((0.10, 0.65, 0.20))
    ball_color: tuple = info((0.95, 0.85, 0.10))
    decoy_color: tuple = info((0.85, 0.15, 0.12))
    # Explicit small offsets: the ~2 cm default would eat the 2-5 mm bore clearances.
    contact_offset: float = info(0.001)

    # Derived (filled in __post_init__).
    bore_center_z: float = field(default=None, init=False)
    eject_tip_x: float = field(default=None, init=False)  # rod-tip x that ejects the ball
    basin_in_x0: float = field(default=None, init=False)  # basin interior bounds
    basin_in_x1: float = field(default=None, init=False)
    basin_in_y: float = field(default=None, init=False)
    rod_rest_z: float = field(default=None, init=False)  # rod centre height on the rack

    def __post_init__(self) -> None:
        self.bore_center_z = self.bore_floor_z + self.bore_half
        # The ball loses floor support once its centre passes the muzzle (x = tube_len);
        # a flat rod tip touching the ball puts the tip one ball radius behind its centre.
        self.eject_tip_x = self.tube_len - self.ball_r
        self.basin_in_x0 = self.basin_x0 + self.wall_t
        self.basin_in_x1 = self.basin_x1 - self.wall_t
        self.basin_in_y = self.basin_y_half - self.wall_t
        self.rod_rest_z = self.rack_sup_h + self.rod_r

        assert 2 * self.bore_half - 2 * self.rod_r >= 0.008, "rod needs >= 8 mm bore clearance"
        assert 2 * self.bore_half - 2 * self.ball_r >= 0.003, "ball must fit the bore"
        assert self.basin_wall_h <= self.bore_floor_z - self.wall_t - 0.004, (
            "basin wall top must clear the overhanging tube bottom")
        assert self.rod_len >= self.eject_tip_x + 0.10, (
            "rod must let the hand stay >= 10 cm outside the entry at full eject stroke")
        assert self.ball_depth_range[0] >= 0.055, (
            "ball must spawn deep enough that no gripper finger reaches it through the bore")
        assert self.ball_depth_range[1] <= self.tube_len - 0.05, (
            "ball must spawn clear of the muzzle (ejection must need a real stroke)")
        assert self.tube_len - self.basin_in_x0 >= 0.008, "basin near wall must sit behind the muzzle"
        assert self.basin_in_x1 - self.tube_len >= 0.10, "basin must extend past the landing zone"
        assert 2 * self.rack_ridge_gap_half >= 2 * self.rod_r + 0.003, "rod must drop into the rack notch"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("ram_eject")
class RamEjectScene(BaseScene):
    cfg: RamEjectSceneCfg

    def __init__(self, cfg: RamEjectSceneCfg | None = None) -> None:
        super().__init__(cfg or RamEjectSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        ax, ay = c.asm_pos
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
            "assembly": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Assembly",
                spawn=_assembly_spawner_cfg(
                    bore_half=c.bore_half, wall_t=c.wall_t, bore_floor_z=c.bore_floor_z,
                    tube_len=c.tube_len, ped_len=c.ped_len, basin_x0=c.basin_x0,
                    basin_x1=c.basin_x1, basin_y_half=c.basin_y_half,
                    basin_wall_h=c.basin_wall_h, basin_floor_t=c.basin_floor_t,
                    tube_color=c.tube_color, basin_color=c.basin_color,
                    pedestal_color=c.pedestal_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(ax, ay, 0.0)),
            ),
            "rack": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=_rack_spawner_cfg(
                    sup_x=c.rack_sup_x, sup_len=c.rack_sup_len, sup_w=c.rack_sup_w,
                    sup_h=c.rack_sup_h, ridge_gap_half=c.rack_ridge_gap_half,
                    ridge_w=c.rack_ridge_w, ridge_h=c.rack_ridge_h, color=c.rack_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_pos[0], c.rack_pos[1], 0.0)),
            ),
            "rod": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rod",
                spawn=sim_utils.CylinderCfg(
                    radius=c.rod_r, height=c.rod_len,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.10,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.rod_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.rod_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_pos[0], c.rack_pos[1], c.rod_rest_z + 0.001),
                    rot=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0),
                ),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.20, angular_damping=0.20,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.5, dynamic_friction=0.5, restitution=0.05),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ball_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(ax + 0.08, ay, c.bore_floor_z + c.ball_r + 0.0005)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.20, angular_damping=0.20),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.5, dynamic_friction=0.5, restitution=0.05),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.decoy_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.decoy_pos[0], c.decoy_pos[1], c.ball_r + 0.002)),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.assembly: RigidObject = env.iscene["assembly"]
        self.rack: RigidObject = env.iscene["rack"]
        self.rod: RigidObject = env.iscene["rod"]
        self.ball: RigidObject = env.iscene["ball"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.s0 = torch.full((n,), 0.08, device=dev)  # ball spawn depth, per episode
        self.d0 = torch.full((n,), 0.30, device=dev)  # rod-spawn -> entry distance
        self.approach_latch = torch.zeros(n, device=dev)
        self.insert_latch = torch.zeros(n, device=dev)
        self.push_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: assembly with xy jitter + yaw (bore axis and basin move
        together), rack with xy jitter + free yaw and the rod seated in its notches
        (axial jitter), ball at a random depth on the bore floor, decoy jittered on the
        table; latches zeroed, approach baseline `d0` and ball depth `s0` captured."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- assembly ---
        asm_xy = torch.tensor(c.asm_pos, device=dev).expand(m, 2).clone()
        asm_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.asm_jitter
        asm_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.asm_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = asm_xy
        st[:, 3] = torch.cos(asm_yaw / 2)
        st[:, 6] = torch.sin(asm_yaw / 2)
        st[:, 0:3] += origin
        self.assembly.write_root_state_to_sim(st, env_ids)

        # --- ball: on the bore floor at random depth, in the SAME sampled asm frame ---
        s0 = c.ball_depth_range[0] + torch.rand(m, device=dev) * (
            c.ball_depth_range[1] - c.ball_depth_range[0])
        ca, sa = torch.cos(asm_yaw), torch.sin(asm_yaw)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = asm_xy[:, 0] + ca * s0
        st[:, 1] = asm_xy[:, 1] + sa * s0
        st[:, 2] = c.bore_floor_z + c.ball_r + 0.0005
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.ball.write_root_state_to_sim(st, env_ids)

        # --- rack + rod seated in the notches (rod axis = rack local x) ---
        rack_xy = torch.tensor(c.rack_pos, device=dev).expand(m, 2).clone()
        rack_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.rack_jitter
        rack_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rack_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = rack_xy
        st[:, 3] = torch.cos(rack_yaw / 2)
        st[:, 6] = torch.sin(rack_yaw / 2)
        st[:, 0:3] += origin
        self.rack.write_root_state_to_sim(st, env_ids)

        ax_j = (torch.rand(m, device=dev) * 2 - 1) * c.rod_axial_jitter
        cr, sr = torch.cos(rack_yaw), torch.sin(rack_yaw)
        c45 = math.cos(math.pi / 4)
        half = rack_yaw / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = rack_xy[:, 0] + cr * ax_j
        st[:, 1] = rack_xy[:, 1] + sr * ax_j
        st[:, 2] = c.rod_rest_z + 0.001
        # lying flat along rack x: q = qz(yaw) * qy(90 deg)
        st[:, 3] = torch.cos(half) * c45
        st[:, 4] = -torch.sin(half) * c45
        st[:, 5] = torch.cos(half) * c45
        st[:, 6] = torch.sin(half) * c45
        st[:, 0:3] += origin
        self.rod.write_root_state_to_sim(st, env_ids)

        # --- decoy ---
        dec_xy = torch.tensor(c.decoy_pos, device=dev).expand(m, 2).clone()
        dec_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.decoy_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = dec_xy
        st[:, 2] = c.ball_r + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.decoy.write_root_state_to_sim(st, env_ids)

        # --- baselines + latches ---
        self.s0[env_ids] = s0
        self.d0[env_ids] = (rack_xy - asm_xy).norm(dim=-1).clamp(min=0.05)
        self.approach_latch[env_ids] = 0.0
        self.insert_latch[env_ids] = 0.0
        self.push_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "assembly": self.assembly.data.root_state_w[env_ids].clone(),
            "rack": self.rack.data.root_state_w[env_ids].clone(),
            "rod": self.rod.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "s0": self.s0[env_ids].clone(),
            "d0": self.d0[env_ids].clone(),
            "approach_latch": self.approach_latch[env_ids].clone(),
            "insert_latch": self.insert_latch[env_ids].clone(),
            "push_latch": self.push_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.assembly.write_root_state_to_sim(state["assembly"], env_ids)
        self.rack.write_root_state_to_sim(state["rack"], env_ids)
        self.rod.write_root_state_to_sim(state["rod"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.s0[env_ids] = state["s0"]
        self.d0[env_ids] = state["d0"]
        self.approach_latch[env_ids] = state["approach_latch"]
        self.insert_latch[env_ids] = state["insert_latch"]
        self.push_latch[env_ids] = state["push_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A gray horizontal tube ({c.tube_len * 1000:.0f} mm long, enclosed on all "
            f"sides, square bore {2 * c.bore_half * 1000:.0f} mm wide, open only at its two "
            f"ends) is mounted {c.bore_floor_z * 1000:.0f} mm above the table on a pedestal. "
            f"One end of the tube overhangs a BLUE walled catch basin standing on the table "
            f"directly below it — that end is the MUZZLE; the opposite open end, away from "
            f"the basin, is the ENTRY. A YELLOW ball ({2 * c.ball_r * 1000:.0f} mm) is "
            f"trapped deep inside the bore: it is visible only through the open ends and no "
            f"gripper can reach it. A GREEN rod ({2 * c.rod_r * 1000:.0f} mm thick, "
            f"{c.rod_len * 1000:.0f} mm long — longer than the tube) rests horizontally in "
            f"the notches of a dark two-post rack nearby. A loose RED ball of the same size "
            f"lies on the table; it is a decoy.\n"
            f"Goal: free the yellow ball and get it into the blue basin. Pick the green rod "
            f"off its rack, insert it into the ENTRY end of the tube, and push it through so "
            f"the yellow ball is driven along the bore, drops out of the muzzle, and comes to "
            f"rest inside the basin. The task is judged on the YELLOW ball settled inside "
            f"the basin: leaving the rod inserted (or anywhere else) is fine. Pushing the "
            f"rod partway in without expelling the ball does not count; the ball landing "
            f"outside the basin does not count; putting the RED decoy in the basin counts "
            f"for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Take the green rod from its rack, insert it into the open end of the gray tube "
            "away from the blue basin, and push it through so the trapped yellow ball is "
            "expelled from the far end and settles inside the basin. Only the yellow ball "
            "in the basin counts; the red ball is a decoy."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _to_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> assembly body frame (origin = entry face at table)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.assembly.data.root_quat_w,
                                  p_w - self.assembly.data.root_pos_w)

    def _rod_axis_w(self) -> torch.Tensor:
        """(N,3) rod axis (local +z) in world frame."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(self.rod.data.root_quat_w, ez)

    def rod_tip_local(self) -> torch.Tensor:
        """(N,3) the rod END closer to the muzzle (larger local x), assembly frame."""
        axis = self._rod_axis_w() * (self.cfg.rod_len / 2)
        e1 = self._to_local(self.rod.data.root_pos_w + axis)
        e2 = self._to_local(self.rod.data.root_pos_w - axis)
        return torch.where((e1[:, 0] >= e2[:, 0]).unsqueeze(-1), e1, e2)

    def ball_local(self) -> torch.Tensor:
        return self._to_local(self.ball.data.root_pos_w)

    def _in_basin(self, loc: torch.Tensor) -> torch.Tensor:
        """(N,) bool: a point (assembly frame) inside the basin interior, below the
        wall top and on the floor — real containment."""
        c = self.cfg
        return ((loc[:, 0] > c.basin_in_x0 + c.basin_margin)
                & (loc[:, 0] < c.basin_in_x1 - c.basin_margin)
                & (loc[:, 1].abs() < c.basin_in_y - c.basin_margin)
                & (loc[:, 2] > c.in_basin_z_min) & (loc[:, 2] < c.in_basin_z_max))

    def ball_in_basin(self) -> torch.Tensor:
        """(N,) bool: the YELLOW ball contained in the basin (identity matters — the
        decoy is judged nowhere)."""
        return self._in_basin(self.ball_local())

    def settled(self) -> torch.Tensor:
        return self.ball.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin

    # ----- graded progress --------------------------------------------------------------------
    def insert_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: instrumental insertion — rod tip progress along the bore
        toward the eject stroke, gated on the tip actually being inside the bore
        cross-section (a rod waved in the air or laid on the table earns nothing)."""
        c = self.cfg
        tip = self.rod_tip_local()
        in_bore = ((tip[:, 0] > 0.0) & (tip[:, 1].abs() < c.bore_half)
                   & (tip[:, 2] > c.bore_floor_z) & (tip[:, 2] < c.bore_floor_z + 2 * c.bore_half))
        return (tip[:, 0] / c.eject_tip_x).clamp(0.0, 1.0) * in_bore.float()

    def push_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: the load-bearing transmission — ball progress from its spawn
        depth toward the muzzle, while the ball is in the bore; 1.0 once it is in the
        basin. Ejecting it backwards or out of the bore sideways earns nothing new."""
        c = self.cfg
        loc = self.ball_local()
        in_bore = ((loc[:, 1].abs() < c.bore_half + 0.01)
                   & (loc[:, 2] > c.bore_floor_z - 0.005))
        denom = (c.tube_len - self.s0).clamp(min=0.01)
        raw = ((loc[:, 0] - self.s0) / denom).clamp(0.0, 1.0) * in_bore.float()
        return torch.where(self.ball_in_basin(), torch.ones_like(raw), raw)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch best rod approach, best instrumental insertion, and best ball push each
        physics substep, so transient progress keeps its credit."""
        d = (self.rod.data.root_pos_w - self.assembly.data.root_pos_w)[:, :2].norm(dim=-1)
        approach = (1.0 - d / self.d0).clamp(0.0, 1.0)
        self.approach_latch = torch.maximum(self.approach_latch, approach)
        self.insert_latch = torch.maximum(self.insert_latch, self.insert_frac())
        self.push_latch = torch.maximum(self.push_latch, self.push_frac())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the yellow ball rests inside the basin, settled."""
        return self.ball_in_basin() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 * latched rod approach + 0.25 * latched
        instrumental insertion + 0.45 * latched ball push (max 0.80), 0.9 once the ball
        is in the basin, 1.0 iff success. Doing nothing scores ~0; the seed's
        rod-partway-in end state caps at the small insertion share."""
        base = (0.10 * self.approach_latch + 0.25 * self.insert_latch
                + 0.45 * self.push_latch).clamp(0.0, 0.80)
        s = torch.where(self.ball_in_basin(), torch.maximum(base, base.new_tensor(0.9)), base)
        return torch.where(self.success(), s.new_tensor(1.0), s)


register_env("simgen", lambda: EnvCfg(scene="ram_eject", robot="null"))
