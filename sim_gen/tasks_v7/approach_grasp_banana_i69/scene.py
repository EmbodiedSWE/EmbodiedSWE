"""TiltMazeScene — steer an untouchable ball through a grated labyrinth by tilting the
whole maze on a 2-DOF spring-centered gimbal (sim_gen task `approach_grasp_banana_i69`).

Derived from pick_place/approach_grasp_banana, but STRATEGICALLY different: the seed is
a PREHENSILE PICK-AND-PLACE — a Franka approaches a banana lying among clutter, closes
its jaw around it and carries it toward a basket; the judged payload is exactly the
thing the gripper holds, and the plan is approach -> grasp -> transport -> release.
Here the judged payload can NEVER be touched: an orange ball (28 mm) lives inside a
walled labyrinth tray whose corridor section is covered by a grate of roof slats with
6 mm gaps — narrower than a fingertip and far narrower than the ball. The only handles
the arm may use are four YELLOW PRESS TABS on the tray's outside rim. The tray hangs
on a two-axis revolute gimbal (pitch + roll, +/-12 deg hard stops) above a fixed
pedestal, spring-centered back to level; pressing a tab tilts the maze and GRAVITY
moves the ball. The maze forks: two terminal wells (20 mm deep) sit at the two ends of
a cross corridor, and a green BEACON post placed per episode marks which well is the
goal. The decoy well is an IRREVERSIBLE trap: escaping it would need a ~25 deg tilt
but the gimbal stops at 12 deg — commit to the wrong branch and the episode is lost.
Success = the ball settled in the beacon-marked well with the tray released back to
level. Plan skeleton: indirect gravity-steering of an unreachable object through an
anchored compliant mechanism, with a per-episode branch decision and an irreversible
failure mode — no grasp, no carry, no proximity goal, nothing in common with the seed
beyond "an object must end up at a target".

Mechanics (carousel_ferry-proven pattern — plain rigid bodies + authored USD joints;
tilt "feel" is an external spring/damper wrench applied in `post_step`, which also
consumes the `tilt_drive` buffer that solve/smoke probes write): fixed kinematic
pedestal --revolute X--> small cradle link --revolute Y--> tray (one rigid compound:
floors, wells, rim walls, corridor fillers, 12 roof slats, 4 press tabs). The ball is
a free rigid body riding the maze floor.

Rubric (graded 0..1, latched credit anchored in the demonstrated solve.py trajectory;
every latch requires PATH CONTINUITY — per-substep ball travel below `cont_max` — so a
teleported ball earns nothing):
  - `departed` (latch): the ball, moving continuously, passes through the roofed ENTRY
    corridor (channel |x| < chan_hw, y in [-0.038, 0.015], on the floor) — it must
    leave the open start bay through the maze, not appear elsewhere.
  - `crossed` (latch): after departing, the ball reaches the CROSS corridor
    (y < -0.038) — the fork row where the branch decision happens.
  - `branch` (latched max in [0,1]): progress toward the GOAL side along the cross
    corridor, clamp(side * x / branch_sat, 0, 1); rolling the wrong way earns 0.
  - `potted` (latch): after crossing with branch > branch_gate, the ball is INSIDE the
    goal well (side * x in the well span, z below the well lip band) — reached by
    falling in, 20 mm below the corridor floor.
  - success(): `potted` AND the ball is currently settled inside the goal well AND the
    tray is released level (tilt < level_tol) and quiet — steering credit only counts
    if the mechanism is let go and the ball stays put.
  - score() = 0.10*departed + 0.15*crossed + 0.30*branch + 0.15*potted + 0.30*success
    -> exactly 1.0 iff success(); ~0 for the null policy; a wrong-well episode caps at
    0.25; disturbing the tray after success falls back to 0.70, never 0.

Per-episode randomization (readback-verified in smoke.py): the goal SIDE (beacon east
or west — the branch decision flips) and the ball's spawn position inside the start
bay. A memorized fixed tilt schedule fails across episodes.

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


# ----- USD authoring helpers (pedestal + tray compound spawners) --------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _friction_material(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _decorate(prim, color, contact_offset: float, material=None) -> None:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None, visual_only: bool = False) -> None:
    """Author one box child prim (translate -> scale; authored once)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    if visual_only:
        UsdGeom.Gprim(seg.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    else:
        _decorate(seg.GetPrim(), color, contact_offset, material)


def _spawn_pedestal(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC pedestal. Local frame: origin on the ground at the plan center.
    A low collider block plus a visual-only neck that reaches up toward the cradle
    (no collider there — the gimbal must clear it at full tilt)."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(10.0)
    mat = _friction_material(stage, f"{prim_path}/mat", 0.6, 0.5)
    _box(stage, f"{prim_path}/base", (0.08, 0.08, cfg.base_h),
         (0.0, 0.0, cfg.base_h / 2), cfg.color, cfg.contact_offset, material=mat)
    _box(stage, f"{prim_path}/neck", (0.036, 0.036, 0.008),
         (0.0, 0.0, cfg.base_h + 0.003), cfg.color, cfg.contact_offset,
         visual_only=True)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC labyrinth tray at `prim_path`. Local frame: origin at the
    GIMBAL PIVOT; the maze floor top sits at local z = floor_top < 0. Layout (local):
      - start bay (open sky): channel |x| < chan_hw, y in [cross_y1, bay_y_hi]
      - roofed entry corridor: same channel, y in [cross_y1 - eps, slat_y_hi]
      - cross corridor: y in [cross_y0, cross_y1], full width between the side walls
      - terminal wells: x in +/-[well_x0, well_x1], y in [cross_y0, cross_y1],
        floor `well_depth` below the main floor
      - 12 roof slats over everything south of the bay (gap = pitch - width = 6 mm)
      - 4 press tabs OUTSIDE the rim walls (N/S/E/W) — the only intended handles.
    Mass, CoM (at the pivot) and inertia are authored EXPLICITLY."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in cfg.inertia]))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.2)
    px.CreateAngularDampingAttr(0.05)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    mat = _friction_material(stage, f"{prim_path}/mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    ft, th = cfg.floor_top, cfg.floor_th
    wx0, wx1 = cfg.well_x0, cfg.well_x1
    cy0, cy1 = cfg.cross_y0, cfg.cross_y1
    ny, sy = cfg.inner_n_y, cfg.inner_s_y
    wt, w_top, w_bot = cfg.wall_t, cfg.wall_top, cfg.wall_bot

    # floors -------------------------------------------------------------------------------
    _box(stage, f"{prim_path}/floor_n", (2 * wx1, ny - cy1, th),
         (0.0, (ny + cy1) / 2, ft - th / 2), cfg.floor_color, co, material=mat)
    _box(stage, f"{prim_path}/floor_s", (2 * wx0, cy1 - cy0, th),
         (0.0, (cy0 + cy1) / 2, ft - th / 2), cfg.floor_color, co, material=mat)
    wft = ft - cfg.well_depth  # well floor top
    for sx, nm in ((+1, "e"), (-1, "w")):
        _box(stage, f"{prim_path}/well_floor_{nm}",
             (wx1 - wx0, cy1 - cy0, 0.008),
             (sx * (wx0 + wx1) / 2, (cy0 + cy1) / 2, wft - 0.004),
             cfg.well_color, co, material=mat)
    # rim walls ----------------------------------------------------------------------------
    wall_h = w_top - w_bot
    wall_zc = (w_top + w_bot) / 2
    _box(stage, f"{prim_path}/wall_n", (2 * wx1 + 2 * wt, wt, wall_h),
         (0.0, ny + wt / 2, wall_zc), cfg.wall_color, co, material=mat)
    _box(stage, f"{prim_path}/wall_s", (2 * wx1 + 2 * wt, wt, wall_h),
         (0.0, sy - wt / 2, wall_zc), cfg.wall_color, co, material=mat)
    for sx, nm in ((+1, "e"), (-1, "w")):
        _box(stage, f"{prim_path}/wall_{nm}", (wt, ny - sy, wall_h),
             (sx * (wx1 + wt / 2), (ny + sy) / 2, wall_zc),
             cfg.wall_color, co, material=mat)
    # corridor fillers (bound the channel; solid from the floor up past the slats) ---------
    fil_w = wx1 - cfg.chan_hw
    for sx, nm in ((+1, "e"), (-1, "w")):
        _box(stage, f"{prim_path}/filler_{nm}", (fil_w, ny - cy1, 0.050),
             (sx * (cfg.chan_hw + fil_w / 2), (ny + cy1) / 2, ft + 0.025),
             cfg.wall_color, co, material=mat)
    # roof slats ---------------------------------------------------------------------------
    slat_y_c = (cfg.slat_y_hi + cy0) / 2 - wt / 2  # cover cy0-eps .. slat_y_hi
    slat_len = cfg.slat_y_hi - cy0 + wt
    for k in range(cfg.slat_n):
        x = -((cfg.slat_n - 1) / 2) * cfg.slat_pitch + k * cfg.slat_pitch
        _box(stage, f"{prim_path}/slat_{k}", (cfg.slat_w, slat_len, cfg.slat_th),
             (x, slat_y_c, cfg.slat_z), cfg.slat_color, co, material=mat)
    # press tabs (outside the rim, flush with the wall tops) -------------------------------
    tb, tz = 2 * cfg.tab_half, cfg.tab_z
    for nm, (tx, ty) in (("tab_n", (0.0, ny + wt + cfg.tab_half)),
                         ("tab_s", (0.0, sy - wt - cfg.tab_half)),
                         ("tab_e", (wx1 + wt + cfg.tab_half, (ny + sy) / 2)),
                         ("tab_w", (-(wx1 + wt + cfg.tab_half), (ny + sy) / 2))):
        _box(stage, f"{prim_path}/{nm}", (tb, tb, cfg.tab_th), (tx, ty, tz),
             cfg.tab_color, co, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tray" not in _SPAWNER_CACHE:

        @configclass
        class PedestalSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pedestal)
            base_h: float = 0.045
            color: tuple = (0.15, 0.15, 0.17)
            contact_offset: float = 0.0015

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            floor_top: float = -0.020
            floor_th: float = 0.016
            well_depth: float = 0.020
            chan_hw: float = 0.033
            cross_y0: float = -0.100
            cross_y1: float = -0.040
            well_x0: float = 0.095
            well_x1: float = 0.155
            inner_n_y: float = 0.110
            inner_s_y: float = -0.100
            wall_t: float = 0.008
            wall_top: float = 0.028
            wall_bot: float = -0.048
            slat_n: int = 12
            slat_w: float = 0.020
            slat_pitch: float = 0.026
            slat_th: float = 0.006
            slat_z: float = 0.025
            slat_y_hi: float = 0.020
            tab_half: float = 0.025
            tab_th: float = 0.010
            tab_z: float = 0.023
            mass: float = 1.2
            inertia: tuple = (0.008, 0.015, 0.022)
            mu_static: float = 0.60
            mu_dynamic: float = 0.50
            floor_color: tuple = (0.75, 0.73, 0.68)
            well_color: tuple = (0.10, 0.15, 0.45)
            wall_color: tuple = (0.38, 0.38, 0.42)
            slat_color: tuple = (0.22, 0.22, 0.25)
            tab_color: tuple = (0.90, 0.78, 0.10)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["pedestal"] = PedestalSpawnerCfg
        _SPAWNER_CACHE["tray"] = TraySpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TiltMazeSceneCfg(BaseCfg):
    """Config for `TiltMazeScene`. Honesty is asserted in __post_init__: the decoy well
    is inescapable within the gimbal's stops, the ball fits everywhere it must travel
    but nothing (finger or ball) fits through the roof grate, the spring re-levels the
    tray within the success tolerance with the ball anywhere in the maze, the drive
    can reach the stops, and one substep of ball travel stays far under the
    path-continuity gate."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    level_tol_deg: float = tunable(3.0)   # success: tray tilt below this (released, re-leveled)
    settle_v: float = tunable(0.04)       # success: max ball |lin vel| (m/s)
    settle_w: float = tunable(0.30)       # success: max tray |ang vel| (rad/s)
    pot_z: float = tunable(-0.020)        # potted/in-well: ball center below this (tray frame)
    cont_max: float = tunable(0.02)       # path continuity: max ball travel per substep (m)
    branch_gate: float = tunable(0.85)    # potted requires branch progress beyond this

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    spawn_x: float = tunable(0.012)       # ball spawn |x| bound inside the start bay (m)
    spawn_y_lo: float = tunable(0.045)    # ball spawn y range (start bay, open sky)
    spawn_y_hi: float = tunable(0.095)

    # --- tunable: plant (difficulty dials) -------------------------------------------------------
    kappa: float = tunable(0.5)           # gimbal centering spring (N*m/rad-ish, on the up vector)
    tilt_damp: float = tunable(0.10)      # gimbal damping (N*m*s/rad)
    drive_max: float = tunable(0.30)      # |tilt_drive| clamp per axis (N*m)
    limit_deg: float = tunable(12.0)      # gimbal hard stops, each axis (deg)
    ball_mass: float = tunable(0.010)
    ball_mu: tuple = tunable((0.70, 0.60))
    tray_mu: tuple = tunable((0.60, 0.50))

    # --- info: structure (tray local frame: origin at the gimbal pivot) -------------------------
    post_pos: tuple = info((0.50, 0.0))   # pedestal plan center (world xy)
    pivot_z: float = info(0.115)          # gimbal pivot height above the ground
    base_h: float = info(0.045)           # pedestal collider height
    cradle_size: tuple = info((0.04, 0.04, 0.02))
    cradle_mass: float = info(0.08)
    cradle_drop: float = info(0.055)      # pivot sits this far above the cradle center
    tray_mass: float = info(1.2)
    tray_inertia: tuple = info((0.008, 0.015, 0.022))
    floor_top: float = info(-0.020)
    floor_th: float = info(0.016)
    well_depth: float = info(0.020)
    chan_hw: float = info(0.033)          # start-bay / entry-corridor channel half-width
    bay_y_hi: float = info(0.110)
    cross_y0: float = info(-0.100)        # cross corridor south edge (inner wall face)
    cross_y1: float = info(-0.040)        # cross corridor north edge
    well_x0: float = info(0.095)          # terminal wells: |x| in [well_x0, well_x1]
    well_x1: float = info(0.155)
    wall_t: float = info(0.008)
    wall_top: float = info(0.028)
    wall_bot: float = info(-0.048)
    slat_n: int = info(12)
    slat_w: float = info(0.020)
    slat_pitch: float = info(0.026)       # gap = pitch - width = 6 mm
    slat_th: float = info(0.006)
    slat_z: float = info(0.025)           # slat center height (bottom at 0.022)
    slat_y_hi: float = info(0.020)        # roof starts here going south (bay is open north)
    tab_half: float = info(0.025)
    tab_th: float = info(0.010)
    tab_z: float = info(0.023)
    ball_r: float = info(0.014)
    beacon_dx: float = info(0.28)         # beacon world x = post_x + side * beacon_dx
    beacon_y: float = info(-0.07)         # beside the cross-corridor row
    beacon_r: float = info(0.02)
    beacon_h: float = info(0.14)
    branch_sat: float = info(0.105)       # branch progress saturates at side*x = this
    finger_t: float = info(0.007)         # Franka fingertip min thickness (grate must beat it)
    base_pos: tuple = info((0.0, 0.0))    # the documented Franka base xy (TASK.md)
    reach: float = info(0.72)             # documented comfortable arm envelope from base_pos
    contact_offset: float = info(0.0015)
    # rubric weights (sum with success weight = 1.0)
    w_depart: float = info(0.10)
    w_cross: float = info(0.15)
    w_branch: float = info(0.30)
    w_pot: float = info(0.15)
    w_succ: float = info(0.30)

    # Derived (filled in __post_init__).
    well_floor_top: float = field(default=None, init=False)
    ball_rest_floor: float = field(default=None, init=False)  # ball center on the main floor
    ball_rest_well: float = field(default=None, init=False)   # ball center on a well floor
    cradle_z: float = field(default=None, init=False)         # cradle center height (world)
    escape_deg: float = field(default=None, init=False)       # tilt needed to leave a well

    def __post_init__(self) -> None:
        self.well_floor_top = self.floor_top - self.well_depth
        self.ball_rest_floor = self.floor_top + self.ball_r
        self.ball_rest_well = self.well_floor_top + self.ball_r
        self.cradle_z = self.pivot_z - self.cradle_drop
        # trap statics: ball center sits d below the lip; escape needs atan(d / sqrt(r^2-d^2))
        d = self.well_depth - self.ball_r
        assert 0.0 < d < self.ball_r, "well depth must leave the ball center below the lip"
        self.escape_deg = math.degrees(math.atan2(d, math.sqrt(self.ball_r**2 - d**2)))
        assert self.escape_deg > self.limit_deg + 8.0, \
            f"decoy trap escapable: escape {self.escape_deg:.1f} deg vs stops {self.limit_deg} deg"
        # the rubric's in-well band actually separates well from corridor floor
        assert self.ball_rest_well < self.pot_z - 0.004 < self.ball_rest_floor - 0.010, \
            "pot_z band does not separate well from corridor"
        # ball fits under the roof and through every corridor
        slat_bot = self.slat_z - self.slat_th / 2
        assert self.ball_rest_floor + self.ball_r + 0.004 < slat_bot, "ball scrapes the roof"
        assert 2 * self.chan_hw - 2 * self.ball_r >= 0.030, "entry channel too tight"
        assert (self.cross_y1 - self.cross_y0) - 2 * self.ball_r >= 0.025, "cross corridor tight"
        assert (self.well_x1 - self.well_x0) - 2 * self.ball_r >= 0.020, "well mouth too tight"
        # the grate defeats both the ball and a fingertip; edge sliver beside the wall too
        gap = self.slat_pitch - self.slat_w
        assert gap < self.finger_t and gap < 2 * self.ball_r, "grate gap admits finger or ball"
        last_edge = ((self.slat_n - 1) / 2) * self.slat_pitch + self.slat_w / 2
        assert self.well_x1 - last_edge < self.finger_t, "edge sliver beside the wall admits finger"
        # spring re-levels within the success band with the ball parked anywhere
        worst = self.ball_mass * 9.81 * self.well_x1 / self.kappa  # rad, small-angle
        assert math.degrees(worst) < self.level_tol_deg - 1.0, "spring cannot re-level to tolerance"
        # the drive can pin the gimbal at its stops (probes must be non-vacuous)
        assert self.drive_max > 2.0 * self.kappa * math.sin(math.radians(self.limit_deg)), \
            "drive_max cannot reach the stops"
        # branch saturation lives inside the goal well
        assert self.well_x0 < self.branch_sat < self.well_x1 - self.ball_r, "branch_sat misplaced"
        # continuity gate: even at the stops the ball cannot cover cont_max in one substep
        v_max = math.sqrt(2 * (5.0 / 7.0) * 9.81 *
                          math.sin(math.radians(self.limit_deg + 6.0)) * 2 * self.well_x1)
        assert 2.0 * v_max / 120.0 < self.cont_max, "continuity gate would reject honest rolling"
        # the tabs are inside the documented arm envelope
        bx, by = self.base_pos
        tab_out = self.well_x1 + self.wall_t + 2 * self.tab_half
        far = math.hypot(self.post_pos[0] + tab_out - bx, abs(self.post_pos[1]) + by)
        assert far < self.reach, f"farthest tab {far:.3f} m outside reach {self.reach} m"
        # weights sum to exactly 1
        s = self.w_depart + self.w_cross + self.w_branch + self.w_pot + self.w_succ
        assert abs(s - 1.0) < 1e-9, "rubric weights must sum to 1"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("tilt_maze")
class TiltMazeScene(BaseScene):
    cfg: TiltMazeSceneCfg

    def __init__(self, cfg: TiltMazeSceneCfg | None = None) -> None:
        super().__init__(cfg or TiltMazeSceneCfg())

    # ----- assets --------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic pedestal, the cradle link, the labyrinth tray
        (gimbal joints authored in bind()), the orange ball, and the kinematic green
        beacon (re-posed per reset to mark the goal well)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        live = dict(sleep_threshold=0.0, stabilization_threshold=0.0)
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        px, py = c.post_pos

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "pedestal": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pedestal",
                spawn=sp["pedestal"](base_h=c.base_h, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0)),
            ),
            "cradle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cradle",
                spawn=sim_utils.CuboidCfg(
                    size=c.cradle_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cradle_mass),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.25, 0.25, 0.28)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, c.cradle_z)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=sp["tray"](
                    floor_top=c.floor_top, floor_th=c.floor_th, well_depth=c.well_depth,
                    chan_hw=c.chan_hw, cross_y0=c.cross_y0, cross_y1=c.cross_y1,
                    well_x0=c.well_x0, well_x1=c.well_x1,
                    inner_n_y=c.bay_y_hi, inner_s_y=c.cross_y0,
                    wall_t=c.wall_t, wall_top=c.wall_top, wall_bot=c.wall_bot,
                    slat_n=c.slat_n, slat_w=c.slat_w, slat_pitch=c.slat_pitch,
                    slat_th=c.slat_th, slat_z=c.slat_z, slat_y_hi=c.slat_y_hi,
                    tab_half=c.tab_half, tab_th=c.tab_th, tab_z=c.tab_z,
                    mass=c.tray_mass, inertia=c.tray_inertia,
                    mu_static=c.tray_mu[0], mu_dynamic=c.tray_mu[1],
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, c.pivot_z)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.02, angular_damping=0.05,
                        **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.ball_mu[0], dynamic_friction=c.ball_mu[1]),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.95, 0.45, 0.05)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px, py + 0.07, c.pivot_z + c.ball_rest_floor + 0.002)),
            ),
            "beacon": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beacon",
                spawn=sim_utils.CylinderCfg(
                    radius=c.beacon_r, height=c.beacon_h, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.10, 0.70, 0.15)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px + c.beacon_dx, py + c.beacon_y, c.beacon_h / 2)),
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

    # ----- lifecycle -------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        self._quat_apply = quat_apply
        self._quat_apply_inv = quat_apply_inverse
        n = env.num_envs
        dev = env.device
        self.pedestal: RigidObject = env.iscene["pedestal"]
        self.cradle: RigidObject = env.iscene["cradle"]
        self.tray: RigidObject = env.iscene["tray"]
        self.ball: RigidObject = env.iscene["ball"]
        self.beacon: RigidObject = env.iscene["beacon"]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        # External drive input (solve/smoke write; post_step consumes + owns the wrench
        # slot — never call set_external_force_and_torque on the tray directly).
        self.tilt_drive = torch.zeros(n, 2, device=dev)  # world (tau_x, tau_y) on the tray
        # Rubric state.
        self.side = torch.ones(n, device=dev)  # +1 = east well is the goal, -1 = west
        self.departed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.crossed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.branch = torch.zeros(n, device=dev)
        self.potted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._p_prev = torch.zeros(n, 3, device=dev)  # ball world pos last substep
        self._prev_valid = torch.zeros(n, dtype=torch.bool, device=dev)

    def _author_joints(self) -> None:
        """Per env: pedestal --revolute X--> cradle --revolute Y--> tray; both anchored
        at the pivot, both limited to +/- limit_deg, joint pairs never collide. Bodies
        spawn at their authored poses (reset re-poses cradle + tray to exactly these,
        about the never-moved kinematic pedestal — the safe-teleport pattern)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        lim = float(c.limit_deg)
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/gimbal_x")
            j.CreateBody0Rel().SetTargets([f"{base}/Pedestal"])
            j.CreateBody1Rel().SetTargets([f"{base}/Cradle"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.pivot_z))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, c.cradle_drop))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-lim)
            j.CreateUpperLimitAttr(lim)
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/gimbal_y")
            j.CreateBody0Rel().SetTargets([f"{base}/Cradle"])
            j.CreateBody1Rel().SetTargets([f"{base}/Tray"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.cradle_drop))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-lim)
            j.CreateUpperLimitAttr(lim)

    # ----- reset -----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: re-pose cradle + tray level at their authored poses about the
        fixed pedestal, sample the goal SIDE and move the beacon there, sample the
        ball's start-bay spawn, zero every latch and the drive buffer."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        px, py = c.post_pos

        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = px, py, c.cradle_z
        st[:, 0:3] += origin
        st[:, 3] = 1.0
        self.cradle.write_root_state_to_sim(st, env_ids)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = px, py, c.pivot_z
        st[:, 0:3] += origin
        st[:, 3] = 1.0
        self.tray.write_root_state_to_sim(st, env_ids)

        # goal side + beacon
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           -torch.ones(m, device=dev), torch.ones(m, device=dev))
        self.side[env_ids] = side
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = px + side * c.beacon_dx
        st[:, 1] = py + c.beacon_y
        st[:, 2] = c.beacon_h / 2
        st[:, 0:3] += origin
        st[:, 3] = 1.0
        self.beacon.write_root_state_to_sim(st, env_ids)

        # ball in the open start bay (tray is level, local == world offsets)
        bx = (torch.rand(m, device=dev) * 2 - 1) * c.spawn_x
        by = c.spawn_y_lo + torch.rand(m, device=dev) * (c.spawn_y_hi - c.spawn_y_lo)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = px + bx
        st[:, 1] = py + by
        st[:, 2] = c.pivot_z + c.ball_rest_floor + 0.002
        st[:, 0:3] += origin
        st[:, 3] = 1.0
        self.ball.write_root_state_to_sim(st, env_ids)

        # rubric + drive state
        self.departed[env_ids] = False
        self.crossed[env_ids] = False
        self.branch[env_ids] = 0.0
        self.potted[env_ids] = False
        self._p_prev[env_ids] = 0.0
        self._prev_valid[env_ids] = False
        self.tilt_drive[env_ids] = 0.0

    # ----- readings --------------------------------------------------------------------------------
    def ball_local(self) -> torch.Tensor:
        """(N, 3) ball center in the TRAY frame (origin at the pivot)."""
        rel = self.ball.data.root_pos_w - self.tray.data.root_pos_w
        return self._quat_apply_inv(self.tray.data.root_quat_w, rel)

    def tray_up(self) -> torch.Tensor:
        """(N, 3) the tray's local +z axis in world coordinates."""
        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        return self._quat_apply(self.tray.data.root_quat_w, ez)

    def tray_tilt(self) -> torch.Tensor:
        """(N,) tray tilt from level (rad)."""
        return torch.arccos(self.tray_up()[:, 2].clamp(-1.0, 1.0))

    def _in_cross(self, p: torch.Tensor) -> torch.Tensor:
        """(N,) bool: ball (tray-frame p) inside the cross-corridor row (any depth)."""
        c = self.cfg
        return ((p[:, 1] > c.cross_y0 - 0.002) & (p[:, 1] < c.cross_y1 + 0.002)
                & (p[:, 0].abs() < c.well_x1 + 0.002) & (p[:, 2] < 0.010))

    def _in_well(self, p: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        """(N,) bool: ball (tray-frame p) inside the `side` well, below the lip band."""
        c = self.cfg
        sx = side * p[:, 0]
        return (self._in_cross(p) & (sx > c.well_x0 - 0.002) & (sx < c.well_x1 + 0.002)
                & (p[:, 2] < c.pot_z))

    def in_goal_well(self) -> torch.Tensor:
        """(N,) bool: ball currently inside the beacon-marked well (physical, live)."""
        return self._in_well(self.ball_local(), self.side)

    def in_decoy_well(self) -> torch.Tensor:
        """(N,) bool: ball currently inside the WRONG well."""
        return self._in_well(self.ball_local(), -self.side)

    # ----- rubric ----------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the ball was steered through the maze into the goal well (latched
        path) and now rests there with the tray released level and quiet."""
        c = self.cfg
        v = self.ball.data.root_lin_vel_w.norm(dim=-1)
        w = self.tray.data.root_ang_vel_w.norm(dim=-1)
        level = self.tray_tilt() < math.radians(c.level_tol_deg)
        return (self.potted & self.in_goal_well()
                & (v < c.settle_v) & (w < c.settle_w) & level)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: latched traversal credit + live success. Exactly 1.0
        iff success(); ~0 for the null policy; wrong-well episodes cap at 0.25;
        disturbing the tray after success falls back to 0.70 (credit never evaporates)."""
        c = self.cfg
        return (c.w_depart * self.departed.float() + c.w_cross * self.crossed.float()
                + c.w_branch * self.branch + c.w_pot * self.potted.float()
                + c.w_succ * self.success().float())

    # ----- step-coupled mechanics (every substep) ---------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Gimbal plant (centering spring + damping + external drive buffer, applied as
        one wrench on the tray), then the continuity-gated traversal latches."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device

        # --- plant: tau_world = kappa * (u x ez) - damp * omega + clamped drive ---
        q = self.tray.data.root_quat_w
        u = self.tray_up()
        om = self.tray.data.root_ang_vel_w
        tau = torch.zeros(n, 3, device=dev)
        tau[:, 0] = c.kappa * u[:, 1] - c.tilt_damp * om[:, 0]
        tau[:, 1] = -c.kappa * u[:, 0] - c.tilt_damp * om[:, 1]
        tau[:, 2] = -c.tilt_damp * om[:, 2]
        tau[:, 0:2] += self.tilt_drive.clamp(-c.drive_max, c.drive_max)
        tau_body = self._quat_apply_inv(q, tau)  # wrench slot takes BODY-frame torque
        self.tray.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), tau_body.reshape(n, 1, 3))

        # --- traversal latches (continuity-gated; a garbage/teleport frame earns NOTHING) ---
        p_w = self.ball.data.root_pos_w
        jump = (p_w - self._p_prev).norm(dim=-1)
        cont = self._prev_valid & (jump < c.cont_max)
        p = torch.nan_to_num(self.ball_local(), nan=1e3, posinf=1e3, neginf=-1e3)

        entry = ((p[:, 0].abs() < c.chan_hw) & (p[:, 1] > c.cross_y1 + 0.002)
                 & (p[:, 1] < c.slat_y_hi - 0.005) & (p[:, 2] < 0.010))
        self.departed = self.departed | (cont & entry)
        in_cross = self._in_cross(p)
        self.crossed = self.crossed | (cont & self.departed & in_cross)
        prog = (self.side * p[:, 0] / c.branch_sat).clamp(0.0, 1.0)
        prog = torch.nan_to_num(prog, nan=0.0, posinf=0.0, neginf=0.0)
        self.branch = torch.maximum(
            self.branch, torch.where(cont & self.crossed & in_cross,
                                     prog, torch.zeros_like(prog)))
        in_goal = self._in_well(p, self.side)
        self.potted = self.potted | (
            cont & self.crossed & (self.branch > c.branch_gate) & in_goal)
        self.branch = torch.where(self.potted, torch.ones_like(self.branch), self.branch)

        self._p_prev = p_w.clone()
        self._prev_valid = torch.ones_like(self._prev_valid)

    # ----- state (full, restorable) -------------------------------------------------------------
    def _bodies(self) -> dict[str, Any]:
        return {"pedestal": self.pedestal, "cradle": self.cradle, "tray": self.tray,
                "ball": self.ball, "beacon": self.beacon}

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self._bodies().items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("side", "departed", "crossed", "branch", "potted",
                               "_p_prev", "_prev_valid", "tilt_drive")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A labyrinth tray hangs on a two-axis gimbal above a small fixed pedestal "
            f"about {c.post_pos[0]:.2f} m in front of the robot, pivot "
            f"{c.pivot_z * 100:.0f} cm above the floor. A spring re-centers the tray to "
            f"level whenever it is released; each tilt axis stops at "
            f"{c.limit_deg:.0f} degrees. Inside the tray an ORANGE ball "
            f"({2 * c.ball_r * 1000:.0f} mm) sits in an open starting bay. From the bay "
            f"a walled channel leads through a roofed section into a cross corridor "
            f"with a deep square well sunk at each end (dark blue floors). The roof is "
            f"a grate of {c.slat_n} dark slats with {1000 * (c.slat_pitch - c.slat_w):.0f} mm "
            f"gaps: neither the ball nor a fingertip fits through — past the bay the "
            f"ball CANNOT be touched, poked, or lifted. Four YELLOW tabs "
            f"({2 * c.tab_half * 100:.0f} cm square) stick out from the tray's outside "
            f"rim at the north, south, east and west; pressing down on a tab tilts the "
            f"maze about the gimbal, and gravity rolls the ball. A green beacon post "
            f"stands on the floor beside ONE end of the cross corridor; its side is "
            f"chosen per episode and marks the GOAL well.\n"
            f"Goal: tilt the maze by pressing the yellow tabs so the ball rolls out of "
            f"the bay, south through the roofed channel, then along the cross corridor "
            f"TOWARD THE BEACON, and drops into the goal well; then release the tabs "
            f"and let the spring re-level the tray with the ball resting in the well. "
            f"Mind the fork: the opposite well is a trap — its walls are too deep for "
            f"any allowed tilt to bring the ball back out. Touching only the yellow "
            f"tabs (or the tray's outer rim) is possible; the ball itself is sealed "
            f"under the grate once it leaves the bay."
        )

    def instruction(self) -> str:
        """SHORT imperative form for VLA training."""
        return (
            "Press the tray's yellow rim tabs to tilt the gimballed labyrinth so the "
            "orange ball rolls from the open bay through the grated corridor and drops "
            "into the well on the side marked by the green beacon, then release the "
            "tabs and let the tray spring back level with the ball in that well. The "
            "ball is sealed under the grate — steer it only by tilting; the opposite "
            "well is an inescapable trap."
        )


# ----- runnable env: scene physics only (NullRobot solve/smoke) -> "simgen.tilt_maze" ----------
register_env("simgen", lambda: EnvCfg(scene="tilt_maze", robot="null"))
