"""TiltLabyrinthScene — steer a ball through a serpentine maze by TILTING its tray
(sim_gen task `block_pyramid_i42`).

Derived from rlbench/block_pyramid, but STRATEGICALLY different: the seed is repeated
color-selective pick-and-place — grasp six green cubes one at a time among red
distractors and stack them into a 3-2-1 pyramid; every judged quantity is a block pose
and the plan is "pick, place, repeat". Here NOTHING is ever picked, placed, or stacked,
and the payload must never be touched at all: the only free object is a ball riding a
maze tray mounted on a two-axis GIMBAL, and the robot's only useful contact is pressing
the tray's rim tabs to TILT it. The task is indirect, closed-loop control of the
gravity vector: roll the ball along a walled serpentine corridor (east, then west, then
east), steer it AROUND a red-bordered trap hole mid-corridor, and drop it through the
green-bordered goal hole so it lands in the catch box below the tray. The plan
(continuous two-axis plate control with a hazard to skirt) and the code structure
(gimbal joints, tray-frame latching, basin containment) share nothing with the seed's
discrete pick-stack loop.

The apparatus (fully procedural, no external assets):
  - BASE (heavy dynamic compound, 40 kg — a joint anchored to a teleported kinematic
    body0 stays world-fixed at spawn on this stack): a floor slab, two journal posts,
    a perimeter GUARD skirt + rim COVER that seal the under-tray volume (nothing can be
    dropped or reached in from outside — the only way into a catch box is through its
    hole in the tray), and two open-top catch boxes on the slab: the GREEN goal box
    under the goal hole and the RED trap box under the trap hole.
  - FRAME: a slim ring on a revolute X-joint to the base (pitch axis).
  - TRAY: the maze plate on a revolute Y-joint to the frame (roll axis) — together a
    gimbal, both limited to +/- `tilt_limit_deg`. The tray is bottom-heavy (authored
    CoM 60 mm below the pivot axes), so it SELF-LEVELS: left alone it returns to level, and
    holding a tilt requires a sustained press on a rim TAB (four amber tabs at the side
    midpoints are the pressing affordances). Maze floor: a serpentine corridor formed
    by two interior walls with gaps at alternating ends; a 40 mm square TRAP hole
    (red border) in the middle corridor; a 70 mm square GOAL hole (green border) at
    the east end of the north corridor.

Judged on PHYSICAL outcomes only: success() = the ball settled INSIDE the green goal
box under the tray (base-frame containment + settle gate) AND the serpentine was
genuinely rolled — the three corridor-stage latches gate success, so a ball that
appears in the box without traversing the maze is refused. score() latches the corridor
stages the ball actually reaches while rolling ON the maze floor (middle corridor 0.20,
north corridor 0.50, goal approach 0.70), 1.0 iff success(); the trap hole yields no
credit and costs any not-yet-latched progress. Latches are monotonic, so credit never
evaporates while the ball is airborne in the goal hole.

Per-episode randomization (readback-verifiable): whole-apparatus yaw (free, +/-180 deg)
+ xy jitter, and the ball's start position sampled along the south corridor — world-
frame memorized tilt sequences fail; the solver must read the maze's orientation.

No discrete execution order is declared: the serpentine geometry itself enforces the
route (walls + the sealed underside are the only path to the goal box).

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _rot_xy(ang: torch.Tensor, xy: tuple) -> torch.Tensor:
    """(m, 2) world offset of a body-local xy point under yaw `ang`."""
    ca, sa = torch.cos(ang), torch.sin(ang)
    x, y = float(xy[0]), float(xy[1])
    return torch.stack([ca * x - sa * y, sa * x + ca * y], dim=-1)


# ----- custom compound spawners (base / frame / tray) --------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _root_xform(prim_path: str, translation, orientation):
    """Define an Xform root and author its (idempotent, single) translate/orient ops."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    return stage, xform.GetPrim()


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable):
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _rigid_dynamic(root, mass: float, *, lin_damp: float, ang_damp: float,
                   iters: int = 16, com: tuple | None = None) -> None:
    """Dynamic rigid-body armor on a compound root: MassAPI mass, damping, no sleeping
    while velocities are judged, depenetration cap. `com` authors an explicit local
    centre of mass — REQUIRED for the tray: with only a mass on the root, PhysX keeps
    the CoM at the body origin (the pivot), which kills the self-levelling pendulum
    (verified on the forge: the gimbal flopped to its joint limits at rest)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    mapi = UsdPhysics.MassAPI.Apply(root)
    mapi.CreateMassAttr(float(mass))
    if com is not None:
        mapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(int(iters))
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _bind_grip(stage_path: str, mat_path: str, static: float, dynamic: float) -> None:
    """Author (once) and bind a friction material (custom spawner colliders otherwise get
    the ~0.5 default with no restitution control — a bouncing ball never settles)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    import omni.usd
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(mat_path).IsValid():
        sim_utils.spawn_rigid_body_material(
            mat_path,
            sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic,
                                           restitution=0.0))
    bind_physics_material(stage_path, mat_path)


def _spawn_base(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the BASE: heavy DYNAMIC compound (local origin at ground level, footprint
    centre). Slab, two journal posts (visual explanation of the gimbal axis), guard
    skirt + rim cover (they seal the under-tray volume so the only way into a catch box
    is through its hole in the tray), and the two open-top catch boxes."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, c.base_mass, lin_damp=1.0, ang_damp=1.0)
    collide = _make_collide(c.contact_offset)
    dark = (0.24, 0.24, 0.27)
    _add_box(stage, f"{prim_path}/slab", center=(0.0, 0.0, 0.02),
             size=(0.70, 0.70, 0.04), color=dark, collide=collide)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/post_{'p' if sgn > 0 else 'n'}",
                 center=(sgn * 0.31, 0.0, 0.145), size=(0.05, 0.05, 0.21),
                 color=dark, collide=collide)
    # guard skirt: 4 plates around the tray footprint, slab top -> 0.108
    gz, gh = (0.04 + 0.108) / 2, 0.108 - 0.04
    for i, (cx, cy, sx, sy) in enumerate((
            (0.0, 0.265, 0.546, 0.008), (0.0, -0.265, 0.546, 0.008),
            (0.265, 0.0, 0.008, 0.546), (-0.265, 0.0, 0.008, 0.546))):
        _add_box(stage, f"{prim_path}/guard_{i}", center=(cx, cy, gz),
                 size=(sx, sy, gh), color=dark, collide=collide)
    # rim cover: horizontal annulus (inner 0.215, outer 0.269) at z 0.096..0.108 —
    # blocks anything dropped between the tray edge and the guard
    for i, (cx, cy, sx, sy) in enumerate((
            (0.0, 0.242, 0.538, 0.054), (0.0, -0.242, 0.538, 0.054),
            (0.242, 0.0, 0.054, 0.430), (-0.242, 0.0, 0.054, 0.430))):
        _add_box(stage, f"{prim_path}/cover_{i}", center=(cx, cy, 0.102),
                 size=(sx, sy, 0.012), color=dark, collide=collide)
    # catch boxes: 4 walls each on the slab (floor = the slab), z 0.04..0.10
    bz, bh = (0.04 + 0.10) / 2, 0.10 - 0.04

    def box_walls(tag: str, cx0: float, cy0: float, inner: float, color) -> None:
        h, t = inner / 2, c.wall_t
        for j, (dx, dy, sx, sy) in enumerate((
                (0.0, h + t / 2, inner + 2 * t, t), (0.0, -h - t / 2, inner + 2 * t, t),
                (h + t / 2, 0.0, t, inner), (-h - t / 2, 0.0, t, inner))):
            _add_box(stage, f"{prim_path}/{tag}_{j}", center=(cx0 + dx, cy0 + dy, bz),
                     size=(sx, sy, bh), color=color, collide=collide)

    box_walls("goalbox", c.goal_box_c[0], c.goal_box_c[1], c.goal_box_inner, c.green)
    box_walls("trapbox", 0.0, 0.0, c.trap_box_inner, c.red)
    return root


def _spawn_frame(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the gimbal FRAME: a slim ring around the tray, origin ON the pivot axis,
    plus the revolute X-joint to the sibling base (authored at spawn; post-play joints
    are dead). The joint pair (frame, base) is collision-filtered by USD default."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, c.frame_mass, lin_damp=0.5, ang_damp=1.0)
    collide = _make_collide(c.contact_offset)
    col = (0.55, 0.57, 0.62)
    for i, (cx, cy, sx, sy) in enumerate((
            (0.0, 0.268, 0.552, 0.016), (0.0, -0.268, 0.552, 0.016),
            (0.268, 0.0, 0.016, 0.520), (-0.268, 0.0, 0.016, 0.520))):
        _add_box(stage, f"{prim_path}/bar_{i}", center=(cx, cy, 0.0),
                 size=(sx, sy, 0.020), color=col, collide=collide)

    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/pitch")
    j.CreateBody0Rel().SetTargets([f"{base}/GimbalBase"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("X")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.pivot_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(c.tilt_limit_deg))
    j.CreateUpperLimitAttr(float(c.tilt_limit_deg))
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the maze TRAY: origin ON the pivot axis, everything hung below it (the
    bottom-heavy pendulum that self-levels the gimbal). Floor slabs leave the two
    square holes open; colored border slabs are FLUSH (same thickness) so they read
    visually without creating bumps. Plus the revolute Y-joint to the sibling frame."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    # explicit CoM 60 mm below the pivot: the bottom-heavy pendulum that self-levels
    # (restoring stiffness ~ tray_mass * g * 0.060 ~ 1.5 N m/rad)
    _rigid_dynamic(root, c.tray_mass, lin_damp=0.5, ang_damp=1.5, iters=32,
                   com=(0.0, 0.0, -0.060))
    collide = _make_collide(c.contact_offset)
    fz = -0.050  # floor slab centre (top -0.045, bottom -0.055)
    ft = 0.010
    gray = (0.78, 0.78, 0.76)
    th, gh = c.trap_hole / 2, c.goal_hole / 2  # hole half-sizes
    tb, gb = th + c.border_w, c.border_w  # trap border half-extent; goal border width
    gx, gy = c.goal_hole_c  # goal hole centre (0.15, 0.15)

    def slab(tag: str, x0: float, x1: float, y0: float, y1: float, color) -> None:
        _add_box(stage, f"{prim_path}/{tag}",
                 center=((x0 + x1) / 2, (y0 + y1) / 2, fz),
                 size=(x1 - x0, y1 - y0, ft), color=color, collide=collide)

    e = 0.20  # floor half-extent
    # neutral floor (5 slabs) around the trap zone [-tb,tb]^2 and goal zone [0.09,e]^2
    gz0 = gx - gh - gb  # goal zone west/south edge (0.09)
    slab("f_s", -e, e, -e, -tb, gray)
    slab("f_wm", -e, -tb, -tb, tb, gray)
    slab("f_em", tb, e, -tb, tb, gray)
    slab("f_mid", -e, e, tb, gz0, gray)
    slab("f_nw", -e, gz0, gz0, e, gray)
    # trap border (red frame between hole and [-tb,tb]^2)
    slab("bt_n", -tb, tb, th, tb, c.red)
    slab("bt_s", -tb, tb, -tb, -th, c.red)
    slab("bt_e", th, tb, -th, th, c.red)
    slab("bt_w", -tb, -th, -th, th, c.red)
    # goal border (green frame between hole and the goal zone [gz0, e]^2)
    slab("bg_s", gz0, e, gz0, gy - gh, c.green)
    slab("bg_n", gz0, e, gy + gh, e, c.green)
    slab("bg_w", gz0, gx - gh, gy - gh, gy + gh, c.green)
    slab("bg_e", gx + gh, e, gy - gh, gy + gh, c.green)

    wall = (0.35, 0.40, 0.50)
    wz, wh = -0.0275, 0.035  # walls: z -0.045 .. -0.010
    t = c.wall_t

    def wbox(tag: str, cx0: float, cy0: float, sx: float, sy: float, color=wall) -> None:
        _add_box(stage, f"{prim_path}/{tag}", center=(cx0, cy0, wz),
                 size=(sx, sy, wh), color=color, collide=collide)

    wbox("w_n", 0.0, e + t / 2, 2 * e + 2 * t, t)
    wbox("w_s", 0.0, -e - t / 2, 2 * e + 2 * t, t)
    wbox("w_e", e + t / 2, 0.0, t, 2 * e)
    wbox("w_w", -e - t / 2, 0.0, t, 2 * e)
    # interior wall A (gap at EAST end), interior wall B (gap at WEST end)
    ga, gbx = c.gap_x, c.gap_x  # walls end at +/-0.13 -> 70 mm gaps
    wbox("w_a", (-e + ga) / 2, -c.mid_y, ga + e, t)
    wbox("w_b", (e - gbx) / 2 + 0.0, c.mid_y, e + gbx, t)
    # press tabs (amber) at the four side midpoints, flush with the wall tops
    amber = (0.92, 0.66, 0.12)
    for tag, cx0, cy0, sx, sy in (("tab_n", 0.0, 0.228, 0.050, 0.036),
                                  ("tab_s", 0.0, -0.228, 0.050, 0.036),
                                  ("tab_e", 0.228, 0.0, 0.036, 0.050),
                                  ("tab_w", -0.228, 0.0, 0.036, 0.050)):
        _add_box(stage, f"{prim_path}/{tag}", center=(cx0, cy0, -0.014),
                 size=(sx, sy, 0.008), color=amber, collide=collide)

    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/roll")
    j.CreateBody0Rel().SetTargets([f"{base}/GimbalFrame"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(c.tilt_limit_deg))
    j.CreateUpperLimitAttr(float(c.tilt_limit_deg))
    _bind_grip(prim_path, "/World/simgenLabyrinthMat", 0.60, 0.50)
    return root


def _spawner_classes(cfg: TiltLabyrinthSceneCfg) -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "base" not in _SPAWNER_CACHE:

        @configclass
        class LabBaseSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_base)
            base_mass: float = 40.0
            contact_offset: float = 0.002
            wall_t: float = 0.010
            goal_box_c: tuple = (0.13, 0.13)
            goal_box_inner: float = 0.16
            trap_box_inner: float = 0.16
            green: tuple = (0.10, 0.65, 0.15)
            red: tuple = (0.85, 0.10, 0.10)

        @configclass
        class LabFrameSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_frame)
            frame_mass: float = 0.6
            contact_offset: float = 0.002
            pivot_z: float = 0.24
            tilt_limit_deg: float = 7.0

        @configclass
        class LabTraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            tray_mass: float = 2.5
            contact_offset: float = 0.002
            wall_t: float = 0.010
            tilt_limit_deg: float = 7.0
            trap_hole: float = 0.040
            goal_hole: float = 0.070
            goal_hole_c: tuple = (0.15, 0.15)
            border_w: float = 0.025
            mid_y: float = 0.075
            gap_x: float = 0.13
            green: tuple = (0.10, 0.65, 0.15)
            red: tuple = (0.85, 0.10, 0.10)

        _SPAWNER_CACHE.update(base=LabBaseSpawnerCfg, frame=LabFrameSpawnerCfg,
                              tray=LabTraySpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class TiltLabyrinthSceneCfg(BaseCfg):
    """Config for `TiltLabyrinthScene`. Corridor widths and hole sizes were chosen so a
    wall-hugging pass clears the trap by ~18 mm while a mindless straight run through
    the middle corridor's centre line falls in."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max ball |v| when judging success (m/s)
    stage_scores: tuple = tunable((0.20, 0.50, 0.70))  # latched credit: mid corridor,
    # north corridor, goal approach (success = 1.0)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_range_deg: float = tunable(180.0)  # whole-apparatus yaw, uniform +/- this
    pos_jitter: float = tunable(0.03)  # whole-apparatus xy jitter (m)
    ball_x_range: tuple = tunable((-0.16, 0.05))  # ball start x band (south corridor, tray frame)
    ball_y_range: tuple = tunable((-0.16, -0.11))  # ball start y band

    # --- tunable: mechanism ----------------------------------------------------------------------
    tilt_limit_deg: float = tunable(7.0)  # gimbal joint limits (both axes)
    ball_mass: float = tunable(0.020)  # kg (rubber ball)
    tray_mass: float = tunable(2.5)  # kg (bottom-heavy pendulum -> self-levelling)

    # --- info: fixed geometry (keep in sync with the spawners) -----------------------------------
    pivot_z: float = info(0.24)  # gimbal axis height (m); tray floor top = pivot_z - 0.045
    floor_top_local: float = info(-0.045)  # tray-frame z of the maze floor top
    floor_half: float = info(0.20)  # maze floor half-extent
    ball_r: float = info(0.015)
    trap_hole: float = info(0.040)  # square, centred at tray (0, 0)
    goal_hole: float = info(0.070)  # square, centred at `goal_hole_c`
    goal_hole_c: tuple = info((0.15, 0.15))
    border_w: float = info(0.025)  # colored border width around each hole
    mid_y: float = info(0.075)  # interior wall centre lines at y = -/+ mid_y
    gap_x: float = info(0.13)  # interior walls end at +/-gap_x (70 mm gaps)
    wall_t: float = info(0.010)
    goal_box_c: tuple = info((0.13, 0.13))  # goal catch box centre (base frame)
    goal_box_inner: float = info(0.16)
    trap_box_inner: float = info(0.16)
    box_wall_top: float = info(0.10)  # catch box wall top (base frame z)
    contact_offset: float = info(0.002)
    apparatus_pos: tuple = info((0.0, 0.0))  # nominal footprint centre


# ----- scene --------------------------------------------------------------------------------------
@SCENES.register("tilt_labyrinth")
class TiltLabyrinthScene(BaseScene):
    cfg: TiltLabyrinthSceneCfg

    def __init__(self, cfg: TiltLabyrinthSceneCfg | None = None) -> None:
        super().__init__(cfg or TiltLabyrinthSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes(c)
        ax, ay = c.apparatus_pos
        # spawn order matters: each joint's body0 must already exist
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
            "gimbal_base": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/GimbalBase",
                spawn=sp["base"](wall_t=c.wall_t, goal_box_c=c.goal_box_c,
                                 goal_box_inner=c.goal_box_inner,
                                 trap_box_inner=c.trap_box_inner,
                                 contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(ax, ay, 0.0)),
            ),
            "gimbal_frame": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/GimbalFrame",
                spawn=sp["frame"](pivot_z=c.pivot_z, tilt_limit_deg=c.tilt_limit_deg,
                                  contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(ax, ay, c.pivot_z)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/MazeTray",
                spawn=sp["tray"](tray_mass=c.tray_mass, tilt_limit_deg=c.tilt_limit_deg,
                                 trap_hole=c.trap_hole, goal_hole=c.goal_hole,
                                 goal_hole_c=c.goal_hole_c, border_w=c.border_w,
                                 mid_y=c.mid_y, gap_x=c.gap_x, wall_t=c.wall_t,
                                 contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(ax, ay, c.pivot_z)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.03, angular_damping=0.03,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.70, dynamic_friction=0.60, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.45, 0.05)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(ax - 0.10, ay - 0.14, c.pivot_z + c.floor_top_local + c.ball_r + 0.003)),
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

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.base: RigidObject = env.iscene["gimbal_base"]
        self.frame: RigidObject = env.iscene["gimbal_frame"]
        self.tray: RigidObject = env.iscene["tray"]
        self.ball: RigidObject = env.iscene["ball"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # monotonic stage latches (rubric credit that never evaporates)
        self.lat_mid = torch.zeros(n, dtype=torch.bool, device=dev)
        self.lat_north = torch.zeros(n, dtype=torch.bool, device=dev)
        self.lat_near = torch.zeros(n, dtype=torch.bool, device=dev)
        self.lat_trap = torch.zeros(n, dtype=torch.bool, device=dev)  # info only

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: one whole-apparatus pose (yaw free +/-yaw_range, xy jitter)
        written consistently to base/frame/tray, ball placed in the south corridor at a
        sampled x; all latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        axy = torch.tensor(c.apparatus_pos, device=dev).unsqueeze(0).expand(m, 2).clone()
        axy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.pos_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_range_deg)
        q = _qz(yaw)

        def write(body, xy: torch.Tensor, z: float, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3:7] = quat
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        write(self.base, axy, 0.0, q)
        write(self.frame, axy, c.pivot_z, q)
        write(self.tray, axy, c.pivot_z, q)

        bx = c.ball_x_range[0] + torch.rand(m, device=dev) * (c.ball_x_range[1] - c.ball_x_range[0])
        by = c.ball_y_range[0] + torch.rand(m, device=dev) * (c.ball_y_range[1] - c.ball_y_range[0])
        ca, sa = torch.cos(yaw), torch.sin(yaw)
        bxy = axy + torch.stack([ca * bx - sa * by, sa * bx + ca * by], dim=-1)
        write(self.ball, bxy, c.pivot_z + c.floor_top_local + c.ball_r + 0.003,
              _qz(torch.zeros(m, device=dev)))

        self.lat_mid[env_ids] = False
        self.lat_north[env_ids] = False
        self.lat_near[env_ids] = False
        self.lat_trap[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "base": self.base.data.root_state_w[env_ids].clone(),
            "frame": self.frame.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "latches": torch.stack([self.lat_mid[env_ids], self.lat_north[env_ids],
                                    self.lat_near[env_ids], self.lat_trap[env_ids]], dim=1).clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.base.write_root_state_to_sim(state["base"], env_ids)
        self.frame.write_root_state_to_sim(state["frame"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        lat = state["latches"]
        self.lat_mid[env_ids] = lat[:, 0]
        self.lat_north[env_ids] = lat[:, 1]
        self.lat_near[env_ids] = lat[:, 2]
        self.lat_trap[env_ids] = lat[:, 3]

    # ----- frames ----------------------------------------------------------------------------------
    def ball_tray_local(self) -> torch.Tensor:
        """(N, 3) ball centre in the TRAY body frame (origin at the pivot)."""
        from isaaclab.utils.math import quat_apply_inverse

        d = self.ball.data.root_pos_w - self.tray.data.root_pos_w
        return quat_apply_inverse(self.tray.data.root_quat_w, d)

    def ball_base_local(self) -> torch.Tensor:
        """(N, 3) ball centre in the BASE body frame (origin at ground level)."""
        from isaaclab.utils.math import quat_apply_inverse

        d = self.ball.data.root_pos_w - self.base.data.root_pos_w
        return quat_apply_inverse(self.base.data.root_quat_w, d)

    def tray_up_base(self) -> torch.Tensor:
        """(N, 3) the tray's up-axis expressed in the BASE frame — (u_x, u_y) is the
        downhill direction a rolling ball accelerates along; |u_xy| = sin(tilt)."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        up_w = quat_apply(self.tray.data.root_quat_w, ez)
        return quat_apply_inverse(self.base.data.root_quat_w, up_w)

    def _on_floor(self, p: torch.Tensor) -> torch.Tensor:
        """(N,) bool: ball rolling ON the maze floor — tray-frame height at rolling level
        and NOT over either hole opening (a ball falling through a hole crosses the floor
        plane inside the opening; that must not latch corridor credit)."""
        c = self.cfg
        z_roll = c.floor_top_local + c.ball_r
        on_z = (p[:, 2] - z_roll).abs() < 0.020
        in_bounds = (p[:, 0].abs() < c.floor_half + 0.005) & (p[:, 1].abs() < c.floor_half + 0.005)
        th, gh = c.trap_hole / 2 + 0.005, c.goal_hole / 2 + 0.005
        gx, gy = c.goal_hole_c
        over_trap = (p[:, 0].abs() < th) & (p[:, 1].abs() < th)
        over_goal = ((p[:, 0] - gx).abs() < gh) & ((p[:, 1] - gy).abs() < gh)
        return on_z & in_bounds & ~over_trap & ~over_goal

    # ----- containment ----------------------------------------------------------------------------
    def in_goal_box(self) -> torch.Tensor:
        """(N,) bool: ball centre inside the GREEN catch box volume (base frame)."""
        c = self.cfg
        p = self.ball_base_local()
        h = c.goal_box_inner / 2 - 0.002
        gx, gy = c.goal_box_c
        return ((p[:, 0] - gx).abs() < h) & ((p[:, 1] - gy).abs() < h) \
            & (p[:, 2] > 0.030) & (p[:, 2] < c.box_wall_top - 0.002)

    def in_trap_box(self) -> torch.Tensor:
        """(N,) bool: ball centre inside the RED trap box volume (base frame)."""
        c = self.cfg
        p = self.ball_base_local()
        h = c.trap_box_inner / 2 - 0.002
        return (p[:, 0].abs() < h) & (p[:, 1].abs() < h) \
            & (p[:, 2] > 0.030) & (p[:, 2] < c.box_wall_top - 0.002)

    def settled(self) -> torch.Tensor:
        """(N,) bool: ball |lin vel| below `settle_speed`."""
        return self.ball.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    # ----- step-coupled latching ------------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch the corridor stages the ball genuinely reaches while rolling on the
        floor (monotonic; runs every physics substep)."""
        c = self.cfg
        p = self.ball_tray_local()
        on = self._on_floor(p)
        mid = on & (p[:, 1] > -c.mid_y + 0.005) & (p[:, 1] < c.mid_y - 0.005)
        north = on & (p[:, 1] > c.mid_y + 0.005)
        near = north & (p[:, 0] > 0.06)
        self.lat_mid |= mid
        self.lat_north |= north
        self.lat_near |= near
        self.lat_trap |= self.in_trap_box()

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A square maze tray (40 cm across, light gray floor, blue-gray walls) is "
            "mounted on a two-axis gimbal above a dark pedestal; the whole apparatus may "
            "face any direction. The tray tilts up to "
            f"{c.tilt_limit_deg:.0f} degrees about each horizontal axis and springs back "
            "level when released (it is bottom-heavy); four small AMBER TABS at the "
            "midpoints of its rim are pressing points — pushing a tab down tilts the tray. "
            "An orange rubber ball (30 mm) rests in the maze. Two interior walls split the "
            "floor into a serpentine corridor of three legs with 70 mm turning gaps at "
            "alternating ends. In the MIDDLE leg there is a 40 mm square hole with a RED "
            "border (a trap); at the far end of the LAST leg there is a 70 mm square hole "
            "with a GREEN border (the goal). Each hole opens into a matching catch box "
            "under the tray; the underside is sealed by a skirt, so the only way into a "
            "box is through its hole.\n"
            "Goal: WITHOUT ever touching the ball, tilt the tray (press the amber rim "
            "tabs) so the ball rolls along the corridor from its start leg, through the "
            "first gap, along the middle leg PAST the red-bordered trap hole (hug the "
            "corridor wall to keep clear of it), through the second gap, and along the "
            "last leg until it drops through the GREEN-bordered hole and comes to rest in "
            "the green catch box below. If the ball falls through the red hole it lands "
            "in the trap box and the task is failed. Identify the maze's orientation "
            "visually: the green border marks the goal, the red border marks the trap, "
            "and the ball starts in the leg farthest from the goal hole."
        )

    def instruction(self) -> str:
        return (
            "Press the maze tray's amber rim tabs to tilt it, rolling the orange ball "
            "along the walled corridor: past the red-bordered trap hole without falling "
            "in, then through the green-bordered hole so it lands in the catch box "
            "below. Never touch the ball directly."
        )

    # ----- rubric ----------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: ball settled inside the green goal catch box (physical containment
        under the tray + settle gate) AND the corridor pathway was genuinely rolled
        (all three stage latches earned on the maze floor) — a ball that appears in the
        box without traversing the serpentine is not a success."""
        return (self.in_goal_box() & self.settled()
                & self.lat_mid & self.lat_north & self.lat_near)

    def score(self) -> torch.Tensor:
        """(N,) float: latched corridor progress (mid 0.20 / north 0.50 / goal approach
        0.70), 1.0 iff success(). Monotonic under correct behavior: the latches never
        clear, so credit survives the airborne drop through the goal hole; the trap
        earns nothing and forfeits any stage not yet latched."""
        c = self.cfg
        s = torch.zeros(self.env.num_envs, device=self.env.device)
        s = torch.where(self.lat_mid, torch.full_like(s, c.stage_scores[0]), s)
        s = torch.where(self.lat_north, torch.full_like(s, c.stage_scores[1]), s)
        s = torch.where(self.lat_near, torch.full_like(s, c.stage_scores[2]), s)
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("simgen", lambda: EnvCfg(scene="tilt_labyrinth", robot="null"))
