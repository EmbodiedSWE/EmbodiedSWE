"""MoveClockScene — make the promotion move, then punch the chess clock (legally).

Derived from rlbench/setup_chess ("set up the chess board"), but the PLAN is a
different game action with a RULE attached. The seed's plan is: 32 independent,
unordered pick-and-place moves onto marked squares — placement is the whole task
and order never matters. Here there is exactly ONE placement, and the task is only
complete when the move is PUNCHED IN on a mechanical chess clock, under the real
chess rule that you must move BEFORE you press your clock:

  1. PROMOTE — stand the ivory QUEEN upright inside the gold PROMOTION SQUARE (a
     low square frame lying on the board; its spot is randomized per episode).
  2. PUNCH THE CLOCK — press the RAISED button of the chess clock standing beside
     the board so its rocker beam tips fully over to the other side. The rocker is
     a real gravity-bistable mechanism (revolute joint, CoM above the pivot): it
     holds whichever side it was left on and must be flipped through contact.
  RULE (declared, chess-legal): pressing the clock while the queen is NOT yet
     standing in the promotion square is an ILLEGAL punch — a FOUL that latches
     permanently and forfeits the episode (success can never fire afterwards).

Strategic deltas vs the seed: a bistable MECHANISM the seed does not have (the
clock rocker — flipping it through its joint is a load-bearing interaction, not a
placement); a TEMPORAL-LEGALITY ordering rule judged by an event latch (the seed's
moves are unordered, and the end state alone cannot distinguish a legal from an
illegal punch — the rubric tracks the event history exactly as a chess arbiter
does); success requires a mechanism state (beam tipped) on top of the one
placement.

Assets are fully procedural (compound-spawner pattern):
  - board: KINEMATIC pedestal (0.40 x 0.40 x 0.10 m) with a visual 6x6
    checkerboard top (no tile colliders — the slab top is the standing surface).
    Pose (yaw + xy jitter) randomized per episode.
  - frame: KINEMATIC gold square outline (interior 90 x 90 mm, walls 6 mm thick,
    4 mm tall, NO floor of its own — the queen stands on the board through it),
    teleported to a randomized board spot each reset = the promotion square.
  - queen: DYNAMIC compound (flanged base, shaft, gold coronet ball),
    base-weighted with authored CoM + diagonal inertia; starts standing on the
    board on the opposite half from the frame.
  - clock: fixed world pose beside the board (never teleported — its rocker is
    jointed to it). ClockBase: KINEMATIC wood housing (base block + two cheek
    plates carrying the pivot). Rocker: DYNAMIC beam on a spawn-authored
    RevoluteJoint (axis x, limits +/-15 deg) with authored CoM 25 mm ABOVE the
    pivot -> gravity-bistable: it rests pressed against whichever stop it is on
    (an over-center toggle, like a real chess clock's rocker). Button caps: IVORY
    on the +y end, BLACK on the -y end. Which end starts RAISED is randomized.

Per-episode randomization (readback-verifiable): board yaw + xy jitter, the
promotion-square spot, the queen's start spot + yaw, and the rocker's start side.

Rubric (0..1; latched partial credit anchored in the demonstrated solve):
  0.45  l_seat — the queen has stood upright inside the promotion square (latched)
  0.30  l_flip — the rocker reached the far stop while l_seat was already latched
                 and no foul had fired (latched)
  FOUL  _foul  — the rocker left its start side before l_seat: latched forever,
                 blocks l_flip and success.
  1.0 iff success() — live: queen standing upright inside the promotion square AND
       rocker fully tipped to the far side AND no foul, everything settled, states
       finite. Non-success capped at 0.75.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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


# ----- custom compound spawners ---------------------------------------------------------------
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


def _friction_material(stage, path: str, mu_s: float, mu_d: float):
    """A physics material prim (custom-spawner colliders otherwise get ~0.5 friction)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu_s))
    api.CreateDynamicFrictionAttr(float(mu_d))
    api.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, contact_offset: float, material=None) -> None:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _add_box(stage, path: str, *, center, size, color, contact_offset=None,
             material=None, collide=True):
    """One box child: translate + scale, displayColor, optional collider.
    `collide=False` authors a purely visual prim (checker tiles)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        _collide(box.GetPrim(), contact_offset, material)
    return box.GetPrim()


def _add_cyl(stage, path: str, *, center, radius, height, color, contact_offset,
             material=None):
    """One z-axis cylinder child with authored extent, displayColor, collider."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr("Z")
    r, h = float(radius), float(height)
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(cyl.GetPrim(), contact_offset, material)
    return cyl.GetPrim()


def _add_ball(stage, path: str, *, center, radius, color, contact_offset, material=None):
    from pxr import Gf, UsdGeom

    ball = UsdGeom.Sphere.Define(stage, path)
    ball.CreateRadiusAttr(float(radius))
    r = float(radius)
    ball.CreateExtentAttr([Gf.Vec3f(-r, -r, -r), Gf.Vec3f(r, r, r)])
    xf = UsdGeom.Xformable(ball.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    ball.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(ball.GetPrim(), contact_offset, material)
    return ball.GetPrim()


def _spawn_board(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the board pedestal: KINEMATIC compound. Local frame: origin at the
    footprint centre ON THE GROUND, z up. Children: slab + a 6x6 VISUAL
    checkerboard (no colliders — the slab's flat top is the standing surface, so
    no proud collider edge exists anywhere on it)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(30.0)
    mat = _friction_material(stage, f"{prim_path}/mat", cfg.mu_s, cfg.mu_d)
    c = cfg
    top = c.board_h
    _add_box(stage, f"{prim_path}/slab",
             center=(0.0, 0.0, top / 2), size=(c.board_xy, c.board_xy, top),
             color=(0.24, 0.16, 0.10), contact_offset=c.contact_offset, material=mat)
    tile = (c.board_xy - 0.02) / 6
    for i in range(6):
        for j in range(6):
            _add_box(stage, f"{prim_path}/tile_{i}_{j}",
                     center=((i - 2.5) * tile, (j - 2.5) * tile, top + 0.0004),
                     size=(tile - 0.003, tile - 0.003, 0.0008),
                     color=(0.90, 0.86, 0.74) if (i + j) % 2 == 0 else (0.30, 0.22, 0.14),
                     collide=False)
    return root


def _spawn_frame(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the promotion square: KINEMATIC gold square OUTLINE (four low wall
    boxes; NO floor — the queen stands on the board top through the opening).
    Local frame: origin at the square centre AT BOARD-TOP HEIGHT."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(1.0)
    mat = _friction_material(stage, f"{prim_path}/mat", 0.30, 0.25)
    c = cfg
    ih, t, h = c.frame_inner_half, c.frame_t, c.frame_h
    gold = (0.85, 0.68, 0.15)
    for tag, cx, cy, sx, sy in (
        ("xp", ih + t / 2, 0.0, t, 2 * (ih + t)),
        ("xn", -(ih + t / 2), 0.0, t, 2 * (ih + t)),
        ("yp", 0.0, ih + t / 2, 2 * ih, t),
        ("yn", 0.0, -(ih + t / 2), 2 * ih, t),
    ):
        _add_box(stage, f"{prim_path}/wall_{tag}",
                 center=(cx, cy, h / 2), size=(sx, sy, h),
                 color=gold, contact_offset=c.contact_offset, material=mat)
    return root


def _spawn_queen(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the queen: DYNAMIC compound rigid body. Local frame: origin at the
    CENTRE of the base bottom face. Children: base cylinder, shaft cylinder,
    collar disc, gold coronet ball. Mass, CoM (base-weighted) and a diagonal
    inertia are authored EXPLICITLY (MassAPI mass alone leaves the CoM at the
    body origin)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, float(cfg.com_z)))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in cfg.inertia]))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.20)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)  # kills GPU phantom-creep artifacts
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    mat = _friction_material(stage, f"{prim_path}/mat", cfg.mu_s, cfg.mu_d)
    co = cfg.contact_offset
    ivory = (0.93, 0.91, 0.82)
    _add_cyl(stage, f"{prim_path}/base",
             center=(0.0, 0.0, cfg.base_h / 2), radius=cfg.base_r,
             height=cfg.base_h, color=ivory, contact_offset=co, material=mat)
    _add_cyl(stage, f"{prim_path}/shaft",
             center=(0.0, 0.0, cfg.base_h + cfg.shaft_h / 2), radius=cfg.shaft_r,
             height=cfg.shaft_h, color=ivory, contact_offset=co, material=mat)
    _add_cyl(stage, f"{prim_path}/collar",
             center=(0.0, 0.0, cfg.base_h + cfg.shaft_h + 0.004),
             radius=cfg.shaft_r + 0.005, height=0.008,
             color=ivory, contact_offset=co, material=mat)
    _add_ball(stage, f"{prim_path}/coronet",
              center=(0.0, 0.0, cfg.base_h + cfg.shaft_h + 0.008 + cfg.crown_r * 0.8),
              radius=cfg.crown_r, color=(0.85, 0.68, 0.15),
              contact_offset=co, material=mat)
    return root


def _spawn_clock_base(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the chess-clock housing: KINEMATIC compound. Local frame: origin at
    the footprint centre ON THE GROUND. Children: wood base block + two cheek
    plates (at +/-x) carrying the pivot at z = pivot_z. The rocker sweep
    (bar half-length x sin 15 deg) stays clear of the base top by >40 mm — the
    joint limits are the only stops (no static prop ever masks them)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(10.0)
    mat = _friction_material(stage, f"{prim_path}/mat", 0.40, 0.35)
    c = cfg
    wood = (0.38, 0.24, 0.12)
    _add_box(stage, f"{prim_path}/block",
             center=(0.0, 0.0, c.base_h / 2), size=(c.base_x, c.base_y, c.base_h),
             color=wood, contact_offset=c.contact_offset, material=mat)
    for tag, sx in (("p", 1.0), ("n", -1.0)):
        _add_box(stage, f"{prim_path}/cheek_{tag}",
                 center=(sx * (c.bar_x / 2 + 0.004 + c.cheek_t / 2), 0.0,
                         c.base_h + (c.pivot_z - c.base_h) / 2 + 0.010),
                 size=(c.cheek_t, 0.06, c.pivot_z - c.base_h + 0.020),
                 color=wood, contact_offset=c.contact_offset, material=mat)
    return root


def _spawn_rocker(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the clock rocker: DYNAMIC beam, body origin AT the pivot. Children:
    bar box along local y, IVORY button cap on the +y end top, BLACK cap on the
    -y end top, red centre boss. CoM is authored 25 mm ABOVE the pivot -> the
    beam is gravity-bistable at the +/-15 deg joint stops (over-center toggle).
    Spawn-authors the RevoluteJoint to the sibling ClockBase: axis X, limits
    +/-15 deg, joint-pair collision OFF."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, float(cfg.com_z)))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in cfg.inertia]))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.10)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    mat = _friction_material(stage, f"{prim_path}/mat", 0.40, 0.35)
    c = cfg
    co = c.contact_offset
    _add_box(stage, f"{prim_path}/bar",
             center=(0.0, 0.0, 0.0), size=(c.bar_x, c.bar_len, c.bar_h),
             color=(0.55, 0.10, 0.10), contact_offset=co, material=mat)
    for tag, sy, col in (("ivory", 1.0, (0.93, 0.91, 0.82)),
                         ("black", -1.0, (0.09, 0.09, 0.11))):
        _add_cyl(stage, f"{prim_path}/btn_{tag}",
                 center=(0.0, sy * c.btn_y, c.bar_h / 2 + c.btn_h / 2),
                 radius=c.btn_r, height=c.btn_h, color=col,
                 contact_offset=co, material=mat)
    _add_ball(stage, f"{prim_path}/boss",
              center=(0.0, 0.0, c.bar_h / 2 + 0.008), radius=0.012,
              color=(0.85, 0.68, 0.15), contact_offset=co, material=mat)

    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/pivot_joint")
    j.CreateBody0Rel().SetTargets([f"{base}/ClockBase"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("X")
    j.CreateCollisionEnabledAttr(False)
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.pivot_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    j.CreateLowerLimitAttr(-float(c.tilt_deg))
    j.CreateUpperLimitAttr(float(c.tilt_deg))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "board" not in _SPAWNER_CACHE:

        @configclass
        class BoardSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_board)
            board_xy: float = 0.40
            board_h: float = 0.10
            mu_s: float = 0.35
            mu_d: float = 0.30
            contact_offset: float = 0.002

        @configclass
        class FrameSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_frame)
            frame_inner_half: float = 0.045
            frame_t: float = 0.006
            frame_h: float = 0.004
            contact_offset: float = 0.002

        @configclass
        class QueenSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_queen)
            base_r: float = 0.022
            base_h: float = 0.014
            shaft_r: float = 0.012
            shaft_h: float = 0.075
            crown_r: float = 0.016
            mass: float = 0.15
            com_z: float = 0.012
            inertia: tuple = (1.6e-4, 1.6e-4, 4.0e-5)
            mu_s: float = 0.30
            mu_d: float = 0.25
            contact_offset: float = 0.002

        @configclass
        class ClockBaseSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_clock_base)
            base_x: float = 0.10
            base_y: float = 0.24
            base_h: float = 0.04
            cheek_t: float = 0.015
            bar_x: float = 0.03
            pivot_z: float = 0.12
            contact_offset: float = 0.002

        @configclass
        class RockerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rocker)
            bar_x: float = 0.03
            bar_len: float = 0.20
            bar_h: float = 0.025
            btn_y: float = 0.083
            btn_r: float = 0.018
            btn_h: float = 0.025
            pivot_z: float = 0.12
            tilt_deg: float = 15.0
            mass: float = 0.15
            com_z: float = 0.025
            inertia: tuple = (6.0e-4, 1.5e-4, 6.0e-4)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(board=BoardSpawnerCfg, frame=FrameSpawnerCfg,
                              queen=QueenSpawnerCfg, clock_base=ClockBaseSpawnerCfg,
                              rocker=RockerSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class MoveClockSceneCfg(BaseCfg):
    """Config for `MoveClockScene`. Tolerances are honest by construction: a queen
    physically standing inside the 90 mm frame interior has its base centre within
    45 - 22 = 23 mm of the frame centre < `seat_tol` (30 mm), so every real
    in-square stand counts, while a queen outside the frame is >= 51 mm off
    (wall outer face + base radius). The rocker's rest angles are the +/-15 deg
    joint stops; `flip_deg` (12) sits 3 deg inside them and `foul_deg` (8) is 7
    deg of margin away from the held stop, far beyond solver wobble."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    seat_tol: float = tunable(0.030)       # queen base centre within this of the frame centre (m)
    upright_max_deg: float = tunable(12.0)  # queen axis within this of world-up when seated
    settle_speed: float = tunable(0.04)    # max queen |lin vel| when judging (m/s)
    rocker_still: float = tunable(0.40)    # max rocker |ang vel| when judging (rad/s)
    flip_deg: float = tunable(12.0)        # rocker past this on the FAR side = tipped
    foul_deg: float = tunable(8.0)         # rocker below this on the START side pre-seat = foul

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    board_yaw_deg: float = tunable(20.0)   # board yaw about nominal (+/- deg)
    board_jitter: float = tunable(0.04)    # board xy jitter (+/- m)
    frame_x_range: tuple = tunable((-0.10, 0.10))  # promotion-square centre x (board frame)
    frame_y_lohi: tuple = tunable((0.03, 0.11))    # |y| range; the SIDE (+/-) is drawn too
    queen_x_range: tuple = tunable((-0.11, 0.11))  # queen start x (board frame)
    queen_y_lohi: tuple = tunable((0.05, 0.12))    # |y| range; always OPPOSITE the frame side

    # --- info: layout ---------------------------------------------------------------------------
    board_pos: tuple = info((0.0, 0.0))
    board_xy: float = info(0.40)
    board_h: float = info(0.10)            # slab top = playing surface height
    frame_inner_half: float = info(0.045)  # promotion-square interior half-width
    frame_t: float = info(0.006)
    frame_h: float = info(0.004)
    clock_pos: tuple = info((0.44, 0.0))   # FIXED world xy (jointed assembly, never teleported)
    pivot_z: float = info(0.12)            # rocker pivot height above the ground
    tilt_deg: float = info(15.0)           # rocker joint stops (+/- deg about local x)
    bar_len: float = info(0.20)
    btn_y: float = info(0.083)             # button centres along the bar (+y ivory, -y black)
    rocker_mass: float = info(0.15)
    rocker_com_z: float = info(0.025)      # CoM above pivot -> gravity-bistable
    # queen
    base_r: float = info(0.022)
    base_h: float = info(0.014)
    shaft_r: float = info(0.012)
    shaft_h: float = info(0.075)
    queen_mass: float = info(0.15)
    # rubric weights (0.45 + 0.30 = 0.75 = the non-success cap)
    w_seat: float = info(0.45)
    w_flip: float = info(0.30)
    contact_offset: float = info(0.002)


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qconj(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qx(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 1] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("move_clock")
class MoveClockScene(BaseScene):
    cfg: MoveClockSceneCfg

    def __init__(self, cfg: MoveClockSceneCfg | None = None) -> None:
        super().__init__(cfg or MoveClockSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        board_spawn = cls["board"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            board_xy=c.board_xy, board_h=c.board_h, contact_offset=c.contact_offset)
        frame_spawn = cls["frame"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            frame_inner_half=c.frame_inner_half, frame_t=c.frame_t, frame_h=c.frame_h,
            contact_offset=c.contact_offset)
        queen_spawn = cls["queen"](
            base_r=c.base_r, base_h=c.base_h, shaft_r=c.shaft_r, shaft_h=c.shaft_h,
            mass=c.queen_mass, contact_offset=c.contact_offset)
        clock_base_spawn = cls["clock_base"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            pivot_z=c.pivot_z, contact_offset=c.contact_offset)
        rocker_spawn = cls["rocker"](
            bar_len=c.bar_len, btn_y=c.btn_y, pivot_z=c.pivot_z, tilt_deg=c.tilt_deg,
            mass=c.rocker_mass, com_z=c.rocker_com_z, contact_offset=c.contact_offset)

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
            "board": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Board",
                spawn=board_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.board_pos[0], c.board_pos[1], 0.0)),
            ),
            "frame": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Frame",
                spawn=frame_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.board_h)),
            ),
            # ClockBase BEFORE Rocker: the rocker's spawn-authored joint targets it.
            "clock_base": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/ClockBase",
                spawn=clock_base_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.clock_pos[0], c.clock_pos[1], 0.0)),
            ),
            "rocker": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rocker",
                spawn=rocker_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.clock_pos[0], c.clock_pos[1], c.pivot_z)),
            ),
            "queen": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Queen",
                spawn=queen_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.6, -0.6, 0.02)),
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
        self.board: RigidObject = env.iscene["board"]
        self.frame: RigidObject = env.iscene["frame"]
        self.clock_base: RigidObject = env.iscene["clock_base"]
        self.rocker: RigidObject = env.iscene["rocker"]
        self.queen: RigidObject = env.iscene["queen"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.start_side = torch.ones(n, device=dev)   # +1: ivory (+y) end starts RAISED
        self.frame_xy = torch.zeros(n, 2, device=dev)  # promotion-square centre (board frame)
        # latches (partial credit survives transients; success is judged live)
        self._l_seat = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l_flip = torch.zeros(n, dtype=torch.bool, device=dev)
        self._foul = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the board (yaw + xy jitter), sample the promotion
        square's spot on one half and teleport the frame there, stand the queen on
        the OPPOSITE half, and cock the rocker onto a random side (rotating it
        about its own origin = the pivot, a pure in-DOF write; the jointed clock
        itself is NEVER teleported). Clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(m, 4, device=dev)  # burn draws (first post-seed draws degenerate)

        # --- board: kinematic, yaw + xy jitter ---
        psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.board_yaw_deg)
        q = _qz(psi)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.board_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.board_jitter
        st[:, 1] = c.board_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.board_jitter
        st[:, 3:7] = q
        bxy = st[:, 0:2].clone()
        st[:, 0:3] += origin
        self.board.write_root_state_to_sim(st, env_ids)

        from isaaclab.utils.math import quat_apply

        def to_world(loc: torch.Tensor) -> torch.Tensor:
            w = torch.zeros(m, 3, device=dev)
            w[:, 0:2] = bxy
            return w + quat_apply(q, loc) + origin

        # --- promotion square: sampled on one half; queen on the other half ---
        t = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        fx = c.frame_x_range[0] + torch.rand(m, device=dev) \
            * (c.frame_x_range[1] - c.frame_x_range[0])
        fy = t * (c.frame_y_lohi[0] + torch.rand(m, device=dev)
                  * (c.frame_y_lohi[1] - c.frame_y_lohi[0]))
        self.frame_xy[env_ids, 0] = fx
        self.frame_xy[env_ids, 1] = fy
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = fx
        loc[:, 1] = fy
        loc[:, 2] = c.board_h
        stf = torch.zeros(m, 13, device=dev)
        stf[:, 0:3] = to_world(loc)
        stf[:, 3:7] = q
        self.frame.write_root_state_to_sim(stf, env_ids)

        qx_ = c.queen_x_range[0] + torch.rand(m, device=dev) \
            * (c.queen_x_range[1] - c.queen_x_range[0])
        qy_ = -t * (c.queen_y_lohi[0] + torch.rand(m, device=dev)
                    * (c.queen_y_lohi[1] - c.queen_y_lohi[0]))
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = qx_
        loc[:, 1] = qy_
        loc[:, 2] = c.board_h + 0.0015
        stq = torch.zeros(m, 13, device=dev)
        stq[:, 0:3] = to_world(loc)
        stq[:, 3:7] = _qmul(q, _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi))
        self.queen.write_root_state_to_sim(stq, env_ids)

        # --- rocker: cock onto a random side (in-DOF rotation about the pivot) ---
        s0 = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        self.start_side[env_ids] = s0
        # write 1 deg INSIDE the stop; gravity presses it onto the stop (over-center)
        ang = s0 * math.radians(c.tilt_deg - 1.0)
        str_ = torch.zeros(m, 13, device=dev)
        str_[:, 0] = c.clock_pos[0]
        str_[:, 1] = c.clock_pos[1]
        str_[:, 2] = c.pivot_z
        str_[:, 0:3] += origin
        str_[:, 3:7] = _qx(ang)
        self.rocker.write_root_state_to_sim(str_, env_ids)

        # --- clear latches ---
        self._l_seat[env_ids] = False
        self._l_flip[env_ids] = False
        self._foul[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "board": self.board.data.root_state_w[env_ids].clone(),
            "frame": self.frame.data.root_state_w[env_ids].clone(),
            "clock_base": self.clock_base.data.root_state_w[env_ids].clone(),
            "rocker": self.rocker.data.root_state_w[env_ids].clone(),
            "queen": self.queen.data.root_state_w[env_ids].clone(),
            "start_side": self.start_side[env_ids].clone(),
            "frame_xy": self.frame_xy[env_ids].clone(),
            "l_seat": self._l_seat[env_ids].clone(),
            "l_flip": self._l_flip[env_ids].clone(),
            "foul": self._foul[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.board.write_root_state_to_sim(state["board"], env_ids)
        self.frame.write_root_state_to_sim(state["frame"], env_ids)
        self.rocker.write_root_state_to_sim(state["rocker"], env_ids)
        self.queen.write_root_state_to_sim(state["queen"], env_ids)
        self.start_side[env_ids] = state["start_side"]
        self.frame_xy[env_ids] = state["frame_xy"]
        self._l_seat[env_ids] = state["l_seat"]
        self._l_flip[env_ids] = state["l_flip"]
        self._foul[env_ids] = state["foul"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A dark-wood chess PEDESTAL BOARD ({c.board_xy * 100:.0f} x "
            f"{c.board_xy * 100:.0f} cm, {c.board_h * 100:.0f} cm tall) stands on the "
            f"floor, its top painted as a pale-and-brown checkerboard. Lying flat on "
            f"the board is a GOLD SQUARE FRAME (a thin square outline, interior "
            f"{2 * c.frame_inner_half * 100:.0f} x {2 * c.frame_inner_half * 100:.0f} cm, "
            f"walls only {c.frame_h * 1000:.0f} mm tall) — the PROMOTION SQUARE. The "
            f"ivory CHESS QUEEN (cylindrical piece, {c.base_r * 2000:.0f} mm base, a "
            f"gold ball coronet on top) starts standing on the other half of the "
            f"board. Beside the board stands a wooden CHESS CLOCK: a red rocker BEAM "
            f"on a pivot, with a round IVORY BUTTON on one end and a round BLACK "
            f"BUTTON on the other. The beam is tipped to one side: one button sits "
            f"RAISED, the other pressed down. The beam is a mechanical toggle — it "
            f"stays on whichever side it was left, and pressing the raised button "
            f"down tips it fully over to the other side. The board's position and "
            f"heading, the promotion square's spot, the queen's spot and which clock "
            f"button starts raised all change every episode.\n"
            f"Goal — make the move, then punch the clock, IN THAT ORDER: (1) MOVE: "
            f"stand the queen upright (within ~{c.upright_max_deg:.0f} deg) with its "
            f"base inside the gold promotion square (base centre within "
            f"~{c.seat_tol * 100:.0f} cm of the square's centre), resting on the "
            f"board. (2) PUNCH: press the RAISED clock button down so the beam tips "
            f"fully onto the other side and stays there. RULE: pressing the clock "
            f"(tipping the beam off its starting side) while the queen is NOT yet "
            f"standing in the promotion square is an ILLEGAL punch and forfeits the "
            f"episode permanently — exactly as in chess, the move must be completed "
            f"before the clock is pressed. Success: queen standing in the gold "
            f"square AND beam fully tipped to the far side, no foul, everything at "
            f"rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Stand the ivory queen upright inside the gold square frame on the "
            "board, then press the raised button of the chess clock beside the "
            "board so its beam tips fully to the other side. Pressing the clock "
            "before the queen stands in the gold square is an illegal punch and "
            "fails the task."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def board_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points (N,3) -> the board's local frame (origin at footprint
        centre on the ground)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.board.data.root_quat_w,
                                  pos_w - self.board.data.root_pos_w)

    def board_to_world(self, loc: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        return self.board.data.root_pos_w + quat_apply(self.board.data.root_quat_w, loc)

    def rocker_angle(self) -> torch.Tensor:
        """(N,) rad: rocker tilt about its hinge (+ = ivory/+y end raised)."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ey = torch.tensor([0.0, 1.0, 0.0], device=self.env.device).expand(n, 3)
        yw = quat_apply(self.rocker.data.root_quat_w, ey)
        return torch.atan2(yw[:, 2], yw[:, 1])

    def upright(self, body) -> torch.Tensor:
        """(N,) bool: body axis within `upright_max_deg` of world-up."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(body.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.upright_max_deg))

    def seated(self) -> torch.Tensor:
        """(N,) bool, geometric: queen base centre within `seat_tol` of the
        promotion square's centre, RESTING on the board top (z window rejects
        hovering / dropped-through / lying poses), upright."""
        c = self.cfg
        loc = self.board_local(self.queen.data.root_pos_w)
        near = (loc[:, :2] - self.frame_xy).norm(dim=-1) < c.seat_tol
        rest = (loc[:, 2] > c.board_h - 0.005) & (loc[:, 2] < c.board_h + 0.010)
        return near & rest & self.upright(self.queen)

    def flipped(self) -> torch.Tensor:
        """(N,) bool: rocker fully tipped onto the FAR side (past `flip_deg`)."""
        return self.start_side * self.rocker_angle() \
            <= -math.radians(self.cfg.flip_deg)

    def on_start_side(self) -> torch.Tensor:
        """(N,) bool: rocker still held on its starting stop."""
        return self.start_side * self.rocker_angle() \
            >= math.radians(self.cfg.flip_deg)

    def settled(self) -> torch.Tensor:
        """(N,) bool: queen slow AND rocker rotationally still."""
        c = self.cfg
        qv = self.queen.data.root_lin_vel_w.norm(dim=-1)
        rw = self.rocker.data.root_ang_vel_w.norm(dim=-1)
        return (qv < c.settle_speed) & (rw < c.rocker_still)

    def _update_latches(self) -> None:
        c = self.cfg
        # foul FIRST: the beam leaving its start side before the seat latch is an
        # illegal punch (7 deg of margin from the held stop; solver wobble << that)
        off_start = self.start_side * self.rocker_angle() < math.radians(c.foul_deg)
        self._foul |= off_start & ~self._l_seat
        slow = self.queen.data.root_lin_vel_w.norm(dim=-1) < 0.25
        self._l_seat |= self.seated() & slow
        self._l_flip |= self.flipped() & self._l_seat & ~self._foul

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool, live: queen standing upright inside the promotion square AND
        rocker fully tipped to the far side, LEGALLY (no foul on record), both
        settled, states finite. The pose clauses are physical outcomes; the foul
        clause is the declared chess-legality rule (move before clock), judged
        from the event history exactly like an arbiter would."""
        self._update_latches()
        finite = torch.isfinite(self.queen.data.root_state_w).all(dim=-1) \
            & torch.isfinite(self.rocker.data.root_state_w).all(dim=-1)
        return self.seated() & self.flipped() & ~self._foul & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.45*seated-latch + 0.30*legal-flip-latch (~0 for
        doing nothing; a fouled episode can never earn the flip credit), capped at
        0.75 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_seat * self._l_seat.float()
                + c.w_flip * self._l_flip.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="move_clock", robot="null"))
