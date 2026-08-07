"""RockerCabinetScene — open the sealed bottom drawer THROUGH the cabinet's own
rocker transmission (sim_gen task `libero_kitchen_scene1_open_bottom_drawer_i87`).

Derived from libero_90/libero_kitchen_scene1_open_bottom_drawer, but STRATEGICALLY
different: the seed's whole skill is grasping the bottom drawer's handle and pulling
it open — the hand actuates the judged part directly, outward, by a guided pull.
Here the judged part CANNOT be actuated directly at all: the bottom drawer's front
is a smooth plate RECESSED 4 mm behind the cabinet face with only a 2 mm perimeter
gap — nothing to hook, pinch, or press usefully (it already sits on its inner hard
stop, so pushing it does nothing either). The only way to open it is the cabinet's
internal ROCKER TRANSMISSION: a plank pivoted on a horizontal axle inside a sealed
rear chamber, its upper arm facing the TOP drawer's rear pusher plate and its lower
arm facing the BOTTOM drawer's. The top drawer starts pulled OUT, its oversized
front lip protruding from the cabinet. PUSHING THE TOP DRAWER FLUSH (an inward
push on the lip — no grasp needed) drives its pusher plate against the rocker's
upper arm; the rocker pivots and its lower arm pushes the bottom drawer OUT.
Plan-level contrast with the seed: the seed pulls the judged drawer by its handle;
here the judged drawer is untouched and untouchable — the robot pushes a DIFFERENT
drawer inward, in the OPPOSITE direction, and a hidden 1:1 mechanical linkage
produces the goal state. The seed's strategy applied to this scene (pull a drawer
outward) is useless: pulling the top drawer just runs it to its outer stop and
disengages the rocker (smoke proves this with real force), and the bottom drawer
offers no purchase to pull.

No stored energy anywhere: the rocker is mass-balanced on its pivot (CoM at the
axle) and both drawers ride horizontal prismatic joints, so the mechanism holds
whatever state it is left in — the outcome persists hands-off.

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide): cabinet (KINEMATIC: sides, back, base, mid shelf-block, top panel —
the rear chamber is sealed, reachable only through the drawer cavities the drawers
themselves fill), top drawer (DYNAMIC: open box + oversized front lip + rear
pusher prong), bottom drawer (DYNAMIC: open box + recessed smooth front plate +
rear pusher prong), rocker (DYNAMIC balanced plank + two contact pads), plus a
black bowl and a plate on the cabinet top (distractors, from the seed scene).
Joints are authored per-env at bind time (cabinet=body0 kinematic, follower=body1;
collision filtering applies to that pair only, so drawer<->rocker contact — the
transmission — still collides; joint LIMITS are the drawer hard stops). Sliding
surfaces carry a bound polished material (mu ~0.06, min combine) — PhysX's default
~0.5 friction would jam the pad-on-plate transmission.

Per-episode randomization (readback-verified by smoke): the top drawer's initial
opening d0 (how far it protrudes, which sets the free travel before the rocker
engages), and the xy poses of the bowl and plate on the cabinet top.

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.15  engaged — top drawer ever pushed >= 3 cm inward from its opening (latched)
  0.25  cracked — bottom drawer ever open >= 3 cm (latched)
  0.25  ajar    — bottom drawer ever open >= 6 cm (latched)
capped at 0.65; exactly 1.0 iff success(): bottom drawer open >= `open_goal`
(8 cm), everything settled and finite. Null policy ~0 (nothing moves). The seed's
strategy — pull a drawer outward — earns ~0.15 at most (it can never crack the
bottom drawer).

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


# ----- custom compound spawners -----------------------------------------------------------------
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


def _span(stage, path: str, *, x, y, z, color, collide: Callable):
    """Box child from axis spans (x0, x1), (y0, y1), (z0, z1)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2))
    xf.AddScaleOp().Set(Gf.Vec3f(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _mk_material(prim_path: str, name: str, mu_s: float, mu_d: float, combine: str) -> str:
    import isaaclab.sim as sim_utils

    mat_path = f"{prim_path}/{name}"
    sim_utils.spawn_rigid_body_material(mat_path, sim_utils.RigidBodyMaterialCfg(
        static_friction=float(mu_s), dynamic_friction=float(mu_d), restitution=0.0,
        friction_combine_mode=combine))
    return mat_path


def _dyn_body(root, mass: float, lin_damp: float, ang_damp: float) -> None:
    """Standard dynamic compound body physics (32/4 iters, no sleep, damped)."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(4)
    prb.CreateLinearDampingAttr(float(lin_damp))
    prb.CreateAngularDampingAttr(float(ang_damp))
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)


def _spawn_cabinet(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the cabinet at `prim_path`: KINEMATIC compound. Local frame: x=0 is
    the FRONT face plane (+x = out of the cabinet, the drawer-opening direction),
    z=0 is the ground. Two drawer cavities open at the front; behind them
    (x < mid_x0) is the sealed rocker chamber (bounded by sides, back, base and
    top panel), reachable only through the cavities the drawers occupy."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/side_p", x=(c.back_x0, 0.0), y=(c.wall_y0, c.wall_y1),
          z=(0.0, c.top_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/side_n", x=(c.back_x0, 0.0), y=(-c.wall_y1, -c.wall_y0),
          z=(0.0, c.top_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/back", x=(c.back_x0, c.back_x1),
          y=(-c.wall_y0, c.wall_y0), z=(0.0, c.top_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/base", x=(c.back_x1, 0.0), y=(-c.wall_y0, c.wall_y0),
          z=(0.0, c.base_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/mid", x=(c.mid_x0, 0.0), y=(-c.wall_y0, c.wall_y0),
          z=(c.mid_z0, c.mid_z1), color=c.trim_color, collide=collide)
    _span(stage, f"{prim_path}/top", x=(c.back_x1, 0.0), y=(-c.wall_y0, c.wall_y0),
          z=(c.top_z0, c.top_z1), color=c.trim_color, collide=collide)
    wood = _mk_material(prim_path, "wood", c.wood_mu_s, c.wood_mu_d, "average")
    bind_physics_material(prim_path, wood)
    return root


def _spawn_top_drawer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the top drawer at its CLOSED pose, root origin = cabinet origin
    (so the prismatic joint anchors coincide and the joint coordinate IS the
    opening). Open box + oversized red front LIP (the push surface, proud of the
    cabinet face by its thickness even when flush) + the rear pusher prong: two
    side rails straddling the rocker plank, carrying the pusher PLATE that meets
    the rocker's upper pad."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/lip", x=(0.0, c.lip_t), y=(-c.lip_hy, c.lip_hy),
          z=(c.lip_z0, c.lip_z1), color=c.lip_color, collide=collide)
    _span(stage, f"{prim_path}/floor", x=(c.body_x0, 0.0), y=(-c.body_hy, c.body_hy),
          z=(c.floor_z0, c.floor_z1), color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/w_rear", x=(c.body_x0, c.body_x0 + 0.015),
          y=(-c.body_hy, c.body_hy), z=(c.floor_z1, c.wall_z1),
          color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/w_yp", x=(c.body_x0, 0.0), y=(c.body_hy - 0.015, c.body_hy),
          z=(c.floor_z1, c.wall_z1), color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/w_yn", x=(c.body_x0, 0.0), y=(-c.body_hy, -c.body_hy + 0.015),
          z=(c.floor_z1, c.wall_z1), color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/rail_p", x=(c.plate_x1, c.body_x0 + 0.015),
          y=(c.rail_y0, c.rail_y1), z=(c.plate_z0, c.plate_z1),
          color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/rail_n", x=(c.plate_x1, c.body_x0 + 0.015),
          y=(-c.rail_y1, -c.rail_y0), z=(c.plate_z0, c.plate_z1),
          color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/plate", x=(c.plate_x0, c.plate_x1),
          y=(-c.plate_hy, c.plate_hy), z=(c.plate_z0, c.plate_z1),
          color=c.plate_color, collide=collide)
    _dyn_body(root, c.mass, c.lin_damp, c.ang_damp)
    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    bind_physics_material(prim_path, slick)
    return root


def _spawn_bottom_drawer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the bottom drawer at its CLOSED pose, root origin = cabinet origin.
    Open box whose FRONT is a smooth plate recessed `recess` behind the cabinet
    face with only a `side_gap` perimeter gap — no handle, no lip, no purchase —
    plus the short rear pusher prong (shaft + plate) that the rocker's lower pad
    pushes outward."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/front", x=(c.front_x0 - c.front_t, c.front_x0),
          y=(-c.front_hy, c.front_hy), z=(c.front_z0, c.front_z1),
          color=c.front_color, collide=collide)
    _span(stage, f"{prim_path}/floor", x=(c.body_x0, c.front_x0 - c.front_t),
          y=(-c.body_hy, c.body_hy), z=(c.floor_z0, c.floor_z1),
          color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/w_rear", x=(c.body_x0, c.body_x0 + 0.015),
          y=(-c.body_hy, c.body_hy), z=(c.floor_z1, c.wall_z1),
          color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/w_yp", x=(c.body_x0, c.front_x0 - c.front_t),
          y=(c.body_hy - 0.015, c.body_hy), z=(c.floor_z1, c.wall_z1),
          color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/w_yn", x=(c.body_x0, c.front_x0 - c.front_t),
          y=(-c.body_hy, -c.body_hy + 0.015), z=(c.floor_z1, c.wall_z1),
          color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/shaft", x=(c.plate_x1, c.body_x0),
          y=(-c.plate_hy, c.plate_hy), z=(c.plate_z0, c.plate_z1),
          color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/plate", x=(c.plate_x0, c.plate_x1),
          y=(-c.plate_hy, c.plate_hy), z=(c.plate_z0, c.plate_z1),
          color=c.plate_color, collide=collide)
    _dyn_body(root, c.mass, c.lin_damp, c.ang_damp)
    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    bind_physics_material(prim_path, slick)
    return root


def _spawn_rocker(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the rocker with root origin AT the pivot (so the revolute anchor is
    the root AND — with MassAPI mass — the CoM sits on the axle: the rocker is
    gravity-balanced and holds any angle; no stored energy). A thin plank spanning
    z in [-arm_r-pad_hz, +arm_r+pad_hz] with two +x contact pads at the arm tips."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    zt = c.arm_r + c.pad_hz
    _span(stage, f"{prim_path}/plank", x=(-c.plank_t / 2, c.plank_t / 2),
          y=(-c.plank_hy, c.plank_hy), z=(-zt, zt), color=c.plank_color, collide=collide)
    _span(stage, f"{prim_path}/pad_u", x=(c.plank_t / 2, c.plank_t / 2 + c.pad_d),
          y=(-c.pad_hy, c.pad_hy), z=(c.arm_r - c.pad_hz, c.arm_r + c.pad_hz),
          color=c.pad_color, collide=collide)
    _span(stage, f"{prim_path}/pad_l", x=(c.plank_t / 2, c.plank_t / 2 + c.pad_d),
          y=(-c.pad_hy, c.pad_hy), z=(-c.arm_r - c.pad_hz, -c.arm_r + c.pad_hz),
          color=c.pad_color, collide=collide)
    _dyn_body(root, c.mass, 0.05, c.ang_damp)
    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    bind_physics_material(prim_path, slick)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cabinet" not in _SPAWNER_CACHE:

        @configclass
        class CabinetSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cabinet)
            wall_y0: float = 0.14
            wall_y1: float = 0.17
            back_x0: float = -0.49
            back_x1: float = -0.47
            base_z1: float = 0.05
            mid_x0: float = -0.26
            mid_z0: float = 0.21
            mid_z1: float = 0.50
            top_z0: float = 0.66
            top_z1: float = 0.70
            body_color: tuple = (0.52, 0.38, 0.24)
            trim_color: tuple = (0.44, 0.31, 0.19)
            contact_offset: float = 0.0015
            wood_mu_s: float = 0.60
            wood_mu_d: float = 0.55

        @configclass
        class TopDrawerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_top_drawer)
            lip_t: float = 0.018
            lip_hy: float = 0.148
            lip_z0: float = 0.492
            lip_z1: float = 0.668
            body_x0: float = -0.20
            body_hy: float = 0.135
            floor_z0: float = 0.5015
            floor_z1: float = 0.5095
            wall_z1: float = 0.63
            rail_y0: float = 0.06
            rail_y1: float = 0.09
            plate_x0: float = -0.395
            plate_x1: float = -0.38
            plate_hy: float = 0.10
            plate_z0: float = 0.535
            plate_z1: float = 0.635
            mass: float = 0.60
            lin_damp: float = 6.0
            ang_damp: float = 4.0
            lip_color: tuple = (0.78, 0.16, 0.12)
            box_color: tuple = (0.68, 0.56, 0.40)
            plate_color: tuple = (0.35, 0.30, 0.24)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.06
            slide_mu_d: float = 0.05

        @configclass
        class BottomDrawerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottom_drawer)
            front_x0: float = -0.004
            front_t: float = 0.018
            front_hy: float = 0.138
            front_z0: float = 0.052
            front_z1: float = 0.208
            body_x0: float = -0.24
            body_hy: float = 0.135
            floor_z0: float = 0.0515
            floor_z1: float = 0.0595
            wall_z1: float = 0.19
            plate_x0: float = -0.272
            plate_x1: float = -0.257
            plate_hy: float = 0.10
            plate_z0: float = 0.09
            plate_z1: float = 0.19
            mass: float = 0.60
            lin_damp: float = 6.0
            ang_damp: float = 4.0
            front_color: tuple = (0.42, 0.44, 0.50)
            box_color: tuple = (0.68, 0.56, 0.40)
            plate_color: tuple = (0.35, 0.30, 0.24)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.06
            slide_mu_d: float = 0.05

        @configclass
        class RockerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rocker)
            arm_r: float = 0.24
            plank_t: float = 0.02
            plank_hy: float = 0.05
            pad_d: float = 0.015
            pad_hy: float = 0.03
            pad_hz: float = 0.025
            mass: float = 0.40
            ang_damp: float = 0.5
            plank_color: tuple = (0.85, 0.45, 0.10)
            pad_color: tuple = (0.90, 0.60, 0.15)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.06
            slide_mu_d: float = 0.05

        _SPAWNER_CACHE["cabinet"] = CabinetSpawnerCfg
        _SPAWNER_CACHE["top_drawer"] = TopDrawerSpawnerCfg
        _SPAWNER_CACHE["bottom_drawer"] = BottomDrawerSpawnerCfg
        _SPAWNER_CACHE["rocker"] = RockerSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RockerCabinetSceneCfg(BaseCfg):
    """Config for `RockerCabinetScene`. The transmission contract is asserted in
    `__post_init__`: the top drawer's full inward travel really drives the rocker
    far enough to push the bottom drawer past the goal opening, the rocker really
    fits its chamber at full swing, its pads really stay on the pusher plates
    through the arc, and the bottom drawer's front really offers no purchase."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    open_goal: float = tunable(0.080)     # bottom-drawer opening for success (m)
    settle_lin: float = tunable(0.05)     # max |lin vel| of movers when judging (m/s)
    settle_ang: float = tunable(0.30)     # max |ang vel| of the rocker when judging (rad/s)
    engage_delta: float = tunable(0.030)  # top pushed this far inward -> "engaged" latch (m)
    crack_open: float = tunable(0.030)    # bottom opening -> "cracked" latch (m)
    ajar_open: float = tunable(0.060)     # bottom opening -> "ajar" latch (m)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    d0_lo: float = tunable(0.125)         # top-drawer initial opening, low (m)
    d0_hi: float = tunable(0.134)         # top-drawer initial opening, high (m)
    bowl_x: tuple = tunable((-0.42, -0.30))   # bowl x band on the cabinet top
    plate_x: tuple = tunable((-0.16, -0.04))  # plate x band on the cabinet top
    topper_y: float = tunable(0.08)       # +/- y jitter for bowl and plate

    # --- info: cabinet (local frame: x=0 front face, +x = opening direction, z=0 ground) ---------
    wall_y0: float = info(0.14)           # cavity half-width (inner side-wall face)
    wall_y1: float = info(0.17)
    back_x0: float = info(-0.49)
    back_x1: float = info(-0.47)          # chamber rear inner face
    base_z1: float = info(0.05)
    mid_x0: float = info(-0.26)           # cavities end / chamber begins
    mid_z0: float = info(0.21)
    mid_z1: float = info(0.50)
    top_z0: float = info(0.66)
    top_z1: float = info(0.70)            # cabinet top surface (distractors stand here)
    # --- info: joints (limits ARE the drawer hard stops; authored pose = closed = 0) -------------
    top_stroke: float = info(0.145)       # top-drawer prismatic upper limit (m)
    bot_stroke: float = info(0.160)       # bottom-drawer prismatic upper limit (m)
    rocker_max_deg: float = info(34.0)    # revolute limit, transmission direction
    rocker_back_deg: float = info(3.0)    # revolute limit, slack direction
    # --- info: rocker (root/pivot at (pivot_x, 0, pivot_z); pads face +x) ------------------------
    pivot_x: float = info(-0.30)
    pivot_z: float = info(0.36)
    arm_r: float = info(0.24)             # pivot -> pad centre
    plank_t: float = info(0.02)
    pad_d: float = info(0.015)
    pad_hz: float = info(0.025)
    rocker_mass: float = info(0.40)
    # --- info: pusher plates (drawer-local = cabinet-local at closed) ----------------------------
    top_plate_x0: float = info(-0.395)    # top prong plate rear face at closed
    top_plate_z0: float = info(0.535)
    top_plate_z1: float = info(0.635)
    bot_plate_x0: float = info(-0.272)    # bottom prong plate rear face at closed
    bot_plate_z0: float = info(0.09)
    bot_plate_z1: float = info(0.19)
    # --- info: bottom-drawer front (the "no purchase" contract) ----------------------------------
    recess: float = info(0.004)           # front plate sits this far behind the cabinet face
    side_gap: float = info(0.002)         # perimeter gap around the recessed plate
    # --- info: drawers ---------------------------------------------------------------------------
    drawer_mass: float = info(0.60)
    lip_t: float = info(0.018)
    # --- info: distractors -----------------------------------------------------------------------
    bowl_r: float = info(0.045)
    bowl_h: float = info(0.050)
    bowl_mass: float = info(0.15)
    plate_r: float = info(0.070)
    plate_h: float = info(0.012)
    plate_mass: float = info(0.20)
    # --- info: materials -------------------------------------------------------------------------
    slide_mu_s: float = info(0.06)        # polished transmission: default ~0.5 would jam it
    slide_mu_d: float = info(0.05)
    contact_offset: float = info(0.0015)
    # --- info: rubric weights (0.15 + 0.25 + 0.25 = 0.65 = the non-success cap) ------------------
    w_engage: float = info(0.15)
    w_crack: float = info(0.25)
    w_ajar: float = info(0.25)

    def __post_init__(self) -> None:
        # pad front face x (cabinet frame, rocker at rest)
        pad_face = self.pivot_x + self.plank_t / 2 + self.pad_d
        # engagement travel: top drawer inward push that the upper pad absorbs
        d_eng = pad_face - self.top_plate_x0
        assert abs(d_eng - 0.120) < 1e-9, "engagement travel is the design's 12 cm"
        # free travel before engagement exists at every sampled opening
        assert self.d0_lo - d_eng >= 0.004, "reset must leave a free gap before contact"
        assert self.d0_hi > self.d0_lo, "randomization band must be non-empty"
        # the pull-out probe (smoke) always has real travel to the outer stop
        assert self.top_stroke - self.d0_hi >= 0.008, "outer stop must be reachable by a pull"
        # rocker swing needed to absorb d_eng, with limit margin
        th_end = math.asin(d_eng / self.arm_r)
        assert math.degrees(th_end) <= self.rocker_max_deg - 3.0, \
            "rocker limit must not clip the working swing"
        # transmitted output beats the goal with margin (1:1 arms, minus slack c0)
        c0 = self.bot_plate_x0 - pad_face
        assert 0.001 <= c0 <= 0.006, "bottom pad-to-plate slack is a few mm"
        out_est = self.arm_r * math.sin(th_end) - c0
        assert out_est >= self.open_goal + 0.025, "full push must clear the goal with margin"
        assert self.bot_stroke >= out_est + 0.010, "bottom stroke must not clip the output"
        assert self.ajar_open < self.open_goal < out_est, "latch ladder must be climbable"
        # rocker fits the chamber at full swing (upper tip sweeps toward the back wall)
        tip_x = self.pivot_x - (self.arm_r + self.pad_hz) * math.sin(math.radians(self.rocker_max_deg)) \
            - self.plank_t / 2
        assert tip_x > self.back_x1 + 0.005, "rocker must clear the back wall at full swing"
        # pads stay on the pusher plates through the working arc
        zu = self.pivot_z + self.arm_r * math.cos(th_end)
        assert zu - self.pad_hz * math.cos(th_end) >= self.top_plate_z0 + 0.005, \
            "upper pad must stay on the top plate at full swing"
        assert self.pivot_z + self.arm_r + self.pad_hz <= self.top_plate_z1, \
            "upper pad must start on the top plate"
        zl = self.pivot_z - self.arm_r * math.cos(th_end)
        assert zl + self.pad_hz * math.cos(th_end) <= self.bot_plate_z1 - 0.005, \
            "lower pad must stay on the bottom plate at full swing"
        assert self.pivot_z - self.arm_r - self.pad_hz >= self.bot_plate_z0, \
            "lower pad must start on the bottom plate"
        # no-purchase contract on the judged drawer's front
        assert self.recess >= 0.004 and self.side_gap <= 0.003, \
            "bottom front must be recessed and gap-sealed (nothing to hook or pinch)"
        # distractor bands never overlap (bowl and plate cannot spawn intersecting)
        assert self.plate_x[0] - self.bowl_x[1] >= self.bowl_r + self.plate_r + 0.02
        assert self.bowl_x[0] >= self.back_x1 + self.bowl_r - 0.02
        assert self.plate_x[1] <= -self.plate_r + 0.04
        assert abs(self.w_engage + self.w_crack + self.w_ajar - 0.65) < 1e-9


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("rocker_cabinet")
class RockerCabinetScene(BaseScene):
    cfg: RockerCabinetSceneCfg

    def __init__(self, cfg: RockerCabinetSceneCfg | None = None) -> None:
        super().__init__(cfg or RockerCabinetSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        cab_spawn = cls["cabinet"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            wall_y0=c.wall_y0, wall_y1=c.wall_y1, back_x0=c.back_x0, back_x1=c.back_x1,
            base_z1=c.base_z1, mid_x0=c.mid_x0, mid_z0=c.mid_z0, mid_z1=c.mid_z1,
            top_z0=c.top_z0, top_z1=c.top_z1, contact_offset=c.contact_offset)
        top_spawn = cls["top_drawer"](
            mass=c.drawer_mass, lip_t=c.lip_t, plate_x0=c.top_plate_x0,
            plate_z0=c.top_plate_z0, plate_z1=c.top_plate_z1,
            contact_offset=c.contact_offset, slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)
        bot_spawn = cls["bottom_drawer"](
            mass=c.drawer_mass, front_x0=-c.recess, front_hy=c.wall_y0 - c.side_gap,
            plate_x0=c.bot_plate_x0, plate_z0=c.bot_plate_z0, plate_z1=c.bot_plate_z1,
            contact_offset=c.contact_offset, slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)
        rocker_spawn = cls["rocker"](
            arm_r=c.arm_r, plank_t=c.plank_t, pad_d=c.pad_d, pad_hz=c.pad_hz,
            mass=c.rocker_mass, contact_offset=c.contact_offset,
            slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.20, angular_damping=0.20,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=4)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "cabinet": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet", spawn=cab_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            # Drawers/rocker are authored IN PLACE at their closed/rest poses: the
            # bind-time joints anchor at these authored positions (i62 pattern).
            "top_drawer": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/TopDrawer", spawn=top_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "bottom_drawer": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BottomDrawer", spawn=bot_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "rocker": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rocker", spawn=rocker_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.pivot_x, 0.0, c.pivot_z))),
            "bowl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl",
                spawn=sim_utils.CylinderCfg(
                    radius=c.bowl_r, height=c.bowl_h, axis="Z",
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
                    rigid_props=rigid, collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.50, dynamic_friction=0.45, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.06, 0.06, 0.08))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.2, 0.6, 0.05))),
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate",
                spawn=sim_utils.CylinderCfg(
                    radius=c.plate_r, height=c.plate_h, axis="Z",
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.plate_mass),
                    rigid_props=rigid, collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.50, dynamic_friction=0.45, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.95))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.2, -0.6, 0.05))),
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
        self.cabinet: RigidObject = env.iscene["cabinet"]
        self.top: RigidObject = env.iscene["top_drawer"]
        self.bot: RigidObject = env.iscene["bottom_drawer"]
        self.rocker: RigidObject = env.iscene["rocker"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.plate: RigidObject = env.iscene["plate"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke)
        self.d0 = torch.zeros(n, device=dev)          # initial top-drawer opening
        # latches (partial credit survives transients; success is judged live)
        self._engage = torch.zeros(n, dtype=torch.bool, device=dev)
        self._crack = torch.zeros(n, dtype=torch.bool, device=dev)
        self._ajar = torch.zeros(n, dtype=torch.bool, device=dev)
        self._author_joints()

    def _author_joints(self) -> None:
        """Per-env joints, authored ONCE at bind time against the AUTHORED poses
        (drawer roots coincide with the cabinet root; the rocker root is its pivot).
        The joint LIMITS are the drawer hard stops; joint collision filtering only
        disables the cabinet<->follower pair, so drawer<->rocker transmission
        contact still collides."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        c = self.cfg
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            for name, body, hi in (("top_slide", "TopDrawer", c.top_stroke),
                                   ("bottom_slide", "BottomDrawer", c.bot_stroke)):
                j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/{name}")
                j.CreateBody0Rel().SetTargets([f"{base}/Cabinet"])
                j.CreateBody1Rel().SetTargets([f"{base}/{body}"])
                j.CreateCollisionEnabledAttr(False)
                j.CreateAxisAttr("X")
                j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLowerLimitAttr(0.0)
                j.CreateUpperLimitAttr(float(hi))
            r = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/rocker_pivot")
            r.CreateBody0Rel().SetTargets([f"{base}/Cabinet"])
            r.CreateBody1Rel().SetTargets([f"{base}/Rocker"])
            r.CreateCollisionEnabledAttr(False)
            r.CreateAxisAttr("Y")
            r.CreateLocalPos0Attr(Gf.Vec3f(float(c.pivot_x), 0.0, float(c.pivot_z)))
            r.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            r.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            r.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            # +angle about +y swings the UPPER arm toward +x; transmission
            # (upper arm pushed to -x) is the NEGATIVE direction. Degrees.
            r.CreateLowerLimitAttr(-float(c.rocker_max_deg))
            r.CreateUpperLimitAttr(float(c.rocker_back_deg))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: cabinet re-asserted at its fixed pose (kinematic joint
        anchors are world-fixed — the cabinet must never move), rocker at rest
        (0 deg), bottom drawer closed, top drawer at a SAMPLED opening d0, bowl
        and plate at sampled poses on the cabinet top, latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def place(body, dx, dy, dz) -> None:
            s = torch.zeros(m, 13, device=dev)
            s[:, 0] = origin[:, 0] + dx
            s[:, 1] = origin[:, 1] + dy
            s[:, 2] = origin[:, 2] + dz
            s[:, 3] = 1.0
            body.write_root_state_to_sim(s, env_ids)

        zeros = torch.zeros(m, device=dev)
        place(self.cabinet, zeros, zeros, zeros)
        place(self.rocker, zeros + c.pivot_x, zeros, zeros + c.pivot_z)
        place(self.bot, zeros, zeros, zeros)

        d0 = c.d0_lo + torch.rand(m, device=dev) * (c.d0_hi - c.d0_lo)
        self.d0[env_ids] = d0
        place(self.top, d0, zeros, zeros)

        u = torch.rand(m, 4, device=dev)
        bx = c.bowl_x[0] + u[:, 0] * (c.bowl_x[1] - c.bowl_x[0])
        by = (u[:, 1] * 2 - 1) * c.topper_y
        place(self.bowl, bx, by, zeros + c.top_z1 + c.bowl_h / 2 + 0.002)
        px = c.plate_x[0] + u[:, 2] * (c.plate_x[1] - c.plate_x[0])
        py = (u[:, 3] * 2 - 1) * c.topper_y
        place(self.plate, px, py, zeros + c.top_z1 + c.plate_h / 2 + 0.002)

        self._engage[env_ids] = False
        self._crack[env_ids] = False
        self._ajar[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cabinet": self.cabinet.data.root_state_w[env_ids].clone(),
            "top": self.top.data.root_state_w[env_ids].clone(),
            "bot": self.bot.data.root_state_w[env_ids].clone(),
            "rocker": self.rocker.data.root_state_w[env_ids].clone(),
            "bowl": self.bowl.data.root_state_w[env_ids].clone(),
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "d0": self.d0[env_ids].clone(),
            "engage": self._engage[env_ids].clone(),
            "crack": self._crack[env_ids].clone(),
            "ajar": self._ajar[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cabinet.write_root_state_to_sim(state["cabinet"], env_ids)
        self.top.write_root_state_to_sim(state["top"], env_ids)
        self.bot.write_root_state_to_sim(state["bot"], env_ids)
        self.rocker.write_root_state_to_sim(state["rocker"], env_ids)
        self.bowl.write_root_state_to_sim(state["bowl"], env_ids)
        self.plate.write_root_state_to_sim(state["plate"], env_ids)
        self.d0[env_ids] = state["d0"]
        self._engage[env_ids] = state["engage"]
        self._crack[env_ids] = state["crack"]
        self._ajar[env_ids] = state["ajar"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden KITCHEN CABINET stands on the ground, {c.top_z1 * 100:.0f} cm "
            f"tall, with a black bowl and a white plate sitting on its top. It has "
            f"two drawers. The TOP drawer starts pulled OUT: its oversized RED "
            f"front panel protrudes about {100 * (0.125 + c.lip_t):.0f} cm from the "
            f"cabinet face (the exact opening varies by episode). The BOTTOM drawer "
            f"is shut: its grey front plate is a smooth panel RECESSED "
            f"{c.recess * 1000:.0f} mm behind the cabinet face with only a "
            f"{c.side_gap * 1000:.0f} mm gap around it — no handle, no lip, no "
            f"edge to hook or pinch, and it already rests on its inner stop, so "
            f"neither pulling nor pushing it directly does anything.\n"
            f"Inside the cabinet's sealed rear chamber (unreachable — every access "
            f"is blocked by the drawers that fill their openings) a balanced ROCKER "
            f"plank pivots on a horizontal axle: its upper arm faces the top "
            f"drawer's rear pusher plate, its lower arm faces the bottom drawer's. "
            f"PUSH THE TOP DRAWER FLUSH — a straight inward push on its red panel, "
            f"no grasp needed — and after about 1 cm of free travel the drawer "
            f"engages the rocker, which pivots and drives the BOTTOM drawer OUT "
            f"about 11 cm. The mechanism is friction-held and mass-balanced: "
            f"wherever you stop, everything stays put, and the transmission only "
            f"works by pushing the top drawer IN (pulling it back out to its outer "
            f"stop just disengages the rocker and leaves the bottom drawer shut).\n"
            f"Goal: the bottom drawer open at least {c.open_goal * 100:.0f} cm, "
            f"everything at rest. The bowl and the plate are bystanders — leave "
            f"them where they sit."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Open the cabinet's bottom drawer. It has no handle and cannot be "
            "pulled: instead push the protruding top drawer flush into the "
            "cabinet — the internal rocker linkage drives the bottom drawer "
            "out. Finish with the bottom drawer open and everything at rest."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _local(self, body) -> torch.Tensor:
        """(N,3) body root position in the cabinet frame (cabinet is fixed at the
        env origin with identity heading)."""
        return body.data.root_pos_w - self.env_origins

    def top_open(self) -> torch.Tensor:
        """(N,) top-drawer opening (its prismatic coordinate; 0 = flush/closed)."""
        return self._local(self.top)[:, 0]

    def bot_open(self) -> torch.Tensor:
        """(N,) bottom-drawer opening (its prismatic coordinate; 0 = closed)."""
        return self._local(self.bot)[:, 0]

    def settled(self) -> torch.Tensor:
        """(N,) bool: drawers, distractors slow; rocker rotation slow."""
        c = self.cfg
        return (self.top.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.bot.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.plate.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.rocker.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.top.data.root_pos_w, self.bot.data.root_pos_w,
                         self.rocker.data.root_pos_w, self.bowl.data.root_pos_w,
                         self.plate.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        self._engage |= ((self.d0 - self.top_open()) >= c.engage_delta) & fin
        b = self.bot_open()
        self._crack |= (b >= c.crack_open) & fin
        self._ajar |= (b >= c.ajar_open) & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the bottom drawer stands open at least `open_goal`, with
        everything settled and finite — a LIVE physical outcome. The drawer's
        front offers no purchase and its joint bottoms out inward, so the only
        physical route to this state is driving the internal rocker by pushing
        the top drawer flush."""
        self._update_latches()
        return (self.bot_open() >= self.cfg.open_goal) & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15 engaged + 0.25 cracked (>=3 cm) + 0.25 ajar
        (>=6 cm), all latched, capped at 0.65; exactly 1.0 iff success() holds
        live. Doing nothing scores ~0; the seed's strategy (pull a drawer
        outward) can only ever move the top drawer to its outer stop and scores
        ~0 (no latch fires)."""
        c = self.cfg
        self._update_latches()
        base = (c.w_engage * self._engage.float() + c.w_crack * self._crack.float()
                + c.w_ajar * self._ajar.float()).clamp(max=0.65)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="rocker_cabinet", robot="null"))
