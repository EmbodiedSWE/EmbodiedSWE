"""BayonetDrawerScene — open the handle-less bottom drawer with a twist-lock KEY
(sim_gen task `libero_kitchen_scene1_open_bottom_drawer_i88`).

Derived from libero_90/libero_kitchen_scene1_open_bottom_drawer, but STRATEGICALLY
different: the seed's whole skill is grasping the bottom drawer's own handle and
pulling. Here the drawer HAS no handle and offers no purchase at all — its front
is a single smooth 30 mm-thick slick slab, flush with the cabinet face, with only
a 4 mm perimeter gap (a Franka fingertip is ~23 mm deep: it cannot reach through
the 16 mm slot and hook the inner face). The ONLY way to open it is a BAYONET
COUPLING: fetch the free T-KEY lying on the floor, insert its horizontal crossbar
through the keyed SLOT in the slab, TWIST the key ~90 degrees so the crossbar
stands vertical BEHIND the slab (46 mm bar vs 16 mm slot — a pure geometric
interlock), and PULL: the crossbar bears on the slab's inner face and drags the
drawer open. Untwisted, the same pull simply slides the key back out; the cabinet
face also carries a visually identical DECOY slot higher up (in a FIXED panel) —
keying and twisting there locks the key uselessly to the cabinet. Plan-level
contrast with the seed: the seed's manipuland IS the goal object (grab drawer,
pull drawer); here the manipuland is a separate free TOOL that must first be
mated to the goal object through an insert+twist sequence before any pull can
transmit — the seed's strategy (pull on the drawer itself) has nothing to grab,
and an untwisted or decoy-slotted key provably transmits nothing (smoke proves
both with real forces).

No stored energy anywhere: the drawer rides a horizontal prismatic joint (limits
are its hard stops), the key is a free rigid body, everything is friction-held
and damped — the outcome persists hands-off.

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide): cabinet (KINEMATIC: sides, back, base, mid shelf, top, and the
fixed decoy front panel with the decoy slot), drawer (DYNAMIC: slotted front
slab + open box), T-key (DYNAMIC: crossbar + shaft + knob, root at the shaft
midpoint), plus a black bowl and a white plate on the cabinet top (distractors,
from the seed scene). The drawer joint is authored per-env at bind time
(cabinet=body0 kinematic, drawer=body1; collision filtering applies to that pair
only, so key<->drawer and key<->cabinet contacts — the coupling — still collide).
Sliding surfaces carry a bound slick material (mu ~0.06, min combine) — PhysX's
default ~0.5 friction would jam the bar-in-slot passage.

Per-episode randomization (readback-verified by smoke): the key's floor pose
(x, y, yaw), and the xy poses of the bowl and plate on the cabinet top.

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.15  keyed  — crossbar ever fully behind the drawer's slab, at slot height
  0.20  locked — keyed with the crossbar twisted past 45 deg (interlock engaged)
  0.25  cracked— drawer ever open >= 3 cm
capped at 0.60; exactly 1.0 iff success(): drawer open >= `open_goal` (8 cm),
everything settled and finite. Null policy ~0 (nothing moves). The seed's
strategy — pull the drawer directly — finds no purchase and scores ~0; pulling
an UNTWISTED inserted key extracts the key and scores <= 0.15.

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
    z=0 is the ground. The lower cavity (base..shelf) holds the drawer; the upper
    cavity (shelf..top) is SEALED by the fixed decoy front panel, which carries a
    slot visually identical to the drawer's — keying it achieves nothing."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/side_p", x=(c.back_x0, 0.0), y=(c.side_y0, c.side_y1),
          z=(0.0, c.top_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/side_n", x=(c.back_x0, 0.0), y=(-c.side_y1, -c.side_y0),
          z=(0.0, c.top_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/back", x=(c.back_x0, c.back_x1),
          y=(-c.side_y0, c.side_y0), z=(0.0, c.top_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/base", x=(c.back_x1, 0.0), y=(-c.side_y0, c.side_y0),
          z=(0.0, c.base_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/shelf", x=(c.back_x1, 0.0), y=(-c.side_y0, c.side_y0),
          z=(c.shelf_z0, c.shelf_z1), color=c.trim_color, collide=collide)
    _span(stage, f"{prim_path}/top", x=(c.back_x1, 0.0), y=(-c.side_y0, c.side_y0),
          z=(c.top_z0, c.top_z1), color=c.trim_color, collide=collide)
    # fixed DECOY front panel (upper cavity), with a slot identical to the drawer's
    _span(stage, f"{prim_path}/dec_lo", x=(-c.panel_t, 0.0), y=(-c.side_y0, c.side_y0),
          z=(c.shelf_z1, c.decoy_z - c.slot_hh), color=c.front_color, collide=collide)
    _span(stage, f"{prim_path}/dec_hi", x=(-c.panel_t, 0.0), y=(-c.side_y0, c.side_y0),
          z=(c.decoy_z + c.slot_hh, c.top_z0), color=c.front_color, collide=collide)
    _span(stage, f"{prim_path}/dec_yp", x=(-c.panel_t, 0.0), y=(c.slot_hw, c.side_y0),
          z=(c.decoy_z - c.slot_hh, c.decoy_z + c.slot_hh), color=c.front_color, collide=collide)
    _span(stage, f"{prim_path}/dec_yn", x=(-c.panel_t, 0.0), y=(-c.side_y0, -c.slot_hw),
          z=(c.decoy_z - c.slot_hh, c.decoy_z + c.slot_hh), color=c.front_color, collide=collide)
    wood = _mk_material(prim_path, "wood", c.wood_mu_s, c.wood_mu_d, "average")
    bind_physics_material(prim_path, wood)
    return root


def _spawn_drawer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the drawer at its CLOSED pose, root origin = cabinet origin (so the
    prismatic joint anchors coincide and the joint coordinate IS the opening).
    The FRONT is a smooth `panel_t`-thick slab, flush with the cabinet face,
    pierced only by the keyed slot (4 spans around it) — no handle, no lip, no
    purchase. Behind it: open box (floor, rear wall, side walls) with free
    interior around the slot for the crossbar to rotate."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/fr_lo", x=(-c.panel_t, 0.0), y=(-c.front_hy, c.front_hy),
          z=(c.front_z0, c.slot_z - c.slot_hh), color=c.front_color, collide=collide)
    _span(stage, f"{prim_path}/fr_hi", x=(-c.panel_t, 0.0), y=(-c.front_hy, c.front_hy),
          z=(c.slot_z + c.slot_hh, c.front_z1), color=c.front_color, collide=collide)
    _span(stage, f"{prim_path}/fr_yp", x=(-c.panel_t, 0.0), y=(c.slot_hw, c.front_hy),
          z=(c.slot_z - c.slot_hh, c.slot_z + c.slot_hh), color=c.front_color, collide=collide)
    _span(stage, f"{prim_path}/fr_yn", x=(-c.panel_t, 0.0), y=(-c.front_hy, -c.slot_hw),
          z=(c.slot_z - c.slot_hh, c.slot_z + c.slot_hh), color=c.front_color, collide=collide)
    _span(stage, f"{prim_path}/floor", x=(c.body_x0, -c.panel_t),
          y=(-c.front_hy, c.front_hy), z=(c.front_z0, c.front_z0 + 0.010),
          color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/w_rear", x=(c.body_x0, c.body_x0 + 0.015),
          y=(-c.front_hy, c.front_hy), z=(c.front_z0 + 0.010, c.wall_z1),
          color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/w_yp", x=(c.body_x0, -c.panel_t),
          y=(c.front_hy - 0.015, c.front_hy), z=(c.front_z0 + 0.010, c.wall_z1),
          color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/w_yn", x=(c.body_x0, -c.panel_t),
          y=(-c.front_hy, -c.front_hy + 0.015), z=(c.front_z0 + 0.010, c.wall_z1),
          color=c.box_color, collide=collide)
    _dyn_body(root, c.mass, c.lin_damp, c.ang_damp)
    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    bind_physics_material(prim_path, slick)
    return root


def _spawn_key(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the T-key with root origin at the SHAFT midpoint, +x toward the
    crossbar: crossbar (the T-head that passes the slot horizontally and locks
    vertically), square shaft (thin enough to spin freely inside the slot), and
    the grip knob (a cube too big to ever pass the slot)."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/crossbar", x=(c.cross_x0, c.cross_x1),
          y=(-c.cross_hy, c.cross_hy), z=(-c.cross_hz, c.cross_hz),
          color=c.cross_color, collide=collide)
    _span(stage, f"{prim_path}/shaft", x=(c.shaft_x0, c.cross_x0),
          y=(-c.shaft_h, c.shaft_h), z=(-c.shaft_h, c.shaft_h),
          color=c.shaft_color, collide=collide)
    _span(stage, f"{prim_path}/knob", x=(c.knob_x0, c.knob_x1),
          y=(-c.knob_h, c.knob_h), z=(-c.knob_h, c.knob_h),
          color=c.knob_color, collide=collide)
    _dyn_body(root, c.mass, c.lin_damp, c.ang_damp)
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
            side_y0: float = 0.15
            side_y1: float = 0.18
            back_x0: float = -0.40
            back_x1: float = -0.38
            base_z1: float = 0.05
            shelf_z0: float = 0.30
            shelf_z1: float = 0.33
            top_z0: float = 0.66
            top_z1: float = 0.70
            panel_t: float = 0.030
            decoy_z: float = 0.475
            slot_hw: float = 0.028
            slot_hh: float = 0.008
            body_color: tuple = (0.52, 0.38, 0.24)
            trim_color: tuple = (0.44, 0.31, 0.19)
            front_color: tuple = (0.42, 0.44, 0.50)
            contact_offset: float = 0.0015
            wood_mu_s: float = 0.60
            wood_mu_d: float = 0.55

        @configclass
        class DrawerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_drawer)
            panel_t: float = 0.030
            front_hy: float = 0.146
            front_z0: float = 0.052
            front_z1: float = 0.298
            slot_z: float = 0.175
            slot_hw: float = 0.028
            slot_hh: float = 0.008
            body_x0: float = -0.30
            wall_z1: float = 0.28
            mass: float = 1.5
            lin_damp: float = 6.0
            ang_damp: float = 4.0
            front_color: tuple = (0.42, 0.44, 0.50)
            box_color: tuple = (0.68, 0.56, 0.40)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.06
            slide_mu_d: float = 0.05

        @configclass
        class KeySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_key)
            cross_x0: float = 0.055
            cross_x1: float = 0.065
            cross_hy: float = 0.023
            cross_hz: float = 0.005
            shaft_x0: float = -0.065
            shaft_h: float = 0.0045
            knob_x0: float = -0.095
            knob_x1: float = -0.065
            knob_h: float = 0.014
            mass: float = 0.10
            lin_damp: float = 3.0
            ang_damp: float = 3.0
            cross_color: tuple = (0.85, 0.20, 0.10)
            shaft_color: tuple = (0.62, 0.64, 0.68)
            knob_color: tuple = (0.10, 0.25, 0.80)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.06
            slide_mu_d: float = 0.05

        _SPAWNER_CACHE["cabinet"] = CabinetSpawnerCfg
        _SPAWNER_CACHE["drawer"] = DrawerSpawnerCfg
        _SPAWNER_CACHE["key"] = KeySpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BayonetDrawerSceneCfg(BaseCfg):
    """Config for `BayonetDrawerScene`. The bayonet contract is asserted in
    `__post_init__`: the horizontal crossbar really passes the slot with
    clearance, the twisted crossbar really cannot (interlock), the shaft really
    spins freely inside the slot, the knob really can never pass, the slab is
    really too thick to hook through, the pull really drives the drawer past the
    goal, and the decoy slot is really discriminated by the keyed predicate."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    open_goal: float = tunable(0.080)     # drawer opening for success (m)
    settle_lin: float = tunable(0.05)     # max |lin vel| of movers when judging (m/s)
    settle_ang: float = tunable(0.50)     # max |ang vel| of the key when judging (rad/s)
    crack_open: float = tunable(0.030)    # drawer opening -> "cracked" latch (m)
    keyed_x_pen: float = tunable(0.002)   # crossbar this far behind the slab -> keyed (m)
    keyed_y_tol: float = tunable(0.050)   # |y| tolerance for keyed (m)
    keyed_z_tol: float = tunable(0.040)   # |z - slot_z| tolerance for keyed (m)
    lock_cos: float = tunable(0.7071)     # |crossbar axis . z| >= this -> locked (45 deg)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    key_x: tuple = tunable((0.32, 0.52))  # key spawn x band on the floor (m)
    key_y: float = tunable(0.35)          # +/- y band for the key spawn (m)
    bowl_x: tuple = tunable((-0.33, -0.27))   # bowl x band on the cabinet top
    plate_x: tuple = tunable((-0.12, -0.07))  # plate x band on the cabinet top
    topper_y: float = tunable(0.07)       # +/- y jitter for bowl and plate

    # --- info: cabinet (local frame: x=0 front face, +x = opening direction, z=0 ground) ---------
    side_y0: float = info(0.15)           # cavity half-width (inner side-wall face)
    side_y1: float = info(0.18)
    back_x0: float = info(-0.40)
    back_x1: float = info(-0.38)
    base_z1: float = info(0.05)
    shelf_z0: float = info(0.30)          # drawer cavity: base_z1..shelf_z0
    shelf_z1: float = info(0.33)
    top_z0: float = info(0.66)
    top_z1: float = info(0.70)            # cabinet top surface (distractors stand here)
    decoy_z: float = info(0.475)          # DECOY slot height (fixed panel, upper cavity)
    # --- info: the slot (same dims in the drawer slab and the decoy panel) ----------------------
    panel_t: float = info(0.030)          # slab/panel thickness (fingertip is ~23 mm: no hook)
    slot_z: float = info(0.175)           # REAL slot height (drawer slab)
    slot_hw: float = info(0.028)          # slot half-width (y)
    slot_hh: float = info(0.008)          # slot half-height (z)
    # --- info: drawer ----------------------------------------------------------------------------
    front_hy: float = info(0.146)         # slab half-width -> 4 mm side gap (no purchase)
    front_z0: float = info(0.052)
    front_z1: float = info(0.298)
    body_x0: float = info(-0.30)
    wall_z1: float = info(0.28)
    drawer_mass: float = info(1.5)
    drawer_stroke: float = info(0.14)     # prismatic upper limit = outer hard stop (m)
    # --- info: T-key (root at shaft midpoint, +x toward the crossbar) ----------------------------
    cross_x0: float = info(0.055)
    cross_x1: float = info(0.065)         # crossbar thickness 10 mm (passes 16 mm slot flat)
    cross_hy: float = info(0.023)         # crossbar half-length 23 mm (locks: 46 mm > slot)
    cross_hz: float = info(0.005)
    shaft_x0: float = info(-0.065)
    shaft_h: float = info(0.0045)         # 9 mm square shaft (spins freely in the slot)
    knob_x0: float = info(-0.095)
    knob_x1: float = info(-0.065)
    knob_h: float = info(0.014)           # 28 mm knob (can never pass the slot)
    key_mass: float = info(0.10)
    key_z0: float = info(0.016)           # spawn root height on the floor
    # --- info: solve anchors (asserted feasible below; read by solve/smoke) ---------------------
    key_ins_x: float = info(0.018)        # key root x at full insertion (crossbar behind slab)
    key_pull_x: float = info(0.118)       # key root x at end of pull (drawer ~9.3 cm open)
    # --- info: materials -------------------------------------------------------------------------
    slide_mu_s: float = info(0.06)        # slick movers: default ~0.5 would jam the slot
    slide_mu_d: float = info(0.05)
    contact_offset: float = info(0.0015)
    # --- info: distractors -----------------------------------------------------------------------
    bowl_r: float = info(0.045)
    bowl_h: float = info(0.050)
    bowl_mass: float = info(0.15)
    plate_r: float = info(0.070)
    plate_h: float = info(0.012)
    plate_mass: float = info(0.20)
    # --- info: rubric weights (0.15 + 0.20 + 0.25 = 0.60 = the non-success cap) ------------------
    w_keyed: float = info(0.15)
    w_locked: float = info(0.20)
    w_crack: float = info(0.25)

    def __post_init__(self) -> None:
        # -- insertion: the FLAT crossbar passes the slot with real clearance
        assert self.slot_hw - self.cross_hy >= 0.004, "crossbar needs y clearance in the slot"
        assert self.slot_hh - self.cross_hz >= 0.0025, "crossbar needs z clearance in the slot"
        # -- the shaft spins freely inside the slot (half-diagonal vs slot half-height)
        assert self.shaft_h * math.sqrt(2.0) <= self.slot_hh - 0.001, \
            "shaft must rotate freely in the slot"
        # -- interlock: the VERTICAL crossbar cannot pass back (>= 10 mm bearing per side)
        assert self.cross_hy - self.slot_hh >= 0.010, "locked crossbar must bear >= 10 mm per side"
        # -- lock engages early: at 15 deg roll the crossbar can no longer escape ...
        th = math.radians(15.0)
        assert self.cross_hy * math.sin(th) + self.cross_hz * math.cos(th) > self.slot_hh + 0.002, \
            "crossbar must be captive beyond 15 deg"
        # -- ... but at <= 4 deg it passes (insertion/extraction at ~zero roll is possible)
        th4 = math.radians(4.0)
        assert self.cross_hy * math.sin(th4) + self.cross_hz * math.cos(th4) <= self.slot_hh - 0.0008, \
            "near-flat crossbar must still pass the slot"
        # -- the knob can never pass the slot (the key cannot be lost inside)
        assert self.knob_h >= self.slot_hh + 0.004, "knob must never fit the slot"
        # -- the slab is too thick to reach through and hook (Franka fingertip ~23 mm)
        assert self.panel_t >= 0.028, "slab must defeat fingertip hooking through the slot"
        # -- no purchase on the slab: flush face, small uniform side gap
        assert 0.003 <= self.side_y0 - self.front_hy <= 0.006, "side gap must be a few mm"
        assert 0.001 <= self.front_z0 - self.base_z1 <= 0.004, "bottom gap must be small"
        assert 0.001 <= self.shelf_z0 - self.front_z1 <= 0.004, "top gap must be small"
        # -- slots sit inside their panels
        assert self.slot_z - self.slot_hh >= self.front_z0 + 0.02
        assert self.slot_z + self.slot_hh <= self.front_z1 - 0.02
        assert self.slot_hw <= self.front_hy - 0.03
        assert self.decoy_z - self.slot_hh >= self.shelf_z1 + 0.02
        assert self.decoy_z + self.slot_hh <= self.top_z0 - 0.02
        # -- keyed depth: at key_ins_x (key yawed pi) the crossbar is FULLY behind the slab
        cross_off = (self.cross_x0 + self.cross_x1) / 2
        cross_front = self.key_ins_x - cross_off + self.cross_hz
        assert cross_front <= -(self.panel_t + self.keyed_x_pen) - 0.002, \
            "insertion must put the crossbar clearly behind the slab"
        # -- and the knob stays clearly outside the cabinet face
        assert self.key_ins_x - self.knob_x1 >= 0.010, "knob must stay outside at full insertion"
        # -- rotation clearance: the crossbar's swing circle stays inside the drawer box
        r = math.hypot(self.cross_hy, self.cross_hz)
        assert self.slot_z - r >= self.front_z0 + 0.010 + 0.005, "swing must clear the drawer floor"
        assert self.slot_z + r <= self.wall_z1 - 0.005, "swing must stay under the wall tops"
        assert r <= self.front_hy - 0.015 - 0.02, "swing must clear the side walls"
        assert self.key_ins_x - cross_off - self.cross_hz >= self.body_x0 + 0.020, \
            "crossbar must stay clear of the rear wall"
        # -- pull kinematics: crossbar-on-slab contact drives the drawer past the goal
        open_out = self.key_pull_x - cross_off + self.cross_hz + self.panel_t
        assert open_out >= self.open_goal + 0.010, "full pull must clear the goal with margin"
        assert self.drawer_stroke >= open_out + 0.020, "outer stop must not clip the pull"
        assert self.crack_open < self.open_goal < open_out, "latch ladder must be climbable"
        # -- decoy discrimination: the keyed predicate's z window excludes the decoy slot
        assert self.decoy_z - self.slot_z >= self.keyed_z_tol + 0.05, \
            "decoy slot must be far outside the keyed z-window"
        # -- key spawns on the open floor, clear of the cabinet face
        assert self.key_x[0] >= 0.095 + 0.10, "key spawn band must be clear of the cabinet"
        assert self.key_x[1] > self.key_x[0]
        # -- distractor bands never overlap (bowl and plate cannot spawn intersecting)
        assert self.plate_x[0] - self.bowl_x[1] >= self.bowl_r + self.plate_r + 0.02
        assert self.bowl_x[0] - self.bowl_r >= self.back_x1
        assert self.plate_x[1] + self.plate_r <= 0.0
        # -- rubric weights
        assert abs(self.w_keyed + self.w_locked + self.w_crack - 0.60) < 1e-9


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("bayonet_drawer")
class BayonetDrawerScene(BaseScene):
    cfg: BayonetDrawerSceneCfg

    def __init__(self, cfg: BayonetDrawerSceneCfg | None = None) -> None:
        super().__init__(cfg or BayonetDrawerSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        cab_spawn = cls["cabinet"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            side_y0=c.side_y0, side_y1=c.side_y1, back_x0=c.back_x0, back_x1=c.back_x1,
            base_z1=c.base_z1, shelf_z0=c.shelf_z0, shelf_z1=c.shelf_z1,
            top_z0=c.top_z0, top_z1=c.top_z1, panel_t=c.panel_t, decoy_z=c.decoy_z,
            slot_hw=c.slot_hw, slot_hh=c.slot_hh, contact_offset=c.contact_offset)
        drw_spawn = cls["drawer"](
            panel_t=c.panel_t, front_hy=c.front_hy, front_z0=c.front_z0, front_z1=c.front_z1,
            slot_z=c.slot_z, slot_hw=c.slot_hw, slot_hh=c.slot_hh, body_x0=c.body_x0,
            wall_z1=c.wall_z1, mass=c.drawer_mass, contact_offset=c.contact_offset,
            slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)
        key_spawn = cls["key"](
            cross_x0=c.cross_x0, cross_x1=c.cross_x1, cross_hy=c.cross_hy, cross_hz=c.cross_hz,
            shaft_x0=c.shaft_x0, shaft_h=c.shaft_h, knob_x0=c.knob_x0, knob_x1=c.knob_x1,
            knob_h=c.knob_h, mass=c.key_mass, contact_offset=c.contact_offset,
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
            # The drawer is authored IN PLACE at its closed pose: the bind-time
            # joint anchors at this authored position (i62/i87 pattern).
            "drawer": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Drawer", spawn=drw_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "key": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Key", spawn=key_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.42, 0.0, c.key_z0))),
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
        from isaaclab.utils.math import quat_apply

        self._quat_apply = quat_apply
        self.cabinet: RigidObject = env.iscene["cabinet"]
        self.drawer: RigidObject = env.iscene["drawer"]
        self.key: RigidObject = env.iscene["key"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.plate: RigidObject = env.iscene["plate"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke): sampled key spawn (x, y, yaw)
        self.k0 = torch.zeros(n, 3, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._keyed_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._locked_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._crack_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._author_joints()

    def _author_joints(self) -> None:
        """Per-env drawer joint, authored ONCE at bind time against the AUTHORED
        poses (drawer root coincides with the cabinet root). The joint LIMITS are
        the drawer hard stops; joint collision filtering only disables the
        cabinet<->drawer pair, so key<->drawer and key<->cabinet contacts — the
        bayonet coupling — still collide."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        c = self.cfg
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/drawer_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Cabinet"])
            j.CreateBody1Rel().SetTargets([f"{base}/Drawer"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(0.0)
            j.CreateUpperLimitAttr(float(c.drawer_stroke))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: cabinet re-asserted at its fixed pose (kinematic joint
        anchors are world-fixed — the cabinet must never move), drawer closed,
        key at a SAMPLED floor pose (x, y, yaw), bowl and plate at sampled poses
        on the cabinet top, latches cleared."""
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
        place(self.drawer, zeros, zeros, zeros)

        u = torch.rand(m, 3, device=dev)
        kx = c.key_x[0] + u[:, 0] * (c.key_x[1] - c.key_x[0])
        ky = (u[:, 1] * 2 - 1) * c.key_y
        kyaw = (u[:, 2] * 2 - 1) * math.pi
        self.k0[env_ids, 0] = kx
        self.k0[env_ids, 1] = ky
        self.k0[env_ids, 2] = kyaw
        s = torch.zeros(m, 13, device=dev)
        s[:, 0] = origin[:, 0] + kx
        s[:, 1] = origin[:, 1] + ky
        s[:, 2] = origin[:, 2] + c.key_z0
        s[:, 3] = torch.cos(kyaw / 2)
        s[:, 6] = torch.sin(kyaw / 2)
        self.key.write_root_state_to_sim(s, env_ids)

        v = torch.rand(m, 4, device=dev)
        bx = c.bowl_x[0] + v[:, 0] * (c.bowl_x[1] - c.bowl_x[0])
        by = (v[:, 1] * 2 - 1) * c.topper_y
        place(self.bowl, bx, by, zeros + c.top_z1 + c.bowl_h / 2 + 0.002)
        px = c.plate_x[0] + v[:, 2] * (c.plate_x[1] - c.plate_x[0])
        py = (v[:, 3] * 2 - 1) * c.topper_y
        place(self.plate, px, py, zeros + c.top_z1 + c.plate_h / 2 + 0.002)

        self._keyed_l[env_ids] = False
        self._locked_l[env_ids] = False
        self._crack_l[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cabinet": self.cabinet.data.root_state_w[env_ids].clone(),
            "drawer": self.drawer.data.root_state_w[env_ids].clone(),
            "key": self.key.data.root_state_w[env_ids].clone(),
            "bowl": self.bowl.data.root_state_w[env_ids].clone(),
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "k0": self.k0[env_ids].clone(),
            "keyed": self._keyed_l[env_ids].clone(),
            "locked": self._locked_l[env_ids].clone(),
            "crack": self._crack_l[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cabinet.write_root_state_to_sim(state["cabinet"], env_ids)
        self.drawer.write_root_state_to_sim(state["drawer"], env_ids)
        self.key.write_root_state_to_sim(state["key"], env_ids)
        self.bowl.write_root_state_to_sim(state["bowl"], env_ids)
        self.plate.write_root_state_to_sim(state["plate"], env_ids)
        self.k0[env_ids] = state["k0"]
        self._keyed_l[env_ids] = state["keyed"]
        self._locked_l[env_ids] = state["locked"]
        self._crack_l[env_ids] = state["crack"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden KITCHEN CABINET stands on the ground, {c.top_z1 * 100:.0f} cm "
            f"tall, with a black bowl and a white plate sitting on its top. Its "
            f"bottom drawer is shut and HAS NO HANDLE: the front is a smooth grey "
            f"slab, {c.panel_t * 1000:.0f} mm thick, flush with the cabinet face, "
            f"with only a {(c.side_y0 - c.front_hy) * 1000:.0f} mm perimeter gap — "
            f"nothing to hook, pinch, or press, and the slab is too thick to reach "
            f"through. The only opening in it is a horizontal SLOT "
            f"({2 * c.slot_hw * 100:.1f} x {2 * c.slot_hh * 100:.1f} cm) at "
            f"{c.slot_z * 100:.0f} cm height. A visually identical DECOY slot sits "
            f"higher up at {c.decoy_z * 100:.1f} cm — but that one is cut in a "
            f"FIXED panel: keying it moves nothing.\n"
            f"On the floor in front of the cabinet lies a free T-KEY (pose varies "
            f"by episode): a red crossbar ({2 * c.cross_hy * 100:.1f} cm long, "
            f"{2 * c.cross_hz * 1000:.0f} mm thick) on a thin steel shaft with a "
            f"blue grip knob. Held FLAT, the crossbar passes through the drawer's "
            f"slot; TWISTED ~90 degrees once inside, the crossbar stands taller "
            f"than the slot and can no longer come out — a bayonet interlock. "
            f"PULL the twisted key and the crossbar bears on the slab's inner "
            f"face, dragging the drawer open. Pulling an untwisted key just "
            f"slides it back out; the knob never fits through the slot.\n"
            f"Goal: the drawer open at least {c.open_goal * 100:.0f} cm, "
            f"everything at rest. The bowl and the plate are bystanders — leave "
            f"them where they sit."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Open the cabinet's handle-less bottom drawer using the T-key on the "
            "floor: insert the flat crossbar through the drawer's lower slot, "
            "twist the key about 90 degrees to lock it behind the front slab, "
            "then pull the key to drag the drawer open. Ignore the identical "
            "decoy slot in the fixed panel above. Finish with the drawer open "
            "and everything at rest."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _local(self, body) -> torch.Tensor:
        """(N,3) body root position in the cabinet frame (cabinet is fixed at the
        env origin with identity heading)."""
        return body.data.root_pos_w - self.env_origins

    def drawer_open(self) -> torch.Tensor:
        """(N,) drawer opening (its prismatic coordinate; 0 = flush/closed)."""
        return self._local(self.drawer)[:, 0]

    def cross_local(self) -> torch.Tensor:
        """(N,3) crossbar CENTRE position in the cabinet frame."""
        c = self.cfg
        off = torch.zeros(self.env.num_envs, 3, device=self.env.device)
        off[:, 0] = (c.cross_x0 + c.cross_x1) / 2
        w = self.key.data.root_pos_w + self._quat_apply(self.key.data.root_quat_w, off)
        return w - self.env_origins

    def cross_tilt(self) -> torch.Tensor:
        """(N,) |crossbar long axis . world z| — 0 flat (passes slot), 1 vertical."""
        ey = torch.zeros(self.env.num_envs, 3, device=self.env.device)
        ey[:, 1] = 1.0
        return self._quat_apply(self.key.data.root_quat_w, ey)[:, 2].abs()

    def keyed(self) -> torch.Tensor:
        """(N,) bool: the crossbar centre sits fully BEHIND the drawer's front
        slab, inside the drawer box at slot height (i.e. it entered through the
        REAL slot — the decoy slot fails the z-window)."""
        c = self.cfg
        p = self.cross_local()
        x_rel = p[:, 0] - self.drawer_open()
        return (x_rel < -(c.panel_t + c.keyed_x_pen)) \
            & (p[:, 1].abs() < c.keyed_y_tol) \
            & ((p[:, 2] - c.slot_z).abs() < c.keyed_z_tol) & self._finite()

    def locked(self) -> torch.Tensor:
        """(N,) bool: keyed AND twisted past 45 deg — the bayonet interlock is
        engaged (the 46 mm crossbar cannot pass the 16 mm slot)."""
        return self.keyed() & (self.cross_tilt() >= self.cfg.lock_cos)

    def decoy_keyed(self) -> torch.Tensor:
        """(N,) bool (diagnostic): crossbar behind the FIXED decoy panel."""
        c = self.cfg
        p = self.cross_local()
        return (p[:, 0] < -(c.panel_t + c.keyed_x_pen)) \
            & (p[:, 1].abs() < c.keyed_y_tol) \
            & ((p[:, 2] - c.decoy_z).abs() < c.keyed_z_tol)

    def settled(self) -> torch.Tensor:
        """(N,) bool: drawer, key, distractors slow; key rotation slow."""
        c = self.cfg
        return (self.drawer.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.key.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.key.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang) \
            & (self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.plate.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.drawer.data.root_pos_w, self.key.data.root_pos_w,
                         self.bowl.data.root_pos_w, self.plate.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        fin = self._finite()
        self._keyed_l |= self.keyed() & fin
        self._locked_l |= self.locked() & fin
        self._crack_l |= (self.drawer_open() >= self.cfg.crack_open) & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the drawer stands open at least `open_goal`, with
        everything settled and finite — a LIVE physical outcome. The slab offers
        no purchase and the joint bottoms out inward, so the only physical route
        to this state is the bayonet coupling: key in the real slot, twisted,
        pulled."""
        self._update_latches()
        return (self.drawer_open() >= self.cfg.open_goal) & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15 keyed + 0.20 locked + 0.25 cracked (>=3 cm),
        all latched, capped at 0.60; exactly 1.0 iff success() holds live. Doing
        nothing scores ~0; the seed's strategy (pull the drawer itself) finds no
        purchase and scores ~0; an untwisted-key pull tops out at 0.15; keying
        the decoy slot scores 0 (the keyed z-window excludes it)."""
        c = self.cfg
        self._update_latches()
        base = (c.w_keyed * self._keyed_l.float() + c.w_locked * self._locked_l.float()
                + c.w_crack * self._crack_l.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="bayonet_drawer", robot="null"))
