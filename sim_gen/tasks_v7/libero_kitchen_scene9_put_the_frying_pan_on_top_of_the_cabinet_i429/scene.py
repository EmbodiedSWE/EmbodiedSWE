"""RatchetPortcullisScene — deliver the frying pan INTO a roofed pantry hutch
whose only doorway is barred by a gravity-closed portcullis gate held open only
by its one-way ratchet (sim_gen task
`libero_kitchen_scene9_put_the_frying_pan_on_top_of_the_cabinet_i429`).

Derived from libero_90/libero_kitchen_scene9_put_the_frying_pan_on_top_of_the_cabinet,
but STRATEGICALLY different. The seed's whole skill is grasp -> transport ->
release: pick the frypan off the table and set it down on an OPEN top surface
(a shelf top with nothing in the way). Here the goal surface is INSIDE a fully
roofed, sealed hutch — placing the pan ON TOP of the structure (the seed's
strategy, literally) scores nothing, and smoke proves it. The only opening is a
low front doorway barred by a portcullis: a vertical sliding gate that gravity
drives shut. The gate cannot be simply held open while the pan is carried
through, because the robot has one arm — the task is solved by the hutch's own
RATCHET: a toothed rack on the gate's front face clicks past a gravity-biased
pawl on the way UP, and the pawl jams each tooth on the way DOWN, so every lift
stroke is retained hands-off. Plan: (1) lift the gate by its bar — the ratchet
latches it open; (2) with both hands free again, slide the pan along the floor
through the doorway onto the stove pad inside; (3) let everything settle. The
plan is order-forced by geometry (roof kills drop-in; the closed gate kills
slide-in) and the delivery is certified by a TRANSIT credential observed only
when the pan physically crosses the doorway plane under a raised gate.

No stored energy is created by success: the gate rests on the pawl (a hard
revolute stop takes the load), the pan rests on the floor — the outcome
persists hands-off indefinitely.

Assets are fully procedural (compound-spawner pattern; children of one body
never self-collide): hutch (KINEMATIC: side walls, back wall, lintel + front
piers, full roof, pawl bracket, and a collider-LESS flush stove-pad marker),
gate (DYNAMIC: panel + 6-tooth rack + lift bar) on a per-env vertical prismatic
joint (limits = hard stops), pawl (DYNAMIC flap) on a per-env revolute joint
with asymmetric limits (down-jam at -3 deg, free swing to +75 deg), skillet
(DYNAMIC: disc + handle) and a bowl distractor that sits ON THE ROOF — exactly
where the seed's strategy would put things. Joint collision filtering applies
only to the hutch<->follower pairs, so gate<->pawl ratchet contact — the
mechanism — still collides. Rack and pawl carry a polished material so teeth
click past the pawl instead of dragging it.

Per-episode randomization (readback-verified by smoke): the pan's start pose
(x, y, yaw — yaw restricted so the handle can trail through the doorway) and
the bowl's xy on the roof.

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.20  opened  — gate ever lifted >= e_open (8 cm)                    (latched)
  0.20  propped — gate open and still for >= prop_steps substeps       (latched)
  0.25  transit — pan crossed the doorway plane under a raised gate    (latched)
capped at 0.65; exactly 1.0 iff success(): transit AND the pan rests upright on
the stove pad inside, everything settled and finite. Null policy ~0 (gravity
keeps the gate shut). The seed's strategy — set the pan on top of the structure
— fires no latch and scores ~0.

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


def _span(stage, path: str, *, x, y, z, color, collide: Callable | None):
    """Box child from axis spans (x0, x1), (y0, y1), (z0, z1). collide=None => visual only."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2))
    xf.AddScaleOp().Set(Gf.Vec3f(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
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


def _spawn_hutch(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the hutch at `prim_path`: KINEMATIC compound. Local frame: x=0 is
    the front face plane (+x = out of the hutch, toward the robot), z=0 the
    ground. The only opening is the front doorway (|y| < door_hw, z < door_h);
    everything else — sides, back, lintel, full roof — is sealed. A pawl
    bracket cheek plate juts forward beside the flap (offset +y, clear of the
    flap's whole swing); the stove-pad marker on the interior floor is VISUAL
    ONLY (no collider: nothing to wall a slid pan)."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/side_p", x=(c.back_x0, 0.0), y=(c.side_y0, c.side_y1),
          z=(0.0, c.roof_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/side_n", x=(c.back_x0, 0.0), y=(-c.side_y1, -c.side_y0),
          z=(0.0, c.roof_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/back", x=(c.back_x0, c.back_x1), y=(-c.side_y0, c.side_y0),
          z=(0.0, c.roof_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/pier_p", x=(c.front_x0, 0.0), y=(c.door_hw, c.side_y1),
          z=(0.0, c.roof_z0), color=c.trim_color, collide=collide)
    _span(stage, f"{prim_path}/pier_n", x=(c.front_x0, 0.0), y=(-c.side_y1, -c.door_hw),
          z=(0.0, c.roof_z0), color=c.trim_color, collide=collide)
    _span(stage, f"{prim_path}/lintel", x=(c.front_x0, 0.0), y=(-c.door_hw, c.door_hw),
          z=(c.door_h, c.roof_z0), color=c.trim_color, collide=collide)
    _span(stage, f"{prim_path}/roof", x=(c.back_x0, 0.0), y=(-c.side_y1, c.side_y1),
          z=(c.roof_z0, c.roof_z1), color=c.roof_color, collide=collide)
    # pawl mount: a SIDE cheek plate, offset +y of the flap band with 1 mm
    # clearance, z spanning the hinge. It must never enter the flap's swept
    # x-z volume: the pawl and hutch are a jointed pair, and any spawn
    # interpenetration rigidly locks the GPU revolute despite the filter.
    _span(stage, f"{prim_path}/bracket", x=(-0.02, c.hinge_x + 0.015),
          y=(c.rack_y1 + 0.001, c.rack_y1 + 0.016),
          z=(c.hinge_z - 0.016, c.hinge_z + 0.016),
          color=c.trim_color, collide=collide)
    # stove-pad goal marker: flush, VISUAL ONLY (no collider, no step to wall a pan)
    _span(stage, f"{prim_path}/pad", x=(c.pad_x - c.pad_hx, c.pad_x + c.pad_hx),
          y=(c.pad_y - c.pad_hy, c.pad_y + c.pad_hy), z=(0.0002, 0.0012),
          color=c.pad_color, collide=None)
    wood = _mk_material(prim_path, "wood", c.wood_mu_s, c.wood_mu_d, "average")
    bind_physics_material(prim_path, wood)
    return root


def _spawn_gate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the portcullis gate at its CLOSED pose, root origin = hutch origin
    (so the vertical prismatic joint coordinate IS the lift). Panel in front of
    the doorway + a 6-tooth rack on its front face (the ratchet's rack, offset
    to the +y side, outside the aperture) + a lift bar the gripper can hook."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/panel", x=(c.panel_x0, c.panel_x1),
          y=(-c.panel_hy, c.panel_hy), z=(c.panel_z0, c.panel_z1),
          color=c.panel_color, collide=collide)
    for k in range(c.n_teeth):
        t0 = c.rack_z0 + k * c.pitch
        _span(stage, f"{prim_path}/tooth_{k}", x=(c.panel_x1, c.tooth_x1),
              y=(c.rack_y0, c.rack_y1), z=(t0, t0 + c.tooth_h),
              color=c.tooth_color, collide=collide)
    _span(stage, f"{prim_path}/bar", x=(c.panel_x1, c.bar_x1),
          y=(-c.bar_hy, c.bar_hy), z=(c.bar_z0, c.bar_z1),
          color=c.bar_color, collide=collide)
    _dyn_body(root, c.mass, c.lin_damp, c.ang_damp)
    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    bind_physics_material(prim_path, slick)
    return root


def _spawn_pawl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the pawl flap with root origin at its GEOMETRIC CENTER (so the
    MassAPI CoM-at-origin quirk puts the CoM there — 17 mm on the tip side of
    the hinge, which is what gravity-biases the flap tip-down into the rack).
    The revolute anchor sits at pawl-local (+hinge_off, 0, 0)."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/flap", x=(-c.half_l, c.half_l), y=(-c.half_w, c.half_w),
          z=(-c.half_t, c.half_t), color=c.flap_color, collide=collide)
    _dyn_body(root, c.mass, 0.05, c.ang_damp)
    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    bind_physics_material(prim_path, slick)
    return root


def _spawn_skillet(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the skillet: flat disc (cylinder) + straight handle along +x.
    Root frame: z=0 at the pan's bottom plane, origin on the disc axis."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import Gf, UsdGeom

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    disc = UsdGeom.Cylinder.Define(stage, f"{prim_path}/disc")
    disc.CreateAxisAttr("Z")
    disc.CreateRadiusAttr(float(c.disc_r))
    disc.CreateHeightAttr(float(c.disc_h))
    UsdGeom.Xformable(disc.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, c.disc_h / 2))
    disc.CreateDisplayColorAttr([Gf.Vec3f(*c.disc_color)])
    disc.CreateExtentAttr([Gf.Vec3f(-c.disc_r, -c.disc_r, -c.disc_h / 2),
                           Gf.Vec3f(c.disc_r, c.disc_r, c.disc_h / 2)])
    collide(disc.GetPrim())
    _span(stage, f"{prim_path}/handle", x=(c.disc_r - 0.008, c.handle_x1),
          y=(-c.handle_hy, c.handle_hy), z=(c.handle_z0, c.handle_z1),
          color=c.handle_color, collide=collide)
    _dyn_body(root, c.mass, c.lin_damp, c.ang_damp)
    mat = _mk_material(prim_path, "steel", c.mu_s, c.mu_d, "average")
    bind_physics_material(prim_path, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "hutch" not in _SPAWNER_CACHE:

        @configclass
        class HutchSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_hutch)
            back_x0: float = -0.37
            back_x1: float = -0.34
            side_y0: float = 0.13
            side_y1: float = 0.16
            front_x0: float = -0.03
            door_hw: float = 0.10
            door_h: float = 0.13
            roof_z0: float = 0.23
            roof_z1: float = 0.26
            hinge_x: float = 0.056
            hinge_z: float = 0.252
            rack_y0: float = 0.095
            rack_y1: float = 0.135
            pad_x: float = -0.24
            pad_y: float = 0.0
            pad_hx: float = 0.06
            pad_hy: float = 0.06
            body_color: tuple = (0.52, 0.38, 0.24)
            trim_color: tuple = (0.44, 0.31, 0.19)
            roof_color: tuple = (0.38, 0.27, 0.17)
            pad_color: tuple = (0.85, 0.25, 0.10)
            contact_offset: float = 0.0015
            wood_mu_s: float = 0.60
            wood_mu_d: float = 0.55

        @configclass
        class GateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gate)
            panel_x0: float = 0.004
            panel_x1: float = 0.016
            panel_hy: float = 0.13
            panel_z0: float = 0.005
            panel_z1: float = 0.170
            n_teeth: int = 5
            rack_z0: float = 0.030
            pitch: float = 0.028
            tooth_h: float = 0.008
            tooth_x1: float = 0.030
            rack_y0: float = 0.095
            rack_y1: float = 0.135
            bar_x1: float = 0.048
            bar_hy: float = 0.05
            bar_z0: float = 0.058
            bar_z1: float = 0.078
            mass: float = 0.30
            lin_damp: float = 2.0
            ang_damp: float = 2.0
            panel_color: tuple = (0.30, 0.34, 0.40)
            tooth_color: tuple = (0.75, 0.62, 0.20)
            bar_color: tuple = (0.78, 0.16, 0.12)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.10
            slide_mu_d: float = 0.08

        @configclass
        class PawlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pawl)
            half_l: float = 0.021
            half_w: float = 0.020
            half_t: float = 0.0045
            mass: float = 0.05
            # a greased pivot: heavy rate damping kills contact chatter while the
            # tip carries the gate at the jam stop (gravity re-seat torque is
            # ~370 rad/s^2 of accel — damping never keeps the pawl from seating)
            ang_damp: float = 8.0
            flap_color: tuple = (0.85, 0.45, 0.10)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.10
            slide_mu_d: float = 0.08

        @configclass
        class SkilletSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_skillet)
            disc_r: float = 0.070
            disc_h: float = 0.022
            handle_x1: float = 0.185
            handle_hy: float = 0.011
            handle_z0: float = 0.010
            handle_z1: float = 0.026
            mass: float = 0.40
            lin_damp: float = 0.5
            ang_damp: float = 1.0
            disc_color: tuple = (0.12, 0.12, 0.14)
            handle_color: tuple = (0.20, 0.20, 0.22)
            contact_offset: float = 0.0015
            mu_s: float = 0.35
            mu_d: float = 0.30

        _SPAWNER_CACHE["hutch"] = HutchSpawnerCfg
        _SPAWNER_CACHE["gate"] = GateSpawnerCfg
        _SPAWNER_CACHE["pawl"] = PawlSpawnerCfg
        _SPAWNER_CACHE["skillet"] = SkilletSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RatchetPortcullisSceneCfg(BaseCfg):
    """Config for `RatchetPortcullisScene`. The mechanism contract is asserted
    in `__post_init__`: the ratchet really has pawl-held gate heights above the
    "opened" threshold and below the joint stop, the pawl really overlaps the
    teeth and fits their gaps, the closed gate really seals the doorway, the
    doorway really passes the pan at every sampled yaw, and the pan really fits
    the interior at the pad."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    e_open: float = tunable(0.080)        # gate lift for the "opened" latch (m)
    prop_steps: int = tunable(60)         # consecutive open+still substeps -> "propped" (0.5 s)
    pan_clear: float = tunable(0.060)     # min gate lift for transit credit (m)
    pad_tol_x: float = tunable(0.060)     # pan-on-pad |dx| tolerance (m)
    pad_tol_y: float = tunable(0.050)     # pan-on-pad |dy| tolerance (m)
    pan_z_lo: float = tunable(-0.005)     # pan root z band on the pad (m)
    pan_z_hi: float = tunable(0.025)
    upright_min: float = tunable(0.90)    # pan body z-axis . world z
    settle_lin: float = tunable(0.05)     # max CoM |lin vel| of every mover when judging (m/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    pan_x: tuple = tunable((0.26, 0.34))  # pan start x band (m, in front of the hutch)
    pan_y: float = tunable(0.08)          # +/- pan start y jitter (m)
    pan_yaw: float = tunable(0.30)        # +/- pan start yaw (rad; keeps the handle trailable)
    bowl_x: tuple = tunable((-0.30, -0.10))  # bowl x band ON THE ROOF
    bowl_y: float = tunable(0.06)         # +/- bowl y jitter on the roof

    # --- info: hutch (local frame: x=0 front face, +x = toward the robot, z=0 ground) ------------
    back_x0: float = info(-0.37)
    back_x1: float = info(-0.34)          # interior rear face
    side_y0: float = info(0.13)           # interior side face
    side_y1: float = info(0.16)
    front_x0: float = info(-0.03)         # front wall inner face
    door_hw: float = info(0.10)           # doorway half-width
    door_h: float = info(0.13)            # doorway height (lintel underside)
    roof_z0: float = info(0.23)           # interior ceiling
    roof_z1: float = info(0.26)           # roof top (the bowl stands here)
    # --- info: stove pad (interior floor, flush, visual-only) ------------------------------------
    pad_x: float = info(-0.24)
    pad_y: float = info(0.0)
    pad_hx: float = info(0.06)
    pad_hy: float = info(0.06)
    # --- info: gate (authored CLOSED; prismatic-Z joint coordinate = lift; limits = stops) -------
    gate_hi: float = info(0.175)          # prismatic upper limit (m)
    panel_x0: float = info(0.004)
    panel_x1: float = info(0.016)
    panel_hy: float = info(0.13)
    panel_z0: float = info(0.005)
    panel_z1: float = info(0.170)
    bar_x1: float = info(0.048)
    bar_z0: float = info(0.058)
    bar_z1: float = info(0.078)
    gate_mass: float = info(0.30)
    # --- info: ratchet rack (on the gate front face, +y side, outside the aperture) --------------
    n_teeth: int = info(5)
    rack_z0: float = info(0.030)          # lowest tooth underside (gate frame)
    pitch: float = info(0.028)
    tooth_h: float = info(0.008)
    tooth_x1: float = info(0.030)         # tooth outer face
    rack_y0: float = info(0.095)
    rack_y1: float = info(0.135)
    # --- info: pawl (root at flap center; hinge at root-local +hinge_off) ------------------------
    hinge_x: float = info(0.056)
    hinge_z: float = info(0.252)
    pawl_half_l: float = info(0.021)
    pawl_half_t: float = info(0.0045)
    hinge_off: float = info(0.017)        # hinge is this far toward the heel from the center
    pawl_mass: float = info(0.05)
    pawl_down_deg: float = info(3.0)      # jam-side limit (down past horizontal)
    pawl_up_deg: float = info(75.0)       # free-swing limit (teeth click past)
    # --- info: skillet -------------------------------------------------------------------------
    disc_r: float = info(0.070)
    disc_h: float = info(0.022)
    handle_x1: float = info(0.185)        # handle tip x in the pan frame
    handle_hy: float = info(0.011)
    pan_mass: float = info(0.40)
    # --- info: bowl distractor (on the roof) -----------------------------------------------------
    bowl_r: float = info(0.045)
    bowl_h: float = info(0.050)
    bowl_mass: float = info(0.15)
    # --- info: transit credential slab (doorway tunnel + just-inside strip) ----------------------
    slab_x0: float = info(-0.09)
    slab_x1: float = info(0.0)
    slab_z1: float = info(0.10)
    # --- info: materials -------------------------------------------------------------------------
    contact_offset: float = info(0.0015)
    slide_mu_s: float = info(0.10)        # rack/pawl polished: teeth click, don't drag
    slide_mu_d: float = info(0.08)
    # --- info: rubric weights (0.20 + 0.20 + 0.25 = 0.65 = the non-success cap) ------------------
    w_open: float = info(0.20)
    w_prop: float = info(0.20)
    w_transit: float = info(0.25)

    def __post_init__(self) -> None:
        # -- ratchet statics: pawl tip catch height and the hold ladder -------------------------
        tip_len = self.pawl_half_l + self.hinge_off       # hinge -> tip
        droop = tip_len * math.sin(math.radians(self.pawl_down_deg))
        z_catch = self.hinge_z + self.pawl_half_t - droop  # tooth underside rests here
        holds = [z_catch - (self.rack_z0 + k * self.pitch) for k in range(self.n_teeth)]
        holds = [e for e in holds if 0.0 < e < self.gate_hi]
        assert holds, "the rack must offer at least one pawl-held lift"
        # every pawl-held height already passes the pan and fires the "opened" latch
        assert min(holds) >= self.e_open + 0.010, \
            "the lowest ratchet hold must clear the opened threshold with margin"
        assert min(holds) >= self.pan_clear + 0.030, "every hold must pass the pan"
        # lifting to the hard stop then releasing drops onto a real catch (small fall)
        drop = self.gate_hi - max(holds)
        assert 0.003 <= drop <= self.pitch, \
            "release from the stop must fall a few mm onto the top catch"
        # at the hard stop the pawl sits INSIDE a tooth gap (not grazing a tooth)
        pawl_lo = self.hinge_z - self.pawl_half_t - droop
        pawl_hi = self.hinge_z + self.pawl_half_t
        for k in range(self.n_teeth):
            t0 = self.rack_z0 + k * self.pitch + self.gate_hi
            assert t0 >= pawl_hi + 0.004 or t0 + self.tooth_h <= pawl_lo - 0.004, \
                f"tooth {k} must clear the pawl band at the hard stop"
        # pawl tip overlaps the teeth by >= 8 mm but clears the panel face by >= 1 mm
        tip_x = self.hinge_x - tip_len
        assert self.tooth_x1 - tip_x >= 0.008, "pawl must overlap the teeth"
        assert tip_x >= self.panel_x1 + 0.001, "pawl tip must clear the panel face"
        # pawl fits the tooth gap with slack; swings clear of the rack at the up-limit
        assert (self.pitch - self.tooth_h) - 2 * self.pawl_half_t >= 0.005, \
            "tooth gaps must swallow the pawl"
        up_x = self.hinge_x - tip_len * math.cos(math.radians(self.pawl_up_deg))
        assert up_x >= self.tooth_x1 + 0.005, "pawl at the up-limit must clear the teeth"
        # the flap's whole swept disc (radius to the tip-top corner) stays in front of
        # the hutch face plane x=0 — no hutch slab (roof, lintel, piers, all at x<=0)
        # can ever intersect the swing; the mount cheek is y-offset out of the flap
        # band. A jointed pair that interpenetrates at spawn rigidly locks the GPU
        # revolute despite the collision filter, so this must hold by construction.
        swing_r = math.hypot(tip_len, self.pawl_half_t)
        assert self.hinge_x - swing_r >= 0.002, \
            "the pawl's swept volume must stay clear of the hutch face plane"
        # -- gate seals the doorway when closed; never touches the ground ----------------------
        assert self.panel_hy >= self.door_hw + 0.02, "closed gate must overhang the piers"
        assert self.panel_z1 >= self.door_h + 0.03, "closed gate must overhang the lintel"
        assert 0.003 <= self.panel_z0 <= 0.008, "closed gate must hover just off the ground"
        # rack stays on the panel; the gate at rest is entirely below the resting pawl
        assert self.rack_z0 + (self.n_teeth - 1) * self.pitch + self.tooth_h <= self.panel_z1
        assert self.panel_z1 <= pawl_lo - 0.05, "closed gate must hang clear of the pawl"
        # -- doorway passability at every sampled pan yaw --------------------------------------
        half_span = self.handle_x1 * math.sin(self.pan_yaw) + self.handle_hy
        assert max(self.disc_r, half_span) <= self.door_hw - 0.015, \
            "the pan must fit the doorway at every sampled yaw"
        pan_top = max(self.disc_h, 0.026)
        assert self.pan_clear - pan_top >= 0.030, "transit gate margin over the pan's height"
        assert self.door_h - pan_top >= 0.08, "the doorway must pass the pan with headroom"
        # -- the pan fits the interior at the pad ----------------------------------------------
        assert self.pad_x + self.handle_x1 <= self.front_x0 - 0.005, \
            "handle-trailing pan on the pad must clear the front wall"
        assert self.pad_x - self.disc_r >= self.back_x1 + 0.020, \
            "pan on the pad must clear the back wall"
        assert abs(self.pad_y) + self.pad_tol_y + self.disc_r <= self.side_y0, \
            "pan anywhere within tolerance must clear the side walls"
        # -- transit slab lies strictly in the sealed doorway/interior region ------------------
        assert self.slab_x1 <= self.panel_x0 and self.slab_x0 >= self.front_x0 - 0.08
        assert self.slab_z1 <= self.door_h - 0.02
        # -- randomization bands: pan spawns clear of the gate; bowl stays on the roof ---------
        assert self.pan_x[0] - self.disc_r >= self.bar_x1 + 0.03, \
            "the pan must spawn clear of the gate hardware"
        assert self.pan_x[1] > self.pan_x[0] and self.pan_yaw > 0.05
        assert self.bowl_x[0] - self.bowl_r >= self.back_x0 + 0.02
        assert self.bowl_x[1] + self.bowl_r <= -0.02
        assert self.bowl_y + self.bowl_r <= self.side_y1
        # -- weights ---------------------------------------------------------------------------
        assert abs(self.w_open + self.w_prop + self.w_transit - 0.65) < 1e-9


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("ratchet_portcullis")
class RatchetPortcullisScene(BaseScene):
    cfg: RatchetPortcullisSceneCfg

    def __init__(self, cfg: RatchetPortcullisSceneCfg | None = None) -> None:
        super().__init__(cfg or RatchetPortcullisSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        hutch_spawn = cls["hutch"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            back_x0=c.back_x0, back_x1=c.back_x1, side_y0=c.side_y0, side_y1=c.side_y1,
            front_x0=c.front_x0, door_hw=c.door_hw, door_h=c.door_h,
            roof_z0=c.roof_z0, roof_z1=c.roof_z1, hinge_x=c.hinge_x, hinge_z=c.hinge_z,
            rack_y0=c.rack_y0, rack_y1=c.rack_y1, pad_x=c.pad_x, pad_y=c.pad_y,
            pad_hx=c.pad_hx, pad_hy=c.pad_hy, contact_offset=c.contact_offset)
        gate_spawn = cls["gate"](
            panel_x0=c.panel_x0, panel_x1=c.panel_x1, panel_hy=c.panel_hy,
            panel_z0=c.panel_z0, panel_z1=c.panel_z1, n_teeth=c.n_teeth,
            rack_z0=c.rack_z0, pitch=c.pitch, tooth_h=c.tooth_h, tooth_x1=c.tooth_x1,
            rack_y0=c.rack_y0, rack_y1=c.rack_y1, bar_x1=c.bar_x1,
            bar_z0=c.bar_z0, bar_z1=c.bar_z1, mass=c.gate_mass,
            contact_offset=c.contact_offset, slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)
        pawl_spawn = cls["pawl"](
            half_l=c.pawl_half_l, half_t=c.pawl_half_t, mass=c.pawl_mass,
            contact_offset=c.contact_offset, slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)
        pan_spawn = cls["skillet"](
            disc_r=c.disc_r, disc_h=c.disc_h, handle_x1=c.handle_x1, handle_hy=c.handle_hy,
            mass=c.pan_mass, contact_offset=c.contact_offset)

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
            "hutch": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Hutch", spawn=hutch_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            # Gate and pawl are authored IN PLACE at their closed/rest poses: the
            # bind-time joints anchor at these authored positions.
            "gate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gate", spawn=gate_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "pawl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pawl", spawn=pawl_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.hinge_x - c.hinge_off, (c.rack_y0 + c.rack_y1) / 2, c.hinge_z))),
            "pan": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pan", spawn=pan_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.0, 0.002))),
            "bowl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl",
                spawn=sim_utils.CylinderCfg(
                    radius=c.bowl_r, height=c.bowl_h, axis="Z",
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
                    rigid_props=rigid, collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.50, dynamic_friction=0.45, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.95))),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.20, 0.0, c.roof_z1 + c.bowl_h / 2 + 0.002))),
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
        self.hutch: RigidObject = env.iscene["hutch"]
        self.gate: RigidObject = env.iscene["gate"]
        self.pawl: RigidObject = env.iscene["pawl"]
        self.pan: RigidObject = env.iscene["pan"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke)
        self.p0 = torch.zeros(n, 3, device=dev)       # pan spawn x, y, yaw
        self.b0 = torch.zeros(n, 2, device=dev)       # bowl spawn x, y (roof)
        # latches (partial credit survives transients; success needs the transit credential)
        self._opened = torch.zeros(n, dtype=torch.bool, device=dev)
        self._prop = torch.zeros(n, dtype=torch.bool, device=dev)
        self._prop_ctr = torch.zeros(n, dtype=torch.long, device=dev)
        self._transit = torch.zeros(n, dtype=torch.bool, device=dev)
        self._author_joints()

    def _author_joints(self) -> None:
        """Per-env joints, authored ONCE at bind time against the AUTHORED poses
        (the gate root coincides with the hutch root, so the prismatic-Z
        coordinate IS the lift; the pawl root sits `hinge_off` tip-ward of the
        hinge, so gravity biases the flap tip-down). Joint LIMITS are the hard
        stops; joint collision filtering disables only the hutch<->follower
        pairs, so gate<->pawl ratchet contact still collides."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        c = self.cfg
        hinge_y = (c.rack_y0 + c.rack_y1) / 2
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/gate_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Hutch"])
            j.CreateBody1Rel().SetTargets([f"{base}/Gate"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(0.0)
            j.CreateUpperLimitAttr(float(c.gate_hi))
            r = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/pawl_pivot")
            r.CreateBody0Rel().SetTargets([f"{base}/Hutch"])
            r.CreateBody1Rel().SetTargets([f"{base}/Pawl"])
            r.CreateCollisionEnabledAttr(False)
            r.CreateAxisAttr("Y")
            r.CreateLocalPos0Attr(Gf.Vec3f(float(c.hinge_x), hinge_y, float(c.hinge_z)))
            r.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            r.CreateLocalPos1Attr(Gf.Vec3f(float(c.hinge_off), 0.0, 0.0))
            r.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            # +angle about +y lifts the -x tip UP (the free ratchet direction);
            # -angle drops the tip into the jam stop. Degrees.
            r.CreateLowerLimitAttr(-float(c.pawl_down_deg))
            r.CreateUpperLimitAttr(float(c.pawl_up_deg))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: hutch re-asserted at its fixed pose (kinematic joint
        anchors are world-fixed — the hutch must never move), gate CLOSED, pawl
        at rest on its jam stop, pan at a SAMPLED outside pose (x, y, yaw), bowl
        at a sampled xy ON THE ROOF, latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def place(body, dx, dy, dz, yaw=None) -> None:
            s = torch.zeros(m, 13, device=dev)
            s[:, 0] = origin[:, 0] + dx
            s[:, 1] = origin[:, 1] + dy
            s[:, 2] = origin[:, 2] + dz
            if yaw is None:
                s[:, 3] = 1.0
            else:
                s[:, 3] = torch.cos(yaw / 2)
                s[:, 6] = torch.sin(yaw / 2)
            body.write_root_state_to_sim(s, env_ids)

        zeros = torch.zeros(m, device=dev)
        place(self.hutch, zeros, zeros, zeros)
        place(self.gate, zeros, zeros, zeros)
        place(self.pawl, zeros + (c.hinge_x - c.hinge_off),
              zeros + (c.rack_y0 + c.rack_y1) / 2, zeros + c.hinge_z)

        u = torch.rand(m, 5, device=dev)
        px = c.pan_x[0] + u[:, 0] * (c.pan_x[1] - c.pan_x[0])
        py = (u[:, 1] * 2 - 1) * c.pan_y
        pyaw = (u[:, 2] * 2 - 1) * c.pan_yaw
        self.p0[env_ids, 0] = px
        self.p0[env_ids, 1] = py
        self.p0[env_ids, 2] = pyaw
        place(self.pan, px, py, zeros + 0.002, yaw=pyaw)

        bx = c.bowl_x[0] + u[:, 3] * (c.bowl_x[1] - c.bowl_x[0])
        by = (u[:, 4] * 2 - 1) * c.bowl_y
        self.b0[env_ids, 0] = bx
        self.b0[env_ids, 1] = by
        place(self.bowl, bx, by, zeros + c.roof_z1 + c.bowl_h / 2 + 0.002)

        self._opened[env_ids] = False
        self._prop[env_ids] = False
        self._prop_ctr[env_ids] = 0
        self._transit[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "hutch": self.hutch.data.root_state_w[env_ids].clone(),
            "gate": self.gate.data.root_state_w[env_ids].clone(),
            "pawl": self.pawl.data.root_state_w[env_ids].clone(),
            "pan": self.pan.data.root_state_w[env_ids].clone(),
            "bowl": self.bowl.data.root_state_w[env_ids].clone(),
            "p0": self.p0[env_ids].clone(),
            "b0": self.b0[env_ids].clone(),
            "opened": self._opened[env_ids].clone(),
            "prop": self._prop[env_ids].clone(),
            "prop_ctr": self._prop_ctr[env_ids].clone(),
            "transit": self._transit[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.hutch.write_root_state_to_sim(state["hutch"], env_ids)
        self.gate.write_root_state_to_sim(state["gate"], env_ids)
        self.pawl.write_root_state_to_sim(state["pawl"], env_ids)
        self.pan.write_root_state_to_sim(state["pan"], env_ids)
        self.bowl.write_root_state_to_sim(state["bowl"], env_ids)
        self.p0[env_ids] = state["p0"]
        self.b0[env_ids] = state["b0"]
        self._opened[env_ids] = state["opened"]
        self._prop[env_ids] = state["prop"]
        self._prop_ctr[env_ids] = state["prop_ctr"]
        self._transit[env_ids] = state["transit"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A roofed wooden PANTRY HUTCH stands on the ground — sealed on every "
            f"side, {c.roof_z1 * 100:.0f} cm tall, with a white bowl sitting on its "
            f"flat roof. Its only opening is a low front DOORWAY "
            f"({2 * c.door_hw * 100:.0f} cm wide, {c.door_h * 100:.0f} cm tall) at "
            f"floor level, and that doorway is barred by a PORTCULLIS: a grey gate "
            f"panel riding a vertical slide, driven shut by its own weight. The "
            f"gate carries a RED LIFT BAR near its bottom edge and, up its front "
            f"face, a rack of gold RATCHET TEETH that runs under an orange gravity "
            f"PAWL mounted over the doorway: the teeth click freely past the pawl "
            f"as the gate RISES, and the pawl jams against a tooth the moment the "
            f"gate tries to FALL — every lift stroke is kept, hands-off, so the "
            f"gate can be walked open in strokes and stays open by itself (it only "
            f"catches once lifted past roughly two-thirds of its travel). A "
            f"flat BLACK SKILLET (a {2 * c.disc_r * 100:.0f} cm disc with a "
            f"straight handle) lies on the ground in front of the hutch (exact "
            f"spot and heading vary by episode). Inside, centred on the hutch "
            f"floor, an orange STOVE PAD marks the goal — flush with the floor, "
            f"reachable only through the doorway (the roof rules out dropping "
            f"anything in from above).\n"
            f"Goal: ratchet the gate open by its lift bar, then slide the skillet "
            f"through the doorway (handle trailing — the doorway is narrower than "
            f"the pan is long) until it rests flat on the stove pad, and let "
            f"everything settle. The bowl on the roof is a bystander — leave it "
            f"where it sits. Setting the skillet on TOP of the hutch achieves "
            f"nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lift the barred gate by its red bar — its ratchet will hold each "
            "gain, leaving the doorway open hands-free. Then slide the black "
            "skillet along the floor through the doorway onto the orange stove "
            "pad inside the hutch, and leave everything at rest."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _local(self, body) -> torch.Tensor:
        """(N,3) body root position in the hutch frame (hutch fixed at the env
        origin with identity heading)."""
        return body.data.root_pos_w - self.env_origins

    def gate_lift(self) -> torch.Tensor:
        """(N,) gate lift (its prismatic coordinate; 0 = closed)."""
        return self._local(self.gate)[:, 2]

    def pan_pos(self) -> torch.Tensor:
        """(N,3) pan root position in the hutch frame."""
        return self._local(self.pan)

    def pan_up(self) -> torch.Tensor:
        """(N,) pan body z-axis . world z (1 = perfectly flat)."""
        q = self.pan.data.root_quat_w
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return 1.0 - 2.0 * (x * x + y * y)

    def on_pad(self) -> torch.Tensor:
        """(N,) bool: pan root within the pad tolerances, at floor height, flat."""
        c = self.cfg
        p = self.pan_pos()
        return ((p[:, 0] - c.pad_x).abs() <= c.pad_tol_x) \
            & ((p[:, 1] - c.pad_y).abs() <= c.pad_tol_y) \
            & (p[:, 2] >= c.pan_z_lo) & (p[:, 2] <= c.pan_z_hi) \
            & (self.pan_up() >= c.upright_min)

    def settled(self) -> torch.Tensor:
        """(N,) bool: every mover's CoM is slow (same gate for all four bodies).

        The pawl is judged by CoM LINEAR velocity like the others, NOT by
        angular velocity: carrying the gate at its jam stop leaves a steady
        GPU resting-contact velocity bias (~0.94 rad/s measured, re-injected by
        the constraint solve every substep while nothing actually moves), so an
        angular gate can never pass in the mechanism's loaded rest state. A
        genuinely swinging pawl re-seats at 5-15 rad/s = 0.09-0.25 m/s CoM
        speed and is still caught by the 0.05 m/s gate."""
        c = self.cfg
        return (self.pan.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.gate.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.pawl.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.gate.data.root_pos_w, self.pawl.data.root_pos_w,
                         self.pan.data.root_pos_w, self.bowl.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        lift = self.gate_lift()
        self._opened |= (lift >= c.e_open) & fin
        still = self.gate.data.root_lin_vel_w[:, 2].abs() < 0.02
        holding = (lift >= c.e_open - 0.005) & still & fin
        self._prop_ctr = torch.where(holding, self._prop_ctr + 1,
                                     torch.zeros_like(self._prop_ctr))
        self._prop |= self._prop_ctr >= c.prop_steps
        # transit credential: the pan root OBSERVED inside the doorway tunnel /
        # just-inside strip, at floor level, under a raised gate. A teleport to
        # the pad jumps this slab and never earns it.
        p = self.pan_pos()
        in_slab = (p[:, 0] > c.slab_x0) & (p[:, 0] < c.slab_x1) \
            & (p[:, 1].abs() < c.door_hw) & (p[:, 2] < c.slab_z1)
        self._transit |= in_slab & (lift >= c.pan_clear) & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the pan rests upright on the stove pad inside the hutch,
        everything settled and finite, AND the pan physically crossed the
        doorway under a raised gate (the latched transit credential — the roof
        seals every other route, so this is the only physical way in)."""
        self._update_latches()
        return self._transit & self.on_pad() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20 opened + 0.20 propped (ratchet holds the
        gate open hands-off) + 0.25 transit, all latched, capped at 0.65;
        exactly 1.0 iff success() holds live. Doing nothing scores ~0 (gravity
        keeps the gate shut); the seed's strategy — set the pan on top of the
        structure — fires no latch and scores ~0."""
        c = self.cfg
        self._update_latches()
        base = (c.w_open * self._opened.float() + c.w_prop * self._prop.float()
                + c.w_transit * self._transit.float()).clamp(max=0.65)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="ratchet_portcullis", robot="null"))
