"""SiloTipPourScene — tip a caged silo with its lever so the cream box pours into the
staged basket.

Derived from libero/libero_pick_cream_cheese ("pick up the cream cheese and put it in
the basket": grasp the cream-cheese box among distractors, carry it through free air,
release it over an open basket — one pick-and-place judged by a containment bbox).
Here the payload strategy is inverted wholesale: the cream box is UNGRASPABLE. It sits
caged inside an elevated SILO HUTCH — roofed, walled, with one open spout barely wider
than the box (15 mm per side: no finger fits beside it) and too little headroom above
it — mounted on a revolute hinge at the spout lip. The only way to move the box is to
actuate the MECHANISM: press down the red LEVER PADDLE that sticks out beside the
silo roof, tipping the whole hutch forward on its hinge until the box slides out of
the spout and falls. And the fall must be RECEIVED: whatever leaves a silo and lands
on the ground or the bare catch pad (instead of inside the basket) is a PERMANENT
fail — so the basket must first be carried from its start spot and staged on the
catch pad under the correct silo's spout. There are two identical silos: one cages
the CREAM box (the target), the other a BROWN decoy; which side is which is shuffled
per episode and the only cue is the box color visible through the spout. The silo is
back-tilted at rest (box rests against the back wall) and self-returns when released:
the pour is a transient the solver must set up in advance, not a state it can hold.

Assets are fully procedural (compound-spawner pattern — child colliders of one body
never self-collide):
  - stand: heavy DYNAMIC compound (30 kg; dynamic, not kinematic — a joint anchored
    to a teleported kinematic body0 stays world-fixed at the spawn pose on this
    stack). Local frame: origin at footprint centre on the ground, spouts face local
    +x. Base slab, two hinge columns, and two thin CATCH PADS on the ground under the
    spouts (pad centres local (0.24, ±0.17)).
  - silos (x2): one rigid compound each (0.35 kg), body origin AT the hinge = the
    front-bottom spout lip. Floor (slick-bound), back wall, side walls, roof
    (interior 108 x 90 x 78 mm), open front spout, plus the red lever bar + paddle
    cantilevered forward on the outboard side. REVOLUTE joint to the stand authored
    at spawn: axis local Y, anchor stand-local (0.16, ±0.17, 0.18), localRot0 =
    rotY(-12 deg) so joint angle 0 = rest pitch -12 deg (back-tilted); limits
    [-0.5 deg, +55 deg] => pitch range [-12.5, +43]. Explicit MassAPI centre of mass
    (-0.060, 0, 0.042) — mass-only authoring leaves the CoM at the body origin (the
    hinge), which would kill the restoring torque. The CoM balance angle (~55 deg
    pitch) is past the +43 deg stop, so a released silo ALWAYS swings back shut.
  - boxes: cream 60 x 60 x 55 mm 150 g target, brown identical decoy, one per silo.
  - basket: one rigid compound (0.30 kg), origin at bottom centre: 180 mm square,
    115 mm tall, 8 mm walls (inner half-width 82 mm), starting on the open ground
    well away from both pads.

Per-episode randomization (readback-verifiable): stand yaw +/- 12 deg + xy jitter,
cream/brown side swap, per-box pose jitter inside the silos, basket ground pose +
free yaw.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.15 * staged — basket ever at rest on the CREAM silo's pad centre (latched)
  0.25 * poured — the cream box ever exits its silo WHILE the silo is tipped forward
                  (pitch >= pour_min); exiting an untipped silo latches the BREACH
                  fail instead (anti-teleport, anti-drag-out)
  0.20 * landed — the cream box ever inside the upright grounded basket after a
                  legal pour (latched)
  fail latches (permanent, freeze credit and block success): breach (above) and
  grounded (the cream box at ground level away from the basket — the pour missed or
  the box was dumped without a receiver).
  1.0 iff success() — cream box inside the basket, brown box NOT inside it, basket
  upright on the ground, no fail latch, everything settled (consecutive-still
  counter) and finite. Non-success capped at 0.60.

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


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat([torch.zeros_like(q[:, :1]), v], dim=-1)
    return _qmul(_qmul(q, qv), _qinv(q))[:, 1:]


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


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


def _add_box(stage, path: str, *, center, size, color, collide: Callable):
    """One box child: translate + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the stand: heavy DYNAMIC compound. Local frame: origin at the footprint
    centre on the ground; spouts face local +x. Children: base slab, two hinge
    columns (inboard of the silos, clear of the lever sweep), two thin catch pads on
    the ground under the spouts."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(30.0)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    c = cfg

    # base slab: x -0.20 .. +0.16, y +/- 0.23
    _add_box(stage, f"{prim_path}/base", center=(-0.02, 0.0, 0.010),
             size=(0.36, 0.46, 0.020), color=c.body_color, collide=collide)
    # hinge columns, INBOARD of each silo (the levers cantilever outboard)
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/column_{tag}",
                 center=(c.hinge_x, sgn * 0.095, 0.100),
                 size=(0.050, 0.030, 0.160), color=c.body_color, collide=collide)
    # catch pads on the ground under the spouts (visual + landing surface)
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/pad_{tag}",
                 center=(c.pad_x, sgn * c.pad_y, 0.002),
                 size=(0.190, 0.160, 0.004), color=c.pad_color, collide=collide)
    return root


def _spawn_silo(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one silo hutch: DYNAMIC compound, body origin AT the hinge (the
    front-bottom spout lip), floor extending back in -x. Children: floor (slick
    material bound), back wall, two side walls, roof, and the red lever (roof
    cross-arm + forward bar + paddle) on the outboard side (cfg.side = +/-1).

    Mass properties are authored EXPLICITLY: mass, centre of mass AND diagonal
    inertia. Mass-only authoring leaves the CoM at the body origin — the hinge —
    which would zero the restoring torque and let the silo rest anywhere.

    The REVOLUTE joint to the sibling stand is authored here at spawn (post-play
    joints are dead): axis Y, anchor stand-local (hinge_x, side*hinge_y, hinge_z),
    localRot0 = rotY(rest_pitch) so joint angle 0 = the back-tilted rest pose.
    Collision with the stand stays joint-filtered (default): the travel stops are
    the joint limits, not contact."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    s = float(c.side)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(c.silo_mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(float(c.com_x), 0.0, float(c.com_z)))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.0009, 0.0009, 0.0009))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(float(c.silo_ang_damping))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)

    # hutch shell (interior 108 long x 90 wide x 78 tall; open front at x = 0)
    _add_box(stage, f"{prim_path}/floor", center=(-0.058, 0.0, -0.004),
             size=(0.116, 0.106, 0.008), color=c.floor_color, collide=collide)
    _add_box(stage, f"{prim_path}/back", center=(-0.112, 0.0, 0.039),
             size=(0.008, 0.106, 0.094), color=c.body_color, collide=collide)
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/side_{tag}", center=(-0.058, sgn * 0.049, 0.039),
                 size=(0.116, 0.008, 0.094), color=c.body_color, collide=collide)
    _add_box(stage, f"{prim_path}/roof", center=(-0.058, 0.0, 0.082),
             size=(0.116, 0.106, 0.008), color=c.body_color, collide=collide)
    # lever: cross-arm off the roof's outboard front corner, forward bar, paddle.
    # Outboard (y = side * 0.075 for the bar) so the paddle sweep clears both the
    # falling box (|y| <= 0.045 silo-local) and the staged basket below.
    _add_box(stage, f"{prim_path}/lever_arm", center=(-0.010, s * 0.064, 0.082),
             size=(0.020, 0.045, 0.008), color=c.lever_color, collide=collide)
    _add_box(stage, f"{prim_path}/lever_bar", center=(0.045, s * 0.075, 0.082),
             size=(0.130, 0.012, 0.008), color=c.lever_color, collide=collide)
    _add_box(stage, f"{prim_path}/paddle", center=(0.115, s * 0.075, 0.086),
             size=(0.035, 0.046, 0.006), color=c.lever_color, collide=collide)

    # moderate-friction floor so the box slides out at ~12 deg beyond level
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    mat_path = f"{prim_path}/floorMat"
    sim_utils.spawn_rigid_body_material(
        mat_path,
        sim_utils.RigidBodyMaterialCfg(static_friction=c.floor_static,
                                       dynamic_friction=c.floor_dynamic,
                                       restitution=0.0))
    bind_physics_material(f"{prim_path}/floor", mat_path)
    bind_physics_material(f"{prim_path}/back", mat_path)

    # revolute hinge to the sibling stand, axis along stand-local Y at the spout lip
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Stand"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(float(c.hinge_x), s * float(c.hinge_y), float(c.hinge_z)))
    half = math.radians(c.rest_pitch_deg) / 2.0
    j.CreateLocalRot0Attr(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, math.sin(half), 0.0)))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    # travel stops: rest against the lower limit; dump pitch = rest + upper limit
    j.CreateLowerLimitAttr(-0.5)
    j.CreateUpperLimitAttr(55.0)
    return root


def _spawn_basket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the basket: DYNAMIC compound, origin at the bottom centre. Floor plus
    four walls; 180 mm square, 115 mm tall, 8 mm walls (inner half-width 82 mm)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.basket_mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.20)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)

    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, 0.004),
             size=(0.180, 0.180, 0.008), color=c.basket_color, collide=collide)
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/wall_x{tag}", center=(sgn * 0.086, 0.0, 0.0615),
                 size=(0.008, 0.180, 0.107), color=c.basket_color, collide=collide)
        _add_box(stage, f"{prim_path}/wall_y{tag}", center=(0.0, sgn * 0.086, 0.0615),
                 size=(0.164, 0.008, 0.107), color=c.basket_color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "stand" not in _SPAWNER_CACHE:

        @configclass
        class StandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            hinge_x: float = 0.16
            pad_x: float = 0.24
            pad_y: float = 0.17
            body_color: tuple = (0.35, 0.37, 0.42)
            pad_color: tuple = (0.16, 0.17, 0.20)
            contact_offset: float = 0.002

        @configclass
        class SiloSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_silo)
            side: float = 1.0
            silo_mass: float = 0.35
            com_x: float = -0.060
            com_z: float = 0.042
            silo_ang_damping: float = 1.2
            hinge_x: float = 0.16
            hinge_y: float = 0.17
            hinge_z: float = 0.18
            rest_pitch_deg: float = -12.0
            floor_static: float = 0.20
            floor_dynamic: float = 0.16
            body_color: tuple = (0.62, 0.62, 0.66)
            floor_color: tuple = (0.50, 0.50, 0.55)
            lever_color: tuple = (0.85, 0.15, 0.10)
            contact_offset: float = 0.002

        @configclass
        class BasketSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_basket)
            basket_mass: float = 0.30
            basket_color: tuple = (0.76, 0.60, 0.35)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["stand"] = StandSpawnerCfg
        _SPAWNER_CACHE["silo"] = SiloSpawnerCfg
        _SPAWNER_CACHE["basket"] = BasketSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SiloTipPourSceneCfg(BaseCfg):
    """Config for `SiloTipPourScene`. The pour gate (`pour_min_deg`) is honest by
    construction: it is far below the pitch a box physically needs to slide out
    (floor friction => ~+12 deg), so any legitimate pour clears it easily, while an
    exit from an untipped silo — a teleport, or a drag out of the open spout at the
    -12.5 deg rest pose — cannot."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    pour_min_deg: float = tunable(2.0)     # exit below this pitch = breach fail
    stage_tol: float = tunable(0.06)       # basket-to-pad-centre distance for `staged` (m)
    settle_speed: float = tunable(0.05)    # max |lin vel| (cream, basket) when judging (m/s)
    silo_settle_avel: float = tunable(0.50)  # max silo |ang vel| when judging (rad/s)
    still_steps: int = tunable(60)         # consecutive still substeps required (0.5 s)

    # --- tunable: randomization (the task-family knobs) -------------------------------------------
    stand_yaw_deg: float = tunable(12.0)   # stand yaw about its nominal heading (+/- deg)
    stand_jitter: float = tunable(0.03)    # stand xy jitter (+/- m)
    side_swap: bool = tunable(True)        # shuffle which silo cages the cream box
    box_jitter_x: float = tunable(0.008)   # box jitter inside the silo (+/- m)
    box_jitter_y: float = tunable(0.010)
    box_yaw_deg: float = tunable(8.0)      # box yaw jitter (+/- deg)
    basket_x: float = tunable(0.45)        # basket start, stand-local x (nominal)
    basket_jitter_x: float = tunable(0.05)
    basket_jitter_y: float = tunable(0.06)

    # --- info: layout (world nominal; spouts face stand-local +x) ---------------------------------
    stand_pos: tuple = info((0.42, 0.0))   # stand origin on the ground (nominal)
    stand_yaw_nom_deg: float = info(180.0)  # nominal heading: spouts face world -x
    # --- info: hinge / silo structure (silo local frame: origin at the hinge = spout lip) ---------
    hinge_x: float = info(0.16)            # hinge anchor, stand-local
    hinge_y: float = info(0.17)
    hinge_z: float = info(0.18)
    rest_pitch_deg: float = info(-12.0)    # joint angle 0 (rests on the -0.5 deg stop)
    max_pitch_deg: float = info(43.0)      # upper joint stop (+55 deg travel)
    silo_int_len: float = info(0.108)      # back wall inner face .. spout lip
    silo_int_w: float = info(0.090)        # spout / interior width
    silo_int_h: float = info(0.078)        # floor top .. roof underside
    silo_mass: float = info(0.35)
    paddle_center: tuple = info((0.115, 0.075, 0.086))  # silo-local (y is * side)
    # --- info: catch pads / basket ----------------------------------------------------------------
    pad_x: float = info(0.24)              # pad centres, stand-local (0.24, +/-0.17)
    pad_y: float = info(0.17)
    basket_outer: float = info(0.18)       # square footprint
    basket_h: float = info(0.115)
    basket_inner_half: float = info(0.082)
    basket_mass: float = info(0.30)
    # --- info: boxes -------------------------------------------------------------------------------
    box_size: tuple = info((0.060, 0.060, 0.055))
    box_mass: float = info(0.15)
    box_slot_x: float = info(-0.066)       # box start, silo-local
    cream_color: tuple = info((0.93, 0.90, 0.78))
    brown_color: tuple = info((0.42, 0.26, 0.13))
    contact_offset: float = info(0.002)
    # --- info: rubric geometry gates ---------------------------------------------------------------
    ground_z_max: float = info(0.045)      # cream centre below this ...
    ground_r_min: float = info(0.09)       # ... AND farther than this from the basket = grounded
    # rubric weights (0.15 + 0.25 + 0.20 = 0.60 = the non-success cap)
    w_staged: float = info(0.15)
    w_poured: float = info(0.25)
    w_landed: float = info(0.20)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("silo_tip_pour")
class SiloTipPourScene(BaseScene):
    cfg: SiloTipPourSceneCfg

    def __init__(self, cfg: SiloTipPourSceneCfg | None = None) -> None:
        super().__init__(cfg or SiloTipPourSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        stand_spawn = cls["stand"](hinge_x=c.hinge_x, pad_x=c.pad_x, pad_y=c.pad_y,
                                   contact_offset=c.contact_offset)
        basket_spawn = cls["basket"](basket_mass=c.basket_mass, contact_offset=c.contact_offset)

        box_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.05, angular_damping=0.10,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.24, dynamic_friction=0.20, restitution=0.0),
        )

        # template poses: each silo MUST spawn consistent with its authored joint
        # frames (stand at the nominal pose -> hinge world pose computed here)
        px, py = c.stand_pos
        yaw0 = math.radians(c.stand_yaw_nom_deg)
        cy, sy = math.cos(yaw0), math.sin(yaw0)
        qz0 = (math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2))
        hp = math.radians(c.rest_pitch_deg) / 2.0
        qy0 = (math.cos(hp), 0.0, math.sin(hp), 0.0)
        q_silo = (qz0[0] * qy0[0], -qz0[3] * qy0[2], qz0[0] * qy0[2], qz0[3] * qy0[0])

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
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=stand_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0), rot=qz0),
            ),
            "basket": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Basket",
                spawn=basket_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px - 0.45, py, 0.003)),
            ),
        }
        for sgn, tag in ((1.0, "p"), (-1.0, "n")):
            silo_spawn = cls["silo"](side=sgn, hinge_x=c.hinge_x, hinge_y=c.hinge_y,
                                     hinge_z=c.hinge_z, rest_pitch_deg=c.rest_pitch_deg,
                                     silo_mass=c.silo_mass, contact_offset=c.contact_offset)
            hx_w = px + cy * c.hinge_x - sy * sgn * c.hinge_y
            hy_w = py + sy * c.hinge_x + cy * sgn * c.hinge_y
            out[f"silo_{tag}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Silo_" + tag.upper(),
                spawn=silo_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hx_w, hy_w, c.hinge_z), rot=q_silo),
            )
        for name in ("cream", "brown"):
            color = c.cream_color if name == "cream" else c.brown_color
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Box_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=c.box_size,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.box_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    **box_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(1.0 if name == "cream" else 1.3, 1.0, 0.05)),
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
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.stand: RigidObject = env.iscene["stand"]
        self.silo_p: RigidObject = env.iscene["silo_p"]
        self.silo_n: RigidObject = env.iscene["silo_n"]
        self.basket: RigidObject = env.iscene["basket"]
        self.cream: RigidObject = env.iscene["cream"]
        self.brown: RigidObject = env.iscene["brown"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # cream_side[e] = +1 / -1: sign of the silo (stand-local y) caging the CREAM box
        self.cream_side = torch.ones(n, dtype=torch.float, device=dev)
        # credit latches (survive transients; success is judged live)
        self._staged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._poured = torch.zeros(n, dtype=torch.bool, device=dev)
        self._landed = torch.zeros(n, dtype=torch.bool, device=dev)
        # fail latches (permanent; freeze credit, block success)
        self._breach = torch.zeros(n, dtype=torch.bool, device=dev)
        self._grounded = torch.zeros(n, dtype=torch.bool, device=dev)
        # exit-event tracking + settle bookkeeping
        self._prev_in = torch.ones(n, dtype=torch.bool, device=dev)
        self._still = torch.zeros(n, dtype=torch.long, device=dev)
        self._steps = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the stand (yaw + xy jitter), seat both silos on their
        hinges at the rest pitch, cage one box per silo (cream side swap + jitter +
        yaw), drop the basket on the open ground (free yaw), clear all latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- stand: nominal heading + yaw + xy jitter ---
        yaw = math.radians(c.stand_yaw_nom_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.stand_yaw_deg)
        q_st = _qz(yaw)
        pp = torch.zeros(m, 3, device=dev)
        pp[:, 0] = c.stand_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.stand_jitter
        pp[:, 1] = c.stand_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.stand_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + origin
        st[:, 3:7] = q_st
        self.stand.write_root_state_to_sim(st, env_ids)

        # --- cream side swap ---
        if c.side_swap:
            side = torch.where(torch.rand(m, device=dev) < 0.5,
                               torch.ones(m, device=dev), -torch.ones(m, device=dev))
        else:
            side = torch.ones(m, device=dev)
        self.cream_side[env_ids] = side

        # --- silos: seated on their hinges at the rest pitch (joint angle 0) ---
        q_rest = _qmul(q_st, _qy(torch.full((m,), math.radians(c.rest_pitch_deg), device=dev)))
        for silo, sgn in ((self.silo_p, 1.0), (self.silo_n, -1.0)):
            anchor = torch.zeros(m, 3, device=dev)
            anchor[:, 0] = c.hinge_x
            anchor[:, 1] = sgn * c.hinge_y
            anchor[:, 2] = c.hinge_z
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pp + _qapply(q_st, anchor) + origin
            st[:, 3:7] = q_rest
            silo.write_root_state_to_sim(st, env_ids)

        # --- boxes: caged, one per silo (cream on `side`), jitter + yaw ---
        for body, sgn in ((self.cream, side), (self.brown, -side)):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = c.box_slot_x + (torch.rand(m, device=dev) * 2 - 1) * c.box_jitter_x
            loc[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.box_jitter_y
            loc[:, 2] = c.box_size[2] / 2 + 0.003
            anchor = torch.zeros(m, 3, device=dev)
            anchor[:, 0] = c.hinge_x
            anchor[:, 1] = sgn * c.hinge_y
            anchor[:, 2] = c.hinge_z
            qb = _qz((torch.rand(m, device=dev) * 2 - 1) * math.radians(c.box_yaw_deg))
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pp + _qapply(q_st, anchor) + _qapply(q_rest, loc) + origin
            st[:, 3:7] = _qmul(q_rest, qb)
            body.write_root_state_to_sim(st, env_ids)

        # --- basket: on the open ground in front, free yaw (always >= 0.16 m from
        # either pad centre by construction of the jitter ranges) ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.basket_x + (torch.rand(m, device=dev) * 2 - 1) * c.basket_jitter_x
        loc[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.basket_jitter_y
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + _qapply(q_st, loc) + origin
        st[:, 2] = 0.003 + origin[:, 2]
        st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
        self.basket.write_root_state_to_sim(st, env_ids)

        # --- clear latches / bookkeeping ---
        self._staged[env_ids] = False
        self._poured[env_ids] = False
        self._landed[env_ids] = False
        self._breach[env_ids] = False
        self._grounded[env_ids] = False
        self._prev_in[env_ids] = True
        self._still[env_ids] = 0
        self._steps[env_ids] = 0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "silo_p": self.silo_p.data.root_state_w[env_ids].clone(),
            "silo_n": self.silo_n.data.root_state_w[env_ids].clone(),
            "basket": self.basket.data.root_state_w[env_ids].clone(),
            "cream": self.cream.data.root_state_w[env_ids].clone(),
            "brown": self.brown.data.root_state_w[env_ids].clone(),
            "cream_side": self.cream_side[env_ids].clone(),
            "staged": self._staged[env_ids].clone(),
            "poured": self._poured[env_ids].clone(),
            "landed": self._landed[env_ids].clone(),
            "breach": self._breach[env_ids].clone(),
            "grounded": self._grounded[env_ids].clone(),
            "prev_in": self._prev_in[env_ids].clone(),
            "still": self._still[env_ids].clone(),
            "steps": self._steps[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.silo_p.write_root_state_to_sim(state["silo_p"], env_ids)
        self.silo_n.write_root_state_to_sim(state["silo_n"], env_ids)
        self.basket.write_root_state_to_sim(state["basket"], env_ids)
        self.cream.write_root_state_to_sim(state["cream"], env_ids)
        self.brown.write_root_state_to_sim(state["brown"], env_ids)
        self.cream_side[env_ids] = state["cream_side"]
        self._staged[env_ids] = state["staged"]
        self._poured[env_ids] = state["poured"]
        self._landed[env_ids] = state["landed"]
        self._breach[env_ids] = state["breach"]
        self._grounded[env_ids] = state["grounded"]
        self._prev_in[env_ids] = state["prev_in"]
        self._still[env_ids] = state["still"]
        self._steps[env_ids] = state["steps"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A heavy grey STAND carries two identical elevated SILO HUTCHES side by "
            f"side, their open square spouts facing you at "
            f"{c.hinge_z * 1000:.0f} mm height. Each silo is a roofed, walled cage "
            f"(interior {c.silo_int_len * 1000:.0f} x {c.silo_int_w * 1000:.0f} x "
            f"{c.silo_int_h * 1000:.0f} mm) holding one box "
            f"({c.box_size[0] * 1000:.0f} x {c.box_size[1] * 1000:.0f} x "
            f"{c.box_size[2] * 1000:.0f} mm, visible through the spout): one silo "
            f"cages the CREAM-WHITE box, the other a BROWN decoy — which side is "
            f"which is shuffled per episode, and the box color seen through the "
            f"spout is the only cue. The boxes CANNOT be grasped: the spout leaves "
            f"only 15 mm beside the box and the roof leaves no room above it. Each "
            f"silo pivots on a hinge at its spout lip and rests tilted BACK 12 "
            f"degrees, the box lying against its back wall; a RED LEVER PADDLE "
            f"sticks out forward beside each silo's roof. Pressing a paddle DOWN "
            f"tips that silo forward (up to {c.max_pitch_deg:.0f} degrees); past "
            f"roughly 12 degrees forward its box slides out of the spout and falls. "
            f"A released silo always swings back to its rest tilt. Below each spout "
            f"a dark CATCH PAD marks the landing spot on the ground, and a tan "
            f"BASKET ({c.basket_outer * 1000:.0f} mm square, "
            f"{c.basket_h * 1000:.0f} mm tall, open top) stands on the ground off "
            f"to the front, away from both pads.\n"
            f"Goal: deliver the CREAM-WHITE box into the basket. Because the box "
            f"can only leave its silo by being poured, first carry the basket onto "
            f"the catch pad under the CREAM silo's spout (centre it on the pad), "
            f"then press that silo's red paddle to tip the silo until the cream box "
            f"slides out, and release — the box must fall INTO the basket and the "
            f"silo swings back shut. Two hard rules: (1) a box may leave its silo "
            f"only while that silo is tipped forward (poured, not dragged or "
            f"flicked out of the resting silo — any exit from an untipped silo is a "
            f"permanent fail); (2) anything poured must be received — if the cream "
            f"box ends up on the ground or the bare pad instead of inside the "
            f"basket, that is a permanent fail. Finish with the cream box inside "
            f"the upright basket on the ground, the brown box NOT in the basket, "
            f"and everything at rest. Staging the basket AFTER pouring is therefore "
            f"impossible — stage first."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Find which silo cages the cream-white box, carry the basket onto the "
            "catch pad under that silo's spout, then press the silo's red lever "
            "paddle to tip it until the cream box pours out and falls into the "
            "basket; release the lever and let everything settle. Leave the brown "
            "box in its own silo."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _stand_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the stand frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.stand.data.root_quat_w,
                                  pos_w - self.stand.data.root_pos_w)

    def _silo_local(self, silo: RigidObject, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> one silo's body frame (origin at the hinge / spout lip)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(silo.data.root_quat_w, pos_w - silo.data.root_pos_w)

    def silo_pitch_deg(self, silo: RigidObject) -> torch.Tensor:
        """(N,) signed pitch in degrees, stand frame (rest ~ -12.5; positive = spout
        tipped down = pouring)."""
        q_rel = _qmul(_qinv(self.stand.data.root_quat_w), silo.data.root_quat_w)
        return torch.rad2deg(2.0 * torch.atan2(q_rel[:, 2], q_rel[:, 0]))

    def cream_silo_pitch_deg(self) -> torch.Tensor:
        """(N,) pitch of whichever silo cages the cream box this episode."""
        return torch.where(self.cream_side > 0, self.silo_pitch_deg(self.silo_p),
                           self.silo_pitch_deg(self.silo_n))

    def _in_silo(self, silo: RigidObject, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool: world point inside one silo's cage volume (silo frame)."""
        loc = self._silo_local(silo, pos_w)
        return (loc[:, 0] > -0.112) & (loc[:, 0] < 0.004) \
            & (loc[:, 1].abs() < 0.049) & (loc[:, 2] > -0.005) & (loc[:, 2] < 0.080)

    def cream_in_silo(self) -> torch.Tensor:
        """(N,) bool: cream box inside ITS silo (the cream_side one)."""
        p = self.cream.data.root_pos_w
        return torch.where(self.cream_side > 0, self._in_silo(self.silo_p, p),
                           self._in_silo(self.silo_n, p))

    def pad_center_w(self, side: torch.Tensor) -> torch.Tensor:
        """(N,3) world position of the catch-pad centre on the given side (+1/-1)."""
        c = self.cfg
        loc = torch.zeros(side.shape[0], 3, device=side.device)
        loc[:, 0] = c.pad_x
        loc[:, 1] = side * c.pad_y
        from isaaclab.utils.math import quat_apply

        return self.stand.data.root_pos_w + quat_apply(self.stand.data.root_quat_w, loc)

    def basket_upright(self) -> torch.Tensor:
        """(N,) bool: basket up-axis within ~12 deg of world up."""
        up = _qapply(self.basket.data.root_quat_w,
                     torch.tensor([[0.0, 0.0, 1.0]], device=self.env.device).expand(
                         self.basket.data.root_quat_w.shape[0], 3))
        return up[:, 2] > math.cos(math.radians(12.0))

    def basket_on_ground(self) -> torch.Tensor:
        """(N,) bool: basket bottom at ground/pad level."""
        z = self.basket.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return z < 0.025

    def basket_staged(self) -> torch.Tensor:
        """(N,) bool (live): basket upright on the cream-side pad centre."""
        d = self.basket.data.root_pos_w - self.pad_center_w(self.cream_side)
        near = d[:, 0:2].norm(dim=-1) < self.cfg.stage_tol
        return near & self.basket_upright() & self.basket_on_ground()

    def in_basket(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool: world point inside the basket's inner volume (basket frame)."""
        from isaaclab.utils.math import quat_apply_inverse

        loc = quat_apply_inverse(self.basket.data.root_quat_w,
                                 pos_w - self.basket.data.root_pos_w)
        # 0.060: a box leaning against an inner wall keeps its centre inside this
        # (wall inner face at 0.082, box half 0.030); anything straddling or outside
        # a wall has |centre| >= ~0.086 and a rim perch has z >= ~0.11.
        return (loc[:, 0].abs() < 0.060) & (loc[:, 1].abs() < 0.060) \
            & (loc[:, 2] > 0.015) & (loc[:, 2] < 0.088)

    def cream_grounded(self) -> torch.Tensor:
        """(N,) bool (live): cream box at ground level, away from the basket."""
        c = self.cfg
        z = self.cream.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        r = (self.cream.data.root_pos_w[:, 0:2]
             - self.basket.data.root_pos_w[:, 0:2]).norm(dim=-1)
        return (z < c.ground_z_max) & (r > c.ground_r_min)

    def settled(self) -> torch.Tensor:
        """(N,) bool: consecutive-still counter satisfied (not an instantaneous
        velocity gate — teleports zero velocities and swings cross zero at turning
        points; the counter runs in post_step)."""
        return (self._still >= self.cfg.still_steps) & (self._steps >= self.cfg.still_steps)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in
                         (self.silo_p, self.silo_n, self.basket, self.cream, self.brown)],
                        dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        # --- exit-event detection (cream box vs its own silo) ---
        now_in = self.cream_in_silo()
        exited = self._prev_in & ~now_in
        pitch = self.cream_silo_pitch_deg()
        failed = self._breach | self._grounded
        self._breach |= exited & (pitch < c.pour_min_deg) & fin & ~failed
        self._grounded |= self.cream_grounded() & fin & ~self._breach
        failed = self._breach | self._grounded
        # --- credit latches (frozen once failed) ---
        slow_b = self.basket.data.root_lin_vel_w.norm(dim=-1) < 0.08
        self._staged |= self.basket_staged() & slow_b & fin & ~failed
        self._poured |= exited & (pitch >= c.pour_min_deg) & fin & ~failed
        self._landed |= self.in_basket(self.cream.data.root_pos_w) & self._poured \
            & self.basket_upright() & self.basket_on_ground() & fin & ~failed
        self._prev_in = now_in
        # --- consecutive-still counter ---
        still = (self.cream.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.basket.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.brown.data.root_lin_vel_w.norm(dim=-1) < 0.10) \
            & (self.silo_p.data.root_ang_vel_w.norm(dim=-1) < c.silo_settle_avel) \
            & (self.silo_n.data.root_ang_vel_w.norm(dim=-1) < c.silo_settle_avel)
        self._still = torch.where(still, self._still + 1, torch.zeros_like(self._still))
        self._steps = self._steps + 1

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: cream box inside the basket, brown box NOT inside it, basket
        upright on the ground, no fail latch, everything settled and finite. The
        containment/upright clauses are live physical outcomes; the fail latches
        enforce that the box got there by a received pour."""
        return self.in_basket(self.cream.data.root_pos_w) \
            & ~self.in_basket(self.brown.data.root_pos_w) \
            & self.basket_upright() & self.basket_on_ground() \
            & ~self._breach & ~self._grounded \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*staged + 0.25*poured + 0.20*landed (latched;
        ~0 for doing nothing — staging needs the basket moved onto the pad, pouring
        needs the silo tipped past `pour_min_deg` while its box exits), capped at
        0.60 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        base = (c.w_staged * self._staged.float() + c.w_poured * self._poured.float()
                + c.w_landed * self._landed.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="silo_tip_pour", robot="null"))
