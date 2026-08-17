"""MastLoweringScene — deploy the catch cradle, then fell the hinged mast so it
comes to rest ON the cradle's crest, spanning OVER the protected bottle, which
must stay standing on its marker pad (sim_gen task `scene_d_i399`).

Derived from calvin/scene_D but STRATEGICALLY different: the CALVIN table is a
menu of independent single-DOF primitives — press the button, flick the switch,
push the slider to its far end, pull the drawer, pick a block in free space.
Every seed objective is "actuate one binary DOF" or "grasp a free block", and
nothing in the scene is ever at risk. Here the REQUIRED action is destructive
— a tall hinged MAST must be pushed past vertical and felled — and the goal is
defined by three things the seed never combines:

  (a) a PROTECTED BYSTANDER: a slender green bottle stands on a marker pad
      directly in the mast's fall corridor. It is judged for NON-disturbance:
      success requires it still standing upright ON its pad. The felled mast
      passes ~4 cm above its cap — but ONLY if it is caught at the right angle.
  (b) a CONTINUOUS catch-fixture placement whose valid band depends on the
      sampled bottle position: a free-standing CRADLE (two guide plates over a
      crest bar) must be stood in the corridor just beyond the bottle; the
      angle the mast is arrested at is set by WHERE the cradle stands
      (phi_rest = 90 deg + atan(drop / d)), not by any end stop.
  (c) a PHYSICALLY FORCED ORDER: with no cradle deployed, the falling mast
      sweeps down to the court floor and its face passes BELOW the bottle's
      cap on the way — felling first provably topples the bottle (asserted in
      cfg). Deploy-then-fell is the only order that can succeed, without any
      order latch: geometry enforces it.

No stored energy: the mast is a plain damped revolute pendulum with a back
stop (it rests leaning 1 deg onto the stop — the null policy is stable), the
cradle and bottle are free bodies — every outcome persists hands-off.

Assets are fully procedural (compound-spawner pattern; children of one body
never self-collide): court floor (KINEMATIC slab — separate from the pylon so
the joint's collision filter cannot kill mast<->floor contact), pylon
(KINEMATIC foot + post + hinge cheeks), mast (DYNAMIC shaft + red head;
explicit CoM at mid-shaft and rod inertia so the hinge statics are real),
cradle (DYNAMIC base + two guide plates + crest bar; root at the base BOTTOM
so the MassAPI CoM-at-origin quirk puts its CoM at ground level — impact
stable), bottle (DYNAMIC slender cylinder), marker pad (KINEMATIC disc). The
mast revolute is authored per-env at bind time (pylon = kinematic body0;
collision filtering applies to that pair only). USD revolute limits are in
DEGREES: back stop at -7 deg, forward limit far past floor contact.

Per-episode randomization (readback-verified by smoke): the bottle pad's spot
(db, py) in the corridor, and the cradle's parking spot + yaw off to the side
(rack_x, rack_y = random side, rack_yaw).

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.20  deployed  — the cradle ever stands upright on the corridor floor
  0.20  committed — the mast ever pitched past `commit_deg` (point of no return)
  0.20  arrested  — the mast ever at rest in the catch band with the bottle safe
capped at 0.60; exactly 1.0 iff success(): mast settled IN the catch band
(92..103 deg — both no-cradle rest poses are asserted OUTSIDE it: the floor
rest ~116 deg AND the mast propped on the bottle's flat cap, ~107 deg or
more at every sampled bottle spot) AND the bottle upright on its pad, all
settled and finite. Null policy ~0. The seed's push-a-DOF reflex (just shove
the mast) tops out at the single commit latch (0.20).

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


def _dyn_body(root, mass: float, lin_damp: float, ang_damp: float,
              com=None, inertia=None) -> None:
    """Standard dynamic compound body physics (32/4 iters, no sleep, damped).
    `com`/`inertia` optionally author an explicit centre of mass and diagonal
    inertia (the MassAPI mass-only quirk leaves the CoM at the body origin)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    massapi = UsdPhysics.MassAPI.Apply(root)
    massapi.CreateMassAttr(float(mass))
    if com is not None:
        massapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    if inertia is not None:
        massapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(4)
    prb.CreateLinearDampingAttr(float(lin_damp))
    prb.CreateAngularDampingAttr(float(ang_damp))
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)


def _spawn_court(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The court floor: ONE kinematic slab. Deliberately a separate asset from
    the pylon: the mast revolute filters only the mast<->pylon pair, so
    mast<->court contact (the floor-felled outcome) stays live."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/slab", x=(c.court_x0, c.court_x1), y=(-c.court_hy, c.court_hy),
          z=(0.0, c.court_z1), color=c.color, collide=collide)
    grip = _mk_material(prim_path, "grip", c.mu_s, c.mu_d, "average")
    bind_physics_material(prim_path, grip)
    return root


def _spawn_pylon(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The hinge pylon: KINEMATIC foot + post + two hinge cheeks flanking the
    mast's shaft in y. The hinge axis (authored per-env at bind time) runs
    along y at (0, 0, hinge_z), between the cheeks."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/foot", x=(-c.foot_half, c.foot_half),
          y=(-c.foot_half, c.foot_half), z=(c.court_z1, c.foot_z1),
          color=c.color, collide=collide)
    _span(stage, f"{prim_path}/post", x=(-c.post_half, c.post_half),
          y=(-c.post_half, c.post_half), z=(c.foot_z1, c.post_z1),
          color=c.color, collide=collide)
    _span(stage, f"{prim_path}/cheek_yp", x=(-c.cheek_hx, c.cheek_hx),
          y=(c.cheek_y0, c.cheek_y1), z=(c.post_z1, c.cheek_z1),
          color=c.cheek_color, collide=collide)
    _span(stage, f"{prim_path}/cheek_yn", x=(-c.cheek_hx, c.cheek_hx),
          y=(-c.cheek_y1, -c.cheek_y0), z=(c.post_z1, c.cheek_z1),
          color=c.cheek_color, collide=collide)
    mat = _mk_material(prim_path, "mat", c.mu_s, c.mu_d, "average")
    bind_physics_material(prim_path, mat)
    return root


def _spawn_mast(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The mast with root AT the hinge point (the revolute anchor): a slender
    shaft (local z in [0, shaft_z1]) topped by a wider RED head. Explicit CoM
    at mid-shaft and rod inertia make the pendulum statics real (mass-only
    MassAPI would leave the CoM at the hinge — a neutrally balanced mast)."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/shaft", x=(-c.shaft_hx, c.shaft_hx),
          y=(-c.shaft_hy, c.shaft_hy), z=(0.0, c.shaft_z1),
          color=c.shaft_color, collide=collide)
    _span(stage, f"{prim_path}/head", x=(-c.head_hx, c.head_hx),
          y=(-c.head_hy, c.head_hy), z=(c.shaft_z1, c.head_z1),
          color=c.head_color, collide=collide)
    _dyn_body(root, c.mass, c.lin_damp, c.ang_damp,
              com=(0.0, 0.0, c.com_z), inertia=(c.i_xy, c.i_xy, c.i_z))
    mat = _mk_material(prim_path, "mat", c.mu_s, c.mu_d, "average")
    bind_physics_material(prim_path, mat)
    return root


def _spawn_cradle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The catch cradle with root at the base BOTTOM centre (MassAPI CoM-at-
    origin quirk -> CoM at ground level, deliberately impact-stable): a square
    base, two tall guide plates flanking an open channel along x, and a crest
    bar spanning the channel below the plate tops — the saddle the falling
    mast is caught in. The crest bar doubles as the carry handle (jaw-sized)."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/base", x=(-c.base_half, c.base_half),
          y=(-c.base_half, c.base_half), z=(0.0, c.base_h),
          color=c.color, collide=collide)
    _span(stage, f"{prim_path}/plate_yp", x=(-c.plate_hx, c.plate_hx),
          y=(c.plate_y0, c.plate_y1), z=(c.base_h, c.plate_z1),
          color=c.color, collide=collide)
    _span(stage, f"{prim_path}/plate_yn", x=(-c.plate_hx, c.plate_hx),
          y=(-c.plate_y1, -c.plate_y0), z=(c.base_h, c.plate_z1),
          color=c.color, collide=collide)
    _span(stage, f"{prim_path}/crest", x=(-c.plate_hx, c.plate_hx),
          y=(-c.plate_y0, c.plate_y0), z=(c.crest_z0, c.crest_z1),
          color=c.crest_color, collide=collide)
    _dyn_body(root, c.mass, c.lin_damp, c.ang_damp)
    grip = _mk_material(prim_path, "grip", c.mu_s, c.mu_d, "average")
    bind_physics_material(prim_path, grip)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "court" not in _SPAWNER_CACHE:

        @configclass
        class CourtSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_court)
            court_x0: float = -0.16
            court_x1: float = 0.80
            court_hy: float = 0.44
            court_z1: float = 0.016
            color: tuple = (0.52, 0.54, 0.50)
            contact_offset: float = 0.0015
            mu_s: float = 0.90
            mu_d: float = 0.85

        @configclass
        class PylonSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pylon)
            court_z1: float = 0.016
            foot_half: float = 0.09
            foot_z1: float = 0.046
            post_half: float = 0.04
            post_z1: float = 0.27
            cheek_hx: float = 0.03
            cheek_y0: float = 0.035
            cheek_y1: float = 0.055
            cheek_z1: float = 0.35
            color: tuple = (0.35, 0.32, 0.30)
            cheek_color: tuple = (0.25, 0.24, 0.26)
            contact_offset: float = 0.0015
            mu_s: float = 0.60
            mu_d: float = 0.55

        @configclass
        class MastSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_mast)
            shaft_hx: float = 0.025
            shaft_hy: float = 0.03
            shaft_z1: float = 0.56
            head_hx: float = 0.035
            head_hy: float = 0.04
            head_z1: float = 0.62
            mass: float = 1.0
            com_z: float = 0.30
            i_xy: float = 0.033
            i_z: float = 0.002
            lin_damp: float = 0.05
            ang_damp: float = 12.0
            shaft_color: tuple = (0.82, 0.78, 0.55)
            head_color: tuple = (0.85, 0.10, 0.08)
            contact_offset: float = 0.0015
            mu_s: float = 0.50
            mu_d: float = 0.45

        @configclass
        class CradleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cradle)
            base_half: float = 0.08
            base_h: float = 0.02
            plate_hx: float = 0.02
            plate_y0: float = 0.052
            plate_y1: float = 0.075
            plate_z1: float = 0.26
            crest_z0: float = 0.204
            crest_z1: float = 0.22
            mass: float = 1.2
            lin_damp: float = 0.50
            ang_damp: float = 2.0
            color: tuple = (0.12, 0.25, 0.75)
            crest_color: tuple = (0.20, 0.45, 0.95)
            contact_offset: float = 0.0015
            mu_s: float = 0.90
            mu_d: float = 0.85

        _SPAWNER_CACHE["court"] = CourtSpawnerCfg
        _SPAWNER_CACHE["pylon"] = PylonSpawnerCfg
        _SPAWNER_CACHE["mast"] = MastSpawnerCfg
        _SPAWNER_CACHE["cradle"] = CradleSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class MastLoweringSceneCfg(BaseCfg):
    """Config for `MastLoweringScene`. The physical contracts are asserted in
    `__post_init__`: the rest-on-cradle angle really lies inside the catch
    band for the whole deployment window while the floor rest angle lies
    OUTSIDE it; the arrested mast really clears the bottle's cap; and a fall
    with NO cradle really drives the mast's face below the cap (the felled
    mast provably topples the bottle — order is physics-forced)."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    phi_lo_deg: float = tunable(92.0)     # catch band lower edge (deg past vertical is > 90)
    phi_hi_deg: float = tunable(103.0)    # catch band upper edge (floor rest AND the mast
    #                                       propped on the bottle's cap are both beyond this)
    commit_deg: float = tunable(15.0)     # pitched past this -> committed to falling (deg)
    bottle_xy_tol: float = tunable(0.035)  # bottle centre within this of its sampled marker (m)
    bottle_z_tol: float = tunable(0.012)  # bottle rest-height tolerance on the pad (m)
    bottle_tilt_deg: float = tunable(10.0)  # bottle "upright" cone (topple angle ~11.3 deg)
    settle_lin: float = tunable(0.06)     # max |lin vel| of bottle/cradle when judging (m/s)
    settle_ang: float = tunable(0.15)     # max mast |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    pad_x: tuple = tunable((0.28, 0.38))    # bottle pad distance band down the corridor (m)
    pad_y_max: float = tunable(0.02)        # bottle pad |y| bound (stays under the mast sweep)
    rack_x: tuple = tunable((0.18, 0.34))   # cradle parking spot x band (m)
    rack_y_mag: tuple = tunable((0.24, 0.30))  # cradle parking |y| band (side sampled) (m)

    # --- info: court / pylon (local frame: hinge axis over x=y=0) --------------------------------
    court_x0: float = info(-0.16)
    court_x1: float = info(0.80)
    court_hy: float = info(0.44)
    court_z1: float = info(0.016)         # court top (the corridor floor)
    foot_half: float = info(0.09)
    foot_z1: float = info(0.046)
    post_half: float = info(0.04)
    post_z1: float = info(0.27)
    cheek_hx: float = info(0.03)
    cheek_y0: float = info(0.035)
    cheek_y1: float = info(0.055)
    cheek_z1: float = info(0.35)
    hinge_z: float = info(0.32)           # revolute axis height (along y)
    # --- info: mast -------------------------------------------------------------------------------
    shaft_hx: float = info(0.025)         # shaft half-thickness along local x (fall direction)
    shaft_hy: float = info(0.03)
    shaft_z1: float = info(0.56)          # shaft top (local z from the hinge)
    head_hx: float = info(0.035)
    head_hy: float = info(0.04)
    head_z1: float = info(0.62)           # mast tip
    mast_mass: float = info(1.0)
    mast_com_z: float = info(0.30)        # explicit CoM: mid-shaft (real pendulum statics)
    mast_i_xy: float = info(0.033)        # rod inertia about the CoM (m L^2 / 12)
    mast_i_z: float = info(0.002)
    mast_ang_damp: float = info(12.0)     # heavy: terminal fall ~1.4 rad/s -> gentle catch
    limit_lo_deg: float = info(-7.0)      # back stop (null policy rests here)
    limit_hi_deg: float = info(128.0)     # forward limit far past floor contact
    phi0_deg: float = info(-6.0)          # reset pitch: settles 1 deg onto the back stop
    # --- info: cradle -------------------------------------------------------------------------------
    base_half: float = info(0.08)
    base_h: float = info(0.02)
    plate_hx: float = info(0.02)          # plate/crest half-length along x
    plate_y0: float = info(0.052)         # channel half-width (shaft is 0.03 half-wide)
    plate_y1: float = info(0.075)
    plate_z1: float = info(0.26)          # guide plate tops (local, from base bottom)
    crest_z0: float = info(0.204)
    crest_z1: float = info(0.22)          # crest top: the catch saddle surface
    cradle_mass: float = info(1.2)
    deploy_off: float = info(0.145)       # solve's cradle target: x = db + deploy_off
    # --- info: bottle / pad -------------------------------------------------------------------------
    bottle_r: float = info(0.016)
    bottle_h: float = info(0.16)
    bottle_mass: float = info(0.05)
    pad_r: float = info(0.035)
    pad_h: float = info(0.006)
    # --- info: deployed() gate ----------------------------------------------------------------------
    dep_x: tuple = info((0.14, 0.60))     # cradle root x band "in the corridor"
    dep_y_max: float = info(0.10)
    dep_z_tol: float = info(0.015)
    dep_tilt_deg: float = info(15.0)
    contact_offset: float = info(0.0015)
    # --- info: rubric weights (0.20 * 3 = 0.60 = the non-success cap) ----------------------------
    w_deploy: float = info(0.20)
    w_commit: float = info(0.20)
    w_arrest: float = info(0.20)

    # ----- derived geometry ----------------------------------------------------------------------
    def crest_top_w(self) -> float:
        """World z of the crest saddle surface (cradle standing on the court)."""
        return self.court_z1 + self.crest_z1

    def drop(self) -> float:
        """Vertical drop from the hinge to the caught shaft's CENTRELINE."""
        return self.hinge_z - (self.crest_top_w() + self.shaft_hx)

    def phi_rest_deg(self, d_c: float) -> float:
        """Rest pitch (deg) of a mast caught on a crest edge at horizontal d_c."""
        return 90.0 + math.degrees(math.atan(self.drop() / d_c))

    def phi_floor_deg(self) -> float:
        """Rest pitch (deg) of a mast felled all the way to the court floor
        (head's leading top corner touching the slab)."""
        rhs = self.court_z1 - self.hinge_z
        r = math.hypot(self.head_z1, self.head_hx)
        delta = math.atan2(self.head_hx, self.head_z1)
        return math.degrees(math.acos(rhs / r) - delta)

    def bottle_top_w(self) -> float:
        return self.court_z1 + self.pad_h + self.bottle_h

    def bottle_rest_z(self) -> float:
        return self.court_z1 + self.pad_h + self.bottle_h / 2

    def _face_z(self, x: float, phi_deg: float) -> float:
        """World z of the mast's UNDERSIDE face at horizontal distance x, at pitch phi."""
        th = math.radians(phi_deg - 90.0)
        return self.hinge_z - x * math.tan(th) - self.shaft_hx / math.cos(th)

    def __post_init__(self) -> None:
        c = self
        # -- basic stacking
        assert c.court_z1 < c.foot_z1 < c.post_z1 < c.hinge_z < c.cheek_z1
        assert c.base_h <= c.crest_z0 < c.crest_z1 < c.plate_z1, "crest must sit below the plate tops"
        assert c.drop() > 0.02, "hinge must sit above the caught centreline"
        # -- mast fits the cradle channel; the wider head fits too (it overhangs past the plates)
        assert c.plate_y0 >= c.shaft_hy + 0.010, "shaft must enter the channel with clearance"
        assert c.plate_y0 >= c.head_hy + 0.008, "head must clear the channel too"
        assert c.cheek_y0 >= c.shaft_hy + 0.004, "shaft must swing between the hinge cheeks"
        # -- catch band: rest-on-cradle inside it (3 deg margin) across the WHOLE deployment window
        for db in (c.pad_x[0], c.pad_x[1]):
            d_c = db + c.deploy_off + c.plate_hx  # contact = the crest's FAR top edge
            pr = c.phi_rest_deg(d_c)
            assert c.phi_lo_deg + 3.0 <= pr <= c.phi_hi_deg - 3.0, \
                f"phi_rest({d_c:.3f}) = {pr:.1f} outside band"
            # contact point stays on the shaft, short of the head
            assert math.hypot(d_c, c.drop()) <= c.shaft_z1 - 0.010, "contact must land on the shaft"
            # the wider head passes BEYOND the far guide plate edge (no snag)
            assert c.shaft_z1 * math.sin(math.radians(pr)) >= d_c + 0.005
            # ARRESTED mast clears the bottle cap by >= 25 mm at the bottle's x
            assert c._face_z(db, pr) >= c.bottle_top_w() + 0.025, \
                f"caught mast must clear the bottle cap at db={db}"
        # -- floor rest angle OUTSIDE the band (a floor-felled mast can never pass) and inside limits
        pf = c.phi_floor_deg()
        assert pf >= c.phi_hi_deg + 3.0, f"floor rest {pf:.1f} must be beyond the band"
        assert c.limit_hi_deg >= pf + 5.0, "forward limit must not mask floor contact"
        assert c.limit_hi_deg - c.limit_lo_deg <= 175.0, "revolute travel must avoid the 180 wrap"
        assert c.limit_lo_deg < c.phi0_deg < 0.0, "reset pitch must press onto the back stop"
        # -- ORDER FORCING: felling with no cradle can NEVER end in band, on either branch.
        # Branch 1 (bottle swept aside): the floor rest pose is below the cap and out of band.
        assert c._face_z(c.pad_x[0], pf) <= c.bottle_top_w() - 0.015, \
            "floor-felled mast must sweep below the bottle cap (order forcing)"
        # Branch 2 (mast lands on the flat cap and PROPS there): at 2 deg past the band's
        # upper edge the underside face is still ABOVE the cap at every sampled bottle spot,
        # so cap contact — and any bottle-propped rest — happens strictly beyond the band.
        for db in (c.pad_x[0], c.pad_x[1]):
            assert c._face_z(db, c.phi_hi_deg + 2.0) >= c.bottle_top_w() + 0.004, \
                f"bottle-propped mast at db={db} must rest beyond the band (order forcing)"
        assert c.shaft_z1 * math.sin(math.radians(pf)) >= c.pad_x[1] + 0.03, \
            "the shaft must reach past the farthest bottle spot"
        assert c.pad_y_max <= c.shaft_hy - 0.004, "bottle axis must sit under the mast footprint"
        # -- bottle: judged upright strictly inside its topple cone
        topple = math.degrees(math.atan(c.bottle_r / (c.bottle_h / 2)))
        assert c.bottle_tilt_deg <= topple - 1.0, "tilt gate must be inside the topple angle"
        assert c.phi_lo_deg >= 91.0, "band must exclude near-vertical (unsupported) poses"
        assert 5.0 <= c.commit_deg <= 30.0
        # -- deployment window: solve target inside the deployed() gate; base clear of the pad
        for db in (c.pad_x[0], c.pad_x[1]):
            assert c.dep_x[0] + 0.01 <= db + c.deploy_off <= c.dep_x[1] - 0.01
        assert c.deploy_off - c.base_half >= c.pad_r + 0.020, \
            "deployed base must not overlap the bottle pad"
        # -- rack (parking spot): fully on the court, clear of pylon/corridor, never 'deployed'
        diag = c.base_half * math.sqrt(2.0)
        assert c.rack_y_mag[1] + diag <= c.court_hy - 0.02
        assert c.rack_y_mag[0] - diag >= c.foot_half + 0.01
        assert c.rack_y_mag[0] - diag >= c.shaft_hy + 0.05, "rack clear of the fall corridor"
        assert c.rack_y_mag[0] >= c.dep_y_max + 0.05, "rack must never start deployed"
        assert c.rack_x[1] + diag <= c.court_x1 - 0.02 and c.rack_x[0] - diag >= c.court_x0 + 0.02
        # -- pad placement: clear of the pylon foot, on the court
        assert c.pad_x[0] - c.pad_r >= c.foot_half + 0.01
        assert c.pad_x[1] + c.pad_r <= c.court_x1 - 0.02
        # -- everything the hand must pinch fits the ~80 mm Franka jaw
        assert 2 * c.plate_hx <= 0.075 and 2 * c.bottle_r <= 0.075 and 2 * c.shaft_hy <= 0.075
        # -- rubric weights
        assert abs(c.w_deploy + c.w_commit + c.w_arrest - 0.60) < 1e-9


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("mast_lowering")
class MastLoweringScene(BaseScene):
    cfg: MastLoweringSceneCfg

    def __init__(self, cfg: MastLoweringSceneCfg | None = None) -> None:
        super().__init__(cfg or MastLoweringSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        court_spawn = cls["court"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            court_x0=c.court_x0, court_x1=c.court_x1, court_hy=c.court_hy, court_z1=c.court_z1,
            contact_offset=c.contact_offset)
        pylon_spawn = cls["pylon"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            court_z1=c.court_z1, foot_half=c.foot_half, foot_z1=c.foot_z1,
            post_half=c.post_half, post_z1=c.post_z1, cheek_hx=c.cheek_hx, cheek_y0=c.cheek_y0,
            cheek_y1=c.cheek_y1, cheek_z1=c.cheek_z1, contact_offset=c.contact_offset)
        mast_spawn = cls["mast"](
            shaft_hx=c.shaft_hx, shaft_hy=c.shaft_hy, shaft_z1=c.shaft_z1, head_hx=c.head_hx,
            head_hy=c.head_hy, head_z1=c.head_z1, mass=c.mast_mass, com_z=c.mast_com_z,
            i_xy=c.mast_i_xy, i_z=c.mast_i_z, ang_damp=c.mast_ang_damp,
            contact_offset=c.contact_offset)
        cradle_spawn = cls["cradle"](
            base_half=c.base_half, base_h=c.base_h, plate_hx=c.plate_hx, plate_y0=c.plate_y0,
            plate_y1=c.plate_y1, plate_z1=c.plate_z1, crest_z0=c.crest_z0, crest_z1=c.crest_z1,
            mass=c.cradle_mass, contact_offset=c.contact_offset)

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.20, angular_damping=0.20,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=4)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)
        bottle_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.60, dynamic_friction=0.55, restitution=0.0)

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "court": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Court", spawn=court_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "pylon": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pylon", spawn=pylon_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            # authored at the joint's zero pose (upright): the bind-time revolute
            # anchors at this authored position.
            "mast": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Mast", spawn=mast_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.hinge_z))),
            "cradle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cradle", spawn=cradle_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.26, 0.27, c.court_z1 + 0.004))),
            "bottle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle",
                spawn=sim_utils.CylinderCfg(
                    radius=c.bottle_r, height=c.bottle_h, axis="Z",
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bottle_mass),
                    rigid_props=rigid, collision_props=coll, physics_material=bottle_mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.65, 0.15))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.33, 0.0, self.cfg.bottle_rest_z()))),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=sim_utils.CylinderCfg(
                    radius=c.pad_r, height=c.pad_h, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll, physics_material=bottle_mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95))),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.33, 0.0, c.court_z1 + c.pad_h / 2))),
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
        self.court: RigidObject = env.iscene["court"]
        self.pylon: RigidObject = env.iscene["pylon"]
        self.mast: RigidObject = env.iscene["mast"]
        self.cradle: RigidObject = env.iscene["cradle"]
        self.bottle: RigidObject = env.iscene["bottle"]
        self.pad: RigidObject = env.iscene["pad"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke): sampled (db, py, rack_x, rack_y, rack_yaw)
        self.layout = torch.zeros(n, 5, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._deploy_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._commit_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._arrest_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._author_joints()

    def _author_joints(self) -> None:
        """Per-env mast revolute, authored ONCE at bind time against the AUTHORED
        poses: axis y at (0, 0, hinge_z) on the kinematic pylon, mast root at the
        anchor. USD revolute limits are in DEGREES; the -7 deg back stop is what
        the resting mast leans on, the +128 deg forward limit sits far past
        floor contact (~116 deg) so it never masks the felled outcome. Joint
        collision filtering disables ONLY the mast<->pylon pair — mast<->court,
        mast<->bottle and mast<->cradle contacts (the physics of the task) stay
        live (the court is a separate asset precisely for this)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        c = self.cfg
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/mast_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Pylon"])
            j.CreateBody1Rel().SetTargets([f"{base}/Mast"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.hinge_z)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(float(c.limit_lo_deg))
            j.CreateUpperLimitAttr(float(c.limit_hi_deg))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: court/pylon re-asserted at their fixed poses, mast
        written just off its back stop (settles leaning on it — the stable null
        pose), bottle standing on its pad at a SAMPLED corridor spot (db, py),
        cradle parked at a SAMPLED spot + yaw off to a SAMPLED side, latches
        cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        u = torch.rand(m, 5, device=dev)
        db = c.pad_x[0] + u[:, 0] * (c.pad_x[1] - c.pad_x[0])
        py = (2.0 * u[:, 1] - 1.0) * c.pad_y_max
        rx = c.rack_x[0] + u[:, 2] * (c.rack_x[1] - c.rack_x[0])
        ry = c.rack_y_mag[0] + u[:, 3] * (c.rack_y_mag[1] - c.rack_y_mag[0])
        side = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        ry = ry * side
        ryaw = (2.0 * u[:, 4] - 1.0) * math.pi
        self.layout[env_ids, 0] = db
        self.layout[env_ids, 1] = py
        self.layout[env_ids, 2] = rx
        self.layout[env_ids, 3] = ry
        self.layout[env_ids, 4] = ryaw

        def write(body, dx, dy, dz, yaw=None, pitch=None) -> None:
            s = torch.zeros(m, 13, device=dev)
            s[:, 0] = origin[:, 0] + dx
            s[:, 1] = origin[:, 1] + dy
            s[:, 2] = origin[:, 2] + dz
            if yaw is not None:
                s[:, 3] = torch.cos(yaw / 2)
                s[:, 6] = torch.sin(yaw / 2)
            elif pitch is not None:
                s[:, 3] = math.cos(pitch / 2)
                s[:, 5] = math.sin(pitch / 2)
            else:
                s[:, 3] = 1.0
            body.write_root_state_to_sim(s, env_ids)

        zeros = torch.zeros(m, device=dev)
        write(self.court, zeros, zeros, zeros)
        write(self.pylon, zeros, zeros, zeros)
        write(self.mast, zeros, zeros, zeros + c.hinge_z, pitch=math.radians(c.phi0_deg))
        write(self.cradle, rx, ry, zeros + c.court_z1 + 0.004, yaw=ryaw)
        write(self.pad, db, py, zeros + c.court_z1 + c.pad_h / 2)
        write(self.bottle, db, py, zeros + c.bottle_rest_z() + 0.002)

        self._deploy_l[env_ids] = False
        self._commit_l[env_ids] = False
        self._arrest_l[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "court": self.court.data.root_state_w[env_ids].clone(),
            "pylon": self.pylon.data.root_state_w[env_ids].clone(),
            "mast": self.mast.data.root_state_w[env_ids].clone(),
            "cradle": self.cradle.data.root_state_w[env_ids].clone(),
            "bottle": self.bottle.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "layout": self.layout[env_ids].clone(),
            "deploy_l": self._deploy_l[env_ids].clone(),
            "commit_l": self._commit_l[env_ids].clone(),
            "arrest_l": self._arrest_l[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.court.write_root_state_to_sim(state["court"], env_ids)
        self.pylon.write_root_state_to_sim(state["pylon"], env_ids)
        self.mast.write_root_state_to_sim(state["mast"], env_ids)
        self.cradle.write_root_state_to_sim(state["cradle"], env_ids)
        self.bottle.write_root_state_to_sim(state["bottle"], env_ids)
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        self.layout[env_ids] = state["layout"]
        self._deploy_l[env_ids] = state["deploy_l"]
        self._commit_l[env_ids] = state["commit_l"]
        self._arrest_l[env_ids] = state["arrest_l"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        d_lo = c.pad_x[0] + c.deploy_off
        d_hi = c.pad_x[1] + c.deploy_off
        return (
            f"A flat COURT slab lies on the ground. Near its left end a squat PYLON holds "
            f"a tall MAST ({c.head_z1 * 100:.0f} cm, pale shaft, RED head) on a stiff "
            f"sideways hinge {c.hinge_z * 100:.0f} cm up; the mast starts leaning about "
            f"1 degree BACK onto a stop, so left alone it stands forever. Pushed forward "
            f"past vertical it is committed: it swings down over the court (heavily "
            f"damped, so it falls slowly) along the +x corridor.\n"
            f"In that corridor, {c.pad_x[0] * 100:.0f}-{c.pad_x[1] * 100:.0f} cm from the "
            f"pylon (spot varies by episode), a slender GREEN BOTTLE "
            f"({c.bottle_h * 100:.0f} cm tall, {2 * c.bottle_r * 100:.1f} cm wide) stands "
            f"on a small WHITE marker pad. It is fragile cargo: it must END the episode "
            f"still standing upright ON its pad. With nothing else in its path a felled "
            f"mast ends badly: it either sweeps the bottle away and slams to the floor, "
            f"or jams propped on the bottle's cap — both rest far past the catch band.\n"
            f"Parked off to one side (spot and heading vary by episode) is a heavy BLUE "
            f"CRADLE: a square base with two tall guide plates over an open channel and a "
            f"bright-blue CREST BAR ({c.crest_top_w() * 100:.1f} cm high, "
            f"{2 * c.plate_hx * 100:.0f} cm wide — a pinchable handle) spanning them. "
            f"Stood upright in the corridor, channel facing the pylon (its long axis "
            f"along the corridor), it catches the falling mast in its saddle: the mast "
            f"then rests tilted just past horizontal, passing a few cm ABOVE the "
            f"bottle's cap.\n"
            f"Goal: the mast at rest CAUGHT ON THE CRADLE — pitched between "
            f"{c.phi_lo_deg:.0f} and {c.phi_hi_deg:.0f} degrees from vertical (resting "
            f"on the floor is ~{c.phi_floor_deg():.0f} degrees: too far — it never "
            f"counts) — with the bottle still standing on its pad. Stand the cradle so "
            f"its crest is {d_lo * 100:.0f}-{d_hi * 100:.0f} cm from the pylon — about "
            f"{c.deploy_off * 100:.0f} cm beyond the bottle is ideal — THEN push the "
            f"mast's shaft forward past vertical and let it fall into the saddle. "
            f"Deploy the cradle FIRST: once the mast is committed there is no saving "
            f"the bottle without it."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Carry the blue cradle into the fall corridor and stand it upright just "
            "beyond the green bottle, channel facing the pylon. Then push the tall "
            "mast forward past vertical so it swings down and comes to rest in the "
            "cradle's saddle, spanning over the bottle. The bottle must stay standing "
            "on its white pad, untouched."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _local(self, body) -> torch.Tensor:
        """(N,3) body root position in the court frame (pylon fixed at the origin)."""
        return body.data.root_pos_w - self.env_origins

    def _up(self, body) -> torch.Tensor:
        """(N,3) the body's local +z axis in world coordinates."""
        n = self.env.num_envs
        ez = torch.zeros(n, 3, device=self.env.device)
        ez[:, 2] = 1.0
        return self._quat_apply(body.data.root_quat_w, ez)

    def phi(self) -> torch.Tensor:
        """(N,) mast pitch from vertical, signed + toward the fall corridor (rad)."""
        a = self._up(self.mast)
        return torch.atan2(a[:, 0], a[:, 2])

    def mast_w(self) -> torch.Tensor:
        """(N,) mast angular velocity about the hinge axis (world y) (rad/s)."""
        return self.mast.data.root_ang_vel_w[:, 1]

    def in_band(self) -> torch.Tensor:
        """(N,) bool: mast pitch inside the catch band [phi_lo, phi_hi] deg."""
        c = self.cfg
        p = self.phi()
        return (p >= math.radians(c.phi_lo_deg)) & (p <= math.radians(c.phi_hi_deg))

    def bottle_ok(self) -> torch.Tensor:
        """(N,) bool: the bottle stands upright ON its sampled marker pad —
        tilt inside the gate cone, centre xy within tol of the sampled marker,
        rest height on the pad (a bottle standing on the court, perched on the
        caught mast, lying anywhere, or displaced all fail)."""
        c = self.cfg
        p = self._local(self.bottle)
        up_z = self._up(self.bottle)[:, 2]
        tilt_ok = up_z >= math.cos(math.radians(c.bottle_tilt_deg))
        xy_ok = (p[:, :2] - self.layout[:, 0:2]).norm(dim=-1) < c.bottle_xy_tol
        z_ok = (p[:, 2] - c.bottle_rest_z()).abs() < c.bottle_z_tol
        return tilt_ok & xy_ok & z_ok

    def deployed(self) -> torch.Tensor:
        """(N,) bool: the cradle stands upright on the corridor floor."""
        c = self.cfg
        p = self._local(self.cradle)
        up_z = self._up(self.cradle)[:, 2]
        return (up_z >= math.cos(math.radians(c.dep_tilt_deg))) \
            & ((p[:, 2] - c.court_z1).abs() <= c.dep_z_tol) \
            & (p[:, 0] >= c.dep_x[0]) & (p[:, 0] <= c.dep_x[1]) \
            & (p[:, 1].abs() <= c.dep_y_max)

    def committed(self) -> torch.Tensor:
        """(N,) bool: mast pitched past the point of no return."""
        return self.phi() >= math.radians(self.cfg.commit_deg)

    def arrested(self) -> torch.Tensor:
        """(N,) bool: the mast at REST in the catch band with the bottle safe.
        The angular-velocity gate keeps this from firing mid-sweep as a doomed
        (cradle-less) fall passes through the band on its way to the floor."""
        return self.in_band() & (self.mast_w().abs() < self.cfg.settle_ang) & self.bottle_ok()

    def settled(self) -> torch.Tensor:
        """(N,) bool: mast rotation slow, bottle and cradle slow."""
        c = self.cfg
        return (self.mast.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang) \
            & (self.bottle.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.cradle.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.mast.data.root_pos_w, self.cradle.data.root_pos_w,
                         self.bottle.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        fin = self._finite()
        self._deploy_l |= self.deployed() & fin
        self._commit_l |= self.committed() & fin
        self._arrest_l |= self.arrested() & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the mast at rest INSIDE the catch band (only the cradle
        can hold it there: the floor rest angle is asserted beyond the band and
        nothing else stands in the corridor) with the bottle still upright on
        its marker pad, everything settled and finite — a LIVE physical
        outcome. A cradle-less fell can never pass: both no-cradle rest poses
        (floor ~116 deg, propped on the bottle cap >= ~107 deg) are asserted
        beyond the band's upper edge (order forcing in cfg)."""
        self._update_latches()
        return self.in_band() & self.bottle_ok() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20 deployed + 0.20 committed + 0.20 arrested,
        all latched, capped at 0.60; exactly 1.0 iff success() holds live.
        Doing nothing scores ~0; the seed's shove-the-DOF reflex (fell with no
        cradle) parks the mast out of band either way and tops out at the
        single commit latch (0.20)."""
        c = self.cfg
        self._update_latches()
        base = (c.w_deploy * self._deploy_l.float() + c.w_commit * self._commit_l.float()
                + c.w_arrest * self._arrest_l.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="mast_lowering", robot="null"))
