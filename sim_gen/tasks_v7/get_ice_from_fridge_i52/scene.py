"""IceDoserScene — meter EXACTLY TWO ice balls into the blue cup with a
single-ball dosing shuttle (sim_gen task `get_ice_from_fridge_i52`).

Derived from rlbench/get_ice_from_fridge, but STRATEGICALLY different: the seed
is "hold a cup against a fridge lever" — one grasp, one continuous press-and-hold,
quantity never judged. Here the dispenser is a VOLUMETRIC DOSING VALVE (the candy
machine / ice-maker chute mechanism): a sealed single-file hopper of five ice
balls stands over a horizontal SHUTTLE that rides in a tunnel and carries a
one-ball POCKET. Only two shuttle states exist, both against hard stops:

  CLOSED (rest): pocket under the hopper throat — one ball sits in the pocket,
      the outlet in the tunnel floor is covered by the shuttle body.
  OPEN: shuttle pushed one stroke toward the outlet — the pocket crosses the
      outlet and its ball falls through; meanwhile the shuttle body seals the
      throat, so NO MORE BALLS can follow. Holding the valve open dispenses
      nothing further (smoke proves it): the machine meters ONE ball per full
      close->open->close cycle, by geometry, not timing.

The task: slide the BLUE cup under the outlet drop zone, pump the shuttle
end-to-end exactly TWICE (exactly two balls into the blue cup), push it back to
its CLOSED stop, and leave the RED decoy cup empty with the three remaining
balls still sealed inside the machine. The hopper is capped and the tunnel
enclosed: the only physical path from hopper to cup is through the pocket, so
every counted ball certifies a full mechanism cycle. Quantity IS the task —
a third cycle (or holding the valve open over the cup after teleport-cheating
more balls in) is an overfill and fails terminally, because balls cannot be
put back into the sealed machine.

Plan-level contrast with the seed (and with every tasks_v7 task read while
building this): the objective is a COUNT with failure on both sides, reached by
a discrete reciprocating pump cycle (push protruding knob to hard stop, push
opposite knob back), preceded by a required STAGING move (cup under the outlet
BEFORE the first cycle — a ball dispensed early is lost on the ground and the
task is unrecoverable). No lever hold, no pouring, no articulated door, no
pose-tracking: the solver must operate a metering machine a discrete number of
times and then STOP.

Assets are fully procedural (compound-spawner pattern; children of one body
never self-collide). All mechanism surfaces get a bound LOW-FRICTION physics
material — PhysX's default 0.5 friction force-closes ball valves.
  - housing (KINEMATIC compound): ground pedestal, elevated tunnel (floor with
    outlet hole, side walls, roof with throat hole), capped single-file hopper
    shaft, green outlet markers. The tunnel cantilevers over open ground on the
    outlet side so a cup (and a hand) fits under the drop zone.
  - shuttle (DYNAMIC compound): tunnel-guided slide with a through-pocket sized
    for one ball, and a knob plate at each end. The knobs are wider than the
    tunnel bore: they are the push handles AND the travel hard stops
    (closed stop / open stop). Pocket depth == ball diameter, so the ball above
    shears off cleanly at the shuttle's top plane (never jams, never doubles).
  - blue cup / red cup (DYNAMIC compounds): floor disc + 12-segment wall ring.
    Blue is the target, red the decoy; both start away from the machine.
  - 5 ice balls (spheres, r 16 mm), spawned as a column in the shaft.

Per-episode randomization (readback-verified): housing yaw FREE (+/-180 deg) +
xy jitter; both cup slots at random bearings/radii around the machine (blue and
red independently, min angular separation).

Rubric (0..1; latched stages anchored in the demonstrated solve):
  0.15  blue cup ever staged under the outlet (settled, upright, on the ground)
  0.25  first ball in the blue cup (red still empty)
  0.30  target count (2) in the blue cup (red still empty)
  capped at 0.70; exactly 1.0 iff success(): blue holds EXACTLY target balls,
  red empty, the other three balls retained inside the machine, shuttle parked
  at its CLOSED stop, everything settled and finite. Null policy ~0; the seed's
  press-and-hold scores at most the 1-ball stages and can never reach success;
  dispensing without staging the cup scores ~0 and is terminal; overfill keeps
  its earned stage credit but success is impossible.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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


def _friction_material(stage, path: str, static: float, dynamic: float):
    """A UsdShade material carrying UsdPhysics friction — custom-spawner colliders
    default to ~0.5 friction unless one is BOUND, and 0.5 force-closes a ball
    valve."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _make_collide(contact_offset: float, material=None) -> Callable:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
        if material is not None:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics")

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable, yaw_deg=None):
    """One box child: translate (+ optional z-rotation) + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw_deg is not None:
        xf.AddRotateZOp().Set(float(yaw_deg))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable):
    """One z-axis cylinder child: translate only, displayColor, collider."""
    from pxr import Gf, UsdGeom

    r, h = float(radius), float(height)
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(r)
    cyl.CreateHeightAttr(h)
    cyl.CreateAxisAttr("Z")
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    return cyl.GetPrim()


def _apply_dynamic(root, mass: float, lin_damp: float, ang_damp: float) -> None:
    """Make a compound root a DYNAMIC rigid body with an explicit total mass."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateLinearDampingAttr(float(lin_damp))
    prb.CreateAngularDampingAttr(float(ang_damp))
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)


def _spawn_housing(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the dispenser housing at `prim_path`: KINEMATIC compound. Local
    frame: origin on the GROUND under the hopper/throat axis, +x toward the
    outlet (the cantilever side), +z up."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    slick = _friction_material(stage, f"{prim_path}/slickMat",
                               cfg.mech_mu_s, cfg.mech_mu_d)
    collide = _make_collide(cfg.contact_offset, slick)
    c = cfg
    # pedestal: solid block from the ground to the tunnel underside, left half only
    # (the outlet side cantilevers over open ground so a cup fits underneath)
    _add_box(stage, f"{prim_path}/pedestal",
             center=((c.ped_x0 + c.ped_x1) / 2, 0.0, c.floor_z0 / 2),
             size=(c.ped_x1 - c.ped_x0, 2 * (c.bore_hw + c.wall_t), c.floor_z0),
             color=c.body_color, collide=collide)
    # tunnel floor (hole = outlet, x in [out_x0, out_x1], |y| < throat_hw)
    fz = (c.floor_z0 + c.floor_z1) / 2
    ft = c.floor_z1 - c.floor_z0
    _add_box(stage, f"{prim_path}/floor_left",
             center=((-c.hx + c.out_x0) / 2, 0.0, fz),
             size=(c.out_x0 + c.hx, 2 * c.bore_hw, ft),
             color=c.body_color, collide=collide)
    _add_box(stage, f"{prim_path}/floor_right",
             center=((c.out_x1 + c.hx) / 2, 0.0, fz),
             size=(c.hx - c.out_x1, 2 * c.bore_hw, ft),
             color=c.body_color, collide=collide)
    for sy, t in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/floor_strip_{t}",
                 center=((c.out_x0 + c.out_x1) / 2,
                         sy * (c.throat_hw + c.bore_hw) / 2, fz),
                 size=(c.out_x1 - c.out_x0, c.bore_hw - c.throat_hw, ft),
                 color=c.outlet_color, collide=collide)
    # tunnel side walls
    for sy, t in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/wall_{t}",
                 center=(0.0, sy * (c.bore_hw + c.wall_t / 2),
                         (c.floor_z0 + c.roof_z1) / 2),
                 size=(2 * c.hx, c.wall_t, c.roof_z1 - c.floor_z0),
                 color=c.body_color, collide=collide)
    # tunnel roof (hole = throat, |x| < throat_hw, |y| < throat_hw)
    rz = (c.roof_z0 + c.roof_z1) / 2
    rt = c.roof_z1 - c.roof_z0
    _add_box(stage, f"{prim_path}/roof_left",
             center=((-c.hx - c.throat_hw) / 2, 0.0, rz),
             size=(c.hx - c.throat_hw, 2 * c.bore_hw, rt),
             color=c.body_color, collide=collide)
    _add_box(stage, f"{prim_path}/roof_right",
             center=((c.hx + c.throat_hw) / 2, 0.0, rz),
             size=(c.hx - c.throat_hw, 2 * c.bore_hw, rt),
             color=c.body_color, collide=collide)
    for sy, t in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/roof_strip_{t}",
                 center=(0.0, sy * (c.throat_hw + c.bore_hw) / 2, rz),
                 size=(2 * c.throat_hw, c.bore_hw - c.throat_hw, rt),
                 color=c.body_color, collide=collide)
    # hopper shaft: single-file vertical guide over the throat, CAPPED (the only
    # way out of the machine is through the shuttle pocket)
    sz = (c.shaft_z0 + c.shaft_z1) / 2
    sh = c.shaft_z1 - c.shaft_z0
    for sx, t in ((1.0, "xp"), (-1.0, "xn")):
        _add_box(stage, f"{prim_path}/shaft_{t}",
                 center=(sx * (c.throat_hw + c.shaft_t / 2), 0.0, sz),
                 size=(c.shaft_t, 2 * (c.throat_hw + c.shaft_t), sh),
                 color=c.shaft_color, collide=collide)
    for sy, t in ((1.0, "yp"), (-1.0, "yn")):
        _add_box(stage, f"{prim_path}/shaft_{t}",
                 center=(0.0, sy * (c.throat_hw + c.shaft_t / 2), sz),
                 size=(2 * c.throat_hw, c.shaft_t, sh),
                 color=c.shaft_color, collide=collide)
    _add_box(stage, f"{prim_path}/cap",
             center=(0.0, 0.0, c.shaft_z1 + c.cap_t / 2),
             size=(2 * (c.throat_hw + c.shaft_t), 2 * (c.throat_hw + c.shaft_t),
                   c.cap_t),
             color=c.cap_color, collide=collide)
    return root


def _spawn_shuttle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the dosing shuttle at `prim_path`: DYNAMIC compound. Local frame:
    ORIGIN AT THE POCKET CENTER (so the shuttle's housing-local x IS the stroke
    displacement d: d=0 pocket under the throat, d=stroke pocket over the
    outlet). Pocket depth == ball diameter: the ball above the pocket ball rests
    exactly on the shuttle's top plane and shears off cleanly when the shuttle
    moves — the geometric one-ball meter."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _apply_dynamic(root, cfg.mass, 0.8, 0.8)
    slick = _friction_material(stage, f"{prim_path}/slickMat",
                               cfg.mech_mu_s, cfg.mech_mu_d)
    collide = _make_collide(cfg.contact_offset, slick)
    c = cfg
    # core blocks flanking the pocket (through hole, |x| < pocket_hw, |y| < pocket_hw)
    _add_box(stage, f"{prim_path}/core_left",
             center=((c.sh_x0 - c.pocket_hw) / 2, 0.0, 0.0),
             size=(-c.pocket_hw - c.sh_x0, 2 * c.sh_hy, 2 * c.sh_hz),
             color=c.shuttle_color, collide=collide)
    _add_box(stage, f"{prim_path}/core_right",
             center=((c.pocket_hw + c.sh_x1) / 2, 0.0, 0.0),
             size=(c.sh_x1 - c.pocket_hw, 2 * c.sh_hy, 2 * c.sh_hz),
             color=c.shuttle_color, collide=collide)
    for sy, t in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/rail_{t}",
                 center=(0.0, sy * (c.pocket_hw + c.sh_hy) / 2, 0.0),
                 size=(2 * c.pocket_hw, c.sh_hy - c.pocket_hw, 2 * c.sh_hz),
                 color=c.shuttle_color, collide=collide)
    # knob plates: push handles AND travel hard stops (wider/taller than the bore)
    for x0, t in ((c.sh_x0 - c.knob_t, "left"), (c.sh_x1, "right")):
        _add_box(stage, f"{prim_path}/knob_{t}",
                 center=(x0 + c.knob_t / 2, 0.0, c.knob_zc),
                 size=(c.knob_t, 2 * c.knob_hy, c.knob_hz),
                 color=c.knob_color, collide=collide)
    return root


def _spawn_cup(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one cup at `prim_path`: DYNAMIC compound, root at the BOTTOM
    CENTER. Floor disc + 12-segment wall ring."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _apply_dynamic(root, cfg.mass, 0.3, 0.5)
    grip = _friction_material(stage, f"{prim_path}/gripMat",
                              cfg.cup_mu, cfg.cup_mu * 0.9)
    collide = _make_collide(cfg.contact_offset, grip)
    c = cfg
    _add_cyl(stage, f"{prim_path}/floor",
             center=(0.0, 0.0, c.floor_t / 2),
             radius=c.in_r + c.wall_t, height=c.floor_t,
             color=c.color, collide=collide)
    n_seg = 12
    rc = c.in_r + c.wall_t / 2
    seg_len = 2 * rc * math.tan(math.pi / n_seg) + 0.004
    for i in range(n_seg):
        a = 2 * math.pi * i / n_seg
        _add_box(stage, f"{prim_path}/wall_{i}",
                 center=(rc * math.cos(a), rc * math.sin(a), c.wall_h / 2),
                 size=(c.wall_t, seg_len, c.wall_h),
                 color=c.color, collide=collide, yaw_deg=math.degrees(a))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "housing" not in _SPAWNER_CACHE:

        @configclass
        class HousingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_housing)
            hx: float = 0.100
            bore_hw: float = 0.0335
            wall_t: float = 0.010
            floor_z0: float = 0.100
            floor_z1: float = 0.110
            roof_z0: float = 0.146
            roof_z1: float = 0.156
            throat_hw: float = 0.020
            out_x0: float = 0.040
            out_x1: float = 0.080
            ped_x0: float = -0.100
            ped_x1: float = -0.025
            shaft_t: float = 0.008
            shaft_z0: float = 0.156
            shaft_z1: float = 0.326
            cap_t: float = 0.008
            mech_mu_s: float = 0.12
            mech_mu_d: float = 0.10
            body_color: tuple = (0.45, 0.48, 0.55)
            shaft_color: tuple = (0.60, 0.63, 0.70)
            cap_color: tuple = (0.20, 0.20, 0.24)
            outlet_color: tuple = (0.10, 0.75, 0.20)
            contact_offset: float = 0.0015

        @configclass
        class ShuttleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shuttle)
            mass: float = 0.15
            sh_x0: float = -0.160
            sh_x1: float = 0.100
            sh_hy: float = 0.031
            sh_hz: float = 0.016
            pocket_hw: float = 0.019
            knob_t: float = 0.012
            knob_hy: float = 0.045
            knob_hz: float = 0.047
            knob_zc: float = 0.0025
            mech_mu_s: float = 0.12
            mech_mu_d: float = 0.10
            shuttle_color: tuple = (0.80, 0.72, 0.30)
            knob_color: tuple = (0.90, 0.55, 0.10)
            contact_offset: float = 0.0015

        @configclass
        class CupSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cup)
            mass: float = 0.10
            in_r: float = 0.036
            wall_t: float = 0.006
            wall_h: float = 0.075
            floor_t: float = 0.008
            cup_mu: float = 0.60
            color: tuple = (0.10, 0.20, 0.85)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["housing"] = HousingSpawnerCfg
        _SPAWNER_CACHE["shuttle"] = ShuttleSpawnerCfg
        _SPAWNER_CACHE["cup"] = CupSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class IceDoserSceneCfg(BaseCfg):
    """Config for `IceDoserScene`. The load-bearing invariants — one ball per
    cycle, the sealed hopper, both stroke hard stops, the throat sealed while
    open, the cup catching every properly staged drop — are all geometric and
    asserted below."""

    # --- tunable: task ---------------------------------------------------------------------------
    target_count: int = tunable(2)         # balls the blue cup must END with (exactly)
    n_balls: int = tunable(5)              # balls loaded in the hopper at reset

    # --- tunable: rubric thresholds --------------------------------------------------------------
    stage_xy: float = tunable(0.015)       # blue cup center within this of the drop line
    incup_xy: float = tunable(0.028)       # ball-in-cup: center within this of the cup axis
    #   (honest by construction: max physical in-cup offset = in_r - ball_r = 20 mm < 28 mm;
    #    a ball OUTSIDE the wall is >= 58 mm off-axis)
    incup_z0: float = tunable(0.010)       # ball-in-cup: center above the cup floor (cup frame)
    incup_z1: float = tunable(0.085)       # ball-in-cup: center below rim + ball_r (cup frame)
    closed_tol: float = tunable(0.015)     # shuttle parked at CLOSED: |d| below this
    cup_up_z: float = tunable(0.90)        # cup counts as upright above this up-vector z
    settle_lin: float = tunable(0.05)      # judge gate: max |v| balls + cups (m/s)
    settle_shu: float = tunable(0.03)      # judge gate: max shuttle |v| (m/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_deg: float = tunable(180.0)        # housing yaw uniform +/- (FREE heading)
    xy_jitter: float = tunable(0.04)       # housing xy jitter (+/- m)
    slot_r: tuple = tunable((0.26, 0.34))  # cup slot radius range around the housing
    slot_sep_deg: float = tunable(55.0)    # min angular separation between the two cups

    # --- info: housing (local frame: origin on the ground under the hopper axis) -----------------
    ball_r: float = info(0.016)
    ball_mass: float = info(0.030)
    hx: float = info(0.100)                # housing half-length in x
    bore_hw: float = info(0.0335)          # tunnel bore half-width (y)
    wall_t: float = info(0.010)
    floor_z0: float = info(0.100)          # tunnel floor underside (cup slides below)
    floor_z1: float = info(0.110)          # tunnel floor top = shuttle ride plane
    roof_z0: float = info(0.146)           # tunnel roof underside (4 mm over the shuttle)
    roof_z1: float = info(0.156)
    throat_hw: float = info(0.020)         # hopper throat / roof hole half-width
    out_x0: float = info(0.040)            # outlet hole in the floor: x span
    out_x1: float = info(0.080)
    ped_x0: float = info(-0.100)           # pedestal x span (outlet side stays open)
    ped_x1: float = info(-0.025)
    shaft_t: float = info(0.008)
    shaft_z0: float = info(0.156)
    shaft_z1: float = info(0.326)
    cap_t: float = info(0.008)
    # --- info: shuttle (root AT the pocket center; local x in housing frame == stroke d) ---------
    shuttle_mass: float = info(0.15)
    sh_x0: float = info(-0.160)            # core left end (shuttle frame)
    sh_x1: float = info(0.100)             # core right end
    sh_hy: float = info(0.031)             # core half-width (y)
    sh_hz: float = info(0.016)             # core half-thickness == ball_r (shear plane!)
    pocket_hw: float = info(0.019)         # pocket half-width (one-ball through hole)
    knob_t: float = info(0.012)
    knob_hy: float = info(0.045)
    knob_hz: float = info(0.047)           # knob plate height (blocks the bore, rides over
    knob_zc: float = info(0.0025)          # the pedestal: bottom at world z 0.105)
    stroke: float = info(0.060)            # CLOSED stop d=0 ... OPEN stop d=stroke
    # --- info: cups ------------------------------------------------------------------------------
    cup_in_r: float = info(0.036)
    cup_wall_t: float = info(0.006)
    cup_wall_h: float = info(0.075)
    cup_floor_t: float = info(0.008)
    cup_mass: float = info(0.10)
    blue_color: tuple = info((0.10, 0.20, 0.85))
    red_color: tuple = info((0.85, 0.10, 0.10))
    # --- info: friction / misc -------------------------------------------------------------------
    mech_mu_s: float = info(0.12)          # mechanism surfaces: bound slick material
    mech_mu_d: float = info(0.10)
    cup_mu: float = info(0.60)
    ball_mu: float = info(0.20)
    contact_offset: float = info(0.0015)
    # retained-in-machine box (housing local)
    keep_x: float = info(0.095)
    keep_y: float = info(0.035)
    keep_z0: float = info(0.105)
    keep_z1: float = info(0.340)
    # rubric weights (0.15 + 0.25 + 0.30 = 0.70 = the non-success cap)
    w_stage: float = info(0.15)
    w_one: float = info(0.25)
    w_all: float = info(0.30)

    # Derived (filled in __post_init__).
    ball_d: float = field(default=0.0, init=False)
    out_c: float = field(default=0.0, init=False)   # outlet / drop-line x (housing local)

    def __post_init__(self) -> None:
        self.ball_d = 2 * self.ball_r
        self.out_c = (self.out_x0 + self.out_x1) / 2

        assert 2 <= self.target_count <= self.n_balls - 2, \
            "need balls left over so overfill AND retention are both live clauses"
        # --- the one-ball meter, by geometry ---
        assert abs(2 * self.sh_hz - self.ball_d) < 1e-9, \
            "pocket depth must EQUAL the ball diameter: the ball above rests on the " \
            "shuttle top plane and shears cleanly (deeper doubles, shallower jams the roof)"
        assert 2 * self.pocket_hw >= self.ball_d + 0.004, "ball must drop freely into the pocket"
        assert 2 * self.pocket_hw <= 2 * self.throat_hw, "pocket no wider than the throat"
        assert 2 * self.throat_hw >= self.ball_d + 0.006, "ball must fall freely down the shaft"
        assert self.roof_z0 - (self.floor_z1 + 2 * self.sh_hz) >= 0.0035, \
            "shuttle must slide under the roof with clearance beyond both contact offsets"
        assert self.bore_hw - self.sh_hy >= 0.002, "shuttle must slide in the bore with play"
        # --- stroke hard stops (knobs vs housing end faces) ---
        assert abs((-self.sh_x0 - self.hx) - self.stroke) < 1e-9, \
            "OPEN stop: left knob face hits the housing at exactly d = stroke"
        assert abs(self.sh_x1 - self.hx) < 1e-9, \
            "CLOSED stop: right knob face hits the housing at exactly d = 0"
        assert abs(self.out_c - self.stroke) < 1e-9, \
            "at the OPEN stop the pocket center sits exactly on the outlet center"
        assert self.knob_hy > self.bore_hw + 0.008 and self.knob_hz > 0.040, \
            "knobs must be wider than the bore (they are the travel stops)"
        assert self.knob_zc - self.knob_hz / 2 + 0.126 >= self.floor_z0 + 0.004, \
            "knob bottom must ride OVER the pedestal top"
        # --- sealing: exactly one hole open at a time, with margin for stop offsets ---
        d_open = self.stroke - 0.004        # worst-case arrest short of the rigid stop
        assert (-self.pocket_hw + d_open) >= self.throat_hw + 0.010, \
            "at OPEN the core seals the throat (no second ball can follow)"
        assert self.sh_x1 >= self.out_x1 + 0.015 and -self.pocket_hw <= self.out_x0 - 0.010, \
            "at CLOSED the core seals the outlet"
        ov_open = min(d_open + self.pocket_hw, self.out_x1) - max(d_open - self.pocket_hw,
                                                                  self.out_x0)
        assert ov_open >= self.ball_d + 0.002, "ball falls through the outlet at the OPEN stop"
        ov_load = min(0.004 + self.pocket_hw, self.throat_hw) - max(0.004 - self.pocket_hw,
                                                                    -self.throat_hw)
        assert ov_load >= self.ball_d + 0.002, "ball loads into the pocket at the CLOSED stop"
        # --- sealed hopper: cap above the full stack with headroom ---
        stack_top = self.floor_z1 + self.ball_r + (self.n_balls - 1) * self.ball_d + self.ball_r
        assert self.shaft_z1 >= stack_top + 0.030, "shaft must hold the full column + headroom"
        assert self.shaft_z0 == self.roof_z1, "shaft sits on the roof"
        # --- the drop: cup fits under the cantilever and catches every staged ball ---
        assert self.floor_z0 >= self.cup_wall_h + 0.015, \
            "cup (and a fingertip above its rim) must fit under the tunnel floor"
        assert self.out_x0 - self.ped_x1 >= 0.05, "pedestal must stay clear of the drop zone"
        # ball leaves the outlet within (hole_hw - ball_r) of the drop line, drifts
        # < 15 mm during the 90 mm fall at shuttle speeds; cup staged within stage_xy:
        scatter = (self.out_x1 - self.out_x0) / 2 - self.ball_r + 0.015 + self.stage_xy
        assert scatter <= self.cup_in_r + 0.003, "staged cup catches every dispensed ball"
        assert self.cup_in_r >= 2 * self.ball_r + 0.002, \
            "cup floor takes two balls side by side (third rests on top, below the rim)"
        # in-cup honesty bounds
        assert self.incup_xy > self.cup_in_r - self.ball_r, \
            "any ball physically inside the cup must count"
        assert self.incup_xy < self.cup_in_r + self.cup_wall_t + self.ball_r - 0.004, \
            "a ball leaning outside the cup wall must NOT count"
        # --- slots clear the machine and the staging zone ---
        reach = max(self.hx + 2 * self.knob_t + self.stroke,
                    math.hypot(self.hx, self.bore_hw + self.wall_t))
        cup_out = self.cup_in_r + self.cup_wall_t
        assert self.slot_r[0] - cup_out >= reach + 0.015, "cup slots clear the machine"
        assert self.slot_r[0] - cup_out >= self.out_c + cup_out + self.stage_xy + 0.02, \
            "no cup can spawn already staged"
        assert self.keep_x >= self.stroke + self.pocket_hw + 0.005, \
            "a ball riding the pocket at OPEN still counts as retained"


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("ice_doser")
class IceDoserScene(BaseScene):
    cfg: IceDoserSceneCfg

    def __init__(self, cfg: IceDoserSceneCfg | None = None) -> None:
        super().__init__(cfg or IceDoserSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        housing_spawn = cls["housing"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            hx=c.hx, bore_hw=c.bore_hw, wall_t=c.wall_t, floor_z0=c.floor_z0,
            floor_z1=c.floor_z1, roof_z0=c.roof_z0, roof_z1=c.roof_z1,
            throat_hw=c.throat_hw, out_x0=c.out_x0, out_x1=c.out_x1,
            ped_x0=c.ped_x0, ped_x1=c.ped_x1, shaft_t=c.shaft_t,
            shaft_z0=c.shaft_z0, shaft_z1=c.shaft_z1, cap_t=c.cap_t,
            mech_mu_s=c.mech_mu_s, mech_mu_d=c.mech_mu_d,
            contact_offset=c.contact_offset)
        shuttle_spawn = cls["shuttle"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass=c.shuttle_mass, sh_x0=c.sh_x0, sh_x1=c.sh_x1, sh_hy=c.sh_hy,
            sh_hz=c.sh_hz, pocket_hw=c.pocket_hw, knob_t=c.knob_t,
            knob_hy=c.knob_hy, knob_hz=c.knob_hz, knob_zc=c.knob_zc,
            mech_mu_s=c.mech_mu_s, mech_mu_d=c.mech_mu_d,
            contact_offset=c.contact_offset)

        def cup_spawn(color):
            return cls["cup"](
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass=c.cup_mass, in_r=c.cup_in_r, wall_t=c.cup_wall_t,
                wall_h=c.cup_wall_h, floor_t=c.cup_floor_t, cup_mu=c.cup_mu,
                color=color, contact_offset=c.contact_offset)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "housing": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Housing", spawn=housing_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "shuttle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shuttle", spawn=shuttle_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.127))),
            "cup_blue": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/CupBlue", spawn=cup_spawn(c.blue_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.30, 0.001))),
            "cup_red": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/CupRed", spawn=cup_spawn(c.red_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, -0.30, 0.001))),
        }
        for i in range(c.n_balls):
            out[f"ball_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball_" + str(i),
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.15, angular_damping=0.25),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.ball_mu, dynamic_friction=c.ball_mu * 0.9,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.86, 0.93, 0.97)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.128 + i * (c.ball_d + 0.001))),
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
        c = self.cfg
        self.housing: RigidObject = env.iscene["housing"]
        self.shuttle: RigidObject = env.iscene["shuttle"]
        self.cups: dict[str, RigidObject] = {
            "blue": env.iscene["cup_blue"], "red": env.iscene["cup_red"]}
        self.balls: list[RigidObject] = [env.iscene[f"ball_{i}"]
                                         for i in range(c.n_balls)]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke)
        self.housing_yaw = torch.zeros(n, device=dev)
        self.slot_ang = torch.zeros(n, 2, device=dev)    # blue, red world bearings
        # latches (partial credit survives regressions; success is judged live)
        self._l_stage = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l_one = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l_all = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the housing (free yaw + xy jitter), park the
        shuttle at its CLOSED stop, drop the ball column down the shaft (the
        bottom ball settles into the pocket), scatter the two cups at random
        bearings around the machine, clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        self.housing_yaw[env_ids] = yaw
        q_yaw = _qz(yaw)
        hp = torch.zeros(m, 3, device=dev)
        hp[:, 0:2] = (torch.rand(m, 2, device=dev) * 2 - 1) * c.xy_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = hp + origin
        st[:, 3:7] = q_yaw
        self.housing.write_root_state_to_sim(st, env_ids)

        cy, sy = torch.cos(yaw), torch.sin(yaw)

        def local_to_world(lx, ly, lz):
            p = torch.zeros(m, 3, device=dev)
            p[:, 0] = hp[:, 0] + lx * cy - ly * sy
            p[:, 1] = hp[:, 1] + lx * sy + ly * cy
            p[:, 2] = lz
            return p + origin

        # shuttle: 3 mm off the CLOSED stop (avoids spawning in contact), 1 mm proud
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = local_to_world(0.003, 0.0, c.floor_z1 + c.sh_hz + 0.001)
        st[:, 3:7] = q_yaw
        self.shuttle.write_root_state_to_sim(st, env_ids)

        # balls: a column down the shaft axis; the bottom one lands in the pocket
        for i, ball in enumerate(self.balls):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = local_to_world(
                0.003, 0.0, c.floor_z1 + c.ball_r + 0.002 + i * (c.ball_d + 0.001))
            st[:, 3] = 1.0
            ball.write_root_state_to_sim(st, env_ids)

        # cups: random bearings/radii around the machine, min angular separation
        a_b = torch.rand(m, device=dev) * 2 * math.pi
        sep_min = math.radians(c.slot_sep_deg)
        a_r = a_b + sep_min + torch.rand(m, device=dev) * (2 * math.pi - 2 * sep_min)
        self.slot_ang[env_ids, 0] = a_b
        self.slot_ang[env_ids, 1] = a_r % (2 * math.pi)
        for k, (name, cup) in enumerate(self.cups.items()):
            ang = self.slot_ang[env_ids, k]
            r = c.slot_r[0] + torch.rand(m, device=dev) * (c.slot_r[1] - c.slot_r[0])
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = hp[:, 0] + r * torch.cos(ang)
            st[:, 1] = hp[:, 1] + r * torch.sin(ang)
            st[:, 2] = 0.001
            st[:, 3:7] = _qz(torch.rand(m, device=dev) * 2 * math.pi)
            st[:, 0:3] += origin
            cup.write_root_state_to_sim(st, env_ids)

        self._l_stage[env_ids] = False
        self._l_one[env_ids] = False
        self._l_all[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "housing": self.housing.data.root_state_w[env_ids].clone(),
            "shuttle": self.shuttle.data.root_state_w[env_ids].clone(),
            "cups": {n: b.data.root_state_w[env_ids].clone()
                     for n, b in self.cups.items()},
            "balls": [b.data.root_state_w[env_ids].clone() for b in self.balls],
            "housing_yaw": self.housing_yaw[env_ids].clone(),
            "slot_ang": self.slot_ang[env_ids].clone(),
            "l_stage": self._l_stage[env_ids].clone(),
            "l_one": self._l_one[env_ids].clone(),
            "l_all": self._l_all[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.housing.write_root_state_to_sim(state["housing"], env_ids)
        self.shuttle.write_root_state_to_sim(state["shuttle"], env_ids)
        for n, b in self.cups.items():
            b.write_root_state_to_sim(state["cups"][n], env_ids)
        for b, s in zip(self.balls, state["balls"]):
            b.write_root_state_to_sim(s, env_ids)
        self.housing_yaw[env_ids] = state["housing_yaw"]
        self.slot_ang[env_ids] = state["slot_ang"]
        self._l_stage[env_ids] = state["l_stage"]
        self._l_one[env_ids] = state["l_one"]
        self._l_all[env_ids] = state["l_all"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"An ICE DISPENSER stands on the ground: a capped vertical hopper "
            f"tube holding {c.n_balls} white ice balls "
            f"({c.ball_d * 1000:.0f} mm) feeds a horizontal DOSING SHUTTLE that "
            f"rides in an enclosed tunnel. The tunnel rests on a pedestal at one "
            f"end and CANTILEVERS over open ground at the other; on that open "
            f"side, GREEN MARKS on the tunnel's underside flank the OUTLET hole "
            f"where balls drop out. The shuttle has an orange KNOB at each end "
            f"and slides between two hard stops. At rest it sits at the CLOSED "
            f"stop (the outlet-side knob flush against the housing, the other "
            f"knob protruding): one ball waits in the shuttle's internal pocket "
            f"under the hopper. Pushing the PROTRUDING knob in until it stops "
            f"carries the pocket over the outlet — that one ball falls out — "
            f"while the shuttle body seals the hopper, so HOLDING it open yields "
            f"nothing more: the machine dispenses EXACTLY ONE ball per full "
            f"push-push cycle (push the protruding knob to the stop, then push "
            f"the other knob back to the CLOSED stop, which reloads the pocket). "
            f"The hopper is capped and the tunnel enclosed — balls can leave "
            f"only through the outlet. Two open cups sit on the ground nearby: "
            f"a BLUE cup and a RED cup; positions and the machine's heading "
            f"change every episode, so read the scene.\n"
            f"Goal: first slide the BLUE cup under the outlet (under the green "
            f"marks, within about {c.stage_xy * 1000:.0f} mm of the drop line — "
            f"a ball dispensed with no cup there is lost on the ground and the "
            f"task cannot be completed), then work the shuttle through exactly "
            f"{c.target_count} full cycles so EXACTLY {c.target_count} ice balls "
            f"end up in the blue cup, and finish with the shuttle pushed back to "
            f"its CLOSED stop. The RED cup must stay empty and the remaining "
            f"{c.n_balls - c.target_count} balls must remain inside the machine. "
            f"More than {c.target_count} balls in the blue cup, any ball in the "
            f"red cup or on the ground, or the shuttle left open, all fail."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        c = self.cfg
        return (
            f"Slide the blue cup under the dispenser outlet, then pump the "
            f"shuttle end-to-end exactly {c.target_count} times to drop exactly "
            f"{c.target_count} ice balls into it, and push the shuttle back to "
            f"its closed stop. No ball may end up in the red cup or on the "
            f"ground, and no extra ball may be dispensed."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _housing_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.housing.data.root_quat_w,
                                  pos_w - self.housing.data.root_pos_w)

    def shuttle_d(self) -> torch.Tensor:
        """(N,) stroke displacement: shuttle root (== pocket center) housing-local x."""
        return self._housing_local(self.shuttle.data.root_pos_w)[:, 0]

    def _ball_pos(self) -> torch.Tensor:
        return torch.stack([b.data.root_pos_w for b in self.balls], dim=1)  # (N,B,3)

    def _ball_vel(self) -> torch.Tensor:
        return torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                            for b in self.balls], dim=1)                    # (N,B)

    def cup_upright(self, name: str) -> torch.Tensor:
        q = self.cups[name].data.root_quat_w
        w, x, y, z = q.unbind(-1)
        return (1 - 2 * (x * x + y * y)) > self.cfg.cup_up_z

    def in_cup(self, name: str) -> torch.Tensor:
        """(N, B) bool, geometric: ball center inside the cup's interior volume,
        judged in the CUP'S BODY FRAME, cup upright."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        cup = self.cups[name]
        pos = self._ball_pos()
        n, b = pos.shape[0], pos.shape[1]
        cq = cup.data.root_quat_w[:, None, :].expand(n, b, 4).reshape(n * b, 4)
        cp = cup.data.root_pos_w[:, None, :]
        loc = quat_apply_inverse(cq, (pos - cp).reshape(n * b, 3)).reshape(n, b, 3)
        near = loc[:, :, :2].norm(dim=-1) < c.incup_xy
        inz = (loc[:, :, 2] > c.incup_z0) & (loc[:, :, 2] < c.incup_z1)
        return near & inz & self.cup_upright(name).unsqueeze(-1)

    def cup_count(self, name: str) -> torch.Tensor:
        return self.in_cup(name).sum(dim=1)

    def retained(self) -> torch.Tensor:
        """(N, B) bool: ball inside the machine (shaft, pocket, or riding the
        shuttle top), in the housing frame."""
        c = self.cfg
        pos = self._ball_pos()
        n, b = pos.shape[0], pos.shape[1]
        flat = self._housing_local(pos.reshape(n * b, 3)).reshape(n, b, 3)
        return (flat[:, :, 0].abs() < c.keep_x) & (flat[:, :, 1].abs() < c.keep_y) \
            & (flat[:, :, 2] > c.keep_z0) & (flat[:, :, 2] < c.keep_z1)

    def staged(self) -> torch.Tensor:
        """(N,) bool: BLUE cup upright on the ground with its axis on the drop
        line (housing local), i.e. under the outlet."""
        c = self.cfg
        loc = self._housing_local(self.cups["blue"].data.root_pos_w)
        dx = loc[:, 0] - c.out_c
        near = (dx * dx + loc[:, 1] * loc[:, 1]).sqrt() < c.stage_xy
        on_ground = loc[:, 2] < 0.012
        return near & on_ground & self.cup_upright("blue")

    def shuttle_closed(self) -> torch.Tensor:
        return self.shuttle_d().abs() < self.cfg.closed_tol

    def settled(self) -> torch.Tensor:
        c = self.cfg
        balls_ok = (self._ball_vel() < c.settle_lin).all(dim=1)
        cups_ok = torch.ones_like(balls_ok)
        for b in self.cups.values():
            cups_ok &= b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        shu_ok = self.shuttle.data.root_lin_vel_w.norm(dim=-1) < c.settle_shu
        return balls_ok & cups_ok & shu_ok

    def _finite(self) -> torch.Tensor:
        p = torch.cat([self._ball_pos(),
                       self.shuttle.data.root_pos_w[:, None, :],
                       self.cups["blue"].data.root_pos_w[:, None, :],
                       self.cups["red"].data.root_pos_w[:, None, :]], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        blue = self.cup_count("blue")
        red_empty = self.cup_count("red") == 0
        cup_slow = self.cups["blue"].data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        self._l_stage |= self.staged() & cup_slow & fin
        self._l_one |= (blue >= 1) & red_empty & fin
        self._l_all |= (blue >= c.target_count) & red_empty & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: EXACTLY target_count balls in the blue cup, red cup empty,
        every other ball retained inside the machine, shuttle parked at its
        CLOSED stop, everything settled and finite. All clauses are live; the
        only physical route for a ball into a cup is through the shuttle pocket
        (capped hopper, enclosed tunnel), so each counted ball certifies one
        full mechanism cycle."""
        c = self.cfg
        self._update_latches()
        blue = self.cup_count("blue")
        red = self.cup_count("red")
        kept = self.retained().sum(dim=1)
        return (blue == c.target_count) & (red == 0) \
            & (kept == c.n_balls - c.target_count) & self.shuttle_closed() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15 blue-cup-staged + 0.25 first-ball-in +
        0.30 target-count-in (latched), capped at 0.70; exactly 1.0 iff
        success() holds live. Null policy ~0; dispensing before staging the cup
        ~0; the seed's press-and-hold reaches at most the one-ball stages."""
        c = self.cfg
        self._update_latches()
        base = (c.w_stage * self._l_stage.float() + c.w_one * self._l_one.float()
                + c.w_all * self._l_all.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="ice_doser", robot="null"))
