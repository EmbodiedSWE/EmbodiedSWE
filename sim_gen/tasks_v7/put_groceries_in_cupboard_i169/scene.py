"""CarouselCupboardScene — stock the red carton into the EMPTY bay of a rotary
carousel cupboard, then rotate the carousel so the carton is stowed out of sight.

Derived from rlbench/put_groceries_in_cupboard ("put the groceries in the
cupboard": named grocery among distractors, grasped and SET DOWN on a passive
open cupboard shelf — any resting pose on the shelf surface, reached by lowering
from above, is terminal). Here the cupboard's storage is a MECHANISM that must be
indexed twice, with the placement sandwiched between the two rotations:

  1. The cabinet is fully roofed and walled except for ONE front window; inside,
     a 3-bay carousel (turntable on a free damped vertical pivot) holds two decoy
     cans in two bays. Only the bay aligned with the window is accessible; the
     EMPTY bay starts rotated away (75-170 deg off the window). There is no
     passive shelf anywhere: the seed's put-it-down move terminates on the roof
     and scores nothing (smoke), and pressing the carton against the wall or a
     divider never gets it inside (smoke, force probe).
  2. The carousel is indexed from OUTSIDE via a crank knob on top of the cabinet
     (the turntable's axle passes through a roof hole and carries a crank arm):
     rotate until the empty bay faces the window.
  3. Insert the carton horizontally through the window and set it on the
     carousel floor in the empty bay.
  4. Rotate the carousel AGAIN (>= stow_min_deg away) so the loaded bay is
     behind the wall — the carton rides the turntable on friction, upright,
     and ends hidden. Success is the stowed, settled, upright carton in its bay
     with both decoy cans still seated in THEIR bays.

Execution order is forced by geometry, not by rubric timestamps: the carton can
only enter through the window into the aligned bay (walls/roof/dividers block
everything else), and it can only end up stowed by riding a rotation that
happens AFTER the insertion.

Assets are fully procedural (compound spawners; per-child density on the
turntable so the pivot inertia is real; root MassAPI on the heavy cabinet —
custom spawners apply no cfg mass schemas, so mass is authored in the funcs):
  - cabinet: heavy DYNAMIC body (a jointed body0 must not be kinematic or the
    anchor stays world-fixed after the reset teleport): round plinth (r 0.24,
    top z 0.10), a 260 deg polygonal wall band (inner face r 0.21, z 0.10-0.32)
    leaving a +/-50 deg front WINDOW, and a roof (z 0.32-0.34) with a small
    axle hole (free gap 12 mm — far smaller than the carton).
  - turntable: DYNAMIC compound on a spawn-authored free RevoluteJoint (axis Z,
    no limits, damped): floor disc r 0.19 (top z 0.127, 7 mm above the plinth),
    hub, three radial dividers every 120 deg (bay centers at turntable-local
    0/120/240 deg), axle through the roof hole, and a crank arm + vertical
    KNOB (D 24 mm, z 0.42-0.48) above the roof — the graspable/pushable index
    handle, reachable at every carousel angle.
  - red CARTON 55 x 55 x 110 mm (the grocery; 0.25 kg), starting upright on a
    small kinematic pickup stand outside the cabinet.
  - BLUE and GREEN cans (D 60 x 95 mm) stowed upright in two bays (which two
    bays, and which can where, is shuffled per episode).

Per-episode randomization (readback-verifiable): cabinet yaw +/-18 deg + xy
jitter, EMPTY bay index (3 values), initial empty-bay bearing +/-U(75, 170) deg
off the window, decoy-to-bay shuffle, pickup-stand bearing/distance + carton yaw.

Rubric (0..1; latched stage credit anchored in the demonstrated solve):
  0.15 * lifted   — carton ever raised well above its stand (latched)
  0.25 * aligned  — empty bay ever brought within align_tol of the window
                    (latched; initial bearing >= 75 deg makes this real work)
  0.30 * loaded   — carton ever upright inside the empty bay (latched, 3-step
                    persistence)
  1.0 iff success() — carton upright and settled on the carousel floor of the
                    (former) empty bay, that bay rotated >= stow_min_deg away
                    from the window (carton fully behind the wall), both decoy
                    cans still seated in their own bays, everything finite.
  Non-success capped at 0.70; null policy ~0.

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


# ----- USD authoring helpers --------------------------------------------------------------------
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             orient=None, density: float | None = None):
    """One box child: translate (+ optional orient) + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom, UsdPhysics

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    if density is not None:
        UsdPhysics.MassAPI.Apply(box.GetPrim()).CreateDensityAttr(float(density))
    return box.GetPrim()


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable,
             density: float | None = None):
    """One z-axis cylinder child."""
    from pxr import Gf, UsdGeom, UsdPhysics

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    if density is not None:
        UsdPhysics.MassAPI.Apply(cyl.GetPrim()).CreateDensityAttr(float(density))
    return cyl.GetPrim()


def _qz_t(ang_deg: float) -> tuple:
    h = math.radians(ang_deg) / 2
    return (math.cos(h), 0.0, 0.0, math.sin(h))


# ----- compound spawn funcs ---------------------------------------------------------------------
def _spawn_cabinet(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The cabinet: heavy DYNAMIC compound. Local frame: origin at the carousel
    axis on the ground; local +x = window centre direction. Round plinth, a
    260 deg polygonal wall band leaving the +/- window_half_deg front window,
    and a roof with a small axle hole."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.cabinet_mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    wood, roofc = c.cabinet_color, c.roof_color

    # plinth (the carton cannot slip under the disc: 7 mm gap, plinth backs it)
    _add_cyl(stage, f"{prim_path}/plinth", center=(0.0, 0.0, c.plinth_h / 2),
             radius=c.plinth_r, height=c.plinth_h, color=wood, collide=collide)

    # wall band: flat segments every seg_deg over the closed arc
    zw = (c.wall_z0 + c.wall_z1) / 2
    hw = c.wall_z1 - c.wall_z0
    seg = c.wall_seg_deg
    a = c.window_half_deg + seg / 2
    i = 0
    chord = 2 * c.wall_r * math.tan(math.radians(seg / 2)) + 0.006
    while a <= 360.0 - c.window_half_deg - seg / 2 + 1e-6:
        ar = math.radians(a)
        _add_box(stage, f"{prim_path}/wall_{i}",
                 center=(c.wall_r * math.cos(ar), c.wall_r * math.sin(ar), zw),
                 size=(c.wall_t, chord, hw), color=wood, collide=collide,
                 orient=_qz_t(a))
        a += seg
        i += 1

    # roof: 4 plates leaving a square axle hole (half-gap hole_half)
    zr = (c.roof_z0 + c.roof_z1) / 2
    hr = c.roof_z1 - c.roof_z0
    ext = c.roof_half
    hh = c.hole_half
    _add_box(stage, f"{prim_path}/roof_yp", center=(0.0, (hh + ext) / 2, zr),
             size=(2 * ext, ext - hh, hr), color=roofc, collide=collide)
    _add_box(stage, f"{prim_path}/roof_yn", center=(0.0, -(hh + ext) / 2, zr),
             size=(2 * ext, ext - hh, hr), color=roofc, collide=collide)
    _add_box(stage, f"{prim_path}/roof_xp", center=((hh + ext) / 2, 0.0, zr),
             size=(ext - hh, 2 * hh, hr), color=roofc, collide=collide)
    _add_box(stage, f"{prim_path}/roof_xn", center=(-(hh + ext) / 2, 0.0, zr),
             size=(ext - hh, 2 * hh, hr), color=roofc, collide=collide)
    return root


def _spawn_turntable(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The carousel: DYNAMIC compound, body origin ON the pivot axis at ground
    level. Floor disc + hub + three radial dividers (bay centres at local
    0/120/240 deg) + axle through the roof + crank arm + knob. Per-child
    DENSITY so the pivot inertia is real. Spawn-authored free RevoluteJoint
    (axis Z, no limits) to the sibling Cabinet."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.2)
    pxrb.CreateAngularDampingAttr(float(c.tt_ang_damping))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    deck, div_c, knob_c = c.deck_color, c.divider_color, c.knob_color
    rho = c.tt_density

    _add_cyl(stage, f"{prim_path}/disc",
             center=(0.0, 0.0, c.disc_top - c.disc_t / 2),
             radius=c.disc_r, height=c.disc_t, color=deck, collide=collide,
             density=rho)
    _add_cyl(stage, f"{prim_path}/hub",
             center=(0.0, 0.0, (c.disc_top + c.hub_z1) / 2),
             radius=c.hub_r, height=c.hub_z1 - c.disc_top, color=div_c,
             collide=collide, density=rho)
    rmid = (c.hub_r + c.disc_r) / 2
    for k, ang in enumerate((60.0, 180.0, 300.0)):
        ar = math.radians(ang)
        _add_box(stage, f"{prim_path}/div_{k}",
                 center=(rmid * math.cos(ar), rmid * math.sin(ar),
                         (c.disc_top + c.div_z1) / 2),
                 size=(c.disc_r - c.hub_r, c.div_t, c.div_z1 - c.disc_top),
                 color=div_c, collide=collide, orient=_qz_t(ang), density=rho)
    _add_cyl(stage, f"{prim_path}/axle",
             center=(0.0, 0.0, (c.hub_z1 + c.crank_z0) / 2),
             radius=c.axle_r, height=c.crank_z0 - c.hub_z1, color=div_c,
             collide=collide, density=rho)
    _add_box(stage, f"{prim_path}/crank",
             center=(c.crank_len / 2 - 0.015, 0.0, c.crank_z0 + 0.01),
             size=(c.crank_len, 0.03, 0.02), color=knob_c, collide=collide,
             density=rho)
    _add_cyl(stage, f"{prim_path}/knob",
             center=(c.knob_r_pos, 0.0, c.crank_z0 + 0.02 + c.knob_h / 2),
             radius=c.knob_r, height=c.knob_h, color=knob_c, collide=collide,
             density=rho)

    # free revolute pivot to the sibling cabinet (vertical axis; no limits)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/pivot")
    j.CreateBody0Rel().SetTargets([f"{base}/Cabinet"])
    j.CreateBody1Rel().SetTargets([prim_path])
    # turntable<->cabinet contact stays ON (USD default for a joint pair is
    # filtered); the authored clearances are >= 7 mm everywhere at every angle
    j.CreateCollisionEnabledAttr(True)
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.20))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.20))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    return root


def _spawn_carton(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The red grocery carton: one box, body origin at its centre, +z up."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.1)
    pxrb.CreateAngularDampingAttr(0.3)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/body", center=(0.0, 0.0, 0.0),
             size=(c.carton_w, c.carton_w, c.carton_h), color=c.carton_color,
             collide=collide, density=c.carton_density)
    return root


def _spawn_can(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A decoy can: one cylinder, body origin at its centre, +z up."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.1)
    pxrb.CreateAngularDampingAttr(0.3)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    _add_cyl(stage, f"{prim_path}/body", center=(0.0, 0.0, 0.0),
             radius=c.can_r, height=c.can_h, color=c.can_color,
             collide=collide, density=c.can_density)
    return root


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The pickup stand: KINEMATIC block (no joints attach to it — safe)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    collide = _make_collide(cfg.contact_offset)
    _add_box(stage, f"{prim_path}/block",
             center=(0.0, 0.0, cfg.stand_h / 2),
             size=(cfg.stand_w, cfg.stand_w, cfg.stand_h),
             color=cfg.stand_color, collide=collide)
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
            cabinet_mass: float = 50.0
            plinth_r: float = 0.24
            plinth_h: float = 0.10
            wall_r: float = 0.22
            wall_t: float = 0.02
            wall_z0: float = 0.10
            wall_z1: float = 0.32
            wall_seg_deg: float = 20.0
            window_half_deg: float = 50.0
            roof_z0: float = 0.32
            roof_z1: float = 0.34
            roof_half: float = 0.245
            hole_half: float = 0.030
            cabinet_color: tuple = (0.55, 0.40, 0.24)
            roof_color: tuple = (0.46, 0.33, 0.20)
            contact_offset: float = 0.002

        @configclass
        class TurntableSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_turntable)
            disc_r: float = 0.19
            disc_t: float = 0.02
            disc_top: float = 0.127
            hub_r: float = 0.03
            hub_z1: float = 0.287
            div_t: float = 0.024
            div_z1: float = 0.302
            axle_r: float = 0.018
            crank_z0: float = 0.40
            crank_len: float = 0.19
            knob_r_pos: float = 0.155
            knob_r: float = 0.012
            knob_h: float = 0.06
            tt_ang_damping: float = 0.8
            tt_density: float = 500.0
            deck_color: tuple = (0.78, 0.66, 0.42)
            divider_color: tuple = (0.68, 0.55, 0.32)
            knob_color: tuple = (0.20, 0.20, 0.22)
            contact_offset: float = 0.002

        @configclass
        class CartonSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_carton)
            carton_w: float = 0.055
            carton_h: float = 0.110
            carton_density: float = 750.0
            carton_color: tuple = (0.85, 0.12, 0.12)
            contact_offset: float = 0.002

        @configclass
        class CanSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_can)
            can_r: float = 0.030
            can_h: float = 0.095
            can_density: float = 600.0
            can_color: tuple = (0.15, 0.30, 0.85)
            contact_offset: float = 0.002

        @configclass
        class StandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            stand_w: float = 0.14
            stand_h: float = 0.12
            stand_color: tuple = (0.50, 0.50, 0.52)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["cabinet"] = CabinetSpawnerCfg
        _SPAWNER_CACHE["turntable"] = TurntableSpawnerCfg
        _SPAWNER_CACHE["carton"] = CartonSpawnerCfg
        _SPAWNER_CACHE["can"] = CanSpawnerCfg
        _SPAWNER_CACHE["stand"] = StandSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class CarouselCupboardSceneCfg(BaseCfg):
    """Config for `CarouselCupboardScene`. The access geometry is honest by
    construction — every claim is asserted numerically in __post_init__: the
    carton fits through the window with margin but cannot pass the roof hole or
    the divider-to-wall gap; a stowed bay hides the carton fully behind the
    wall; the initial empty-bay bearing can never hand out alignment credit."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    align_tol_deg: float = tunable(30.0)   # empty-bay |bearing| below this latches `aligned`
    stow_min_deg: float = tunable(100.0)   # |bearing| at/above this counts as stowed
    bay_r_min: float = tunable(0.045)      # in-bay radial band (turntable-local, m)
    bay_r_max: float = tunable(0.185)
    bay_ang_tol_deg: float = tunable(48.0)  # in-bay angular half-width about the bay centre
    bay_z_min: float = tunable(0.162)      # in-bay body-centre height band (on the disc)
    bay_z_max: float = tunable(0.215)
    carton_upright_deg: float = tunable(20.0)  # carton +z within this of world-up
    can_upright_deg: float = tunable(30.0)     # can axis within this of world-up
    settle_speed: float = tunable(0.05)    # max carton/cabinet |lin vel| when judging (m/s)
    tt_settle_avel: float = tunable(0.30)  # max turntable |ang vel| when judging (rad/s)
    lift_z: float = tunable(0.27)          # carton centre above this latches `lifted` (m)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    cab_yaw_deg: float = tunable(18.0)     # cabinet yaw jitter about nominal (+/- deg)
    cab_jitter: float = tunable(0.04)      # cabinet xy jitter (+/- m)
    bearing0_range: tuple = tunable((75.0, 170.0))  # initial empty-bay |bearing| U(range), deg
    stand_bear_deg: float = tunable(35.0)  # stand bearing about the window direction (+/- deg)
    stand_dist_range: tuple = tunable((0.42, 0.52))  # stand distance from the axis (m)

    # --- info: layout ---------------------------------------------------------------------------
    cab_pos: tuple = info((0.50, 0.0))     # carousel axis on the ground (nominal)
    cab_yaw_nom_deg: float = info(180.0)   # nominal heading: window (+x local) faces the robot
    cabinet_mass: float = info(50.0)
    plinth_r: float = info(0.24)
    plinth_h: float = info(0.10)
    wall_r: float = info(0.22)             # wall band centreline radius (inner face 0.21)
    wall_t: float = info(0.02)
    wall_z0: float = info(0.10)
    wall_z1: float = info(0.32)
    window_half_deg: float = info(50.0)    # window half arc about cabinet-local +x
    roof_z0: float = info(0.32)
    roof_z1: float = info(0.34)
    roof_half: float = info(0.245)
    hole_half: float = info(0.030)         # roof axle-hole half gap
    # turntable
    disc_r: float = info(0.19)
    disc_top: float = info(0.127)          # carousel floor height (the bay shelf)
    hub_r: float = info(0.03)
    div_z1: float = info(0.302)            # divider top (18 mm under the roof)
    axle_r: float = info(0.018)
    knob_r_pos: float = info(0.155)        # knob radius on the crank (m)
    knob_r: float = info(0.012)
    knob_top: float = info(0.48)
    tt_ang_damping: float = info(0.8)
    tt_density: float = info(500.0)
    slot_r: float = info(0.115)            # nominal item radius inside a bay (m)
    # groceries
    carton_w: float = info(0.055)
    carton_h: float = info(0.110)
    carton_density: float = info(750.0)
    can_r: float = info(0.030)
    can_h: float = info(0.095)
    can_density: float = info(600.0)
    # pickup stand
    stand_w: float = info(0.14)
    stand_h: float = info(0.12)
    # rubric weights (0.15 + 0.25 + 0.30 = 0.70 = the non-success cap)
    w_lift: float = info(0.15)
    w_align: float = info(0.25)
    w_load: float = info(0.30)
    # colors
    cabinet_color: tuple = info((0.55, 0.40, 0.24))
    roof_color: tuple = info((0.46, 0.33, 0.20))
    deck_color: tuple = info((0.78, 0.66, 0.42))
    divider_color: tuple = info((0.68, 0.55, 0.32))
    knob_color: tuple = info((0.20, 0.20, 0.22))
    carton_color: tuple = info((0.85, 0.12, 0.12))
    can_blue_color: tuple = info((0.15, 0.30, 0.85))
    can_green_color: tuple = info((0.10, 0.62, 0.20))
    stand_color: tuple = info((0.50, 0.50, 0.52))
    contact_offset: float = info(0.002)

    def __post_init__(self) -> None:
        """Audit the access geometry (all lengths in metres, angles in deg)."""
        wall_in = self.wall_r - self.wall_t / 2
        diag = self.carton_w * math.sqrt(2)
        # the window admits the carton with generous margin (chord at the inner face)
        window_chord = 2 * wall_in * math.sin(math.radians(self.window_half_deg))
        assert window_chord > diag + 0.15, (window_chord, diag)
        # the window is tall enough for the upright carton held above the disc
        assert self.wall_z1 - self.disc_top > self.carton_h + 0.06
        # the roof hole cannot pass the carton: widest free span beside the axle
        # (hole half-diagonal minus axle radius) is far under the carton width
        assert self.hole_half * math.sqrt(2) - self.axle_r < 0.5 * self.carton_w
        # the divider-to-wall and disc-to-wall gaps cannot pass the carton or a can
        assert wall_in - self.disc_r < min(self.carton_w, 2 * self.can_r) - 0.01
        # a stowed carton is FULLY behind the wall: bay bearing at stow_min puts
        # the carton's whole angular extent outside the window arc
        half_ext = math.degrees(math.atan2(diag / 2, self.slot_r))
        assert self.stow_min_deg - half_ext > self.window_half_deg + 5.0
        # only the aligned bay is insertable: a neighbour bay's nearest in-bay
        # position stays outside the window when this bay is centred
        assert 120.0 - self.bay_ang_tol_deg > self.window_half_deg + 15.0
        # the initial empty-bay bearing can never hand out alignment credit
        assert self.bearing0_range[0] > self.align_tol_deg + 10.0
        # stow threshold is well beyond alignment (the two latches are distinct)
        assert self.stow_min_deg > self.align_tol_deg + 40.0
        # bay z band sits on the disc: an upright carton centre is inside it, a
        # toppled carton centre is below it
        assert self.bay_z_min < self.disc_top + self.carton_h / 2 < self.bay_z_max
        assert self.bay_z_min < self.disc_top + self.can_h / 2 < self.bay_z_max
        assert self.disc_top + self.carton_w / 2 < self.bay_z_min
        assert self.disc_top + self.can_r < self.bay_z_min
        # jaw feasibility: knob and carton fit a Franka parallel jaw (< 80 mm)
        assert 2 * self.knob_r < 0.06 and self.carton_w < 0.07
        # the pickup stand stays clear of the cabinet at every sampled distance
        assert self.stand_dist_range[0] > self.plinth_r + self.stand_w / 2 * 1.5 + 0.05
        # lift latch is above both the stand-top and the on-disc carton pose
        assert self.lift_z > self.stand_h + self.carton_h / 2 + 0.06
        assert self.lift_z > self.disc_top + self.carton_h / 2 + 0.06


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


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _wrap_deg(a: torch.Tensor) -> torch.Tensor:
    return (a + 180.0) % 360.0 - 180.0


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("carousel_cupboard")
class CarouselCupboardScene(BaseScene):
    cfg: CarouselCupboardSceneCfg

    def __init__(self, cfg: CarouselCupboardSceneCfg | None = None) -> None:
        super().__init__(cfg or CarouselCupboardSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        cab_spawn = cls["cabinet"](
            cabinet_mass=c.cabinet_mass, plinth_r=c.plinth_r, plinth_h=c.plinth_h,
            wall_r=c.wall_r, wall_t=c.wall_t, wall_z0=c.wall_z0, wall_z1=c.wall_z1,
            window_half_deg=c.window_half_deg, roof_z0=c.roof_z0, roof_z1=c.roof_z1,
            roof_half=c.roof_half, hole_half=c.hole_half,
            cabinet_color=c.cabinet_color, roof_color=c.roof_color,
            contact_offset=c.contact_offset)
        tt_spawn = cls["turntable"](
            disc_r=c.disc_r, disc_top=c.disc_top, hub_r=c.hub_r,
            div_z1=c.div_z1, axle_r=c.axle_r, knob_r_pos=c.knob_r_pos,
            knob_r=c.knob_r, tt_ang_damping=c.tt_ang_damping,
            tt_density=c.tt_density, deck_color=c.deck_color,
            divider_color=c.divider_color, knob_color=c.knob_color,
            contact_offset=c.contact_offset)
        carton_spawn = cls["carton"](
            carton_w=c.carton_w, carton_h=c.carton_h,
            carton_density=c.carton_density, carton_color=c.carton_color,
            contact_offset=c.contact_offset)
        can_b_spawn = cls["can"](can_r=c.can_r, can_h=c.can_h,
                                 can_density=c.can_density,
                                 can_color=c.can_blue_color,
                                 contact_offset=c.contact_offset)
        can_g_spawn = cls["can"](can_r=c.can_r, can_h=c.can_h,
                                 can_density=c.can_density,
                                 can_color=c.can_green_color,
                                 contact_offset=c.contact_offset)
        stand_spawn = cls["stand"](stand_w=c.stand_w, stand_h=c.stand_h,
                                   stand_color=c.stand_color,
                                   contact_offset=c.contact_offset)

        # template poses: turntable spawns joint-consistent with the cabinet
        px, py = c.cab_pos
        q0 = _qz_t(c.cab_yaw_nom_deg)
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
            "cabinet": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet",
                spawn=cab_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0), rot=q0),
            ),
            "turntable": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Turntable",
                spawn=tt_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0), rot=q0),
            ),
            "carton": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Carton",
                spawn=carton_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, 1.0, 0.06)),
            ),
            "can_blue": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/CanBlue",
                spawn=can_b_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.4, 1.0, 0.051)),
            ),
            "can_green": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/CanGreen",
                spawn=can_g_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.6, 1.0, 0.051)),
            ),
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=stand_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, -1.0, 0.0)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # solve/smoke drive bodies via set_external_force_and_torque;
                # without this flag wrenches are under-applied across TGS iterations
                "enable_external_forces_every_iteration": True,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.cabinet: RigidObject = env.iscene["cabinet"]
        self.turntable: RigidObject = env.iscene["turntable"]
        self.carton: RigidObject = env.iscene["carton"]
        self.can_blue: RigidObject = env.iscene["can_blue"]
        self.can_green: RigidObject = env.iscene["can_green"]
        self.stand: RigidObject = env.iscene["stand"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.empty_bay = torch.zeros(n, dtype=torch.long, device=dev)
        self.blue_bay = torch.ones(n, dtype=torch.long, device=dev)
        self.green_bay = torch.full((n,), 2, dtype=torch.long, device=dev)
        self.bearing0 = torch.zeros(n, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._aligned = torch.zeros(n, dtype=torch.bool, device=dev)
        self._loaded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._load_cnt = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the cabinet (yaw + xy jitter), rotate the
        carousel so the sampled EMPTY bay sits 75-170 deg off the window, seat
        the shuffled decoy cans in the other two bays (poses written in the
        turntable frame), stand the carton on the pickup stand at a random
        bearing in front, clear the latches. The whole linkage is written
        together (teleporting one body of a jointed pair gets depenetrated
        back by the other)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        yaw = math.radians(c.cab_yaw_nom_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.cab_yaw_deg)
        q_cab = _qz(yaw)
        cp = torch.zeros(m, 3, device=dev)
        cp[:, 0] = c.cab_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.cab_jitter
        cp[:, 1] = c.cab_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.cab_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = cp + origin
        st[:, 3:7] = q_cab
        self.cabinet.write_root_state_to_sim(st, env_ids)

        # empty bay + decoy shuffle (torch.rand comparisons — the first randint
        # after manual_seed is near-degenerate)
        e = (torch.rand(m, device=dev) * 3.0).clamp(max=2.999).long()
        self.empty_bay[env_ids] = e
        others = torch.stack([(e + 1) % 3, (e + 2) % 3], dim=1)  # the occupied bays
        swap = torch.rand(m, device=dev) < 0.5
        self.blue_bay[env_ids] = torch.where(swap, others[:, 0], others[:, 1])
        self.green_bay[env_ids] = torch.where(swap, others[:, 1], others[:, 0])

        # carousel angle: empty-bay bearing = sign * U(range) off the window
        b0, b1 = c.bearing0_range
        mag = b0 + torch.rand(m, device=dev) * (b1 - b0)
        sgn = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        bearing = sgn * mag
        self.bearing0[env_ids] = bearing
        phi = torch.deg2rad(bearing - 120.0 * e.float())
        q_tt = _qmul(q_cab, _qz(phi))
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = cp + origin
        st[:, 3:7] = q_tt
        self.turntable.write_root_state_to_sim(st, env_ids)

        # decoy cans: seated at the slot radius of their bays (turntable frame)
        for body, bay in ((self.can_blue, self.blue_bay[env_ids]),
                          (self.can_green, self.green_bay[env_ids])):
            ang = torch.deg2rad(120.0 * bay.float())
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = c.slot_r * torch.cos(ang)
            loc[:, 1] = c.slot_r * torch.sin(ang)
            loc[:, 2] = c.disc_top + c.can_h / 2 + 0.003
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = cp + quat_apply(q_tt, loc) + origin
            st[:, 3:7] = q_tt
            body.write_root_state_to_sim(st, env_ids)

        # pickup stand + carton: random bearing about the window direction
        d0, d1 = c.stand_dist_range
        dist = d0 + torch.rand(m, device=dev) * (d1 - d0)
        sb = yaw + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.stand_bear_deg)
        sp = torch.zeros(m, 3, device=dev)
        sp[:, 0] = cp[:, 0] + dist * torch.cos(sb)
        sp[:, 1] = cp[:, 1] + dist * torch.sin(sb)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = sp + origin
        st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
        self.stand.write_root_state_to_sim(st, env_ids)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = sp[:, 0:2] + origin[:, 0:2]
        st[:, 2] = origin[:, 2] + c.stand_h + c.carton_h / 2 + 0.003
        st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
        self.carton.write_root_state_to_sim(st, env_ids)

        self._lifted[env_ids] = False
        self._aligned[env_ids] = False
        self._loaded[env_ids] = False
        self._load_cnt[env_ids] = 0

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cabinet": self.cabinet.data.root_state_w[env_ids].clone(),
            "turntable": self.turntable.data.root_state_w[env_ids].clone(),
            "carton": self.carton.data.root_state_w[env_ids].clone(),
            "can_blue": self.can_blue.data.root_state_w[env_ids].clone(),
            "can_green": self.can_green.data.root_state_w[env_ids].clone(),
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "empty_bay": self.empty_bay[env_ids].clone(),
            "blue_bay": self.blue_bay[env_ids].clone(),
            "green_bay": self.green_bay[env_ids].clone(),
            "bearing0": self.bearing0[env_ids].clone(),
            "lifted": self._lifted[env_ids].clone(),
            "aligned": self._aligned[env_ids].clone(),
            "loaded": self._loaded[env_ids].clone(),
            "load_cnt": self._load_cnt[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cabinet.write_root_state_to_sim(state["cabinet"], env_ids)
        self.turntable.write_root_state_to_sim(state["turntable"], env_ids)
        self.carton.write_root_state_to_sim(state["carton"], env_ids)
        self.can_blue.write_root_state_to_sim(state["can_blue"], env_ids)
        self.can_green.write_root_state_to_sim(state["can_green"], env_ids)
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.empty_bay[env_ids] = state["empty_bay"]
        self.blue_bay[env_ids] = state["blue_bay"]
        self.green_bay[env_ids] = state["green_bay"]
        self.bearing0[env_ids] = state["bearing0"]
        self._lifted[env_ids] = state["lifted"]
        self._aligned[env_ids] = state["aligned"]
        self._loaded[env_ids] = state["loaded"]
        self._load_cnt[env_ids] = state["load_cnt"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A round wooden CUPBOARD stands on the ground: a low plinth, a "
            "cylindrical wall (inner diameter ~420 mm, from 100 to 320 mm "
            "height) and a full roof. The wall has ONE opening — a front WINDOW "
            f"spanning {2 * c.window_half_deg:.0f} degrees of arc — and the "
            "roof is closed except for a small axle hole; nothing fits through "
            "the roof, and there is no open shelf anywhere on the outside. "
            "Inside, a CAROUSEL turntable (floor disc at 127 mm height, a hub, "
            "and three radial dividers) splits the interior into three equal "
            "BAYS; only the bay currently facing the window can be reached. "
            "The turntable's axle sticks up through the roof and carries a "
            "horizontal CRANK ARM with a vertical dark KNOB (24 mm across, at "
            "155 mm radius, 420-480 mm height): pushing or dragging the knob "
            "in a circle rotates the carousel — it turns freely and stays "
            "where it is left. Two bays hold groceries already stowed: a BLUE "
            "can and a GREEN can, one each. One bay is EMPTY. The carousel "
            "starts rotated so the empty bay does NOT face the window "
            f"({c.bearing0_range[0]:.0f}-{c.bearing0_range[1]:.0f} degrees "
            "off, either direction). In front of the cupboard a small grey "
            "STAND holds the grocery to put away: an upright RED CARTON "
            "(55 x 55 x 110 mm).\n"
            "Goal: stock the red carton into the EMPTY bay and stow it. "
            "Rotate the carousel by its top knob until the empty bay faces "
            "the window (look through the window: the accessible bay must "
            "contain nothing), carry the carton in horizontally through the "
            "window and set it UPRIGHT on the carousel floor inside the empty "
            "bay, then rotate the carousel again — at least "
            f"{c.stow_min_deg:.0f} degrees, either direction — so the "
            "carton's bay is behind the wall and the carton is no longer "
            "visible through the window. The carton rides the turntable; "
            "rotate gently so it stays upright and in its bay. The blue and "
            "green cans must remain seated in their own bays throughout. "
            "Setting the carton on the roof, on the ground, at the window "
            "mouth, into an occupied bay, or leaving its bay still facing "
            "the window does not count. Success: carton upright and settled "
            "on the carousel floor in the formerly-empty bay, that bay "
            f"rotated at least {c.stow_min_deg:.0f} degrees away from the "
            "window centre, both cans still in their bays, everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Rotate the cupboard's carousel by the knob on its roof until the "
            "empty bay faces the front window, put the red carton upright onto "
            "the carousel floor in that empty bay through the window, then "
            "rotate the carousel at least 100 degrees further so the carton is "
            "hidden behind the wall. Keep the blue and green cans seated in "
            "their own bays and leave the carton upright and inside its bay."
        )

    # ----- frames / live predicates -------------------------------------------------------------
    def _tt_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.turntable.data.root_quat_w,
                                  pos_w - self.turntable.data.root_pos_w)

    def rel_yaw_deg(self) -> torch.Tensor:
        """(N,) float: turntable yaw relative to the cabinet (deg, wrapped)."""
        qc = self.cabinet.data.root_quat_w
        qt = self.turntable.data.root_quat_w
        qc_inv = qc * torch.tensor([1.0, -1.0, -1.0, -1.0], device=qc.device)
        rel = _qmul(qc_inv, qt)
        return _wrap_deg(torch.rad2deg(2.0 * torch.atan2(rel[:, 3], rel[:, 0])))

    def bay_bearing_deg(self, bay: torch.Tensor) -> torch.Tensor:
        """(N,) float: bay-centre bearing off the window centre (deg, wrapped)."""
        return _wrap_deg(self.rel_yaw_deg() + 120.0 * bay.float())

    def target_bearing_deg(self) -> torch.Tensor:
        return self.bay_bearing_deg(self.empty_bay)

    def _upright(self, body, max_deg: float) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device) \
            .expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)[:, 2] \
            >= math.cos(math.radians(max_deg))

    def in_bay(self, body, bay: torch.Tensor) -> torch.Tensor:
        """(N,) bool, geometric: body centre inside bay's sector on the disc
        (turntable frame): radial band, angular band about the bay centre, and
        the on-the-disc height band."""
        c = self.cfg
        loc = self._tt_local(body.data.root_pos_w)
        r = loc[:, :2].norm(dim=-1)
        ang = torch.rad2deg(torch.atan2(loc[:, 1], loc[:, 0]))
        dang = _wrap_deg(ang - 120.0 * bay.float()).abs()
        zok = (loc[:, 2] > c.bay_z_min) & (loc[:, 2] < c.bay_z_max)
        return (r > c.bay_r_min) & (r < c.bay_r_max) \
            & (dang < c.bay_ang_tol_deg) & zok

    def carton_in_target(self) -> torch.Tensor:
        return self.in_bay(self.carton, self.empty_bay) \
            & self._upright(self.carton, self.cfg.carton_upright_deg)

    def decoys_home(self) -> torch.Tensor:
        """(N,) bool: both cans seated upright in their own bays."""
        c = self.cfg
        return self.in_bay(self.can_blue, self.blue_bay) \
            & self._upright(self.can_blue, c.can_upright_deg) \
            & self.in_bay(self.can_green, self.green_bay) \
            & self._upright(self.can_green, c.can_upright_deg)

    def stowed(self) -> torch.Tensor:
        return self.target_bearing_deg().abs() >= self.cfg.stow_min_deg

    def settled(self) -> torch.Tensor:
        c = self.cfg
        return (self.carton.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.turntable.data.root_ang_vel_w[:, 2].abs() < c.tt_settle_avel) \
            & (self.cabinet.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in
                         (self.cabinet, self.turntable, self.carton,
                          self.can_blue, self.can_green)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Advance the latches ONCE per physics step."""
        fin = self._finite()
        z = self.carton.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        self._lifted |= (z > self.cfg.lift_z) & fin
        self._aligned |= (self.target_bearing_deg().abs()
                          < self.cfg.align_tol_deg) & fin
        slow = self.carton.data.root_lin_vel_w.norm(dim=-1) < 0.5
        in_t = self.carton_in_target() & slow & fin
        self._load_cnt = torch.where(in_t, self._load_cnt + 1,
                                     torch.zeros_like(self._load_cnt))
        self._loaded |= self._load_cnt >= 3

    # ----- rubric -------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the carton is STOCKED AND STOWED — upright and settled on
        the carousel floor of the formerly-empty bay, that bay rotated at least
        stow_min_deg away from the window (the carton is fully behind the
        wall), both decoy cans still seated in their own bays, all finite. All
        clauses are live physical outcomes."""
        return self.carton_in_target() & self.stowed() & self.decoys_home() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*lifted + 0.25*aligned + 0.30*loaded (all
        latched; ~0 for the null policy — the carousel holds its angle on the
        damped pivot and the carton starts on the stand), capped at 0.70 — and
        exactly 1.0 iff success() holds live."""
        c = self.cfg
        base = (c.w_lift * self._lifted.float()
                + c.w_align * self._aligned.float()
                + c.w_load * self._loaded.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="carousel_cupboard", robot="null"))
