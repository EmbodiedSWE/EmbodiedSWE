"""TiltLabyrinthScene — route the SEALED red checker into the green-marked corner
pocket by TILTING the whole case; the piece itself can never be touched.

Derived from rlbench/setup_checkers ("place the checkers on the board in the
starting arrangement": 24 free checkers lying exposed on a table are picked up one
by one and laid flat onto marked board squares — repetitive, unordered, direct
pick-and-place onto an open horizontal surface). Here the SINGLE checker is sealed
inside a closed labyrinth case and the manipuland is the ENVIRONMENT, not the
piece:

  - the case is a shallow tray mounted on a two-axis GIMBAL (outer frame on an
    X-axis hinge to the stand, tray on a Y-axis hinge to the frame, both spring-
    centered to level and limited to +/-12 deg). Pressing down on the tray's rim
    tilts the play field; releasing it lets the return springs re-level it;
  - the checker (a red disc) lies UNDER a barred grille roof: the 12 mm slots make
    it fully visible but physically non-extractable and non-graspable (smoke
    force-proves the seal). Gravity, commanded through tilt, is the only way to
    move it;
  - the interior is a maze: the start chamber funnels (two 45 deg guide walls)
    into a central GAP in a full-height baffle; behind the baffle, the far wall
    strip leads to TWO sunken corner pockets (7 mm deep). A green BEACON POST on
    the stand marks the target corner; the mirrored pocket is a decoy, and both
    pockets retain the disc against full tilt — a wrong-pocket drop is permanent;
  - goal: disc settled in the BEACON-side pocket with the case back level and at
    rest.

A solver therefore needs a different plan and different code than the seed's:
no grasp, no per-piece placement loop — instead perception of the beacon side,
a two-leg TILT ROUTE (funnel+gap leg in +x, then a signed +/-y leg along the far
wall), and a controlled release. The seed's whole skill (lay a checker flat on a
marked square) is inexpressible here; its nearest analog — a checker laid on TOP
of the case over the right pocket — is smoke-rejected.

Assets are fully procedural (compound spawners; memory: custom spawners apply no
cfg schemas, so mass/material/iters are authored in the funcs):
  - stand: heavy DYNAMIC pedestal (base slab, centre column, two bearing towers)
    — dynamic, not kinematic, because a joint anchored to a kinematic body stays
    world-fixed when the linkage is teleported at reset;
  - frame: dynamic gimbal ring, spawn-authored RevoluteJoint (axis X) to the
    stand, +/-12 deg limits, spring-return drive to 0;
  - tray: dynamic case compound (floor with two pocket holes, sunken pocket
    floors, perimeter rim walls, funnel guides, baffle, grille rails), spawn-
    authored RevoluteJoint (axis Y) to the frame, same limits/drive;
  - disc: the red checker, 48 mm diameter x 12 mm, 30 g, slick material;
  - beacon: kinematic green post re-posed each reset to the target corner's side.

Per-episode randomization (readback-verifiable): stand yaw +/-20 deg + xy jitter
+/-5 cm, disc start cell in the start chamber, and the target-pocket side (beacon
side), sampled via torch.rand comparison (first-randint-after-seed is degenerate).

Rubric (0..1; latched stage credit anchored in the demonstrated solve):
  0.25 * gap_latch    — disc crossed the baffle into the far chamber, INSIDE the
                        play volume (z-banded: a disc on the grille roof earns
                        nothing), 3-step persistence;
  0.35 * pocket_latch — disc inside the TARGET pocket volume (any case attitude),
                        3-step persistence;
  1.0 iff success()   — disc settled in the target pocket AND the case level
                        (within 3 deg) and everything at rest and finite.
  Non-success capped at 0.60; null policy ~0 (springs hold the case level and
  the disc parked in the start chamber).

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


def _add_box(stage, path: str, *, center, size, color, contact_offset,
             material=None, orient=None, density: float | None = None):
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
    _collide(box.GetPrim(), contact_offset, material)
    if density is not None:
        UsdPhysics.MassAPI.Apply(box.GetPrim()).CreateDensityAttr(float(density))
    return box.GetPrim()


def _rb(root, *, kinematic: bool = False, mass: float | None = None,
        lin_damp: float = 0.05, ang_damp: float = 0.05, iters: bool = True):
    """RigidBodyAPI + PhysX body armor on a compound root."""
    from pxr import PhysxSchema, UsdPhysics

    api = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        api.CreateKinematicEnabledAttr(True)
    if mass is not None:
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    if iters:
        px.CreateSolverPositionIterationCountAttr(16)
        px.CreateSolverVelocityIterationCountAttr(4)  # TGS vel iters max 4
    return px


def _hinge(stage, path: str, body0: str, body1: str, *, axis: str,
           local_pos0, local_pos1, limit_deg: float, k_drive: float, d_drive: float):
    """Spawn-authored RevoluteJoint with symmetric limits and a spring-return
    angular drive to 0. The joint pair stays collision-FILTERED (USD default) —
    gimbal clearances are visual; the limits are the mechanical stops."""
    from pxr import Gf, UsdPhysics

    j = UsdPhysics.RevoluteJoint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([body0])
    j.CreateBody1Rel().SetTargets([body1])
    j.CreateAxisAttr(axis)
    j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in local_pos0]))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(*[float(v) for v in local_pos1]))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(limit_deg))
    j.CreateUpperLimitAttr(float(limit_deg))
    drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    drv.CreateTypeAttr("force")
    drv.CreateTargetPositionAttr(0.0)
    drv.CreateStiffnessAttr(float(k_drive))
    drv.CreateDampingAttr(float(d_drive))
    return j


# ----- compound spawn funcs ---------------------------------------------------------------------
def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The stand: heavy DYNAMIC pedestal — base slab, centre column (top LOW so
    the double-tilted tray corner clears it), two bearing towers at x = +/-tower_x
    reaching the pivot height. Origin at the footprint centre on the ground."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rb(root, mass=c.stand_mass, lin_damp=2.0, ang_damp=2.0)
    co = c.contact_offset
    col = c.stand_color
    _add_box(stage, f"{prim_path}/base", center=(0.0, 0.0, 0.02),
             size=(c.base_x, c.base_y, 0.04), color=col, contact_offset=co)
    _add_box(stage, f"{prim_path}/column", center=(0.0, 0.0, (0.04 + c.column_top) / 2),
             size=(0.08, 0.08, c.column_top - 0.04), color=col, contact_offset=co)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/tower_{'p' if sgn > 0 else 'n'}",
                 center=(sgn * c.tower_x, 0.0, (0.04 + c.pivot_z + 0.02) / 2),
                 size=(0.05, 0.05, c.pivot_z + 0.02 - 0.04), color=col, contact_offset=co)
    return root


def _spawn_frame(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The gimbal frame: dynamic rectangular ring around the tray, origin at the
    PIVOT. Carries the spawn-authored X-axis hinge to the sibling Stand."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rb(root, ang_damp=0.05)
    co = c.contact_offset
    col = c.frame_color
    s = c.frame_sec  # 0.03 square section
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/rail_{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * c.frame_ry, 0.0),
                 size=(2 * c.frame_rx + s, s, s), color=col, contact_offset=co,
                 density=c.frame_density)
        _add_box(stage, f"{prim_path}/bar_{'p' if sgn > 0 else 'n'}",
                 center=(sgn * c.frame_rx, 0.0, 0.0),
                 size=(s, 2 * c.frame_ry - s, s), color=col, contact_offset=co,
                 density=c.frame_density)
    base = prim_path.rsplit("/", 1)[0]
    _hinge(stage, f"{prim_path}/roll_hinge", f"{base}/Stand", prim_path,
           axis="X", local_pos0=(0.0, 0.0, c.pivot_z), local_pos1=(0.0, 0.0, 0.0),
           limit_deg=c.tilt_limit_deg, k_drive=c.drive_k, d_drive=c.drive_d)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The sealed labyrinth case: floor (with two pocket holes), sunken pocket
    floors, perimeter rim walls, two 45-deg funnel guide walls, the gapped
    baffle, and the grille rails. Origin at the PIVOT (local z=0). Carries the
    spawn-authored Y-axis hinge to the sibling Frame. Slick material on every
    interior collider; per-child density so the pivot inertia is real."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rb(root, ang_damp=0.05)
    co = c.contact_offset
    mat = _friction_material(stage, f"{prim_path}/slickMat", c.mu_s, c.mu_d)
    dn = c.tray_density
    ix, iy, wt = c.ix, c.iy, c.wall_t
    f0, f1 = c.floor_z0, c.floor_z1              # floor slab bottom/top
    p0, p1 = c.pocket_z0, c.pocket_z1            # pocket floor bottom/top
    wtop, itop = c.rim_top, c.inner_top          # rim wall top / interior wall top
    g0, g1 = c.grille_z0, c.grille_z1

    def box(name, center, size, color, orient=None):
        _add_box(stage, f"{prim_path}/{name}", center=center, size=size, color=color,
                 contact_offset=co, material=mat, orient=orient, density=dn)

    dark, light = c.tray_color, c.tray_color2
    # --- floor: everything except the two pocket holes (x [pk_x0, ix], |y| [pk_y0, iy])
    zf = (f0 + f1) / 2
    tf = f1 - f0
    box("floor_main", ((-ix + c.pk_x0) / 2, 0.0, zf), (c.pk_x0 + ix, 2 * iy, tf), dark)
    box("floor_strip", ((c.pk_x0 + ix) / 2, 0.0, zf), (ix - c.pk_x0, 2 * c.pk_y0, tf), dark)
    # --- sunken pocket floors (both corners identical — identity comes from the beacon)
    zp = (p0 + p1) / 2
    for sgn in (1.0, -1.0):
        box(f"pocket_{'p' if sgn > 0 else 'n'}",
            ((c.pk_x0 + ix) / 2, sgn * (c.pk_y0 + iy) / 2, zp),
            (ix - c.pk_x0, iy - c.pk_y0, p1 - p0), light)
    # --- perimeter rim walls (floor bottom up to the press rim)
    zw = (f0 + wtop) / 2
    hw = wtop - f0
    box("rim_xp", (ix + wt / 2, 0.0, zw), (wt, 2 * iy + 2 * wt, hw), dark)
    box("rim_xn", (-ix - wt / 2, 0.0, zw), (wt, 2 * iy + 2 * wt, hw), dark)
    box("rim_yp", (0.0, iy + wt / 2, zw), (2 * ix, wt, hw), dark)
    box("rim_yn", (0.0, -iy - wt / 2, zw), (2 * ix, wt, hw), dark)
    # --- interior structures: funnel guides + gapped baffle (floor top .. inner_top)
    zi = (f1 + itop) / 2
    hi = itop - f1
    fx0, fx1 = c.fun_x0, c.fun_x1
    fy0, fy1 = iy, c.gap_half - 0.003            # wall line runs to just past the gap edge
    ln = math.hypot(fx1 - fx0, fy1 - fy0) + 0.008
    ang = math.atan2(fy1 - fy0, fx1 - fx0)       # negative for the +y wall
    for sgn in (1.0, -1.0):
        q = (math.cos(sgn * ang / 2), 0.0, 0.0, math.sin(sgn * ang / 2))
        box(f"funnel_{'p' if sgn > 0 else 'n'}",
            ((fx0 + fx1) / 2, sgn * (fy0 + fy1) / 2, zi), (ln, 0.010, hi), light, orient=q)
        box(f"baffle_{'p' if sgn > 0 else 'n'}",
            (0.0, sgn * (c.gap_half + iy) / 2, zi), (2 * c.baffle_half, iy - c.gap_half, hi),
            light)
    # --- grille rails (along x, sealed roof with slots far narrower than the disc)
    zg = (g0 + g1) / 2
    k = 0
    y = -iy + c.grille_pitch / 2
    while y < iy:
        box(f"grille_{k}", (0.0, y, zg), (2 * ix, c.grille_w, g1 - g0), dark)
        y += c.grille_pitch
        k += 1
    base = prim_path.rsplit("/", 1)[0]
    _hinge(stage, f"{prim_path}/pitch_hinge", f"{base}/Frame", prim_path,
           axis="Y", local_pos0=(0.0, 0.0, 0.0), local_pos1=(0.0, 0.0, 0.0),
           limit_deg=c.tilt_limit_deg, k_drive=c.drive_k, d_drive=c.drive_d)
    return root


def _spawn_disc(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The red checker: one slick cylinder, authored mass (not density — root
    MassAPI mass; a custom spawner applies no cfg mass schema)."""
    from pxr import Gf, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rb(root, mass=c.disc_mass, lin_damp=0.02, ang_damp=0.05)
    mat = _friction_material(stage, f"{prim_path}/slickMat", c.mu_s, c.mu_d)
    cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/body")
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(float(c.disc_r))
    cyl.CreateHeightAttr(float(c.disc_h))
    cyl.CreateExtentAttr([Gf.Vec3f(-c.disc_r, -c.disc_r, -c.disc_h / 2),
                          Gf.Vec3f(c.disc_r, c.disc_r, c.disc_h / 2)])
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*c.disc_color)])
    _collide(cyl.GetPrim(), c.contact_offset, mat)
    return root


def _spawn_beacon(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The green target-marker post: KINEMATIC (no joints attach to it, so
    kinematic is safe; it is re-posed to the target side each reset)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rb(root, kinematic=True, iters=False)
    _add_box(stage, f"{prim_path}/post", center=(0.0, 0.0, 0.12),
             size=(0.05, 0.05, 0.16), color=c.beacon_color,
             contact_offset=c.contact_offset)
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
            stand_mass: float = 40.0
            base_x: float = 0.72
            base_y: float = 0.52
            column_top: float = 0.14
            tower_x: float = 0.30
            pivot_z: float = 0.26
            stand_color: tuple = (0.35, 0.35, 0.38)
            contact_offset: float = 0.002

        @configclass
        class FrameSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_frame)
            frame_rx: float = 0.24
            frame_ry: float = 0.15
            frame_sec: float = 0.03
            pivot_z: float = 0.26
            tilt_limit_deg: float = 12.0
            drive_k: float = 0.06
            drive_d: float = 0.008
            frame_density: float = 1200.0
            frame_color: tuple = (0.20, 0.22, 0.55)
            contact_offset: float = 0.002

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            ix: float = 0.15
            iy: float = 0.11
            wall_t: float = 0.016
            floor_z0: float = -0.020
            floor_z1: float = -0.012
            pocket_z0: float = -0.027
            pocket_z1: float = -0.019
            rim_top: float = 0.030
            inner_top: float = 0.010
            grille_z0: float = 0.010
            grille_z1: float = 0.016
            grille_pitch: float = 0.020
            grille_w: float = 0.008
            pk_x0: float = 0.088
            pk_y0: float = 0.048
            gap_half: float = 0.041
            baffle_half: float = 0.006
            fun_x0: float = -0.085
            fun_x1: float = -0.006
            tilt_limit_deg: float = 12.0
            drive_k: float = 0.06
            drive_d: float = 0.008
            mu_s: float = 0.12
            mu_d: float = 0.10
            tray_density: float = 900.0
            tray_color: tuple = (0.80, 0.72, 0.55)
            tray_color2: tuple = (0.45, 0.38, 0.28)
            contact_offset: float = 0.002

        @configclass
        class DiscSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_disc)
            disc_r: float = 0.024
            disc_h: float = 0.012
            disc_mass: float = 0.030
            mu_s: float = 0.12
            mu_d: float = 0.10
            disc_color: tuple = (0.85, 0.10, 0.10)
            contact_offset: float = 0.002

        @configclass
        class BeaconSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beacon)
            beacon_color: tuple = (0.10, 0.75, 0.20)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["stand"] = StandSpawnerCfg
        _SPAWNER_CACHE["frame"] = FrameSpawnerCfg
        _SPAWNER_CACHE["tray"] = TraySpawnerCfg
        _SPAWNER_CACHE["disc"] = DiscSpawnerCfg
        _SPAWNER_CACHE["beacon"] = BeaconSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class TiltLabyrinthSceneCfg(BaseCfg):
    """Config for `TiltLabyrinthScene`. The maze geometry is honest by
    construction — the load-bearing clauses are asserted in __post_init__."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    level_max_deg: float = tunable(3.0)     # case counts as level within this of horizontal
    settle_lin: float = tunable(0.04)       # max disc |lin vel| when judging (m/s; above the
    # GPU cylinder-creep artifact — vel iters 4 authored on the disc)
    settle_ang: float = tunable(0.25)       # max tray/frame |ang vel| when judging (rad/s)
    pocket_x_pad: float = tunable(0.006)    # pocket-entry margin on the x ledge side
    pocket_y_pad: float = tunable(0.006)    # pocket-entry margin on the y ledge side
    pocket_z_slack: float = tunable(0.0035)  # disc centre must sit below floor_top+h/2-this

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    stand_yaw_deg: float = tunable(20.0)    # stand yaw about nominal (+/- deg)
    stand_jitter: float = tunable(0.05)     # stand xy jitter (+/- m)
    disc_x_range: tuple = tunable((-0.135, -0.095))  # disc start cell (tray-local x)
    disc_y_range: tuple = tunable((-0.070, 0.070))   # disc start cell (tray-local y)

    # --- info: layout ---------------------------------------------------------------------------
    stand_pos: tuple = info((0.35, 0.0))    # stand origin on the ground (nominal)
    pivot_z: float = info(0.26)             # gimbal pivot height above the ground
    stand_mass: float = info(40.0)
    base_x: float = info(0.72)
    base_y: float = info(0.52)
    column_top: float = info(0.14)
    tower_x: float = info(0.30)
    frame_rx: float = info(0.24)
    frame_ry: float = info(0.15)
    tilt_limit_deg: float = info(12.0)
    drive_k: float = info(0.06)             # spring-return drive stiffness (USD angular units)
    drive_d: float = info(0.008)
    # tray interior (tray-local, origin at the pivot)
    ix: float = info(0.15)                  # interior half-extent x
    iy: float = info(0.11)                  # interior half-extent y
    wall_t: float = info(0.016)             # rim wall thickness (the press surface width)
    floor_z1: float = info(-0.012)          # main floor TOP
    pocket_z1: float = info(-0.019)         # pocket floor TOP (7 mm sunken)
    rim_top: float = info(0.030)            # press-rim top
    inner_top: float = info(0.010)          # interior wall / grille-bottom height
    grille_z1: float = info(0.016)
    grille_pitch: float = info(0.020)
    grille_w: float = info(0.008)           # slot width = pitch - w = 12 mm << disc diameter
    pk_x0: float = info(0.088)              # pocket hole x span [pk_x0, ix]
    pk_y0: float = info(0.048)              # pocket hole |y| span [pk_y0, iy]
    gap_half: float = info(0.041)           # baffle gap half-width (centred at y=0)
    baffle_half: float = info(0.006)
    fun_x0: float = info(-0.085)            # funnel guide walls run x [fun_x0, fun_x1]
    fun_x1: float = info(-0.006)
    # disc
    disc_r: float = info(0.024)
    disc_h: float = info(0.012)
    disc_mass: float = info(0.030)
    mu_s: float = info(0.12)
    mu_d: float = info(0.10)
    # beacon (stand-local)
    beacon_x: float = info(0.10)
    beacon_y: float = info(0.21)
    # rubric weights (0.25 + 0.35 = 0.60 = the non-success cap)
    w_gap: float = info(0.25)
    w_pocket: float = info(0.35)
    gap_x_min: float = info(0.030)          # entered-far-chamber latch: disc x beyond this
    inband_z_max: float = info(0.005)       # ...and z below this (a disc ON the grille: ~0.022)
    # colors / misc
    stand_color: tuple = info((0.35, 0.35, 0.38))
    frame_color: tuple = info((0.20, 0.22, 0.55))
    tray_color: tuple = info((0.80, 0.72, 0.55))
    tray_color2: tuple = info((0.45, 0.38, 0.28))
    disc_color: tuple = info((0.85, 0.10, 0.10))
    beacon_color: tuple = info((0.10, 0.75, 0.20))
    contact_offset: float = info(0.002)

    def __post_init__(self) -> None:
        """Audit the maze geometry (all lengths in metres)."""
        r, h = self.disc_r, self.disc_h
        # the disc passes the baffle gap and the funnel exit with real margin
        assert self.gap_half - r >= 0.015
        # the grille seals: slots far narrower than the disc, and the clear height
        # (floor top -> grille bottom) is well below the disc DIAMETER, so the disc
        # can neither exit nor stand up / flip inside
        assert self.grille_pitch - self.grille_w <= 0.012 < 2 * r
        clear = self.inner_top - self.floor_z1
        assert h + 0.004 <= clear < 2 * r - 0.020
        # each pocket admits the disc flat with >= 10 mm xy slack, and the 7 mm
        # ledge retains it (ledge depth >~ half the disc height)
        assert (self.ix - self.pk_x0) >= 2 * r + 0.010
        assert (self.iy - self.pk_y0) >= 2 * r + 0.010
        depth = self.floor_z1 - self.pocket_z1
        assert depth >= 0.5 * h
        # the funnel exit feeds the gap: exit half-aperture (wall line end minus
        # half wall thickness) admits the disc and lands inside the gap span
        exit_half = self.gap_half - 0.003
        assert exit_half - 0.005 > r + 0.005
        assert exit_half <= self.gap_half
        # pocket predicate z-threshold separates in-pocket from on-floor by > 2 mm
        z_pocket = self.pocket_z1 + h / 2
        z_floor = self.floor_z1 + h / 2
        z_th = self.floor_z1 + h / 2 - self.pocket_z_slack
        assert z_pocket < z_th - 0.002 and z_floor > z_th + 0.002
        # embodiment: pressing the rim to the tilt stop needs only a few newtons
        # (drive_k is authored in USD angular units, torque per DEGREE)
        press_f = self.drive_k * self.tilt_limit_deg / self.ix
        assert press_f < 8.0
        # rim press surface is a real fingertip pad
        assert self.wall_t >= 0.012
        # the tilted tray clears the stand column: worst corner at BOTH hinge
        # limits — combined tilt theta = acos(cos^2(limit)), lowest local point
        # is the pocket bottom (pocket_z1 - 8 mm slab)
        theta = math.acos(math.cos(math.radians(self.tilt_limit_deg)) ** 2)
        dip = math.hypot(self.ix + self.wall_t, self.iy + self.wall_t) * math.sin(theta)
        low = self.pivot_z + (self.pocket_z1 - 0.008) - dip
        assert low > self.column_top + 0.005


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


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("tilt_labyrinth")
class TiltLabyrinthScene(BaseScene):
    cfg: TiltLabyrinthSceneCfg

    def __init__(self, cfg: TiltLabyrinthSceneCfg | None = None) -> None:
        super().__init__(cfg or TiltLabyrinthSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        stand_spawn = cls["stand"](
            stand_mass=c.stand_mass, base_x=c.base_x, base_y=c.base_y,
            column_top=c.column_top, tower_x=c.tower_x, pivot_z=c.pivot_z,
            stand_color=c.stand_color, contact_offset=c.contact_offset)
        frame_spawn = cls["frame"](
            frame_rx=c.frame_rx, frame_ry=c.frame_ry, pivot_z=c.pivot_z,
            tilt_limit_deg=c.tilt_limit_deg, drive_k=c.drive_k, drive_d=c.drive_d,
            frame_color=c.frame_color, contact_offset=c.contact_offset)
        tray_spawn = cls["tray"](
            ix=c.ix, iy=c.iy, wall_t=c.wall_t, floor_z1=c.floor_z1,
            pocket_z1=c.pocket_z1, rim_top=c.rim_top, inner_top=c.inner_top,
            grille_z1=c.grille_z1, grille_pitch=c.grille_pitch, grille_w=c.grille_w,
            pk_x0=c.pk_x0, pk_y0=c.pk_y0, gap_half=c.gap_half,
            baffle_half=c.baffle_half, fun_x0=c.fun_x0, fun_x1=c.fun_x1,
            tilt_limit_deg=c.tilt_limit_deg, drive_k=c.drive_k, drive_d=c.drive_d,
            mu_s=c.mu_s, mu_d=c.mu_d, tray_color=c.tray_color,
            tray_color2=c.tray_color2, contact_offset=c.contact_offset)
        disc_spawn = cls["disc"](
            disc_r=c.disc_r, disc_h=c.disc_h, disc_mass=c.disc_mass,
            mu_s=c.mu_s, mu_d=c.mu_d, disc_color=c.disc_color,
            contact_offset=c.contact_offset)
        beacon_spawn = cls["beacon"](beacon_color=c.beacon_color,
                                     contact_offset=c.contact_offset)

        px, py = c.stand_pos
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
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=stand_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0)),
            ),
            "frame": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Frame",
                spawn=frame_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, c.pivot_z)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=tray_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, c.pivot_z)),
            ),
            "disc": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Disc",
                spawn=disc_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px - 0.11, py, c.pivot_z - 0.004)),
            ),
            "beacon": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beacon",
                spawn=beacon_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px + c.beacon_x, py + c.beacon_y, 0.0)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # solve/smoke drive the case via set_external_force_and_torque;
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
        self.stand: RigidObject = env.iscene["stand"]
        self.frame: RigidObject = env.iscene["frame"]
        self.tray: RigidObject = env.iscene["tray"]
        self.disc: RigidObject = env.iscene["disc"]
        self.beacon: RigidObject = env.iscene["beacon"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # target pocket side: +1 -> the +y pocket (tray-local), -1 -> the -y pocket
        self.target_sign = torch.ones(n, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._gap_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self._gap_cnt = torch.zeros(n, dtype=torch.long, device=dev)
        self._pocket_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self._pocket_cnt = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the stand (yaw + xy jitter), hang frame and tray
        LEVEL on their hinges (poses consistent with the authored joint frames —
        the whole linkage is written together: teleporting one body of a jointed
        pair gets depenetrated back by the other), drop the disc into a random
        start cell, park the beacon at the sampled target side, clear latches.
        Side sampling uses torch.rand (first randint after a manual_seed is
        near-degenerate)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.stand_yaw_deg)
        q = _qz(yaw)
        sp = torch.zeros(m, 3, device=dev)
        sp[:, 0] = c.stand_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.stand_jitter
        sp[:, 1] = c.stand_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.stand_jitter

        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = sp + origin
        st[:, 3:7] = q
        self.stand.write_root_state_to_sim(st, env_ids)

        pivot = sp.clone()
        pivot[:, 2] = c.pivot_z
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pivot + origin
        st[:, 3:7] = q
        self.frame.write_root_state_to_sim(st, env_ids)
        self.tray.write_root_state_to_sim(st.clone(), env_ids)

        # disc: random start cell in the start chamber, resting on the floor
        loc = torch.zeros(m, 3, device=dev)
        x0, x1 = c.disc_x_range
        y0, y1 = c.disc_y_range
        loc[:, 0] = x0 + torch.rand(m, device=dev) * (x1 - x0)
        loc[:, 1] = y0 + torch.rand(m, device=dev) * (y1 - y0)
        loc[:, 2] = c.floor_z1 + c.disc_h / 2 + 0.002
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pivot + quat_apply(q, loc) + origin
        st[:, 3:7] = q
        self.disc.write_root_state_to_sim(st, env_ids)

        # target side + beacon
        sgn = torch.where(torch.rand(m, device=dev) < 0.5,
                          torch.ones(m, device=dev), -torch.ones(m, device=dev))
        self.target_sign[env_ids] = sgn
        bloc = torch.zeros(m, 3, device=dev)
        bloc[:, 0] = c.beacon_x
        bloc[:, 1] = sgn * c.beacon_y
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = sp + quat_apply(q, bloc) + origin
        st[:, 3:7] = q
        self.beacon.write_root_state_to_sim(st, env_ids)

        self._gap_latch[env_ids] = False
        self._gap_cnt[env_ids] = 0
        self._pocket_latch[env_ids] = False
        self._pocket_cnt[env_ids] = 0

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "frame": self.frame.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "disc": self.disc.data.root_state_w[env_ids].clone(),
            "beacon": self.beacon.data.root_state_w[env_ids].clone(),
            "target_sign": self.target_sign[env_ids].clone(),
            "gap_latch": self._gap_latch[env_ids].clone(),
            "gap_cnt": self._gap_cnt[env_ids].clone(),
            "pocket_latch": self._pocket_latch[env_ids].clone(),
            "pocket_cnt": self._pocket_cnt[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.frame.write_root_state_to_sim(state["frame"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.disc.write_root_state_to_sim(state["disc"], env_ids)
        self.beacon.write_root_state_to_sim(state["beacon"], env_ids)
        self.target_sign[env_ids] = state["target_sign"]
        self._gap_latch[env_ids] = state["gap_latch"]
        self._gap_cnt[env_ids] = state["gap_cnt"]
        self._pocket_latch[env_ids] = state["pocket_latch"]
        self._pocket_cnt[env_ids] = state["pocket_cnt"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A tilt-labyrinth game stands on a grey pedestal: a shallow rectangular "
            "CASE (interior 300 x 220 mm, tan floor and rim) hangs at about "
            f"{c.pivot_z * 100:.0f} cm height in a blue two-axis GIMBAL — the case "
            "pivots on a lengthwise hinge inside the blue ring, and the ring pivots "
            "on a crosswise hinge in the pedestal's bearing towers. Both hinges are "
            f"spring-centred to level and stop at +/-{c.tilt_limit_deg:.0f} degrees. "
            "PRESSING DOWN on the case's flat rim (16 mm wide, all four sides) tilts "
            "the play field toward the pressed edge; releasing lets the springs "
            "re-level it. A RED CHECKER (a 48 mm disc) lies inside the case UNDER a "
            "slotted grille roof: you can see it through the 12 mm slots but you can "
            "never touch, grasp or lift it — gravity, commanded through tilt, is the "
            "only way to move it.\n"
            "The interior is a maze. The checker starts in the chamber at one end "
            "(the funnel end, farther from the pockets). Two angled dark guide walls "
            "funnel that chamber into a central GAP (82 mm) in a full-height dark "
            "baffle that spans the case. Behind the baffle, at the far end, the "
            "floor has TWO square sunken POCKETS, one in each corner, 7 mm deep — "
            "deep enough that a checker that drops in cannot come back out at any "
            "tilt. Only ONE pocket is the goal: a GREEN POST standing on the "
            "pedestal beside the target corner marks it; the mirrored pocket is a "
            "decoy, and a checker settled in the decoy is a permanent failure.\n"
            "Goal: tilt the case (tilt toward the baffle to send the checker "
            "through the funnel and gap, then tilt toward the green post's side to "
            "roll it along the far wall) until the red checker drops into the "
            "GREEN-MARKED corner pocket, then release the rim and let the springs "
            "level the case. Success: the checker settled in the beacon-side "
            f"pocket with the case level (within {c.level_max_deg:.0f} degrees) and "
            "everything at rest. The pedestal's heading, the checker's start cell "
            "and which corner is the target all vary per episode."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Press down on the case rim to tilt the labyrinth and roll the sealed "
            "red checker through the baffle gap, then steer it into the sunken "
            "corner pocket marked by the green post. Do not let it drop into the "
            "opposite decoy pocket — that is a permanent failure. Finish with the "
            "rim released so the case springs back level, checker in the marked "
            "pocket."
        )

    # ----- frames / live predicates -------------------------------------------------------------
    def disc_local(self) -> torch.Tensor:
        """(N, 3): disc centre in the TRAY frame (origin at the pivot)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.tray.data.root_quat_w,
                                  self.disc.data.root_pos_w - self.tray.data.root_pos_w)

    def tray_up(self) -> torch.Tensor:
        """(N, 3): the tray's +z axis in world."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device) \
            .expand(self.env.num_envs, 3)
        return quat_apply(self.tray.data.root_quat_w, ez)

    def tilt_deg(self) -> torch.Tensor:
        """(N,) float: case tilt from horizontal, degrees."""
        return torch.rad2deg(torch.acos(self.tray_up()[:, 2].clamp(-1.0, 1.0)))

    def level(self) -> torch.Tensor:
        return self.tilt_deg() <= self.cfg.level_max_deg

    def in_pocket(self, sign: torch.Tensor) -> torch.Tensor:
        """(N,) bool, geometric: disc centre inside the sunken pocket volume on
        the given side — xy inside the hole (with entry margin) AND centre BELOW
        the on-floor height (the disc has really dropped in). Judged in the tray
        frame, so it holds at any case attitude."""
        c = self.cfg
        p = self.disc_local()
        in_x = (p[:, 0] > c.pk_x0 + c.pocket_x_pad) & (p[:, 0] < c.ix)
        ys = p[:, 1] * sign
        in_y = (ys > c.pk_y0 + c.pocket_y_pad) & (ys < c.iy)
        z_th = c.floor_z1 + c.disc_h / 2 - c.pocket_z_slack
        return in_x & in_y & (p[:, 2] < z_th) & (p[:, 2] > c.pocket_z1 - 0.01)

    def in_target_pocket(self) -> torch.Tensor:
        return self.in_pocket(self.target_sign)

    def in_decoy_pocket(self) -> torch.Tensor:
        return self.in_pocket(-self.target_sign)

    def entered_far(self) -> torch.Tensor:
        """(N,) bool: disc past the baffle, INSIDE the play volume (z-banded so a
        disc resting on the grille roof does not count)."""
        c = self.cfg
        p = self.disc_local()
        return (p[:, 0] > c.gap_x_min) & (p[:, 2] < c.inband_z_max) \
            & (p[:, 1].abs() < c.iy + 0.01)

    def settled(self) -> torch.Tensor:
        c = self.cfg
        return (self.disc.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.tray.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang) \
            & (self.frame.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang) \
            & (self.stand.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in
                         (self.stand, self.frame, self.tray, self.disc)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Advance the latches ONCE per physics step (streak-gated: transients of
        1-2 steps do not latch)."""
        fin = self._finite()
        far = self.entered_far() & fin
        self._gap_cnt = torch.where(far, self._gap_cnt + 1,
                                    torch.zeros_like(self._gap_cnt))
        self._gap_latch |= self._gap_cnt >= 3
        pk = self.in_target_pocket() & fin
        self._pocket_cnt = torch.where(pk, self._pocket_cnt + 1,
                                       torch.zeros_like(self._pocket_cnt))
        self._pocket_latch |= self._pocket_cnt >= 3

    # ----- rubric -------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the red checker rests IN the beacon-side pocket with the
        case LEVEL and everything at rest and finite. All clauses are live
        physical outcomes — the pocket is a real sunken volume (smoke proves it
        retains the disc against full tilt) and level+settled means the rim has
        genuinely been released."""
        return self.in_target_pocket() & self.level() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25*gap_latch + 0.35*pocket_latch (latched; ~0
        for the null policy — the springs hold the case level and the disc parked),
        capped at 0.60 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        base = (c.w_gap * self._gap_latch.float()
                + c.w_pocket * self._pocket_latch.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; the case is driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="tilt_labyrinth", robot="null"))
