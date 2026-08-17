"""BalanceShelfScene — the cupboard's only shelf is a free-pivoting BEAM BALANCE:
evict the heavy blue can that pins it, then load the two red cans one into each
pocket so the beam settles LEVEL.

Derived from rlbench/put_groceries_in_cupboard ("put the groceries in the
cupboard": named grocery among distractors, grasped and SET DOWN on a passive
open shelf — any resting pose on the shelf surface, reached by lowering from
above, is terminal). Here the shelf itself is a DYNAMIC mechanism and the goal
is a physical equilibrium of that mechanism, not a resting pose:

  1. The cupboard's single shelf is a beam on a free revolute pivot (axis along
     the cupboard's depth, hard stops at +/- stop_deg). Two fenced POCKETS sit
     at the beam's ends; a weighted keel under the pivot makes the empty (or
     symmetrically loaded) beam self-level, while ANY single load pins it hard
     against a stop.
  2. A heavy BLUE decoy can starts seated in one pocket (random side), so the
     shelf starts slammed to that stop. The seed's strategy — set a grocery
     down on the shelf — cannot succeed: with the decoy aboard the beam can
     never level (the decoy out-torques a red can even head-to-head), so the
     decoy must first be TAKEN OFF the shelf and set aside.
  3. The two RED cans (the groceries, on the ground in front) must then be
     loaded ONE INTO EACH pocket. A single red can pins the beam to its stop;
     only the symmetric two-can load balances. The second can must be placed
     into a pocket that is TILTED at the stop — and the beam levels only after
     the load is symmetric.
  Success is live equilibrium: both red cans seated upright one per pocket,
  the beam level within level_tol_deg, the blue can NOT on the beam, and
  everything settled and finite.

Execution order is forced by physics, not by rubric timestamps: while the blue
can is aboard the beam cannot level, and the beam levels only once the red
loading is symmetric.

Assets are fully procedural (compound spawners; per-child density on the beam
so the pivot inertia and the keel restoring torque are real; root MassAPI on
the heavy cupboard — custom spawners apply no cfg mass schemas, so mass is
authored in the funcs):
  - cupboard: heavy DYNAMIC body (a jointed body0 must not be kinematic or the
    anchor stays world-fixed after the reset teleport): plinth 340 x 500 mm
    (top z 0.05), side walls (inner faces y +/-0.235, z 0.05-0.44), a back
    wall (inner face x -0.155), a roof (z 0.44-0.46), OPEN front, and two
    pivot posts flanking the beam's axle stubs.
  - beam: DYNAMIC compound on a spawn-authored RevoluteJoint (axis X, limits
    +/- stop_deg, damped): two 92 mm square pocket plates (top 30 mm below the
    pivot) at y +/-0.15 with 30 mm fences, a narrow bridge, a keel post and a
    1.0 kg BOB 121 mm under the pivot (the restoring pendulum), and two axle
    stubs along +/-x.
  - two RED cans D64 x 90 mm, 0.30 kg (the groceries), upright on the ground
    in front of the cupboard at random bearings.
  - one BLUE can D68 x 115 mm, 0.62 kg (the decoy), seated in a random pocket.

Torque budget (audited in __post_init__, from the authored densities): keel
restoring at the stop ~0.26 N*m; one red can drives ~0.42 N*m (pins, ratio
>1.3); blue-vs-red mismatch drives ~0.46 N*m (pins even head-to-head); worst
case in-pocket placement slack tilts the balanced beam ~1.7 deg << level tol.

Per-episode randomization (readback-verifiable): cupboard yaw +/-15 deg + xy
jitter, decoy pocket side (the beam settles to the matching stop during the
initial settle), red-can ground bearings/distances/yaws + which-can-where.

Rubric (0..1; latched stage credit anchored in the demonstrated solve):
  0.15 * unloaded — blue can off the beam AND beam level (latched, 3-step
                    persistence; requires eviction AND waiting out the swing)
  0.25 * first    — a red can ever seated upright in either pocket (latched)
  0.30 * both     — both red cans seated, one per pocket (latched)
  1.0 iff success() — matched red load, beam level, blue can off the beam,
                    all settled, all finite (live).
  Non-success capped at 0.70; null policy ~0 (the decoy pins the beam from
  step 0, so `unloaded` can never latch without eviction).

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

_G = 9.81

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
             axis: str = "Z", density: float | None = None):
    """One cylinder child along `axis` ("Z" or "X")."""
    from pxr import Gf, UsdGeom, UsdPhysics

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr(axis)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    if axis == "Z":
        ext = [Gf.Vec3f(-radius, -radius, -height / 2),
               Gf.Vec3f(radius, radius, height / 2)]
    else:  # X
        ext = [Gf.Vec3f(-height / 2, -radius, -radius),
               Gf.Vec3f(height / 2, radius, radius)]
    cyl.CreateExtentAttr(ext)
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
def _spawn_cupboard(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The cupboard: heavy DYNAMIC compound. Local frame: origin on the ground
    under the pivot; local +x = OPEN FRONT direction. Plinth, two side walls,
    a back wall, a roof, and two pivot posts flanking the beam axle."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.cup_mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    wood, post_c = c.cup_color, c.post_color

    dx, dy = c.cup_dx, c.cup_dy               # plinth footprint (x depth, y width)
    wz0, wz1 = c.plinth_h, c.roof_z0          # wall vertical span
    zw, hw = (wz0 + wz1) / 2, wz1 - wz0
    # plinth
    _add_box(stage, f"{prim_path}/plinth", center=(0.0, 0.0, c.plinth_h / 2),
             size=(dx, dy, c.plinth_h), color=wood, collide=collide)
    # side walls: inner faces at y = +/- wall_in
    yc = c.wall_in + c.wall_t / 2
    _add_box(stage, f"{prim_path}/wall_yp", center=(0.0, yc, zw),
             size=(dx, c.wall_t, hw), color=wood, collide=collide)
    _add_box(stage, f"{prim_path}/wall_yn", center=(0.0, -yc, zw),
             size=(dx, c.wall_t, hw), color=wood, collide=collide)
    # back wall: inner face at x = -back_in (front at +x stays OPEN)
    _add_box(stage, f"{prim_path}/wall_back",
             center=(-(c.back_in + c.wall_t / 2), 0.0, zw),
             size=(c.wall_t, dy, hw), color=wood, collide=collide)
    # roof
    _add_box(stage, f"{prim_path}/roof",
             center=(0.0, 0.0, (c.roof_z0 + c.roof_z1) / 2),
             size=(dx, dy, c.roof_z1 - c.roof_z0), color=wood, collide=collide)
    # pivot posts flanking the axle stubs (inner faces at x = +/- post_in)
    xc = c.post_in + c.post_w / 2
    zp = (c.plinth_h + c.post_top) / 2
    for s, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/post_{tag}",
                 center=(s * xc, 0.0, zp),
                 size=(c.post_w, c.post_w, c.post_top - c.plinth_h),
                 color=post_c, collide=collide)
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The balance beam: DYNAMIC compound, body origin ON the pivot axis.
    Two fenced pocket plates at y = +/- well_y, a narrow bridge, a hub, a keel
    post + heavy BOB under the pivot, and two axle stubs along +/- x.
    Per-child DENSITY so the restoring pendulum torque is real. Spawn-authored
    RevoluteJoint (axis X, limits +/- stop_deg) to the sibling Cupboard."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.2)
    pxrb.CreateAngularDampingAttr(float(c.beam_damping))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    deck, fence_c, bob_c = c.deck_color, c.fence_color, c.bob_color
    rho = c.rho_struct

    zp = c.plate_top - c.plate_t / 2          # plate centre z
    zf = c.plate_top + c.fence_h / 2          # fence centre z
    for s, tag in ((1.0, "p"), (-1.0, "n")):
        ys = s * c.well_y
        _add_box(stage, f"{prim_path}/plate_{tag}", center=(0.0, ys, zp),
                 size=(c.well_w, c.well_w, c.plate_t), color=deck,
                 collide=collide, density=rho)
        # fences: inner clear square well_inner x well_inner, top at local z 0
        off = c.well_inner / 2 + c.fence_t / 2
        _add_box(stage, f"{prim_path}/fence_{tag}_yp",
                 center=(0.0, ys + off, zf),
                 size=(c.well_w, c.fence_t, c.fence_h), color=fence_c,
                 collide=collide, density=rho)
        _add_box(stage, f"{prim_path}/fence_{tag}_yn",
                 center=(0.0, ys - off, zf),
                 size=(c.well_w, c.fence_t, c.fence_h), color=fence_c,
                 collide=collide, density=rho)
        _add_box(stage, f"{prim_path}/fence_{tag}_xp",
                 center=(off, ys, zf),
                 size=(c.fence_t, c.well_inner, c.fence_h), color=fence_c,
                 collide=collide, density=rho)
        _add_box(stage, f"{prim_path}/fence_{tag}_xn",
                 center=(-off, ys, zf),
                 size=(c.fence_t, c.well_inner, c.fence_h), color=fence_c,
                 collide=collide, density=rho)
    # bridge between the pockets (too narrow/off-centre to count as a pocket)
    _add_box(stage, f"{prim_path}/bridge", center=(0.0, 0.0, zp),
             size=(c.bridge_w, c.bridge_len, c.plate_t), color=deck,
             collide=collide, density=rho)
    # hub: carries the axle down to the bridge
    _add_box(stage, f"{prim_path}/hub",
             center=(0.0, 0.0, (c.plate_top - c.plate_t + 0.008) / 2),
             size=(c.hub_w, c.hub_w, 0.008 - (c.plate_top - c.plate_t)),
             color=fence_c, collide=collide, density=rho)
    # keel post + bob: the restoring pendulum
    _add_box(stage, f"{prim_path}/keel",
             center=(0.0, 0.0, c.keel_z),
             size=(c.keel_w, c.keel_w, c.keel_h), color=fence_c,
             collide=collide, density=rho)
    _add_box(stage, f"{prim_path}/bob",
             center=(0.0, 0.0, c.bob_z),
             size=(c.bob_s, c.bob_s, c.bob_s), color=bob_c,
             collide=collide, density=c.rho_bob)
    # axle stubs along +/- x (between the cupboard posts, never touching)
    ln = c.stub_x1 - c.stub_x0
    for s, tag in ((1.0, "p"), (-1.0, "n")):
        _add_cyl(stage, f"{prim_path}/stub_{tag}",
                 center=(s * (c.stub_x0 + ln / 2), 0.0, 0.0),
                 radius=c.stub_r, height=ln, color=bob_c, collide=collide,
                 axis="X", density=rho)

    # revolute pivot to the sibling cupboard (axis X, hard stops)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/pivot")
    j.CreateBody0Rel().SetTargets([f"{base}/Cupboard"])
    j.CreateBody1Rel().SetTargets([prim_path])
    # beam<->cupboard contact stays ON (USD default for a joint pair is
    # filtered); authored clearances are >= 7 mm everywhere in the sweep
    j.CreateCollisionEnabledAttr(True)
    j.CreateAxisAttr("X")
    j.CreateLowerLimitAttr(float(-c.stop_deg))
    j.CreateUpperLimitAttr(float(c.stop_deg))
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.pivot_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    return root


def _spawn_can(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A can: one z-cylinder, body origin at its centre, +z up."""
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


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cupboard" not in _SPAWNER_CACHE:

        @configclass
        class CupboardSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cupboard)
            cup_mass: float = 40.0
            cup_dx: float = 0.34
            cup_dy: float = 0.50
            plinth_h: float = 0.05
            wall_in: float = 0.235
            back_in: float = 0.155
            wall_t: float = 0.015
            roof_z0: float = 0.44
            roof_z1: float = 0.46
            post_in: float = 0.09
            post_w: float = 0.04
            post_top: float = 0.24
            cup_color: tuple = (0.55, 0.40, 0.24)
            post_color: tuple = (0.35, 0.25, 0.15)
            contact_offset: float = 0.002

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            pivot_z: float = 0.22
            stop_deg: float = 12.0
            well_y: float = 0.15
            well_w: float = 0.092
            well_inner: float = 0.076
            plate_top: float = -0.030
            plate_t: float = 0.012
            fence_t: float = 0.008
            fence_h: float = 0.030
            bridge_w: float = 0.030
            bridge_len: float = 0.208
            hub_w: float = 0.030
            keel_w: float = 0.020
            keel_h: float = 0.060
            keel_z: float = -0.066
            bob_s: float = 0.050
            bob_z: float = -0.121
            stub_r: float = 0.008
            stub_x0: float = 0.015
            stub_x1: float = 0.082
            rho_struct: float = 400.0
            rho_bob: float = 8000.0
            beam_damping: float = 8.0
            deck_color: tuple = (0.80, 0.68, 0.44)
            fence_color: tuple = (0.62, 0.48, 0.26)
            bob_color: tuple = (0.22, 0.22, 0.24)
            contact_offset: float = 0.002

        @configclass
        class CanSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_can)
            can_r: float = 0.032
            can_h: float = 0.090
            can_density: float = 1036.0
            can_color: tuple = (0.85, 0.12, 0.12)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["cupboard"] = CupboardSpawnerCfg
        _SPAWNER_CACHE["beam"] = BeamSpawnerCfg
        _SPAWNER_CACHE["can"] = CanSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class BalanceShelfSceneCfg(BaseCfg):
    """Config for `BalanceShelfScene`. The torque budget is honest by
    construction — every claim is asserted numerically in __post_init__ from
    the authored densities and geometry: one red can pins the beam at a stop;
    the blue can out-torques a red head-to-head; worst-case in-pocket slack
    tilts the balanced beam far less than the level tolerance; the blue can
    cannot prop up a pocket from the plinth."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    level_tol_deg: float = tunable(6.0)     # |beam tilt| below this counts as LEVEL
    stop_deg: float = tunable(12.0)         # revolute hard-stop half-travel (deg)
    seat_xy_tol: float = tunable(0.030)     # |beam-local xy offset| from pocket centre (m)
    seat_z_lo: float = tunable(0.008)       # seat z band about the expected centre (m)
    seat_z_hi: float = tunable(0.020)
    upright_tol_deg: float = tunable(20.0)  # can axis within this of the beam normal
    settle_lin: float = tunable(0.05)       # max can/cupboard |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.10)       # max beam |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    cup_yaw_deg: float = tunable(15.0)      # cupboard yaw jitter about nominal (+/- deg)
    cup_jitter: float = tunable(0.04)       # cupboard xy jitter (+/- m)
    can_bear_range: tuple = tunable((18.0, 40.0))   # red-can |bearing| off the front dir (deg)
    can_dist_range: tuple = tunable((0.40, 0.52))   # red-can distance from the pivot axis (m)

    # --- info: cupboard layout ------------------------------------------------------------------
    cup_pos: tuple = info((0.50, 0.0))      # pivot axis on the ground (nominal)
    cup_yaw_nom_deg: float = info(180.0)    # nominal heading: open front (+x local) faces robot
    cup_mass: float = info(40.0)
    cup_dx: float = info(0.34)              # plinth/roof footprint: x (depth)
    cup_dy: float = info(0.50)              # y (width)
    plinth_h: float = info(0.05)
    wall_in: float = info(0.235)            # side-wall inner faces at y = +/- this
    back_in: float = info(0.155)            # back-wall inner face at x = - this
    wall_t: float = info(0.015)
    roof_z0: float = info(0.44)
    roof_z1: float = info(0.46)
    post_in: float = info(0.09)             # post inner faces at x = +/- this
    post_w: float = info(0.04)
    post_top: float = info(0.24)
    # beam
    pivot_z: float = info(0.22)             # pivot height above the ground (cupboard local)
    well_y: float = info(0.15)              # pocket centres at beam-local y = +/- this
    well_w: float = info(0.092)             # pocket plate square side
    well_inner: float = info(0.076)         # fence inner clear square side
    plate_top: float = info(-0.030)         # pocket floor, beam-local z
    plate_t: float = info(0.012)
    fence_t: float = info(0.008)
    fence_h: float = info(0.030)            # fence rim top at beam-local z 0
    bridge_w: float = info(0.030)
    bridge_len: float = info(0.208)
    hub_w: float = info(0.030)
    keel_w: float = info(0.020)
    keel_h: float = info(0.060)
    keel_z: float = info(-0.066)
    bob_s: float = info(0.050)
    bob_z: float = info(-0.121)             # bob centre, beam-local z (the pendulum)
    stub_r: float = info(0.008)
    stub_x0: float = info(0.015)
    stub_x1: float = info(0.082)
    rho_struct: float = info(400.0)
    rho_bob: float = info(8000.0)
    beam_damping: float = info(8.0)
    # cans
    can_r: float = info(0.032)              # red grocery can
    can_h: float = info(0.090)
    can_density: float = info(1036.0)       # -> ~0.30 kg
    decoy_r: float = info(0.034)            # blue decoy can
    decoy_h: float = info(0.115)
    decoy_density: float = info(1484.0)     # -> ~0.62 kg
    # rubric weights (0.15 + 0.25 + 0.30 = 0.70 = the non-success cap)
    w_unload: float = info(0.15)
    w_first: float = info(0.25)
    w_both: float = info(0.30)
    # colors
    cup_color: tuple = info((0.55, 0.40, 0.24))
    post_color: tuple = info((0.35, 0.25, 0.15))
    deck_color: tuple = info((0.80, 0.68, 0.44))
    fence_color: tuple = info((0.62, 0.48, 0.26))
    bob_color: tuple = info((0.22, 0.22, 0.24))
    can_color: tuple = info((0.85, 0.12, 0.12))
    decoy_color: tuple = info((0.15, 0.30, 0.85))
    contact_offset: float = info(0.002)

    # ----- derived masses / torques (from the authored densities) -------------------------------
    def _beam_pendulum(self) -> tuple:
        """(beam mass, restoring coefficient K = M*g*depth [N*m per sin(tilt)])."""
        rs = self.rho_struct
        parts = []  # (mass, z)
        zp = self.plate_top - self.plate_t / 2
        parts += [(rs * self.well_w**2 * self.plate_t, zp)] * 2
        zf = self.plate_top + self.fence_h / 2
        m_fy = rs * self.well_w * self.fence_t * self.fence_h
        m_fx = rs * self.fence_t * self.well_inner * self.fence_h
        parts += [(m_fy, zf)] * 4 + [(m_fx, zf)] * 4
        parts += [(rs * self.bridge_w * self.bridge_len * self.plate_t, zp)]
        hub_z0, hub_z1 = self.plate_top - self.plate_t, 0.008
        parts += [(rs * self.hub_w**2 * (hub_z1 - hub_z0), (hub_z0 + hub_z1) / 2)]
        parts += [(rs * self.keel_w**2 * self.keel_h, self.keel_z)]
        parts += [(self.rho_bob * self.bob_s**3, self.bob_z)]
        m_stub = rs * math.pi * self.stub_r**2 * (self.stub_x1 - self.stub_x0)
        parts += [(m_stub, 0.0)] * 2
        mass = sum(m for m, _ in parts)
        k = -_G * sum(m * z for m, z in parts)  # z < 0 -> restoring
        return mass, k

    def _can_masses(self) -> tuple:
        m_r = self.can_density * math.pi * self.can_r**2 * self.can_h
        m_d = self.decoy_density * math.pi * self.decoy_r**2 * self.decoy_h
        return m_r, m_d

    def _drive_torque(self, m: float, com_z: float) -> float:
        """Torque a mass in a pocket exerts about the pivot at the stop angle."""
        th = math.radians(self.stop_deg)
        return m * _G * (self.well_y * math.cos(th) - com_z * math.sin(th))

    def __post_init__(self) -> None:
        """Audit the torque budget and access geometry (m, deg, N*m)."""
        m_beam, k_rest = self._beam_pendulum()
        m_r, m_d = self._can_masses()
        rest_stop = k_rest * math.sin(math.radians(self.stop_deg))
        com_r = self.plate_top + self.can_h / 2
        com_d = self.plate_top + self.decoy_h / 2
        # one red can PINS the beam at the stop (with margin)
        assert self._drive_torque(m_r, com_r) > 1.3 * rest_stop, \
            (self._drive_torque(m_r, com_r), rest_stop)
        # the decoy out-torques a red can head-to-head (so no level with it aboard)
        assert self._drive_torque(m_d, com_d) - self._drive_torque(m_r, com_r) \
            > 1.1 * rest_stop
        # the decoy alone pins hard
        assert self._drive_torque(m_d, com_d) > 2.0 * rest_stop
        # worst-case in-pocket slack tilts the balanced beam far under the tolerance
        slack = self.well_inner - 2 * self.can_r          # total opposing offset
        tilt = math.degrees(math.asin(m_r * _G * slack / k_rest))
        assert tilt < self.level_tol_deg / 2, (tilt, self.level_tol_deg)
        # pinned (stop) and LEVEL are clearly distinct states
        assert self.stop_deg >= self.level_tol_deg + 3.0
        # both cans fit the pocket with slack; jaws fit a Franka gripper (<80 mm)
        assert 2 * self.can_r < self.well_inner - 0.008
        assert 2 * self.decoy_r < self.well_inner - 0.004
        assert 2 * self.can_r < 0.075 and 2 * self.decoy_r < 0.075
        # the bridge cannot satisfy `seated` (its span ends short of the pockets)
        assert self.bridge_len / 2 < self.well_y - self.seat_xy_tol
        # seat z band separates upright (inside) from toppled (below)
        assert self.seat_z_lo < self.can_h / 2 - self.can_r, "toppled can must exit the band"
        # anti-prop: the decoy standing on the plinth cannot reach the LEVEL
        # pocket underside (so a propped beam cannot fake equilibrium)
        assert self.plinth_h + self.decoy_h + 0.005 < self.pivot_z + self.plate_top - self.plate_t
        # sweep clearances (>= 7 mm): axle stubs vs posts; pockets vs side
        # walls; bob vs plinth at the stop; pocket outer corner vs plinth
        assert self.stub_x1 + 0.007 <= self.post_in
        y_ext = self.well_y + self.well_w / 2
        assert y_ext + 0.007 <= self.wall_in
        # back wall is along x; the beam's x half-extent (unchanged by tilt about X)
        x_ext = max(self.well_w / 2, self.stub_x1, self.bob_s / 2,
                    self.bridge_w / 2, self.hub_w / 2)
        assert x_ext + 0.007 <= self.back_in
        th = math.radians(self.stop_deg)
        bob_low = self.pivot_z - (self.bob_s / 2) * math.sin(th) \
            + (self.bob_z - self.bob_s / 2) * math.cos(th)
        assert bob_low > self.plinth_h + 0.007, bob_low
        corner = self.pivot_z - y_ext * math.sin(th) \
            + (self.plate_top - self.plate_t) * math.cos(th)
        assert corner > self.plinth_h + 0.007, corner
        # roof clearance above a can raised in the high pocket at the stop
        top = self.pivot_z + y_ext * math.sin(th) + self.plate_top + self.decoy_h
        assert top + 0.05 < self.roof_z0, top
        # red cans spawn clear of the cupboard footprint
        circum = math.hypot(self.cup_dx / 2, self.cup_dy / 2)
        assert self.can_dist_range[0] > circum + self.can_r + self.cup_jitter + 0.02
        # rubric weights: latched sum equals the non-success cap
        assert abs(self.w_unload + self.w_first + self.w_both - 0.70) < 1e-9


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
@SCENES.register("balance_shelf")
class BalanceShelfScene(BaseScene):
    cfg: BalanceShelfSceneCfg

    def __init__(self, cfg: BalanceShelfSceneCfg | None = None) -> None:
        super().__init__(cfg or BalanceShelfSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        cup_spawn = cls["cupboard"](
            cup_mass=c.cup_mass, cup_dx=c.cup_dx, cup_dy=c.cup_dy,
            plinth_h=c.plinth_h, wall_in=c.wall_in, back_in=c.back_in,
            wall_t=c.wall_t, roof_z0=c.roof_z0, roof_z1=c.roof_z1,
            post_in=c.post_in, post_w=c.post_w, post_top=c.post_top,
            cup_color=c.cup_color, post_color=c.post_color,
            contact_offset=c.contact_offset)
        beam_spawn = cls["beam"](
            pivot_z=c.pivot_z, stop_deg=c.stop_deg, well_y=c.well_y,
            well_w=c.well_w, well_inner=c.well_inner, plate_top=c.plate_top,
            plate_t=c.plate_t, fence_t=c.fence_t, fence_h=c.fence_h,
            bridge_w=c.bridge_w, bridge_len=c.bridge_len, hub_w=c.hub_w,
            keel_w=c.keel_w, keel_h=c.keel_h, keel_z=c.keel_z,
            bob_s=c.bob_s, bob_z=c.bob_z, stub_r=c.stub_r,
            stub_x0=c.stub_x0, stub_x1=c.stub_x1, rho_struct=c.rho_struct,
            rho_bob=c.rho_bob, beam_damping=c.beam_damping,
            deck_color=c.deck_color, fence_color=c.fence_color,
            bob_color=c.bob_color, contact_offset=c.contact_offset)
        red_a = cls["can"](can_r=c.can_r, can_h=c.can_h,
                           can_density=c.can_density, can_color=c.can_color,
                           contact_offset=c.contact_offset)
        red_b = cls["can"](can_r=c.can_r, can_h=c.can_h,
                           can_density=c.can_density, can_color=c.can_color,
                           contact_offset=c.contact_offset)
        decoy = cls["can"](can_r=c.decoy_r, can_h=c.decoy_h,
                           can_density=c.decoy_density,
                           can_color=c.decoy_color,
                           contact_offset=c.contact_offset)

        px, py = c.cup_pos
        q0 = _qz_t(c.cup_yaw_nom_deg)
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
            "cupboard": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cupboard",
                spawn=cup_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0), rot=q0),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=beam_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px, py, c.pivot_z), rot=q0),
            ),
            "can_a": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/CanA",
                spawn=red_a,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, 1.0, 0.048)),
            ),
            "can_b": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/CanB",
                spawn=red_b,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.4, 1.0, 0.048)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=decoy,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.6, 1.0, 0.061)),
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
        self.cupboard: RigidObject = env.iscene["cupboard"]
        self.beam: RigidObject = env.iscene["beam"]
        self.can_a: RigidObject = env.iscene["can_a"]
        self.can_b: RigidObject = env.iscene["can_b"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.decoy_side = torch.ones(n, device=dev)   # +1 -> pocket at beam +y
        # latches (partial credit survives transients; success is judged live)
        self._unloaded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._first = torch.zeros(n, dtype=torch.bool, device=dev)
        self._both = torch.zeros(n, dtype=torch.bool, device=dev)
        self._unload_cnt = torch.zeros(n, dtype=torch.long, device=dev)
        self._first_cnt = torch.zeros(n, dtype=torch.long, device=dev)
        self._both_cnt = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the cupboard (yaw + xy jitter), write the beam
        LEVEL and joint-consistent on its pivot, seat the blue decoy in a
        random pocket (the beam tips to that stop during the initial settle),
        stand the two red cans on the ground at random bearings in front,
        clear the latches. The whole linkage is written together (teleporting
        one body of a jointed pair gets depenetrated back by the other)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        yaw = math.radians(c.cup_yaw_nom_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.cup_yaw_deg)
        q_cup = _qz(yaw)
        cp = torch.zeros(m, 3, device=dev)
        cp[:, 0] = c.cup_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.cup_jitter
        cp[:, 1] = c.cup_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.cup_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = cp + origin
        st[:, 3:7] = q_cup
        self.cupboard.write_root_state_to_sim(st, env_ids)

        # beam: LEVEL at the pivot (it tips to the decoy's stop while settling)
        pivot_w = cp.clone()
        pivot_w[:, 2] = c.pivot_z
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pivot_w + origin
        st[:, 3:7] = q_cup
        self.beam.write_root_state_to_sim(st, env_ids)

        # decoy: seated in a random pocket (torch.rand comparison — the first
        # randint after manual_seed is near-degenerate)
        side = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0).to(dev)
        self.decoy_side[env_ids] = side
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 1] = side * c.well_y
        loc[:, 2] = c.plate_top + c.decoy_h / 2 + 0.003
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pivot_w + quat_apply(q_cup, loc) + origin
        st[:, 3:7] = q_cup
        self.decoy.write_root_state_to_sim(st, env_ids)

        # red cans: on the ground in front, one bearing each side of the front
        # direction (random magnitudes, random which-can-where, random yaw)
        b0, b1 = c.can_bear_range
        d0, d1 = c.can_dist_range
        sgn_a = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0).to(dev)
        for body, sgn in ((self.can_a, sgn_a), (self.can_b, -sgn_a)):
            mag = b0 + torch.rand(m, device=dev) * (b1 - b0)
            dist = d0 + torch.rand(m, device=dev) * (d1 - d0)
            bear = yaw + sgn * torch.deg2rad(mag)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = cp[:, 0] + dist * torch.cos(bear) + origin[:, 0]
            st[:, 1] = cp[:, 1] + dist * torch.sin(bear) + origin[:, 1]
            st[:, 2] = origin[:, 2] + c.can_h / 2 + 0.003
            st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
            body.write_root_state_to_sim(st, env_ids)

        self._unloaded[env_ids] = False
        self._first[env_ids] = False
        self._both[env_ids] = False
        self._unload_cnt[env_ids] = 0
        self._first_cnt[env_ids] = 0
        self._both_cnt[env_ids] = 0

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cupboard": self.cupboard.data.root_state_w[env_ids].clone(),
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "can_a": self.can_a.data.root_state_w[env_ids].clone(),
            "can_b": self.can_b.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "decoy_side": self.decoy_side[env_ids].clone(),
            "unloaded": self._unloaded[env_ids].clone(),
            "first": self._first[env_ids].clone(),
            "both": self._both[env_ids].clone(),
            "unload_cnt": self._unload_cnt[env_ids].clone(),
            "first_cnt": self._first_cnt[env_ids].clone(),
            "both_cnt": self._both_cnt[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cupboard.write_root_state_to_sim(state["cupboard"], env_ids)
        self.beam.write_root_state_to_sim(state["beam"], env_ids)
        self.can_a.write_root_state_to_sim(state["can_a"], env_ids)
        self.can_b.write_root_state_to_sim(state["can_b"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.decoy_side[env_ids] = state["decoy_side"]
        self._unloaded[env_ids] = state["unloaded"]
        self._first[env_ids] = state["first"]
        self._both[env_ids] = state["both"]
        self._unload_cnt[env_ids] = state["unload_cnt"]
        self._first_cnt[env_ids] = state["first_cnt"]
        self._both_cnt[env_ids] = state["both_cnt"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A wooden CUPBOARD stands on the ground: a low plinth "
            "(340 x 500 mm, 50 mm tall), side and back walls, a roof at "
            "440-460 mm, and an OPEN FRONT facing you. Its only shelf is a "
            "BEAM BALANCE: a beam on a free horizontal pivot (axis along the "
            "cupboard depth, 220 mm high, hard stops at "
            f"+/-{c.stop_deg:.0f} deg) carrying two fenced square POCKETS "
            "(inner 76 x 76 mm, floor 30 mm below the pivot, 30 mm rims), one "
            "at each end, 150 mm left and right of the pivot. A weighted keel "
            "hangs under the pivot: an EMPTY or SYMMETRICALLY loaded beam "
            "slowly self-levels, but any single can aboard out-torques the "
            "keel and slams the beam against a stop. A heavy BLUE can "
            "(68 mm diameter, 115 mm tall, ~0.62 kg) starts seated in one "
            "pocket, so the shelf starts tipped hard to that side; the blue "
            "can is heavier than any red can, so while it is anywhere on the "
            "beam the shelf can NEVER sit level. On the ground in front of "
            "the cupboard stand two RED cans (64 mm diameter, 90 mm tall, "
            "~0.30 kg each) — the groceries.\n"
            "Goal: put the groceries in the cupboard so the shelf balances. "
            "First take the BLUE can off the shelf and set it aside anywhere "
            "off the beam (ground or plinth), and let the beam swing level. "
            "Then place the two RED cans upright ONE INTO EACH pocket — the "
            "first can tips the beam to its stop, and the second (lowered "
            "into the RAISED, tilted pocket) restores balance. Simply setting "
            "cans down while the blue can is aboard, stacking both reds in "
            "one pocket, or leaving a can on the bridge, plinth, roof or "
            "ground does not count. Success: both red cans standing upright, "
            "one seated in each pocket (within 30 mm of the pocket centre), "
            f"the beam LEVEL within {c.level_tol_deg:.0f} deg, the blue can "
            "not on the beam, and everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Take the blue can off the cupboard's balance shelf and set it "
            "aside off the shelf, then put the two red cans upright one into "
            "each pocket of the shelf so the beam settles level. Leave both "
            "red cans seated in their pockets, the beam balanced, and the "
            "blue can off the beam."
        )

    # ----- frames / live predicates -------------------------------------------------------------
    def _beam_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.beam.data.root_quat_w,
                                  pos_w - self.beam.data.root_pos_w)

    def tilt_deg(self) -> torch.Tensor:
        """(N,) float: beam tilt about the pivot relative to the cupboard
        (deg, wrapped; positive = beam +y end UP)."""
        qc = self.cupboard.data.root_quat_w
        qb = self.beam.data.root_quat_w
        qc_inv = qc * torch.tensor([1.0, -1.0, -1.0, -1.0], device=qc.device)
        rel = _qmul(qc_inv, qb)
        return _wrap_deg(torch.rad2deg(2.0 * torch.atan2(rel[:, 1], rel[:, 0])))

    def seated(self, body, h: float, side: torch.Tensor) -> torch.Tensor:
        """(N,) bool, geometric in the BEAM frame: body centre within
        seat_xy_tol of the pocket centre at beam-local y = side*well_y, its z
        in the on-the-plate band, and its axis within upright_tol_deg of the
        beam normal. `side` is a (N,) tensor of +/-1 (or a scalar)."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        loc = self._beam_local(body.data.root_pos_w)
        if not torch.is_tensor(side):
            side = torch.full((loc.shape[0],), float(side), device=loc.device)
        ze = c.plate_top + h / 2
        ok = (loc[:, 0].abs() < c.seat_xy_tol) \
            & ((loc[:, 1] - side * c.well_y).abs() < c.seat_xy_tol) \
            & (loc[:, 2] > ze - c.seat_z_lo) & (loc[:, 2] < ze + c.seat_z_hi)
        ez = torch.tensor([0.0, 0.0, 1.0], device=loc.device) \
            .expand(loc.shape[0], 3)
        axis_w = quat_apply(body.data.root_quat_w, ez)
        axis_b = quat_apply_inverse(self.beam.data.root_quat_w, axis_w)
        return ok & (axis_b[:, 2] >= math.cos(math.radians(c.upright_tol_deg)))

    def aboard(self, body) -> torch.Tensor:
        """(N,) bool: body centre anywhere in the beam's carry volume (beam
        frame). Ground/plinth/roof positions fall far outside the z band."""
        c = self.cfg
        loc = self._beam_local(body.data.root_pos_w)
        return (loc[:, 0].abs() < 0.10) & (loc[:, 1].abs() < 0.22) \
            & (loc[:, 2] > -0.05) & (loc[:, 2] < 0.20)

    def matched(self) -> torch.Tensor:
        """(N,) bool: the two red cans seated ONE PER POCKET (either pairing)."""
        c = self.cfg
        ap = self.seated(self.can_a, c.can_h, 1.0)
        an = self.seated(self.can_a, c.can_h, -1.0)
        bp = self.seated(self.can_b, c.can_h, 1.0)
        bn = self.seated(self.can_b, c.can_h, -1.0)
        return (ap & bn) | (an & bp)

    def any_seated(self) -> torch.Tensor:
        c = self.cfg
        return self.seated(self.can_a, c.can_h, 1.0) \
            | self.seated(self.can_a, c.can_h, -1.0) \
            | self.seated(self.can_b, c.can_h, 1.0) \
            | self.seated(self.can_b, c.can_h, -1.0)

    def level(self) -> torch.Tensor:
        return self.tilt_deg().abs() < self.cfg.level_tol_deg

    def settled(self) -> torch.Tensor:
        c = self.cfg
        return (self.can_a.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.can_b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.beam.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang) \
            & (self.cupboard.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in
                         (self.cupboard, self.beam, self.can_a,
                          self.can_b, self.decoy)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Advance the latches ONCE per physics step (3-step persistence with
        slow gates, so solver transients cannot flash credit)."""
        fin = self._finite()
        slow_beam = self.beam.data.root_ang_vel_w.norm(dim=-1) < 0.5
        slow_cans = (self.can_a.data.root_lin_vel_w.norm(dim=-1) < 0.5) \
            & (self.can_b.data.root_lin_vel_w.norm(dim=-1) < 0.5)
        un = (~self.aboard(self.decoy)) & self.level() & slow_beam & fin
        self._unload_cnt = torch.where(un, self._unload_cnt + 1,
                                       torch.zeros_like(self._unload_cnt))
        self._unloaded |= self._unload_cnt >= 3
        fi = self.any_seated() & slow_cans & fin
        self._first_cnt = torch.where(fi, self._first_cnt + 1,
                                      torch.zeros_like(self._first_cnt))
        self._first |= self._first_cnt >= 3
        bo = self.matched() & slow_cans & fin
        self._both_cnt = torch.where(bo, self._both_cnt + 1,
                                     torch.zeros_like(self._both_cnt))
        self._both |= self._both_cnt >= 3

    # ----- rubric -------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the shelf is BALANCED AND STOCKED — both red cans seated
        upright one per pocket, the beam level within level_tol_deg, the blue
        can not on the beam, everything settled and finite. All clauses are
        live physical outcomes."""
        return self.matched() & self.level() & (~self.aboard(self.decoy)) \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*unloaded + 0.25*first + 0.30*both (all
        latched; ~0 for the null policy — the decoy pins the beam from step 0
        so `unloaded` cannot latch, and no red can is seated), capped at
        0.70 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        base = (c.w_unload * self._unloaded.float()
                + c.w_first * self._first.float()
                + c.w_both * self._both.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="balance_shelf", robot="null"))
