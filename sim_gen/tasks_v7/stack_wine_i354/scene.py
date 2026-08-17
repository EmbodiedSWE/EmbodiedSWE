"""BottleSpringBayScene — seat a wine bottle into a SPRING-LOADED capture bay
(sim_gen task `stack_wine_i354`, battery-compartment mechanics).

Derived from rlbench/stack_wine, but STRATEGICALLY different: the seed is a pure
support-from-below pick-and-place — grasp THE wine bottle and LAY it on THE rack;
the rack imposes no constraint on the path and the goal contact is "rests on top".
Here the goal state is a bottle CLAMPED HORIZONTALLY inside a sprung fixture: its
neck sits inside a spring-loaded socket CUP that presses the bottle axially so its
base face is pinned against the bay's end wall, captured UNDER an overhanging lip.
The relaxed cup-to-wall gap is SHORTER than the bottle (G0 = L - seat_c), so the
seed's set-down cannot produce the state: the only way in is a spring-compression
cycle — enter the open-top bay PITCHED (the level bottle does not fit), drive the
neck into the cup and PRESS the spring well past the seat point so the base drops
below the lip, LOWER the base to the deck, then RELEASE and let the spring shove
the bottle back under the lip until the base seats on the end wall. A bottle laid
on top of the bay walls (the seed's strategy), left resting in the bay without
compression, seated in the wrong bay, inserted base-first, or merely HELD at full
compression is all rejected by the rubric; only the released, spring-retained
clamp with the seat-band compression counts.

success() (judged live on physical poses; the fixture is at its authored pose):
  bottle horizontal in the TARGET bay (the one marked by the amber beacon), axis
  aligned with the bay (neck toward the cup within `align_deg`), neck tip inside
  the cup against its back plate, the target plunger's spring compression in the
  SEAT band [c_lo, c_hi] (a held press reads ~c_need, far outside), base face at
  the end wall under the lip, bottle at deck lying height, everything settled.
score() = latched stage credit anchored in the demonstrated solution: 0.10 ever
  lifted + 0.20 neck ever in the target cup + 0.20 spring ever pressed past
  `press_c` with the neck in the cup + 0.10 ever fully seated for
  `seat_latch_steps` consecutive substeps, capped at 0.60; exactly 1.0 iff
  success(). Doing nothing scores ~0; the seed strategy scores <= 0.10.

Assets are fully procedural (no external files):
  - rack: ONE kinematic compound body — a slick deck plate carrying TWO identical
    open-top bays (axis along x): low side walls, an end wall at +x with an inward
    overhanging LIP (underside `lip_gap` above a lying bottle's top), open at -x
    where the plunger rides.
  - plungers (one per bay): DYNAMIC gravity-free bodies on a per-env X prismatic
    joint to the rack (pair collision disabled; symmetric limits +/- travel), each
    carrying a square socket CUP (opening passes the neck, refuses the body; a top
    wall blocks vertical neck escape). A post_step spring f = k*(home - x) - c*v
    returns each plunger to its extended home.
  - bottle: ONE dynamic compound body (body cylinder + neck cylinder), root at the
    centre of mass, mass/CoM/inertia authored explicitly (custom spawn funcs apply
    no cfg schemas).
  - beacon: a kinematic amber floor tile teleported in front of the TARGET bay.
The clamping principle is honest by construction (asserted in __post_init__): the
level bottle does not fit the relaxed gap, dropping the base past the lip forces a
transient compression c_need far beyond the seat band, the spring force at the
seat exceeds sliding friction with margin (release genuinely seats), the lip
catches a lifted base within millimetres, and a base-first insertion rests at a
compression far outside the seat band.

Per-episode randomization (verified by READBACK in smoke): target bay (beacon
teleports between the two bays) + bottle spawn xy jitter and free yaw. The
rack/plunger jointed pairs are deliberately FIXED at their authored pose: on this
PhysX stack a per-episode teleport of a jointed pair is unreliable (the joint
frame stays anchored at the authored pose), so the mechanism never moves and only
the free bodies randomize. Heavy imports (isaaclab, pxr) are deferred so
importing this module stays app-free.
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

# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float | None) -> None:
    """A colored axis-aligned box prim; collides iff `contact_offset` is not None."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(seg.GetPrim(), contact_offset)


def _cyl(stage, path: str, radius: float, height: float, center_z: float, color,
         contact_offset: float | None) -> None:
    """A z-axis cylinder prim at local (0, 0, center_z); collides iff offset given."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr("Z")
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(center_z)))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(cyl.GetPrim(), contact_offset)


def _phys_material(stage, path: str, static: float, dynamic: float) -> Any:
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _bind_material(stage, prim_path: str, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(prim_path)).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC two-bay rack at `prim_path`. Root origin = deck centre at
    FLOOR level; every part is an axis-aligned box. Slick material (the spring must
    out-pull deck friction — asserted in the scene cfg)."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(30.0)

    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.friction_static,
                         cfg.friction_dynamic)
    for name, cx, cy, cz, sx, sy, sz in cfg.parts:
        if name.startswith("deck"):
            color = cfg.deck_color
        elif name.startswith("lip"):
            color = cfg.lip_color
        else:
            color = cfg.wall_color
        _box(stage, f"{prim_path}/{name}", (sx, sy, sz), (cx, cy, cz), color,
             cfg.contact_offset)
        _bind_material(stage, f"{prim_path}/{name}", mat)
    return root


def _spawn_plunger(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one DYNAMIC gravity-free plunger (the sprung socket cup) at `prim_path`.
    Root origin = the CENTRE of the cup's back-plate INNER face (the surface the neck
    tip presses); the cup channel opens toward +x. Mass/CoM/inertia authored
    explicitly — custom spawn funcs apply no cfg schemas."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)

    UsdPhysics.RigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(cfg.mass))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in cfg.inertia]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateDisableGravityAttr(True)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.05)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.friction_static,
                         cfg.friction_dynamic)
    s, d, t, bt = cfg.s_half, cfg.cup_d, cfg.wall_t, cfg.back_t
    o = s + t  # outer half width of the channel walls
    parts = (
        ("back", (bt, 2 * o, 2 * o), (-bt / 2, 0.0, 0.0)),
        ("top", (d, 2 * o, t), (d / 2, 0.0, s + t / 2)),
        ("bot", (d, 2 * o, t), (d / 2, 0.0, -(s + t / 2))),
        ("left", (d, t, 2 * s), (d / 2, -(s + t / 2), 0.0)),
        ("right", (d, t, 2 * s), (d / 2, s + t / 2, 0.0)),
    )
    for name, size, center in parts:
        _box(stage, f"{prim_path}/{name}", size, center, cfg.color, cfg.contact_offset)
        _bind_material(stage, f"{prim_path}/{name}", mat)
    return root


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC bottle at `prim_path`: body + neck cylinders along local +z
    (base at -z, neck at +z), root origin at the compound CENTRE OF MASS."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)

    UsdPhysics.RigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(cfg.mass))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in cfg.inertia]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(cfg.lin_damping))
    px.CreateAngularDampingAttr(float(cfg.ang_damping))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.friction_static,
                         cfg.friction_dynamic)
    _cyl(stage, f"{prim_path}/body", cfg.body_r, cfg.body_h, cfg.body_cz,
         cfg.body_color, cfg.contact_offset)
    _cyl(stage, f"{prim_path}/neck", cfg.neck_r, cfg.neck_h, cfg.neck_cz,
         cfg.neck_color, cfg.contact_offset)
    for part in ("body", "neck"):
        _bind_material(stage, f"{prim_path}/{part}", mat)
    return root


def _rack_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rack" not in _SPAWNER_CACHE:

        @configclass
        class RackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            parts: tuple = ()
            deck_color: tuple = (0.55, 0.57, 0.60)
            wall_color: tuple = (0.30, 0.33, 0.40)
            lip_color: tuple = (0.72, 0.45, 0.12)
            friction_static: float = 0.05
            friction_dynamic: float = 0.05
            contact_offset: float = 0.001

        _SPAWNER_CACHE["rack"] = RackSpawnerCfg
    return _SPAWNER_CACHE["rack"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), **kw)


def _plunger_spawner_cfg(**kw: Any) -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "plunger" not in _SPAWNER_CACHE:

        @configclass
        class PlungerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plunger)
            s_half: float = 0.017
            cup_d: float = 0.020
            wall_t: float = 0.008
            back_t: float = 0.010
            mass: float = 0.06
            inertia: tuple = (3e-5, 3e-5, 3e-5)
            color: tuple = (0.75, 0.15, 0.12)
            friction_static: float = 0.2
            friction_dynamic: float = 0.2
            contact_offset: float = 0.001

        _SPAWNER_CACHE["plunger"] = PlungerSpawnerCfg
    return _SPAWNER_CACHE["plunger"](**kw)


def _bottle_spawner_cfg(**kw: Any) -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bottle" not in _SPAWNER_CACHE:

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            body_r: float = 0.030
            body_h: float = 0.190
            neck_r: float = 0.011
            neck_h: float = 0.060
            body_cz: float = 0.0
            neck_cz: float = 0.0
            mass: float = 0.45
            inertia: tuple = (1e-3, 1e-3, 2e-4)
            lin_damping: float = 0.1
            ang_damping: float = 0.3
            body_color: tuple = (0.16, 0.32, 0.14)
            neck_color: tuple = (0.16, 0.32, 0.14)
            friction_static: float = 0.2
            friction_dynamic: float = 0.2
            contact_offset: float = 0.001

        _SPAWNER_CACHE["bottle"] = BottleSpawnerCfg
    return _SPAWNER_CACHE["bottle"](**kw)


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BottleSpringBaySceneCfg(BaseCfg):
    """Config for `BottleSpringBayScene`. The spring-capture honesty conditions are
    asserted in `__post_init__` (see the assert messages)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    align_deg: float = tunable(10.0)  # bottle axis within this of the bay's -x direction
    tip_y_tol: float = tunable(0.010)  # neck tip |y - bay_y| (cup physically bounds at 6 mm)
    base_y_tol: float = tunable(0.015)  # base centre |y - bay_y| (bay physically bounds 10 mm)
    z_tol: float = tunable(0.008)  # bottle CoM |z - lying axis height|
    c_lo: float = tunable(0.004)  # target-plunger SEAT compression band (m); the released
    c_hi: float = tunable(0.016)  # clamp rests at seat_c = 0.010; a held press reads ~0.028
    tip_back_lo: float = tunable(-0.004)  # neck tip x minus cup-back face x: at the back
    tip_back_hi: float = tunable(0.006)
    base_wall_tol: float = tunable(0.006)  # base face within this of the end wall (m)
    settle_lin: float = tunable(0.10)  # judging thresholds — above the GPU phantom band;
    settle_ang: float = tunable(0.60)  # the tight position/compression windows do the work
    plunger_still: float = tunable(0.08)  # target plunger |v| when judging (m/s)
    seat_latch_steps: int = tunable(24)  # consecutive seated substeps to latch stage credit
    lifted_z: float = tunable(0.15)  # bottle CoM ever above this (env z) = lifted credit
    press_c: float = tunable(0.024)  # compression that counts as "pressed past the seat"

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    spawn_x: tuple = tunable((-0.42, -0.32))  # bottle spawn x band (standing, on the floor)
    spawn_y: tuple = tunable((-0.12, 0.12))  # bottle spawn y band
    spawn_yaw_deg: float = tunable(180.0)  # free spawn yaw (bottle is axisymmetric)

    # --- info: bottle -------------------------------------------------------------------------
    body_r: float = info(0.030)  # 60 mm body — fits the 80 mm Franka jaw with margin
    body_h: float = info(0.190)
    neck_r: float = info(0.011)  # 22 mm neck — passes the 34 mm cup opening freely
    neck_h: float = info(0.060)
    mass: float = info(0.45)
    lin_damping: float = info(0.1)
    ang_damping: float = info(0.3)
    bottle_color: tuple = info((0.16, 0.32, 0.14))
    bottle_mu: float = info(0.2)

    # --- info: rack / bays --------------------------------------------------------------------
    deck_t: float = info(0.010)  # deck plate thickness (bay floor at this height)
    bay_dy: float = info(0.110)  # bay centrelines at y = +/- bay_dy
    w_half: float = info(0.040)  # bay inner half width (80 mm channel; 10 mm per side)
    wall_t: float = info(0.010)  # side wall thickness
    wall_h: float = info(0.045)  # side wall height above the deck (open top; top of wall
    # sits ABOVE the lying bottle's axis but BELOW its top surface)
    wall_x0: float = info(-0.085)  # side walls span [wall_x0, x_wall]
    x_wall: float = info(0.140)  # end wall INNER face (fixture-local x)
    wall_tx: float = info(0.012)  # end wall thickness
    end_h: float = info(0.100)  # end wall height above the deck
    lip_d: float = info(0.014)  # lip protrusion inward (-x) from the end wall face
    lip_gap: float = info(0.006)  # lip underside clearance above a lying bottle's top
    lip_t: float = info(0.012)  # lip thickness
    deck_x0: float = info(-0.170)  # deck plate x extent
    deck_margin_y: float = info(0.010)
    rack_mu: float = info(0.05)  # slick deck — the spring must out-pull friction

    # --- info: plunger / spring ---------------------------------------------------------------
    s_half: float = info(0.017)  # cup opening half width (34 mm square)
    cup_d: float = info(0.020)  # cup channel depth (face to back-plate inner face)
    cup_wall_t: float = info(0.008)
    cup_back_t: float = info(0.010)
    plunger_mass: float = info(0.06)
    seat_c: float = info(0.010)  # seated spring compression: G0 = L - seat_c
    travel: float = info(0.040)  # prismatic joint limits +/- travel about home
    spring_k: float = info(120.0)  # N/m; seat force 1.2 N vs ~0.55 N deck friction
    spring_cd: float = info(5.0)  # N*s/m (~critical for the 60 g plunger)
    plunger_color: tuple = info((0.75, 0.15, 0.12))
    plunger_mu: float = info(0.2)

    # --- info: beacon / misc ------------------------------------------------------------------
    beacon_x: float = info(-0.22)  # amber target-bay tile, on the floor
    beacon_size: tuple = info((0.06, 0.06, 0.004))
    beacon_color: tuple = info((0.95, 0.70, 0.10))
    contact_offset: float = info(0.001)
    jaw_max: float = info(0.080)  # Franka parallel-jaw stroke (embodiment honesty)
    payload_max: float = info(3.0)  # Franka payload (embodiment honesty)

    # Derived (filled in __post_init__).
    parts: tuple = field(default=None, init=False)  # rack box segments (fixture-local)
    L: float = field(default=None, init=False)  # bottle overall length
    com_z: float = field(default=None, init=False)  # CoM height above the base face
    tip_off: float = field(default=None, init=False)  # neck tip offset in the root frame (+z)
    base_off: float = field(default=None, init=False)  # base face offset (-z, negative)
    body_c: float = field(default=None, init=False)  # body centre, root frame
    neck_c: float = field(default=None, init=False)  # neck centre, root frame
    inertia: tuple = field(default=None, init=False)  # bottle diagonal inertia about CoM
    axis_h: float = field(default=None, init=False)  # lying bottle axis height (deck + r)
    G0: float = field(default=None, init=False)  # relaxed cup-back -> end-wall gap
    x_cb0: float = field(default=None, init=False)  # cup back inner face at home (local x)
    lip_z_lo: float = field(default=None, init=False)  # lip underside height
    c_need: float = field(default=None, init=False)  # compression to drop the base past the lip
    c_rev: float = field(default=None, init=False)  # rest compression of a base-first insertion

    def __post_init__(self) -> None:
        c = self
        # ---- bottle mass properties (uniform density over the two cylinders) ----
        c.L = c.body_h + c.neck_h
        vb = math.pi * c.body_r**2 * c.body_h
        vn = math.pi * c.neck_r**2 * c.neck_h
        mb, mn = (c.mass * v / (vb + vn) for v in (vb, vn))
        zb = c.body_h / 2
        zn = c.body_h + c.neck_h / 2
        c.com_z = (mb * zb + mn * zn) / c.mass
        c.tip_off = c.L - c.com_z
        c.base_off = -c.com_z
        c.body_c = zb - c.com_z
        c.neck_c = zn - c.com_z

        def i_trans(m: float, r: float, h: float, z: float) -> float:
            return m * (3 * r**2 + h**2) / 12 + m * (z - c.com_z) ** 2

        ixx = i_trans(mb, c.body_r, c.body_h, zb) + i_trans(mn, c.neck_r, c.neck_h, zn)
        izz = (mb * c.body_r**2 + mn * c.neck_r**2) / 2
        c.inertia = (ixx, ixx, izz)

        # ---- fixture geometry ----
        c.axis_h = c.deck_t + c.body_r
        c.G0 = c.L - c.seat_c
        c.x_cb0 = c.x_wall - c.G0
        c.lip_z_lo = c.deck_t + 2 * c.body_r + c.lip_gap
        c.c_need = c.seat_c + c.lip_d + 0.004  # clear the lip with 4 mm x margin
        c.c_rev = c.seat_c + c.cup_d  # base-first: base at the cup FACE, neck at the wall

        deck_x1 = c.x_wall + c.wall_tx
        deck_cx = (c.deck_x0 + deck_x1) / 2
        deck_sy = 2 * (c.bay_dy + c.w_half + c.wall_t + c.deck_margin_y)
        parts: list[tuple] = [
            ("deck", deck_cx, 0.0, c.deck_t / 2, deck_x1 - c.deck_x0, deck_sy, c.deck_t)]
        wall_cx = (c.wall_x0 + c.x_wall) / 2
        wall_sx = c.x_wall - c.wall_x0
        for k, sgn in ((0, -1.0), (1, 1.0)):
            y = sgn * c.bay_dy
            for side, ssg in (("l", -1.0), ("r", 1.0)):
                parts.append((f"bay{k}_wall_{side}", wall_cx, y + ssg * (c.w_half + c.wall_t / 2),
                              c.deck_t + c.wall_h / 2, wall_sx, c.wall_t, c.wall_h))
            parts.append((f"bay{k}_end", c.x_wall + c.wall_tx / 2, y,
                          c.deck_t + c.end_h / 2, c.wall_tx, 2 * (c.w_half + c.wall_t), c.end_h))
            parts.append((f"bay{k}_lip", c.x_wall - c.lip_d / 2, y,
                          c.lip_z_lo + c.lip_t / 2, c.lip_d, 2 * c.w_half, c.lip_t))
        c.parts = tuple(parts)

        # ---- honesty-by-construction asserts ----
        assert c.s_half >= c.neck_r + 0.005, "the neck must enter the cup opening freely"
        assert c.s_half <= c.body_r - 0.010, "the body must NOT enter the cup opening"
        assert c.neck_h >= c.cup_d + 0.020, (
            "the neck must reach the cup back with the shoulder well outside the face")
        assert c.s_half - c.neck_r <= 0.010, (
            "the cup top wall must catch a lifted neck within millimetres (captive)")
        assert c.seat_c >= 0.006, (
            "the relaxed gap must be genuinely shorter than the bottle — a level "
            "set-down (the seed strategy) must be geometrically impossible")
        assert c.c_need <= c.travel - 0.008, (
            "the press-to-insert compression must fit the travel with headroom")
        assert c.c_need >= c.c_hi + 0.008, (
            "a held press must read FAR outside the seat band — release is load-bearing")
        assert 0.003 <= c.lip_gap <= 0.010, (
            "the lip must clear a sliding bottle but catch a lifted base within mm")
        assert c.lip_z_lo + c.lip_t < c.deck_t + c.end_h, "lip sits below the end wall top"
        mu_pair = (c.rack_mu + c.bottle_mu) / 2
        assert c.spring_k * c.seat_c >= 1.8 * mu_pair * c.mass * 9.81, (
            "the spring at seat compression must out-pull deck friction with margin — "
            "the release stage must genuinely drive the bottle under the lip")
        assert c.L * math.cos(math.radians(25.0)) <= c.G0 - 0.008, (
            "a bottle pitched ~25 deg must fit the relaxed gap — insertion is feasible")
        assert c.c_rev >= c.c_hi + 0.010, (
            "a base-first insertion must rest FAR outside the seat band")
        assert c.c_rev <= c.travel - 0.008, (
            "a base-first insertion must settle inside the travel (rejected, not jammed)")
        assert c.deck_t + c.wall_h > c.axis_h + 0.010, (
            "side walls must corral the lying bottle (top of wall above its axis)")
        assert c.deck_t + c.wall_h < c.deck_t + 2 * c.body_r, (
            "side walls must stay below the lying bottle's top (open-top bay)")
        assert (c.deck_t + c.wall_h + c.body_r) - c.axis_h >= 0.030, (
            "a bottle laid ON TOP of the walls must rest far above the seat z band")
        assert 2 * c.w_half >= 2 * c.body_r + 0.012, "fingers must fit beside the body"
        assert 2 * c.w_half <= 2 * c.body_r + 0.030, "the bay must still corral the bottle"
        assert c.axis_h - (c.s_half + c.cup_wall_t) - c.deck_t >= 0.003, (
            "the plunger cup must ride clear of the deck")
        assert c.deck_x0 <= c.x_cb0 - c.cup_back_t - c.travel - 0.005, (
            "the deck must extend under the plunger at full compression")
        assert 2 * c.body_r <= c.jaw_max - 0.015, "the bottle must be trivially graspable"
        assert c.mass <= c.payload_max, "the bottle must be liftable"
        assert c.spawn_x[1] + c.body_r <= c.beacon_x - 0.025, (
            "the spawn band must stay clear of the beacon tile")
        assert c.tip_y_tol >= (c.s_half - c.neck_r) + 0.002, (
            "tip_y_tol must admit every physically cupped neck (honesty by construction)")
        assert c.base_y_tol >= (c.w_half - c.body_r) + 0.004, (
            "base_y_tol must admit every physically bayed base")


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("bottle_spring_bay")
class BottleSpringBayScene(BaseScene):
    cfg: BottleSpringBaySceneCfg

    def __init__(self, cfg: BottleSpringBaySceneCfg | None = None) -> None:
        super().__init__(cfg or BottleSpringBaySceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "rack": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=_rack_spawner_cfg(
                    parts=c.parts, friction_static=c.rack_mu, friction_dynamic=c.rack_mu,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "bottle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle",
                spawn=_bottle_spawner_cfg(
                    body_r=c.body_r, body_h=c.body_h, neck_r=c.neck_r, neck_h=c.neck_h,
                    body_cz=c.body_c, neck_cz=c.neck_c, mass=c.mass, inertia=c.inertia,
                    lin_damping=c.lin_damping, ang_damping=c.ang_damping,
                    body_color=c.bottle_color, neck_color=c.bottle_color,
                    friction_static=c.bottle_mu, friction_dynamic=c.bottle_mu,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=((c.spawn_x[0] + c.spawn_x[1]) / 2, 0.0, c.com_z + 0.002)),
            ),
            "beacon": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beacon",
                spawn=sim_utils.CuboidCfg(
                    size=c.beacon_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.beacon_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.beacon_x, c.bay_dy, c.beacon_size[2] / 2)),
            ),
        }
        for i in range(2):
            y = (-1.0, 1.0)[i] * c.bay_dy
            out[f"plunger_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plunger_" + str(i),
                spawn=_plunger_spawner_cfg(
                    s_half=c.s_half, cup_d=c.cup_d, wall_t=c.cup_wall_t,
                    back_t=c.cup_back_t, mass=c.plunger_mass, color=c.plunger_color,
                    friction_static=c.plunger_mu, friction_dynamic=c.plunger_mu,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.x_cb0, y, c.axis_h)),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # external-wrench servos (solve.py) and the post_step spring are
                # under-applied across TGS iterations without this flag
                "enable_external_forces_every_iteration": True,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.rack: RigidObject = env.iscene["rack"]
        self.bottle: RigidObject = env.iscene["bottle"]
        self.beacon: RigidObject = env.iscene["beacon"]
        self.plungers: list[RigidObject] = [env.iscene[f"plunger_{i}"] for i in range(2)]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.target_bay = torch.zeros(n, dtype=torch.long, device=dev)
        self.lifted_latch = torch.zeros(n, device=dev)
        self.neckin_latch = torch.zeros(n, device=dev)
        self.pressed_latch = torch.zeros(n, device=dev)
        self.seated_latch = torch.zeros(n, device=dev)
        self.seat_ctr = torch.zeros(n, dtype=torch.long, device=dev)
        self.max_seat_ctr = torch.zeros(n, dtype=torch.long, device=dev)  # telemetry
        self._plunger_home = self.env_origins.clone()  # world home positions (n, 2, 3)
        self._plunger_home = torch.stack(
            [self.env_origins + torch.tensor([c.x_cb0, s * c.bay_dy, c.axis_h], device=dev)
             for s in (-1.0, 1.0)], dim=1)
        self._author_joints()
        # authored-mass readback (custom spawners apply no cfg schemas — verify)
        for name, obj, want in (("bottle", self.bottle, c.mass),
                                ("plunger", self.plungers[0], c.plunger_mass)):
            m_back = float(obj.root_physx_view.get_masses().cpu().view(-1)[0])
            if abs(m_back - want) > 1e-3:
                print(f"[bottle_spring_bay] MASS AUTHORING FAILED ({name}): readback "
                      f"{m_back:.4f} != {want:.4f} — verdicts are void", flush=True)

    def _author_joints(self) -> None:
        """Per env, per bay: the plunger's X prismatic slide on the rack — pair
        collision disabled (the joint limit is the mechanical stop), SYMMETRIC limits
        +/- travel about the extended home (the GPU sign-convention hedge)."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            for k, sgn in ((0, -1.0), (1, 1.0)):
                j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/plunger_slide_{k}")
                j.CreateBody0Rel().SetTargets([f"{base}/Rack"])
                j.CreateBody1Rel().SetTargets([f"{base}/Plunger_{k}"])
                j.CreateCollisionEnabledAttr(False)
                j.CreateAxisAttr("X")
                j.CreateLocalPos0Attr(Gf.Vec3f(c.x_cb0, sgn * c.bay_dy, c.axis_h))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLowerLimitAttr(-c.travel)
                j.CreateUpperLimitAttr(c.travel)
                lim = PhysxSchema.PhysxLimitAPI.Apply(j.GetPrim(), "linear")
                if hasattr(lim, "CreateContactDistanceAttr"):  # removed in Isaac Sim 5.1
                    lim.CreateContactDistanceAttr(0.001)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: rack re-pinned at its FIXED authored pose (jointed pairs must
        never teleport on this stack), plungers re-pinned at home along their DOF with
        zero velocity, target bay sampled (beacon teleported in front of it), bottle
        standing on the floor in the spawn band with xy jitter + free yaw, latches
        zeroed, plunger force buffers zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        st = torch.zeros(m, 13, device=dev)
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.rack.write_root_state_to_sim(st, env_ids)
        for k in range(2):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = self._plunger_home[env_ids, k]
            st[:, 3] = 1.0
            self.plungers[k].write_root_state_to_sim(st, env_ids)

        # target bay: burn draws first (the FIRST post-seed draw is near-constant
        # across seeds on this stack), then a fair coin
        _ = torch.rand(m, 3, device=dev)
        self.target_bay[env_ids] = (torch.rand(m, device=dev) < 0.5).long()
        sgn = self.target_bay[env_ids].float() * 2.0 - 1.0
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.beacon_x
        st[:, 1] = sgn * c.bay_dy
        st[:, 2] = c.beacon_size[2] / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.beacon.write_root_state_to_sim(st, env_ids)

        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.spawn_x[0] + torch.rand(m, device=dev) * (c.spawn_x[1] - c.spawn_x[0])
        st[:, 1] = c.spawn_y[0] + torch.rand(m, device=dev) * (c.spawn_y[1] - c.spawn_y[0])
        st[:, 2] = c.com_z + 0.002
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.spawn_yaw_deg) / 2
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.bottle.write_root_state_to_sim(st, env_ids)

        self.lifted_latch[env_ids] = 0.0
        self.neckin_latch[env_ids] = 0.0
        self.pressed_latch[env_ids] = 0.0
        self.seated_latch[env_ids] = 0.0
        self.seat_ctr[env_ids] = 0
        self.max_seat_ctr[env_ids] = 0
        # Zero the plunger force buffers: a stale spring force from the previous episode
        # would kick the gravity-free plungers for one substep before post_step runs.
        n = self.env.num_envs
        for p in self.plungers:
            p.set_external_force_and_torque(
                torch.zeros(n, 1, 3, device=dev), torch.zeros(n, 1, 3, device=dev))

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "rack": self.rack.data.root_state_w[env_ids].clone(),
            "bottle": self.bottle.data.root_state_w[env_ids].clone(),
            "beacon": self.beacon.data.root_state_w[env_ids].clone(),
            "plungers": [p.data.root_state_w[env_ids].clone() for p in self.plungers],
            "target_bay": self.target_bay[env_ids].clone(),
            "latches": {k: getattr(self, k)[env_ids].clone()
                        for k in ("lifted_latch", "neckin_latch", "pressed_latch",
                                  "seated_latch", "seat_ctr")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.rack.write_root_state_to_sim(state["rack"], env_ids)
        self.bottle.write_root_state_to_sim(state["bottle"], env_ids)
        self.beacon.write_root_state_to_sim(state["beacon"], env_ids)
        for p, st in zip(self.plungers, state["plungers"]):
            p.write_root_state_to_sim(st, env_ids)
        self.target_bay[env_ids] = state["target_bay"]
        for k, v in state["latches"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A low metal rack sits on the floor: a slick grey deck plate carrying TWO "
            f"identical open-top bays side by side, each an {2 * c.w_half * 1000:.0f} mm "
            f"wide channel between low blue side walls. At the FAR (+x) end of each bay "
            f"stands an end wall with an orange overhanging LIP jutting "
            f"{c.lip_d * 1000:.0f} mm back over the channel; at the NEAR (-x) end rides "
            f"a red SPRING-LOADED PLUNGER carrying a square socket CUP "
            f"({2 * c.s_half * 1000:.0f} mm opening, {c.cup_d * 1000:.0f} mm deep) that "
            f"faces the end wall. The spring holds each cup extended toward its end "
            f"wall, and the relaxed cup-to-wall gap is {c.G0 * 1000:.0f} mm — SHORTER "
            f"than the green wine bottle ({c.L * 1000:.0f} mm long, "
            f"{2 * c.body_r * 1000:.0f} mm body, {2 * c.neck_r * 1000:.0f} mm neck) "
            f"standing on the floor in front of the rack. A small amber beacon tile on "
            f"the floor marks the TARGET bay.\n"
            f"Goal: clamp the bottle horizontally into the TARGET bay like a battery "
            f"into a compartment. Because the bottle is longer than the relaxed gap it "
            f"cannot be laid straight in: lower it into the bay PITCHED base-up with "
            f"the neck toward the cup, drive the neck into the cup opening and PRESS "
            f"the plunger back against its spring (about "
            f"{c.c_need * 1000:.0f} mm of compression) until the raised base can drop "
            f"past the orange lip, LOWER the base onto the deck in front of the lip, "
            f"then RELEASE: the spring shoves the bottle back so its base slides under "
            f"the lip and seats against the end wall, leaving the bottle clamped — "
            f"neck sprung into the cup, base pinned under the lip. A bottle laid on "
            f"top of the bay walls, resting anywhere without spring compression, "
            f"seated in the WRONG bay, inserted base-first, or merely held pressed at "
            f"full compression does not count: success is the released, spring-"
            f"retained clamp in the beacon's bay, at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Clamp the green wine bottle horizontally into the bay marked by the amber "
            "beacon: lower it in pitched base-up, press its neck into the red sprung "
            "cup until the base can drop past the orange lip, lower the base to the "
            "deck, and release so the spring seats the base under the lip against the "
            "end wall. Laying it on top, using the wrong bay, inserting it base-first, "
            "or just holding it pressed does not count."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _bottle_axis(self) -> torch.Tensor:
        """(N,3) world direction of the bottle's base->neck (+z local) axis."""
        from isaaclab.utils.math import quat_apply

        q = self.bottle.data.root_quat_w
        ez = torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], 3)
        return quat_apply(q, ez)

    def _keypoints(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(neck tip, base face centre, axis) in ENV-LOCAL coordinates (the fixture is
        at the env origin with identity yaw, so env frame == fixture frame)."""
        c = self.cfg
        p = self.bottle.data.root_pos_w - self.env_origins
        a = self._bottle_axis()
        return p + a * c.tip_off, p + a * c.base_off, a

    def compression(self) -> torch.Tensor:
        """(N,2) spring compression per plunger: home_x - x (positive = pressed)."""
        x = torch.stack([p.data.root_pos_w[:, 0] for p in self.plungers], dim=1)
        return self._plunger_home[:, :, 0] - x

    # ----- predicates -------------------------------------------------------------------------
    def seated_now(self) -> torch.Tensor:
        """(N,) bool: the full geometric clamp clause in the TARGET bay (excluding the
        settle gates): bottle horizontal, neck toward the cup, neck tip at the live cup
        back, target spring compression in the SEAT band, base face at the end wall,
        bottle at lying height and corralled in y."""
        c = self.cfg
        tip, base, a = self._keypoints()
        sgn = self.target_bay.float() * 2.0 - 1.0
        y_t = sgn * c.bay_dy
        comp = self.compression().gather(1, self.target_bay.view(-1, 1)).squeeze(1)
        back_x = torch.stack([p.data.root_pos_w[:, 0] for p in self.plungers],
                             dim=1).gather(1, self.target_bay.view(-1, 1)).squeeze(1) \
            - self.env_origins[:, 0]
        p_z = self.bottle.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return ((a[:, 0] <= -math.cos(math.radians(c.align_deg)))
                & ((tip[:, 1] - y_t).abs() <= c.tip_y_tol)
                & ((base[:, 1] - y_t).abs() <= c.base_y_tol)
                & ((p_z - c.axis_h).abs() <= c.z_tol)
                & ((tip[:, 0] - back_x) >= c.tip_back_lo)
                & ((tip[:, 0] - back_x) <= c.tip_back_hi)
                & (comp >= c.c_lo) & (comp <= c.c_hi)
                & (base[:, 0] >= c.x_wall - c.base_wall_tol))

    def settled(self) -> torch.Tensor:
        """(N,) bool: bottle AND target plunger at rest."""
        c = self.cfg
        lin = self.bottle.data.root_lin_vel_w.norm(dim=-1)
        ang = self.bottle.data.root_ang_vel_w.norm(dim=-1)
        pv = torch.stack([p.data.root_lin_vel_w.norm(dim=-1) for p in self.plungers],
                         dim=1).gather(1, self.target_bay.view(-1, 1)).squeeze(1)
        return (lin < c.settle_lin) & (ang < c.settle_ang) & (pv < c.plunger_still)

    # ----- mechanics + progress latches (every physics substep) -------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Apply the plunger return springs and latch the stage credits."""
        c = self.cfg
        dev = self.env.device
        n = self.env.num_envs

        comp = self.compression()  # (N,2)
        for k, p in enumerate(self.plungers):
            v_x = p.data.root_lin_vel_w[:, 0]
            f_x = c.spring_k * comp[:, k] - c.spring_cd * v_x
            f = torch.zeros(n, 1, 3, device=dev)
            f[:, 0, 0] = f_x
            p.set_external_force_and_torque(f, torch.zeros(n, 1, 3, device=dev))

        tip, _base, _a = self._keypoints()
        p_z = self.bottle.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        lin = self.bottle.data.root_lin_vel_w.norm(dim=-1)
        self.lifted_latch = torch.maximum(
            self.lifted_latch, ((p_z > c.lifted_z) & (lin < 1.0)).float())

        sgn = self.target_bay.float() * 2.0 - 1.0
        back_x = torch.stack([p.data.root_pos_w[:, 0] for p in self.plungers],
                             dim=1).gather(1, self.target_bay.view(-1, 1)).squeeze(1) \
            - self.env_origins[:, 0]
        neck_in = (((tip[:, 0] - back_x) >= c.tip_back_lo)
                   & ((tip[:, 0] - back_x) <= c.cup_d)
                   & ((tip[:, 1] - sgn * c.bay_dy).abs() <= c.s_half)
                   & ((tip[:, 2] - c.axis_h).abs() <= c.s_half))
        self.neckin_latch = torch.maximum(self.neckin_latch, neck_in.float())
        t_comp = comp.gather(1, self.target_bay.view(-1, 1)).squeeze(1)
        self.pressed_latch = torch.maximum(
            self.pressed_latch, (neck_in & (t_comp >= c.press_c)).float())

        ok = self.seated_now() & (lin < c.settle_lin)
        self.seat_ctr = torch.where(ok, self.seat_ctr + 1, torch.zeros_like(self.seat_ctr))
        self.max_seat_ctr = torch.maximum(self.max_seat_ctr, self.seat_ctr)
        self.seated_latch = torch.maximum(
            self.seated_latch, (self.seat_ctr >= c.seat_latch_steps).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the bottle is clamped in the TARGET bay (full geometric clause)
        and everything is settled — judged live; pull the bottle out or hold it pressed
        and success is gone."""
        return self.seated_now() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: latched stage credit — 0.10 lifted + 0.20 neck ever in
        the target cup + 0.20 ever pressed past `press_c` with the neck cupped + 0.10
        ever seated for `seat_latch_steps` substeps, capped at 0.60; exactly 1.0 iff
        success(). Doing nothing scores ~0; the seed strategy scores <= 0.10."""
        base = (0.10 * self.lifted_latch + 0.20 * self.neckin_latch
                + 0.20 * self.pressed_latch + 0.10 * self.seated_latch).clamp(0.0, 0.60)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="bottle_spring_bay", robot="null", env_spacing=3))
