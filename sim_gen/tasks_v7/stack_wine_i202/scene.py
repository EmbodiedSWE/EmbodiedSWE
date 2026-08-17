"""BottleHangRailScene — hang three collared carafes BELOW an overhead keyhole rail
(sim_gen task `stack_wine_i202`).

Derived from rlbench/stack_wine, but STRATEGICALLY different: the seed is a pure
support-from-below pick-and-place — grasp THE wine bottle and LAY it on THE rack; the
final contact is "object rests on top of the fixture" and the whole plan is
approach-grasp-carry-set. Here the goal contact is INVERTED and gated by a mechanism:
each carafe must end up HANGING under the rail, held from ABOVE by its collar, and the
only way in is a three-stage keyhole maneuver — raise the carafe under the rail so its
collar passes UP through a wide port cut in the plate, TRANSLATE it rearward so the
neck runs down a narrow slot the collar cannot pass, then lower/release so the collar
lands on the plate top and the carafe swings free below. The seed's end state
transplanted here (bottle resting ON TOP of the rail) is explicitly rejected by the
rubric; a carafe released while still at the wide port simply falls back through
(physics rejects it). The plan is insert-through-aperture -> captive slide -> hanging
release, per object, versus the seed's place-on-top.

success() (all live, judged on physical poses in the RAIL's frame):
  every carafe hangs in its own keyhole — collar centre within `x_tol` of the slot
  centreline, at least `hang_y_min` down the slot (far past the port, where the collar
  physically cannot lift out), collar resting in the seat band on the plate top, carafe
  plumb (axis within `hang_tilt_deg` of vertical) with its body fully below the plate —
  one carafe per keyhole (all three keyholes used), everything settled.
score() = latched stage credit anchored in the demonstrated solution: per carafe
  0.05 ever LIFTED to rail height + 0.10 collar ever INSIDE a keyhole above the plate
  + 0.10 ever HUNG (full geometric hang, slow, for `hang_latch_steps` consecutive
  substeps), capped at 0.75; exactly 1.0 iff success(). Doing nothing scores ~0.

Assets are fully procedural (no external files):
  - rail: ONE kinematic compound body — a walnut plate 12 mm thick, 300 mm above the
    floor on two corner posts, with three keyhole cutouts built from axis-aligned box
    segments: a 46x46 mm square PORT near the FRONT edge continuing into a 24 mm wide
    slot (CHANNEL) running to the back. All plate segments share one flat top face, so
    a collar slides across the port/channel boundary without snagging.
  - carafes: three identical DYNAMIC compound bodies (body cylinder 56 mm dia x
    100 mm, neck 14 mm dia x 50 mm, tan collar disc 34 mm dia x 8 mm), root origin AT
    the centre of mass (applied forces induce no spurious torque), mass + CoM +
    diagonal inertia authored explicitly in the spawner (custom spawn funcs apply no
    cfg schemas).
The hanging principle is honest by construction (asserted in __post_init__): the
collar fits through the port but not the channel, the neck slides the channel freely,
the body cannot pass the port, the hang band starts far beyond the lift-out bound, and
the band is shorter than a collar diameter so a keyhole physically holds ONE carafe.

Per-episode randomization (verified by READBACK in smoke): rail pose (xy jitter +
yaw) and carafe spawn slots (permuted + jittered + free yaw). Heavy imports
(isaaclab, pxr) are deferred so importing this module stays app-free.
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


def _spawn_rail(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC keyhole rail at `prim_path`. Root origin = plate CENTRE
    (mid-thickness); the two posts hang down to the floor. All segments are
    axis-aligned boxes sharing one flat top face, so the collar slides across the
    port/channel boundary without snagging."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(20.0)

    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.friction_static,
                         cfg.friction_dynamic)
    for k, (name, cx, cy, cz, sx, sy, sz) in enumerate(cfg.parts):
        color = cfg.post_color if name.startswith("post") else cfg.plate_color
        _box(stage, f"{prim_path}/{name}", (sx, sy, sz), (cx, cy, cz), color,
             cfg.contact_offset)
        _bind_material(stage, f"{prim_path}/{name}", mat)
    return root


def _spawn_carafe(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one DYNAMIC carafe at `prim_path`. Root origin = the compound CENTRE OF
    MASS (so applied forces at the origin induce no spurious torque); mass, CoM and
    diagonal inertia are authored EXPLICITLY (custom spawn funcs apply no cfg
    schemas — density-derived masses and origin-CoM are the classic silent traps)."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)

    UsdPhysics.RigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(cfg.mass))
    from pxr import Gf

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
         cfg.body_color, cfg.contact_offset)
    _cyl(stage, f"{prim_path}/collar", cfg.collar_r, cfg.collar_h, cfg.collar_cz,
         cfg.collar_color, cfg.contact_offset)
    for part in ("body", "neck", "collar"):
        _bind_material(stage, f"{prim_path}/{part}", mat)
    return root


def _rail_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rail" not in _SPAWNER_CACHE:

        @configclass
        class RailSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rail)
            parts: tuple = ()
            plate_color: tuple = (0.45, 0.30, 0.18)
            post_color: tuple = (0.28, 0.19, 0.11)
            friction_static: float = 0.4
            friction_dynamic: float = 0.35
            contact_offset: float = 0.001

        _SPAWNER_CACHE["rail"] = RailSpawnerCfg
    return _SPAWNER_CACHE["rail"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), **kw)


def _carafe_spawner_cfg(**kw: Any) -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "carafe" not in _SPAWNER_CACHE:

        @configclass
        class CarafeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_carafe)
            body_r: float = 0.028
            body_h: float = 0.10
            neck_r: float = 0.007
            neck_h: float = 0.05
            collar_r: float = 0.017
            collar_h: float = 0.008
            body_cz: float = 0.0
            neck_cz: float = 0.0
            collar_cz: float = 0.0
            mass: float = 0.30
            inertia: tuple = (1e-4, 1e-4, 1e-4)
            lin_damping: float = 0.1
            ang_damping: float = 0.3
            body_color: tuple = (0.10, 0.30, 0.14)
            collar_color: tuple = (0.85, 0.72, 0.45)
            friction_static: float = 0.4
            friction_dynamic: float = 0.35
            contact_offset: float = 0.001

        _SPAWNER_CACHE["carafe"] = CarafeSpawnerCfg
    return _SPAWNER_CACHE["carafe"](**kw)


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BottleHangRailSceneCfg(BaseCfg):
    """Config for `BottleHangRailScene`. The keyhole-hanging honesty conditions are
    asserted in `__post_init__` (see the assert messages)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    x_tol: float = tunable(0.008)  # collar centre across-the-slot tolerance (m); the
    # channel physically bounds a hanging neck at chan_half - neck_r = 5 mm < x_tol,
    # so any carafe physically hanging in a slot passes this clause by construction
    hang_y_min: float = tunable(0.015)  # collar centre at least this far down the slot
    # (from the port/channel boundary datum, see __post_init__) — far beyond the
    # lift-out bound, where the collar CANNOT pass back up through the port
    hang_y_max: float = tunable(0.034)  # ...and at most this far (channel end bound)
    collar_z_lo: float = tunable(0.004)  # collar-centre seat band, rail-local z (m):
    collar_z_hi: float = tunable(0.018)  # resting = plate_t/2 + collar_h/2 = 10 mm
    hang_tilt_deg: float = tunable(15.0)  # carafe axis within this of world up (plumb)
    body_below_z: float = tunable(-0.060)  # body centre below this (rail-local z):
    # the carafe truly hangs UNDER the plate (rejects the inverted-on-top spoof)
    settle_lin: float = tunable(0.10)  # max |lin vel| when judging (m/s) — sits above
    # the GPU phantom-velocity artifact; the tight hang position window does the work
    settle_ang: float = tunable(0.60)  # max |ang vel| when judging (rad/s)
    slow_lin: float = tunable(0.15)  # "slow" gate for the hang latch
    hang_latch_steps: int = tunable(24)  # consecutive slow-hung substeps to latch credit
    lifted_z: float = tunable(0.20)  # collar ever above this (env z) = lifted credit

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    rail_xy_jitter: tuple = tunable((0.030, 0.020))  # +/- rail pose jitter (m)
    rail_yaw_deg: float = tunable(8.0)  # +/- rail yaw jitter
    bottle_slots: tuple = tunable((-0.14, 0.0, 0.14))  # spawn x slots (permuted)
    bottle_row_y: float = tunable(-0.30)  # spawn row y (in front of the rail)
    bottle_xy_jitter: tuple = tunable((0.020, 0.030))  # +/- spawn jitter per carafe
    bottle_yaw_deg: float = tunable(180.0)  # free spawn yaw (carafes are symmetric)

    # --- info: rail structure -----------------------------------------------------------------
    shelf_top_z: float = info(0.300)  # plate TOP face height above the floor
    plate_t: float = info(0.012)  # plate thickness
    cells_x: tuple = info((-0.11, 0.0, 0.11))  # keyhole centrelines (rail-local x)
    cell_half: float = info(0.055)  # half width of one keyhole cell
    depth_half: float = info(0.065)  # plate half depth (rail-local y)
    port_half: float = info(0.023)  # square port half width (46 x 46 mm opening)
    port_cy: float = info(-0.020)  # port centre (rail-local y; front = -y)
    chan_half: float = info(0.012)  # channel (slot) half width — 24 mm slot
    chan_y1: float = info(0.038)  # channel rear end (rail-local y)
    cap_hw: float = info(0.020)  # end-cap half width beyond the outer cells
    post_w: float = info(0.030)  # post cross-section
    post_x: float = info(0.190)  # post centreline (rail-local x)
    post_y: float = info(0.030)  # post centreline (rail-local y)
    plate_color: tuple = info((0.45, 0.30, 0.18))
    post_color: tuple = info((0.28, 0.19, 0.11))

    # --- info: carafe structure ---------------------------------------------------------------
    body_r: float = info(0.028)  # 56 mm body — fits the 80 mm Franka jaw with margin
    body_h: float = info(0.100)
    neck_r: float = info(0.007)  # 14 mm neck — slides the 24 mm channel freely
    neck_h: float = info(0.050)
    collar_r: float = info(0.017)  # 34 mm collar — passes the port, NOT the channel
    collar_h: float = info(0.008)
    mass: float = info(0.30)
    lin_damping: float = info(0.1)
    ang_damping: float = info(0.3)  # a hanging carafe is a pendulum — damp the swing
    body_color: tuple = info((0.10, 0.30, 0.14))
    collar_color: tuple = info((0.85, 0.72, 0.45))
    friction_static: float = info(0.4)
    friction_dynamic: float = info(0.35)
    contact_offset: float = info(0.001)
    jaw_max: float = info(0.080)  # Franka parallel-jaw stroke (embodiment honesty)
    payload_max: float = info(3.0)  # Franka payload (embodiment honesty)

    # Derived (filled in __post_init__).
    parts: tuple = field(default=None, init=False)  # rail box segments (rail-local)
    shelf_root_z: float = field(default=None, init=False)  # rail root height (plate centre)
    chan_y0: float = field(default=None, init=False)  # port/channel boundary (datum)
    total_h: float = field(default=None, init=False)  # carafe overall height
    com_z: float = field(default=None, init=False)  # CoM height above the carafe bottom
    bot_off: float = field(default=None, init=False)  # bottom z in the root (CoM) frame
    body_c: float = field(default=None, init=False)  # body centre, root frame
    collar_c: float = field(default=None, init=False)  # collar centre, root frame
    inertia: tuple = field(default=None, init=False)  # diagonal inertia about the CoM
    seat_z_loc: float = field(default=None, init=False)  # collar centre when seated (rail z)

    def __post_init__(self) -> None:
        c = self
        # ---- carafe mass properties (uniform density over the three cylinders) ----
        vb = math.pi * c.body_r**2 * c.body_h
        vn = math.pi * c.neck_r**2 * c.neck_h
        vc = math.pi * c.collar_r**2 * c.collar_h
        vt = vb + vn + vc
        mb, mn, mc = (c.mass * v / vt for v in (vb, vn, vc))
        zb = c.body_h / 2
        zn = c.body_h + c.neck_h / 2
        zc = c.body_h + c.neck_h + c.collar_h / 2
        c.total_h = c.body_h + c.neck_h + c.collar_h
        c.com_z = (mb * zb + mn * zn + mc * zc) / c.mass
        c.bot_off = -c.com_z
        c.body_c = zb - c.com_z
        c.collar_c = zc - c.com_z

        def i_trans(m: float, r: float, h: float, z: float) -> float:
            return m * (3 * r**2 + h**2) / 12 + m * (z - c.com_z) ** 2

        ixx = (i_trans(mb, c.body_r, c.body_h, zb) + i_trans(mn, c.neck_r, c.neck_h, zn)
               + i_trans(mc, c.collar_r, c.collar_h, zc))
        izz = (mb * c.body_r**2 + mn * c.neck_r**2 + mc * c.collar_r**2) / 2
        c.inertia = (ixx, ixx, izz)

        # ---- rail box partition (rail-local; root at plate centre) ----
        c.shelf_root_z = c.shelf_top_z - c.plate_t / 2
        c.chan_y0 = c.port_cy + c.port_half  # port/channel boundary — the slot datum
        parts: list[tuple] = []
        t, dh = c.plate_t, c.depth_half
        p0, p1 = c.port_cy - c.port_half, c.port_cy + c.port_half  # port y extent
        for k, xk in enumerate(c.cells_x):
            ch = c.cell_half
            segs = [
                ("front", 0.0, (p0 - dh) / 2, dh + p0, 2 * ch),  # y in [-dh, p0]
                ("back", 0.0, (c.chan_y1 + dh) / 2, dh - c.chan_y1, 2 * ch),
                ("pl", -(ch + c.port_half) / 2, c.port_cy, p1 - p0, ch - c.port_half),
                ("pr", (ch + c.port_half) / 2, c.port_cy, p1 - p0, ch - c.port_half),
                ("cl", -(ch + c.chan_half) / 2, (p1 + c.chan_y1) / 2,
                 c.chan_y1 - p1, ch - c.chan_half),
                ("cr", (ch + c.chan_half) / 2, (p1 + c.chan_y1) / 2,
                 c.chan_y1 - p1, ch - c.chan_half),
            ]
            for name, dx, cy, sy, sx in segs:
                parts.append((f"cell{k}_{name}", xk + dx, cy, 0.0, sx, sy, t))
        span = c.cells_x[-1] + c.cell_half
        for side, sgn in (("l", -1.0), ("r", 1.0)):
            parts.append((f"cap_{side}", sgn * (span + c.cap_hw), 0.0, 0.0,
                          2 * c.cap_hw, 2 * dh, t))
            post_h = c.shelf_root_z - c.plate_t / 2  # floor -> plate bottom
            parts.append((f"post_{side}", sgn * c.post_x, c.post_y,
                          -c.plate_t / 2 - post_h / 2, c.post_w, c.post_w, post_h))
        c.parts = tuple(parts)
        c.seat_z_loc = c.plate_t / 2 + c.collar_h / 2

        # ---- honesty-by-construction asserts ----
        assert c.collar_r >= c.chan_half + 0.004, (
            "the collar must NOT pass the channel — hanging retention is the task")
        assert c.collar_r <= c.port_half - 0.005, (
            "the collar must pass the port with clearance — insertion must be feasible")
        assert c.neck_r <= c.chan_half - 0.004, (
            "the neck must slide the channel freely")
        assert c.body_r >= c.port_half + 0.004, (
            "the body must NOT fit through the port — no drop-through shortcut")
        # collar can lift out only while its disc fits inside the port void:
        # collar centre y <= chan_y0 - collar_r. The hang band must start far beyond.
        assert c.hang_y_min - (0.0 - c.collar_r) >= 0.025, (
            "the hang band must start >= 25 mm beyond the collar lift-out bound")
        assert c.hang_y_max >= c.hang_y_min + 0.010, "hang band must be reachable"
        reach = (c.chan_y1 - c.chan_y0) - c.neck_r  # deepest achievable slot travel
        assert reach > c.hang_y_min + 0.008, (
            "the channel must extend usefully past the hang threshold")
        assert (min(c.hang_y_max, reach) - c.hang_y_min) < 2 * c.collar_r - 0.004, (
            "the hang band must be shorter than a collar diameter — ONE carafe per slot")
        assert c.neck_h >= c.plate_t + 0.020, (
            "the neck must be long enough that a hanging body clears the plate")
        assert c.total_h < c.shelf_top_z - c.plate_t - 0.060, (
            "a carafe standing under the rail must fit with insertion headroom")
        assert c.shelf_top_z - (c.total_h - c.collar_h) > 0.05, (
            "a hanging carafe must swing clear of the floor")
        assert c.collar_z_lo < c.seat_z_loc < c.collar_z_hi, "seat band brackets the seat"
        assert c.x_tol >= (c.chan_half - c.neck_r) + 0.002, (
            "x_tol must admit every physically hanging carafe (honesty by construction)")
        assert 2 * c.body_r <= c.jaw_max - 0.015, "carafes must be trivially graspable"
        assert c.mass <= c.payload_max, "carafes must be liftable"
        assert (c.cells_x[1] - c.cells_x[0]) >= 2 * c.cell_half - 1e-9, "cells must tile"
        # carafe spawn row stays clear of the rail footprint under worst-case jitter
        rail_front = (-c.depth_half) * 1.0 - c.rail_xy_jitter[1] - (
            (c.cells_x[-1] + c.cell_half + 2 * c.cap_hw) * math.sin(math.radians(c.rail_yaw_deg)))
        assert (rail_front - (c.bottle_row_y + c.bottle_xy_jitter[1])) > 0.08, (
            "carafe spawns must stay clear of the rail under worst-case jitter")


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("bottle_hang_rail")
class BottleHangRailScene(BaseScene):
    cfg: BottleHangRailSceneCfg

    def __init__(self, cfg: BottleHangRailSceneCfg | None = None) -> None:
        super().__init__(cfg or BottleHangRailSceneCfg())

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
            "rail": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rail",
                spawn=_rail_spawner_cfg(
                    parts=c.parts, plate_color=c.plate_color, post_color=c.post_color,
                    friction_static=c.friction_static, friction_dynamic=c.friction_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.shelf_root_z)),
            ),
        }
        for i in range(3):
            out[f"carafe_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Carafe_" + str(i),
                spawn=_carafe_spawner_cfg(
                    body_r=c.body_r, body_h=c.body_h, neck_r=c.neck_r, neck_h=c.neck_h,
                    collar_r=c.collar_r, collar_h=c.collar_h,
                    body_cz=c.body_c, neck_cz=c.body_h + c.neck_h / 2 - c.com_z,
                    collar_cz=c.collar_c, mass=c.mass, inertia=c.inertia,
                    lin_damping=c.lin_damping, ang_damping=c.ang_damping,
                    body_color=c.body_color, collar_color=c.collar_color,
                    friction_static=c.friction_static, friction_dynamic=c.friction_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bottle_slots[i], c.bottle_row_y, c.com_z + 0.002)),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # external-wrench servos (solve.py) are under-applied across TGS
                # iterations without this flag — IsaacLab warns about it at startup
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
        self.rail: RigidObject = env.iscene["rail"]
        self.bottles: list[RigidObject] = [env.iscene[f"carafe_{i}"] for i in range(3)]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.lifted_latch = torch.zeros(n, 3, device=dev)
        self.ported_latch = torch.zeros(n, 3, device=dev)
        self.hung_latch = torch.zeros(n, 3, device=dev)
        self.hang_ctr = torch.zeros(n, 3, dtype=torch.long, device=dev)
        self.max_hang_ctr = torch.zeros(n, 3, dtype=torch.long, device=dev)  # telemetry
        # authored-mass readback (custom spawners apply no cfg schemas — verify)
        m_back = float(self.bottles[0].root_physx_view.get_masses().cpu().view(-1)[0])
        if abs(m_back - self.cfg.mass) > 1e-3:
            print(f"[bottle_hang_rail] MASS AUTHORING FAILED: readback {m_back:.4f} != "
                  f"{self.cfg.mass:.4f} — verdicts are void", flush=True)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter the rail pose (xy + yaw), scatter the carafes standing
        upright on permuted slots (+ jitter + free yaw), zero all latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- rail: one kinematic root carries the whole keyhole assembly ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = (torch.rand(m, device=dev) * 2 - 1) * c.rail_xy_jitter[0]
        st[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.rail_xy_jitter[1]
        st[:, 2] = c.shelf_root_z
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rail_yaw_deg) / 2
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.rail.write_root_state_to_sim(st, env_ids)

        # --- carafes: permuted slots + jitter + free yaw, standing upright ---
        slots = torch.tensor(c.bottle_slots, device=dev)
        perm = torch.rand(m, 3, device=dev).argsort(dim=1)
        yaw_amp = math.radians(c.bottle_yaw_deg)
        for i, b in enumerate(self.bottles):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = slots[perm[:, i]]
            st[:, 0] += (torch.rand(m, device=dev) * 2 - 1) * c.bottle_xy_jitter[0]
            st[:, 1] = c.bottle_row_y + (torch.rand(m, device=dev) * 2 - 1) * c.bottle_xy_jitter[1]
            st[:, 2] = c.com_z + 0.002
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            b.write_root_state_to_sim(st, env_ids)

        # --- latches ---
        self.lifted_latch[env_ids] = 0.0
        self.ported_latch[env_ids] = 0.0
        self.hung_latch[env_ids] = 0.0
        self.hang_ctr[env_ids] = 0
        self.max_hang_ctr[env_ids] = 0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "rail": self.rail.data.root_state_w[env_ids].clone(),
            "bottles": [b.data.root_state_w[env_ids].clone() for b in self.bottles],
            "lifted_latch": self.lifted_latch[env_ids].clone(),
            "ported_latch": self.ported_latch[env_ids].clone(),
            "hung_latch": self.hung_latch[env_ids].clone(),
            "hang_ctr": self.hang_ctr[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.rail.write_root_state_to_sim(state["rail"], env_ids)
        for b, st in zip(self.bottles, state["bottles"]):
            b.write_root_state_to_sim(st, env_ids)
        self.lifted_latch[env_ids] = state["lifted_latch"]
        self.ported_latch[env_ids] = state["ported_latch"]
        self.hung_latch[env_ids] = state["hung_latch"]
        self.hang_ctr[env_ids] = state["hang_ctr"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"An overhead hanging rail stands before you: a walnut-brown plate "
            f"({2 * (c.cells_x[-1] + c.cell_half + 2 * c.cap_hw) * 1000:.0f} x "
            f"{2 * c.depth_half * 1000:.0f} mm, {c.plate_t * 1000:.0f} mm thick) held "
            f"horizontally {c.shelf_top_z * 1000:.0f} mm above the floor by two dark "
            f"corner posts at its back corners. Cut through the plate are three "
            f"identical KEYHOLES, evenly spaced along the rail: each is a wide square "
            f"PORT ({2 * c.port_half * 1000:.0f} x {2 * c.port_half * 1000:.0f} mm "
            f"opening) near the rail's FRONT edge (the side facing you) that continues "
            f"into a narrow SLOT ({2 * c.chan_half * 1000:.0f} mm wide, "
            f"{(c.chan_y1 - c.chan_y0) * 1000:.0f} mm long) running toward the BACK "
            f"edge.\n"
            f"On the floor in front of the rail stand three identical green carafes "
            f"({2 * c.body_r * 1000:.0f} mm diameter body, {c.total_h * 1000:.0f} mm "
            f"tall): each has a narrow neck ({2 * c.neck_r * 1000:.0f} mm) topped by a "
            f"wider tan COLLAR disc ({2 * c.collar_r * 1000:.0f} mm diameter). The "
            f"collar fits up through a port but CANNOT pass the narrow slot — that is "
            f"the hanging principle.\n"
            f"Goal: hang all three carafes BELOW the rail, one per keyhole. For each "
            f"carafe: hold it upright under a free keyhole's port, raise it so the "
            f"collar passes up through the port, slide it toward the BACK so the neck "
            f"travels down the narrow slot (get it at least ~{c.hang_y_min * 1000:.0f} mm "
            f"past the slot's start — pushing gently until it stops at the slot's far "
            f"end is safest), then lower and release it so the collar rests on top of "
            f"the plate and the carafe hangs freely below the rail. A carafe standing "
            f"on the floor, resting on TOP of the rail, or released while still at the "
            f"wide port (it falls straight back through) does not count. Any carafe "
            f"may go to any keyhole, exactly one carafe per keyhole, in any order. "
            f"Finish with all three hanging plumb and at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Hang each of the three green carafes below the overhead rail: raise the "
            "carafe under a free keyhole so its tan collar passes up through the wide "
            "port, slide its neck to the far end of the narrow slot, and release it to "
            "hang from the collar. One carafe per keyhole; a carafe left standing on "
            "the floor, resting on top of the rail, or released at the wide port does "
            "not count."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _bottle_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,3,3) root positions (world) and (N,3,4) root quats for the carafes."""
        pos = torch.stack([b.data.root_pos_w for b in self.bottles], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.bottles], dim=1)
        return pos, quat

    def _bottle_axis(self) -> torch.Tensor:
        """(N,3,3) world direction of each carafe's body +z (up) axis."""
        from isaaclab.utils.math import quat_apply

        pos, quat = self._bottle_tensors()
        n = quat.shape[0]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(n * 3, 3)
        return quat_apply(quat.reshape(n * 3, 4), ez).reshape(n, 3, 3)

    def _points_rail_local(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Collar centres and body centres in the RAIL's frame, shapes (N,3,3). The
        rubric lives in this frame so the jittered/yawed rail judges identically."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        pos, quat = self._bottle_tensors()
        n = pos.shape[0]
        axis = self._bottle_axis()
        collar_w = pos + axis * c.collar_c
        body_w = pos + axis * c.body_c
        rq = self.rail.data.root_quat_w[:, None, :].expand(n, 3, 4).reshape(n * 3, 4)
        rp = self.rail.data.root_pos_w[:, None, :]
        collar_l = quat_apply_inverse(rq, (collar_w - rp).reshape(n * 3, 3)).reshape(n, 3, 3)
        body_l = quat_apply_inverse(rq, (body_w - rp).reshape(n * 3, 3)).reshape(n, 3, 3)
        return collar_l, body_l

    # ----- predicates -------------------------------------------------------------------------
    def hang_cells(self) -> tuple[torch.Tensor, torch.Tensor]:
        """((N,3) bool hung, (N,3) long nearest-cell index). hung = the full geometric
        hang clause in the rail frame: collar on the slot centreline, deep down the
        slot (far past the lift-out bound), collar in the seat band on the plate top,
        carafe plumb with its body below the plate."""
        c = self.cfg
        collar_l, body_l = self._points_rail_local()
        cells = torch.tensor(c.cells_x, device=collar_l.device)
        dx = collar_l[..., 0:1] - cells.view(1, 1, 3)  # (N,3,3)
        cell = dx.abs().argmin(dim=-1)  # (N,3)
        xerr = dx.gather(-1, cell.unsqueeze(-1)).squeeze(-1)
        along = collar_l[..., 1] - c.chan_y0  # slot travel past the port/channel datum
        axis_up = self._bottle_axis()[..., 2]
        hung = ((xerr.abs() < c.x_tol)
                & (along >= c.hang_y_min) & (along <= c.hang_y_max)
                & (collar_l[..., 2] >= c.collar_z_lo) & (collar_l[..., 2] <= c.collar_z_hi)
                & (axis_up >= math.cos(math.radians(c.hang_tilt_deg)))
                & (body_l[..., 2] < c.body_below_z))
        return hung, cell

    def settled(self) -> torch.Tensor:
        """(N,3) bool: carafe lin AND ang velocity below thresholds."""
        lin = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.bottles], dim=1)
        ang = torch.stack([b.data.root_ang_vel_w.norm(dim=-1) for b in self.bottles], dim=1)
        return (lin < self.cfg.settle_lin) & (ang < self.cfg.settle_ang)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch lifted (collar ever at rail height while not flying), ported (collar
        ever inside a keyhole above the plate) and hung (full hang clause, slow, for
        `hang_latch_steps` consecutive substeps), every physics substep."""
        c = self.cfg
        collar_l, _body_l = self._points_rail_local()
        pos, _q = self._bottle_tensors()
        lin = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.bottles], dim=1)
        collar_env_z = (pos + self._bottle_axis() * c.collar_c)[..., 2] \
            - self.env_origins[:, None, 2]
        self.lifted_latch = torch.maximum(
            self.lifted_latch, ((collar_env_z > c.lifted_z) & (lin < 1.0)).float())
        cells = torch.tensor(c.cells_x, device=collar_l.device)
        dx = (collar_l[..., 0:1] - cells.view(1, 1, 3)).abs().min(dim=-1).values
        in_keyhole = ((dx < c.port_half)
                      & (collar_l[..., 1] > c.port_cy - c.port_half)
                      & (collar_l[..., 1] < c.chan_y1)
                      & (collar_l[..., 2] > c.collar_z_lo)
                      & (collar_l[..., 2] < 0.030))
        self.ported_latch = torch.maximum(self.ported_latch, in_keyhole.float())
        hung, _cell = self.hang_cells()
        ok = hung & (lin < c.slow_lin)
        self.hang_ctr = torch.where(ok, self.hang_ctr + 1, torch.zeros_like(self.hang_ctr))
        self.max_hang_ctr = torch.maximum(self.max_hang_ctr, self.hang_ctr)
        self.hung_latch = torch.maximum(
            self.hung_latch, (self.hang_ctr >= c.hang_latch_steps).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: every carafe hanging (full geometric clause), settled, and the
        three occupied keyholes are DISTINCT (all three used) — judged live."""
        hung, cell = self.hang_cells()
        all_hung = (hung & self.settled()).all(dim=1)
        one_hot = torch.zeros(cell.shape[0], 3, device=cell.device)
        one_hot.scatter_(1, cell, 1.0)  # marks cells claimed by ANY carafe
        distinct = (one_hot > 0).all(dim=1)
        return all_hung & distinct

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: latched stage credit — 0.05 lifted + 0.10 collar in a
        keyhole + 0.10 hung, per carafe, capped at 0.75; exactly 1.0 iff success().
        Doing nothing scores ~0; latched credit never evaporates."""
        base = (0.05 * self.lifted_latch.sum(dim=1)
                + 0.10 * self.ported_latch.sum(dim=1)
                + 0.10 * self.hung_latch.sum(dim=1)).clamp(0.0, 0.75)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="bottle_hang_rail", robot="null"))
