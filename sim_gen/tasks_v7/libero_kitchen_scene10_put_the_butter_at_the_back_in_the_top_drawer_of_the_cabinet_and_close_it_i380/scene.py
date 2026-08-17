"""ButterHatchScene — close the drawer FIRST, then post the butter through the
countertop hatch into the closed drawer.

Derived from libero_90/kitchen_scene10 "put the butter at the back in the top drawer of
the cabinet and close it", with the seed's plan INVERTED. The seed opens a drawer,
lowers the butter into the exposed cavity from above, and pushes the drawer shut —
open / insert / close on a top-accessible sliding volume. Here the drawer cavity is
NEVER top-accessible: a fixed countertop plus a front canopy roof covers the drawer's
entire travel with a 6 mm gap (a 26 mm butter cannot enter — smoke-tested denial).
The ONLY way into the drawer is a square HATCH cut through the countertop, ringed by
an orange collar chute — and the hatch lines up with the drawer cavity ONLY WHEN THE
DRAWER IS FULLY CLOSED. While the drawer is open, the hatch column continues straight
down past the drawer plane into a REJECT CELLAR on the cabinet floor: a deposit made
before closing is not merely unrewarded, it is physically LOST below the drawer
(smoke-tested). So the required order is the seed's plan backwards: push the drawer
shut first, then pick the butter up and drop it through the hatch so gravity delivers
it into the sealed drawer. The closure and the alignment are the same act — closing
the drawer is not the finishing touch, it is what CREATES the deposit path. A white
paraffin block of identical shape must stay out of the drawer.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - cabinet: KINEMATIC compound — base slab, side walls, back wall, two front pillars,
    a countertop authored as four strips around the 100 x 100 mm hatch hole, and the
    collar chute (four orange walls, 55 mm tall) on top of the hole. The countertop
    extends 180 mm past the bay front as a CANOPY over the drawer's pulled-out sweep.
    Origin at the ground centre of the drawer bay.
  - drawer: DYNAMIC compound tray — floor plate, four 60 mm walls, a face plate and a
    protruding push KNOB bar (the knob tip clears the canopy front edge even at full
    open, so the arm can push it home from the front). Rides a bind-time horizontal
    PrismaticJoint (cabinet -> drawer, axis X, limits [-travel, 0]; joint-pair
    collision disabled — the joint owns alignment, all clearances are asserted at
    import time). q = 0 is fully closed (hard stop), q = -travel fully open.
  - butter (yellow) / paraffin (white): 58 x 32 x 26 mm blocks — the 32 mm faces are
    the parallel-jaw grasp feature.

Per-episode randomization (readback-verifiable): the drawer's initial opening q0 ~
U[-0.14, -0.06] (well clear of both joint stops), a random swap of {butter, paraffin}
over the two floor spawn slots, per-object xy jitter and free yaw.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.10 * approach — butter approach to the hatch mouth point, measured against the
                    per-episode spawn distance (running max; exactly 0 for idling)
  0.20 * closed   — drawer ever fully closed and still (latched bool)
  0.25 * chute    — butter ever inside the collar/hatch column WHILE the drawer is
                    closed (latched bool: the aligned deposit in flight)
  0.30 * inside   — butter ever settled inside the drawer cavity with the drawer
                    closed (latched bool)
  1.0 iff success() — butter settled in the cavity, drawer fully closed, paraffin
                    out, everything at rest. Non-success cap 0.85.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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
# One rigid body per object, several child colliders, authored with raw pxr APIs; only
# `isaaclab.sim.utils.clone` is borrowed (regex-resolve + per-env replication).

_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _root_xform(prim_path: str, translation, orientation):
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


def _dynamic_body(root, mass: float, *, lin_damp: float, ang_damp: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(lin_damp)
    pxrb.CreateAngularDampingAttr(ang_damp)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)


def _spawn_cabinet(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the hatch cabinet: KINEMATIC compound. Origin at the ground centre of
    the drawer bay. +x = toward the back wall, -x = toward the robot / portico front.
    The countertop is four strips around the hatch hole; the collar chute rings the
    hole on top; the canopy region (countertop forward of the bay) is held up by two
    corner pillars."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    top_c = c.top_under + c.top_t / 2  # countertop slab centre z
    bay_cx = (-c.bay_half_x + c.bay_back_x) / 2  # x centre of base/side walls
    bay_lx = c.bay_back_x + c.bay_half_x  # x length of base/side walls
    # base slab (the reject-cellar floor)
    _add_box(stage, f"{prim_path}/base", center=(bay_cx, 0.0, c.base_t / 2),
             size=(bay_lx, 2 * c.width_half_out, c.base_t), color=c.wood, collide=collide)
    # side walls + back wall (ground -> countertop underside)
    wy = (c.width_half_in + c.width_half_out) / 2
    wt = c.width_half_out - c.width_half_in
    for sgn, nm in ((1.0, "wall_yp"), (-1.0, "wall_yn")):
        _add_box(stage, f"{prim_path}/{nm}", center=(bay_cx, sgn * wy, c.top_under / 2),
                 size=(bay_lx, wt, c.top_under), color=c.wood, collide=collide)
    _add_box(stage, f"{prim_path}/wall_back",
             center=(c.bay_back_x - wt / 2, 0.0, c.top_under / 2),
             size=(wt, 2 * c.width_half_out, c.top_under), color=c.wood, collide=collide)
    # front canopy pillars
    for sgn, nm in ((1.0, "pillar_yp"), (-1.0, "pillar_yn")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(c.canopy_front + c.pillar_t / 2, sgn * wy, c.top_under / 2),
                 size=(c.pillar_t, wt, c.top_under), color=c.wood, collide=collide)
    # countertop: four strips around the hatch hole
    px0, px1 = c.port_cx - c.port_half, c.port_cx + c.port_half
    _add_box(stage, f"{prim_path}/top_front",
             center=((c.canopy_front + px0) / 2, 0.0, top_c),
             size=(px0 - c.canopy_front, 2 * c.width_half_out, c.top_t),
             color=c.wood_top, collide=collide)
    _add_box(stage, f"{prim_path}/top_back",
             center=((px1 + c.bay_back_x) / 2, 0.0, top_c),
             size=(c.bay_back_x - px1, 2 * c.width_half_out, c.top_t),
             color=c.wood_top, collide=collide)
    sy = (c.port_half_y + c.width_half_out) / 2
    st = c.width_half_out - c.port_half_y
    for sgn, nm in ((1.0, "top_yp"), (-1.0, "top_yn")):
        _add_box(stage, f"{prim_path}/{nm}", center=(c.port_cx, sgn * sy, top_c),
                 size=(2 * c.port_half, st, c.top_t), color=c.wood_top, collide=collide)
    # collar chute ringing the hole (inner faces flush with the hole edges)
    cz = c.top_under + c.top_t + c.collar_h / 2
    _add_box(stage, f"{prim_path}/collar_xn",
             center=(px0 - c.collar_t / 2, 0.0, cz),
             size=(c.collar_t, 2 * (c.port_half_y + c.collar_t), c.collar_h),
             color=c.collar_color, collide=collide)
    _add_box(stage, f"{prim_path}/collar_xp",
             center=(px1 + c.collar_t / 2, 0.0, cz),
             size=(c.collar_t, 2 * (c.port_half_y + c.collar_t), c.collar_h),
             color=c.collar_color, collide=collide)
    for sgn, nm in ((1.0, "collar_yp"), (-1.0, "collar_yn")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(c.port_cx, sgn * (c.port_half_y + c.collar_t / 2), cz),
                 size=(2 * c.port_half, c.collar_t, c.collar_h),
                 color=c.collar_color, collide=collide)
    return root


def _spawn_drawer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the drawer: DYNAMIC compound, origin at the CENTRE OF THE CAVITY FLOOR
    TOP plane. Children: floor plate, back/front walls (full width), side walls, face
    plate beyond the front wall, and the protruding push knob bar."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _dynamic_body(root, cfg.mass_props.mass, lin_damp=cfg.lin_damp, ang_damp=0.05)
    collide = _make_collide(cfg)
    c = cfg
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, -c.floor_t / 2),
             size=(c.out_half_x * 2, c.out_half_y * 2, c.floor_t),
             color=c.wood, collide=collide)
    wz = c.wall_h / 2
    for sgn, nm in ((1.0, "wall_xp"), (-1.0, "wall_xn")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(sgn * (c.cav_half_x + c.wall_t / 2), 0.0, wz),
                 size=(c.wall_t, c.out_half_y * 2, c.wall_h), color=c.wood, collide=collide)
    for sgn, nm in ((1.0, "wall_yp"), (-1.0, "wall_yn")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(0.0, sgn * (c.cav_half_y + c.wall_t / 2), wz),
                 size=(c.cav_half_x * 2, c.wall_t, c.wall_h), color=c.wood, collide=collide)
    _add_box(stage, f"{prim_path}/face",
             center=(-(c.face_x0 + c.face_t / 2), 0.0, c.face_cz),
             size=(c.face_t, c.face_half_y * 2, c.face_h), color=c.face_color,
             collide=collide)
    _add_box(stage, f"{prim_path}/knob",
             center=(-(c.face_x0 + c.face_t + c.knob_l / 2), 0.0, c.knob_cz),
             size=(c.knob_l, c.knob_w, c.knob_w), color=c.knob_color, collide=collide)
    return root


def _spawn_block(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a free block (butter / paraffin): DYNAMIC box, origin at its centre."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _dynamic_body(root, cfg.mass_props.mass, lin_damp=0.05, ang_damp=0.10)
    collide = _make_collide(cfg)
    _add_box(stage, f"{prim_path}/body", center=(0.0, 0.0, 0.0),
             size=(cfg.bx, cfg.by, cfg.bz), color=cfg.color, collide=collide)
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
            bay_half_x: float = 0.117
            bay_back_x: float = 0.133
            width_half_in: float = 0.096
            width_half_out: float = 0.112
            base_t: float = 0.012
            top_under: float = 0.216
            top_t: float = 0.014
            canopy_front: float = -0.297
            pillar_t: float = 0.016
            port_cx: float = 0.040
            port_half: float = 0.050
            port_half_y: float = 0.050
            collar_h: float = 0.055
            collar_t: float = 0.012
            wood: tuple = (0.55, 0.38, 0.22)
            wood_top: tuple = (0.36, 0.25, 0.16)
            collar_color: tuple = (0.90, 0.45, 0.10)
            contact_offset: float = 0.002

        @configclass
        class DrawerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_drawer)
            out_half_x: float = 0.112
            out_half_y: float = 0.092
            floor_t: float = 0.012
            cav_half_x: float = 0.100
            cav_half_y: float = 0.080
            wall_t: float = 0.012
            wall_h: float = 0.060
            face_x0: float = 0.112  # face inner plane distance from origin
            face_t: float = 0.014
            face_half_y: float = 0.092
            face_h: float = 0.100
            face_cz: float = 0.010
            knob_l: float = 0.050
            knob_w: float = 0.024
            knob_cz: float = 0.010
            lin_damp: float = 2.0
            wood: tuple = (0.72, 0.54, 0.32)
            face_color: tuple = (0.60, 0.42, 0.25)
            knob_color: tuple = (0.18, 0.18, 0.20)
            contact_offset: float = 0.002

        @configclass
        class BlockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_block)
            bx: float = 0.058
            by: float = 0.032
            bz: float = 0.026
            color: tuple = (0.5, 0.5, 0.5)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(cabinet=CabinetSpawnerCfg, drawer=DrawerSpawnerCfg,
                              block=BlockSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ButterHatchSceneCfg(BaseCfg):
    """Config for `ButterHatchScene`. The order-forcing is METRIC and asserted at
    import time: the roof gap over the drawer walls (6 mm) is smaller than the
    thinnest butter dimension (26 mm), so top insertion is impossible at any drawer
    position; the hatch column overlaps the drawer cavity with >= 10 mm margin only
    at closed, and clears the whole drawer body by >= 20 mm at full open, where it
    drops past the drawer plane into the reject cellar."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    close_tol: float = tunable(0.008)  # |q| below this counts as fully closed (m)
    cav_x_tol: float = tunable(0.088)  # in-cavity: |x| bound in the DRAWER frame (m)
    cav_y_tol: float = tunable(0.070)  # in-cavity: |y| bound in the DRAWER frame (m)
    cav_z_lo: float = tunable(0.004)  # in-cavity: z band above the cavity floor top ...
    cav_z_hi: float = tunable(0.056)  # ... (m); above = perched on walls / roof
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    q0_min: float = tunable(-0.14)  # initial drawer opening lower bound (m, negative=open)
    q0_max: float = tunable(-0.06)  # initial drawer opening upper bound
    spawn_jitter: float = tunable(0.03)  # per-block spawn xy jitter (+/- m)
    yaw_free: bool = tunable(True)  # free spawn yaw (demo sets False)
    swap_slots: bool = tunable(True)  # random butter/paraffin slot swap (demo sets False)

    # --- info: layout (single Franka base near the origin) ---------------------------------------
    cab_pos: tuple = info((0.58, 0.0))  # cabinet bay-centre xy
    spawn_slots: tuple = info(((0.24, 0.30), (0.24, -0.30)))  # butter/paraffin floor slots

    # --- info: cabinet structure (cabinet frame; +x = back, -x = portico front) ------------------
    bay_half_x: float = info(0.117)  # side walls' front edge (bay mouth) at -bay_half_x
    bay_back_x: float = info(0.133)  # back wall outer face
    width_half_in: float = info(0.096)  # side wall inner face
    width_half_out: float = info(0.112)  # cabinet outer half-width
    base_t: float = info(0.012)  # cellar floor slab thickness
    top_under: float = info(0.216)  # countertop underside height
    top_t: float = info(0.014)  # countertop thickness
    canopy_front: float = info(-0.297)  # countertop/canopy front edge
    pillar_t: float = info(0.016)
    port_cx: float = info(0.040)  # hatch hole centre x
    port_half: float = info(0.050)  # hatch hole half-length (x)
    port_half_y: float = info(0.050)  # hatch hole half-width (y)
    collar_h: float = info(0.055)  # collar chute height above the countertop
    collar_t: float = info(0.012)

    # --- info: drawer ----------------------------------------------------------------------------
    plane_z: float = info(0.150)  # drawer origin height = cavity floor TOP plane
    travel: float = info(0.160)  # prismatic stroke; q in [-travel, 0], 0 = closed
    out_half_x: float = info(0.112)  # floor plate outer half-length
    out_half_y: float = info(0.092)
    floor_t: float = info(0.012)
    cav_half_x: float = info(0.100)  # cavity inner half-length
    cav_half_y: float = info(0.080)
    wall_t: float = info(0.012)
    wall_h: float = info(0.060)
    face_x0: float = info(0.112)  # face plate inner plane (drawer frame, -x side)
    face_t: float = info(0.014)
    face_half_y: float = info(0.092)
    face_h: float = info(0.100)  # face spans z [-0.040, 0.060] about face_cz=0.010
    face_cz: float = info(0.010)
    knob_l: float = info(0.050)
    knob_w: float = info(0.024)
    knob_cz: float = info(0.010)
    drawer_mass: float = info(0.50)
    drawer_lin_damp: float = info(2.0)

    # --- info: free blocks -----------------------------------------------------------------------
    butter_size: tuple = info((0.058, 0.032, 0.026))
    butter_mass: float = info(0.08)
    butter_color: tuple = info((0.93, 0.80, 0.22))  # yellow — the payload
    paraffin_color: tuple = info((0.92, 0.92, 0.90))  # white — identical shape, stays out

    contact_offset: float = info(0.002)
    # rubric weights (0.10 + 0.20 + 0.25 + 0.30 = 0.85 = the non-success cap)
    w_app: float = info(0.10)
    w_close: float = info(0.20)
    w_chute: float = info(0.25)
    w_in: float = info(0.30)

    # Derived (filled in __post_init__).
    mouth_z: float = field(default=None, init=False)  # hatch mouth hover point height
    chute_z_lo: float = field(default=None, init=False)
    chute_z_hi: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        c = self
        c.chute_z_lo = c.top_under - 0.004  # just under the countertop underside
        c.chute_z_hi = c.top_under + c.top_t + c.collar_h + 0.025
        c.mouth_z = c.chute_z_hi + 0.005

        # ---- import-time geometry audit: the order-forcing is metric, not hopeful ----
        top_gap = c.top_under - (c.plane_z + c.wall_h)
        assert 0.004 < top_gap < min(c.butter_size), \
            f"roof gap {top_gap} must be positive and below the butter's thinnest side"
        # hatch inside the cavity at closed, with margin
        assert c.port_cx - c.port_half > -c.cav_half_x + 0.010
        assert c.port_cx + c.port_half < c.cav_half_x - 0.010
        assert c.port_half_y < c.cav_half_y - 0.010
        # hatch fully clear of the drawer body at full open -> reject cellar below
        drawer_max_x_open = c.out_half_x - c.travel
        assert drawer_max_x_open < (c.port_cx - c.port_half) - 0.020, \
            "open-drawer hatch column not clear of the drawer body"
        # canopy covers the cavity top over the whole stroke
        assert c.canopy_front < -c.cav_half_x - c.travel - 0.020
        # hole admits a falling butter with margin (largest horizontal diagonal)
        diag = math.hypot(c.butter_size[0], c.butter_size[1])
        assert 2 * min(c.port_half, c.port_half_y) > diag + 0.025
        # reject drop clears under the drawer floor plate
        assert c.plane_z - c.floor_t > c.base_t + max(c.butter_size) + 0.010
        # knob tip protrudes beyond the canopy at full open (pushable from the front)
        knob_tip_open = -(c.face_x0 + c.face_t + c.knob_l) - c.travel
        assert knob_tip_open < c.canopy_front - 0.020
        # face plate clears the side walls / pillars laterally
        assert c.width_half_in - c.face_half_y >= 0.004
        assert c.width_half_in - c.out_half_y >= 0.004
        # face top stays under the countertop
        assert c.plane_z + c.face_cz + c.face_h / 2 < c.top_under - 0.004
        # q0 sampling band clear of both joint stops
        assert c.q0_min > -c.travel + 0.015 and c.q0_max < -c.close_tol - 0.020


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("butter_hatch")
class ButterHatchScene(BaseScene):
    cfg: ButterHatchSceneCfg

    def __init__(self, cfg: ButterHatchSceneCfg | None = None) -> None:
        super().__init__(cfg or ButterHatchSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        cabinet_spawn = spawners["cabinet"](
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            bay_half_x=c.bay_half_x, bay_back_x=c.bay_back_x,
            width_half_in=c.width_half_in, width_half_out=c.width_half_out,
            base_t=c.base_t, top_under=c.top_under, top_t=c.top_t,
            canopy_front=c.canopy_front, pillar_t=c.pillar_t, port_cx=c.port_cx,
            port_half=c.port_half, port_half_y=c.port_half_y, collar_h=c.collar_h,
            collar_t=c.collar_t, contact_offset=c.contact_offset,
        )
        drawer_spawn = spawners["drawer"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.drawer_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=4),
            out_half_x=c.out_half_x, out_half_y=c.out_half_y, floor_t=c.floor_t,
            cav_half_x=c.cav_half_x, cav_half_y=c.cav_half_y, wall_t=c.wall_t,
            wall_h=c.wall_h, face_x0=c.face_x0, face_t=c.face_t,
            face_half_y=c.face_half_y, face_h=c.face_h, face_cz=c.face_cz,
            knob_l=c.knob_l, knob_w=c.knob_w, knob_cz=c.knob_cz,
            lin_damp=c.drawer_lin_damp, contact_offset=c.contact_offset,
        )

        def block_spawn(color):
            return spawners["block"](
                mass_props=sim_utils.MassPropertiesCfg(mass=c.butter_mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16, solver_velocity_iteration_count=4),
                bx=c.butter_size[0], by=c.butter_size[1], bz=c.butter_size[2],
                color=color, contact_offset=c.contact_offset,
            )

        cx, cy = c.cab_pos
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
                spawn=cabinet_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, 0.0)),
            ),
            "drawer": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Drawer",
                spawn=drawer_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(cx - 0.10, cy, c.plane_z)),
            ),
            "butter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Butter",
                spawn=block_spawn(c.butter_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.spawn_slots[0][0], c.spawn_slots[0][1],
                         c.butter_size[2] / 2 + 0.002)),
            ),
            "paraffin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Paraffin",
                spawn=block_spawn(c.paraffin_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.spawn_slots[1][0], c.spawn_slots[1][1],
                         c.butter_size[2] / 2 + 0.002)),
            ),
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
        self.cabinet: RigidObject = env.iscene["cabinet"]
        self.drawer: RigidObject = env.iscene["drawer"]
        self.butter: RigidObject = env.iscene["butter"]
        self.paraffin: RigidObject = env.iscene["paraffin"]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._app_max = torch.zeros(n, device=dev)  # hatch-mouth approach, running max
        self._closed = torch.zeros(n, dtype=torch.bool, device=dev)  # drawer ever closed+still
        self._chuted = torch.zeros(n, dtype=torch.bool, device=dev)  # butter in chute, closed
        self._inside = torch.zeros(n, dtype=torch.bool, device=dev)  # butter in cavity, closed
        self._d_init = torch.full((n,), 0.5, device=dev)  # spawn->mouth distance (reset-set)

    def _author_joints(self) -> None:
        """Per env, one bind-time PrismaticJoint cabinet -> drawer along +x, limits
        [-travel, 0] about the CLOSED pose (a hard stop at q=0 backs the closed state
        against deposit impacts, which push the drawer toward +x = shut). Joint-pair
        collision disabled: the joint owns the drawer's alignment; every static
        clearance along the sweep is asserted at import time, so the sweep volume is
        carved by construction."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/drawer_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Cabinet"])
            j.CreateBody1Rel().SetTargets([f"{base}/Drawer"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.plane_z)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-float(c.travel))
            j.CreateUpperLimitAttr(0.0)

    # ----- frames / predicates -------------------------------------------------------------------
    def q(self) -> torch.Tensor:
        """(N,) drawer joint coordinate: 0 = fully closed, -travel = fully open."""
        return (self.drawer.data.root_pos_w[:, 0] - self.env_origins[:, 0]
                - self.cfg.cab_pos[0])

    def drawer_closed(self) -> torch.Tensor:
        return self.q().abs() <= self.cfg.close_tol

    def in_cavity(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body centre inside the drawer cavity box, judged in the DRAWER
        frame (the drawer never rotates), between the cavity floor and the wall top."""
        c = self.cfg
        rel = body.data.root_pos_w - self.drawer.data.root_pos_w
        return ((rel[:, 0].abs() <= c.cav_x_tol) & (rel[:, 1].abs() <= c.cav_y_tol)
                & (rel[:, 2] > c.cav_z_lo) & (rel[:, 2] < c.cav_z_hi))

    def in_chute(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body centre inside the hatch/collar column (cabinet frame)."""
        c = self.cfg
        rel = body.data.root_pos_w - self.cabinet.data.root_pos_w
        return ((rel[:, 0] - c.port_cx).abs() <= c.port_half) \
            & (rel[:, 1].abs() <= c.port_half_y) \
            & (rel[:, 2] > c.chute_z_lo) & (rel[:, 2] < c.chute_z_hi)

    def in_cellar(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body centre in the reject cellar under the drawer plane."""
        c = self.cfg
        rel = body.data.root_pos_w - self.cabinet.data.root_pos_w
        return ((rel[:, 0] > -c.bay_half_x) & (rel[:, 0] < c.bay_back_x)
                & (rel[:, 1].abs() < c.width_half_in)
                & (rel[:, 2] < c.plane_z - c.floor_t - 0.005))

    def _mouth_world(self) -> torch.Tensor:
        c = self.cfg
        m = torch.tensor([c.cab_pos[0] + c.port_cx, c.cab_pos[1], c.mouth_z],
                         device=self.env.device)
        return m.expand(self.env.num_envs, 3) + self.env_origins

    def _still(self, body: RigidObject) -> torch.Tensor:
        return body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: cabinet re-asserted at its fixed pose; drawer re-posed
        follower-only along its unchanged slide to a random opening q0 (clear of both
        stops); butter and paraffin randomly SWAPPED over the two floor slots with xy
        jitter and free yaw; latches cleared; approach baseline recorded."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        torch.rand(3, device=dev)  # burn post-seed draws (first-draw degeneracy)

        # --- cabinet (kinematic, fixed) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = c.cab_pos
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.cabinet.write_root_state_to_sim(st, env_ids)

        # --- drawer: random opening along the unchanged joint (follower-only) ---
        q0 = c.q0_min + torch.rand(m, device=dev) * (c.q0_max - c.q0_min)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.cab_pos[0] + q0
        st[:, 1] = c.cab_pos[1]
        st[:, 2] = c.plane_z
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.drawer.write_root_state_to_sim(st, env_ids)

        # --- blocks: random slot swap + jitter + yaw ---
        slots = torch.tensor(c.spawn_slots, device=dev)  # (2, 2)
        if c.swap_slots:
            swap = torch.randint(0, 2, (m,), device=dev)
        else:
            swap = torch.zeros(m, dtype=torch.long, device=dev)
        butter_xy = None
        for k, body in enumerate((self.butter, self.paraffin)):
            xy = slots[(swap + k) % 2] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.spawn_jitter
            if k == 0:
                butter_xy = xy
            half = ((torch.rand(m, device=dev) * 2 - 1) * math.pi / 2
                    if c.yaw_free else torch.zeros(m, device=dev))
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = c.butter_size[2] / 2 + 0.002
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- approach baseline: butter spawn -> hatch mouth (null scores exactly 0) ---
        mouth = torch.tensor([c.cab_pos[0] + c.port_cx, c.cab_pos[1], c.mouth_z],
                             device=dev).expand(m, 3)
        spawn = torch.cat([butter_xy,
                           torch.full((m, 1), c.butter_size[2] / 2 + 0.002, device=dev)],
                          dim=1)
        self._d_init[env_ids] = (spawn - mouth).norm(dim=-1).clamp(min=0.05)

        # --- clear latches ---
        self._app_max[env_ids] = 0.0
        self._closed[env_ids] = False
        self._chuted[env_ids] = False
        self._inside[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cabinet": self.cabinet.data.root_state_w[env_ids].clone(),
            "drawer": self.drawer.data.root_state_w[env_ids].clone(),
            "butter": self.butter.data.root_state_w[env_ids].clone(),
            "paraffin": self.paraffin.data.root_state_w[env_ids].clone(),
            "app_max": self._app_max[env_ids].clone(),
            "closed": self._closed[env_ids].clone(),
            "chuted": self._chuted[env_ids].clone(),
            "inside": self._inside[env_ids].clone(),
            "d_init": self._d_init[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cabinet.write_root_state_to_sim(state["cabinet"], env_ids)
        self.drawer.write_root_state_to_sim(state["drawer"], env_ids)
        self.butter.write_root_state_to_sim(state["butter"], env_ids)
        self.paraffin.write_root_state_to_sim(state["paraffin"], env_ids)
        self._app_max[env_ids] = state["app_max"]
        self._closed[env_ids] = state["closed"]
        self._chuted[env_ids] = state["chuted"]
        self._inside[env_ids] = state["inside"]
        self._d_init[env_ids] = state["d_init"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden counter CABINET stands on the floor, its open portico front "
            f"facing you. A single DRAWER (light wood, with a dark push KNOB bar "
            f"sticking out of its face at {(c.plane_z + c.knob_cz) * 100:.0f} cm height) "
            f"rides a horizontal slide; it starts pulled partway OUT toward you. The "
            f"countertop extends forward as a fixed CANOPY that roofs the drawer over "
            f"its whole travel, leaving only a {(c.top_under - c.plane_z - c.wall_h) * 1000:.0f} mm "
            f"gap above the drawer walls — you can NEVER reach or drop anything into "
            f"the drawer cavity from above or from the side. The only way in is the "
            f"HATCH: a {2 * c.port_half * 100:.0f} cm square hole through the countertop "
            f"near the back, ringed by an ORANGE collar chute. The hatch column lines "
            f"up with the drawer cavity ONLY when the drawer is FULLY CLOSED (pushed "
            f"all the way in). While the drawer is open, the hatch drops straight past "
            f"the drawer into a reject cellar on the cabinet floor — anything posted "
            f"before the drawer is shut is lost below the drawer and cannot be "
            f"recovered into it. On the floor beside the cabinet lie two blocks of "
            f"identical shape whose positions shuffle between episodes — identify by "
            f"COLOR: a YELLOW butter stick and a WHITE paraffin block.\n"
            f"Goal: get the YELLOW butter inside the CLOSED drawer. Required order, "
            f"forced by the geometry: FIRST push the drawer fully shut (press the dark "
            f"knob or the drawer face straight in until it stops), THEN pick up the "
            f"yellow butter, carry it above the orange collar, and drop it into the "
            f"hatch so it falls through the countertop into the drawer below. Success: "
            f"the butter lying settled inside the drawer cavity, the drawer fully "
            f"closed, the white paraffin left outside the drawer, everything at rest. "
            f"Posting the butter while the drawer is still open (it falls into the "
            f"cellar), posting the white paraffin instead, leaving the drawer ajar, or "
            f"parking the butter on the countertop or in the collar does not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Push the drawer fully shut first, then pick up the yellow butter stick "
            "and drop it through the orange hatch collar on the countertop so it "
            "lands inside the closed drawer. Posting it while the drawer is open "
            "loses it in the cellar and fails; the white paraffin block must stay "
            "out of the drawer."
        )

    # ----- rubric --------------------------------------------------------------------------------
    def _update_latches(self) -> None:
        c = self.cfg
        d = (self.butter.data.root_pos_w - self._mouth_world()).norm(dim=-1)
        app = (1.0 - d / self._d_init).clamp(0.0, 1.0)
        app = torch.nan_to_num(app, nan=0.0, posinf=0.0, neginf=0.0)
        self._app_max = torch.maximum(self._app_max, app)
        closed = self.drawer_closed()
        self._closed |= closed & self._still(self.drawer)
        self._chuted |= self.in_chute(self.butter) & closed
        self._inside |= (self.in_cavity(self.butter) & closed
                         & self._still(self.butter))

    # ----- step-coupled bookkeeping (every step) -------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No plant forces — the mechanism is passive (joint + damping). Latch rubric
        progress every step so the in-flight chute transit is never missed."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: butter settled inside the drawer cavity, drawer fully closed and
        still, paraffin NOT in the cavity, butter still. Physical outcomes only: the
        cavity test is a live drawer-frame pose readback, containment is real (the
        cavity is roofed by the countertop, walled by the drawer)."""
        self._update_latches()
        return (self.in_cavity(self.butter) & self._still(self.butter)
                & self.drawer_closed() & self._still(self.drawer)
                & ~self.in_cavity(self.paraffin))

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10*mouth approach (vs the episode's own spawn
        distance) + 0.20*closed + 0.25*chute-transit-while-closed + 0.30*inside — all
        latched/rising-only, exactly 0 for doing nothing, capped 0.85 — and exactly
        1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_app * self._app_max + c.w_close * self._closed.float()
                + c.w_chute * self._chuted.float()
                + c.w_in * self._inside.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="butter_hatch", robot="null"))
