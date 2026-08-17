"""RackTrayServeScene — fetch the stowed serving tray from its captive dish rack,
seat it flat in the serving stand's well, then serve the FRONT bowl on it.

Derived from libero_90/kitchen_scene2_put_the_black_bowl_at_the_front_on_the_plate
("put the black bowl at the front on the plate": pick the front one of three
identical black bowls and set it on a plate — one unordered pick-and-place judged by
a final xy/z window above a plate that is already sitting there, ready to receive).
What carries over is only the cast: three identical black bowls in a row, of which
the FRONT one is the target, and a flat white "plate" as the serving surface. The
strategic content is replaced:

  - the serving surface (a square white TRAY — the task's plate) does NOT start at
    the goal. It starts STOWED ON EDGE inside a tall dish RACK: a slotted channel
    (16 mm gap) with a TOP RAIL that leaves only ~6 mm of headroom over the stowed
    tray, so the seed's universal primitive — pick the thing up — is geometrically
    impossible; the tray's only escape is its single sliding DoF, out through the
    rack's open mouth onto a low-fenced apron;
  - the goal fixture is a serving STAND with a shallow square WELL, empty at reset.
    Success requires the tray SEATED flat on the well floor (a rim-perched or
    tilted or misyawed tray sits high and is rejected), and only then the FRONT
    bowl settled centered ON the tray, with both decoy bowls off it;
  - the execution order is forced by physics, not by fiat: dumping the front bowl
    into the well first (the literal seed plan — "put the bowl at the goal") leaves
    an obstruction on which a dropped tray rests high and tilted, so the rubric
    rejects it and the bowl must be removed again before the tray can seat.

So a solver needs a different PLAN (prepare the destination before serving: extract
-> reorient 90 degrees -> install -> then place) and different code structure (a
constrained in-channel slide, a large-object reorientation and recess seating, then
a stacking placement) instead of the seed's single grasp-carry-set-down.

Assets are fully procedural (compound-spawner pattern; child colliders of one body
never self-collide):
  - counter: static box, top at `surface_z`;
  - rack: ONE kinematic compound (floor strip, two tall slot walls, back wall, top
    rail, two low apron fences), origin at the MOUTH center on the counter, channel
    along local +x; re-posed each reset with xy jitter and a yaw jitter;
  - stand: ONE kinematic compound (base slab + four low rim walls forming a square
    well, inner 190 mm, floor top 10 mm above the counter), xy jitter + FREE yaw;
  - tray: DYNAMIC white board 170 x 170 x 10 mm, spawned standing on edge inside
    the rack slot at a jittered stow depth;
  - bowls: three DYNAMIC identical black open-box bowls (96 mm outer, 48 mm tall)
    dealt over the three row slots by a fresh permutation each episode; the TARGET
    is whichever body landed on the FRONT slot (largest x — the end of the row
    nearest the counter edge that the rack mouth faces, where the robot stands),
    latched from that deal exactly like the seed latches "the bowl at the front".

Geometry facts the rubric leans on (rack-local x from the mouth, z from counter top):
  - stowed tray spans x ~ [-0.135, +0.047]: its rear ~100 mm sits under the top
    rail (z 0.182, tray top 0.176 -> ~6 mm lift headroom) and the slot walls are
    200 mm tall, so from stow the tray cannot leave upward; the mouth at x=0 is the
    only exit, reached by sliding ~140 mm along +x;
  - tray seated in the well: tray center z = 0.015 above the counter; a tray
    resting on the 12 mm rim walls sits at ~0.027+ and a tray on a bowl in the well
    at ~0.06+, both far outside the +-4 mm seat band; a 45-degree-misyawed tray
    (diagonal 240 mm > 190 mm well) can only rest ON the rim;
  - bowl on tray: bowl bottom within (-3, +20) mm of the tray top and 55 mm of the
    tray center; a bowl in the BARE well sits a full tray thickness low and, more
    to the point, fails the tray-seated clause entirely.

Rubric (0..1; latched/running-max credit, monotone, ~0 for the null policy):
  0.00-0.30  extraction progress — running max of the tray's rack-frame +x travel
             from its latched stow depth toward the mouth (5 mm deadband)
  0.35  s_out    — tray fully clear of the rack (center past the mouth by 100 mm)
  0.60  s_seated — tray seated flat on the well floor
  0.80  s_loaded — tray seated AND the target bowl on it
  1.0   iff success(): seated + target bowl centered on the tray + decoys off the
        tray + everything settled.
The task is reversible (a bowl dumped in the well can be removed), so there are no
spoil latches.

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


def _add_box(stage, path: str, *, center, size, color, collide: Callable):
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _phys_material(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _bind_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC dish rack. Local origin: the MOUTH center on the counter top plane;
    the storage channel extends toward local -x, the extraction direction (and the
    low-fenced apron) toward local +x. Children:
      - floor strip (slick) under channel + apron;
      - two tall slot walls (x in [-slot_len, 0], inner gap `gap`);
      - back wall closing the -x end;
      - TOP RAIL over the rear of the channel, ~6 mm above the stowed tray's top
        edge — the captivity element;
      - two LOW apron fences continuing the gap beyond the mouth (the extracted
        tray stages on edge between them)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    slick = _phys_material(stage, f"{prim_path}/slickmat", c.mu_static, c.mu_dynamic)
    _bind_material(root, slick)
    half_gap = c.gap / 2
    wall_cy = half_gap + c.wall_t / 2
    # floor strip: channel back .. apron end
    x0, x1 = -c.slot_len, c.apron_len
    _add_box(stage, f"{prim_path}/floor",
             center=((x0 + x1) / 2, 0.0, c.floor_top - c.floor_t / 2),
             size=(x1 - x0, c.gap + 2 * c.wall_t, c.floor_t),
             color=(0.55, 0.50, 0.42), collide=collide)
    # tall slot walls
    for sgn, tag in ((1.0, "l"), (-1.0, "r")):
        _add_box(stage, f"{prim_path}/wall_{tag}",
                 center=(-c.slot_len / 2, sgn * wall_cy, c.floor_top + c.wall_h / 2),
                 size=(c.slot_len, c.wall_t, c.wall_h),
                 color=(0.42, 0.36, 0.28), collide=collide)
    # back wall
    _add_box(stage, f"{prim_path}/back",
             center=(-c.slot_len - c.wall_t / 2, 0.0, c.floor_top + c.wall_h / 2),
             size=(c.wall_t, c.gap + 2 * c.wall_t, c.wall_h),
             color=(0.42, 0.36, 0.28), collide=collide)
    # top rail (captivity): z rail_z0..rail_z0+rail_t over x in [-slot_len, rail_x1]
    _add_box(stage, f"{prim_path}/rail",
             center=((-c.slot_len + c.rail_x1) / 2, 0.0, c.rail_z0 + c.rail_t / 2),
             size=(c.rail_x1 + c.slot_len, c.gap + 2 * c.wall_t, c.rail_t),
             color=(0.30, 0.26, 0.20), collide=collide)
    # low apron fences
    for sgn, tag in ((1.0, "l"), (-1.0, "r")):
        _add_box(stage, f"{prim_path}/fence_{tag}",
                 center=(c.apron_len / 2, sgn * wall_cy, c.floor_top + c.fence_h / 2),
                 size=(c.apron_len, c.wall_t, c.fence_h),
                 color=(0.42, 0.36, 0.28), collide=collide)
    return root


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC serving stand: base slab + four low rim walls forming a shallow
    square WELL (inner `well_in`, floor top `base_h` above the counter). Local
    origin at the base footprint center on the counter top."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/base", center=(0.0, 0.0, c.base_h / 2),
             size=(c.base_w, c.base_w, c.base_h), color=(0.35, 0.40, 0.45),
             collide=collide)
    wc = c.well_in / 2 + c.rim_t / 2
    wlen = c.well_in + 2 * c.rim_t
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/rim_x{'p' if sgn > 0 else 'n'}",
                 center=(sgn * wc, 0.0, c.base_h + c.rim_h / 2),
                 size=(c.rim_t, wlen, c.rim_h), color=(0.22, 0.26, 0.30),
                 collide=collide)
        _add_box(stage, f"{prim_path}/rim_y{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * wc, c.base_h + c.rim_h / 2),
                 size=(c.well_in, c.rim_t, c.rim_h), color=(0.22, 0.26, 0.30),
                 collide=collide)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC serving tray: one white board, local origin at its CENTER. MassAPI
    mass + diagonal inertia authored explicitly (custom spawners apply no cfg
    schemas); slick-ish material so the in-channel slide is honest."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    rb.CreateSolverPositionIterationCountAttr(16)
    rb.CreateSolverVelocityIterationCountAttr(4)
    rb.CreateMaxDepenetrationVelocityAttr(0.5)
    rb.CreateLinearDampingAttr(0.05)
    rb.CreateAngularDampingAttr(0.10)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(4.0e-4, 4.0e-4, 8.0e-4))
    collide = _make_collide(cfg.contact_offset)
    mat = _phys_material(stage, f"{prim_path}/traymat", cfg.mu_static, cfg.mu_dynamic)
    _bind_material(root, mat)
    _add_box(stage, f"{prim_path}/board", center=(0.0, 0.0, 0.0),
             size=(cfg.side, cfg.side, cfg.thick), color=(0.93, 0.93, 0.90),
             collide=collide)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC black open-box bowl, local origin at the BOTTOM CENTER: base plate +
    four walls. MassAPI mass + low CoM + diagonal inertia authored explicitly."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    rb.CreateSolverPositionIterationCountAttr(16)
    rb.CreateSolverVelocityIterationCountAttr(4)
    rb.CreateMaxDepenetrationVelocityAttr(0.5)
    rb.CreateLinearDampingAttr(0.05)
    rb.CreateAngularDampingAttr(0.10)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.012))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(2.0e-4, 2.0e-4, 3.0e-4))
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    inner = c.outer - 2 * c.wall_t
    _add_box(stage, f"{prim_path}/base", center=(0.0, 0.0, c.base_t / 2),
             size=(inner, inner, c.base_t), color=c.color, collide=collide)
    wz = c.base_t + (c.height - c.base_t) / 2
    wh = c.height - c.base_t
    wc = inner / 2 + c.wall_t / 2
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/wall_x{'p' if sgn > 0 else 'n'}",
                 center=(sgn * wc, 0.0, wz), size=(c.wall_t, c.outer, wh),
                 color=c.color, collide=collide)
        _add_box(stage, f"{prim_path}/wall_y{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * wc, wz), size=(inner, c.wall_t, wh),
                 color=c.color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rack" not in _SPAWNER_CACHE:

        @configclass
        class RackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            slot_len: float = 0.15
            apron_len: float = 0.21
            gap: float = 0.016
            wall_t: float = 0.012
            wall_h: float = 0.20
            floor_top: float = 0.006
            floor_t: float = 0.008
            rail_x1: float = -0.035
            rail_z0: float = 0.182
            rail_t: float = 0.020
            fence_h: float = 0.035
            mu_static: float = 0.15
            mu_dynamic: float = 0.12
            contact_offset: float = 0.0015

        @configclass
        class StandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            base_w: float = 0.24
            base_h: float = 0.010
            well_in: float = 0.190
            rim_t: float = 0.015
            rim_h: float = 0.012
            contact_offset: float = 0.0015

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            side: float = 0.170
            thick: float = 0.010
            mass: float = 0.15
            mu_static: float = 0.25
            mu_dynamic: float = 0.20
            contact_offset: float = 0.0015

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            outer: float = 0.096
            height: float = 0.048
            base_t: float = 0.008
            wall_t: float = 0.008
            mass: float = 0.10
            color: tuple = (0.07, 0.07, 0.08)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["rack"] = RackSpawnerCfg
        _SPAWNER_CACHE["stand"] = StandSpawnerCfg
        _SPAWNER_CACHE["tray"] = TraySpawnerCfg
        _SPAWNER_CACHE["bowl"] = BowlSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RackTrayServeSceneCfg(BaseCfg):
    """Config for `RackTrayServeScene`. The seat band is honest by construction: a
    seated tray's center sits 15 mm above the counter; the nearest wrong rests are
    rim-perch (27+ mm) and tray-on-bowl (60+ mm), both >= 3x the +-4 mm band. The
    45-degree-misyawed tray (240 mm diagonal > 190 mm well) physically cannot reach
    the seat band. The captivity headroom (6 mm) is < the 176 mm rise a stowed tray
    would need to clear the slot walls."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    out_x: float = tunable(0.10)          # rack-frame tray-center x for "fully out of the rack"
    prog_deadband: float = tunable(0.005)  # extraction progress deadband (m)
    seat_xy_tol: float = tunable(0.015)   # tray center within this of the well center
    seat_dz_tol: float = tunable(0.004)   # |tray center z - seated z| below this
    seat_up_cos: float = tunable(0.99905)  # tray +z within ~2.5 deg of world up
    bowl_xy_tol: float = tunable(0.055)   # target bowl center within this of the tray center
    bowl_dz_lo: float = tunable(-0.003)   # bowl bottom minus tray top, lower gate
    bowl_dz_hi: float = tunable(0.020)    # ... upper gate
    bowl_up_cos: float = tunable(0.90)    # bowl upright gate
    decoy_xy: float = tunable(0.10)       # decoy exclusion radius around the tray axis
    decoy_dz_lo: float = tunable(-0.010)  # decoy exclusion z band (vs tray top)
    decoy_dz_hi: float = tunable(0.060)
    settle_speed: float = tunable(0.05)   # max |lin vel| for "still" (m/s)
    settle_ang: float = tunable(0.5)      # max |ang vel| for "still" (rad/s)
    still_steps: int = tunable(12)        # consecutive still steps -> "at rest"
    warmup_steps: int = tunable(30)       # latch grace after reset (spawn settle)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    slot_jitter: float = tunable(0.02)    # uniform +/- xy jitter per bowl row slot (m)
    stand_jitter: float = tunable(0.02)   # uniform +/- xy jitter on the stand (m)
    stand_free_yaw: bool = tunable(True)  # stand yaw uniform over the full circle
    rack_jitter: float = tunable(0.015)   # uniform +/- xy jitter on the rack (m)
    rack_yaw_deg: float = tunable(8.0)    # uniform +/- yaw jitter on the rack (deg)
    stow_x0: float = tunable(-0.050)      # tray stow depth range (rack-frame center x)
    stow_x1: float = tunable(-0.038)
    shuffle_bowls: bool = tunable(True)   # permute bowl bodies over row slots

    # --- info: layout (counter-top frame; counter top at surface_z) ------------------------------
    surface_z: float = info(0.40)
    counter_size: tuple = info((0.90, 0.90, 0.40))
    rack_xy: tuple = info((0.19, 0.30))   # rack MOUTH center; channel/extraction along +x
    stand_xy: tuple = info((0.10, -0.02))
    slot_xs: tuple = info((0.22, 0.02, -0.18))  # bowl row along x; slot 0 = FRONT (largest x)
    slot_y: float = info(-0.24)
    # --- info: rack structure (rack-local; z from the counter top) -------------------------------
    slot_len: float = info(0.15)
    apron_len: float = info(0.21)
    gap: float = info(0.016)
    rack_floor_top: float = info(0.006)
    rail_z0: float = info(0.182)
    rail_x1: float = info(-0.035)
    # --- info: stand structure -------------------------------------------------------------------
    base_w: float = info(0.24)
    base_h: float = info(0.010)           # well floor top above the counter
    well_in: float = info(0.190)
    rim_h: float = info(0.012)            # rim wall height above the well floor
    # --- info: tray / bowls ----------------------------------------------------------------------
    tray_side: float = info(0.170)
    tray_thick: float = info(0.010)
    tray_mass: float = info(0.15)
    bowl_outer: float = info(0.096)
    bowl_height: float = info(0.048)
    bowl_mass: float = info(0.10)
    contact_offset: float = info(0.0015)
    # rubric stage values (monotone; non-success cap = w_loaded)
    w_prog: float = info(0.30)
    w_out: float = info(0.35)
    w_seated: float = info(0.60)
    w_loaded: float = info(0.80)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("rack_tray_serve")
class RackTrayServeScene(BaseScene):
    cfg: RackTrayServeSceneCfg

    def __init__(self, cfg: RackTrayServeSceneCfg | None = None) -> None:
        super().__init__(cfg or RackTrayServeSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        z0 = c.surface_z

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
            "counter": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Counter",
                spawn=sim_utils.CuboidCfg(
                    size=c.counter_size,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.30)),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, z0 - c.counter_size[2] / 2)),
            ),
            "rack": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=cls["rack"](contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.rack_xy[0], c.rack_xy[1], z0)),
            ),
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=cls["stand"](contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.stand_xy[0], c.stand_xy[1], z0)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=cls["tray"](contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_xy[0] - 0.045, c.rack_xy[1], z0 + 0.093),
                    rot=(0.70711, 0.70711, 0.0, 0.0)),
            ),
        }
        for i in range(3):
            out[f"bowl_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl_" + str(i),
                spawn=cls["bowl"](contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_xs[i], c.slot_y, z0 + 0.002)),
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
        n = env.num_envs
        dev = env.device
        self.rack: RigidObject = env.iscene["rack"]
        self.stand: RigidObject = env.iscene["stand"]
        self.tray: RigidObject = env.iscene["tray"]
        self.bowls: list[RigidObject] = [env.iscene[f"bowl_{i}"] for i in range(3)]
        self.env_origins = env.iscene.env_origins
        # randomization readback
        self.target = torch.zeros(n, dtype=torch.long, device=dev)  # body on the FRONT slot
        self.slot_of = torch.arange(3, device=dev).unsqueeze(0).expand(n, 3).clone()
        self._stow_x = torch.zeros(n, device=dev)  # latched stow depth (rack-frame x)
        # monitors / latches
        self._warmup = torch.zeros(n, dtype=torch.long, device=dev)
        self._still_tray = torch.zeros(n, dtype=torch.long, device=dev)
        self._still_bowl = torch.zeros(n, 3, dtype=torch.long, device=dev)
        self._prog = torch.zeros(n, device=dev)          # running-max extraction progress
        self._s_out = torch.zeros(n, dtype=torch.bool, device=dev)
        self._s_seated = torch.zeros(n, dtype=torch.bool, device=dev)
        self._s_loaded = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: re-pose the rack (xy + yaw jitter) and the stand (xy jitter
        + free yaw), stow the tray on edge in the rack slot at a jittered depth, deal
        the three bowl BODIES over the three row slots by a fresh permutation, latch
        the target = the body dealt onto the FRONT slot, clear all monitors."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        z0 = c.surface_z

        torch.rand(m, device=dev)  # burn one draw (first post-seed draw is degenerate)

        # --- rack: jittered xy, small yaw jitter ---
        ryaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rack_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.rack_xy[0] + (torch.rand(m, device=dev) * 2 - 1) * c.rack_jitter
        st[:, 1] = c.rack_xy[1] + (torch.rand(m, device=dev) * 2 - 1) * c.rack_jitter
        st[:, 2] = z0
        st[:, 3] = torch.cos(ryaw / 2)
        st[:, 6] = torch.sin(ryaw / 2)
        rack_pos = st[:, 0:3].clone()
        st[:, 0:3] += origin
        self.rack.write_root_state_to_sim(st, env_ids)

        # --- stand: jittered xy, free yaw ---
        if c.stand_free_yaw:
            syaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        else:
            syaw = torch.zeros(m, device=dev)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.stand_xy[0] + (torch.rand(m, device=dev) * 2 - 1) * c.stand_jitter
        st[:, 1] = c.stand_xy[1] + (torch.rand(m, device=dev) * 2 - 1) * c.stand_jitter
        st[:, 2] = z0
        st[:, 3] = torch.cos(syaw / 2)
        st[:, 6] = torch.sin(syaw / 2)
        st[:, 0:3] += origin
        self.stand.write_root_state_to_sim(st, env_ids)

        # --- tray: standing on edge in the slot at a jittered stow depth ---
        # q = qz(rack_yaw) * qx(90 deg): tray local +x stays along the channel,
        # local +z (the board normal) turns horizontal.
        stow = c.stow_x0 + torch.rand(m, device=dev) * (c.stow_x1 - c.stow_x0)
        self._stow_x[env_ids] = stow
        cos_y, sin_y = torch.cos(ryaw), torch.sin(ryaw)
        tz = c.rack_floor_top + c.tray_side / 2 + 0.002
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = rack_pos[:, 0] + stow * cos_y
        st[:, 1] = rack_pos[:, 1] + stow * sin_y
        st[:, 2] = z0 + tz
        half = ryaw / 2
        r2 = math.sqrt(0.5)
        st[:, 3] = torch.cos(half) * r2
        st[:, 4] = torch.cos(half) * r2
        st[:, 5] = torch.sin(half) * r2
        st[:, 6] = torch.sin(half) * r2
        st[:, 0:3] += origin
        self.tray.write_root_state_to_sim(st, env_ids)

        # --- bowls: fresh permutation over row slots + jitter + free yaw ---
        if c.shuffle_bowls:
            perm = torch.rand(m, 3, device=dev).argsort(dim=1)  # slot index per body
        else:
            perm = torch.arange(3, device=dev).unsqueeze(0).expand(m, 3).contiguous()
        self.slot_of[env_ids] = perm
        self.target[env_ids] = (perm == 0).float().argmax(dim=1)
        xs = torch.tensor(c.slot_xs, device=dev)
        for i in range(3):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = xs[perm[:, i]] + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            st[:, 1] = c.slot_y + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            st[:, 2] = z0 + 0.002
            half = (torch.rand(m, device=dev) * 2 - 1) * math.pi / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            self.bowls[i].write_root_state_to_sim(st, env_ids)

        # --- clear monitors ---
        self._warmup[env_ids] = c.warmup_steps
        self._still_tray[env_ids] = 0
        self._still_bowl[env_ids] = 0
        self._prog[env_ids] = 0.0
        self._s_out[env_ids] = False
        self._s_seated[env_ids] = False
        self._s_loaded[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "rack": self.rack.data.root_state_w[env_ids].clone(),
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "bowls": [b.data.root_state_w[env_ids].clone() for b in self.bowls],
            "target": self.target[env_ids].clone(),
            "slot_of": self.slot_of[env_ids].clone(),
            "stow_x": self._stow_x[env_ids].clone(),
            "warmup": self._warmup[env_ids].clone(),
            "still_tray": self._still_tray[env_ids].clone(),
            "still_bowl": self._still_bowl[env_ids].clone(),
            "prog": self._prog[env_ids].clone(),
            "s_out": self._s_out[env_ids].clone(),
            "s_seated": self._s_seated[env_ids].clone(),
            "s_loaded": self._s_loaded[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.rack.write_root_state_to_sim(state["rack"], env_ids)
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        for b, s in zip(self.bowls, state["bowls"]):
            b.write_root_state_to_sim(s, env_ids)
        self.target[env_ids] = state["target"]
        self.slot_of[env_ids] = state["slot_of"]
        self._stow_x[env_ids] = state["stow_x"]
        self._warmup[env_ids] = state["warmup"]
        self._still_tray[env_ids] = state["still_tray"]
        self._still_bowl[env_ids] = state["still_bowl"]
        self._prog[env_ids] = state["prog"]
        self._s_out[env_ids] = state["s_out"]
        self._s_seated[env_ids] = state["s_seated"]
        self._s_loaded[env_ids] = state["s_loaded"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On a kitchen counter stand, from back to front: a tall wooden dish RACK, "
            f"a blue-grey serving STAND, and a row of three IDENTICAL black open bowls "
            f"({c.bowl_outer * 1000:.0f} mm square, {c.bowl_height * 1000:.0f} mm tall). "
            f"A square white serving TRAY ({c.tray_side * 1000:.0f} x "
            f"{c.tray_side * 1000:.0f} x {c.tray_thick * 1000:.0f} mm) is STOWED ON EDGE "
            f"inside the rack's slot. The rack is a narrow channel "
            f"({c.gap * 1000:.0f} mm wide) between two 200 mm walls with a roof rail "
            f"over its rear, so the stowed tray CANNOT be lifted out — it can only "
            f"SLIDE along the channel and out through the rack's open mouth (the end "
            f"with the low fences, facing the front counter edge). The stand carries a "
            f"shallow square WELL ({c.well_in * 1000:.0f} mm inner, walls "
            f"{c.rim_h * 1000:.0f} mm high) which is EMPTY at reset. The stand's "
            f"position and heading, the rack's exact pose, the tray's stow depth, and "
            f"the order of the bowls in the row are randomized every episode.\n"
            f"Goal, in the only order physics allows: (1) slide the tray out of the "
            f"rack, (2) lay it FLAT inside the stand's well so it sits level on the "
            f"well floor (its edges must be aligned with the well — the well is only "
            f"{(c.well_in - c.tray_side) * 500:.0f} mm larger per side, and a "
            f"misaligned or tilted tray rests on the rim and does not count), then "
            f"(3) place the bowl at the FRONT of the row — the end nearest the "
            f"counter's front edge, the same side the rack mouth faces — upright and "
            f"centered on the tray (within {c.bowl_xy_tol * 1000:.0f} mm). Only the "
            f"front bowl counts, and the other two bowls must stay OFF the tray. "
            f"Putting the bowl into the bare well first is a dead end: the tray "
            f"cannot seat on top of it, so the bowl would have to be removed again."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the white tray out of the dish rack's slot, lay it flat inside "
            "the square well of the serving stand so it sits level on the well "
            "floor, then place the front black bowl (the one nearest you) upright "
            "and centered on the tray. Keep the other two bowls off the tray."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    @staticmethod
    def _yaw_of(q: torch.Tensor) -> torch.Tensor:
        """(..., 4) wxyz quat -> (...,) yaw about world z."""
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def tray_local_x(self) -> torch.Tensor:
        """(N,) tray center x in the RACK frame (mouth at 0, +x = out)."""
        ryaw = self._yaw_of(self.rack.data.root_quat_w)
        d = self.tray.data.root_pos_w - self.rack.data.root_pos_w
        return d[:, 0] * torch.cos(ryaw) + d[:, 1] * torch.sin(ryaw)

    def tray_rack_dist(self) -> torch.Tensor:
        """(N,) horizontal distance tray center to rack origin."""
        d = self.tray.data.root_pos_w - self.rack.data.root_pos_w
        return d[:, :2].norm(dim=-1)

    def _up_z(self, body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        return quat_apply(body.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)

    def tray_out(self) -> torch.Tensor:
        """(N,) bool: the tray is fully clear of the rack channel."""
        return (self.tray_local_x() > self.cfg.out_x) | (self.tray_rack_dist() > 0.30)

    def tray_seated(self) -> torch.Tensor:
        """(N,) bool, geometric: tray center over the well center, at seated height
        (center 15 mm above the counter: on the well FLOOR, not the rim, not a bowl),
        level within ~2.5 deg."""
        c = self.cfg
        d = self.tray.data.root_pos_w - self.stand.data.root_pos_w
        xy_ok = d[:, :2].norm(dim=-1) < c.seat_xy_tol
        seat_z = c.base_h + c.tray_thick / 2
        z_ok = (d[:, 2] - seat_z).abs() < c.seat_dz_tol
        return xy_ok & z_ok & (self._up_z(self.tray) >= c.seat_up_cos)

    def _bowl_rel_tray(self) -> tuple[torch.Tensor, torch.Tensor]:
        """((N,3) bowl-center xy distance to the tray axis, (N,3) bowl bottom minus
        tray top). Bowl origins are at their bottom centers."""
        pos = torch.stack([b.data.root_pos_w for b in self.bowls], dim=1)
        tp = self.tray.data.root_pos_w[:, None, :]
        d_xy = (pos[:, :, :2] - tp[:, :, :2]).norm(dim=-1)
        dz = pos[:, :, 2] - (tp[:, :, 2] + self.cfg.tray_thick / 2)
        return d_xy, dz

    def on_tray(self) -> torch.Tensor:
        """(N, 3) bool, geometric: bowl upright on the tray within `bowl_xy_tol` of
        its center."""
        c = self.cfg
        d_xy, dz = self._bowl_rel_tray()
        up = torch.stack([torch.zeros_like(d_xy[:, 0]) for _ in range(3)], dim=1)
        for i, b in enumerate(self.bowls):
            up[:, i] = self._up_z(b)
        return (d_xy < c.bowl_xy_tol) & (dz > c.bowl_dz_lo) & (dz < c.bowl_dz_hi) \
            & (up >= c.bowl_up_cos)

    def decoys_clear(self) -> torch.Tensor:
        """(N,) bool: neither decoy bowl intrudes on the tray (exclusion cylinder
        around the tray axis in the tray's z band)."""
        c = self.cfg
        d_xy, dz = self._bowl_rel_tray()
        intrude = (d_xy < c.decoy_xy) & (dz > c.decoy_dz_lo) & (dz < c.decoy_dz_hi)
        n = self.env.num_envs
        is_tgt = torch.zeros(n, 3, dtype=torch.bool, device=self.env.device)
        is_tgt.scatter_(1, self.target.unsqueeze(1), True)
        return ~(intrude & ~is_tgt).any(dim=1)

    def _finite(self) -> torch.Tensor:
        p = torch.cat([self.tray.data.root_pos_w]
                      + [b.data.root_pos_w for b in self.bowls], dim=-1)
        return torch.isfinite(p).all(dim=-1)

    def _gather_tgt(self, per_bowl: torch.Tensor) -> torch.Tensor:
        return per_bowl.gather(1, self.target.unsqueeze(1)).squeeze(1)

    # ----- trajectory monitors -------------------------------------------------------------------
    def _update_monitors(self) -> None:
        """Called once per physics step (post_step). Streak counters, running-max
        extraction progress, stage latches."""
        c = self.cfg
        lin_t = self.tray.data.root_lin_vel_w.norm(dim=-1)
        ang_t = self.tray.data.root_ang_vel_w.norm(dim=-1)
        still_t = (lin_t < c.settle_speed) & (ang_t < c.settle_ang)
        self._still_tray = torch.where(still_t, self._still_tray + 1,
                                       torch.zeros_like(self._still_tray))
        lin_b = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.bowls], dim=1)
        ang_b = torch.stack([b.data.root_ang_vel_w.norm(dim=-1) for b in self.bowls], dim=1)
        still_b = (lin_b < c.settle_speed) & (ang_b < c.settle_ang)
        self._still_bowl = torch.where(still_b, self._still_bowl + 1,
                                       torch.zeros_like(self._still_bowl))

        warm = self._warmup > 0
        self._warmup = (self._warmup - 1).clamp(min=0)
        live = ~warm

        travel = self.tray_local_x() - self._stow_x
        span = (c.out_x - self._stow_x).clamp(min=1e-6)
        prog = ((travel - c.prog_deadband) / span).clamp(0.0, 1.0)
        prog = torch.where(live, prog, torch.zeros_like(prog))
        self._prog = torch.maximum(self._prog, prog)

        self._s_out |= live & self.tray_out()
        seated = self.tray_seated()
        self._s_seated |= live & seated
        self._s_loaded |= live & seated & self._gather_tgt(self.on_tray())

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_monitors()

    # ----- rubric --------------------------------------------------------------------------------
    def at_rest(self) -> torch.Tensor:
        """(N,) bool: tray AND target bowl still for `still_steps` consecutive steps."""
        c = self.cfg
        return (self._still_tray >= c.still_steps) \
            & (self._gather_tgt(self._still_bowl) >= c.still_steps)

    def success(self) -> torch.Tensor:
        """(N,) bool: tray seated flat on the well floor, the TARGET bowl upright and
        centered on it, both decoys off the tray, everything at rest. All geometric
        clauses are live physical readback (the task is reversible; only the target
        identity is a latched episode fact)."""
        return self.tray_seated() & self._gather_tgt(self.on_tray()) \
            & self.decoys_clear() & self.at_rest() & self._finite() \
            & (self._warmup == 0)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: running-max extraction progress up to 0.30, then
        latched stages (0.35 out of the rack, 0.60 seated, 0.80 loaded; monotone, ~0
        for the null policy), exactly 1.0 iff success() holds live."""
        c = self.cfg
        s = c.w_prog * self._prog
        s = torch.where(self._s_out, torch.full_like(s, c.w_out).maximum(s), s)
        s = torch.where(self._s_seated, torch.full_like(s, c.w_seated), s)
        s = torch.where(self._s_loaded, torch.full_like(s, c.w_loaded), s)
        return torch.where(self.success(), torch.ones_like(s), s)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="rack_tray_serve", robot="null"))
