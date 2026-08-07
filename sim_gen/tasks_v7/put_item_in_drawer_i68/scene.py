"""BallastLiftScene — load the red cube into the balance-beam bucket, then hoist it to
the delivery shroud by counterweighting the pan with BOTH steel cylinders.

Derived from rlbench/put_item_in_drawer (grasp the drawer handle, pull the prismatic
joint open, lower the item into the opened drawer from above), but the receptacle is a
MASS-ACTUATED MECHANISM the robot never opens, pulls or pushes: the "drawer" is an
open-top BUCKET fixed to one arm of a balance beam (a revolute seesaw, limits +/-20
deg); the other arm carries a walled BALLAST PAN. The beam is bucket-heavy, so at rest
the bucket hangs LOW with its mouth open to the sky — the item can simply be dropped
in. The goal position, however, is the TOP stop, where the bucket docks 12 mm under a
fixed delivery SHROUD plate ("the drawer pushed home"): the only way to get it there is
to load enough counterweight into the pan. Two DARK-GREY STEEL cylinders (600 g each)
are provided along with a same-shaped WHITE FOAM decoy (15 g): by the torque budget,
BOTH steels are required — one steel (even plus the foam) cannot overcome the beam's
built-in bucket-side bias plus the loaded cube, while two steels flip it decisively.
The 12 mm shroud gap is smaller than the 45 mm cube, so a raised bucket CANNOT be
loaded: deposit must precede the lift (recoverable — unloading the pan lets the bucket
sink back down). The robot therefore actuates the fixture purely by WHERE IT PLACES
MASS; no part of the beam is ever handled.

Assets are fully procedural (tilt-bin-pattern compound spawners; child colliders of one
body never self-collide):
  - pedestal: KINEMATIC compound at a FIXED pose (never re-posed: joints anchored to a
    teleported kinematic body0 stay world-fixed) — ground slab, pivot column, shroud
    riser + arm, and the SHROUD PLATE tilted +20 deg so it parallels the raised
    bucket's mouth plane at a uniform 12 mm gap.
  - beam: DYNAMIC compound — center bar, bucket (open-top, interior 110 x 110 x 70 mm)
    at local +y, ballast pan (interior 150(x) x 60(y) x 75 mm) at local -y. Origin ON
    the pivot axis, so reset re-poses it as a pure joint-coordinate rotation. Explicit
    MassAPI mass AND centerOfMass (+y bias: spawn-authored compounds keep CoM at the
    body origin otherwise). Sleep/stabilization zeroed (gravity-driven, judged still).
  - pivot: bind-time UsdPhysics.RevoluteJoint pedestal->beam about +X, limits
    [-20, +20] deg, joint-pair collision disabled. +theta = bucket up.
  - cube (red, 45 mm, 120 g), steel_a / steel_b (dark-grey cylinders, 32 mm dia x
    70 mm, 600 g), foam (white, same shape, 15 g): DYNAMIC, scattered on the floor.

Torque budget (kg*m about the pivot, worst-case world arms at the -20 deg rest):
  keep-down  = beam bias 1.7 kg x 103 mm = 0.176  (+ cube 0.030..0.037 when loaded)
  one steel  <= 0.6 x 277 mm (outer wall) = 0.166 -> stays down even EMPTY (0.1 N*m
                                                    margin; 0.4 N*m loaded; foam +0.004)
  two steels >= 1.2 x 250 mm (inner wall) = 0.300 -> flips     (0.9 N*m margin loaded)
The pan interior is deliberately NARROW along the beam (60 mm vs 32 mm cylinders) so
ballast arms cannot be gamed; the walls keep weights captive through the swing.

Per-episode randomization (readback-verifiable): the four loose objects are dealt onto
the four floor slots by a fresh PERMUTATION (identity must be read from color/shape,
not memorized), each with xy jitter and free yaw, and the beam starts at a random
partially-raised angle in [-20, -8] deg that falls back to the rest stop in the first
settle (visual variety, no credit).

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.25 * loaded    — cube ever inside the bucket (latched bool)
  0.25 * ballast   — latched max of (steel cylinders currently in the pan)/2
  0.35 * lift      — latched max of (theta - rest)/(top - rest), counted only while
                     the cube is CURRENTLY inside the bucket (raising an empty bucket
                     earns nothing)
  1.0 iff success() — cube inside the bucket, beam at the top stop (theta >= 17 deg),
                     BOTH steels resting in the pan, everything still. Cap 0.85.

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
_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             orient=None) -> None:
    from pxr import Gf, UsdGeom

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


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
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


def _spawn_pedestal(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC pedestal at a fixed pose: ground slab, pivot column, shroud riser +
    arm, and the shroud plate tilted +cfg.tilt_deg about +x (parallel to the raised
    bucket mouth). Origin ON the pivot axis."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/slab",
             center=(0.0, 0.0, c.slab_z), size=(0.28, 0.56, 0.02),
             color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/column",
             center=(0.0, 0.0, (c.slab_z + 0.01 - 0.030) / 2 - 0.0),
             size=(0.055, 0.050, -(c.slab_z + 0.01) - 0.030 + 0.0),
             color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/riser",
             center=(c.riser_x, c.plate_y, (c.slab_z + 0.01 + c.riser_top) / 2),
             size=(0.04, 0.04, c.riser_top - (c.slab_z + 0.01)),
             color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/arm",
             center=(c.riser_x / 2 - 0.02, c.plate_y, c.riser_top - 0.01),
             size=(c.riser_x + 0.08, 0.04, 0.02),
             color=c.color, collide=collide)
    half = math.radians(c.tilt_deg) / 2
    _add_box(stage, f"{prim_path}/plate",
             center=(0.0, c.plate_y, c.plate_z),
             size=(c.plate_sx, c.plate_sy, c.plate_t),
             color=c.plate_color, collide=collide,
             orient=(math.cos(half), math.sin(half), 0.0, 0.0))
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC balance beam: center bar, bucket at +y, ballast pan at -y. Origin on
    the pivot axis. Explicit mass AND centerOfMass (+y bias holds the bucket down);
    sleep/stabilization zeroed (gravity-driven, judged for stillness)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass_props.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, float(cfg.com_y), 0.0))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.02)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/bar",
             center=(0.0, 0.0, 0.0), size=(0.036, 0.64, 0.024),
             color=c.bar_color, collide=collide)
    # bucket: floor + 4 walls, interior bk_w x bk_w x bk_h, floor top at z = fl_top
    t, fl_top = c.wall_t, c.floor_top
    bo = c.bk_w + 2 * t
    _add_box(stage, f"{prim_path}/bk_floor",
             center=(0.0, c.arm_len, fl_top - t / 2), size=(bo, bo, t),
             color=c.bucket_color, collide=collide)
    for sgn, nm in ((1.0, "bk_wall_o"), (-1.0, "bk_wall_i")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(0.0, c.arm_len + sgn * (c.bk_w / 2 + t / 2), fl_top + c.bk_h / 2),
                 size=(bo, t, c.bk_h), color=c.bucket_color, collide=collide)
    for sgn, nm in ((1.0, "bk_wall_l"), (-1.0, "bk_wall_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(sgn * (c.bk_w / 2 + t / 2), c.arm_len, fl_top + c.bk_h / 2),
                 size=(t, c.bk_w, c.bk_h), color=c.bucket_color, collide=collide)
    # pan: floor + 4 walls, interior pan_x x pan_y x pan_h
    po_x, po_y = c.pan_x + 2 * t, c.pan_y + 2 * t
    _add_box(stage, f"{prim_path}/pan_floor",
             center=(0.0, -c.arm_len, fl_top - t / 2), size=(po_x, po_y, t),
             color=c.pan_color, collide=collide)
    for sgn, nm in ((1.0, "pan_wall_i"), (-1.0, "pan_wall_o")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(0.0, -c.arm_len + sgn * (c.pan_y / 2 + t / 2), fl_top + c.pan_h / 2),
                 size=(po_x, t, c.pan_h), color=c.pan_color, collide=collide)
    for sgn, nm in ((1.0, "pan_wall_l"), (-1.0, "pan_wall_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(sgn * (c.pan_x / 2 + t / 2), -c.arm_len, fl_top + c.pan_h / 2),
                 size=(t, c.pan_y, c.pan_h), color=c.pan_color, collide=collide)
    return root


def _spawn_cyl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC cylinder along local +z (a ballast weight or the foam decoy)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.30)  # rollers: settle instead of orbiting the pan
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    r, h = cfg.radius, cfg.height
    cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/body")
    cyl.CreateRadiusAttr(r)
    cyl.CreateHeightAttr(h)
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    _make_collide(cfg.contact_offset)(cyl.GetPrim())
    return root


def _spawn_cube(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC cube (the item)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    _add_box(stage, f"{prim_path}/body",
             center=(0.0, 0.0, 0.0), size=(cfg.edge, cfg.edge, cfg.edge),
             color=cfg.color, collide=_make_collide(cfg.contact_offset))
    return root


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pedestal" not in _SPAWNER_CACHE:

        @configclass
        class PedestalSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pedestal)
            slab_z: float = -0.33
            riser_x: float = 0.12
            riser_top: float = 0.245
            plate_y: float = 0.2261
            plate_z: float = 0.1973
            plate_sx: float = 0.20
            plate_sy: float = 0.19
            plate_t: float = 0.012
            tilt_deg: float = 20.0
            color: tuple = (0.45, 0.45, 0.48)
            plate_color: tuple = (0.85, 0.55, 0.10)
            contact_offset: float = 0.002

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            arm_len: float = 0.28
            wall_t: float = 0.008
            floor_top: float = 0.020
            bk_w: float = 0.110
            bk_h: float = 0.070
            pan_x: float = 0.150
            pan_y: float = 0.060
            pan_h: float = 0.075
            com_y: float = 0.11
            bar_color: tuple = (0.35, 0.40, 0.50)
            bucket_color: tuple = (0.82, 0.68, 0.30)
            pan_color: tuple = (0.15, 0.25, 0.55)
            contact_offset: float = 0.002

        @configclass
        class CylSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cyl)
            radius: float = 0.016
            height: float = 0.070
            color: tuple = (0.5, 0.5, 0.5)
            contact_offset: float = 0.002

        @configclass
        class CubeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cube)
            edge: float = 0.045
            color: tuple = (0.8, 0.1, 0.1)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(pedestal=PedestalSpawnerCfg, beam=BeamSpawnerCfg,
                              cyl=CylSpawnerCfg, cube=CubeSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BallastLiftSceneCfg(BaseCfg):
    """Config for `BallastLiftScene`. The quantity threshold is metric: one steel
    cylinder's best-case pan torque (0.166 kg*m, resting against the outer wall) is
    below the beam's bucket-side bias (0.176 kg*m) even before the cube is loaded,
    while two steels' worst case (0.300 kg*m) exceeds bias + cube (0.211 kg*m) by
    ~0.9 N*m."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    lift_min_deg: float = tunable(17.0)  # beam counts as at the top stop above this
    settle_speed: float = tunable(0.05)  # max |lin vel| of cube/steels when judging (m/s)
    settle_omega: float = tunable(0.30)  # max |ang vel| of the beam when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    slot_jitter: float = tunable(0.03)  # per-object spawn xy jitter (+/- m)
    shuffle_slots: bool = tunable(True)  # deal objects onto slots by fresh permutation
    ajar_max_deg: float = tunable(12.0)  # initial beam angle in [rest, rest + this]

    # --- tunable: mechanism plant ----------------------------------------------------------------
    pivot_damp: float = tunable(0.9)  # viscous pivot damping (N*m*s/rad), post_step-owned

    # --- info: layout (single Franka base at the origin, facing +x) ------------------------------
    pivot_pos: tuple = info((0.50, 0.0, 0.34))  # the beam pivot (revolute about +x)
    slots: tuple = info(((0.30, -0.24), (0.34, -0.08), (0.34, 0.08), (0.30, 0.24)))

    # --- info: beam / bucket / pan structure (mirrors BeamSpawnerCfg) ---------------------------
    arm_len: float = info(0.28)  # bucket centre +y, pan centre -y
    limit_deg: float = info(20.0)  # revolute limits +/- this; rest = -limit (bucket down)
    wall_t: float = info(0.008)
    floor_top: float = info(0.020)  # interior floor plane (beam local z)
    bk_w: float = info(0.110)  # bucket interior width (x and y)
    bk_h: float = info(0.070)  # bucket interior height
    pan_x: float = info(0.150)  # pan interior along x (across the beam)
    pan_y: float = info(0.060)  # pan interior along y (narrow: bounds the ballast arm)
    pan_h: float = info(0.075)
    beam_mass: float = info(1.7)
    com_y: float = info(0.11)  # explicit CoM bias toward the bucket

    # --- info: shroud (parallel to the raised mouth, 12 mm gap) ---------------------------------
    shroud_gap: float = info(0.012)
    plate_sx: float = info(0.20)
    plate_sy: float = info(0.19)
    plate_t: float = info(0.012)

    # --- info: loose objects ---------------------------------------------------------------------
    cube_edge: float = info(0.045)
    cube_mass: float = info(0.12)
    cube_color: tuple = info((0.80, 0.10, 0.10))  # red
    steel_r: float = info(0.016)
    steel_h: float = info(0.070)
    steel_mass: float = info(0.60)
    steel_color: tuple = info((0.25, 0.26, 0.30))  # dark metallic grey
    foam_mass: float = info(0.015)
    foam_color: tuple = info((0.93, 0.93, 0.90))  # white

    contact_offset: float = info(0.002)
    # rubric weights (0.25 + 0.25 + 0.35 = 0.85 = the non-success cap)
    w_load: float = info(0.25)
    w_ballast: float = info(0.25)
    w_lift: float = info(0.35)

    # Derived (filled in __post_init__): interior boxes in the BEAM body frame.
    bucket_lo: tuple = field(default=None, init=False)
    bucket_hi: tuple = field(default=None, init=False)
    pan_lo: tuple = field(default=None, init=False)
    pan_hi: tuple = field(default=None, init=False)

    def __post_init__(self) -> None:
        m = 0.005  # containment margin beyond the interior faces
        a, ft = self.arm_len, self.floor_top
        self.bucket_lo = (-self.bk_w / 2 - m, a - self.bk_w / 2 - m, ft - m)
        self.bucket_hi = (self.bk_w / 2 + m, a + self.bk_w / 2 + m, ft + self.bk_h + m)
        self.pan_lo = (-self.pan_x / 2 - m, -a - self.pan_y / 2 - m, ft - m)
        # z cap at wall-top + margin: a steel PERCHED ON THE RIM does not count as
        # ballast in the pan (stacked/wedged pairs inside top out near z 0.052).
        self.pan_hi = (self.pan_x / 2 + m, -a + self.pan_y / 2 + m, ft + self.pan_h + m)

    @property
    def mouth_local(self) -> tuple:
        """Bucket mouth centre in the beam body frame."""
        return (0.0, self.arm_len, self.floor_top + self.bk_h)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("ballast_lift")
class BallastLiftScene(BaseScene):
    cfg: BallastLiftSceneCfg

    def __init__(self, cfg: BallastLiftSceneCfg | None = None) -> None:
        super().__init__(cfg or BallastLiftSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        pedestal = sp["pedestal"](
            mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            contact_offset=c.contact_offset,
        )
        beam = sp["beam"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.beam_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            arm_len=c.arm_len, wall_t=c.wall_t, floor_top=c.floor_top,
            bk_w=c.bk_w, bk_h=c.bk_h, pan_x=c.pan_x, pan_y=c.pan_y, pan_h=c.pan_h,
            com_y=c.com_y, contact_offset=c.contact_offset,
        )

        def cyl(mass, color):
            return sp["cyl"](
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                radius=c.steel_r, height=c.steel_h, color=color,
                contact_offset=c.contact_offset,
            )

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
            "pedestal": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pedestal",
                spawn=pedestal,
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.pivot_pos),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=beam,
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.pivot_pos),
            ),
            "cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                spawn=sp["cube"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    edge=c.cube_edge, color=c.cube_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slots[0][0], c.slots[0][1], c.cube_edge / 2 + 0.002)),
            ),
        }
        for i, (name, mass, color) in enumerate((
                ("steel_a", c.steel_mass, c.steel_color),
                ("steel_b", c.steel_mass, c.steel_color),
                ("foam", c.foam_mass, c.foam_color))):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.capitalize(),
                spawn=cyl(mass, color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slots[i + 1][0], c.slots[i + 1][1], c.steel_h / 2 + 0.002)),
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
        self.pedestal: RigidObject = env.iscene["pedestal"]
        self.beam: RigidObject = env.iscene["beam"]
        self.cube: RigidObject = env.iscene["cube"]
        self.steel_a: RigidObject = env.iscene["steel_a"]
        self.steel_b: RigidObject = env.iscene["steel_b"]
        self.foam: RigidObject = env.iscene["foam"]
        self.env_origins = env.iscene.env_origins
        self._author_pivot()
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._load = torch.zeros(n, dtype=torch.bool, device=dev)  # cube ever in bucket
        self._ballast_max = torch.zeros(n, device=dev)  # steels-in-pan/2, running max
        self._lift_max = torch.zeros(n, device=dev)  # lift progress while loaded, max

    def _author_pivot(self) -> None:
        """Per env: a +X revolute joint pedestal->beam at the pivot, limits
        [-limit, +limit] deg, joint-pair collision disabled. The pedestal is kinematic
        and NEVER re-posed, so the world-fixed anchor stays correct."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/beam_pivot")
            j.CreateBody0Rel().SetTargets([f"{base}/Pedestal"])
            j.CreateBody1Rel().SetTargets([f"{base}/Beam"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-float(c.limit_deg))
            j.CreateUpperLimitAttr(float(c.limit_deg))

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pedestal re-asserted at its FIXED pose, beam re-posed to a
        random partially-raised angle (pure joint-coordinate rotation about the
        unchanged pivot; falls to the rest stop in the first settle), and the four
        loose objects dealt onto the four slots by a fresh permutation + xy jitter +
        free yaw. Latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = torch.tensor(c.pivot_pos, device=dev)
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.pedestal.write_root_state_to_sim(st, env_ids)

        theta0 = -math.radians(c.limit_deg) \
            + torch.rand(m, device=dev) * math.radians(c.ajar_max_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = torch.tensor(c.pivot_pos, device=dev)
        st[:, 3] = torch.cos(theta0 / 2)
        st[:, 4] = torch.sin(theta0 / 2)
        st[:, 0:3] += origin
        self.beam.write_root_state_to_sim(st, env_ids)

        bodies = [(self.cube, c.cube_edge / 2), (self.steel_a, c.steel_h / 2),
                  (self.steel_b, c.steel_h / 2), (self.foam, c.steel_h / 2)]
        if c.shuffle_slots:
            perm = torch.rand(m, 4, device=dev).argsort(dim=1)  # slot index per body
        else:
            perm = torch.arange(4, device=dev).expand(m, 4)
        slots = torch.tensor(c.slots, device=dev)  # (4, 2)
        for b, (body, half_h) in enumerate(bodies):
            xy = slots[perm[:, b]] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            half = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = half_h + 0.002
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        self._load[env_ids] = False
        self._ballast_max[env_ids] = 0.0
        self._lift_max[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {nm: getattr(self, nm).data.root_state_w[env_ids].clone()
               for nm in ("pedestal", "beam", "cube", "steel_a", "steel_b", "foam")}
        out["load"] = self._load[env_ids].clone()
        out["ballast_max"] = self._ballast_max[env_ids].clone()
        out["lift_max"] = self._lift_max[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm in ("pedestal", "beam", "cube", "steel_a", "steel_b", "foam"):
            getattr(self, nm).write_root_state_to_sim(state[nm], env_ids)
        self._load[env_ids] = state["load"]
        self._ballast_max[env_ids] = state["ballast_max"]
        self._lift_max[env_ids] = state["lift_max"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A grey pedestal carries a BALANCE BEAM (a seesaw pivoting about a "
            f"horizontal axis, swing limited to +/-{c.limit_deg:.0f} deg). One arm ends "
            f"in an open-top YELLOW BUCKET (interior {c.bk_w * 100:.0f} x "
            f"{c.bk_w * 100:.0f} x {c.bk_h * 100:.0f} cm — this is the delivery drawer); "
            f"the other arm ends in a walled BLUE BALLAST PAN. The beam is bucket-heavy: "
            f"at rest the bucket hangs LOW with its mouth open to the sky, and the pan "
            f"rides high. Above the bucket's RAISED position an ORANGE SHROUD PLATE is "
            f"fixed, parallel to the raised mouth at a {c.shroud_gap * 1000:.0f} mm gap "
            f"— when the bucket is up, nothing wider than {c.shroud_gap * 1000:.0f} mm "
            f"can pass into it, and the {c.cube_edge * 1000:.0f} mm cube cannot. "
            f"Scattered on the floor between you and the pedestal lie four objects "
            f"whose positions shuffle every episode — identify them by color and shape: "
            f"a RED CUBE ({c.cube_edge * 1000:.0f} mm, the item to deliver), two "
            f"DARK-GREY STEEL cylinders ({2 * c.steel_r * 1000:.0f} mm dia x "
            f"{c.steel_h * 1000:.0f} mm, heavy — {c.steel_mass * 1000:.0f} g each) and "
            f"one WHITE FOAM cylinder of the same shape (almost weightless — "
            f"{c.foam_mass * 1000:.0f} g).\n"
            f"Goal: the red cube must ride INSIDE the bucket with the bucket hoisted to "
            f"its TOP stop (docked under the shroud) and everything at rest. The beam "
            f"is operated only by loading mass: put the cube into the low, open bucket "
            f"first, then set BOTH steel cylinders into the blue pan — one steel (or "
            f"one steel plus the foam) is NOT enough to tip the loaded beam; both "
            f"steels flip it decisively and the bucket rises to the stop. The foam "
            f"cylinder is a decoy: it helps nothing, and it may be left anywhere. If "
            f"you ballast the pan before loading the cube, the bucket rises empty and "
            f"the shroud blocks the mouth — take weight back out of the pan to lower "
            f"the bucket again. Success is judged on the settled physical state: cube "
            f"in the bucket, beam at the top stop (>= {c.lift_min_deg:.0f} deg), both "
            f"steels resting in the pan."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Drop the red cube into the yellow bucket while it hangs low, then load "
            "both dark-grey steel cylinders into the blue ballast pan so the beam tips "
            "and hoists the loaded bucket up to its top stop under the orange shroud. "
            "The white foam cylinder is too light to help."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def beam_angle(self) -> torch.Tensor:
        """(N,) pivot angle in rad (+ = bucket up; rest = -limit). The beam only ever
        rotates about the pivot +x axis, so the root quat is (cos t/2, sin t/2, 0, 0)."""
        q = self.beam.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 1], q[:, 0])

    def _in_box(self, body: RigidObject, lo: tuple, hi: tuple) -> torch.Tensor:
        """(N,) bool: body's origin inside a beam-body-frame box (frame-attached, so a
        tilted beam still contains its load)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - self.beam.data.root_pos_w
        loc = quat_apply_inverse(self.beam.data.root_quat_w, rel)
        lo_t = torch.tensor(lo, device=loc.device)
        hi_t = torch.tensor(hi, device=loc.device)
        return ((loc >= lo_t) & (loc <= hi_t)).all(dim=-1)

    def cube_in_bucket(self) -> torch.Tensor:
        return self._in_box(self.cube, self.cfg.bucket_lo, self.cfg.bucket_hi)

    def steels_in_pan(self) -> torch.Tensor:
        """(N,) count of steel cylinders currently inside the pan volume."""
        c = self.cfg
        return (self._in_box(self.steel_a, c.pan_lo, c.pan_hi).float()
                + self._in_box(self.steel_b, c.pan_lo, c.pan_hi).float())

    def _update_latches(self) -> None:
        c = self.cfg
        inside = self.cube_in_bucket()
        self._load |= inside
        ballast = self.steels_in_pan() / 2.0
        ballast = torch.nan_to_num(ballast, nan=0.0, posinf=0.0, neginf=0.0)
        self._ballast_max = torch.maximum(self._ballast_max, ballast)
        rest = -math.radians(c.limit_deg)
        top = math.radians(c.limit_deg)
        prog = ((self.beam_angle() - rest) / (top - rest)).clamp(0.0, 1.0)
        prog = torch.nan_to_num(prog * inside.float(), nan=0.0, posinf=0.0, neginf=0.0)
        self._lift_max = torch.maximum(self._lift_max, prog)

    # ----- step-coupled mechanics (every substep) --------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Pivot plant: viscous damping about the pivot axis (+x; torque along the
        rotation axis is invariant to the wrench frame-drag quirk), so the ballast
        flip is a stately swing instead of a slam. Owns the beam's wrench slot. Then
        latch rubric progress."""
        n = self.env.num_envs
        w_x = self.beam.data.root_ang_vel_w[:, 0]
        torque = torch.zeros(n, 1, 3, device=self.env.device)
        torque[:, 0, 0] = -self.cfg.pivot_damp * w_x
        self.beam.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=self.env.device), torque)
        self._update_latches()

    def settled(self) -> torch.Tensor:
        c = self.cfg
        beam_still = self.beam.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega
        cube_still = self.cube.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        a_still = self.steel_a.data.root_lin_vel_w.norm(dim=-1) < 2 * c.settle_speed
        b_still = self.steel_b.data.root_lin_vel_w.norm(dim=-1) < 2 * c.settle_speed
        return beam_still & cube_still & a_still & b_still

    def success(self) -> torch.Tensor:
        """(N,) bool: cube inside the bucket, beam at the top stop, BOTH steels resting
        in the pan, everything still. Physical outcomes only."""
        c = self.cfg
        self._update_latches()
        up = self.beam_angle() >= math.radians(c.lift_min_deg)
        return (self.cube_in_bucket() & up & (self.steels_in_pan() >= 2.0)
                & self.settled())

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25*loaded + 0.25*ballast + 0.35*lift-while-loaded —
        all latched, ~0 for doing nothing, capped 0.85 — and exactly 1.0 iff success()
        holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_load * self._load.float() + c.w_ballast * self._ballast_max
                + c.w_lift * self._lift_max).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="ballast_lift", robot="null"))
