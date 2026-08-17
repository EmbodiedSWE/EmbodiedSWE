"""TimerFlipDockScene — pick up a marble-bearing two-chamber TIMER capsule that stands
collar-end-down on the ground, turn it fully UPSIDE-DOWN in the air, and dock it foot-end
first into a snug square well until its collar flange seats on the amber rim (seed:
pick_place/approach_grasp_knife, strategically inverted).

The seed task servos a gripper to a knife on a table, closes the jaw, and lifts; success
is a few frames of stable grasp — pose of the carried object never matters. Here the
grasp-and-lift is only the zeroth step, worth 0.10; everything the rubric actually pays
for is what the seed never asks:

- REORIENTATION IS THE TASK: the capsule spawns standing collar-down and only docks fully
  inverted (its up-axis flipped past 150 deg). Carrying it upright — the seed's whole
  strategy — parks the run at 0.10 forever, and pressing it into the well upright is
  physically refused: the collar flange sits only 40 mm from the collar end, so the
  wrong-way capsule catches on the rim 60 mm proud of the seat.
- THE KEY IS ONE-WAY BY CONSTRUCTION: the 62 mm collar cannot enter the 50 mm well, and
  the 40 mm square tube cannot enter turned 45 deg (its 57 mm diagonal jams). Inserted
  foot-first the collar is the seating STOP (the foot cap hovers 10 mm above the well
  floor), so full depth is a hard, repeatable pose.
- A CAPTIVE INTERNAL PAYLOAD PROVES THE FLIP THROUGH PHYSICS: a 12 mm marble is sealed
  inside the capsule's collar-end chamber. Only real inversion makes it fall through the
  waist between the two internal ledges into the foot-end chamber, and success() checks
  the marble is IN the foot chamber live — teleport-faking the capsule pose without an
  honest inversion + settle leaves the marble on the wrong side.
- EXECUTION ORDER IS GEOMETRICALLY FORCED: flip strictly before insert (the well admits
  only the inverted capsule), and the marble crossing needs the flip to happen while the
  capsule is aloft or docked, not lying on its side (the waist passes the marble only
  along the tube axis).

Geometry honesty is asserted in `__post_init__`: jaw fit; collar > well (wrong-way and
seat stop are real); well > tube with clearance; tube diagonal > well (45 deg refused);
collar stops the capsule before the foot bottoms out; the waist aperture passes the
marble with margin while the ledges can never shelf it (wall clearance keeps the marble
center inboard of the ledge edges); perch and seat heights are 60 mm apart; the spawn
ring never overlaps the dock.

Rubric: score() = 0.10 lifted (both ends airborne, latched) + 0.15 inverted-aloft
(latched) + 0.15 foot-tip-in-well while inverted (latched) + 0.30 * latched max insertion
depth (rim -> seat, gated on the tip being in the well) + 0.15 marble-crossed-to-foot-
chamber while inverted (latched); exactly 1.0 iff success() = live seated pose (inverted
>= 150 deg, centered, collar on the rim) AND the marble live in the foot chamber AND
capsule + marble settled. Null policy scores 0; latched credit never evaporates.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- custom compound spawner (the timer capsule: caps + walls + waist ledges + collar) --------
_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_timer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC timer capsule at `prim_path`. Origin = tube centre; local +z runs
    from the collar end (-z, crimson cap) to the foot end (+z, white cap). Custom spawn
    funcs apply no cfg schemas, so mass (explicit CoM + diagonal inertia), damping, solver
    iterations and collision offsets are all authored here."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))

    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, float(cfg.com_z)))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in cfg.diag_inertia]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.10)
    px.CreateAngularDampingAttr(0.50)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    for tag, size, center, rgb in cfg.boxes:
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/{tag}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
        seg.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        pc = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        pc.CreateContactOffsetAttr(float(cfg.contact_offset))
        pc.CreateRestOffsetAttr(0.0)
    return root


def _timer_spawner_cfg(*, boxes: tuple, mass: float, com_z: float, diag_inertia: tuple,
                       contact_offset: float) -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "timer" not in _SPAWNER_CACHE:

        @configclass
        class TimerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_timer)
            boxes: tuple = ()
            mass: float = 0.30
            com_z: float = 0.0
            diag_inertia: tuple = (6.0e-4, 6.0e-4, 8.0e-5)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["timer"] = TimerSpawnerCfg

    return _SPAWNER_CACHE["timer"](
        boxes=boxes, mass=mass, com_z=com_z, diag_inertia=diag_inertia,
        contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TimerFlipDockSceneCfg(BaseCfg):
    """Config for `TimerFlipDockScene`. Dock geometry lives in the DOCK-LOCAL frame: origin
    at the well centre on the ground, z = world height. The dock is jittered and yawed per
    episode; the capsule spawns standing on a free-azimuth ring around it with free yaw,
    the captive marble jittered inside its collar-end chamber."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    lift_z: float = tunable(0.030)  # lifted latch: BOTH end tips above this height (m)
    invert_cos: float = tunable(0.85)  # inverted latch: -up_z >= this (~32 deg from flipped)
    tip_gate_cos: float = tunable(0.50)  # insertion gates need -up_z >= this (past 120 deg)
    seat_up_cos: float = tunable(0.90)  # success needs -up_z >= this (within ~26 deg)
    tip_xy: float = tunable(0.015)  # foot tip in-well: dock-local Chebyshev |xy| <= this
    seat_tol: float = tunable(0.008)  # success: |centre z - seat_z| <= this (m)
    center_tol: float = tunable(0.010)  # success: dock-local Chebyshev centre |xy| <= this
    mb_xy: float = tunable(0.014)  # marble-in-foot-chamber: body-local |x|,|y| <= this
    mb_z: tuple = tunable((0.020, 0.070))  # ... and body-local z inside this band (m)
    settle_lin: float = tunable(0.12)  # capsule |lin vel| < this at success (m/s)
    settle_ang: float = tunable(0.80)  # capsule |ang vel| < this at success (rad/s)
    settle_marble: float = tunable(0.20)  # marble |lin vel| < this at success (m/s)
    # (settle thresholds sit above the GPU phantom-velocity readback band)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    dock_pos: tuple = tunable((0.42, 0.0))  # nominal well centre on the ground (m)
    dock_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the dock per episode
    dock_yaw_deg: float = tunable(25.0)  # uniform +/- yaw of the dock per episode
    spawn_r: tuple = tunable((0.16, 0.24))  # capsule spawn ring radius around the well (m)
    marble_jitter: float = tunable(0.004)  # uniform +/- xy jitter of the marble in-chamber

    # --- info: capsule structure (body-local; +z collar end -> foot end) ---------------------
    tube_w: float = info(0.040)  # square tube outer width
    half_len: float = info(0.075)  # tube half length (overall 150 mm)
    wall_t: float = info(0.006)
    cap_t: float = info(0.006)
    ledge_w: float = info(0.003)  # waist ledge width (x, per side)
    ledge_t: float = info(0.006)  # waist ledge height (z)
    ledge_cx: float = info(0.0125)  # waist ledge centreline |x|
    collar_w: float = info(0.062)  # collar flange outer width (the one-way key)
    collar_t: float = info(0.010)
    collar_z: float = info(-0.030)  # collar centre, body-local (COLLAR-end half of the tube)
    timer_mass: float = info(0.30)
    timer_com_z: float = info(-0.006)  # collar pulls the CoM slightly toward -z
    timer_diag_inertia: tuple = info((6.0e-4, 6.0e-4, 8.0e-5))  # summed-boxes estimate
    marble_r: float = info(0.006)
    marble_mass: float = info(0.010)
    marble_home_z: float = info(-0.062)  # marble rest, body-local (collar-end chamber)

    # --- info: dock structure (dock-local; z = world height above the ground) ---------------
    dock_w: float = info(0.140)  # dock outer footprint (square)
    base_t: float = info(0.010)
    well_w: float = info(0.050)  # square well opening
    rim_z: float = info(0.120)  # rim top = well mouth height
    rim_t: float = info(0.008)  # amber rim band thickness (top of the walls)
    contact_offset: float = info(0.0015)
    color_body: tuple = info((0.16, 0.45, 0.80))  # capsule tube: blue
    color_foot: tuple = info((0.92, 0.92, 0.92))  # foot cap: white
    color_top: tuple = info((0.82, 0.12, 0.10))  # collar-end cap: crimson
    color_collar: tuple = info((0.62, 0.64, 0.68))  # collar flange: steel
    color_marble: tuple = info((0.95, 0.55, 0.05))  # marble: orange
    color_dock: tuple = info((0.28, 0.28, 0.32))  # dock body: charcoal
    color_rim: tuple = info((0.95, 0.62, 0.08))  # rim band: amber

    # Derived (filled in __post_init__).
    timer_boxes: tuple = field(default=None, init=False)
    dock_pieces: tuple = field(default=None, init=False)
    seat_z: float = field(default=None, init=False)  # capsule centre height at full seat
    perch_z: float = field(default=None, init=False)  # centre height of the wrong-way perch
    foot_seat_z: float = field(default=None, init=False)  # foot tip height at full seat
    tip_z_gate: float = field(default=None, init=False)  # tip below this counts as in-well
    spawn_z: float = field(default=None, init=False)  # standing spawn centre height
    aperture: float = field(default=None, init=False)  # waist opening between the ledges

    def __post_init__(self) -> None:
        tw, hl, wt, ct = self.tube_w, self.half_len, self.wall_t, self.cap_t
        iw = tw - 2 * wt  # interior width (0.028)
        wall_h = 2 * hl - 2 * ct  # tube walls between the caps
        body, foot, top, steel = (self.color_body, self.color_foot, self.color_top,
                                  self.color_collar)
        self.timer_boxes = (
            ("cap_top", (tw, tw, ct), (0.0, 0.0, -(hl - ct / 2)), top),
            ("cap_foot", (tw, tw, ct), (0.0, 0.0, hl - ct / 2), foot),
            ("wall_xn", (wt, tw, wall_h), (-(tw - wt) / 2, 0.0, 0.0), body),
            ("wall_xp", (wt, tw, wall_h), ((tw - wt) / 2, 0.0, 0.0), body),
            ("wall_yn", (iw, wt, wall_h), (0.0, -(tw - wt) / 2, 0.0), body),
            ("wall_yp", (iw, wt, wall_h), (0.0, (tw - wt) / 2, 0.0), body),
            ("ledge_xn", (self.ledge_w, iw, self.ledge_t), (-self.ledge_cx, 0.0, 0.0), body),
            ("ledge_xp", (self.ledge_w, iw, self.ledge_t), (self.ledge_cx, 0.0, 0.0), body),
            # collar = a RING of four flange boxes around the tube (never inside it: the
            # marble's path through the waist must stay clear)
            ("collar_xn", ((self.collar_w - tw) / 2 + 0.002, self.collar_w, self.collar_t),
             (-(tw / 2 + (self.collar_w - tw) / 4 - 0.001), 0.0, self.collar_z), steel),
            ("collar_xp", ((self.collar_w - tw) / 2 + 0.002, self.collar_w, self.collar_t),
             (tw / 2 + (self.collar_w - tw) / 4 - 0.001, 0.0, self.collar_z), steel),
            ("collar_yn", (tw, (self.collar_w - tw) / 2 + 0.002, self.collar_t),
             (0.0, -(tw / 2 + (self.collar_w - tw) / 4 - 0.001), self.collar_z), steel),
            ("collar_yp", (tw, (self.collar_w - tw) / 2 + 0.002, self.collar_t),
             (0.0, tw / 2 + (self.collar_w - tw) / 4 - 0.001, self.collar_z), steel),
        )
        dw, bt, ww, rz, rt = self.dock_w, self.base_t, self.well_w, self.rim_z, self.rim_t
        wall_th = (dw - ww) / 2  # 0.045
        cx = ww / 2 + wall_th / 2  # 0.0475
        wh = rz - rt - bt  # charcoal wall height (0.102)
        wz = bt + wh / 2
        rzc = rz - rt / 2
        dk, am = self.color_dock, self.color_rim
        self.dock_pieces = (
            ("base", (dw, dw, bt), (0.0, 0.0, bt / 2), dk),
            ("wall_xn", (wall_th, dw, wh), (-cx, 0.0, wz), dk),
            ("wall_xp", (wall_th, dw, wh), (cx, 0.0, wz), dk),
            ("wall_yn", (ww, wall_th, wh), (0.0, -cx, wz), dk),
            ("wall_yp", (ww, wall_th, wh), (0.0, cx, wz), dk),
            ("rim_xn", (wall_th, dw, rt), (-cx, 0.0, rzc), am),
            ("rim_xp", (wall_th, dw, rt), (cx, 0.0, rzc), am),
            ("rim_yn", (ww, wall_th, rt), (0.0, -cx, rzc), am),
            ("rim_yp", (ww, wall_th, rt), (0.0, cx, rzc), am),
        )
        # Seating: inverted (foot down), the collar UNDERSIDE (body-local z = collar_z +
        # collar_t/2, which faces down when flipped) lands on the rim top.
        collar_under = -(self.collar_z + self.collar_t / 2)  # 0.025 above centre, inverted
        self.seat_z = rz - collar_under  # 0.095
        insert_depth = hl + collar_under  # foot tip -> collar underside = 0.100
        # Wrong-way (upright) the collar underside is body-local z = collar_z - collar_t/2.
        wrong_entry = hl - (-(self.collar_z - self.collar_t / 2))  # 0.040
        self.perch_z = rz + (hl - wrong_entry)  # 0.155
        self.foot_seat_z = self.seat_z - hl  # 0.020
        self.tip_z_gate = rz - 0.005
        self.spawn_z = hl + 0.002
        self.aperture = 2 * (self.ledge_cx - self.ledge_w / 2)  # 0.022

        # --- honesty asserts -----------------------------------------------------------------
        md = 2 * self.marble_r
        # A parallel jaw grasps the bare tube; standing, a 100 mm band sits above the collar.
        assert tw <= 0.075, "tube must fit the Franka jaw"
        assert (hl - ct) - (self.collar_z + self.collar_t / 2) >= 0.05, "grasp band above collar"
        # One-way key: collar refused by the well; tube admitted straight, refused at 45 deg.
        assert self.collar_w - ww >= 0.008, "collar must NOT enter the well"
        assert ww - tw >= 0.008, "well must admit the tube with clearance"
        assert tw * math.sqrt(2.0) - ww >= 0.004, "well must refuse the tube turned 45 deg"
        # The collar is the stop: the foot never reaches the well floor; the wrong-way perch
        # sits far above the seat (the two outcomes are unambiguous).
        well_depth = rz - bt
        assert well_depth - insert_depth >= 0.005, "collar must stop the capsule first"
        assert self.perch_z - self.seat_z >= 0.040, "perch and seat must be far apart"
        assert self.perch_z - self.seat_z > 4 * self.seat_tol, "seat_tol separates the two"
        # Marble passage: the waist passes the marble with margin, the chambers hold it with
        # clearance, and the ledges can never shelf it (the walls keep the marble centre
        # inboard of the ledge inner edges).
        assert self.aperture - md >= 0.006, "waist must pass the marble"
        assert iw - md >= 0.008, "chamber must hold the marble with clearance"
        # The collar ring never intrudes into the tube interior (the marble's path).
        assert tw / 2 - 0.002 >= tw / 2 - wt, "collar ring stays outside the interior"
        assert iw / 2 - self.marble_r <= (self.aperture / 2) - 0.002, "ledge can never shelf"
        # Marble band honesty: the foot-chamber rest pose sits inside the credited band with
        # margin; the collar-chamber rest pose sits far outside it.
        foot_rest = (hl - ct) - self.marble_r  # 0.063
        assert self.mb_z[0] + 0.005 <= foot_rest <= self.mb_z[1] - 0.005, "band holds the rest"
        assert self.marble_home_z <= -(self.mb_z[0]) - 0.020, "home chamber far outside band"
        assert abs(self.marble_home_z) + self.marble_r <= hl - ct, "marble fits its chamber"
        # In-well gates stay inside the cavity; the seated centre tolerance covers the full
        # mechanical play of the tube in the well.
        assert self.tip_xy <= ww / 2 - 0.005, "tip gate stays inside the well cavity"
        assert self.center_tol >= (ww - tw) / 2, "center_tol covers the well play"
        # Seated, a >= 40 mm band of bare tube stands proud of the rim for the release.
        assert self.seat_z + hl - rz >= 0.040, "seated capsule stands proud of the rim"
        # The spawn ring can never overlap the dock (circumradii + margin).
        dock_circ = (dw / 2) * math.sqrt(2.0)
        timer_circ = (self.collar_w / 2) * math.sqrt(2.0)
        assert self.spawn_r[0] >= dock_circ + timer_circ + 0.010, "spawn ring clears the dock"
        assert self.spawn_r[1] > self.spawn_r[0], "spawn ring is a real interval"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("timer_flip_dock")
class TimerFlipDockScene(BaseScene):
    cfg: TimerFlipDockSceneCfg

    def __init__(self, cfg: TimerFlipDockSceneCfg | None = None) -> None:
        super().__init__(cfg or TimerFlipDockSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the 9 kinematic dock pieces at their canonical pose (reset()
        re-places everything), the compound timer capsule, and the captive marble."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        slick = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.06, dynamic_friction=0.05, restitution=0.0)
        grippy = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.35, dynamic_friction=0.30, restitution=0.0)

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
        }
        for name, size, ctr, rgb in c.dock_pieces:
            # well walls + rim slick (the capsule slides in cleanly); base grippy
            mat = grippy if name == "base" else slick
            out["dock_" + name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dock_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.dock_pos[0] + ctr[0], c.dock_pos[1] + ctr[1], ctr[2])),
            )
        out["timer"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Timer",
            spawn=_timer_spawner_cfg(
                boxes=c.timer_boxes, mass=c.timer_mass, com_z=c.timer_com_z,
                diag_inertia=c.timer_diag_inertia, contact_offset=c.contact_offset),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.dock_pos[0] + c.spawn_r[1], c.dock_pos[1], c.spawn_z)),
        )
        out["marble"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Marble",
            spawn=sim_utils.SphereCfg(
                radius=c.marble_r,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    max_depenetration_velocity=0.5,
                    linear_damping=0.05,
                    angular_damping=0.05,
                    sleep_threshold=0.0,
                    stabilization_threshold=0.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.marble_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=grippy,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.color_marble),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.dock_pos[0] + c.spawn_r[1], c.dock_pos[1],
                     c.spawn_z + c.marble_home_z)),
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the per-env dock frame, the sampled spawn, the latches."""
        super().bind(env)
        c = self.cfg
        n = env.num_envs
        dev = env.device
        self.dock: dict[str, RigidObject] = {name: env.iscene["dock_" + name]
                                             for name, _s, _c, _rgb in c.dock_pieces}
        self.timer: RigidObject = env.iscene["timer"]
        self.marble: RigidObject = env.iscene["marble"]
        self.env_origins = env.iscene.env_origins
        self.d_pos = torch.zeros(n, 2, device=dev)  # dock centre xy (env-local)
        self.d_yaw = torch.zeros(n, device=dev)
        self.sp_r = torch.zeros(n, device=dev)  # sampled spawn radius (readback reference)
        self.sp_az = torch.zeros(n, device=dev)  # sampled spawn azimuth
        self.t_yaw0 = torch.zeros(n, device=dev)  # sampled capsule spawn yaw
        # latches
        self.lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        self.inverted = torch.zeros(n, dtype=torch.bool, device=dev)
        self.tip_in = torch.zeros(n, dtype=torch.bool, device=dev)
        self.marble_across = torch.zeros(n, dtype=torch.bool, device=dev)
        self.depth_max = torch.zeros(n, device=dev)

    # ----- dock-frame transforms -----------------------------------------------------------------
    def local_to_world(self, p_local: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """(m, 3) dock-local points -> world, applying yaw, dock origin, env origin.
        Dock-local z IS world height (the dock sits on the ground)."""
        cy, sy = torch.cos(self.d_yaw[env_ids]), torch.sin(self.d_yaw[env_ids])
        out = torch.empty_like(p_local)
        out[:, 0] = self.d_pos[env_ids, 0] + cy * p_local[:, 0] - sy * p_local[:, 1]
        out[:, 1] = self.d_pos[env_ids, 1] + sy * p_local[:, 0] + cy * p_local[:, 1]
        out[:, 2] = p_local[:, 2]
        return out + self.env_origins[env_ids]

    def world_to_local(self, p_world: torch.Tensor) -> torch.Tensor:
        """(N, 3) world points -> dock-local, all envs."""
        p = p_world - self.env_origins
        cy, sy = torch.cos(self.d_yaw), torch.sin(self.d_yaw)
        dx = p[:, 0] - self.d_pos[:, 0]
        dy = p[:, 1] - self.d_pos[:, 1]
        out = torch.empty_like(p)
        out[:, 0] = cy * dx + sy * dy
        out[:, 1] = -sy * dx + cy * dy
        out[:, 2] = p[:, 2]
        return out

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the dock frame (xy jitter + yaw), re-pin the 9 kinematic
        pieces, stand the capsule collar-down on a free-azimuth ring with free yaw, drop
        the marble at its jittered home in the collar-end chamber, clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        _ = torch.rand(m, device=dev)  # burn the degenerate first post-seed draw

        self.d_pos[env_ids, 0] = c.dock_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.dock_jitter
        self.d_pos[env_ids, 1] = c.dock_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.dock_jitter
        self.d_yaw[env_ids] = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.dock_yaw_deg)
        self.sp_r[env_ids] = (c.spawn_r[0]
                              + torch.rand(m, device=dev) * (c.spawn_r[1] - c.spawn_r[0]))
        self.sp_az[env_ids] = torch.rand(m, device=dev) * 2 * math.pi
        self.t_yaw0[env_ids] = torch.rand(m, device=dev) * 2 * math.pi

        half = self.d_yaw[env_ids] / 2
        qw, qz = torch.cos(half), torch.sin(half)
        for name, _size, ctr, _rgb in c.dock_pieces:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = self.local_to_world(
                torch.tensor(ctr, device=dev).expand(m, 3).clone(), env_ids)
            st[:, 3] = qw
            st[:, 6] = qz
            self.dock[name].write_root_state_to_sim(st, env_ids)

        # capsule: standing collar-end DOWN (identity up-axis) on the spawn ring
        tx = self.d_pos[env_ids, 0] + self.sp_r[env_ids] * torch.cos(self.sp_az[env_ids])
        ty = self.d_pos[env_ids, 1] + self.sp_r[env_ids] * torch.sin(self.sp_az[env_ids])
        th = self.t_yaw0[env_ids] / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = tx
        st[:, 1] = ty
        st[:, 2] = c.spawn_z
        st[:, 0:3] += self.env_origins[env_ids]
        st[:, 3] = torch.cos(th)
        st[:, 6] = torch.sin(th)
        self.timer.write_root_state_to_sim(st, env_ids)

        # marble: at its body-local home in the collar-end chamber (yaw-rotated jitter)
        jx = (torch.rand(m, device=dev) * 2 - 1) * c.marble_jitter
        jy = (torch.rand(m, device=dev) * 2 - 1) * c.marble_jitter
        cy, sy = torch.cos(self.t_yaw0[env_ids]), torch.sin(self.t_yaw0[env_ids])
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = tx + cy * jx - sy * jy
        st[:, 1] = ty + sy * jx + cy * jy
        st[:, 2] = c.spawn_z + c.marble_home_z
        st[:, 0:3] += self.env_origins[env_ids]
        st[:, 3] = 1.0
        self.marble.write_root_state_to_sim(st, env_ids)

        self.lifted[env_ids] = False
        self.inverted[env_ids] = False
        self.tip_in[env_ids] = False
        self.marble_across[env_ids] = False
        self.depth_max[env_ids] = 0.0

    # ----- geometry queries ---------------------------------------------------------------------
    def timer_up(self) -> torch.Tensor:
        """(N, 3) the capsule's body +z (foot direction) in world coordinates."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(self.timer.data.root_quat_w, ez)

    def end_tips(self) -> tuple[torch.Tensor, torch.Tensor]:
        """((N,3), (N,3)) world positions of the foot tip (+z end) and collar-end tip."""
        from isaaclab.utils.math import quat_apply

        hl = self.cfg.half_len
        n = self.env.num_envs
        dev = self.env.device
        q = self.timer.data.root_quat_w
        p = self.timer.data.root_pos_w
        foot = p + quat_apply(q, torch.tensor([0.0, 0.0, hl], device=dev).expand(n, 3))
        top = p + quat_apply(q, torch.tensor([0.0, 0.0, -hl], device=dev).expand(n, 3))
        return foot, top

    def marble_local(self) -> torch.Tensor:
        """(N, 3) the marble centre in the capsule's BODY frame."""
        from isaaclab.utils.math import quat_apply_inverse

        d = self.marble.data.root_pos_w - self.timer.data.root_pos_w
        return quat_apply_inverse(self.timer.data.root_quat_w, d)

    def marble_in_foot(self) -> torch.Tensor:
        """(N,) bool LIVE: marble inside the credited foot-chamber band (body frame)."""
        c = self.cfg
        ml = self.marble_local()
        return ((ml[:, 0].abs() <= c.mb_xy) & (ml[:, 1].abs() <= c.mb_xy)
                & (ml[:, 2] >= c.mb_z[0]) & (ml[:, 2] <= c.mb_z[1]))

    def settled(self) -> torch.Tensor:
        """(N,) bool: capsule + marble below the settle thresholds."""
        c = self.cfg
        return ((self.timer.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.timer.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
                & (self.marble.data.root_lin_vel_w.norm(dim=-1) < c.settle_marble))

    # ----- mechanics (every substep) --------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch bookkeeping: lifted, inverted-aloft, foot-tip-in-well, gated insertion
        depth (latched max), marble crossed while flipped."""
        c = self.cfg
        foot, top = self.end_tips()
        min_end = torch.minimum(foot[:, 2], top[:, 2]) - self.env_origins[:, 2]
        up_z = self.timer_up()[:, 2]
        aloft = min_end > c.lift_z
        self.lifted |= aloft
        self.inverted |= aloft & (up_z < -c.invert_cos)

        fl = self.world_to_local(foot)
        gate = ((up_z < -c.tip_gate_cos)
                & (fl[:, 0].abs() <= c.tip_xy) & (fl[:, 1].abs() <= c.tip_xy)
                & (fl[:, 2] < c.tip_z_gate))
        self.tip_in |= gate
        d = ((c.tip_z_gate - fl[:, 2]) / (c.tip_z_gate - c.foot_seat_z)).clamp(0.0, 1.0)
        self.depth_max = torch.maximum(self.depth_max, torch.where(gate, d, torch.zeros_like(d)))

        self.marble_across |= (up_z < -c.tip_gate_cos) & self.marble_in_foot()

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "timer": self.timer.data.root_state_w[env_ids].clone(),
            "marble": self.marble.data.root_state_w[env_ids].clone(),
            "dock": {"d_pos": self.d_pos[env_ids].clone(), "d_yaw": self.d_yaw[env_ids].clone(),
                     "sp_r": self.sp_r[env_ids].clone(), "sp_az": self.sp_az[env_ids].clone(),
                     "t_yaw0": self.t_yaw0[env_ids].clone()},
            "latches": {"lifted": self.lifted[env_ids].clone(),
                        "inverted": self.inverted[env_ids].clone(),
                        "tip_in": self.tip_in[env_ids].clone(),
                        "marble_across": self.marble_across[env_ids].clone(),
                        "depth_max": self.depth_max[env_ids].clone()},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.d_pos[env_ids] = state["dock"]["d_pos"]
        self.d_yaw[env_ids] = state["dock"]["d_yaw"]
        self.sp_r[env_ids] = state["dock"]["sp_r"]
        self.sp_az[env_ids] = state["dock"]["sp_az"]
        self.t_yaw0[env_ids] = state["dock"]["t_yaw0"]
        dev = self.env.device
        m = len(env_ids)
        half = self.d_yaw[env_ids] / 2
        for name, _size, ctr, _rgb in self.cfg.dock_pieces:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = self.local_to_world(
                torch.tensor(ctr, device=dev).expand(m, 3).clone(), env_ids)
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            self.dock[name].write_root_state_to_sim(st, env_ids)
        self.timer.write_root_state_to_sim(state["timer"], env_ids)
        self.marble.write_root_state_to_sim(state["marble"], env_ids)
        self.lifted[env_ids] = state["latches"]["lifted"]
        self.inverted[env_ids] = state["latches"]["inverted"]
        self.tip_in[env_ids] = state["latches"]["tip_in"]
        self.marble_across[env_ids] = state["latches"]["marble_across"]
        self.depth_max[env_ids] = state["latches"]["depth_max"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A charcoal DOCK block ({c.dock_w * 1000:.0f} mm square, "
            f"{c.rim_z * 1000:.0f} mm tall, amber band around its top rim) sits on the "
            f"ground; its position and heading vary per episode. A square well "
            f"({c.well_w * 1000:.0f} mm opening, {(c.rim_z - c.base_t) * 1000:.0f} mm deep) "
            f"drops from the centre of its top face. A blue TIMER capsule stands on the "
            f"ground somewhere on a ring around the dock: a {c.tube_w * 1000:.0f} mm square "
            f"tube, {2 * c.half_len * 1000:.0f} mm long, with a WHITE cap on one end (the "
            f"foot), a CRIMSON cap on the other, and a steel collar flange "
            f"({c.collar_w * 1000:.0f} mm square) around the tube {40:.0f} mm from the "
            f"crimson end. It spawns standing CRIMSON-END DOWN. Sealed inside, an orange "
            f"marble ({2 * c.marble_r * 1000:.0f} mm) rests in the crimson-end chamber; a "
            f"narrow internal waist between two ledges separates the crimson-end chamber "
            f"from the white-end chamber, and the marble can only pass through the waist "
            f"along the tube's axis.\n"
            f"Goal: dock the capsule in the well UPSIDE-DOWN and fully seated — white foot "
            f"end down inside the well, crimson end up, the steel collar resting on the "
            f"amber rim (capsule centre within {c.center_tol * 1000:.0f} mm of the well "
            f"axis, seated within {c.seat_tol * 1000:.0f} mm of the collar stop) — with the "
            f"marble fallen through the internal waist into the white-end chamber, and "
            f"everything at rest. The well only admits the capsule foot-first and square-on: "
            f"the collar cannot enter (it is wider than the well) so inserting crimson-end "
            f"first catches the collar on the rim {(c.perch_z - c.seat_z) * 1000:.0f} mm too "
            f"high, and the tube cannot enter turned 45 degrees. Lift the capsule, flip it "
            f"fully upside-down in the air (the marble drops through the waist), lower the "
            f"foot into the well, and let the collar seat on the rim."
        )

    def instruction(self) -> str:
        return (
            "Pick up the standing blue capsule, flip it completely upside-down so its white "
            "foot end points down and the sealed marble falls into the foot chamber, then "
            "lower it foot-first into the dock's square well until the steel collar seats "
            "on the amber rim. Leave it fully seated, upside-down, and at rest; inserting "
            "it crimson-end first is blocked by the collar and does not count."
        )

    # ----- progress / rubric ----------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool LIVE: seated inverted pose (up-axis flipped, centred on the well axis,
        collar on the rim) AND the marble in the foot-chamber band AND everything settled."""
        c = self.cfg
        up_z = self.timer_up()[:, 2]
        tl = self.world_to_local(self.timer.data.root_pos_w)
        seated = ((up_z <= -c.seat_up_cos)
                  & (tl[:, 0].abs() <= c.center_tol) & (tl[:, 1].abs() <= c.center_tol)
                  & ((tl[:, 2] - c.seat_z).abs() <= c.seat_tol))
        return seated & self.marble_in_foot() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10 lifted + 0.15 inverted-aloft + 0.15 tip-in-well +
        0.30 * gated latched insertion depth + 0.15 marble-across; exactly 1.0 iff
        success(). Latched credit never evaporates; the null policy scores 0."""
        s = (0.10 * self.lifted.float() + 0.15 * self.inverted.float()
             + 0.15 * self.tip_in.float() + 0.30 * self.depth_max.clamp(0.0, 1.0)
             + 0.15 * self.marble_across.float())
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("simgen", lambda: EnvCfg(scene="timer_flip_dock", robot="null", env_spacing=3))
