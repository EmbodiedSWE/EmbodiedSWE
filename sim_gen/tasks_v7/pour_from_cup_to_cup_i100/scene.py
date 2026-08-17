"""TipDumpScene — drain the trunnion hopper's balls into the catch basin by holding
down the correct rocker pedal (sim_gen task `pour_from_cup_to_cup_i100`).

Derived from rlbench/pour_from_cup_to_cup ("pour the contents of the source cup into
the target cup among distractor cups"), but the MANIPULATION MODEL is replaced
wholesale. The seed's plan is: grasp the free-standing source cup, CARRY it above the
target cup, tilt it with the wrist, pour, set it down — a transport-plus-reorient of a
held vessel, with look-alike distractor cups as the only decision. Here NOTHING is
grasped and NOTHING is carried: the source vessel is a TIPPING HOPPER mounted on
trunnions atop a stand, held upright by a return spring. A rocker CROSSBAR through the
trunnion axle ends in two yellow PEDALS; pressing a pedal down tips the hopper toward
that pedal's side. The open-top green CATCH BASIN sits on ONE side of the station
(randomized per episode). The solver must:
  (1) find the basin side visually and choose the pedal on THAT side (pressing the
      other pedal dumps the balls onto the open floor — an irrecoverable spill the
      rubric scores ~0);
  (2) PRESS AND HOLD the pedal against the return spring past the ~57 deg drain angle
      — a sustained force interaction, not a positioning move: let go early and the
      spring slams the hopper back with the balls still inside;
  (3) wait while the balls roll over the hopper's low spout lip, fly a real ballistic
      arc, and land in the basin (gravity does the transfer — no contact of the
      solver with any ball is required or intended);
  (4) RELEASE the pedal so the spring returns the hopper upright: success is judged
      with every present ball settled INSIDE the basin and the hopper back at rest
      near vertical.
A solver needs a different PLAN (mechanism actuation with a hold phase and a
direction commitment) and a different code structure (spring-hinge plant, drain
monitoring, release endgame) — not different numbers on the seed's plan.

Assets are fully procedural (compound spawners; explicit MassAPI + authored CoM;
friction MATERIALS bound to every collider — custom-spawner colliders otherwise get
the ~0.5 default with no restitution control):
  - stand: KINEMATIC compound — narrow base slab + two pillars carrying the trunnion
    bearings at (0, +-0.15, 0.30). The hinge joint filters ALL stand<->hopper
    collision (trunnion stubs pass through the bearing line by design).
  - hopper: DYNAMIC compound on an authored Y-axis revolute joint through its body
    origin (the pivot): open box (interior 120 x 120 mm, floor 45 mm below the
    pivot, tall side walls along +-y, LOW SPOUT LIPS 9 mm tall along +-x — the lip
    height sets the ~57 deg drain angle for a 20 mm-radius ball), a trunnion stub
    reaching to y = -0.115 and the 360 mm rocker crossbar with a yellow pedal plate
    at each end (x = +-0.155). CoM authored 20 mm below the pivot (pendulum-stable).
  - spring plant: post_step applies tau = press - k*theta - c*omega about the WORLD
    Y axis (the hinge axis; a torque parallel to the rotation axis is immune to the
    external-wrench frame-drag quirk). `press_tau` is the ONLY external input slot
    (solve.py's pedal press and smoke's probes write it; the scene clamps it to
    `press_tau_max`, the physical bound of a fingertip press on the 155 mm pedal
    arm). Discrete-stability audit at 120 Hz with the AUTHORED inertia
    (I_yy ~ 0.0102 kg m^2 about the pivot): omega_n*dt ~ 0.09, zeta ~ 1.4
    (overdamped return), c*dt/I ~ 0.25 << 1.
  - basin: KINEMATIC open-top box (interior 260 x 184 mm, walls 100 mm tall — deep
    vertical walls retain the ~2 m/s landing slam), teleported per episode to the
    +x or -x side of the station.
  - balls: 2..4 dynamic spheres (r 20 mm, 30 g, velocity iters 4 against the GPU
    sphere-creep artifact), distinct colors, spawned on the hopper floor.

Per-episode randomization (readback-verifiable): basin SIDE (+x / -x), basin xy
jitter, present-ball COUNT (2..4, subset-sampled), ball slot permutation + jitter,
hopper start tilt jitter (+-3 deg).

Rubric (0..1, anchored in the demonstrated solve trajectory):
  0.15  tip_latch — the hinge ever tipped past `drain_latch_deg` TOWARD the basin
        side (signed; a wrong-side dump never latches) (latched)
  0.60 * frac — fraction of present balls currently inside the basin (live, physical)
  1.0 iff success(): every present ball settled inside the basin AND the hopper
        released back upright at rest. Non-success is capped at 0.75.
Null policy scores ~0 (spring holds the hopper upright; nothing drains).

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


# ----- compound spawner helpers (the block_pyramid / fire_crib pattern) -------------------------
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable | None):
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(box.GetPrim())
    return box.GetPrim()


def _add_cyl_y(stage, path: str, *, center, radius, height, color):
    """Visual-only cylinder along local Y (the trunnion axle — no collision: the
    stand<->hopper pair is joint-filtered anyway, other bodies never reach it)."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Y")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return cyl.GetPrim()


def _rigid_dynamic(root, mass: float, *, lin_damp: float, ang_damp: float,
                   iters: int = 16, vel_iters: int = 4, com: tuple | None = None,
                   inertia: tuple | None = None) -> None:
    """Dynamic rigid-body armor on a compound root: MassAPI mass (+ authored local CoM —
    with only a mass, PhysX keeps the CoM at the body origin = the pivot, killing the
    pendulum stability; + authored diagonal inertia so the hinge plant's discrete
    stability is exact by construction, not hostage to the shape-derived estimate),
    damping, no sleeping while velocities are judged."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    mapi = UsdPhysics.MassAPI.Apply(root)
    mapi.CreateMassAttr(float(mass))
    if com is not None:
        mapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    if inertia is not None:
        mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(int(iters))
    px.CreateSolverVelocityIterationCountAttr(int(vel_iters))
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _rigid_kinematic(root) -> None:
    from pxr import UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)


def _bind_mat(stage_path: str, mat_path: str, static: float, dynamic: float) -> None:
    """Author (once) and bind a friction material (custom-spawner colliders otherwise
    get the ~0.5 default with no restitution control — a bouncing ball never settles)."""
    import isaaclab.sim as sim_utils
    import omni.usd
    from isaaclab.sim.utils import bind_physics_material

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(mat_path).IsValid():
        sim_utils.spawn_rigid_body_material(
            mat_path,
            sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic,
                                           restitution=0.0))
    bind_physics_material(stage_path, mat_path)


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC trunnion stand: local origin at ground level under the pivot.
    Base slab (narrow in x so the basin can sit close) + two bearing pillars."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_kinematic(root)
    collide = _make_collide(c.contact_offset)
    grey = (0.35, 0.33, 0.30)
    _add_box(stage, f"{prim_path}/slab", center=(0.0, 0.0, c.slab_t / 2),
             size=(c.slab_x, c.slab_y, c.slab_t), color=(0.25, 0.24, 0.22), collide=collide)
    for tag, sy in (("pillar_n", -1.0), ("pillar_p", 1.0)):
        _add_box(stage, f"{prim_path}/{tag}",
                 center=(0.0, sy * c.pillar_y, c.slab_t + c.pillar_h / 2),
                 size=(c.pillar_w, c.pillar_w, c.pillar_h), color=grey, collide=collide)
        # visual bearing cap at the pivot height
        _add_box(stage, f"{prim_path}/{tag}_cap",
                 center=(0.0, sy * c.pillar_y, c.pivot_z),
                 size=(c.pillar_w + 0.008, c.pillar_w + 0.008, 0.024),
                 color=(0.15, 0.15, 0.17), collide=None)
    _bind_mat(prim_path, "/World/mats/stand", 0.6, 0.5)
    return root


def _spawn_hopper(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC tipping hopper. Local origin = the PIVOT (trunnion axis = local Y).
    Open box below/around the pivot + trunnion stub + rocker crossbar + two pedals."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, c.hopper_mass, lin_damp=0.05, ang_damp=0.05,
                   com=(0.0, 0.0, -0.020), inertia=(0.010, 0.010, 0.010))
    collide = _make_collide(c.contact_offset)
    steel = (0.46, 0.48, 0.52)
    dark = (0.20, 0.20, 0.22)
    yellow = (0.95, 0.85, 0.10)
    hi = c.hop_in_half            # interior half width (x and y)
    t = c.hop_wall_t
    zf = c.floor_top_local        # interior floor top (local z)
    zw = c.side_wall_top_local    # tall side-wall top (local z)
    # floor
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, zf - t / 2),
             size=(2 * hi + 2 * t, 2 * hi + 2 * t, t), color=steel, collide=collide)
    # tall side walls along +-y
    for tag, sy in (("wall_n", -1.0), ("wall_p", 1.0)):
        _add_box(stage, f"{prim_path}/{tag}",
                 center=(0.0, sy * (hi + t / 2), (zf + zw) / 2),
                 size=(2 * hi + 2 * t, t, zw - zf), color=steel, collide=collide)
    # LOW spout lips along +-x (lip height sets the drain angle)
    for tag, sx in (("lip_n", -1.0), ("lip_p", 1.0)):
        _add_box(stage, f"{prim_path}/{tag}",
                 center=(sx * (hi + t / 2), 0.0, zf + c.lip_h / 2),
                 size=(t, 2 * hi, c.lip_h), color=(0.65, 0.30, 0.10), collide=collide)
    # visual trunnion axle through the pivot
    _add_cyl_y(stage, f"{prim_path}/axle", center=(0.0, 0.0, 0.0),
               radius=0.008, height=2 * c.pillar_y + 0.06, color=dark)
    # trunnion stub from the hopper wall out to the rocker bar (stays clear of the
    # bearing pillar at y = -pillar_y: max |y| here is 0.130 < the pillar face 0.135)
    _add_box(stage, f"{prim_path}/stub",
             center=(0.0, (-(hi + t) + c.bar_y) / 2, 0.0),
             size=(0.024, abs(c.bar_y) + 0.01 - (hi + t) + 0.02, 0.016),
             color=dark, collide=collide)
    # rocker crossbar + pedals
    _add_box(stage, f"{prim_path}/bar", center=(0.0, c.bar_y, 0.0),
             size=(c.bar_len, c.bar_w, c.bar_h), color=dark, collide=collide)
    for tag, sx in (("pedal_n", -1.0), ("pedal_p", 1.0)):
        _add_box(stage, f"{prim_path}/{tag}",
                 center=(sx * c.pedal_x, c.bar_y, c.bar_h / 2 + c.pedal_t / 2),
                 size=(c.pedal_w, c.pedal_w, c.pedal_t), color=yellow, collide=collide)
    _bind_mat(prim_path, "/World/mats/hopper", 0.30, 0.25)
    return root


def _spawn_basin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC open-top catch basin: local origin at the footprint centre on the
    ground. Deep vertical walls (they retain the landing slam)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_kinematic(root)
    collide = _make_collide(c.contact_offset)
    green = (0.13, 0.52, 0.21)
    ix, iy = c.basin_in_x_half, c.basin_in_y_half
    t, ft, wh = c.basin_wall_t, c.basin_floor_t, c.basin_wall_h
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, ft / 2),
             size=(2 * ix + 2 * t, 2 * iy + 2 * t, ft), color=green, collide=collide)
    for tag, sx in (("wx_n", -1.0), ("wx_p", 1.0)):
        _add_box(stage, f"{prim_path}/{tag}", center=(sx * (ix + t / 2), 0.0, ft + wh / 2),
                 size=(t, 2 * iy + 2 * t, wh), color=green, collide=collide)
    for tag, sy in (("wy_n", -1.0), ("wy_p", 1.0)):
        _add_box(stage, f"{prim_path}/{tag}", center=(0.0, sy * (iy + t / 2), ft + wh / 2),
                 size=(2 * ix, t, wh), color=green, collide=collide)
    _bind_mat(prim_path, "/World/mats/basin", 0.55, 0.50)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "stand" not in _SPAWNER_CACHE:

        def _mk(fn):
            @configclass
            class _Cfg(RigidObjectSpawnerCfg):
                func: Callable = clone(fn)
                # geometry (mirrors TipDumpSceneCfg — filled at assets() time)
                contact_offset: float = 0.002
                pivot_z: float = 0.30
                slab_x: float = 0.056
                slab_y: float = 0.36
                slab_t: float = 0.02
                pillar_y: float = 0.15
                pillar_w: float = 0.03
                pillar_h: float = 0.29
                hop_in_half: float = 0.06
                hop_wall_t: float = 0.008
                floor_top_local: float = -0.045
                side_wall_top_local: float = 0.055
                lip_h: float = 0.009
                hopper_mass: float = 0.40
                bar_y: float = -0.115
                bar_len: float = 0.36
                bar_w: float = 0.024
                bar_h: float = 0.016
                pedal_x: float = 0.155
                pedal_w: float = 0.05
                pedal_t: float = 0.015
                basin_in_x_half: float = 0.13
                basin_in_y_half: float = 0.092
                basin_wall_t: float = 0.008
                basin_floor_t: float = 0.008
                basin_wall_h: float = 0.10

            return _Cfg

        _SPAWNER_CACHE.update(stand=_mk(_spawn_stand), hopper=_mk(_spawn_hopper),
                              basin=_mk(_spawn_basin))
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TipDumpSceneCfg(BaseCfg):
    """Config for `TipDumpScene`. The geometric claims the task rests on are asserted in
    `__post_init__` (drain angle from the lip height, pedal-vs-basin clearance at the
    joint limit, ballistic capture window, spring holdability within the press bound)."""

    # --- tunable: rubric thresholds -----------------------------------------------------------
    upright_tol_deg: float = tunable(8.0)    # |hinge| below this = hopper back upright
    settle_speed: float = tunable(0.05)      # max ball |lin vel| when judging (m/s)
    hinge_settle_omega: float = tunable(0.30)  # max |hinge rate| (rad/s) when judging upright
    drain_latch_deg: float = tunable(55.0)   # tip latch arms past this angle TOWARD the basin
    basin_margin: float = tunable(0.004)     # in-basin test shrink margin (m)

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    n_min: int = tunable(2)                  # min present balls
    n_max: int = tunable(4)                  # max present balls (= number of ball bodies)
    basin_x_jitter: float = tunable(0.015)   # +- jitter of the basin stand-off distance
    basin_y_jitter: float = tunable(0.020)   # +- jitter of the basin y position
    ball_jitter: float = tunable(0.007)      # +- xy jitter of each ball on its floor slot
    start_tilt_jitter_deg: float = tunable(3.0)  # +- hopper start tilt about the hinge

    # --- tunable: hinge plant (difficulty dials) ------------------------------------------------
    spring_k: float = tunable(1.2)           # return spring (N*m/rad toward upright)
    spring_c: float = tunable(0.30)          # viscous hinge damping (N*m*s/rad)
    press_tau_max: float = tunable(2.5)      # physical bound on the consumed press torque
    # (= ~16 N of fingertip force on the 155 mm pedal arm; the scene clamps ANY writer)

    # --- info: station geometry -----------------------------------------------------------------
    pivot_z: float = info(0.30)              # trunnion axis height (world z; axis = world +y)
    slab_x: float = info(0.056)
    slab_y: float = info(0.36)
    slab_t: float = info(0.02)
    pillar_y: float = info(0.15)             # bearing pillars at (0, +-pillar_y)
    pillar_w: float = info(0.03)
    hop_in_half: float = info(0.060)         # hopper interior half width (x and y)
    hop_wall_t: float = info(0.008)
    floor_top_local: float = info(-0.045)    # interior floor top, hopper (pivot) frame
    side_wall_top_local: float = info(0.055)  # tall +-y wall top, hopper frame
    lip_h: float = info(0.009)               # LOW +-x spout lip height (sets the drain angle)
    hopper_mass: float = info(0.40)
    bar_y: float = info(-0.115)              # rocker crossbar centreline (hopper frame)
    bar_len: float = info(0.36)
    bar_w: float = info(0.024)
    bar_h: float = info(0.016)
    pedal_x: float = info(0.155)             # pedal centres at (+-pedal_x, bar_y)
    pedal_w: float = info(0.05)
    pedal_t: float = info(0.015)
    joint_limit_deg: float = info(68.0)      # hinge hard stops
    hold_deg: float = info(63.0)             # intended hold angle (> drain, < limit)
    # --- info: basin -----------------------------------------------------------------------------
    basin_in_x_half: float = info(0.130)
    basin_in_y_half: float = info(0.092)
    basin_wall_t: float = info(0.008)
    basin_floor_t: float = info(0.008)
    basin_wall_h: float = info(0.100)
    basin_near_face: float = info(0.038)     # |x| of the basin's inner near-wall face
    # --- info: balls -----------------------------------------------------------------------------
    ball_r: float = info(0.020)
    ball_mass: float = info(0.030)
    ball_colors: tuple = info(((0.85, 0.10, 0.10), (0.10, 0.25, 0.85),
                               (0.95, 0.55, 0.05), (0.55, 0.15, 0.65)))
    ball_color_names: tuple = info(("red", "blue", "orange", "violet"))
    depot: tuple = info((1.4, 1.4))          # ground depot for absent balls
    contact_offset: float = info(0.002)
    # rubric weights
    w_latch: float = info(0.15)
    w_frac: float = info(0.60)

    # Derived (filled in __post_init__).
    drain_deg: float = field(default=None, init=False)
    basin_cx: float = field(default=None, init=False)   # |basin centre x|
    basin_rim_z: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        r, h = self.ball_r, self.lip_h
        # Ball-over-lip statics: ball resting on the floor against the low lip contacts
        # the lip's top edge; it rolls over when the tilt exceeds atan(d / (r - h)) with
        # d = sqrt(r^2 - (r-h)^2) the horizontal centre-to-edge offset.
        assert 0.0 < h < r, "spout lip must be lower than the ball radius (edge contact)"
        d = math.sqrt(r * r - (r - h) * (r - h))
        self.drain_deg = math.degrees(math.atan2(d, r - h))
        assert 50.0 < self.drain_deg < 62.0, f"drain angle {self.drain_deg:.1f} out of band"
        assert self.drain_latch_deg < self.drain_deg, "latch must arm before full drain"
        assert self.drain_deg + 3.0 < self.hold_deg < self.joint_limit_deg - 4.0, \
            "hold angle must clear the drain angle and stay inside the joint limits"
        # Upright retention: hopping the lip needs sqrt(2 g h) of ball speed.
        assert math.sqrt(2 * 9.81 * h) > 0.35, "lip too low: balls could hop out at rest"
        # Basin placement + rim height.
        self.basin_cx = self.basin_near_face + self.basin_in_x_half
        self.basin_rim_z = self.basin_floor_t + self.basin_wall_h
        # Pedal never enters the basin walls: lowest pedal corner at the joint limit.
        th = math.radians(self.joint_limit_deg)
        ped_c_z = (self.pivot_z - self.pedal_x * math.sin(th)
                   + (self.bar_h / 2 + self.pedal_t / 2) * math.cos(th))
        ped_lo = ped_c_z - (self.pedal_w / 2 * math.sin(th) + self.pedal_t / 2 * math.cos(th))
        assert ped_lo > self.basin_rim_z + 0.010, \
            f"pedal dips to {ped_lo:.3f} at the limit (rim {self.basin_rim_z:.3f})"
        # Ballistic capture window at the hold angle: a zero-speed dribble off the lip
        # falls inside the near wall; a fast far-side roller lands short of the far wall.
        thh = math.radians(self.hold_deg)
        lip_x = self.hop_in_half * math.cos(thh) - self.floor_top_local * math.sin(thh)
        assert lip_x > self.basin_near_face + self.ball_r / 2, \
            "dribble balls off the lip must clear the basin near wall"
        lip_z = (self.pivot_z - self.hop_in_half * math.sin(thh)
                 + self.floor_top_local * math.cos(thh))
        v = math.sqrt(10.0 / 7.0 * 9.81 * 2 * self.hop_in_half * math.sin(thh))  # rolling
        vx, vz = v * math.cos(thh), v * math.sin(thh)
        drop = lip_z - (self.basin_floor_t + self.ball_r)
        t_fly = (-vz + math.sqrt(vz * vz + 2 * 9.81 * drop)) / 9.81
        assert lip_x + vx * t_fly < self.basin_cx + self.basin_in_x_half - 0.02, \
            "fast far-side rollers must land inside the far wall"
        # Spring holdable within the press bound (with ball-shift margin).
        need = (self.spring_k * math.radians(self.hold_deg)
                + self.hopper_mass * 9.81 * 0.02 * math.sin(thh)
                + self.n_max * self.ball_mass * 9.81 * self.hop_in_half)
        assert need < 0.85 * self.press_tau_max, \
            f"hold torque {need:.2f} too close to the press bound {self.press_tau_max}"


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qy(ang: torch.Tensor) -> torch.Tensor:
    """Quaternion for a rotation of `ang` (rad) about +Y, shape (m, 4) wxyz."""
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("tip_dump_station")
class TipDumpScene(BaseScene):
    cfg: TipDumpSceneCfg

    def __init__(self, cfg: TipDumpSceneCfg | None = None) -> None:
        super().__init__(cfg or TipDumpSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()

        def geo(cfg_cls):
            return cfg_cls(
                contact_offset=c.contact_offset, pivot_z=c.pivot_z,
                slab_x=c.slab_x, slab_y=c.slab_y, slab_t=c.slab_t,
                pillar_y=c.pillar_y, pillar_w=c.pillar_w, pillar_h=c.pivot_z - 0.01 - c.slab_t,
                hop_in_half=c.hop_in_half, hop_wall_t=c.hop_wall_t,
                floor_top_local=c.floor_top_local, side_wall_top_local=c.side_wall_top_local,
                lip_h=c.lip_h, hopper_mass=c.hopper_mass,
                bar_y=c.bar_y, bar_len=c.bar_len, bar_w=c.bar_w, bar_h=c.bar_h,
                pedal_x=c.pedal_x, pedal_w=c.pedal_w, pedal_t=c.pedal_t,
                basin_in_x_half=c.basin_in_x_half, basin_in_y_half=c.basin_in_y_half,
                basin_wall_t=c.basin_wall_t, basin_floor_t=c.basin_floor_t,
                basin_wall_h=c.basin_wall_h,
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
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=geo(cls["stand"]),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "hopper": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Hopper",
                spawn=geo(cls["hopper"]),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.pivot_z)),
            ),
            "basin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Basin",
                spawn=geo(cls["basin"]),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.basin_cx, 0.0, 0.0)),
            ),
        }
        for i in range(c.n_max):
            out[f"ball_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball_" + str(i),
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.2,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,  # GPU sphere-creep artifact fix
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.30, dynamic_friction=0.25, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ball_colors[i]),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.depot[0], c.depot[1] + 0.06 * i, c.ball_r + 0.002)),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # The hinge plant lives in an external wrench; without this flag the
                # wrench is under-applied across TGS iterations (the hopper stalled at
                # ~52 deg under a press whose static equilibrium is 105 deg) and the
                # velocity readback carries large phantom components.
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
        c = self.cfg
        n = env.num_envs
        dev = env.device
        self.stand: RigidObject = env.iscene["stand"]
        self.hopper: RigidObject = env.iscene["hopper"]
        self.basin: RigidObject = env.iscene["basin"]
        self.balls: list[RigidObject] = [env.iscene[f"ball_{i}"] for i in range(c.n_max)]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        # Episode state.
        self.side = torch.ones(n, dtype=torch.long, device=dev)  # +1: basin at +x, -1: -x
        self.present = torch.ones(n, c.n_max, dtype=torch.bool, device=dev)
        self._tip_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        # Finite-difference hinge rate: under external wrenches the PhysX angular
        # velocity readback carries large phantom components (observed +-5 rad/s while
        # the angle marches smoothly), so the plant derives its own rate from the
        # (clean) angle history.
        self._th_prev = torch.zeros(n, device=dev)
        self._fd_rate = torch.zeros(n, device=dev)
        # External input: press torque about the hinge (+y tips toward +x). The scene
        # clamps it to press_tau_max; post_step OWNS the hopper's external-wrench slot.
        self.press_tau = torch.zeros(n, device=dev)

    def _author_joints(self) -> None:
        """Per env: the Y-axis trunnion revolute joint stand->hopper through the pivot.
        The joint filters ALL stand<->hopper collision (stub/axle pass the bearings)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/trunnion")
            j.CreateBody0Rel().SetTargets([f"{base}/Stand"])
            j.CreateBody1Rel().SetTargets([f"{base}/Hopper"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.pivot_z))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-c.joint_limit_deg)
            j.CreateUpperLimitAttr(c.joint_limit_deg)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: stand home, basin teleported to a random SIDE (+x/-x) with xy
        jitter, hopper re-posed about the unchanged trunnion at a small start tilt,
        present-ball subset sampled and laid on the hopper floor (slot permutation +
        jitter), absent balls parked in the ground depot, latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- stand: fixed home pose ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.stand.write_root_state_to_sim(st, env_ids)

        # --- basin: side +-1, distance/y jitter ---
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.ones(m, device=dev), -torch.ones(m, device=dev))
        self.side[env_ids] = side.long()
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = side * (c.basin_cx + (torch.rand(m, device=dev) * 2 - 1) * c.basin_x_jitter)
        st[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.basin_y_jitter
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.basin.write_root_state_to_sim(st, env_ids)

        # --- hopper: small start tilt about the trunnion (pure joint-coordinate re-pose) ---
        th0 = torch.deg2rad((torch.rand(m, device=dev) * 2 - 1) * c.start_tilt_jitter_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 2] = c.pivot_z
        st[:, 3:7] = _qy(th0)
        st[:, 0:3] += origin
        self.hopper.write_root_state_to_sim(st, env_ids)

        # --- balls: subset sample, slot permutation, on the (slightly tilted) floor ---
        k = torch.randint(c.n_min, c.n_max + 1, (m,), device=dev)
        rank = torch.rand(m, c.n_max, device=dev).argsort(dim=1).argsort(dim=1)  # (m, 4)
        self.present[env_ids] = rank < k.unsqueeze(1)
        slots = torch.tensor([(-0.022, -0.022), (0.022, -0.022),
                              (-0.022, 0.022), (0.022, 0.022)], device=dev)  # (4, 2)
        zloc = c.floor_top_local + c.ball_r + 0.004
        cth, sth = torch.cos(th0), torch.sin(th0)
        for i in range(c.n_max):
            loc = slots[rank[:, i]] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.ball_jitter
            lx, ly = loc[:, 0], loc[:, 1]
            # hopper frame -> world through the start tilt (rotation about +y)
            ball_w = torch.stack([lx * cth + zloc * sth, ly,
                                  c.pivot_z - lx * sth + zloc * cth], dim=-1)
            park = torch.tensor([c.depot[0], c.depot[1], 0.0], device=dev).expand(m, 3).clone()
            park[:, 1] = park[:, 1] + 0.06 * i
            park[:, 2] = c.ball_r + 0.002
            pres = self.present[env_ids, i].unsqueeze(1)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.where(pres, ball_w, park)
            st[:, 3] = 1.0
            self.balls[i].write_root_state_to_sim(st, env_ids)

        self._tip_latch[env_ids] = False
        self.press_tau[env_ids] = 0.0
        self._th_prev[env_ids] = th0  # sync the FD history to the teleported tilt
        self._fd_rate[env_ids] = 0.0

    # ----- readings -----------------------------------------------------------------------------
    def hinge_rad(self) -> torch.Tensor:
        """(N,) hinge angle (rad, + = mouth tips toward +x). The hopper only ever
        rotates about the world-Y trunnion, so theta = 2*atan2(q_y, q_w)."""
        q = self.hopper.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 2], q[:, 0])

    def hinge_deg(self) -> torch.Tensor:
        return torch.rad2deg(self.hinge_rad())

    def hinge_rate(self) -> torch.Tensor:
        """(N,) signed hinge rate (rad/s) — finite-differenced from the hinge angle
        (the raw angular-velocity readback is unreliable under external wrenches)."""
        return self._fd_rate

    def _ball_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos_w (N,B,3), |lin vel| (N,B)) for all ball bodies."""
        pos = torch.stack([b.data.root_pos_w for b in self.balls], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.balls], dim=1)
        return pos, vel

    def in_basin(self) -> torch.Tensor:
        """(N, B) bool, geometric: ball centre inside the basin interior volume,
        below the rim."""
        c = self.cfg
        pos, _v = self._ball_tensors()
        bp = self.basin.data.root_pos_w[:, None, :]
        dx = (pos[:, :, 0] - bp[:, :, 0]).abs()
        dy = (pos[:, :, 1] - bp[:, :, 1]).abs()
        dz = pos[:, :, 2] - bp[:, :, 2]
        return ((dx < c.basin_in_x_half - c.basin_margin)
                & (dy < c.basin_in_y_half - c.basin_margin)
                & (dz > c.basin_floor_t) & (dz < c.basin_rim_z - 0.004))

    def in_hopper(self) -> torch.Tensor:
        """(N, B) bool: ball centre inside the hopper interior (hopper/pivot frame)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        pos, _v = self._ball_tensors()
        n, b = pos.shape[0], pos.shape[1]
        hq = self.hopper.data.root_quat_w[:, None, :].expand(n, b, 4).reshape(n * b, 4)
        hp = self.hopper.data.root_pos_w[:, None, :]
        loc = quat_apply_inverse(hq, (pos - hp).reshape(n * b, 3)).reshape(n, b, 3)
        return ((loc[:, :, 0].abs() < c.hop_in_half - 0.002)
                & (loc[:, :, 1].abs() < c.hop_in_half - 0.002)
                & (loc[:, :, 2] > c.floor_top_local - 0.005)
                & (loc[:, :, 2] < c.side_wall_top_local + 0.03))

    def frac_in_basin(self) -> torch.Tensor:
        """(N,) fraction of PRESENT balls currently inside the basin."""
        inb = self.in_basin() & self.present
        return inb.sum(dim=1).float() / self.present.sum(dim=1).clamp(min=1).float()

    def balls_settled(self) -> torch.Tensor:
        """(N,) bool: every present ball |lin vel| below `settle_speed`."""
        _p, vel = self._ball_tensors()
        return ((vel < self.cfg.settle_speed) | ~self.present).all(dim=1)

    def hopper_upright(self) -> torch.Tensor:
        """(N,) bool: hinge back within `upright_tol_deg` of vertical and settled."""
        c = self.cfg
        return ((self.hinge_deg().abs() < c.upright_tol_deg)
                & (self.hinge_rate().abs() < c.hinge_settle_omega))

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Hinge plant: return spring + viscous damping + the (clamped) external press
        torque, applied about the WORLD Y hinge axis (parallel to the rotation axis ->
        immune to the external-wrench frame-drag quirk). Then latch the signed tip."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        th = self.hinge_rad()
        # finite-difference rate (robust to the wrench-corrupted velocity readback)
        w = torch.nan_to_num((th - self._th_prev) / self.env.dt,
                             nan=0.0, posinf=0.0, neginf=0.0)
        self._th_prev = th.clone()
        self._fd_rate = w
        press = self.press_tau.clamp(-c.press_tau_max, c.press_tau_max)
        tau = press - c.spring_k * th - c.spring_c * w
        tau = torch.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
        ey = torch.tensor([0.0, 1.0, 0.0], device=dev)
        self.hopper.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), tau.reshape(n, 1, 1) * ey.view(1, 1, 3))
        # tip latch: signed — only a tip TOWARD the basin side ever arms it
        signed = torch.rad2deg(th) * self.side.float()
        signed = torch.nan_to_num(signed, nan=0.0, posinf=0.0, neginf=0.0)
        self._tip_latch |= signed >= c.drain_latch_deg

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: every present ball settled inside the basin AND the hopper
        released back upright at rest. All clauses are live physical outcomes."""
        pos, _v = self._ball_tensors()
        finite = torch.isfinite(pos).all(dim=-1).all(dim=-1) \
            & torch.isfinite(self.hopper.data.root_pos_w).all(dim=-1)
        all_in = (self.in_basin() | ~self.present).all(dim=1)
        return all_in & self.balls_settled() & self.hopper_upright() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15*tip_latch (latched, signed toward the basin)
        + 0.60*frac_in_basin (live) — max 0.75 without success; exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        base = c.w_latch * self._tip_latch.float() + c.w_frac * self.frac_in_basin()
        return torch.where(self.success(), torch.ones_like(base), base)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "hopper": self.hopper.data.root_state_w[env_ids].clone(),
            "basin": self.basin.data.root_state_w[env_ids].clone(),
            "balls": [b.data.root_state_w[env_ids].clone() for b in self.balls],
            "side": self.side[env_ids].clone(),
            "present": self.present[env_ids].clone(),
            "tip_latch": self._tip_latch[env_ids].clone(),
            "press_tau": self.press_tau[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.hopper.write_root_state_to_sim(state["hopper"], env_ids)
        self.basin.write_root_state_to_sim(state["basin"], env_ids)
        for b, s in zip(self.balls, state["balls"]):
            b.write_root_state_to_sim(s, env_ids)
        self.side[env_ids] = state["side"]
        self.present[env_ids] = state["present"]
        self._tip_latch[env_ids] = state["tip_latch"]
        self.press_tau[env_ids] = state["press_tau"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A trunnion DUMP STATION stands on the floor: two pillars carry an open-top "
            f"steel HOPPER (interior {2 * c.hop_in_half * 100:.0f} x "
            f"{2 * c.hop_in_half * 100:.0f} cm) that pivots about a horizontal axle at "
            f"height {c.pivot_z:.2f} m. A return spring holds the hopper upright. Inside "
            f"the hopper lie 2 to 4 loose BALLS ({2 * c.ball_r * 1000:.0f} mm across; "
            f"possible colors red, blue, orange, violet) — count what you see. Through the "
            f"axle, on the near side of the hopper, runs a horizontal ROCKER BAR with a "
            f"square YELLOW PEDAL at each end (pedal centres {c.pedal_x * 100:.0f} cm from "
            f"the axle, at axle height). Pressing a pedal DOWN tips the hopper toward that "
            f"pedal's side; two low orange SPOUT LIPS let the balls roll out once the tip "
            f"exceeds about {c.drain_deg:.0f} degrees. On ONE side of the station — left or "
            f"right, randomized each episode — sits an open green CATCH BASIN "
            f"({2 * c.basin_in_x_half * 100:.0f} x {2 * c.basin_in_y_half * 100:.0f} cm "
            f"inside, walls {c.basin_wall_h * 100:.0f} cm tall) directly under the hopper's "
            f"spill path on that side. The other side is bare floor.\n"
            f"Goal: get EVERY ball from the hopper into the green basin, then leave the "
            f"hopper hanging upright and still. The intended way: press and HOLD DOWN the "
            f"yellow pedal on the SAME side as the basin (about "
            f"{c.hold_deg:.0f} degrees of tip, roughly "
            f"{c.spring_k * math.radians(c.hold_deg) / c.pedal_x:.0f} N at the pedal), keep "
            f"holding while all balls roll over the lip and drop into the basin, then "
            f"release so the spring returns the hopper upright. Warnings: pressing the "
            f"pedal on the WRONG side dumps the balls onto the bare floor; releasing "
            f"before every ball has drained leaves stragglers in the hopper; a partial tip "
            f"below the drain angle spills nothing. Success is judged on the settled end "
            f"state: all present balls at rest INSIDE the basin, hopper upright and still."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Press and hold down the yellow pedal on the same side as the green basin "
            "until every ball drains from the tipping hopper into the basin, then release "
            "the pedal and let the hopper swing back upright. Do not tip the hopper "
            "toward the bare-floor side."
        )


# Scene-level task: no robot in the slot; the hinge is driven through `press_tau`.
register_env("simgen", lambda: EnvCfg(scene="tip_dump_station", robot="null"))
