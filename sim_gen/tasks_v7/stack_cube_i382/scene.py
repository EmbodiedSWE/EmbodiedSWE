"""SwingPumpScene — pump a caged pendulum with repeated timed pushes from a restricted
ground-level window until it swings past a one-way check flap and rests CAPTURED behind
it (sim_gen task `stack_cube_i382`).

Derived from maniskill/stack_cube, but STRATEGICALLY different: the seed is ONE precise
pick-and-place — grasp the red cube and set it statically on the blue cube so their
relative pose matches (a bbox detector on relative position is the whole rubric). Here
the red cube is WELDED to a rod as a pendulum bob: it can never be grasped free, carried,
or placed, and no relative object pose is ever judged. The judged quantity is a latched
DYNAMIC trajectory: mechanical ENERGY must be accumulated over several swing cycles by
repeated, timed, bounded pushes delivered only inside a small exposed window at the
bottom of the arc (the upper arc is sealed between two enclosure plates), until the bob
ballistically coasts up past a gravity-returned one-way CHECK FLAP high on the arc and
comes to rest trapped in the pocket behind it. The plan (resonant energy pumping through
an access-restricted window, then a ratchet capture) and the rubric (up-crossing
amplitude latch + flap-transit latch + captured rest) share nothing with the seed's
single static grasp-and-align, and no cube is ever stacked on, aligned to, or judged
against anything.

The apparatus (fully procedural, no external assets):
  - FRAME (heavy dynamic compound, 45 kg): ballasted base slab, two masts carrying the
    pendulum axle 0.55 m up, two vertical ENCLOSURE PLATES sandwiching the swing plane
    (62 mm slot for the 50 mm bob) from z = 0.357 m up to 0.86 m, end columns sealing
    the slot ends, and a top beam. The bob is reachable ONLY below the plates' lower
    edge — an arc window of about +/-50 deg around the bottom.
  - PENDULUM (one rigid body on a revolute Y-hinge to the frame, travel +/-132 deg):
    a steel rod with the RED 50 mm cube bob at radius 0.30 m; authored CoM at the bob.
  - CHECK FLAP (light amber blade on its own revolute Y-hinge to the frame, hinged
    OUTSIDE the bob's swept band at arc angle `flap_angle_deg` = ~100 deg on the frame's
    +x side, marked by an amber axle pin protruding through both plates): at rest
    gravity holds it against its closed stop, blocking the arc like a turnstile tooth.
    A bob rising along the arc shoves it open (free direction), passes, and the blade
    falls shut behind; the returning bob presses the blade toward its closed stop and
    is ARRESTED — a pure gravity ratchet. The pocket between flap and the +132 deg
    limit stop is deep inside the plate sandwich, unreachable from outside.

Physics makes multi-pass pumping mandatory (asserted in cfg.__post_init__): at the
declared push cap (2 N) the static hold angle (~32 deg) is far below the window edge
(~50 deg), and even a full window transit at the cap injects less energy than the climb
to the flap needs — so no single quasi-static push or single shove can reach the flap;
energy must be banked across swings and the last stretch is ALWAYS a hands-off
ballistic coast above the window.

Judged on PHYSICAL outcomes only: success() = pendulum resting still (pose-FD streak)
in the pocket band past the flap, flap re-closed, AND both trajectory latches earned —
lat_amp (a genuine continuous UP-crossing of the 60 deg amplitude line, above the
window, below the flap) and lat_pass (a genuine continuous crossing of the pass line
WHILE the flap was deflected open — a teleported pendulum fails the per-substep
continuity guard and a pendulum written into the pocket never deflects the flap).
score() latches monotonically: amplitude 0.30, flap transit 0.60, 1.0 iff success().

Per-episode randomization (readback-verifiable): whole-apparatus yaw (free, +/-180 deg)
+ xy jitter — the capture side points anywhere in the world — plus a random initial
pendulum angle (released swinging, random phase). Memorized world-frame push directions
fail; the solve reads the frame back every step.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _quat_y(deg: float) -> tuple:
    """wxyz quat for a rotation of `deg` about +Y."""
    h = math.radians(deg) / 2
    return (math.cos(h), 0.0, math.sin(h), 0.0)


# ----- custom compound spawners (frame / pendulum / check flap) ----------------------------------
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable, quat=None):
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if quat is not None:
        w, x, y, z = (float(v) for v in quat)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _rigid_dynamic(root, mass: float, *, lin_damp: float, ang_damp: float,
                   iters: int = 16, com: tuple | None = None,
                   inertia: tuple | None = None) -> None:
    """Dynamic rigid-body armor on a compound root: MassAPI mass, damping, no sleeping
    while poses are judged, depenetration cap. `com` authors an explicit local centre
    of mass — REQUIRED for the pendulum (PhysX otherwise keeps the CoM at the body
    origin = the hinge, killing gravity return and any pumping torque) and for the
    flap's gravity return. `inertia` authors a diagonal inertia about the CoM so the
    hinge dynamics are auditable (shape-derived inertia is opaque)."""
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
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _bind_grip(stage_path: str, mat_path: str, static: float, dynamic: float) -> None:
    """Author (once) and bind a friction material (custom spawner colliders otherwise
    get the ~0.5 default with no restitution control)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    import omni.usd
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(mat_path).IsValid():
        sim_utils.spawn_rigid_body_material(
            mat_path,
            sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic,
                                           restitution=0.0))
    bind_physics_material(stage_path, mat_path)


def _spawn_frame(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the FRAME: heavy DYNAMIC compound (local origin at ground level under the
    hinge). Ballasted base, two masts + axle 0.55 m up, two enclosure plates sandwiching
    the swing plane from `window_z` up, end columns sealing the slot ends, a top beam,
    and the amber flap axle pin (protrudes through both plates — the visible marker of
    the capture side). Frame<->pendulum and frame<->flap pairs are joint-filtered, so
    the visual axle/pin overlaps are free."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, c.frame_mass, lin_damp=1.0, ang_damp=1.0, com=(0.0, 0.0, 0.06))
    collide = _make_collide(c.contact_offset)
    dark = (0.23, 0.24, 0.27)
    light = (0.62, 0.64, 0.68)
    amber = (0.92, 0.66, 0.12)
    h = c.hinge_h
    pz0, pz1 = c.window_z, c.plate_top  # plate lower / upper edges
    # ballasted base slab
    _add_box(stage, f"{prim_path}/base", center=(0.0, 0.0, 0.011),
             size=(0.92, 0.30, 0.022), color=dark, collide=collide)
    # masts + pendulum axle (visual; the revolute joint is authored on the pendulum)
    for sgn, tag in ((1.0, "l"), (-1.0, "r")):
        _add_box(stage, f"{prim_path}/mast_{tag}", center=(0.0, sgn * 0.10, 0.451),
                 size=(0.036, 0.036, 0.878), color=dark, collide=collide)
    _add_box(stage, f"{prim_path}/axle", center=(0.0, 0.0, h),
             size=(0.024, 0.236, 0.024), color=light, collide=collide)
    # enclosure plates sandwiching the swing plane (62 mm slot for the 50 mm bob)
    for sgn, tag in ((1.0, "l"), (-1.0, "r")):
        _add_box(stage, f"{prim_path}/plate_{tag}",
                 center=(0.0, sgn * (c.slot_half_w + 0.004), (pz0 + pz1) / 2),
                 size=(0.84, 0.008, pz1 - pz0), color=light, collide=collide)
    # end columns sealing the slot ends (base to plate top)
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/endcol_{tag}", center=(sgn * 0.435, 0.0, 0.441),
                 size=(0.030, 0.100, 0.838), color=dark, collide=collide)
    # top beam tying the masts above the plates
    _add_box(stage, f"{prim_path}/topbeam", center=(0.0, 0.0, pz1 + 0.014),
             size=(0.90, 0.236, 0.026), color=dark, collide=collide)
    # amber flap axle pin: protrudes through both plates — the capture-side marker
    fh = c.flap_hinge_local
    _add_box(stage, f"{prim_path}/flap_pin", center=(fh[0], 0.0, fh[2]),
             size=(0.012, 0.088, 0.012), color=amber, collide=collide)
    _bind_grip(prim_path, "/World/simgenFrameMat", 0.70, 0.60)
    return root


def _spawn_pendulum(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the PENDULUM: one rigid compound, local origin AT the hinge; steel rod
    down to the RED cube bob at radius `L_bob`; authored CoM/inertia at the bob. Plus
    the revolute Y hinge to the sibling Frame (travel +/-`swing_limit_deg`)."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, c.pend_mass, lin_damp=0.0, ang_damp=0.02, iters=32,
                   com=(0.0, 0.0, -c.pend_com_d), inertia=tuple(c.pend_inertia_com))
    collide = _make_collide(c.contact_offset)
    steel = (0.42, 0.47, 0.58)
    red = (0.85, 0.15, 0.12)
    rod_len = c.L_bob - c.bob_size / 2
    _add_box(stage, f"{prim_path}/hub", center=(0.0, 0.0, 0.0),
             size=(0.026, 0.055, 0.026), color=steel, collide=collide)
    _add_box(stage, f"{prim_path}/rod", center=(0.0, 0.0, -rod_len / 2),
             size=(c.rod_w, c.rod_w, rod_len), color=steel, collide=collide)
    _add_box(stage, f"{prim_path}/bob", center=(0.0, 0.0, -c.L_bob),
             size=(c.bob_size, c.bob_size, c.bob_size), color=red, collide=collide)

    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Frame"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.hinge_h)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(c.swing_limit_deg))
    j.CreateUpperLimitAttr(float(c.swing_limit_deg))
    _bind_grip(prim_path, "/World/simgenPendMat", 0.40, 0.35)
    return root


def _spawn_flap(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the CHECK FLAP: light amber blade, local origin AT its hinge (radius
    `flap_radius` from the pendulum hinge at arc angle `flap_angle_deg`, OUTSIDE the
    bob's swept band), pointing inward across the band. The blade is HINGE-WEIGHTED
    (a thickened boss toward the hinge, visualized by the hubcap): authored CoM only
    `flap_com_d` out along the blade and a correspondingly small inertia, so gravity
    re-closes it FAST after a pass. Its revolute Y hinge to the sibling Frame has a
    closed backstop at -`flap_closed_stop_deg` (gravity holds the blade there — CoM
    torque is restoring over the whole travel, over-center is at ~`flap_angle_deg` deg
    open, limit `flap_open_limit_deg` keeps 20+ deg of margin) and a free opening
    direction (+) for a rising bob."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    bc = c.blade_center_local
    fc = c.flap_com_local
    _rigid_dynamic(root, c.flap_mass, lin_damp=0.05, ang_damp=0.03, iters=16,
                   com=(fc[0], 0.0, fc[2]),
                   inertia=tuple(c.flap_inertia_com))
    collide = _make_collide(c.contact_offset)
    amber = (0.92, 0.66, 0.12)
    _add_box(stage, f"{prim_path}/hubcap", center=(0.0, 0.0, 0.0),
             size=(0.016, 0.042, 0.016), color=amber, collide=collide)
    _add_box(stage, f"{prim_path}/blade", center=(bc[0], 0.0, bc[2]),
             size=(c.blade_len, c.blade_w, c.blade_t), color=amber, collide=collide,
             quat=_quat_y(90.0 - c.flap_angle_deg))

    base = prim_path.rsplit("/", 1)[0]
    fh = c.flap_hinge_local
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Frame"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in fh]))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    # POSITIVE joint angle = blade swept open (pass direction); lower limit = closed stop
    j.CreateLowerLimitAttr(-float(c.flap_closed_stop_deg))
    j.CreateUpperLimitAttr(float(c.flap_open_limit_deg))
    _bind_grip(prim_path, "/World/simgenFlapMat", 0.25, 0.20)
    return root


def _spawner_classes(cfg: SwingPumpSceneCfg) -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "frame" not in _SPAWNER_CACHE:

        @configclass
        class FrameSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_frame)
            frame_mass: float = 45.0
            hinge_h: float = 0.55
            window_z: float = 0.357
            plate_top: float = 0.86
            slot_half_w: float = 0.031
            flap_hinge_local: tuple = (0.369, 0.0, 0.615)
            contact_offset: float = 0.002

        @configclass
        class PendulumSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pendulum)
            pend_mass: float = 0.33
            pend_com_d: float = 0.285
            pend_inertia_com: tuple = (0.00104, 0.00104, 0.00015)
            L_bob: float = 0.30
            bob_size: float = 0.05
            rod_w: float = 0.014
            hinge_h: float = 0.55
            swing_limit_deg: float = 140.0
            contact_offset: float = 0.002

        @configclass
        class FlapSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_flap)
            flap_mass: float = 0.02
            flap_angle_deg: float = 100.0
            blade_len: float = 0.105
            blade_w: float = 0.040
            blade_t: float = 0.010
            blade_center_local: tuple = (-0.0517, 0.0, -0.0091)
            flap_com_local: tuple = (-0.0374, 0.0, -0.0066)
            flap_inertia_com: tuple = (3.0e-6, 1.35e-5, 1.5e-5)
            flap_hinge_local: tuple = (0.369, 0.0, 0.615)
            flap_open_limit_deg: float = 70.0
            flap_closed_stop_deg: float = 1.0
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(frame=FrameSpawnerCfg, pendulum=PendulumSpawnerCfg,
                              flap=FlapSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class SwingPumpSceneCfg(BaseCfg):
    """Config for `SwingPumpScene`. __post_init__ derives the flap/blade geometry from
    the arc parameters and ASSERTS the physics that makes the task honest: the declared
    push cap cannot hold the bob at the window edge, a single full window transit at
    the cap cannot pay for the climb to the flap (so multi-pass pumping is mandatory),
    the coast target clears the flap but not the travel limit, the blade covers the
    bob's swept band when closed and clears it at the open limit, and the flap's
    gravity return has 20+ deg of over-center margin."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    stage_scores: tuple = tunable((0.30, 0.60))  # latched credit: amplitude, flap transit
    amp_latch_deg: float = tunable(60.0)  # up-crossing amplitude latch (above window, below flap)
    pass_margin_deg: float = tunable(6.0)  # pass line = flap_angle + this (bob centre past blade)
    flap_pass_open_deg: float = tunable(25.0)  # flap must be at least this open AT the crossing
    cont_max_step_rad: float = tunable(0.05)  # per-substep continuity bound (teleports jump more)
    pocket_margin_deg: float = tunable(3.0)  # pocket band starts flap_angle + this
    flap_closed_deg: float = tunable(8.0)  # "flap re-closed" bound for success
    still_rate: float = tunable(0.25)  # rad/s, pose-FD stillness bound (phantom-vel proof)
    still_steps: int = tunable(60)  # substeps (0.5 s) the pendulum must hold still in the pocket

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_range_deg: float = tunable(180.0)  # whole-apparatus yaw, uniform +/- this
    pos_jitter: float = tunable(0.03)  # whole-apparatus xy jitter (m)
    theta0_range_deg: float = tunable(20.0)  # initial pendulum angle, uniform +/- this (at rest)

    # --- tunable: mechanism ----------------------------------------------------------------------
    pend_mass: float = tunable(0.33)  # kg (rod + bob)
    pend_com_d: float = tunable(0.285)  # authored CoM distance below the hinge (m)
    L_bob: float = tunable(0.30)  # bob centre radius (m)
    hinge_h: float = tunable(0.55)  # pendulum hinge height (m)
    swing_limit_deg: float = tunable(140.0)  # revolute travel (both sides)
    flap_angle_deg: float = tunable(100.0)  # arc angle of the check flap (frame +x side)
    flap_mass: float = tunable(0.02)  # kg (blade)
    flap_open_limit_deg: float = tunable(70.0)  # open stop (over-center is at ~flap_angle_deg)
    pump_force_max: float = tunable(2.0)  # N — the declared bounded push cap (solve + probes)
    target_over_deg: float = tunable(26.0)  # solve's coast target = flap_angle + this

    # --- info: fixed geometry (keep in sync with the spawners) -----------------------------------
    bob_size: float = info(0.05)  # the red cube bob edge (m)
    rod_w: float = info(0.014)
    window_z: float = info(0.357)  # enclosure plates' lower edge (bob exposed below this)
    plate_top: float = info(0.86)
    slot_half_w: float = info(0.031)  # plate inner face |y| (62 mm slot for the 50 mm bob)
    flap_radius: float = info(0.375)  # flap hinge radius from the pendulum hinge (m)
    blade_len: float = info(0.105)
    blade_w: float = info(0.040)
    blade_t: float = info(0.010)
    flap_com_d: float = info(0.038)  # hinge-weighted blade: CoM this far out along the blade
    flap_inertia_com: tuple = info((3.0e-6, 1.35e-5, 1.5e-5))  # about the authored CoM
    flap_closed_stop_deg: float = info(1.0)
    pend_inertia_com: tuple = info((0.00104, 0.00104, 0.00015))  # about the authored CoM
    frame_mass: float = info(45.0)
    contact_offset: float = info(0.002)
    apparatus_pos: tuple = info((0.0, 0.0))  # nominal frame-footprint centre
    sim_dt: float = info(1.0 / 120.0)

    def __post_init__(self) -> None:
        th_f = math.radians(self.flap_angle_deg)
        # derived geometry (frame-local; pendulum hinge at (0, 0, hinge_h))
        self.flap_hinge_local = (self.flap_radius * math.sin(th_f), 0.0,
                                 self.hinge_h - self.flap_radius * math.cos(th_f))
        d0 = (-math.sin(th_f), math.cos(th_f))  # closed blade direction (unit, xz)
        self.blade_center_local = (0.5 * self.blade_len * d0[0], 0.0,
                                   0.5 * self.blade_len * d0[1])
        self.flap_com_local = (self.flap_com_d * d0[0], 0.0, self.flap_com_d * d0[1])
        # derived dynamics
        self.mgd = self.pend_mass * _G * self.pend_com_d
        self.i_hinge = self.pend_inertia_com[1] + self.pend_mass * self.pend_com_d ** 2
        th_w = math.acos((self.hinge_h - self.window_z) / self.L_bob)  # window edge angle
        self.theta_w_deg = math.degrees(th_w)
        self.e_flap = self.mgd * (1.0 - math.cos(th_f))
        self.e_target = self.mgd * (1.0 - math.cos(th_f + math.radians(self.target_over_deg)))
        self.e_limit = self.mgd * (1.0 - math.cos(math.radians(self.swing_limit_deg)))
        self.theta_pass_deg = self.flap_angle_deg + self.pass_margin_deg
        self.pocket_lo_deg = self.flap_angle_deg + self.pocket_margin_deg
        self.pocket_hi_deg = self.swing_limit_deg - 1.0

        # --- honesty-by-construction asserts ------------------------------------------------
        # 1. the cap cannot statically hold the bob anywhere near the window edge
        hold_deg = math.degrees(math.atan(self.pump_force_max / (self.pend_mass * _G)))
        assert hold_deg < self.theta_w_deg - 10.0, \
            f"push cap holds statically to {hold_deg:.1f} deg — window edge {self.theta_w_deg:.1f}"
        # 2. one full window transit at the cap cannot pay for the climb to the flap
        e_single = self.pump_force_max * 2.0 * self.L_bob * math.sin(th_w)
        assert e_single < 0.95 * self.e_flap, \
            f"single-transit energy {e_single:.3f} J >= flap climb {self.e_flap:.3f} J"
        # 3. the coast target clears the flap with margin but stays well below the limit stop
        assert self.e_target < 0.92 * self.e_limit, "coast target too close to the travel limit"
        assert self.e_target > 1.10 * self.e_flap, "coast target margin over the flap too thin"
        # 4. flap gravity return: open limit keeps 20+ deg below over-center (~flap_angle)
        assert self.flap_open_limit_deg <= self.flap_angle_deg - 20.0, \
            "flap open limit too close to over-center — gravity return not guaranteed"
        # 5. closed blade covers the bob's swept band; hinge sits outside it
        r_sweep = self.L_bob + self.bob_size * math.sqrt(2.0) / 2.0
        tip_r = self.flap_radius - self.blade_len
        bob_inner = self.L_bob - self.bob_size / 2.0
        assert bob_inner - 0.008 < tip_r < bob_inner, \
            f"blade tip radius {tip_r:.3f} must sit just inside the bob face {bob_inner:.3f}"
        assert self.flap_radius >= r_sweep + 0.030, "flap hinge inside the bob's swept band"
        # 6. blade fully clears the swept band at the open limit
        phi = math.radians(self.flap_open_limit_deg)
        hp = (self.flap_radius * math.sin(th_f), -self.flap_radius * math.cos(th_f))
        dphi = (d0[0] * math.cos(phi) + d0[1] * math.sin(phi),
                d0[1] * math.cos(phi) - d0[0] * math.sin(phi))
        min_r = min(math.hypot(hp[0] + s * dphi[0], hp[1] + s * dphi[1])
                    for s in [self.blade_len * k / 10.0 for k in range(11)])
        assert min_r >= r_sweep + 0.015, \
            f"open blade min radius {min_r:.3f} clips the swept band {r_sweep:.3f}"
        # 7. rubric bands are coherent
        assert self.theta_w_deg + 5.0 <= self.amp_latch_deg <= self.flap_angle_deg - 20.0, \
            "amplitude latch must sit above the window edge and well below the flap"
        assert self.pocket_lo_deg + 2.0 < self.pocket_hi_deg, "pocket band empty"
        assert self.flap_angle_deg + self.target_over_deg <= self.swing_limit_deg - 6.0, \
            "coast target angle reaches the travel limit"
        # 8. the bob fits the slot; the pocket is inside the enclosure
        assert self.slot_half_w - self.bob_size / 2.0 >= 0.004, "bob does not clear the slot"
        pocket_z = self.hinge_h - self.L_bob * math.cos(math.radians(self.pocket_lo_deg))
        assert pocket_z > self.window_z + 0.05, "pocket not enclosed by the plates"
        # 9. the ratchet race: the blade re-closes (conservative bound, weakest-torque
        # accel over the whole fall) faster than the bob's ballistic dwell above the
        # pass line at the coast target — the returning bob ALWAYS meets a shut blade
        i_flap = self.flap_inertia_com[1] + self.flap_mass * self.flap_com_d ** 2
        a_min = (self.flap_mass * _G * self.flap_com_d
                 * math.sin(th_f - math.radians(self.flap_open_limit_deg)) / i_flap)
        t_fall = math.sqrt(2.0 * math.radians(self.flap_open_limit_deg) / a_min)
        w_pass = math.sqrt(2.0 * max(self.e_target - self.mgd
                                     * (1.0 - math.cos(math.radians(self.theta_pass_deg))),
                                     1e-9) / self.i_hinge)
        t_dwell = 2.0 * w_pass / (self.mgd / self.i_hinge)  # decel overestimated => lower bound
        assert t_dwell > 1.30 * t_fall, \
            f"ratchet race too tight: dwell {t_dwell:.3f} s vs blade fall {t_fall:.3f} s"


# ----- scene --------------------------------------------------------------------------------------
@SCENES.register("swing_pump")
class SwingPumpScene(BaseScene):
    cfg: SwingPumpSceneCfg

    def __init__(self, cfg: SwingPumpSceneCfg | None = None) -> None:
        super().__init__(cfg or SwingPumpSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes(c)
        ax, ay = c.apparatus_pos
        fh = c.flap_hinge_local
        # spawn order matters: the joints' body0 (Frame) must already exist
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.60, dynamic_friction=0.50, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "frame": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Frame",
                spawn=sp["frame"](frame_mass=c.frame_mass, hinge_h=c.hinge_h,
                                  window_z=c.window_z, plate_top=c.plate_top,
                                  slot_half_w=c.slot_half_w, flap_hinge_local=fh,
                                  contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(ax, ay, 0.0)),
            ),
            "pendulum": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pendulum",
                spawn=sp["pendulum"](pend_mass=c.pend_mass, pend_com_d=c.pend_com_d,
                                     pend_inertia_com=tuple(c.pend_inertia_com),
                                     L_bob=c.L_bob, bob_size=c.bob_size, rod_w=c.rod_w,
                                     hinge_h=c.hinge_h, swing_limit_deg=c.swing_limit_deg,
                                     contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(ax, ay, c.hinge_h)),
            ),
            "flap": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/CheckFlap",
                spawn=sp["flap"](flap_mass=c.flap_mass, flap_angle_deg=c.flap_angle_deg,
                                 blade_len=c.blade_len, blade_w=c.blade_w,
                                 blade_t=c.blade_t,
                                 blade_center_local=tuple(c.blade_center_local),
                                 flap_com_local=tuple(c.flap_com_local),
                                 flap_inertia_com=tuple(c.flap_inertia_com),
                                 flap_hinge_local=fh,
                                 flap_open_limit_deg=c.flap_open_limit_deg,
                                 flap_closed_stop_deg=c.flap_closed_stop_deg,
                                 contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(ax + fh[0], ay + fh[1], fh[2])),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=self.cfg.sim_dt,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "enable_external_forces_every_iteration": True,
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.frame: RigidObject = env.iscene["frame"]
        self.pendulum: RigidObject = env.iscene["pendulum"]
        self.flap: RigidObject = env.iscene["flap"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # monotonic latches (rubric credit that never evaporates)
        self.lat_amp = torch.zeros(n, dtype=torch.bool, device=dev)
        self.lat_pass = torch.zeros(n, dtype=torch.bool, device=dev)
        # per-substep pose history: continuity guard + pose-FD stillness streak
        self._th_prev = torch.zeros(n, device=dev)
        self._still_streak = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: one whole-apparatus pose (yaw free +/-yaw_range, xy jitter)
        written consistently to frame, pendulum (at its hinge, at a random initial
        angle, at rest) and flap (at its hinge, closed); latches and pose history
        cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(m, 2, device=dev)  # burn the (near-degenerate) first post-seed draw

        axy = torch.tensor(c.apparatus_pos, device=dev).unsqueeze(0).expand(m, 2).clone()
        axy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.pos_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_range_deg)
        q = _qz(yaw)
        th0 = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.theta0_range_deg)

        def write(body, lx: float, ly: float, z: float, quat: torch.Tensor) -> None:
            ca, sa = torch.cos(yaw), torch.sin(yaw)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = axy[:, 0] + ca * lx - sa * ly
            st[:, 1] = axy[:, 1] + sa * lx + ca * ly
            st[:, 2] = z
            st[:, 3:7] = quat
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        from isaaclab.utils.math import quat_mul

        write(self.frame, 0.0, 0.0, 0.0, q)
        # pendulum orientation q * Ry(-th0) puts the bob on the frame's +x side for th0 > 0
        write(self.pendulum, 0.0, 0.0, c.hinge_h, quat_mul(q, _qy(-th0)))
        fh = c.flap_hinge_local
        write(self.flap, fh[0], fh[1], fh[2], q)

        self.lat_amp[env_ids] = False
        self.lat_pass[env_ids] = False
        self._th_prev[env_ids] = th0
        self._still_streak[env_ids] = 0

    # ----- state (full, restorable) ---------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "frame": self.frame.data.root_state_w[env_ids].clone(),
            "pendulum": self.pendulum.data.root_state_w[env_ids].clone(),
            "flap": self.flap.data.root_state_w[env_ids].clone(),
            "latches": torch.stack([self.lat_amp[env_ids], self.lat_pass[env_ids]],
                                   dim=1).clone(),
            "th_prev": self._th_prev[env_ids].clone(),
            "still_streak": self._still_streak[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.frame.write_root_state_to_sim(state["frame"], env_ids)
        self.pendulum.write_root_state_to_sim(state["pendulum"], env_ids)
        self.flap.write_root_state_to_sim(state["flap"], env_ids)
        lat = state["latches"]
        self.lat_amp[env_ids] = lat[:, 0]
        self.lat_pass[env_ids] = lat[:, 1]
        self._th_prev[env_ids] = state["th_prev"]
        self._still_streak[env_ids] = state["still_streak"]

    # ----- frames / readbacks ---------------------------------------------------------------------
    def frame_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N, 3) world point expressed in the FRAME body frame (origin at ground level
        under the pendulum hinge)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.frame.data.root_quat_w,
                                  pos_w - self.frame.data.root_pos_w)

    def theta(self) -> torch.Tensor:
        """(N,) pendulum arc angle in rad from the hanging rest, POSITIVE toward the
        flap side (frame +x). Hinge-angle readback from the frame->pendulum relative
        quaternion (Y-revolute; the +x convention is the negative joint direction)."""
        from isaaclab.utils.math import quat_conjugate, quat_mul

        qr = quat_mul(quat_conjugate(self.frame.data.root_quat_w),
                      self.pendulum.data.root_quat_w)
        qr = torch.where(qr[:, 0:1] < 0, -qr, qr)
        return -2.0 * torch.atan2(qr[:, 2], qr[:, 0])

    def flap_open(self) -> torch.Tensor:
        """(N,) check-flap opening angle in rad, POSITIVE = blade swept toward the
        pocket (the pass direction); ~-flap_closed_stop at the gravity-held backstop."""
        from isaaclab.utils.math import quat_conjugate, quat_mul

        qr = quat_mul(quat_conjugate(self.frame.data.root_quat_w),
                      self.flap.data.root_quat_w)
        qr = torch.where(qr[:, 0:1] < 0, -qr, qr)
        return 2.0 * torch.atan2(qr[:, 2], qr[:, 0])

    def bob_pos_w(self) -> torch.Tensor:
        """(N, 3) world position of the bob centre (pendulum-pose readback)."""
        from isaaclab.utils.math import quat_apply

        v = torch.tensor([0.0, 0.0, -self.cfg.L_bob], device=self.env.device)
        v = v.unsqueeze(0).expand(self.env.num_envs, 3)
        return self.pendulum.data.root_pos_w + quat_apply(self.pendulum.data.root_quat_w, v)

    def bob_frame_local(self) -> torch.Tensor:
        return self.frame_local(self.bob_pos_w())

    def bob_exposed(self) -> torch.Tensor:
        """(N,) bool: bob centre below the enclosure plates' lower edge — the only
        region where a robot (or the solve's bounded push) can touch it."""
        return self.bob_frame_local()[:, 2] < self.cfg.window_z - 0.010

    def in_pocket(self) -> torch.Tensor:
        """(N,) bool: pendulum angle inside the capture pocket (past the flap, short of
        the travel limit)."""
        c = self.cfg
        th = self.theta()
        return ((th > math.radians(c.pocket_lo_deg)) & (th < math.radians(c.pocket_hi_deg)))

    def flap_closed(self) -> torch.Tensor:
        return self.flap_open().abs() < math.radians(self.cfg.flap_closed_deg)

    def pend_still(self) -> torch.Tensor:
        """(N,) bool: pose-FD stillness streak satisfied (immune to phantom-velocity
        readback spikes — judged on poses, teleports reset the streak)."""
        return self._still_streak >= self.cfg.still_steps

    # ----- step-coupled latching ------------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch (monotonic, every physics substep):
        - lat_amp: a genuine continuous UP-crossing of the amplitude line (|theta|
          rises through amp_latch within one substep's continuity bound — a teleport
          jumps the angle and earns nothing; a bob RESTING beyond the line never
          crosses it);
        - lat_pass: a genuine continuous crossing of the pass line WHILE the check
          flap is deflected open (a pendulum written into the pocket never deflects
          the flap; a teleport fails the continuity bound).
        Also maintains the pose-FD stillness streak used by success()."""
        c = self.cfg
        th = self.theta()
        dth = th - self._th_prev
        cont = dth.abs() < c.cont_max_step_rad
        amp = math.radians(c.amp_latch_deg)
        self.lat_amp |= cont & (self._th_prev.abs() < amp) & (th.abs() >= amp)
        pas = math.radians(c.theta_pass_deg)
        self.lat_pass |= (cont & (self._th_prev < pas) & (th >= pas)
                          & (self.flap_open() > math.radians(c.flap_pass_open_deg)))
        still = dth.abs() < c.still_rate * c.sim_dt
        self._still_streak = torch.where(still, self._still_streak + 1,
                                         torch.zeros_like(self._still_streak))
        self._th_prev = th

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A tall dark steel stand (about 0.9 m wide, 0.9 m high) rests on the floor; "
            "the whole apparatus may face any direction. From an axle between its two "
            "masts, 0.55 m up, hangs a rigid PENDULUM: a steel rod with a RED CUBE bob "
            "(5 cm) at 0.30 m radius, swinging in the stand's vertical mid-plane, and it "
            "may already be swaying gently. The upper arc is sealed between two light "
            "ENCLOSURE PLATES (a 62 mm slot, closed at both ends and on top): the bob "
            f"is exposed and touchable ONLY while it hangs below the plates' lower edge "
            f"(below ~{c.window_z:.2f} m, roughly +/-{c.theta_w_deg:.0f} deg of arc "
            "around the bottom). High on one side — the side marked by the small AMBER "
            f"AXLE PIN protruding through both plates — an amber CHECK FLAP blade "
            f"crosses the arc at ~{c.flap_angle_deg:.0f} deg up from the bottom, inside "
            "the enclosure. Gravity holds the blade shut against a backstop like a "
            "turnstile tooth: a bob rising along the arc shoves it open and can pass, "
            "then the blade falls shut behind, and anything pressing it from beyond "
            "just wedges it against the stop — a one-way gate.\n"
            "Goal: get the pendulum to rest CAPTURED in the pocket beyond the check "
            "flap. No single push can do it: from the exposed window a light push "
            f"(~{c.pump_force_max:.0f} N) can neither hold the bob near the window edge "
            "nor fling it to the flap in one pass. Instead, push the bob back and forth "
            "in time with its swing — always along its motion, only while it is below "
            "the plates — so the swing grows pass by pass, until the bob coasts up past "
            "the flap on the marked side and settles against the closed blade. The "
            "pendulum, rod, and bob are one rigid piece: the bob cannot be detached, "
            "grasped away, or carried, and the pocket is sealed inside the plates."
        )

    def instruction(self) -> str:
        return (
            "Push the red pendulum bob back and forth from the open zone under the "
            "enclosure, in rhythm with its swing, until it swings up past the amber "
            "one-way flap on the marked side and comes to rest trapped behind it."
        )

    # ----- rubric ----------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: pendulum resting still in the capture pocket with the check flap
        re-closed AND the trajectory genuinely earned — the amplitude up-crossing latch
        and the flap-transit latch both set. A pendulum that appears in the pocket
        without swinging there through the deflected flap is refused."""
        return (self.in_pocket() & self.flap_closed() & self.pend_still()
                & self.lat_amp & self.lat_pass)

    def score(self) -> torch.Tensor:
        """(N,) float: latched progress — genuine 60 deg amplitude up-crossing 0.30,
        genuine flap transit 0.60 — and 1.0 iff success(). Monotonic under correct
        behavior: the latches never clear, so credit survives the swing decaying or
        the bob being knocked back out of the pocket."""
        c = self.cfg
        s = torch.zeros(self.env.num_envs, device=self.env.device)
        s = torch.where(self.lat_amp, torch.full_like(s, c.stage_scores[0]), s)
        s = torch.where(self.lat_pass, torch.full_like(s, c.stage_scores[1]), s)
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("simgen", lambda: EnvCfg(scene="swing_pump", robot="null"))
