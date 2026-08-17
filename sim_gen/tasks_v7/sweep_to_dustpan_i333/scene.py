"""RiddleTrayScene — SORT a mixed load already sitting on a hinged riddle tray: pick the
orange cubes off the tray into an open crate by hand, then press the tray's red paddle
to TILT the tray against gravity so every marble rolls out through the low end-slot
into a sealed hopper; release, and let the tray fall back onto its rest stop.

Derived from rlbench/sweep_to_dustpan ("sweep dirt to dustpan": grasp a broom tool and
SWEEP five identical dirt cubes across the open ground into a wide, ground-level
dustpan mouth), but the MANIPULATION MODEL is replaced wholesale. The seed's plan is
tool-mediated planar transport of undifferentiated debris on the ground toward a
receptacle approachable from anywhere. Here that plan earns nothing:

  * the debris does not lie on the ground — it sits ON A MACHINE (a hinged sieve
    tray); nothing can be swept along the ground into the hopper, whose only mouth
    hides behind the tray's end-slot and a sub-ball-diameter gap ring;
  * the debris is MIXED and the two species part ways: marbles (d 22 mm) pass the
    28 mm end-slot, cubes (32 mm) geometrically cannot — and high friction keeps
    them put at full tilt, so the slot is a SIZE/SHAPE RIDDLE, not a door;
  * the transport actuator for the marbles is the TRAY ITSELF: the agent operates a
    mechanism (sustained paddle press against the tray's own weight) instead of
    dragging debris with a tool; releasing early or never tilting discharges nothing
    (the rest RECLINE drains the bed AWAY from the slot, so the null tray is
    strictly retentive);
  * the cubes go the opposite way, by per-piece pick-and-place into a crate that is
    randomized on the ground — two destinations, two mechanics, one mixed pile.

Assets are fully procedural (compound-spawner pattern; child colliders of one body
never self-collide; the dock<->tray joint pair never collides):

  - dock: KINEMATIC stand at the hinge point. Two ground posts carry the hinge; a
    sealed HOPPER box (solid floor, side walls, back wall, roof) sits beyond it. The
    hopper's only mouth faces the tray's discharge end, and every gap around that
    mouth (end gap 16 mm, sky slot 20 mm, flank slits 6 mm) is narrower than a
    marble, so nothing can be posted in by hand or dropped in from above.
  - tray: DYNAMIC riddle tray, LOCAL ORIGIN AT THE HINGE AXIS, joined to the dock by
    a spawn-authored Y-axis revolute joint, limits [rest_deg, tilt_max_deg] =
    [-2.5, +12] deg. Gravity (CoM well behind the hinge) rests it reclined on the
    LOWER stop, so at rest everything aboard rolls AWAY from the slot. The discharge
    end carries a full-width slot (28 mm tall) under a header, a canopy sealing the
    last 150 mm of the top (nothing can be dropped straight into the slot region),
    and a raised red PADDLE on a beam beyond the hinge — pressing the paddle down
    (~0.75 N*m tray torque + payload) is the one handle that tilts it.
  - crate: KINEMATIC open-top crate on the ground, pose randomized per episode.
  - pieces: up to 4 steel marbles (spheres r 11 mm) and up to 3 orange cubes
    (32 mm), spawned on the tray bed behind the ridge; counts sampled per episode.

Per-episode randomization (readback-verifiable): marble count in {2..4}, cube count
in {1..3}, piece-to-slot permutation over 7 tray-bed slots with xy jitter + cube free
yaw, crate xy + yaw. The dock and tray poses are FIXED on purpose: the hinge anchor
is spawn-authored in the dock's frame and a teleported fixture leaves its joint
anchor behind (measured house quirk), so the fixture is never randomized.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.20 * crated   — latched per cube (in the crate and calm), fraction of PRESENT
  0.15 * tilted   — tray ever past tilt_credit_deg while a present marble was still
                    aboard (a genuine discharge attempt; empty-tilt stays dark)
  0.45 * binned   — latched per marble (inside the hopper, below the mouth),
                    fraction of PRESENT marbles
  1.0 iff success() — every present marble settled in the hopper, every present cube
                    settled in the crate, the tray back at rest on its stop, all
                    finite. Non-success capped at 0.80.

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


# ----- custom compound spawners ------------------------------------------------------------------
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


def _add_box(stage, path: str, *, center, size, color, contact_offset: float):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(box.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    return box.GetPrim()


def _bind_material(stage, root_path: str, prims, *, static: float, dynamic: float,
                   restitution: float):
    """Author one physics material under the compound root and bind it to `prims`.
    Custom spawn funcs get NO material from any Cfg schema (measured house quirk:
    they default to ~0.5 friction), so friction that the task's margins rest on is
    authored explicitly here."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, f"{root_path}/physmat")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(static))
    api.CreateDynamicFrictionAttr(float(dynamic))
    api.CreateRestitutionAttr(float(restitution))
    for p in prims:
        UsdShade.MaterialBindingAPI.Apply(p).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_dock(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC dock compound. Local origin: THE HINGE AXIS POINT (0.14 m up).
    Children: two ground posts + axle stubs at the hinge, the sealed hopper box
    beyond it (+x), a front wall under the hopper mouth, a roof whose front edge
    leaves only a 20 mm sky slot, and two flank seals closing the mouth sides.
    Every opening around the mouth is narrower than a marble."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    c = cfg
    co = c.contact_offset
    kids = []
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        kids.append(_add_box(stage, f"{prim_path}/post_{tag}",
                             center=(0.0, sgn * 0.150, -0.065), size=(0.030, 0.030, 0.150),
                             color=c.frame_color, contact_offset=co))
        kids.append(_add_box(stage, f"{prim_path}/axle_{tag}",
                             center=(0.0, sgn * 0.140, 0.0), size=(0.012, 0.018, 0.012),
                             color=c.frame_color, contact_offset=co))
        kids.append(_add_box(stage, f"{prim_path}/seal_{tag}",
                             center=(0.005, sgn * 0.1405, -0.005), size=(0.050, 0.009, 0.110),
                             color=c.frame_color, contact_offset=co))
        kids.append(_add_box(stage, f"{prim_path}/hop_side_{tag}",
                             center=(0.133, sgn * 0.126, -0.0475), size=(0.230, 0.008, 0.185),
                             color=c.hopper_color, contact_offset=co))
    kids.append(_add_box(stage, f"{prim_path}/hop_front",
                         center=(c.wall_x, 0.0, -0.0975), size=(c.wall_t, 0.260, 0.085),
                         color=c.hopper_color, contact_offset=co))
    kids.append(_add_box(stage, f"{prim_path}/hop_floor",
                         center=(0.133, 0.0, -0.134), size=(0.230, 0.260, 0.012),
                         color=c.hopper_color, contact_offset=co))
    kids.append(_add_box(stage, f"{prim_path}/hop_back",
                         center=(0.244, 0.0, -0.0475), size=(0.008, 0.260, 0.185),
                         color=c.hopper_color, contact_offset=co))
    kids.append(_add_box(stage, f"{prim_path}/hop_roof",
                         center=(0.134, 0.0, 0.049), size=(0.228, 0.260, 0.008),
                         color=c.roof_color, contact_offset=co))
    _bind_material(stage, prim_path, kids, static=0.60, dynamic=0.55, restitution=0.0)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC riddle tray, LOCAL ORIGIN AT THE HINGE AXIS (so the revolute joint
    pose is pos-only). The bed spans local x in [-0.42, 0]; the discharge slot is at
    x = 0 under a header; positive hinge angle lifts the back (-x) end so the bed
    drains toward the slot. Explicit MassAPI (mass, CoM, diagonal inertia — PhysX
    stops deriving CoM once MassAPI is authored, so author everything)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.20)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    c = cfg
    co = c.contact_offset
    kids = [
        _add_box(stage, f"{prim_path}/floor", center=(-0.210, 0.0, -0.006),
                 size=(0.420, 0.260, 0.012), color=c.bed_color, contact_offset=co),
        _add_box(stage, f"{prim_path}/back", center=(-0.416, 0.0, 0.024),
                 size=(0.008, 0.260, 0.060), color=c.bed_color, contact_offset=co),
        _add_box(stage, f"{prim_path}/header", center=(-0.004, 0.0, 0.039),
                 size=(0.008, 0.260, 0.022), color=c.bed_color, contact_offset=co),
        _add_box(stage, f"{prim_path}/canopy", center=(-0.075, 0.0, 0.040),
                 size=(0.150, 0.260, 0.008), color=c.canopy_color, contact_offset=co),
        _add_box(stage, f"{prim_path}/beam", center=(0.085, 0.0, 0.116),
                 size=(0.190, 0.016, 0.008), color=c.frame_color, contact_offset=co),
        _add_box(stage, f"{prim_path}/paddle", center=(c.paddle_x, 0.0, 0.118),
                 size=(0.040, 0.120, 0.012), color=c.paddle_color, contact_offset=co),
    ]
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        kids.append(_add_box(stage, f"{prim_path}/fence_{tag}",
                             center=(-0.210, sgn * 0.126, 0.022), size=(0.420, 0.008, 0.056),
                             color=c.bed_color, contact_offset=co))
        kids.append(_add_box(stage, f"{prim_path}/hpost_{tag}",
                             center=(-0.004, sgn * 0.050, 0.085), size=(0.008, 0.008, 0.070),
                             color=c.frame_color, contact_offset=co))
    _bind_material(stage, prim_path, kids, static=c.mu_tray, dynamic=c.mu_tray - 0.05,
                   restitution=0.0)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(c.tray_mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in c.tray_com]))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in c.tray_inertia]))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    return root


def _spawn_crate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC open-top crate. Local origin: footprint centre on the ground."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    c = cfg
    co = c.contact_offset
    ih, t, h = c.in_hw, c.wall_t, c.wall_h
    oh = ih + t
    kids = [_add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.floor_t / 2),
                     size=(2 * oh, 2 * oh, c.floor_t), color=c.crate_color, contact_offset=co)]
    zc = c.floor_t + h / 2
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        kids.append(_add_box(stage, f"{prim_path}/wx_{tag}",
                             center=(sgn * (ih + t / 2), 0.0, zc), size=(t, 2 * oh, h),
                             color=c.crate_color, contact_offset=co))
        kids.append(_add_box(stage, f"{prim_path}/wy_{tag}",
                             center=(0.0, sgn * (ih + t / 2), zc), size=(2 * ih, t, h),
                             color=c.crate_color, contact_offset=co))
    _bind_material(stage, prim_path, kids, static=0.80, dynamic=0.75, restitution=0.0)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "dock" not in _SPAWNER_CACHE:

        @configclass
        class DockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_dock)
            wall_x: float = 0.020
            wall_t: float = 0.008
            frame_color: tuple = (0.30, 0.32, 0.36)
            hopper_color: tuple = (0.42, 0.44, 0.48)
            roof_color: tuple = (0.34, 0.36, 0.40)
            contact_offset: float = 0.002

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            paddle_x: float = 0.180
            mu_tray: float = 0.55
            tray_mass: float = 0.42
            tray_com: tuple = (-0.18, 0.0, 0.01)
            tray_inertia: tuple = (0.004, 0.010, 0.013)
            bed_color: tuple = (0.35, 0.40, 0.48)
            canopy_color: tuple = (0.26, 0.30, 0.38)
            frame_color: tuple = (0.30, 0.32, 0.36)
            paddle_color: tuple = (0.85, 0.12, 0.10)
            contact_offset: float = 0.002

        @configclass
        class CrateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_crate)
            in_hw: float = 0.080
            wall_t: float = 0.008
            wall_h: float = 0.055
            floor_t: float = 0.010
            crate_color: tuple = (0.55, 0.38, 0.20)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(dock=DockSpawnerCfg, tray=TraySpawnerCfg, crate=CrateSpawnerCfg)
    return _SPAWNER_CACHE


def _qz(ang: torch.Tensor) -> torch.Tensor:
    """Yaw quaternions (N,4) from angles (N,)."""
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class RiddleTraySceneCfg(BaseCfg):
    """Config for `RiddleTrayScene`. The claims the task rests on are geometric or
    frictional and asserted in `__post_init__`: marbles pass the slot and cubes
    cannot; every gap around the hopper mouth is narrower than a marble; cubes stay
    put at full tilt; the rest recline retains marbles against solver creep; spawn
    bands keep clearance from static walls and stay clear of the canopy."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    ball_settle: float = tunable(0.10)   # max marble |lin vel| when judging success (m/s;
    # above the measured GPU phantom-velocity band)
    cube_settle: float = tunable(0.10)   # max cube |lin vel| when judging success (m/s)
    latch_speed: float = tunable(0.15)   # max |lin vel| for the crate latch to arm
    tilt_credit_deg: float = tunable(8.0)   # tilt latch: tray past this while a marble aboard
    rest_max_deg: float = tunable(-1.0)  # success: tray angle back below this (rest stop
    # sits at -2.5 deg; a tray held up by anything reads higher)
    rest_still: float = tunable(0.35)    # success: max tray |ang vel| (rad/s; above the
    # phantom band, far below any real swing)
    hop_z_max: float = tunable(-0.080)   # dock-frame z bound for "in the hopper": well
    # BELOW the mouth (mouth sill -0.055, marble rest -0.117) — containment is judged
    # below the aperture, so doorway transients never count
    crate_xy: float = tunable(0.072)     # crate-frame |x|,|y| bound (interior half 0.080,
    # cube half 0.016: a wall-hugging cube centre sits at 0.064)
    crate_z_lo: float = tunable(0.015)   # crate-frame z band (floor top 0.010; resting
    crate_z_hi: float = tunable(0.075)   # cube centre 0.026; a wall-top perch reads 0.081)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    min_balls: int = tunable(2)          # marble count sampled in {min_balls .. n_balls}
    min_cubes: int = tunable(1)          # cube count sampled in {min_cubes .. n_cubes}
    slot_jitter: float = tunable(0.012)  # per-piece xy jitter at its bed slot (+/- m)
    cube_yaw_deg: float = tunable(180.0)  # per-cube free yaw (+/- deg)
    crate_jitter: float = tunable(0.05)  # crate xy jitter (+/- m)
    crate_yaw_deg: float = tunable(30.0)  # crate yaw (+/- deg)

    # --- info: fixture layout (world; the dock/tray pose is FIXED on purpose — the
    # hinge anchor is spawn-authored in the dock's frame and a teleported fixture
    # leaves its joint anchor behind, so the fixture is never randomized) -----------------------
    hinge_pos: tuple = info((0.30, 0.0, 0.14))  # world hinge axis point (axis = world y)
    rest_deg: float = info(-2.5)         # lower joint stop: reclined rest (drains BACKWARD)
    tilt_max_deg: float = info(12.0)     # upper joint stop: full discharge tilt
    # --- info: tray geometry (mirrors the spawner defaults; single source here) ------------------
    floor_len: float = info(0.42)        # bed length (local x in [-floor_len, 0])
    floor_t: float = info(0.012)
    tray_hw: float = info(0.13)          # outer half width (fence outer face)
    fence_t: float = info(0.008)
    slot_h: float = info(0.028)          # discharge slot height (bed top -> header bottom)
    canopy_z0: float = info(0.036)       # canopy underside (cube must fit beneath)
    canopy_len: float = info(0.15)
    paddle_x: float = info(0.180)        # paddle centre (local x, beyond the hinge)
    paddle_z0: float = info(0.112)       # beam/paddle underside (local z)
    tray_mass: float = info(0.42)
    tray_com: tuple = info((-0.18, 0.0, 0.01))
    tray_inertia: tuple = info((0.004, 0.010, 0.013))
    mu_tray: float = info(0.55)
    # --- info: dock / hopper geometry (dock frame = hinge frame) ---------------------------------
    wall_x: float = info(0.020)          # hopper front wall centre (local x)
    wall_t: float = info(0.008)
    wall_top_z: float = info(-0.055)     # front wall top = the mouth sill
    roof_front_x: float = info(0.020)    # roof front edge (local x) -> 20 mm sky slot
    roof_z0: float = info(0.045)         # roof underside
    hop_x_in: tuple = info((0.024, 0.240))  # hopper interior x span
    hop_hw: float = info(0.122)          # hopper interior half width
    hop_floor_top: float = info(-0.128)  # hopper floor top (marble rest z -0.117)
    # --- info: pieces ----------------------------------------------------------------------------
    ball_r: float = info(0.011)
    ball_mass: float = info(0.03)
    n_balls: int = info(4)
    cube_size: float = info(0.032)
    cube_mass: float = info(0.08)
    n_cubes: int = info(3)
    mu_cube: float = info(1.0)
    ball_color: tuple = info((0.62, 0.65, 0.70))
    cube_color: tuple = info((0.90, 0.45, 0.08))
    # 7 bed slots (tray-local xy), all behind the ridge, clear of walls and pairwise
    slots: tuple = info(((-0.365, -0.075), (-0.365, 0.075), (-0.295, -0.075),
                         (-0.295, 0.075), (-0.225, -0.075), (-0.225, 0.075),
                         (-0.335, 0.0)))
    spawn_hover: float = info(0.008)     # spawn drop height along the tray normal
    depot: tuple = info((1.40, 1.40))    # off-stage ground parking for absent pieces
    # --- info: crate -----------------------------------------------------------------------------
    crate_pos: tuple = info((0.02, 0.40))
    crate_in_hw: float = info(0.080)
    crate_wall_t: float = info(0.008)
    crate_wall_h: float = info(0.055)
    crate_floor_t: float = info(0.010)
    contact_offset: float = info(0.002)
    # rubric weights (0.20 + 0.15 + 0.45 = 0.80 = the non-success cap)
    w_crate: float = info(0.20)
    w_tilt: float = info(0.15)
    w_bin: float = info(0.45)
    cap: float = info(0.80)

    def __post_init__(self) -> None:
        s, tilt = math.sin(math.radians(self.tilt_max_deg)), math.radians(self.tilt_max_deg)
        ball_d = 2 * self.ball_r
        # the slot is a RIDDLE: marbles pass, cubes geometrically cannot
        assert ball_d <= self.slot_h - 0.003, "slot must pass a marble with margin"
        assert self.cube_size >= self.slot_h + 0.003, "slot must refuse a cube with margin"
        # a cube still fits UNDER the canopy (it must ride the bed, just never exit)
        assert self.canopy_z0 >= self.cube_size + 0.003, "canopy would trap a cube"
        # every gap around the hopper mouth is narrower than a marble:
        # (a) end gap tray->front wall, at its widest (tray end retreats when tilted)
        gap = (self.wall_x - self.wall_t / 2) + self.floor_t * s
        assert gap < ball_d - 0.003, "end gap must not pass a marble"
        # (b) sky slot between the tray end plane and the roof front edge
        assert self.roof_front_x < ball_d - 0.001, "sky slot must not pass a marble"
        # cubes stay put on the bed at full tilt (pair-averaged contact friction)
        assert (self.mu_cube + self.mu_tray) / 2 >= 2 * math.tan(tilt), \
            "cube friction margin under 2x at full tilt"
        # null retention: the rest RECLINE drains the bed away from the slot, and it is
        # steep enough that solver-creep (~0.04 m/s band) cannot climb it toward the
        # slot (creep KE buys < 0.2 mm of height; the recline costs sin(rest)*run >> that)
        assert self.rest_deg <= -1.5, "rest recline too shallow to retain marbles"
        assert abs(math.sin(math.radians(self.rest_deg))) * 0.05 \
            > 5 * (0.7 * 0.04 ** 2 / 9.81), "recline below the creep band"
        # spawn band clearance from the back wall / fences (worst cube corner, ~5 mm rule)
        corner = self.cube_size / 2 * math.sqrt(2)
        assert min(x for x, _y in self.slots) - self.slot_jitter - corner \
            > -(self.floor_len - self.fence_t) + 0.004, "spawn band too close to the back wall"
        assert max(abs(y) for _x, y in self.slots) + self.slot_jitter + corner \
            < self.tray_hw - self.fence_t - 0.004, "spawn band too close to a fence"
        # the spawn band stays BEHIND the canopy: cubes must be graspable from above
        assert max(x for x, _y in self.slots) + self.slot_jitter + corner \
            < -self.canopy_len - 0.004, "spawn band reaches under the canopy"
        # the paddle/beam clears the hopper roof through the whole sweep (cosmetic set
        # is collision-filtered via the joint pair, but the camera should see daylight)
        z_lo = -(self.paddle_x + 0.020) * s + self.paddle_z0 * math.cos(tilt)
        assert z_lo > self.roof_z0 + self.wall_t + 0.008, "paddle sweeps into the hopper roof"
        # "in the hopper" is judged BELOW the mouth sill
        assert self.hop_z_max <= self.wall_top_z - 0.02, "hop_z_max must sit below the mouth"
        # crate: three cubes fit side by side; the xy bound admits a wall-hugger
        assert 2 * self.crate_in_hw > 3 * self.cube_size + 0.03, "crate too small for 3 cubes"
        assert self.crate_xy > self.crate_in_hw - self.cube_size / 2 - 0.010, \
            "crate_xy rejects a wall-hugging cube"


# ----- scene -------------------------------------------------------------------------------------
@SCENES.register("riddle_tray")
class RiddleTrayScene(BaseScene):
    cfg: RiddleTraySceneCfg

    BALL_NAMES = ("ball_0", "ball_1", "ball_2", "ball_3")
    CUBE_NAMES = ("cube_0", "cube_1", "cube_2")

    def __init__(self, cfg: RiddleTraySceneCfg | None = None) -> None:
        super().__init__(cfg or RiddleTraySceneCfg())

    # ----- assets --------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        hx, hy, hz = c.hinge_pos
        half = math.radians(c.rest_deg) / 2
        rest_quat = (math.cos(half), 0.0, math.sin(half), 0.0)
        ball_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.20, angular_damping=0.20,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.50, dynamic_friction=0.45, restitution=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
        )
        cube_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.05, angular_damping=0.05,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=c.mu_cube, dynamic_friction=c.mu_cube - 0.05,
                restitution=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
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
            "dock": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dock",
                spawn=cls["dock"](contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, hz)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=cls["tray"](tray_mass=c.tray_mass, mu_tray=c.mu_tray,
                                  contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, hz), rot=rest_quat),
            ),
            "crate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Crate",
                spawn=cls["crate"](contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.crate_pos[0], c.crate_pos[1], 0.0)),
            ),
        }
        for i, name in enumerate(self.BALL_NAMES):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball_" + name,
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.ball_color, roughness=0.25),
                    **ball_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.depot[0] + 0.1 * i, c.depot[1], c.ball_r + 0.003)),
            )
        for i, name in enumerate(self.CUBE_NAMES):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube_size,) * 3,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.cube_color),
                    **cube_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.depot[0] + 0.1 * i, c.depot[1] + 0.15, c.cube_size / 2 + 0.003)),
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
        c = self.cfg
        self.dock: RigidObject = env.iscene["dock"]
        self.tray: RigidObject = env.iscene["tray"]
        self.crate: RigidObject = env.iscene["crate"]
        self.balls: dict[str, RigidObject] = {n: env.iscene[n] for n in self.BALL_NAMES}
        self.cubes: dict[str, RigidObject] = {n: env.iscene[n] for n in self.CUBE_NAMES}
        self.env_origins = env.iscene.env_origins
        self._author_hinge()
        n = env.num_envs
        dev = env.device
        self.present_ball = torch.ones(n, c.n_balls, dtype=torch.bool, device=dev)
        self.present_cube = torch.ones(n, c.n_cubes, dtype=torch.bool, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._ball_l = torch.zeros(n, c.n_balls, dtype=torch.bool, device=dev)
        self._cube_l = torch.zeros(n, c.n_cubes, dtype=torch.bool, device=dev)
        self._tilt = torch.zeros(n, dtype=torch.bool, device=dev)

    def _author_hinge(self) -> None:
        """Per env: a Y-axis revolute joint dock->tray at the shared hinge origin.
        Limits [rest_deg, tilt_max_deg]: gravity (tray CoM behind the hinge) rests
        the tray reclined on the LOWER stop; pressing the paddle (beyond the hinge)
        drives it toward the upper stop. The joint pair never collides. The dock is
        kinematic and never teleported, so the anchor stays true across resets."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/tray_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Dock"])
            j.CreateBody1Rel().SetTargets([f"{base}/Tray"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(float(c.rest_deg))
            j.CreateUpperLimitAttr(float(c.tilt_max_deg))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: dock re-pinned, tray re-seated on its rest stop, crate
        re-randomized on the ground, piece counts sampled, the 7 pieces permuted
        over the 7 bed slots with jitter (+ cube free yaw), absent pieces parked in
        the depot, latches cleared. Pieces spawn a small hover above the reclined
        bed and drop in (the drop torques the tray INTO its stop, never past it)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        hx, hy, hz = c.hinge_pos
        rest = math.radians(c.rest_deg)
        cosr, sinr = math.cos(rest), math.sin(rest)

        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = hx, hy, hz
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.dock.write_root_state_to_sim(st, env_ids)

        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = hx, hy, hz
        st[:, 3], st[:, 5] = math.cos(rest / 2), math.sin(rest / 2)
        st[:, 0:3] += origin
        self.tray.write_root_state_to_sim(st, env_ids)

        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.crate_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.crate_jitter
        st[:, 1] = c.crate_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.crate_jitter
        st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1)
                         * math.radians(c.crate_yaw_deg))
        st[:, 0:3] += origin
        self.crate.write_root_state_to_sim(st, env_ids)

        # piece counts (burn a draw first: the first post-seed draw is degenerate)
        _ = torch.rand(m, device=dev)
        span_b = c.n_balls - c.min_balls + 1
        kb = c.min_balls + (torch.rand(m, device=dev) * span_b).long().clamp(max=span_b - 1)
        span_c = c.n_cubes - c.min_cubes + 1
        kc = c.min_cubes + (torch.rand(m, device=dev) * span_c).long().clamp(max=span_c - 1)
        rank = torch.rand(m, c.n_balls, device=dev).argsort(dim=1).argsort(dim=1)
        self.present_ball[env_ids] = rank < kb.unsqueeze(1)
        rank = torch.rand(m, c.n_cubes, device=dev).argsort(dim=1).argsort(dim=1)
        self.present_cube[env_ids] = rank < kc.unsqueeze(1)

        # permute the 7 pieces over the 7 bed slots (tray-local), hover above the bed
        n_pc = c.n_balls + c.n_cubes
        perm = torch.rand(m, n_pc, device=dev).argsort(dim=1)
        slots = torch.tensor(c.slots, device=dev, dtype=torch.float)  # (7, 2)
        yaw_amp = math.radians(c.cube_yaw_deg)
        for i, name in enumerate(self.BALL_NAMES + self.CUBE_NAMES):
            is_ball = i < c.n_balls
            body = self.balls[name] if is_ball else self.cubes[name]
            h = (c.ball_r if is_ball else c.cube_size / 2) + c.spawn_hover
            xy = slots[perm[:, i]] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            lx, ly = xy[:, 0], xy[:, 1]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = hx + lx * cosr + h * sinr
            st[:, 1] = hy + ly
            st[:, 2] = hz - lx * sinr + h * cosr
            if is_ball:
                st[:, 3] = 1.0
            else:
                st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * yaw_amp)
            park = torch.zeros(m, 3, device=dev)
            j = i if is_ball else i - c.n_balls
            park[:, 0] = c.depot[0] + 0.1 * j
            park[:, 1] = c.depot[1] + (0.0 if is_ball else 0.15)
            park[:, 2] = (c.ball_r if is_ball else c.cube_size / 2) + 0.003
            pres = (self.present_ball[env_ids, j] if is_ball
                    else self.present_cube[env_ids, j]).unsqueeze(1)
            st[:, 0:3] = torch.where(pres, st[:, 0:3], park)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        self._ball_l[env_ids] = False
        self._cube_l[env_ids] = False
        self._tilt[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "crate": self.crate.data.root_state_w[env_ids].clone(),
            "balls": {n: b.data.root_state_w[env_ids].clone() for n, b in self.balls.items()},
            "cubes": {n: b.data.root_state_w[env_ids].clone() for n, b in self.cubes.items()},
            "present_ball": self.present_ball[env_ids].clone(),
            "present_cube": self.present_cube[env_ids].clone(),
            "ball_l": self._ball_l[env_ids].clone(),
            "cube_l": self._cube_l[env_ids].clone(),
            "tilt": self._tilt[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.crate.write_root_state_to_sim(state["crate"], env_ids)
        for n, b in self.balls.items():
            b.write_root_state_to_sim(state["balls"][n], env_ids)
        for n, b in self.cubes.items():
            b.write_root_state_to_sim(state["cubes"][n], env_ids)
        self.present_ball[env_ids] = state["present_ball"]
        self.present_cube[env_ids] = state["present_cube"]
        self._ball_l[env_ids] = state["ball_l"]
        self._cube_l[env_ids] = state["cube_l"]
        self._tilt[env_ids] = state["tilt"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A RIDDLE TRAY — a {c.floor_len * 1000:.0f} x {2 * c.tray_hw * 1000:.0f} mm "
            f"steel sieve tray on a hinge — stands {c.hinge_pos[2] * 1000:.0f} mm up on a "
            f"dock, reclined on its rest stop so its bed drains AWAY from the discharge "
            f"end. On the bed lies a MIXED load: between {c.min_balls} and {c.n_balls} "
            f"steel MARBLES ({2 * c.ball_r * 1000:.0f} mm) and between {c.min_cubes} and "
            f"{c.n_cubes} orange CUBES ({c.cube_size * 1000:.0f} mm) — count what you "
            f"see. The discharge end carries a full-width slot only "
            f"{c.slot_h * 1000:.0f} mm tall: marbles fit through it, cubes cannot. "
            f"Behind the slot sits a SEALED HOPPER whose only mouth faces the slot; "
            f"every other gap around it is narrower than a marble, so nothing can be "
            f"posted in by hand or dropped in from above. A red PADDLE on a raised beam "
            f"beyond the hinge is the tray's handle: pressing it down tilts the bed "
            f"toward the slot (up to {c.tilt_max_deg:.0f} deg); releasing it lets the "
            f"tray fall back onto its stop. An open wooden CRATE stands on the ground "
            f"nearby.\n"
            f"Goal: SORT the load. First move every orange cube from the bed into the "
            f"crate (over the tray's side fences — the slot will not pass them), then "
            f"press the paddle and hold the tray tilted until every marble has rolled "
            f"through the slot into the hopper, and release. The task is done when all "
            f"marbles rest inside the hopper, all cubes rest in the crate, and the tray "
            f"lies back on its rest stop. Marbles left on the bed, cubes dropped on the "
            f"ground or into the hopper, or a tray still held up do not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Sort the tray: lift each orange cube off the tray bed into the wooden "
            "crate, then press the red paddle down and hold the tray tilted until "
            "every marble rolls through the end slot into the hopper, then release "
            "the paddle so the tray settles back on its stop."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _local(self, body, pos_w: torch.Tensor) -> torch.Tensor:
        """World points (N,3) -> `body`'s frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(body.data.root_quat_w, pos_w - body.data.root_pos_w)

    def _family(self, bodies: dict[str, Any], frame) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos in `frame`'s local coords (N,P,3), |lin vel| (N,P)) for a piece family."""
        pos = torch.stack([b.data.root_pos_w for b in bodies.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in bodies.values()], dim=1)
        n, p = pos.shape[0], pos.shape[1]
        loc = self._local(frame, pos.reshape(n * p, 3)).reshape(n, p, 3)
        return loc, vel

    def tray_angle_deg(self) -> torch.Tensor:
        """(N,) signed hinge angle (deg): rest stop reads rest_deg (-2.5), full
        discharge tilt reads +tilt_max_deg. Computed from the tray's world quat
        (rotation of local +x about the world-y hinge axis)."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(n, 3)
        d = quat_apply(self.tray.data.root_quat_w, ex)
        return torch.rad2deg(torch.atan2(-d[:, 2], d[:, 0]))

    def tray_at_rest(self) -> torch.Tensor:
        """(N,) bool: tray back on (or within a degree of) its gravity rest stop and
        still — a tray held tilted, propped, or mid-swing reads False."""
        c = self.cfg
        return (self.tray_angle_deg() < c.rest_max_deg) \
            & (self.tray.data.root_ang_vel_w.norm(dim=-1) < c.rest_still)

    def balls_in_hopper(self) -> torch.Tensor:
        """(N,B) bool: marble centre inside the hopper interior, dock frame, judged
        BELOW the mouth sill (z < hop_z_max < wall_top_z): a marble in the doorway or
        perched anywhere on the fixture can never satisfy this; the z floor rejects
        one tunnelled under the hopper floor."""
        c = self.cfg
        loc, _v = self._family(self.balls, self.dock)
        return (loc[..., 0] > c.hop_x_in[0] + 0.005) & (loc[..., 0] < c.hop_x_in[1]) \
            & (loc[..., 1].abs() < c.hop_hw - 0.002) \
            & (loc[..., 2] > c.hop_floor_top - 0.006) & (loc[..., 2] < c.hop_z_max)

    def cubes_in_crate(self) -> torch.Tensor:
        """(N,C) bool: cube centre inside the crate box, crate frame (xy bound admits
        a wall-hugging cube; the z band rejects wall-top perches and under-floor)."""
        c = self.cfg
        loc, _v = self._family(self.cubes, self.crate)
        return (loc[..., 0].abs() < c.crate_xy) & (loc[..., 1].abs() < c.crate_xy) \
            & (loc[..., 2] > c.crate_z_lo) & (loc[..., 2] < c.crate_z_hi)

    def balls_on_tray(self) -> torch.Tensor:
        """(N,B) bool: marble riding the tray bed (tray frame; z ceiling excludes a
        marble perched on the canopy top)."""
        c = self.cfg
        loc, _v = self._family(self.balls, self.tray)
        return (loc[..., 0] > -(c.floor_len - c.fence_t)) & (loc[..., 0] < 0.0) \
            & (loc[..., 1].abs() < c.tray_hw - c.fence_t) \
            & (loc[..., 2] > -0.005) & (loc[..., 2] < 0.048)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in
                         list(self.balls.values()) + list(self.cubes.values()) + [self.tray]],
                        dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        self._ball_l |= self.balls_in_hopper() & self.present_ball & fin.unsqueeze(-1)
        _loc, cvel = self._family(self.cubes, self.crate)
        self._cube_l |= self.cubes_in_crate() & (cvel < c.latch_speed) \
            & self.present_cube & fin.unsqueeze(-1)
        aboard = (self.balls_on_tray() & self.present_ball).any(dim=1)
        self._tilt |= (self.tray_angle_deg() > c.tilt_credit_deg) & aboard & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: every PRESENT marble inside the hopper and settled, every
        PRESENT cube inside the crate and settled, the tray back at rest on its
        stop, everything finite. Judged live on the settled state."""
        c = self.cfg
        self._update_latches()
        _bloc, bvel = self._family(self.balls, self.dock)
        balls_ok = ((self.balls_in_hopper() & (bvel < c.ball_settle))
                    | ~self.present_ball).all(dim=1)
        _cloc, cvel = self._family(self.cubes, self.crate)
        cubes_ok = ((self.cubes_in_crate() & (cvel < c.cube_settle))
                    | ~self.present_cube).all(dim=1)
        return balls_ok & cubes_ok & self.tray_at_rest() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20*(crated fraction) + 0.15*tilted-with-load +
        0.45*(binned fraction), all latched; non-success capped at 0.80; exactly
        1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        kb = self.present_ball.sum(dim=1).clamp(min=1).float()
        kc = self.present_cube.sum(dim=1).clamp(min=1).float()
        base = c.w_crate * (self._cube_l & self.present_cube).sum(dim=1).float() / kc \
            + c.w_tilt * self._tilt.float() \
            + c.w_bin * (self._ball_l & self.present_ball).sum(dim=1).float() / kb
        base = base.clamp(max=c.cap)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="riddle_tray", robot="null"))
