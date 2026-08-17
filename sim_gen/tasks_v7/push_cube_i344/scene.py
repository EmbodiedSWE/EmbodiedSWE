"""BalanceTriageScene — WEIGH the two identical cubes against each other on a balance
scale, then sort them: the heavier cube into the red bin, the lighter into the blue.

Derived from maniskill/push_cube ("push and move a cube to a goal region": one cube,
one visible goal, one straight planar push — no hidden state anywhere). Here the
task-defining ingredient is HIDDEN STATE: two cubes that are visually IDENTICAL
(same size, same brushed-steel color) but one is ~3x heavier, and WHICH one is heavy
is randomized per episode. No amount of looking — and no straight push into a region —
can solve the task: the solver must first MEASURE, by loading both cubes onto the
opposite pans of a passive balance scale and letting gravity tip the beam, and only
then deliver each cube to the bin matching its verdict. The scale is an instrument
(a sensor), not an actuator: the beam is a free revolute body whose tilt direction
IS the answer.

Assets are fully procedural:
  - scale: a heavy DYNAMIC pedestal (wide foot + pillar, 6 kg) carrying a DYNAMIC
    beam on a spawn-authored revolute hinge (axis horizontal, limits +/-10 deg).
    The beam has a walled pan TRAY at each end; the tray floors sit AT the hinge
    plane (angle-invariant lever arms) and the beam's CoM is authored slightly BELOW
    the hinge, so the empty beam self-levels — a clean null state.
  - cubes: two 45 mm DYNAMIC cubes, identical color/size; per episode the pair
    (heavy ~0.38-0.46 kg, light ~0.11-0.17 kg) is written to the physics engine
    (masses + scaled inertias) and the two spawn spots are coin-flip swapped.
  - bins: two KINEMATIC walled trays on the floor, RED (heavy verdict) and BLUE
    (light verdict).

The comparison event ("weighed"): both cubes simultaneously resting in-footprint on
OPPOSITE pans, everything calm, and the beam deflected >= `theta_min_deg` toward the
TRUE heavy side, sustained for `weigh_streak` consecutive steps -> latched. A single
cube tipping the beam, both cubes stacked on one pan, or a tilted empty beam never
latches it (all constructed and rejected in smoke.py).

Rubric (0..1; latched partial credit anchored in the demonstrated solve):
  0.10 * heavy cube ever rested on a pan                                  (latched)
  0.10 * light cube ever rested on a pan                                  (latched)
  0.25 * weighed — the comparison event above                             (latched)
  0.15 * heavy settled in the RED bin, only latchable AFTER weighed       (latched)
  0.15 * light settled in the BLUE bin, only latchable AFTER weighed      (latched)
  capped at 0.70; exactly 1.0 iff success() = weighed AND heavy live in the red
  bin AND light live in the blue bin, cubes settled, everything finite. A blind
  guess that happens to sort correctly scores ~0 and never succeeds: the bin
  credit and success are gated on the physically verified weighing.

Honesty geometry (asserted in `__post_init__`):
  - the worst-case mass gap tips the beam decisively (>= 5x the beam's own
    self-leveling torque at the stop) — the verdict is always readable;
  - a cube parked on a full-tilt pan is retained by friction and the pan walls;
  - two cubes cannot rest side by side inside one pan footprint; a cube stacked on
    another (or resting on a bin wall) falls outside the acceptance z-bands;
  - spawn bands keep cubes clear of the pedestal, the bins and each other under
    all jitters; every interaction point is within single-Franka reach.

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


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4), pure torch."""
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


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


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, *, center, size, color, contact_offset: float):
    """One collidable box child: translate + scale, displayColor."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(box.GetPrim(), contact_offset)
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


def _rigid_armor(root, *, mass: float, com, inertia) -> None:
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    return px


def _spawn_pedestal(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC pedestal: origin at the foot's bottom center; a wide foot
    slab plus a square pillar whose top is the hinge point at z=hinge_h."""
    stage, root = _root_xform(prim_path, translation, orientation)
    px = _rigid_armor(root, mass=cfg.mass, com=(0.0, 0.0, 0.03),
                      inertia=(0.05, 0.05, 0.06))
    px.CreateLinearDampingAttr(0.2)
    px.CreateAngularDampingAttr(0.2)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    c = cfg
    kids = [
        _box(stage, f"{prim_path}/foot",
             center=(0.0, 0.0, c.foot_t / 2), size=(2 * c.foot_hx, 2 * c.foot_hy, c.foot_t),
             color=c.color, contact_offset=c.contact_offset),
        _box(stage, f"{prim_path}/pillar",
             center=(0.0, 0.0, (c.foot_t + c.hinge_h) / 2),
             size=(c.pillar_w, c.pillar_w, c.hinge_h - c.foot_t),
             color=c.color, contact_offset=c.contact_offset),
    ]
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC beam: origin ON the hinge point; bar along local x with a
    walled pan tray at each end, tray FLOOR TOPS at local z=0 (the hinge plane).
    A revolute joint (axis Y) to the sibling Pedestal is authored in-spawn."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    px = _rigid_armor(root, mass=cfg.mass, com=(0.0, 0.0, cfg.com_z),
                      inertia=(8e-4, 5e-3, 5.5e-3))
    px.CreateLinearDampingAttr(0.02)
    px.CreateAngularDampingAttr(cfg.ang_damping)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    c = cfg
    co = c.contact_offset
    kids = [
        _box(stage, f"{prim_path}/bar",
             center=(0.0, 0.0, -0.008 - 0.006), size=(2 * c.arm_l + 0.04, 0.024, 0.012),
             color=c.color, contact_offset=co),
    ]
    half_f = c.pan_w / 2
    for s, side in ((1.0, "a"), (-1.0, "b")):
        kids.append(_box(stage, f"{prim_path}/floor_{side}",
                         center=(s * c.arm_l, 0.0, -c.pan_floor_t / 2),
                         size=(c.pan_w, c.pan_w, c.pan_floor_t),
                         color=c.pan_color, contact_offset=co))
        for w, axis in ((1.0, "x"), (-1.0, "x"), (1.0, "y"), (-1.0, "y")):
            if axis == "x":
                cen = (s * c.arm_l + w * (half_f - c.pan_wall_t / 2), 0.0,
                       c.pan_wall_h / 2)
                sz = (c.pan_wall_t, c.pan_w, c.pan_wall_h)
            else:
                cen = (s * c.arm_l, w * (half_f - c.pan_wall_t / 2), c.pan_wall_h / 2)
                sz = (c.pan_w, c.pan_wall_t, c.pan_wall_h)
            kids.append(_box(stage, f"{prim_path}/wall_{side}_{axis}{'p' if w > 0 else 'n'}",
                             center=cen, size=sz, color=c.pan_color, contact_offset=co))
    for k in kids:
        _bind_material(k, mat)
    # revolute hinge to the sibling Pedestal (pair collision FILTERED by the joint;
    # the tilt stops are the joint limits themselves)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Pedestal"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(False)
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(cfg.hinge_h)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(cfg.limit_deg))
    j.CreateUpperLimitAttr(float(cfg.limit_deg))
    return root


def _spawn_bin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a KINEMATIC walled floor bin: origin at the footprint's bottom center."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    mat = _phys_material(stage, f"{prim_path}/physmat", 0.6, 0.5)
    c = cfg
    co = c.contact_offset
    half = c.inner / 2 + c.wall_t
    kids = [
        _box(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.floor_t / 2),
             size=(2 * half, 2 * half, c.floor_t), color=c.color, contact_offset=co),
        _box(stage, f"{prim_path}/wall_xp", center=(half - c.wall_t / 2, 0.0, c.wall_h / 2),
             size=(c.wall_t, 2 * half, c.wall_h), color=c.color, contact_offset=co),
        _box(stage, f"{prim_path}/wall_xn", center=(-(half - c.wall_t / 2), 0.0, c.wall_h / 2),
             size=(c.wall_t, 2 * half, c.wall_h), color=c.color, contact_offset=co),
        _box(stage, f"{prim_path}/wall_yp", center=(0.0, half - c.wall_t / 2, c.wall_h / 2),
             size=(2 * half - 2 * c.wall_t, c.wall_t, c.wall_h),
             color=c.color, contact_offset=co),
        _box(stage, f"{prim_path}/wall_yn", center=(0.0, -(half - c.wall_t / 2), c.wall_h / 2),
             size=(2 * half - 2 * c.wall_t, c.wall_t, c.wall_h),
             color=c.color, contact_offset=co),
    ]
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pedestal" not in _SPAWNER_CACHE:

        @configclass
        class PedestalSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pedestal)
            foot_hx: float = 0.13
            foot_hy: float = 0.10
            foot_t: float = 0.024
            pillar_w: float = 0.04
            hinge_h: float = 0.14
            mass: float = 6.0
            mu_static: float = 0.8
            mu_dynamic: float = 0.7
            color: tuple = (0.30, 0.32, 0.36)
            contact_offset: float = 0.002

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            arm_l: float = 0.16
            pan_w: float = 0.100
            pan_floor_t: float = 0.008
            pan_wall_t: float = 0.006
            pan_wall_h: float = 0.026
            hinge_h: float = 0.14
            limit_deg: float = 10.0
            mass: float = 0.25
            com_z: float = -0.020
            ang_damping: float = 0.30
            mu_static: float = 0.6
            mu_dynamic: float = 0.5
            color: tuple = (0.85, 0.60, 0.15)
            pan_color: tuple = (0.75, 0.52, 0.12)
            contact_offset: float = 0.002

        @configclass
        class BinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bin)
            inner: float = 0.130
            wall_t: float = 0.008
            wall_h: float = 0.035
            floor_t: float = 0.008
            color: tuple = (0.7, 0.2, 0.2)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["pedestal"] = PedestalSpawnerCfg
        _SPAWNER_CACHE["beam"] = BeamSpawnerCfg
        _SPAWNER_CACHE["bin"] = BinSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BalanceTriageSceneCfg(BaseCfg):
    """Config for `BalanceTriageScene`. The mass gap always tips the beam decisively,
    pans retain a parked cube at full tilt, and the acceptance bands reject stacked /
    wall-perched rests. All asserted in `__post_init__`."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    theta_min_deg: float = tunable(6.0)     # beam deflection that counts as a verdict
    weigh_streak: int = tunable(25)         # consecutive calm steps to latch `weighed`
    pan_xy_tol: float = tunable(0.035)      # cube center within this of a pan center (beam frame)
    pan_z_lo: float = tunable(0.010)        # cube center z band over the pan floor (beam frame)
    pan_z_hi: float = tunable(0.042)
    calm_beam_w: float = tunable(0.60)      # max |beam ang vel| during the comparison (rad/s)
    calm_cube_v: float = tunable(0.20)      # max cube |lin vel| during the comparison (m/s)
    bin_xy_tol: float = tunable(0.045)      # cube center within this of the bin center (bin frame)
    bin_z_lo: float = tunable(0.018)        # cube center z band over the bin floor (bin frame)
    bin_z_hi: float = tunable(0.045)
    settle_speed: float = tunable(0.05)     # max cube |lin vel| when judging success

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    heavy_mass_lo: float = tunable(0.38)    # heavy cube mass band (kg)
    heavy_mass_hi: float = tunable(0.46)
    light_mass_lo: float = tunable(0.11)    # light cube mass band (kg)
    light_mass_hi: float = tunable(0.17)
    scale_jitter: float = tunable(0.020)    # pedestal xy jitter (+/- m)
    scale_yaw_deg: float = tunable(12.0)    # pedestal yaw jitter (+/- deg, about beam-across-y)
    bin_jitter: float = tunable(0.030)      # each bin xy jitter (+/- m)
    bin_yaw_deg: float = tunable(25.0)      # each bin yaw jitter (+/- deg)
    spot_jitter: float = tunable(0.025)     # cube spawn-spot xy jitter (+/- m)

    # --- info: layout (world nominal, ground z = 0; Franka base pose ~ (0,0)) -------------------
    scale_pos: tuple = info((0.46, 0.0))    # pedestal center
    scale_yaw_nom_deg: float = info(90.0)   # beam runs across (+/- world y)
    bin_x: float = info(0.22)               # both bins at this x, +/- bin_y
    bin_y: float = info(0.30)               # red at +y, blue at -y
    spot_x: float = info(0.20)              # cube spawn spots at this x, +/- spot_y
    spot_y: float = info(0.09)
    # --- info: scale structure (must match the spawner cfgs) -------------------------------------
    foot_hx: float = info(0.13)
    foot_hy: float = info(0.10)
    foot_t: float = info(0.024)
    hinge_h: float = info(0.14)
    arm_l: float = info(0.16)
    pan_w: float = info(0.100)
    pan_wall_t: float = info(0.006)
    pan_wall_h: float = info(0.026)
    limit_deg: float = info(10.0)
    beam_mass: float = info(0.25)
    beam_com_z: float = info(-0.020)
    pedestal_mass: float = info(6.0)
    beam_mu: float = info(0.6)
    # --- info: cubes + bins ----------------------------------------------------------------------
    cube: float = info(0.045)               # cube edge (both cubes — visually identical)
    cube_spawn_mass: float = info(0.28)     # authored spawn mass (overwritten per reset)
    cube_color: tuple = info((0.58, 0.60, 0.63))
    bin_inner: float = info(0.130)
    bin_wall_t: float = info(0.008)
    bin_wall_h: float = info(0.035)
    bin_floor_t: float = info(0.008)
    red_color: tuple = info((0.72, 0.15, 0.12))
    blue_color: tuple = info((0.12, 0.25, 0.72))
    contact_offset: float = info(0.002)
    ground_mu: float = info(0.6)
    # --- info: rubric weights (0.10+0.10+0.25+0.15+0.15 = 0.75; non-success cap 0.70) -----------
    w_pan: float = info(0.10)
    w_weigh: float = info(0.25)
    w_bin: float = info(0.15)

    def __post_init__(self) -> None:
        g = 9.81
        # the verdict is always decisive: worst-case mass gap vs beam self-leveling
        dm_min = self.heavy_mass_lo - self.light_mass_hi
        tip = dm_min * g * self.arm_l * math.cos(math.radians(self.limit_deg))
        restore = self.beam_mass * g * abs(self.beam_com_z) * math.sin(
            math.radians(self.limit_deg))
        assert dm_min > 0.15, "mass bands overlap or gap too small to be robust"
        assert tip > 5.0 * restore, "beam self-leveling could mask the verdict"
        assert self.heavy_mass_lo / self.light_mass_hi > 2.0, "mass ratio not decisive"
        # verdict threshold is inside the physical stop
        assert self.theta_min_deg + 2.0 <= self.limit_deg < 170.0, "theta_min vs stop"
        # a parked cube is retained on a full-tilt pan (friction + walls)
        assert math.tan(math.radians(self.limit_deg)) < 0.8 * self.beam_mu, \
            "cube would slide on a tilted pan"
        # pan floor tops at the hinge plane (authored so in the spawner: z=0 local)
        # one pan cannot seat two cubes side by side; a stacked cube leaves the z band
        pan_inner = self.pan_w - 2 * self.pan_wall_t
        assert pan_inner < 2 * self.cube, "two cubes could sit side by side in one pan"
        assert pan_inner > self.cube + 0.030, "cube would not drop into the pan freely"
        assert self.cube / 2 + self.cube > self.pan_z_hi + 0.015, \
            "a stacked cube would still count as on-pan"
        assert self.pan_z_lo < self.cube / 2 < self.pan_z_hi, "resting cube outside pan band"
        # bins: cube fits, wall-top and stacked rests rejected by the z band
        assert self.bin_inner / 2 - self.cube / 2 > self.bin_xy_tol - 0.010, \
            "bin xy tol vs geometry"
        assert self.bin_wall_h + self.cube / 2 > self.bin_z_hi + 0.010, \
            "a wall-perched cube would still count as binned"
        assert self.bin_z_lo < self.bin_floor_t + self.cube / 2 < self.bin_z_hi, \
            "resting cube outside bin band"
        assert self.bin_floor_t + self.cube / 2 + self.cube > self.bin_z_hi + 0.015, \
            "a cube stacked on another in the bin would still count"
        # spawn clearances under all jitters
        cube_r = self.cube * math.sqrt(2) / 2
        spot_gap = 2 * self.spot_y - 2 * self.spot_jitter - 2 * cube_r
        assert spot_gap > 0.02, "cube spawn spots could collide"
        foot_near_x = self.scale_pos[0] - self.scale_jitter - self.foot_hy - 0.005
        # (pedestal yaw ~90 deg: world-x half extent is foot_hy)
        assert self.spot_x + self.spot_jitter + cube_r < foot_near_x, \
            "cube could spawn against the pedestal foot"
        bin_half = self.bin_inner / 2 + self.bin_wall_t
        bin_reach = math.sqrt(2) * bin_half + self.bin_jitter
        d_spot_bin = math.hypot(self.bin_x - self.spot_x, self.bin_y - self.spot_y)
        assert d_spot_bin > bin_reach + self.spot_jitter + cube_r + 0.02, \
            "cube could spawn inside a bin"
        d_bins = 2 * self.bin_y - 2 * self.bin_jitter
        assert d_bins > 2 * bin_reach + 0.02, "bins could overlap"
        # single-arm reach from a base at the origin
        far = max(
            math.hypot(self.scale_pos[0] + self.scale_jitter,
                       self.arm_l + self.pan_w / 2),
            math.hypot(self.bin_x + self.bin_jitter, self.bin_y + self.bin_jitter
                       + bin_half),
        )
        assert far < 0.68, f"layout out of comfortable Franka reach ({far:.3f} m)"
        # score arithmetic
        assert abs(2 * self.w_pan + self.w_weigh + 2 * self.w_bin - 0.75) < 1e-9


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("balance_triage")
class BalanceTriageScene(BaseScene):
    cfg: BalanceTriageSceneCfg

    def __init__(self, cfg: BalanceTriageSceneCfg | None = None) -> None:
        super().__init__(cfg or BalanceTriageSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        yaw0 = math.radians(c.scale_yaw_nom_deg)
        q0 = (math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2))

        def cube_cfg(path: str, pos) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + path,
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube, c.cube, c.cube),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.cube_color, roughness=0.35),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.05,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_spawn_mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            )

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.ground_mu, dynamic_friction=c.ground_mu - 0.1,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light_rig": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "pedestal": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pedestal",
                spawn=cls["pedestal"](),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.scale_pos[0], c.scale_pos[1], 0.002), rot=q0),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=cls["beam"](),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.scale_pos[0], c.scale_pos[1], 0.002 + c.hinge_h), rot=q0),
            ),
            "bin_red": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BinRed",
                spawn=cls["bin"](color=c.red_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.bin_x, c.bin_y, 0.0)),
            ),
            "bin_blue": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BinBlue",
                spawn=cls["bin"](color=c.blue_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.bin_x, -c.bin_y, 0.0)),
            ),
            # the two suspects — identical spawns; masses + spots randomized at reset
            "cube_h": cube_cfg("CubeA", (c.spot_x, c.spot_y, c.cube / 2 + 0.004)),
            "cube_l": cube_cfg("CubeB", (c.spot_x, -c.spot_y, c.cube / 2 + 0.004)),
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
        self.pedestal: RigidObject = env.iscene["pedestal"]
        self.beam: RigidObject = env.iscene["beam"]
        self.bin_red: RigidObject = env.iscene["bin_red"]
        self.bin_blue: RigidObject = env.iscene["bin_blue"]
        self.cube_h: RigidObject = env.iscene["cube_h"]
        self.cube_l: RigidObject = env.iscene["cube_l"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches (partial credit survives transients; success re-judges live state)
        self._pan_h = torch.zeros(n, dtype=torch.bool, device=dev)
        self._pan_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._weighed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._hbin = torch.zeros(n, dtype=torch.bool, device=dev)
        self._lbin = torch.zeros(n, dtype=torch.bool, device=dev)
        self._streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._inertia0 = None  # captured lazily at first reset (spawn values)
        self._mass0 = None

    # ----- mass plumbing (per-episode hidden state) ----------------------------------------------
    def _write_masses(self, env_ids: torch.Tensor, mh: torch.Tensor,
                      ml: torch.Tensor) -> None:
        """Write per-episode masses (and proportionally scaled inertias) to physx."""
        if self._inertia0 is None:
            self._mass0 = {b: b.root_physx_view.get_masses().clone()
                           for b in (self.cube_h, self.cube_l)}
            self._inertia0 = {b: b.root_physx_view.get_inertias().clone()
                              for b in (self.cube_h, self.cube_l)}
        ids_cpu = env_ids.detach().cpu()
        for body, new in ((self.cube_h, mh), (self.cube_l, ml)):
            view = body.root_physx_view
            m = self._mass0[body].clone()
            flat = m.view(m.shape[0], -1)
            flat[ids_cpu] = new.detach().cpu().view(-1, 1)
            view.set_masses(m, ids_cpu)
            inr = self._inertia0[body].clone()
            scale = (flat / self._mass0[body].view(m.shape[0], -1))
            inr_flat = inr.view(inr.shape[0], -1)
            inr_flat *= scale[:, :1]
            view.set_inertias(inr, ids_cpu)

    def masses(self) -> tuple[float, float]:
        """(heavy, light) mass readback for env 0 — ground truth, oracle-only."""
        return (float(self.cube_h.root_physx_view.get_masses().view(-1)[0]),
                float(self.cube_l.root_physx_view.get_masses().view(-1)[0]))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pedestal+beam written consistently (one linkage, level),
        bins jittered, per-episode masses written to physx, and the two cubes'
        spawn spots COIN-FLIP SWAPPED (the hidden assignment). Latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        for _ in range(6):  # burn post-seed draws (early Philox draws are seed-correlated)
            torch.rand(2 * m, device=dev)

        def rnd(k: float) -> torch.Tensor:
            return (torch.rand(m, device=dev) * 2 - 1) * k

        def write(body, pos, q, ) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = q
            body.write_root_state_to_sim(st, env_ids)

        # --- scale: pedestal + level beam, one linkage, written back to back ---
        yaw = math.radians(c.scale_yaw_nom_deg) + rnd(math.radians(c.scale_yaw_deg))
        q_ped = _qz(yaw)
        ppos = torch.zeros(m, 3, device=dev)
        ppos[:, 0] = c.scale_pos[0] + rnd(c.scale_jitter)
        ppos[:, 1] = c.scale_pos[1] + rnd(c.scale_jitter)
        ppos[:, 2] = 0.002
        write(self.pedestal, ppos, q_ped)
        hinge = torch.tensor([0.0, 0.0, c.hinge_h], device=dev).expand(m, 3)
        write(self.beam, ppos + _qapply(q_ped, hinge), q_ped.clone())

        # --- bins: xy jitter + free-ish yaw ---
        for body, sy in ((self.bin_red, 1.0), (self.bin_blue, -1.0)):
            bpos = torch.zeros(m, 3, device=dev)
            bpos[:, 0] = c.bin_x + rnd(c.bin_jitter)
            bpos[:, 1] = sy * c.bin_y + rnd(c.bin_jitter)
            write(body, bpos, _qz(rnd(math.radians(c.bin_yaw_deg))))

        # --- hidden state: per-episode masses + coin-flip spot assignment ---
        mh = c.heavy_mass_lo + torch.rand(m, device=dev) * (c.heavy_mass_hi - c.heavy_mass_lo)
        ml = c.light_mass_lo + torch.rand(m, device=dev) * (c.light_mass_hi - c.light_mass_lo)
        self._write_masses(env_ids, mh, ml)
        flip = torch.rand(m, device=dev) > 0.5
        sy_h = torch.where(flip, torch.full((m,), 1.0, device=dev),
                           torch.full((m,), -1.0, device=dev))
        for body, sy in ((self.cube_h, sy_h), (self.cube_l, -sy_h)):
            cpos = torch.zeros(m, 3, device=dev)
            cpos[:, 0] = c.spot_x + rnd(c.spot_jitter)
            cpos[:, 1] = sy * c.spot_y + rnd(c.spot_jitter)
            cpos[:, 2] = c.cube / 2 + 0.004
            write(body, cpos, _qz(rnd(math.pi)))

        self._pan_h[env_ids] = False
        self._pan_l[env_ids] = False
        self._weighed[env_ids] = False
        self._hbin[env_ids] = False
        self._lbin[env_ids] = False
        self._streak[env_ids] = 0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"pedestal": self.pedestal, "beam": self.beam, "bin_red": self.bin_red,
                  "bin_blue": self.bin_blue, "cube_h": self.cube_h, "cube_l": self.cube_l}
        out = {k: b.data.root_state_w[env_ids].clone() for k, b in bodies.items()}
        out["latches"] = torch.stack([self._pan_h[env_ids], self._pan_l[env_ids],
                                      self._weighed[env_ids], self._hbin[env_ids],
                                      self._lbin[env_ids]], dim=1).clone()
        out["streak"] = self._streak[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"pedestal": self.pedestal, "beam": self.beam, "bin_red": self.bin_red,
                  "bin_blue": self.bin_blue, "cube_h": self.cube_h, "cube_l": self.cube_l}
        for k, b in bodies.items():
            b.write_root_state_to_sim(state[k], env_ids)
        lat = state["latches"]
        self._pan_h[env_ids], self._pan_l[env_ids] = lat[:, 0], lat[:, 1]
        self._weighed[env_ids], self._hbin[env_ids] = lat[:, 2], lat[:, 3]
        self._lbin[env_ids] = lat[:, 4]
        self._streak[env_ids] = state["streak"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A BALANCE SCALE stands on the floor ahead: a dark pedestal carrying a "
            f"free-tilting amber beam with a small walled PAN at each end (pans about "
            f"{c.pan_w * 1000:.0f} mm square, roughly {c.hinge_h * 1000:.0f} mm up, "
            f"one to your left, one to your right; the beam tilts up to "
            f"{c.limit_deg:.0f} deg). In front of the scale sit TWO IDENTICAL brushed-"
            f"steel cubes ({c.cube * 1000:.0f} mm) side by side on the floor. They look "
            f"exactly the same, but ONE is about three times heavier than the other, "
            f"and which one it is changes every episode — you cannot tell by looking. "
            f"To the sides stand two open floor bins: a RED bin (for the HEAVY cube) "
            f"and a BLUE bin (for the LIGHT cube); the bins' exact spots and headings "
            f"vary.\n"
            f"Goal: find out which cube is heavier by WEIGHING them against each "
            f"other — place one cube in each pan, both on the scale at the same time, "
            f"let go, and let the beam settle tipped toward the heavier side (it must "
            f"hold its verdict for a moment) — then deliver the cubes: the HEAVIER "
            f"cube into the RED bin, the LIGHTER cube into the BLUE bin, each resting "
            f"on the bin floor, and leave them settled. Sorting the cubes into bins "
            f"WITHOUT a completed both-pans weighing does not count, even if the "
            f"guess happens to be right; tipping the beam with a single cube, or "
            f"with both cubes piled on one pan, is not a weighing. You may re-weigh "
            f"or move cubes between pans and bins at any time; only the weighing "
            f"event and the final resting bins matter."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Weigh the two identical steel cubes against each other on the balance "
            "scale — one cube on each pan at the same time, beam settled — then put "
            "the heavier cube in the red bin and the lighter cube in the blue bin. "
            "Sorting without the two-pan weighing fails."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def beam_angle(self) -> torch.Tensor:
        """(N,) beam hinge angle in RADIANS: 0 level, >0 = pan A (+x local) side DOWN."""
        q_rel = _qmul(_qinv(self.pedestal.data.root_quat_w), self.beam.data.root_quat_w)
        n = self.env.num_envs
        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(n, 3)
        d = _qapply(q_rel, ex)
        return torch.atan2(-d[:, 2], d[:, 0])

    def _beam_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        return _qapply(_qinv(self.beam.data.root_quat_w),
                       pos_w - self.beam.data.root_pos_w)

    def on_pan(self, body) -> torch.Tensor:
        """(N,2) bool: cube resting in-footprint on pan A (+x) / pan B (-x)."""
        c = self.cfg
        loc = self._beam_local(body.data.root_pos_w)
        z_ok = (loc[:, 2] > c.pan_z_lo) & (loc[:, 2] < c.pan_z_hi)
        y_ok = loc[:, 1].abs() < c.pan_xy_tol
        a = ((loc[:, 0] - c.arm_l).abs() < c.pan_xy_tol) & y_ok & z_ok
        b = ((loc[:, 0] + c.arm_l).abs() < c.pan_xy_tol) & y_ok & z_ok
        return torch.stack([a, b], dim=1)

    def in_bin(self, body, bin_body) -> torch.Tensor:
        """(N,) bool: cube resting on the bin floor, inside the walls."""
        c = self.cfg
        loc = _qapply(_qinv(bin_body.data.root_quat_w),
                      body.data.root_pos_w - bin_body.data.root_pos_w)
        return (loc[:, 0].abs() < c.bin_xy_tol) & (loc[:, 1].abs() < c.bin_xy_tol) \
            & (loc[:, 2] > c.bin_z_lo) & (loc[:, 2] < c.bin_z_hi)

    def cubes_settled(self) -> torch.Tensor:
        """(N,) bool: both cubes |lin vel| below `settle_speed`."""
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in (self.cube_h, self.cube_l)], dim=1)
        return (v < self.cfg.settle_speed).all(dim=1)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in
                         (self.pedestal, self.beam, self.cube_h, self.cube_l)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        oph = self.on_pan(self.cube_h)
        opl = self.on_pan(self.cube_l)
        self._pan_h |= fin & oph.any(dim=1)
        self._pan_l |= fin & opl.any(dim=1)
        # the comparison event: opposite pans, calm, beam tipped toward the TRUE
        # heavy side by at least theta_min — sustained weigh_streak steps
        opposite = (oph[:, 0] & opl[:, 1]) | (oph[:, 1] & opl[:, 0])
        ang = self.beam_angle()
        th = math.radians(c.theta_min_deg)
        sign_ok = torch.where(oph[:, 0], ang >= th, ang <= -th)
        calm = (self.beam.data.root_ang_vel_w.norm(dim=-1) < c.calm_beam_w) \
            & (self.cube_h.data.root_lin_vel_w.norm(dim=-1) < c.calm_cube_v) \
            & (self.cube_l.data.root_lin_vel_w.norm(dim=-1) < c.calm_cube_v)
        cond = fin & opposite & sign_ok & calm
        self._streak = torch.where(cond, self._streak + 1,
                                   torch.zeros_like(self._streak))
        self._weighed |= self._streak >= c.weigh_streak
        # verdict-informed delivery: bin credit only latches AFTER a weighing
        still_h = self.cube_h.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        still_l = self.cube_l.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        self._hbin |= self._weighed & fin & still_h & self.in_bin(self.cube_h, self.bin_red)
        self._lbin |= self._weighed & fin & still_l & self.in_bin(self.cube_l, self.bin_blue)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: a completed two-pan weighing has occurred (latched physical
        event) AND the heavy cube rests in the RED bin AND the light cube rests in
        the BLUE bin, both settled, everything finite."""
        return self._weighed & self.in_bin(self.cube_h, self.bin_red) \
            & self.in_bin(self.cube_l, self.bin_blue) & self.cubes_settled() \
            & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: latched 0.10+0.10 pan visits, 0.25 weighed, 0.15+0.15
        verdict-gated bin deliveries (capped 0.70); exactly 1.0 iff success(). The
        null policy scores ~0 (cubes start on the floor, beam level); a blind sort
        without weighing scores ~0 (bin credit is gated on the weighing)."""
        c = self.cfg
        base = (c.w_pan * self._pan_h.float() + c.w_pan * self._pan_l.float()
                + c.w_weigh * self._weighed.float()
                + c.w_bin * self._hbin.float() + c.w_bin * self._lbin.float()
                ).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="balance_triage", robot="null"))
