"""TiltDispenserScene — read which port of a sealed pivoting ball cage is open,
press the yellow paddle on THAT side, and hold the cage at its tilt stop until the
caged ball rolls out of the open port and drops into the catch bin below it
(sim_gen task `basketball_in_hoop_i225`).

Derived from rlbench/basketball_in_hoop, but STRATEGICALLY different: the seed is a
grasp-and-drop — pick the ball off its stand, carry it above the hoop, release, and
gravity delivers it into the judged region. Here the robot NEVER TOUCHES THE BALL:

- the ball is 88 mm across — wider than the Franka parallel jaw span (80 mm,
  asserted in `__post_init__`) — AND it is sealed inside a slotted, roofed CAGE
  whose viewing slots are far narrower than the ball, so grasping, carrying, or
  even direct pushing of the ball is impossible;
- the cage pivots +-15 deg on a central hinge atop a stand; its floor is a shallow
  8-deg V-dish that self-centres the ball at the valley whenever the cage is level
  (the null policy scores ~0 and a ball-induced tilt cannot dispense — both
  asserted from the authored masses);
- both cage ends are open PORTS, but one port is covered per episode by a RED
  SHUTTER (which side is random): the only way to deliver the ball is to READ the
  scene, press down the yellow PADDLE on the open side, and HOLD the cage at its
  tilt stop while the ball rolls out the open port and falls into that side's
  catch bin — then release and let it settle;
- pressing the WRONG paddle tilts the ball into the shutter, which retains it over
  the dish; on release the cage swings back and the ball self-centres — a
  recoverable no-score outcome (proved in smoke).

A solver needs a different PLAN from the seed (a binary scene-reading decision plus
one sustained mechanism press-and-hold on an intermediary — no object transport at
all) and a different CODE STRUCTURE (hold-a-hinge-at-its-stop control instead of a
grasp pose and a release point).

success(): the ball rests inside the OPEN side's catch bin — a canonical-frame
containment window that accepts every physically-possible resting pose inside that
bin and rejects ground, wrong-bin, wall-perch and in-cage poses (asserted in
`__post_init__`) — with the ball settled.

score() is graded and latched (credit never evaporates): 0.2 for each demonstrated
stage — cage held past the dispense angle toward the open side, ball out through
the open port, ball inside the bin window (cap 0.60) — and 1.0 iff success().

Per-episode randomization (readback-verified in smoke): open side (the shutter
teleports to the blocked port), stand yaw + xy jitter, ball start jitter in the
dish. Assets are fully procedural compound spawners (boxes + one sphere); the cage
hangs on a spawn-authored USD revolute joint whose anchor body (the stand) is
DYNAMIC. Heavy imports (isaaclab, pxr) are deferred so importing this module — and
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

# colors
_GREEN = (0.13, 0.45, 0.20)
_STEEL = (0.35, 0.45, 0.60)
_DARK = (0.30, 0.33, 0.40)
_YELLOW = (0.95, 0.85, 0.10)
_KEEL = (0.25, 0.25, 0.28)
_GREY = (0.45, 0.47, 0.50)
_WHITE = (0.85, 0.85, 0.88)
_RED = (0.80, 0.12, 0.10)
_ORANGE = (0.95, 0.45, 0.10)


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


def _material(stage, path: str, static: float = 0.6, dynamic: float = 0.5):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, contact_offset: float, material) -> None:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float, material=None,
         orient=None, density: float | None = None) -> None:
    """Author one box child prim (translate -> orient -> scale, authored once)."""
    from pxr import Gf, UsdGeom, UsdPhysics

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)
    if density is not None:
        UsdPhysics.MassAPI.Apply(seg.GetPrim()).CreateDensityAttr(float(density))


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float | None,
                kinematic: bool = False):
    """Author one rigid-body root Xform with the standard physics armor (zero
    sleep/stabilization thresholds — a sleeping body silently ignores applied
    wrenches; velocity iterations 4 — the sphere-on-face phantom-creep fix).
    `mass=None` leaves mass to per-child densities (true CoM + inertia, which the
    cage's keel-below-hinge stability depends on)."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    if mass is not None:
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return root, pxrb


def _cage_parts(c: Any) -> list[tuple]:
    """The cage's box list, local frame with the ORIGIN ON THE HINGE AXIS.
    Each entry: (name, size, center, color, density, orient-or-None). Shared by the
    spawner and the cfg's mass/stability asserts so geometry and rubric cannot
    drift apart.

    Layout: 8-deg V-dish floor (valley at x=0), slotted side walls, recessed roof,
    corner posts, a yellow press paddle beyond each port, and a dense keel far
    below the hinge (the restoring pendulum mass)."""
    a = math.radians(c.dish_deg)
    t = c.plate_t
    hx = c.cage_half_x
    sl = hx / math.cos(a) + 0.008  # dish plate slope length (+valley lap)
    ca2, sa2 = math.cos(a / 2), math.sin(a / 2)
    sa, ca = math.sin(a), math.cos(a)
    wy2 = 2 * (c.wall_y + t / 2)  # dish/roof width (spans under the walls)
    mzx = hx / 2  # surface midpoint of the +x plate
    mzz = c.dish_valley_z + mzx * math.tan(a)
    wall_cz = c.wall_top - c.wall_h / 2
    roof_cz = c.roof_zbot + t / 2
    post_h = c.roof_zbot - c.wall_top
    d = c.struct_density
    parts: list[tuple] = [
        ("dish_p", (sl, wy2, t), (mzx + sa * t / 2, 0.0, mzz - ca * t / 2),
         _GREEN, d, (ca2, 0.0, -sa2, 0.0)),
        ("dish_n", (sl, wy2, t), (-(mzx + sa * t / 2), 0.0, mzz - ca * t / 2),
         _GREEN, d, (ca2, 0.0, +sa2, 0.0)),
        ("wall_p", (2 * hx, t, c.wall_h), (0.0, +c.wall_y, wall_cz), _STEEL, d, None),
        ("wall_n", (2 * hx, t, c.wall_h), (0.0, -c.wall_y, wall_cz), _STEEL, d, None),
        ("roof", (2 * c.roof_half_x, wy2, t), (0.0, 0.0, roof_cz), _DARK, d, None),
        ("keel", (c.keel_sx, c.keel_sy, c.keel_sz), (0.0, 0.0, c.keel_z),
         _KEEL, c.keel_density, None),
    ]
    for sx in (1.0, -1.0):
        parts.append((f"paddle_{'p' if sx > 0 else 'n'}",
                      (c.paddle_s, c.paddle_s, t), (sx * c.paddle_cx, 0.0, roof_cz),
                      _YELLOW, d, None))
        for sy in (1.0, -1.0):
            parts.append((f"post_{'p' if sx > 0 else 'n'}{'p' if sy > 0 else 'n'}",
                          (t, t, post_h),
                          (sx * c.post_x, sy * c.wall_y, c.wall_top + post_h / 2),
                          _STEEL, d, None))
    return parts


def _cage_dyn(c: Any) -> tuple[float, float, float]:
    """(mass, CoM z, I about the hinge/y axis) of the cage from its authored part
    densities (box inertia, part orientation neglected — the dish tilt is 8 deg)."""
    m_tot, mz, iy = 0.0, 0.0, 0.0
    for _nm, (sx, sy, sz), (cx, _cy, cz), _col, rho, _q in _cage_parts(c):
        m = sx * sy * sz * rho
        m_tot += m
        mz += m * cz
        iy += m * ((sx * sx + sz * sz) / 12.0 + cx * cx + cz * cz)
    return m_tot, mz / m_tot, iy


def _stand_parts(c: Any) -> list[tuple]:
    """The stand's box list (local frame: origin at the footprint centre on the
    ground): base plate, two hinge pillars with axle stubs, and a mirrored catch
    bin under each port (floor, low near wall, tall backboard, side walls)."""
    wt = c.bin_wall_t
    bx0, bx1 = c.bin_x0, c.bin_x1
    bfx = (bx0 + bx1) / 2
    parts: list[tuple] = [
        ("base", (0.14, 0.34, 0.03), (0.0, 0.0, 0.015), _GREY, None),
    ]
    for sy in (1.0, -1.0):
        tag = "p" if sy > 0 else "n"
        parts.append((f"pillar_{tag}", (0.04, 0.04, 0.36),
                      (0.0, sy * 0.10, 0.21), _GREY, None))
        parts.append((f"stub_{tag}", (0.04, 0.05, 0.04),
                      (0.0, sy * 0.075, c.hinge_h), _GREY, None))
    for sx in (1.0, -1.0):
        tag = "p" if sx > 0 else "n"
        parts += [
            (f"bin_floor_{tag}", (bx1 - bx0, 2 * c.bin_half_y + 2 * wt, c.bin_floor_top),
             (sx * bfx, 0.0, c.bin_floor_top / 2), _GREY, None),
            (f"bin_near_{tag}", (wt, 2 * c.bin_half_y + 2 * wt, c.bin_near_top),
             (sx * (bx0 + wt / 2), 0.0, c.bin_near_top / 2), _WHITE, None),
            (f"bin_back_{tag}", (wt, 2 * c.bin_half_y + 2 * wt, c.bin_back_top),
             (sx * (bx1 - wt / 2), 0.0, c.bin_back_top / 2), _WHITE, None),
        ]
        for sy in (1.0, -1.0):
            parts.append((f"bin_side_{tag}{'p' if sy > 0 else 'n'}",
                          (bx1 - bx0, wt, c.bin_side_top),
                          (sx * bfx, sy * (c.bin_half_y + wt / 2), c.bin_side_top / 2),
                          _WHITE, None))
    return parts


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The heavy DYNAMIC stand (root mass — its CoM at the ground-level body origin
    only makes it more stable). It is the hinge joint's anchor body: a DYNAMIC
    anchor keeps the joint frame attached through reset teleports (a kinematic
    body0's anchor would stay world-fixed)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, 28.0)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    mat = _material(stage, f"{prim_path}/phys_mat")
    for name, size, center, color, orient in _stand_parts(cfg):
        _box(stage, f"{prim_path}/{name}", size, center, color, cfg.contact_offset,
             material=mat, orient=orient)
    return root


def _spawn_cage(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The DYNAMIC cage: origin ON the hinge axis, per-child DENSITY masses (root
    mass_props on a custom spawner is silently ignored on this stack; densities
    yield the true CoM + inertia that the keel-pendulum stability needs), plus the
    spawn-authored REVOLUTE joint to the sibling stand (post-play joints are dead)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, None)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(float(cfg.cage_ang_damping))
    mat = _material(stage, f"{prim_path}/phys_mat")
    for name, size, center, color, density, orient in _cage_parts(cfg):
        _box(stage, f"{prim_path}/{name}", size, center, color, cfg.contact_offset,
             material=mat, orient=orient, density=density)

    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Stand"])
    j.CreateBody1Rel().SetTargets([prim_path])
    # cage never needs to touch the stand: leave the joint pair's default
    # collision filtering ON (the tilt stops are the joint limits)
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(cfg.hinge_h)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(cfg.tilt_stop_deg))
    j.CreateUpperLimitAttr(+float(cfg.tilt_stop_deg))
    return root


def _spawn_shutter(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC red shutter: a panel covering the local -x port (teleported to
    the blocked side each reset by rotating the whole body pi about z), standing on
    two ground pillars that straddle the bin side walls."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 5.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    c = cfg
    pz = (c.panel_z0 + c.panel_z1) / 2
    _box(stage, f"{prim_path}/panel",
         (c.panel_t, 2 * c.panel_half_y, c.panel_z1 - c.panel_z0),
         (-c.panel_x, 0.0, pz), _RED, c.contact_offset, material=mat)
    for sy in (1.0, -1.0):
        _box(stage, f"{prim_path}/pillar_{'p' if sy > 0 else 'n'}",
             (0.04, 0.03, c.panel_z1), (-c.panel_x, sy * 0.155, c.panel_z1 / 2),
             _RED, c.contact_offset, material=mat)
    return root


def _spawn_ball(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The oversized ball: one orange sphere (wider than the Franka jaw span —
    asserted in the scene cfg) with mild damping so bin rolling settles."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.15)
    pxrb.CreateAngularDampingAttr(0.5)
    mat = _material(stage, f"{prim_path}/phys_mat")
    r = cfg.radius
    seg = UsdGeom.Sphere.Define(stage, f"{prim_path}/ball")
    seg.CreateRadiusAttr(float(r))
    seg.CreateExtentAttr([Gf.Vec3f(-r, -r, -r), Gf.Vec3f(r, r, r)])
    seg.CreateDisplayColorAttr([Gf.Vec3f(*_ORANGE)])
    _collide(seg.GetPrim(), cfg.contact_offset, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "stand" not in _SPAWNER_CACHE:

        @configclass
        class StandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            contact_offset: float = 0.002
            hinge_h: float = 0.34
            bin_x0: float = 0.12
            bin_x1: float = 0.46
            bin_wall_t: float = 0.015
            bin_floor_top: float = 0.05
            bin_near_top: float = 0.10
            bin_back_top: float = 0.45
            bin_side_top: float = 0.16
            bin_half_y: float = 0.11

        @configclass
        class CageSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cage)
            contact_offset: float = 0.002
            hinge_h: float = 0.34
            dish_deg: float = 8.0
            tilt_stop_deg: float = 15.0
            cage_half_x: float = 0.177
            dish_valley_z: float = -0.075
            plate_t: float = 0.012
            wall_y: float = 0.061
            wall_h: float = 0.085
            wall_top: float = 0.004
            roof_half_x: float = 0.15
            roof_zbot: float = 0.065
            post_x: float = 0.14
            paddle_cx: float = 0.195
            paddle_s: float = 0.09
            keel_z: float = -0.16
            keel_sx: float = 0.20
            keel_sy: float = 0.06
            keel_sz: float = 0.05
            struct_density: float = 700.0
            keel_density: float = 5200.0
            cage_ang_damping: float = 4.0

        @configclass
        class ShutterSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shutter)
            contact_offset: float = 0.002
            panel_x: float = 0.205
            panel_t: float = 0.015
            panel_z0: float = 0.10
            panel_z1: float = 0.32
            panel_half_y: float = 0.104

        @configclass
        class BallSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ball)
            mass: float = 0.25
            radius: float = 0.044
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(stand=StandSpawnerCfg, cage=CageSpawnerCfg,
                              shutter=ShutterSpawnerCfg, ball=BallSpawnerCfg)
    return _SPAWNER_CACHE


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TiltDispenserSceneCfg(BaseCfg):
    """Config for `TiltDispenserScene`. The no-touch premise and every rubric
    window's honesty are asserted in `__post_init__` FROM THE AUTHORED GEOMETRY AND
    DENSITIES: the ball cannot be grasped or extracted, a level cage self-centres
    it (null-safe), the press torque is Franka-scale and sufficient, the shutter
    retains a wrong-side dispense recoverably, the drop lands inside the bin, and
    the bin window accepts every physically-in-bin rest while rejecting ground /
    wrong-bin / perch / in-cage poses."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    tilt_latch_deg: float = tunable(10.0)  # canonical (open-side-down) tilt latch
    exit_x_lo: float = tunable(0.16)  # ball-out-the-port window (canonical frame):
    exit_x_hi: float = tunable(0.60)
    exit_y: float = tunable(0.13)
    exit_z_hi: float = tunable(0.27)  # below every in-cage ball pose at |x|>=exit_x_lo
    bin_x_lo: float = tunable(0.17)  # in-open-bin window (canonical frame):
    bin_x_hi: float = tunable(0.41)
    bin_y: float = tunable(0.08)
    bin_z_lo: float = tunable(0.07)  # bin rest reads 0.094; ground rest reads 0.044
    bin_z_hi: float = tunable(0.13)  # near-wall-top perch reads 0.144
    # Settle gates sit ABOVE the GPU phantom-velocity readback artifact: a ball whose
    # position is frozen (< 1 mm/s true motion, verified by position telemetry) still
    # reads |v| up to ~0.13 m/s and |w| up to ~3.4 rad/s on this stack. Honesty is
    # carried by the tight bin position window + the >= 3.5 s hands-off persistence
    # (a ball really moving 0.18 m/s cannot hold z inside the 60 mm rest band of the
    # 0.24 m window for that long); real thrown/spun probes (0.30 m/s, 9 rad/s) are
    # still firmly rejected — smoke check 13 proves it.
    settle_lin: float = tunable(0.18)  # max ball |lin vel| at judging (m/s)
    settle_ang: float = tunable(4.5)  # max ball |ang vel| at judging (rad/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    yaw_jitter_deg: float = tunable(20.0)  # structure yaw jitter (full width)
    pos_jitter: float = tunable(0.03)  # +- structure xy jitter
    ball_x_jitter: float = tunable(0.05)  # +- ball start x in the dish (transient)
    ball_y_jitter: float = tunable(0.010)  # +- ball start y

    # --- info: mechanism geometry (what the spawners author) ---------------------------------
    hinge_h: float = info(0.34)  # hinge axis height (stand frame z)
    dish_deg: float = info(8.0)  # V-dish plate angle (self-centring)
    tilt_stop_deg: float = info(15.0)  # joint limit both ways
    cage_half_x: float = info(0.177)  # port planes at local x = +-this
    dish_valley_z: float = info(-0.075)  # valley surface top (cage frame, z from hinge)
    plate_t: float = info(0.012)
    wall_y: float = info(0.061)  # side-wall centre |y| (interior width 0.110)
    wall_h: float = info(0.085)
    wall_top: float = info(0.004)  # wall top (cage frame)
    roof_half_x: float = info(0.15)  # roof recessed from the ports
    roof_zbot: float = info(0.065)  # roof underside; slot gap = this - wall_top
    post_x: float = info(0.14)
    paddle_cx: float = info(0.195)  # paddle centre |x| (press arm)
    paddle_s: float = info(0.09)
    keel_z: float = info(-0.16)  # dense keel centre (pendulum mass)
    keel_sx: float = info(0.20)
    keel_sy: float = info(0.06)
    keel_sz: float = info(0.05)
    struct_density: float = info(700.0)  # cage structure density (kg/m^3)
    keel_density: float = info(5200.0)
    cage_ang_damping: float = info(4.0)  # return-swing overshoot < dish angle (asserted)
    press_torque: float = info(2.2)  # the solve's hinge torque (N m) = ~11 N on a paddle
    # --- info: stand / bins ------------------------------------------------------------------
    bin_x0: float = info(0.12)  # bin x span (near-wall outer face .. backboard outer)
    bin_x1: float = info(0.46)
    bin_wall_t: float = info(0.015)
    bin_floor_top: float = info(0.05)
    bin_near_top: float = info(0.10)  # low entry wall under the port
    bin_back_top: float = info(0.45)  # tall backboard catches the flight
    bin_side_top: float = info(0.16)
    bin_half_y: float = info(0.11)  # bin interior half-width
    # --- info: shutter -----------------------------------------------------------------------
    panel_x: float = info(0.205)  # panel centre |x| (covers the blocked port)
    panel_t: float = info(0.015)
    panel_z0: float = info(0.10)
    panel_z1: float = info(0.32)
    # Narrower than the bin interior (2*0.104 < 2*0.11): the kinematic panel must
    # NEVER touch the dynamic stand's side walls — an embedded kinematic contact
    # walks the stand out of its jitter band during settle (seen at +10 mm in y).
    panel_half_y: float = info(0.104)
    # --- info: ball / embodiment -------------------------------------------------------------
    ball_r: float = info(0.044)
    ball_mass: float = info(0.25)
    jaw_span: float = info(0.080)  # Franka parallel-jaw max opening
    max_press_force: float = info(25.0)  # Franka-comfortable sustained push (N)
    contact_offset: float = info(0.002)

    def __post_init__(self) -> None:
        c = self
        r, g = c.ball_r, 9.81
        a = math.radians(c.dish_deg)
        ts = math.radians(c.tilt_stop_deg)
        # -- the no-touch premise is real: the ball cannot be grasped or reached --
        assert 2 * r > c.jaw_span + 0.005, "ball must exceed the Franka jaw span"
        slot = c.roof_zbot - c.wall_top
        assert slot < 2 * r - 0.015, "the wall/roof slot must not pass the ball"
        assert c.cage_half_x - c.roof_half_x < 2 * r - 0.015, \
            "the roof port recess must not pass the ball from above"
        # -- the ports do pass the ball; the interior fits it --
        edge_top = c.dish_valley_z + c.cage_half_x * math.tan(a)
        assert c.roof_zbot - edge_top > 2 * r + 0.012, "port opening too small"
        assert 2 * c.wall_y - c.plate_t > 2 * r + 0.015, "cage interior too narrow"
        # -- authored masses: level cage self-centres the ball (null-safe) --
        m_cage, com_z, i_y = _cage_dyn(c)
        mgd = m_cage * g * (-com_z)
        assert com_z < -0.08, "cage CoM must hang well below the hinge"
        x_max = c.cage_half_x - r * math.sin(a)  # in-cage ball max |x|
        tau_ball = c.ball_mass * g * x_max
        assert mgd * math.sin(a) > 1.3 * tau_ball, "dish must dominate the ball torque"
        ball_tilt = math.degrees(math.asin(tau_ball / mgd))
        assert ball_tilt < c.dish_deg - 2.0, "ball-induced tilt must stay sub-dish"
        assert ball_tilt < c.tilt_latch_deg - 3.0, "ball-induced tilt must not latch"
        # -- the press is sufficient and Franka-scale --
        assert c.press_torque > 1.25 * mgd * math.sin(ts), "press torque insufficient"
        assert c.press_torque / c.paddle_cx < c.max_press_force, "press force too high"
        # -- release return-swing overshoot stays sub-dish (no spurious latch) --
        zeta = c.cage_ang_damping * math.sqrt(i_y / mgd) / 2
        over = math.exp(-math.pi * zeta / math.sqrt(1 - zeta * zeta)) * c.tilt_stop_deg
        assert over < c.dish_deg - 1.0, "return overswing must stay below the dish angle"
        # -- exit window honesty: every in-cage ball pose at |x'|>=exit_x_lo reads above --
        bex = c.cage_half_x - r * math.sin(a)  # ball centre at the dish edge (local)
        bez = edge_top + r * math.cos(a)
        in_cage_min_z = c.hinge_h - bex * math.sin(ts) + bez * math.cos(ts)
        assert in_cage_min_z > c.exit_z_hi + 0.015, "in-cage poses must sit above exit z"
        assert bex * math.cos(ts) + abs(bez) * math.sin(ts) + 0.01 > c.exit_x_lo, \
            "the exiting ball must be inside the exit window in x"
        # -- the drop lands inside the bin (even a zero-speed dribble) --
        exit_x = bex * math.cos(ts) - bez * math.sin(ts)  # >= horizontal exit reach
        assert min(exit_x, bex * math.cos(ts)) > c.bin_x0 + c.bin_wall_t - 0.01, \
            "a dribbled ball must clear the bin near wall"
        # -- bin window honesty --
        rest_z = c.bin_floor_top + r
        assert c.bin_z_lo < rest_z < c.bin_z_hi, "in-bin rest must read inside"
        assert r < c.bin_z_lo - 0.01, "a ground rest must read below"
        assert c.bin_near_top + r > c.bin_z_hi + 0.01, "a near-wall perch must read above"
        assert c.bin_side_top + r > c.bin_z_hi + 0.01, "a side-wall perch must read above"
        assert c.bin_x_lo < c.bin_x0 + c.bin_wall_t + r, "window must accept wall-hugging"
        assert c.bin_x_hi > c.bin_x1 - c.bin_wall_t - r, "window must accept backboard rest"
        assert c.bin_y > c.bin_half_y - r, "window must accept side-wall-hugging rests"
        assert c.bin_x0 - r < c.bin_x_lo, "an outside-the-near-wall rest must read out"
        # -- wrong-side press is retained by the shutter, recoverably --
        panel_in = c.panel_x - c.panel_t / 2
        assert panel_in - r < c.cage_half_x * math.cos(ts) - 0.005, \
            "shutter must stop the ball centre over the dish (recoverable)"
        assert c.panel_z1 > c.hinge_h - c.cage_half_x * math.sin(ts) + 0.02, \
            "panel must cover the lowered port"
        assert c.panel_z0 <= c.bin_near_top, "panel must reach down to the near-wall top"
        assert c.panel_half_y > c.wall_y + c.plate_t, "panel must span the ports"
        # widest ball surface at the port: centre |y| <= interior half-width
        assert c.panel_half_y > (c.wall_y - c.plate_t / 2 - r) + r + 0.003, \
            "panel must span every ball line through the port"
        assert c.panel_half_y < c.bin_half_y - 0.005, \
            "the kinematic panel must clear the bin side walls (embedded kinematic " \
            "contact walks the dynamic stand out of its jitter band)"
        # -- the swinging paddle clears the shutter panel --
        px, pz = c.paddle_cx + c.paddle_s / 2, c.roof_zbot
        rad = math.hypot(px, pz)
        beta = math.atan2(pz, px)
        assert c.hinge_h + rad * math.sin(beta - ts) > c.panel_z1 + 0.015, \
            "paddle sweep must clear the shutter panel"
        # -- score geometry: null start is scoreless --
        assert c.ball_x_jitter + r < c.exit_x_lo, "spawned ball must be outside exit x"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("tilt_dispenser")
class TiltDispenserScene(BaseScene):
    cfg: TiltDispenserSceneCfg

    def __init__(self, cfg: TiltDispenserSceneCfg | None = None) -> None:
        super().__init__(cfg or TiltDispenserSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        cage_geo = {k: getattr(c, k) for k in (
            "hinge_h", "dish_deg", "tilt_stop_deg", "cage_half_x", "dish_valley_z",
            "plate_t", "wall_y", "wall_h", "wall_top", "roof_half_x", "roof_zbot",
            "post_x", "paddle_cx", "paddle_s", "keel_z", "keel_sx", "keel_sy",
            "keel_sz", "struct_density", "keel_density", "cage_ang_damping")}
        stand_geo = {k: getattr(c, k) for k in (
            "hinge_h", "bin_x0", "bin_x1", "bin_wall_t", "bin_floor_top",
            "bin_near_top", "bin_back_top", "bin_side_top", "bin_half_y")}
        shut_geo = {k: getattr(c, k) for k in (
            "panel_x", "panel_t", "panel_z0", "panel_z1", "panel_half_y")}

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            # NOTE: the stand must precede the cage — the cage's spawn-authored
            # revolute joint targets the sibling /Stand prim.
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=sp["stand"](
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    contact_offset=c.contact_offset, **stand_geo),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "cage": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cage",
                spawn=sp["cage"](
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    contact_offset=c.contact_offset, **cage_geo),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.hinge_h)),
            ),
            "shutter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shutter",
                spawn=sp["shutter"](
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    contact_offset=c.contact_offset, **shut_geo),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=sp["ball"](
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.ball_mass, radius=c.ball_r,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, c.hinge_h - 0.020)),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
                # external-wrench plant recipe: without this the hinge-torque press
                # is under-applied across TGS iterations
                "enable_external_forces_every_iteration": True,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.stand: RigidObject = env.iscene["stand"]
        self.cage: RigidObject = env.iscene["cage"]
        self.shutter: RigidObject = env.iscene["shutter"]
        self.ball: RigidObject = env.iscene["ball"]
        self.env_origins = env.iscene.env_origins
        # per-env open side: +1 -> the local +x port is open (shutter on -x),
        # -1 -> mirrored. Canonical frame folds x,y by this sign.
        self.side = torch.ones(n, device=dev)
        # progress latches (post_step)
        self.tilt_latch = torch.zeros(n, device=dev)
        self.exit_latch = torch.zeros(n, device=dev)
        self.bin_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the open side (torch.rand comparison — the
        first-randint degeneracy), pose the stand with yaw + xy jitter, write the
        WHOLE hinge linkage coherently (stand + cage level at the hinge height —
        teleporting one member of a joint pair gets depenetrated back), flip the
        shutter to the blocked port, drop the ball into the dish; latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        self.side[env_ids] = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        s = self.side[env_ids]

        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_jitter_deg) / 2
        jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.pos_jitter
        ch, sh = torch.cos(half), torch.sin(half)

        stand = torch.zeros(m, 13, device=dev)
        stand[:, 0:2] = jit
        stand[:, 3] = ch
        stand[:, 6] = sh
        stand[:, 0:3] += origin
        self.stand.write_root_state_to_sim(stand, env_ids)

        cage = stand.clone()
        cage[:, 2] += c.hinge_h  # yaw never moves the on-axis point (0,0,h)
        self.cage.write_root_state_to_sim(cage, env_ids)

        shut = stand.clone()
        # blocked port is local -s x; the shutter is authored at local -x, so for
        # s=-1 rotate it pi about z: (ch,0,0,sh) (x) (0,0,0,1) = (-sh,0,0,ch)
        shut[:, 3] = torch.where(s > 0, ch, -sh)
        shut[:, 6] = torch.where(s > 0, sh, ch)
        self.shutter.write_root_state_to_sim(shut, env_ids)

        bx = (torch.rand(m, device=dev) * 2 - 1) * c.ball_x_jitter
        by = (torch.rand(m, device=dev) * 2 - 1) * c.ball_y_jitter
        cy, sy = torch.cos(2 * half), torch.sin(2 * half)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = stand[:, 0] - origin[:, 0] + cy * bx - sy * by
        st[:, 1] = stand[:, 1] - origin[:, 1] + sy * bx + cy * by
        st[:, 2] = c.hinge_h - 0.020  # just above the dish rest everywhere in the jitter
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.ball.write_root_state_to_sim(st, env_ids)

        for latch in (self.tilt_latch, self.exit_latch, self.bin_latch):
            latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"stand": self.stand, "cage": self.cage, "shutter": self.shutter,
                  "ball": self.ball}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in bodies.items()},
            "side": self.side[env_ids].clone(),
            "latches": torch.stack([self.tilt_latch[env_ids],
                                    self.exit_latch[env_ids],
                                    self.bin_latch[env_ids]], dim=1),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"stand": self.stand, "cage": self.cage, "shutter": self.shutter,
                  "ball": self.ball}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.side[env_ids] = state["side"]
        lt = state["latches"]
        self.tilt_latch[env_ids] = lt[:, 0]
        self.exit_latch[env_ids] = lt[:, 1]
        self.bin_latch[env_ids] = lt[:, 2]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A BALL DISPENSER stands on the floor: a slotted, roofed CAGE that "
            "pivots on a central hinge atop a grey stand, holding a single large "
            f"ORANGE BALL ({2 * c.ball_r * 1000:.0f} mm across — wider than a "
            "parallel-jaw gripper opens, and sealed behind viewing slots far "
            "narrower than the ball, so it can never be grasped, carried or even "
            "touched). The cage floor is a shallow V that keeps the ball centred "
            "while the cage is level. Both cage ends are open PORTS with a walled "
            "CATCH BIN below each, but one port is covered by a RED SHUTTER panel "
            "— which side is blocked changes between episodes, so look before you "
            "act. A square YELLOW PADDLE sticks out at roof level beyond each "
            "port: pressing a paddle down tilts the whole cage toward that side "
            "until it reaches its tilt stop, and holding it there lets the ball "
            "roll down the V-floor and out through that side's port. If the open "
            "side was chosen the ball drops into the catch bin below; if the "
            "blocked side was chosen the red shutter simply holds the ball in, "
            "and on release the cage swings level and re-centres it. The cage "
            "always swings itself back level when released.\n"
            "Goal: find the port NOT covered by the red shutter, press that "
            "side's yellow paddle down and hold the cage tilted at its stop until "
            "the ball rolls out of the open port and falls into the catch bin "
            "underneath, then release the paddle and let everything settle. Only "
            "a ball at rest inside the open side's catch bin is success — a ball "
            "still in the cage, on the floor, or balanced anywhere else scores "
            "nothing at the end."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Find the cage port that is not covered by the red shutter. Press "
            "down the yellow paddle on that open side and hold the cage tilted "
            "at its stop so the orange ball rolls out through the open port and "
            "drops into the catch bin below, then release the paddle. The ball "
            "is sealed in the cage and too big to grasp — only the paddle press "
            "can deliver it. Finish with the ball at rest inside that bin."
        )

    # ----- frames / live readouts -------------------------------------------------------------
    def tilt_deg(self) -> torch.Tensor:
        """(N,) float: hinge angle in degrees from the stand->cage relative
        quaternion about the hinge (y) axis; POSITIVE = local +x port down. There
        is no joint-state API on a plain spawn-authored USD joint."""
        qs = self.stand.data.root_quat_w
        qc = self.cage.data.root_quat_w
        qs_inv = qs * torch.tensor([1.0, -1.0, -1.0, -1.0], device=qs.device)
        rel = _qmul(qs_inv, qc)
        ang = torch.rad2deg(2.0 * torch.atan2(rel[:, 2], rel[:, 0]))
        ang = torch.where(ang > 180.0, ang - 360.0, ang)
        ang = torch.where(ang < -180.0, ang + 360.0, ang)
        return ang

    def tilt_canon_deg(self) -> torch.Tensor:
        """(N,) float: tilt folded by the open side (+ = open port going DOWN)."""
        return self.side * self.tilt_deg()

    def ball_canon(self) -> torch.Tensor:
        """(N, 3): ball centre in the CANONICAL stand frame (stand-local, x and y
        folded by the open side so the open port is always at +x and one window
        set serves both mirror cases)."""
        from isaaclab.utils.math import quat_apply_inverse

        loc = quat_apply_inverse(self.stand.data.root_quat_w,
                                 self.ball.data.root_pos_w - self.stand.data.root_pos_w)
        loc = loc.clone()
        loc[:, 0] = self.side * loc[:, 0]
        loc[:, 1] = self.side * loc[:, 1]
        return loc

    def canon_to_world(self, canon: torch.Tensor) -> torch.Tensor:
        """(N, 3) canonical points -> world (solve/smoke placement transform)."""
        from isaaclab.utils.math import quat_apply

        loc = canon.clone()
        loc[:, 0] = self.side * loc[:, 0]
        loc[:, 1] = self.side * loc[:, 1]
        return self.stand.data.root_pos_w + quat_apply(self.stand.data.root_quat_w, loc)

    # ----- region predicates (canonical frame) ------------------------------------------------
    def ball_exited(self) -> torch.Tensor:
        """(N,) bool, geometric: ball out through the OPEN port — beyond the port
        plane and below every in-cage pose (the z gate separates it from a ball
        still inside the tilted cage; asserted in __post_init__)."""
        c = self.cfg
        p = self.ball_canon()
        return (p[:, 0] >= c.exit_x_lo) & (p[:, 0] <= c.exit_x_hi) \
            & (p[:, 1].abs() <= c.exit_y) & (p[:, 2] <= c.exit_z_hi)

    def in_open_bin(self) -> torch.Tensor:
        """(N,) bool, geometric: ball centre inside the OPEN side's catch-bin
        containment window (accepts every physically-in-bin resting pose, rejects
        ground / wrong-bin / wall-perch / in-cage poses; asserted in
        __post_init__)."""
        c = self.cfg
        p = self.ball_canon()
        return (p[:, 0] >= c.bin_x_lo) & (p[:, 0] <= c.bin_x_hi) \
            & (p[:, 1].abs() <= c.bin_y) \
            & (p[:, 2] >= c.bin_z_lo) & (p[:, 2] <= c.bin_z_hi)

    def settled(self) -> torch.Tensor:
        """(N,) bool: ball at rest."""
        c = self.cfg
        return (self.ball.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.ball.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch each demonstrated stage every physics step: cage ever held past
        the dispense angle TOWARD THE OPEN SIDE, ball ever out through the open
        port, ball ever inside the open bin window."""
        c = self.cfg
        tilt_ok = (self.tilt_canon_deg() >= c.tilt_latch_deg).float()
        self.tilt_latch = torch.maximum(self.tilt_latch, tilt_ok)
        self.exit_latch = torch.maximum(self.exit_latch, self.ball_exited().float())
        self.bin_latch = torch.maximum(self.bin_latch, self.in_open_bin().float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the ball rests inside the OPEN side's catch bin. The press
        is physically necessary: the ball is sealed in the cage (jaw-proof size,
        sub-ball slots) and only a held tilt past the dish angle dispenses it."""
        return self.in_open_bin() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.2 per latched stage (open-side tilt held, ball
        out the open port, ball in the bin window; cap 0.60); 1.0 iff success().
        Latched — credit never evaporates; the null policy scores ~0 (the level
        cage self-centres the ball; asserted from the authored masses)."""
        base = (0.2 * (self.tilt_latch + self.exit_latch + self.bin_latch)).clamp(0.0, 0.60)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="tilt_dispenser", robot="null", env_spacing=6.0))
